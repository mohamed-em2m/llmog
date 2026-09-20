import datetime as _dt
import logging
import os
import re
import subprocess
import threading
import time
import requests

logger = logging.getLogger("detection_pipeline.server")


# ─── Loading-stage patterns ──────────────────────────────────────────────
# vLLM startup takes minutes and prints a wall of framework logs. These
# patterns turn key milestone lines into one-line human progress updates.
# First match wins; ``None`` label means "track silently, don't announce".
_STAGE_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (
        re.compile(r"Uvicorn running on|Application startup complete"),
        "api-server-up",
    ),
    (
        re.compile(r"Resolved architecture:\s*(\S+)"),
        "architecture-resolved",
    ),
    (
        re.compile(r"Initializing a V\d? LLM engine"),
        "engine-init",
    ),
    (
        re.compile(r"Loading model from scratch"),
        "weights-load-start",
    ),
    (
        re.compile(r"Parse safetensors files"),
        "checkpoints-parse",
    ),
    (
        re.compile(r"Time spent downloading weights for \S+: ([\d.]+)"),
        "weights-downloaded",
    ),
    (
        re.compile(r"Loading safetensors checkpoint shards:\s*100%"),
        "shards-loaded",
    ),
    (
        re.compile(r"Loading weights took ([\d.]+) seconds"),
        "weights-loaded",
    ),
    (
        re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
        "model-in-vram",
    ),
    (
        re.compile(r"Encoder cache will be initialized"),
        "vision-encoder-init",
    ),
    (
        re.compile(r"Compiling a graph for|torch\.compile took"),
        "cuda-graphs-compile",
    ),
    (
        re.compile(r"collected artifacts:|saved AOT compiled"),
        "compile-cache-saved",
    ),
]

_STAGE_LABELS = {
    "api-server-up": "API server up — engine is now loading the model (this takes minutes)",
    "architecture-resolved": "model architecture resolved",
    "engine-init": "initializing inference engine",
    "weights-load-start": "loading model weights",
    "checkpoints-parse": "reading checkpoint files",
    "weights-downloaded": "weights downloaded",
    "shards-loaded": "checkpoint shards loaded",
    "weights-loaded": "weights loaded",
    "model-in-vram": "model resident in VRAM",
    "vision-encoder-init": "initializing vision encoder cache",
    "cuda-graphs-compile": "compiling CUDA graphs (one-time cost, can take minutes)",
    "compile-cache-saved": "compile cache saved",
}

# Lines that look scary (ERROR/WARNING) but are routine on older GPUs.
_LEGACY_GPU_PATTERNS = [
    re.compile(r"compute capability 7\.5"),
    re.compile(r"doesn't support torch\.bfloat16"),
    re.compile(r"Cannot use FA version 2"),
    re.compile(r"FlashInfer .* unsupported compute capability"),
    re.compile(r"Falling back to the Triton"),
]


# ─── vLLM Server Class ─────────────────────────────────────────────────────
class VllmServerManager:
    # ─── Configuration & Initialization ────────────────────────────────────
    def __init__(
        self,
        model: str = "Qwen/Qwen3.6-27B",
        host: str = "0.0.0.0",
        port: int = 8000,
        served_model_name: str | None = None,
        max_model_len: int = 20000,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str | None = None,  # e.g. "fp8", "awq", "gptq", "marlin"
        kv_cache_dtype: str = "auto",  # e.g. "fp8", "fp8_e5m2"
        max_num_seqs: int = 16,
        enforce_eager: bool = False,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = True,
        speculative_model: str | None = None,
        num_speculative_tokens: int | None = None,
        trust_remote_code: bool = True,
        limit_mm_per_prompt: str | None = None,  # e.g. "image=2"
        chat_template: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.served_model_name = served_model_name or model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.dtype = dtype
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.enable_chunked_prefill = enable_chunked_prefill
        self.enable_prefix_caching = enable_prefix_caching
        self.speculative_model = speculative_model
        self.num_speculative_tokens = num_speculative_tokens
        self.trust_remote_code = trust_remote_code
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.chat_template = chat_template
        self.extra_args = extra_args or []

        self.process = None
        self.server_ready_event = threading.Event()
        self.logs = []
        self.log_lock = threading.Lock()
        self._started_at: float | None = None
        self._last_health_latency_ms: float | None = None
        # Human-readable loading progress, updated by the monitor thread.
        self.current_stage: str = "starting"
        self.stage_history: list[tuple[str, float]] = []
        self._legacy_gpu_note_logged = False

        req_host = "localhost" if self.host == "0.0.0.0" else self.host
        self.server_url = f"http://{req_host}:{self.port}"

    # ─── Server Launcher ────────────────────────────────────────────────────
    def start_vllm_server(self):
        """
        Spawns the `vllm serve` subprocess and launches a daemon thread
        to monitor stdout/stderr for the readiness signal.
        """
        if self.process is not None and self.process.poll() is None:
            print("ℹ️ Server is already running.")
            return

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model_name,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
            "--dtype",
            self.dtype,
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
        ]

        if self.quantization:
            cmd.extend(["--quantization", self.quantization])
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.limit_mm_per_prompt:
            cmd.extend(["--limit-mm-per-prompt", self.limit_mm_per_prompt])
        if self.chat_template:
            cmd.extend(["--chat-template", self.chat_template])
        if self.speculative_model:
            # vLLM expects a JSON-ish dict string for --speculative-config
            spec_cfg = {"model": self.speculative_model}
            if self.num_speculative_tokens is not None:
                spec_cfg["num_speculative_tokens"] = self.num_speculative_tokens
            cmd.extend(["--speculative-config", str(spec_cfg)])

        cmd.extend(self.extra_args)

        logger.info("Spawning vLLM: %s", " ".join(cmd))

        self.server_ready_event.clear()
        with self.log_lock:
            self.logs = []
        self._started_at = time.time()
        self._last_health_latency_ms = None

        env = os.environ.copy()
        # Common fix for stale AOT compile cache issues across restarts
        env.setdefault("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )

        def monitor_output():
            last_raw = None
            for line in self.process.stdout:
                self._note_stage(line)
                ts = _dt.datetime.now().strftime("%H:%M:%S")
                stamped = f"[{ts}] {line}" if not line.startswith("[") else line
                with self.log_lock:
                    self.logs.append(stamped)
                    if len(self.logs) > 3000:
                        self.logs.pop(0)
                try:
                    logger.debug(stamped.rstrip())
                except Exception:
                    pass
                # Collapse exact consecutive duplicates (vLLM re-prints 100%
                # progress bars and shard counters) to keep the console readable.
                if line != last_raw:
                    print(stamped, end="", flush=True)
                    last_raw = line
                if (
                    "Uvicorn running on" in line
                    or "Application startup complete" in line
                    or "Started server process" in line
                ):
                    self.server_ready_event.set()

        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()

    def _note_stage(self, line: str) -> None:
        """Match a raw server line against known loading milestones.

        On a new milestone, update ``current_stage`` and emit one concise
        INFO line so users see progress instead of a wall of framework logs.
        Also emits a one-time reassurance when legacy-GPU fallback lines
        (which look like ERRORs) are detected.
        """
        if not self._legacy_gpu_note_logged and any(
            p.search(line) for p in _LEGACY_GPU_PATTERNS
        ):
            self._legacy_gpu_note_logged = True
            logger.info(
                "vLLM: older GPU detected — compatible fallback kernels in use. "
                "Related FA2/FlashInfer/bfloat16 ERROR/WARNING lines are "
                "expected and harmless."
            )
        seen = {s for s, _ in self.stage_history} | {self.current_stage}
        for pattern, stage in _STAGE_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            if stage is None or stage in seen:
                return
            self.current_stage = stage
            elapsed = time.time() - self._started_at if self._started_at else 0.0
            with self.log_lock:
                self.stage_history.append((stage, elapsed))
            label = _STAGE_LABELS.get(stage, stage)
            detail = ""
            if m.groups():
                detail = f" ({', '.join(g for g in m.groups() if g)})"
            logger.info("vLLM loading: %s%s [+%.0fs]", label, detail, elapsed)
            return

    def get_logs(self) -> str:
        """Return all captured logs so far as a single string."""
        with self.log_lock:
            return "".join(self.logs)

    # ─── Wait for server ────────────────────────────────────────────────────
    def wait_for_server(self, timeout: int = 600):
        """
        Waits for the output event trigger and polls the health endpoint
        until the server confirms it is fully loaded. vLLM's engine init
        (weight loading + CUDA graph capture) can take much longer than
        llama.cpp's, hence the longer default timeout.
        """
        print("⏳ Waiting for vLLM server...")
        self.server_ready_event.wait(timeout=timeout)

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"✅ Server ready at {self.server_url}")
                    return True
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("❌ Server did not become healthy in time.")

    # ─── Warmup Request ─────────────────────────────────────────────────────
    def warmup_model(self, test_prompt: str = "explain what is qwen"):
        """
        Performs an initial run on the loaded model to trigger any
        remaining lazy compilation (e.g. CUDA graphs) and warm caches.
        """
        try:
            print("🔥 Warming up model...")
            requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": self.served_model_name,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=120,
            )
            print("✅ Warmup done — TTFT will be faster for real requests")
        except Exception as e:
            print(f"⚠️  Warmup failed (non-fatal): {e}")

    # ─── Shutdown ───────────────────────────────────────────────────────────
    def stop_vllm_server(self, timeout: float = 10.0):
        """
        Terminates the vLLM server process cleanly.

        Attempts a graceful termination (SIGTERM) first to allow vLLM to
        release GPU VRAM (KV cache blocks, CUDA graphs) before falling
        back to a forceful kill (SIGKILL) if the process hangs.
        """
        if self.process is None:
            print("⚠️ No active process provided to stop.")
            return

        if self.process.poll() is not None:
            print(
                f"ℹ️ Server process has already stopped (Exit code: {self.process.poll()})"
            )
            self.process = None
            return

        print("🛑 Sending termination signal (SIGTERM)...")
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
            print(f"✅ Server stopped cleanly. (Exit code: {self.process.returncode})")

        except subprocess.TimeoutExpired:
            print(
                f"⚠️ Server did not exit within {timeout} seconds. Forcing shutdown (SIGKILL)..."
            )
            try:
                self.process.kill()
                self.process.wait()
                print("✅ Server process forcefully terminated.")
            except Exception as e:
                print(f"❌ Failed to forcefully kill the process: {e}")

        except Exception as e:
            print(f"❌ An error occurred during termination: {e}")
        finally:
            self.process = None

    # ─── Orchestrator / Entry Point ─────────────────────────────────────────
    def run_pipeline(self):
        """
        Coordinates launching, waiting, and warming up the model.
        """
        self.start_vllm_server()
        self.wait_for_server()
        self.warmup_model()

    def is_healthy(self) -> bool:
        """Check if the vLLM server is running and responsive."""
        if self.process is None or self.process.poll() is not None:
            self._last_health_latency_ms = None
            return False
        t0 = time.time()
        try:
            r = requests.get(f"{self.server_url}/health", timeout=1)
            self._last_health_latency_ms = (time.time() - t0) * 1000
            return r.status_code == 200
        except Exception:
            self._last_health_latency_ms = None
            return False

    def get_detailed_status(self) -> dict:
        pid = self.process.pid if self.process and self.process.poll() is None else None
        uptime_s = int(time.time() - self._started_at) if self._started_at else 0
        return {
            "pid": pid,
            "port": self.port,
            "host": self.host,
            "url": self.server_url,
            "model": self.model,
            "uptime_s": uptime_s,
            "health_latency_ms": self._last_health_latency_ms,
            "healthy": self.is_healthy() if pid else False,
            "log_lines": len(self.logs),
            "current_stage": self.current_stage,
            "stages_seen": [s for s, _ in self.stage_history],
        }

    # ─── Context Manager ────────────────────────────────────────────────────
    def __enter__(self):
        self.run_pipeline()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_vllm_server()


# ─── Execution ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: FP8 W8A16 on an L4 24GB GPU, eager mode to sidestep the
    # Marlin W4A16 shape-mismatch bug seen with gemma4_unified on vLLM 0.22.x
    server = VllmServerManager(
        model="google/gemma-4-12b-it",
        port=8000,
        max_model_len=20000,
        gpu_memory_utilization=0.90,
        quantization="fp8",
        kv_cache_dtype="fp8",
        enforce_eager=True,
    )

    server.run_pipeline()

    try:
        server.process.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop_vllm_server()
