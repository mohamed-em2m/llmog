import os
import subprocess
import sys
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def _detect_gpu_count() -> int:
    """Return the number of NVIDIA GPUs detected; falls back to 1 on any error."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        count = sum(1 for line in out.splitlines() if line.strip())
        return max(count, 1)
    except Exception:
        return 1


num_gpus = _detect_gpu_count()
tensor_split = ",".join(["1"] * num_gpus)


# ─── Llama-CPP-Python Server Class ───────────────────────────────────────────
class LlamaCppPythonManager:
    """Manage a local llama-cpp-python OpenAI-compatible server.

    Unlike :class:`LlamaServerManager` -- which spawns the standalone
    ``llama-server`` binary shipped with the llama.cpp C++ repo -- this manager
    launches the server bundled inside the ``llama-cpp-python`` package:

        python -m llama_cpp.server --model ... --host ... --port ...

    Because both backends expose the same OpenAI-compatible HTTP API, this
    class intentionally mirrors :class:`LlamaServerManager`'s public surface
    (start / wait / warmup / stop / is_healthy / context-manager) so callers
    and the ``servers_factory`` registry can treat them interchangeably.
    """

    def __init__(
        self,
        model: str = "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
        host: str = "0.0.0.0",
        port: int = 8080,
        ctx_size: int = 20000,
        parallel_slots: int = 1,
        n_threads: int = -1,
        gpu_layers: int = -1,
        tensor_split: str = tensor_split,
        main_gpu: int = 0,
        temp: float = 0.4,
        top_p: float = 0.95,
        top_k: int = 64,
        fa: str = "auto",
        enable_thinking: bool = False,
        batch_size: int = 1024,
        ubatch_size: int = 512,
        kv_cache_type: str = "q4_0",
        image_min_tokens: int = 1024,
        image_max_tokens: int = 4096,
        chat_format: str = "chatml",
        verbose: bool = False,
        extra_args: list = None,
        log_disable: bool = True,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.ctx_size = ctx_size
        self.parallel_slots = parallel_slots
        self.n_threads = n_threads
        self.gpu_layers = gpu_layers
        self.tensor_split = tensor_split
        self.main_gpu = main_gpu
        self.temp = temp
        self.top_p = top_p
        self.top_k = top_k
        self.fa = fa
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.kv_cache_type = kv_cache_type
        self.image_min_tokens = image_min_tokens
        self.image_max_tokens = image_max_tokens
        self.chat_format = chat_format
        self.verbose = verbose
        self.extra_args = list(extra_args or [])
        self.log_disable = log_disable

        self.process = None
        self.server_ready_event = threading.Event()
        self.logs = []
        self.log_lock = threading.Lock()

        # Resolve connection URL (use localhost if binding to all interfaces)
        req_host = "localhost" if self.host == "0.0.0.0" else self.host
        self.server_url = f"http://{req_host}:{self.port}"

    # ─── Server Launcher ──────────────────────────────────────────────────────
    def start_llama_server(self):
        """
        Spawns the ``python -m llama_cpp.server`` subprocess and launches a
        daemon thread to monitor stdout for the readiness signal.
        """
        if self.process is not None and self.process.poll() is None:
            print("ℹ️ Server is already running.")
            return

        cmd = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n_ctx",
            str(self.ctx_size),
            "--n_gpu_layers",
            str(self.gpu_layers),
            "--main_gpu",
            str(self.main_gpu),
            "--n_parallel",
            str(self.parallel_slots),
            "--tensor_split",
            self.tensor_split,
            "--chat_format",
            self.chat_format,
        ]
        if self.n_threads is not None and self.n_threads > 0:
            cmd.extend(["--n_threads", str(self.n_threads)])
        if self.verbose:
            cmd.append("--verbose")
        else:
            cmd.append("--verbose_error")

        if self.batch_size is not None and self.batch_size > 0:
            cmd.extend(["--n_batch", str(self.batch_size)])
        if self.ubatch_size is not None and self.ubatch_size > 0:
            cmd.extend(["--n_ubatch", str(self.ubatch_size)])
        if self.kv_cache_type is not None:
            cmd.extend(
                [
                    "--cache_type_k",
                    self.kv_cache_type,
                    "--cache_type_v",
                    self.kv_cache_type,
                ]
            )
        if self.extra_args:
            cmd.extend(self.extra_args)

        print("cmd: ", cmd)
        self.server_ready_event.clear()
        with self.log_lock:
            self.logs = []

        # Start the subprocess
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Background reader thread to monitor server startup output
        def monitor_output():
            for line in self.process.stdout:
                with self.log_lock:
                    self.logs.append(line)
                    if len(self.logs) > 2000:
                        self.logs.pop(0)
                print(line, end="", flush=True)
                if "Uvicorn running" in line or "Application startup complete" in line:
                    self.server_ready_event.set()

        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()

    def get_logs(self) -> str:
        """Return all captured logs so far as a single string."""
        with self.log_lock:
            return "".join(self.logs)

    # ─── Wait for server ──────────────────────────────────────────────────────
    def wait_for_server(self, timeout: int = 180):
        """
        Waits for the output event trigger and polls the health endpoint
        until the server confirms it is fully loaded.
        """
        print("⏳ Waiting for llama-cpp-python server...")
        self.server_ready_event.wait(timeout=timeout)

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "ok":
                        print(f"✅ Server ready at {self.server_url}")
                        return True
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("❌ Server did not become healthy in time.")

    # ─── Warmup Request ───────────────────────────────────────────────────────
    def warmup_model(self, test_prompt: str = "explain what is qwen"):
        """
        Performs an initial run on the loaded model to initialize paths and KV caches.
        """
        try:
            print("🔥 Warming up model...")
            requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=60,
            )
            print("✅ Warmup done — TTFT will be faster for real requests")
        except Exception as e:
            print(f"⚠️  Warmup failed (non-fatal): {e}")

    # ─── Shutdown ─────────────────────────────────────────────────────────────
    def stop_llama_server(self, timeout: float = 5.0):
        """
        Terminates the llama-cpp-python server process cleanly.
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

    # ─── Orchestrator / Entry Point ───────────────────────────────────────────
    def run_pipeline(self):
        """
        Coordinates launching, waiting, and warming up the model.
        """
        self.start_llama_server()
        self.wait_for_server()
        self.warmup_model()

    def is_healthy(self) -> bool:
        """Check if the llama-cpp-python server is running and responsive."""
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            r = requests.get(f"{self.server_url}/health", timeout=1)
            if r.status_code == 200:
                data = r.json()
                return data.get("status") == "ok"
        except Exception:
            pass
        return False

    # ─── Context Manager ──────────────────────────────────────────────────────
    def __enter__(self):
        self.run_pipeline()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_llama_server()


# ─── Execution ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = LlamaCppPythonManager(
        model="unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
        port=8080,
        ctx_size=20000,
    )

    server.run_pipeline()

    try:
        server.process.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop_llama_server()
