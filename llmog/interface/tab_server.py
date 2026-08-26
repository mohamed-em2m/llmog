"""
Llama Server tab UI and lifecycle server management logic.
"""

import tempfile
import time
import logging
from typing import Dict, Any
import gradio as gr

from servers import LlamaServerManager, LlamaCppPythonManager, VllmServerManager
from servers.llama_server_manager import num_gpus as _num_gpus
from interface.state import (
    state,
    MODEL_PRESETS,
    panel_header,
    _section_title,
)

logger = logging.getLogger("detection_pipeline")


def _stop_manager():
    """Stop whatever server manager is active regardless of backend."""
    manager = state.server_manager
    if manager is None:
        return
    if hasattr(manager, "stop_vllm_server"):
        manager.stop_vllm_server()
    else:
        manager.stop_llama_server()


def _start_manager():
    """Start whatever server manager is active regardless of backend."""
    manager = state.server_manager
    if manager is None:
        return
    if hasattr(manager, "start_vllm_server"):
        manager.start_vllm_server()
    else:
        manager.start_llama_server()


def start_server_wrapper(
    model,
    port,
    host,
    enable_thinking,
    enable_mtp,
    ctx_size,
    gpu_layers,
    kv_cache_type,
    image_min_tokens,
    image_max_tokens,
    parallel_slots,
    batch_size=1024,
    ubatch_size=1024,
    disable_log=False,
    server_type="llama_cpp",
    vllm_tp=1,
    vllm_gpu_util=0.90,
    vllm_max_seq=20000,
):
    # Coerce numeric inputs — gr.Number can yield None when cleared
    try:
        parallel_slots = max(1, int(parallel_slots or 1))
    except (TypeError, ValueError):
        parallel_slots = 1
    try:
        ctx_size = int(ctx_size or 10000) * parallel_slots
    except (TypeError, ValueError):
        ctx_size = 10000 * parallel_slots
    port = int(port or 8080)
    gpu_layers = int(gpu_layers if gpu_layers is not None else -1)

    with state.server_lock:
        if state.server_manager is not None and state.server_manager.is_healthy():
            yield (
                "Server is already running and healthy.",
                f'<span class="status-badge badge-running">RUNNING (Port {state.server_manager.port})</span>',
            )
            return

        yield (
            "Stopping any existing server instance...",
            '<span class="status-badge badge-starting">CLEANING UP...</span>',
        )
        if state.server_manager is not None:
            try:
                _stop_manager()
            except Exception as e:
                logger.warning(f"Error stopping old server: {e}")
            state.server_manager = None

        yield (
            "Configuring server...",
            '<span class="status-badge badge-starting">INITIALIZING...</span>',
        )

        tensor_split = ",".join(["1"] * _num_gpus)
        spec_type = "draft-mtp" if enable_mtp else "none"
        if server_type == "llama_cpp_python":
            state.server_manager = LlamaCppPythonManager(
                model=model,
                host=host,
                port=int(port),
                ctx_size=int(ctx_size),
                parallel_slots=parallel_slots,
                n_threads=-1,
                gpu_layers=int(gpu_layers),
                tensor_split=tensor_split,
                main_gpu=0,
                temp=0.4,
                top_p=0.95,
                top_k=64,
                enable_thinking=enable_thinking,
                batch_size=int(batch_size) if batch_size else 2048,
                ubatch_size=int(ubatch_size) if ubatch_size else 512,
                kv_cache_type=kv_cache_type,
                image_min_tokens=(
                    int(image_min_tokens) if image_min_tokens is not None else 1024
                ),
                image_max_tokens=(
                    int(image_max_tokens) if image_max_tokens is not None else 4096
                ),
                log_disable=bool(disable_log),
            )
        elif server_type == "vllm":
            state.server_manager = VllmServerManager(
                model=model,
                host=host,
                port=int(port),
                max_model_len=int(vllm_max_seq) if vllm_max_seq else 20000,
                gpu_memory_utilization=(
                    float(vllm_gpu_util) if vllm_gpu_util else 0.90
                ),
                tensor_parallel_size=int(vllm_tp) if vllm_tp else 1,
                max_num_seqs=int(parallel_slots) if parallel_slots else 16,
            )
        else:
            state.server_manager = LlamaServerManager(
                model=model,
                host=host,
                port=int(port),
                ctx_size=int(ctx_size),
                parallel_slots=parallel_slots,
                n_threads=-1,
                gpu_layers=int(gpu_layers),
                tensor_split=tensor_split,
                main_gpu=0,
                temp=0.4,
                top_p=0.95,
                top_k=64,
                spec_type=spec_type,
                spec_draft_n_max=4 if enable_mtp else 0,
                enable_thinking=enable_thinking,
                batch_size=int(batch_size) if batch_size else 2048,
                ubatch_size=int(ubatch_size) if ubatch_size else 512,
                kv_cache_type=kv_cache_type,
                image_min_tokens=(
                    int(image_min_tokens) if image_min_tokens is not None else 1024
                ),
                image_max_tokens=(
                    int(image_max_tokens) if image_max_tokens is not None else 4096
                ),
                log_disable=bool(disable_log),
            )

        yield (
            "Spawning server process...",
            '<span class="status-badge badge-starting">STARTING...</span>',
        )
        try:
            _start_manager()
        except Exception as e:
            state.server_manager = None
            yield (
                f"Failed to start server process: {e}",
                '<span class="status-badge badge-error">PROCESS ERROR</span>',
            )
            return

    start_time = time.time()
    # vLLM engine init (weight loading + CUDA graph capture) takes far longer
    # than llama.cpp, so give it a much bigger health-check budget.
    timeout = 600 if server_type == "vllm" else 180
    healthy = False

    while time.time() - start_time < timeout:
        with state.server_lock:
            if state.server_manager is None:
                yield (
                    "Server initialization aborted.",
                    '<span class="status-badge badge-stopped">STOPPED</span>',
                )
                return
            if (
                state.server_manager.process
                and state.server_manager.process.poll() is not None
            ):
                exit_code = state.server_manager.process.poll()
                logs = state.server_manager.get_logs()
                state.server_manager = None
                yield (
                    f"Server process exited with code {exit_code}.\n\n--- Logs ---\n{logs}",
                    '<span class="status-badge badge-error">CRASHED</span>',
                )
                return
            if state.server_manager.is_healthy():
                healthy = True
                break

            logs = state.server_manager.get_logs()
            elapsed = int(time.time() - start_time)
            yield (
                f"Waiting for model to load into memory... ({elapsed}s elapsed)\n\n--- Latest Output ---\n{logs[-1200:]}",
                '<span class="status-badge badge-starting">STARTING...</span>',
            )
        time.sleep(2)

    if healthy:
        yield (
            "Server is up. Running warmup request...",
            '<span class="status-badge badge-starting">WARMING UP...</span>',
        )
        try:
            with state.server_lock:
                if state.server_manager:
                    state.server_manager.warmup_model()
            yield (
                "Server started and warmed up. Ready for detection tasks.",
                f'<span class="status-badge badge-running">RUNNING (Port {port})</span>',
            )
        except Exception as e:
            yield (
                f"Server is healthy, but warmup failed: {e}",
                f'<span class="status-badge badge-running">RUNNING (Port {port})</span>',
            )
    else:
        yield (
            "Timed out waiting for the server to report healthy status.",
            '<span class="status-badge badge-error">TIMEOUT</span>',
        )


def stop_server_wrapper():
    with state.server_lock:
        if state.server_manager is None:
            return (
                "No server running.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        try:
            _stop_manager()
            state.server_manager = None
            return (
                "Server stopped successfully.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        except Exception as e:
            return (
                f"Error stopping server: {e}",
                '<span class="status-badge badge-error">STOP ERROR</span>',
            )


def on_server_backend_change(backend: str):
    """Toggle backend-specific advanced groups when the runtime changes.

    llama.cpp / llama-cpp-python share the llama option set (incl. MTP);
    vLLM has its own tensor-parallel / memory options.
    """
    is_llama = backend in ("llama_cpp", "llama_cpp_python")
    return (
        gr.update(visible=is_llama),  # llama_advanced_group
        gr.update(visible=backend == "vllm"),  # vllm_advanced_group
        gr.update(interactive=is_llama),  # server_mtp_chk
    )


def _fmt_uptime(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _render_status_html(details: dict | None, healthy: bool | None) -> str:
    if details is None:
        return '<div style="color:#6B6B6B;font-size:13px;padding:8px;border:1px dashed #ddd;border-radius:6px;">No local server — start one or switch to External API.</div>'
    pid = details.get("pid") or "—"
    port = details.get("port", "—")
    model = details.get("model", "—")
    url = details.get("url", "—")
    uptime = _fmt_uptime(details.get("uptime_s", 0)) if details.get("uptime_s") else "—"
    latency = details.get("health_latency_ms")
    latency_s = f"{latency:.0f} ms" if isinstance(latency, (int, float)) else "—"
    log_lines = details.get("log_lines", 0)
    health_dot = "🟢" if healthy else ("🟡" if details.get("pid") else "🔴")
    health_txt = (
        "Healthy" if healthy else ("Starting" if details.get("pid") else "Stopped")
    )
    # Shorten model for display
    short_model = model.split("/")[-1] if "/" in str(model) else str(model)
    if len(short_model) > 48:
        short_model = short_model[:46] + "…"
    return f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
  <div><b>{health_dot} {health_txt}</b><br/><span style="color:#6B6B6B">PID {pid} · Port {port}</span></div>
  <div style="text-align:right;color:#6B6B6B">Uptime {uptime}<br/>Health {latency_s} · {log_lines} log lines</div>
  <div style="grid-column:1/-1;color:#333;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{model}">📦 {short_model}</div>
  <div style="grid-column:1/-1;color:#6B6B6B;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{url}">{url}</div>
</div>
"""


def get_server_status_and_logs():
    with state.server_lock:
        if state.server_manager is None:
            return (
                "No server instance exists.\nTip: Start a local server below or switch Endpoint Mode to External API.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
                _render_status_html(None, None),
            )
        mgr = state.server_manager
        # Prefer detailed status if manager implements it
        try:
            details = (
                mgr.get_detailed_status()
                if hasattr(mgr, "get_detailed_status")
                else None
            )
        except Exception:
            details = None
        if details is None:
            # Fallback minimal details
            pid = (
                mgr.process.pid if mgr.process and mgr.process.poll() is None else None
            )
            details = {
                "pid": pid,
                "port": getattr(mgr, "port", "—"),
                "host": getattr(mgr, "host", "—"),
                "url": getattr(mgr, "server_url", "—"),
                "model": getattr(mgr, "model", "—"),
                "uptime_s": 0,
                "health_latency_ms": getattr(mgr, "_last_health_latency_ms", None),
                "log_lines": len(getattr(mgr, "logs", [])),
            }
        is_dead = mgr.process and mgr.process.poll() is not None
        if is_dead:
            exit_code = mgr.process.poll()
            logs = mgr.get_logs()
            # Keep last ~80 lines instead of raw char slice so timestamps stay intact
            tail = "\n".join(logs.splitlines()[-80:])
            return (
                f"Server process is dead (Exit code: {exit_code}).\n\n--- Last 80 log lines ---\n{tail}",
                '<span class="status-badge badge-error">CRASHED</span>',
                _render_status_html(details, False),
            )
        # Health check (also updates latency)
        healthy = mgr.is_healthy()
        logs = mgr.get_logs()
        tail = "\n".join(logs.splitlines()[-120:])
        badge = (
            f'<span class="status-badge badge-running">RUNNING (Port {mgr.port}) · {details.get("health_latency_ms", 0):.0f}ms</span>'
            if healthy and isinstance(details.get("health_latency_ms"), (int, float))
            else f'<span class="status-badge badge-running">RUNNING (Port {mgr.port})</span>'
            if healthy
            else '<span class="status-badge badge-starting">STARTING...</span>'
        )
        prefix = (
            "Server is healthy and running."
            if healthy
            else "Server is starting or unhealthy."
        )
        return (
            f"{prefix}\nUptime {_fmt_uptime(details.get('uptime_s', 0))} · {len(logs.splitlines())} lines\n\n--- Last 120 log lines ---\n{tail}",
            badge,
            _render_status_html(details, healthy),
        )


def clear_server_logs():
    """Clear in-memory logs; keeps server running."""
    with state.server_lock:
        if state.server_manager is None:
            return "No server to clear.", _render_status_html(None, None)
        with state.server_manager.log_lock:
            state.server_manager.logs.clear()
            # leave a marker so user sees action
            ts = time.strftime("%H:%M:%S")
            state.server_manager.logs.append(f"[{ts}] [UI] Logs cleared by user.\n")
        return state.server_manager.get_logs()[-2000:], _render_status_html(
            state.server_manager.get_detailed_status()
            if hasattr(state.server_manager, "get_detailed_status")
            else None,
            state.server_manager.is_healthy(),
        )


def download_server_logs():
    """Write full logs to a temp file and return its path for gr.File download."""
    with state.server_lock:
        if state.server_manager is None:
            # Create a small temp file with message so Download still works
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".log", mode="w", encoding="utf-8"
            )
            tmp.write("No server instance — no logs to download.\n")
            tmp.close()
            return tmp.name
        logs = state.server_manager.get_logs()
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".log", mode="w", encoding="utf-8"
        )
        # Add header with status
        try:
            details = (
                state.server_manager.get_detailed_status()
                if hasattr(state.server_manager, "get_detailed_status")
                else {}
            )
            header = (
                f"# LLM Server Logs\n# Model: {details.get('model', '?')}\n"
                f"# URL: {details.get('url', '?')}  PID: {details.get('pid', '?')}  "
                f"Uptime: {_fmt_uptime(details.get('uptime_s', 0))}\n"
                f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# Lines: {len(logs.splitlines())}\n\n"
            )
            tmp.write(header)
        except Exception:
            pass
        tmp.write(logs)
        tmp.close()
        return tmp.name


def _build_server_tab(server_status_badge: gr.HTML) -> Dict[str, Any]:
    """Build the unified Model / Endpoint tab (Local Server + External API) — symmetric + dropdown options."""

    gr.HTML('<p class="section-label">🧠 Model / Endpoint Configuration</p>')
    # ── Global endpoint mode – single source of truth for all tabs ─────
    endpoint_mode = gr.Radio(
        label="🔌 Endpoint Mode (global)",
        choices=["Local Server", "External API"],
        value="Local Server",
        info="Local Server: spawn llama-server/vLLM locally. External API: use any OpenAI-compatible endpoint (applies to Batch, Draw, Realtime).",
    )
    # Hidden checkbox for backward-compat with existing handlers that check use_external_api_chk
    use_external_api_chk = gr.Checkbox(value=False, visible=False)

    # ── Status hero — always visible, Bauhaus card ──
    with gr.Group(elem_classes=["config-card"]):
        gr.HTML('<div class="config-card-title">📡 Server Status — live</div>')
        gr.HTML(
            '<p style="color:#6B6B6B;font-size:12px;margin:4px 0 0 0;">Start a local VLM or connect to an external API. All tabs respect this Endpoint Mode.</p>'
        )

    with gr.Row(equal_height=False, elem_classes=["draw-tab-row", "twin-screens-row"]):
        # ── Left: Quick Start + Advanced (progressive disclosure) ──
        with gr.Column(scale=1, min_width=420):
            gr.HTML('<p class="section-label">🖥️ Local Server — Quick Start</p>')
            with gr.Accordion(
                "⚙️ Server Options — Model, Runtime & Advanced", open=True
            ):
                with gr.Group(visible=True) as local_server_group:
                    gr.HTML(_section_title("🦙", "Local Model Selection"))
                    server_preset = gr.Dropdown(
                        label="Recommended Model Presets",
                        choices=MODEL_PRESETS,
                        value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                        interactive=True,
                    )
                    server_backend = gr.Dropdown(
                        label="Server Backend",
                        choices=[
                            ("llama.cpp (native binary)", "llama_cpp"),
                            ("llama-cpp-python (bundled server)", "llama_cpp_python"),
                            ("vLLM (CUDA)", "vllm"),
                        ],
                        value="llama_cpp",
                        interactive=True,
                        info="llama.cpp spawns the native 'llama-server' binary; "
                        "llama-cpp-python uses the OpenAI-compatible server bundled in "
                        "the llama-cpp-python package; vLLM serves HuggingFace models "
                        "on CUDA.",
                    )
                    server_model_input = gr.Textbox(
                        label="Model GGUF Path or HF Repo ID",
                        value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                        placeholder="e.g. C:/models/qwen.gguf or HF ID",
                        interactive=True,
                    )
                    gr.HTML(_section_title("⚙️", "Runtime Options"))
                    server_port_input = gr.Number(
                        label="Port Number",
                        value=8080,
                        precision=0,
                        interactive=True,
                    )
                    with gr.Row():
                        server_thinking_chk = gr.Checkbox(
                            label="Thinking Mode", value=False, interactive=True
                        )
                        server_mtp_chk = gr.Checkbox(
                            label="MTP Speculative Drafting",
                            value=True,
                            interactive=True,
                        )

                    with gr.Accordion("Advanced Server Parameters", open=False):
                        # Backend-aware groups — llama-only vs vLLM options are
                        # shown/hidden by server_backend.change (less noise).
                        with gr.Group(visible=True) as llama_advanced_group:
                            gr.HTML(_section_title("🖧", "Network"))
                            server_host_input = gr.Textbox(
                                label="Host Binding", value="0.0.0.0"
                            )
                            gr.HTML(_section_title("🎛️", "Compute"))
                            server_ctx_input = gr.Number(
                                label="Context Size per Slot",
                                value=10000,
                                precision=0,
                            )
                            server_parallel_slots_input = gr.Number(
                                label="Parallel Slots", value=1, precision=0
                            )
                            server_gpu_layers = gr.Number(
                                label="GPU Layers (-ngl)", value=-1, precision=0
                            )
                            server_kv_cache = gr.Dropdown(
                                label="KV Cache Type",
                                choices=[
                                    "f32",
                                    "f16",
                                    "bf16",
                                    "q8_0",
                                    "q4_0",
                                    "q4_1",
                                    "iq4_nl",
                                    "q5_0",
                                    "q5_1",
                                ],
                                value="q4_0",
                            )
                            gr.HTML(_section_title("⚡", "Batch Processing Sizes"))
                            with gr.Row():
                                server_batch_size = gr.Number(
                                    label="Batch Size (-b / --batch-size)",
                                    value=1024,
                                    precision=0,
                                    info="Logical batch size for prompt processing.",
                                )
                                server_ubatch_size = gr.Number(
                                    label="Micro-Batch Size (-ub / --ubatch-size)",
                                    value=1024,
                                    precision=0,
                                    info="Physical micro-batch size submitted to GPU.",
                                )
                            gr.HTML(_section_title("🖼️", "Vision / Image Tokens"))
                            with gr.Row():
                                server_img_min_tokens = gr.Number(
                                    label="Min Image Tokens (--image-min-tokens)",
                                    value=1024,
                                    precision=0,
                                    info="Minimum tokens for image encoding. Lower = faster but lower quality.",
                                )
                                server_img_max_tokens = gr.Number(
                                    label="Max Image Tokens (--image-max-tokens)",
                                    value=4096,
                                    precision=0,
                                    info="Maximum tokens for image encoding. Higher = more detail but slower.",
                                )
                            server_log_disable = gr.Checkbox(
                                label="Disable Server Console Logs (--log-disable)",
                                value=False,
                            )

                        with gr.Group(visible=False) as vllm_advanced_group:
                            gr.HTML(_section_title("🧠", "vLLM Options"))
                            with gr.Row():
                                server_vllm_tp = gr.Number(
                                    label="Tensor Parallel Size",
                                    value=1,
                                    precision=0,
                                    info="Number of GPUs to shard the model across (vLLM).",
                                )
                                server_vllm_gpu_util = gr.Number(
                                    label="GPU Memory Utilization",
                                    value=0.90,
                                    precision=None,
                                    info="Fraction of GPU memory to use (vLLM, e.g. 0.90).",
                                )
                            server_vllm_max_seq = gr.Number(
                                label="Max Model Length",
                                value=20000,
                                precision=0,
                                info="Maximum sequence length (vLLM --max-model-len).",
                            )

                with gr.Group(visible=False) as ext_api_group:
                    gr.HTML(
                        '<div class="config-card"><div class="config-card-title">🌐 External API (global)</div>'
                        '<p style="color:#7d8590;font-size:0.85rem;margin:0;">Used by <b>Batch</b>, <b>Draw & Recognize</b> and <b>Realtime</b> tabs when Endpoint Mode is External.</p></div>'
                    )
                    ext_api_url = gr.Textbox(
                        label="Base URL",
                        value="https://api.openai.com/v1",
                        placeholder="https://api.openai.com/v1",
                    )
                    ext_api_key = gr.Textbox(
                        label="API Key",
                        placeholder="sk-...",
                        value="",
                        type="password",
                    )
                    ext_model_name = gr.Textbox(label="Model Name", value="gpt-4o")

            with gr.Row(elem_classes=["btn-group"]):
                start_server_btn = gr.Button(
                    "▶  Start Server", variant="primary", scale=2
                )
                stop_server_btn = gr.Button(
                    "⏹  Stop Server", variant="secondary", scale=1
                )

        # ── Right: Live Status + Logs (improved) ─────────────────────────
        with gr.Column(scale=1, min_width=420):
            gr.HTML('<p class="section-label">📡 Live Server Status</p>')
            with gr.Group(elem_classes=["config-card"]):
                server_details = gr.HTML(value=_render_status_html(None, None))

            gr.HTML('<p class="section-label">Server Output Console</p>')
            gr.HTML(
                '<div class="output-panel" id="server-log-panel">'
                + panel_header(
                    "Live Logs — auto-refresh 5s, timestamps added", "server-log-ta"
                )
            )
            with gr.Group(elem_classes=["out-md-wrap"]):
                server_logs_viewer = gr.Textbox(
                    lines=22,
                    max_lines=32,
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="server-log-ta",
                    placeholder="Logs appear here after you start a local server. External API mode has no local logs.",
                )
            with gr.Row(elem_classes=["btn-group"]):
                clear_logs_btn = gr.Button(
                    "🗑 Clear", variant="secondary", size="sm", scale=1
                )
                download_logs_btn = gr.Button(
                    "⬇ Download logs", variant="secondary", size="sm", scale=1
                )
                refresh_logs_btn = gr.Button(
                    "↻ Refresh", variant="secondary", size="sm", scale=1
                )
            # Hidden file for log download (triggered by Download button)
            log_download_file = gr.File(visible=False)
            gr.HTML(
                '<p style="color:#6B6B6B;font-size:11px;margin:4px 0 0 0;">'
                "Logs are timestamped <code>[HH:MM:SS]</code> on capture, kept last 3000 lines in RAM, "
                "and also streamed to the Python <code>detection_pipeline.server</code> logger.</p>"
            )
            gr.HTML("</div>")

    return dict(
        endpoint_mode=endpoint_mode,
        use_external_api_chk=use_external_api_chk,
        local_server_group=local_server_group,
        ext_api_group=ext_api_group,
        ext_api_url=ext_api_url,
        ext_api_key=ext_api_key,
        ext_model_name=ext_model_name,
        server_preset=server_preset,
        server_backend=server_backend,
        server_model_input=server_model_input,
        server_port_input=server_port_input,
        server_host_input=server_host_input,
        server_thinking_chk=server_thinking_chk,
        server_mtp_chk=server_mtp_chk,
        server_ctx_input=server_ctx_input,
        server_parallel_slots_input=server_parallel_slots_input,
        server_gpu_layers=server_gpu_layers,
        server_kv_cache=server_kv_cache,
        server_batch_size=server_batch_size,
        server_ubatch_size=server_ubatch_size,
        server_vllm_tp=server_vllm_tp,
        server_vllm_gpu_util=server_vllm_gpu_util,
        server_vllm_max_seq=server_vllm_max_seq,
        server_img_min_tokens=server_img_min_tokens,
        server_img_max_tokens=server_img_max_tokens,
        server_log_disable=server_log_disable,
        llama_advanced_group=llama_advanced_group,
        vllm_advanced_group=vllm_advanced_group,
        start_server_btn=start_server_btn,
        stop_server_btn=stop_server_btn,
        server_logs_viewer=server_logs_viewer,
        server_details=server_details,
        clear_logs_btn=clear_logs_btn,
        download_logs_btn=download_logs_btn,
        refresh_logs_btn=refresh_logs_btn,
        log_download_file=log_download_file,
    )
