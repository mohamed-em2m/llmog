"""
Llama Server tab UI and lifecycle server management logic.
"""

import time
import logging
from typing import Dict, Any
import gradio as gr

from servers import LlamaServerManager
from servers.llama_server_manager import num_gpus as _num_gpus
from interface.state import (
    state,
    LOG_TAIL_BYTES,
    MODEL_PRESETS,
    panel_header,
    _section_title,
    handle_preset_change,
)

logger = logging.getLogger("detection_pipeline")


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
):
    ctx_size = ctx_size * parallel_slots

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
                state.server_manager.stop_llama_server()
            except Exception as e:
                logger.warning(f"Error stopping old server: {e}")
            state.server_manager = None

        yield (
            "Configuring server...",
            '<span class="status-badge badge-starting">INITIALIZING...</span>',
        )

        tensor_split = ",".join(["1"] * _num_gpus)
        spec_type = "draft-mtp" if enable_mtp else "none"
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
            "Spawning llama-server process...",
            '<span class="status-badge badge-starting">STARTING...</span>',
        )
        try:
            state.server_manager.start_llama_server()
        except Exception as e:
            state.server_manager = None
            yield (
                f"Failed to start server process: {e}",
                '<span class="status-badge badge-error">PROCESS ERROR</span>',
            )
            return

    start_time = time.time()
    timeout = 180
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
            state.server_manager.stop_llama_server()
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


def get_server_status_and_logs():
    with state.server_lock:
        if state.server_manager is None:
            return (
                "No server instance exists.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        if (
            state.server_manager.process
            and state.server_manager.process.poll() is not None
        ):
            exit_code = state.server_manager.process.poll()
            return (
                f"Server process is dead (Exit code: {exit_code}).\n\n--- Logs ---\n{state.server_manager.get_logs()}",
                '<span class="status-badge badge-error">CRASHED</span>',
            )
        logs = state.server_manager.get_logs()
        if state.server_manager.is_healthy():
            return (
                f"Server is healthy and running.\n\n--- Logs ---\n{logs[-2000:]}",
                f'<span class="status-badge badge-running">RUNNING (Port {state.server_manager.port})</span>',
            )
        return (
            f"Server is starting or unhealthy.\n\n--- Logs ---\n{logs[-2000:]}",
            '<span class="status-badge badge-starting">STARTING...</span>',
        )


def _build_server_tab(server_status_badge: gr.HTML) -> Dict[str, Any]:
    """Build the Llama Server tab and return all interactive components."""

    gr.HTML('<p class="section-label">Model Server Configuration</p>')
    with gr.Row(equal_height=False):
        # ── Left: Config ──────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML(
                '<div class="config-card"><div class="config-card-title">🦙 Model Selection</div>'
            )
            server_preset = gr.Dropdown(
                label="Recommended Model Presets",
                choices=MODEL_PRESETS,
                value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                interactive=True,
            )
            server_model_input = gr.Textbox(
                label="Model GGUF Path or HF Repo ID",
                value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                placeholder="e.g. C:/models/qwen.gguf or HF ID",
                interactive=True,
            )
            gr.HTML("</div>")

            gr.HTML(
                '<div class="config-card"><div class="config-card-title">⚙️ Runtime Options</div>'
            )
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
            gr.HTML("</div>")

            with gr.Accordion("Advanced Server Parameters", open=False):
                gr.HTML(_section_title("🖧", "Network"))
                server_host_input = gr.Textbox(label="Host Binding", value="0.0.0.0")
                gr.HTML(_section_title("🎛️", "Compute"))
                server_ctx_input = gr.Number(
                    label="Context Size per Slot",
                    value=10000,
                    precision=0,
                )
                server_parallel_slots_input = gr.Number(
                    label="Parallel Slots", value=2, precision=0
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
                with gr.Row():
                    server_log_disable = gr.Checkbox(
                        label="Disable Server Console Logs (--log-disable)",
                        value=False,
                    )

            gr.HTML('<div class="btn-group" style="margin-top:0.75rem;">')
            with gr.Row():
                start_server_btn = gr.Button("▶  Start Server", variant="primary")
                stop_server_btn = gr.Button(
                    "⏹  Stop Server", variant="secondary", size="sm"
                )
                refresh_logs_btn = gr.Button(
                    "🔄 Refresh Logs",
                    variant="secondary",
                    size="sm",
                )
            gr.HTML("</div>")

        # ── Right: Logs ───────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<p class="section-label">Server Output Console</p>')
            gr.HTML(
                '<div class="output-panel" id="server-log-panel">'
                + panel_header("Live Logs", "server-log-ta")
            )
            with gr.Group(elem_classes=["out-md-wrap"]):
                server_logs_viewer = gr.Textbox(
                    lines=22,
                    max_lines=32,
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="server-log-ta",
                )
            gr.HTML("</div>")

    return dict(
        server_preset=server_preset,
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
        server_img_min_tokens=server_img_min_tokens,
        server_img_max_tokens=server_img_max_tokens,
        server_log_disable=server_log_disable,
        start_server_btn=start_server_btn,
        stop_server_btn=stop_server_btn,
        refresh_logs_btn=refresh_logs_btn,
        server_logs_viewer=server_logs_viewer,
    )
