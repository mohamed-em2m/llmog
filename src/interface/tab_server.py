"""
Llama Server tab UI and lifecycle server management logic.
"""

import html
import logging
from typing import Dict, Any
import gradio as gr

from servers import LlamaServerManager
from interface.state import server_manager, server_lock, _STATUS_PILL, LOG_TAIL_BYTES, MODEL_PRESETS, handle_preset_change

logger = logging.getLogger("detection_pipeline")


def _get_server_status_html() -> str:
    global server_manager
    with server_lock:
        if server_manager is None:
            return '<span class="status-badge badge-stopped">STOPPED</span>'
        if server_manager.process and server_manager.process.poll() is None:
            url = f"http://{server_manager.host}:{server_manager.port}"
            return f'<span class="status-badge badge-running">RUNNING</span> <a href="{url}/health" target="_blank" style="color:#7d8590; font-size:0.75rem; text-decoration:none;">({url})</a>'
        elif server_manager.process and server_manager.process.poll() is not None:
            exit_code = server_manager.process.poll()
            return f'<span class="status-badge badge-error">CRASHED (Exit code {exit_code})</span>'
        else:
            return '<span class="status-badge badge-stopped">STOPPED</span>'


def start_server_wrapper(
    model_path, port, host, enable_thinking, enable_mtp, ctx_size, n_gpu_layers,
    cache_type_k, img_min_tokens, img_max_tokens, parallel_slots, batch_size, ubatch_size,
    disable_log=False,
):
    global server_manager
    with server_lock:
        if server_manager is not None:
            if server_manager.process and server_manager.process.poll() is None:
                return server_manager.get_logs()[-LOG_TAIL_BYTES:], _get_server_status_html()

        if not model_path:
            return "Error: Model path/name is required.", _get_server_status_html()

        kwargs = {}
        if ctx_size: kwargs["ctx_size"] = int(ctx_size)
        if n_gpu_layers is not None and n_gpu_layers != -1: kwargs["gpu_layers"] = int(n_gpu_layers)
        if cache_type_k: kwargs["kv_cache_type"] = cache_type_k
        if img_min_tokens: kwargs["image_min_tokens"] = int(img_min_tokens)
        if img_max_tokens: kwargs["image_max_tokens"] = int(img_max_tokens)
        if parallel_slots: kwargs["parallel_slots"] = int(parallel_slots)
        if batch_size: kwargs["batch_size"] = int(batch_size)
        if ubatch_size: kwargs["ubatch_size"] = int(ubatch_size)

        try:
            spec_type = "draft-mtp" if enable_mtp else "none"
            server_manager = LlamaServerManager(
                model=model_path,
                port=int(port),
                host=host,
                enable_thinking=enable_thinking,
                spec_type=spec_type,
                log_disable=bool(disable_log),
                **kwargs,
            )
            server_manager.start_llama_server()
            logs = server_manager.get_logs()[-LOG_TAIL_BYTES:]
            return logs, _get_server_status_html()
        except Exception as exc:
            logger.exception("Failed to start server")
            return f"Failed to start server: {exc}", '<span class="status-badge badge-error">ERROR</span>'


def stop_server_wrapper():
    global server_manager
    with server_lock:
        if server_manager is not None:
            try:
                server_manager.stop_llama_server()
            except Exception as exc:
                logger.warning("Error stopping server: %s", exc)
            logs = server_manager.get_logs()[-LOG_TAIL_BYTES:]
            server_manager = None
            return logs, '<span class="status-badge badge-stopped">STOPPED</span>'
        return "Server is not running.", '<span class="status-badge badge-stopped">STOPPED</span>'


def get_server_status_and_logs():
    global server_manager
    with server_lock:
        if server_manager is None:
            return "", '<span class="status-badge badge-stopped">STOPPED</span>'
        logs = server_manager.get_logs()[-LOG_TAIL_BYTES:]
        return logs, _get_server_status_html()


def _build_server_tab(server_status_badge: gr.HTML) -> Dict[str, Any]:
    gr.HTML('<p class="section-label">Server Instance Manager</p>')

    with gr.Row():
        with gr.Column(scale=1):
            server_preset = gr.Dropdown(
                label="Model Preset",
                choices=MODEL_PRESETS,
                value=MODEL_PRESETS[0],
            )
            server_model_input = gr.Textbox(
                label="Model (HuggingFace repo/file or local path)",
                value=MODEL_PRESETS[0],
            )
            with gr.Row():
                server_host_input = gr.Textbox(label="Host", value="127.0.0.1", scale=2)
                server_port_input = gr.Number(label="Port", value=8080, precision=0, scale=1)

            with gr.Row():
                server_thinking_chk = gr.Checkbox(label="Enable Thinking Mode", value=True)
                server_mtp_chk = gr.Checkbox(label="Enable MTP Mode", value=True)

            with gr.Accordion("Advanced Server Flags", open=False):
                with gr.Row():
                    server_ctx_input = gr.Number(label="Context Size (-c)", value=16384, precision=0)
                    server_gpu_layers = gr.Number(label="GPU Layers (-ngl)", value=-1, precision=0)
                with gr.Row():
                    server_cache_type = gr.Dropdown(
                        label="KV Cache K Type (-ctk)",
                        choices=["f16", "q8_0", "q4_0"],
                        value="q8_0",
                    )
                    server_parallel_slots = gr.Number(label="Parallel Slots (-np)", value=1, precision=0)
                with gr.Row():
                    server_img_min = gr.Number(label="Min Image Tokens", value=1024, precision=0)
                    server_img_max = gr.Number(label="Max Image Tokens", value=4096, precision=0)
                with gr.Row():
                    server_batch_size = gr.Number(label="Batch Size (-b)", value=2048, precision=0)
                    server_ubatch_size = gr.Number(label="Micro Batch Size (-ub)", value=512, precision=0)
                with gr.Row():
                    server_log_disable = gr.Checkbox(label="Disable Server Console Logs (--log-disable)", value=False)

            with gr.Row():
                start_server_btn = gr.Button("🚀 Start Server", variant="primary", scale=2)
                stop_server_btn = gr.Button("🛑 Stop Server", variant="stop", scale=1)

        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">Server Logs</p>')
            server_logs_viewer = gr.Code(
                value="",
                language="markdown",
                label="stdout / stderr",
                lines=18,
                interactive=False,
            )
            refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm")
            log_timer = gr.Timer(2.0)

    return dict(
        server_preset=server_preset,
        server_model_input=server_model_input,
        server_host_input=server_host_input,
        server_port_input=server_port_input,
        server_thinking_chk=server_thinking_chk,
        server_mtp_chk=server_mtp_chk,
        server_ctx_input=server_ctx_input,
        server_gpu_layers=server_gpu_layers,
        server_kv_cache=server_cache_type,
        server_parallel_slots_input=server_parallel_slots,
        server_img_min_tokens=server_img_min,
        server_img_max_tokens=server_img_max,
        server_batch_size=server_batch_size,
        server_ubatch_size=server_ubatch_size,
        server_log_disable=server_log_disable,
        start_server_btn=start_server_btn,
        stop_server_btn=stop_server_btn,
        server_logs_viewer=server_logs_viewer,
        refresh_logs_btn=refresh_logs_btn,
        log_timer=log_timer,
    )
