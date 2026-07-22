"""
Aggregator application builder assembling interface tabs, CSS styling, and event handlers.
"""

import gradio as gr

from interface.console_theme import theme
from interface.state import custom_css, CONSOLE_JS
from interface.tab_server import _build_server_tab, get_server_status_and_logs
from interface.tab_batch import (
    _build_batch_tab,
    toggle_external_api,
    toggle_run_btn,
    run_batch_detection_gui,
    cancel_pipeline,
    on_explorer_image_change,
    on_explorer_round_change,
)
from interface.tab_prompts import _build_prompts_tab
from interface.tab_realtime import _build_realtime_tab, _wire_realtime_events


def _wire_events(c_srv, c_bat, c_pmt, server_status_badge, batch_id_state):
    """Wire all event handlers across server, batch, and prompt tabs."""
    from interface.state import handle_preset_change
    from interface.tab_server import start_server_wrapper, stop_server_wrapper

    c_srv["server_preset"].change(
        handle_preset_change,
        c_srv["server_preset"],
        c_srv["server_model_input"],
    )

    c_srv["start_server_btn"].click(
        start_server_wrapper,
        inputs=[
            c_srv["server_model_input"],
            c_srv["server_port_input"],
            c_srv["server_host_input"],
            c_srv["server_thinking_chk"],
            c_srv["server_mtp_chk"],
            c_srv["server_ctx_input"],
            c_srv["server_gpu_layers"],
            c_srv["server_kv_cache"],
            c_srv["server_img_min_tokens"],
            c_srv["server_img_max_tokens"],
            c_srv["server_parallel_slots_input"],
            c_srv["server_batch_size"],
            c_srv["server_ubatch_size"],
            c_srv["server_log_disable"],
        ],
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )

    c_srv["stop_server_btn"].click(
        stop_server_wrapper,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )

    c_srv["refresh_logs_btn"].click(
        get_server_status_and_logs,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )

    c_srv["log_timer"].tick(
        get_server_status_and_logs,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )

    c_bat["use_external_api_chk"].change(
        toggle_external_api,
        inputs=[c_bat["use_external_api_chk"]],
        outputs=[
            c_srv["start_server_btn"],
            c_srv["stop_server_btn"],
            c_srv["server_preset"],
            c_srv["server_model_input"],
            c_srv["server_port_input"],
            c_srv["server_thinking_chk"],
            c_srv["server_mtp_chk"],
            c_bat["ext_api_group"],
        ],
    )

    c_bat["run_btn"].click(
        fn=lambda: toggle_run_btn(is_running=True),
        inputs=None,
        outputs=[c_bat["run_btn"], c_bat["stop_run_btn"]],
        queue=False,
    ).then(
        fn=run_batch_detection_gui,
        inputs=[
            c_bat["input_images"],
            c_bat["categories_input"],
            c_bat["category_defs_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_bat["rounds_slider"],
            c_bat["score_threshold_slider"],
            c_bat["det_temp_slider"],
            c_bat["jdg_temp_slider"],
            c_bat["concurrency_slider"],
            c_pmt["customize_prompts_chk"],
            c_pmt["custom_det_prompt"],
            c_pmt["custom_jdg_prompt"],
            c_bat["prep_enabled_chk"],
            c_bat["prep_short_edge_slider"],
            c_bat["prep_pad_square_chk"],
            c_bat["prep_contrast_dropdown"],
            c_bat["prep_gamma_slider"],
            c_bat["prep_denoise_dropdown"],
            c_bat["prep_sharpen_chk"],
            c_bat["prep_wb_chk"],
            c_bat["prep_grid_dropdown"],
            c_bat["prep_som_chk"],
            c_bat["prep_tiling_chk"],
            c_bat["prep_tile_size_slider"],
            c_bat["prep_tile_overlap_slider"],
            c_bat["prep_cv_chk"],
            c_bat["prep_cv_padding_slider"],
            c_bat["prep_grid_step_slider"],
            c_bat["prep_grid_line_width_slider"],
            c_bat["prep_grid_font_size_slider"],
            c_bat["prep_grid_line_color_dropdown"],
            c_bat["prep_grid_line_color_custom"],
            c_bat["prep_grid_text_color_dropdown"],
            c_bat["prep_grid_text_color_custom"],
            c_bat["prep_grid_backing_color_dropdown"],
            c_bat["prep_grid_backing_color_custom"],
            c_bat["prep_send_pixel_bounds_chk"],
            c_bat["prep_min_pixels"],
            c_bat["prep_max_pixels"],
            c_bat["prep_custom_resize_chk"],
            c_bat["prep_custom_resize_width"],
            c_bat["prep_custom_resize_height"],
            c_bat["judge_thinking_chk"],
            c_bat["feedback_image_mode_dropdown"],
        ],
        outputs=[
            c_bat["pipeline_status"],
            c_bat["progress_html"],
            c_bat["download_results_box"],
            batch_id_state,
            c_bat["explorer_image_select"],
            c_bat["pipeline_logs_viewer"],
            c_bat["batch_status_table"],
        ],
        concurrency_limit=1,
    ).then(
        fn=lambda: toggle_run_btn(is_running=False),
        inputs=None,
        outputs=[c_bat["run_btn"], c_bat["stop_run_btn"]],
        queue=False,
    )

    c_bat["stop_run_btn"].click(
        fn=cancel_pipeline,
        outputs=[c_bat["pipeline_status"]],
        queue=False,
    )

    _explorer_outputs = [
        c_bat["source_image_viewer"],
        c_bat["best_annotated_viewer"],
        c_bat["round_score_display"],
        c_bat["round_feedback_display"],
        c_bat["round_raw_response_display"],
        c_bat["round_parse_error_display"],
        c_bat["detections_json_box"],
    ]
    _explorer_inputs = [
        c_bat["explorer_image_select"],
        c_bat["explorer_round_select"],
        batch_id_state,
        c_bat["show_grid_chk"],
    ]

    c_bat["explorer_image_select"].change(
        on_explorer_image_change,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_round_select"]],
    ).then(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )

    c_bat["explorer_round_select"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )

    c_bat["show_grid_chk"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=theme, css=custom_css, title="LLM Object Detection Console"
    ) as app:
        gr.HTML(CONSOLE_JS)

        gr.HTML("""
        <div class="app-header">
            <div>
                <h1><span>🔍</span> LLM Object Detection Console</h1>
                <p>// vision-LLM detector/judge pipeline · local or external endpoint</p>
            </div>
            <div class="app-header-meta" id="header-status-meta">
            </div>
        </div>""")

        server_status_badge = gr.HTML(
            value='<span class="status-badge badge-stopped">STOPPED</span>',
        )

        batch_id_state = gr.State("")

        with gr.Tabs():
            with gr.TabItem("⚡ Real-Time & Video"):
                c_real = _build_realtime_tab()

            with gr.TabItem("🦙 Llama Server"):
                c_srv = _build_server_tab(server_status_badge)

            with gr.TabItem("🧪 Batch Sandbox"):
                c_bat = _build_batch_tab()

            with gr.TabItem("✍️ Prompts"):
                c_pmt = _build_prompts_tab()

        _wire_events(c_srv, c_bat, c_pmt, server_status_badge, batch_id_state)
        _wire_realtime_events(c_real, c_srv, c_bat)

        status_timer = gr.Timer(value=5.0)
        app.load(
            get_server_status_and_logs,
            outputs=[c_srv["server_logs_viewer"], server_status_badge],
        )
        status_timer.tick(
            get_server_status_and_logs,
            outputs=[c_srv["server_logs_viewer"], server_status_badge],
        )

    return app
