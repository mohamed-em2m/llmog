"""
Aggregator application builder assembling interface tabs, CSS styling, and event handlers.
"""

import warnings

# Suppress noisy Starlette/Gradio deprecation (HTTP_422_UNPROCESSABLE_ENTITY → CONTENT)
# Gradio 6 routes.py:1379 still uses the old name; fixed upstream in 5.8+ but we pin 6.19
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_CONTENT.*")
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="gradio.*")
except Exception:
    pass
try:
    from starlette.warnings import StarletteDeprecationWarning  # type: ignore

    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except Exception:
    pass

import gradio as gr

from interface.console_theme import theme
from interface.state import (
    custom_css,
    CONSOLE_JS,
    handle_preset_change,
    toggle_custom_color_field,
    _cache_get as _hero_cache_get,
)
from interface.viewer_utils import pipeline_detections_to_annotations, build_viewer_payload as _build_hero_payload
from interface.tab_server import (
    _build_server_tab,
    start_server_wrapper,
    stop_server_wrapper,
    get_server_status_and_logs,
    on_server_backend_change,
)
from interface.tab_batch import (
    _build_batch_tab,
    toggle_run_btn,
    run_batch_dispatcher,
    cancel_pipeline,
    on_explorer_image_change,
    on_explorer_round_change,
    on_explorer_prev,
    on_explorer_next,
    on_explorer_first,
    on_explorer_last,
    on_explorer_pos,
)
from interface.batch.components import (
    on_batch_preset_change,
    on_batch_strategy_change,
    handle_batch_upload,
)
from interface.tab_prompts import _build_prompts_tab
from interface.tab_realtime import _build_realtime_tab, _wire_realtime_events
from interface.tab_draw import build_draw_tab, wire_draw_events
from interface.tab_realtime_interactive import (
    build_realtime_interactive_tab,
    wire_realtime_interactive_events,
)


def _hero_view_for_selection(selected_image: str, selected_round: str, batch_id: str):
    """Sync top hero preview to current explorer selection (instant feedback)."""
    try:
        batch_results = _hero_cache_get(batch_id)
        if not batch_results or not selected_image or selected_image not in batch_results:
            return None, None, '<div class="hero-empty">No results yet — run batch to see a large live preview here.</div>'
        img_data = batch_results[selected_image]
        viewer_base = img_data.get("raw_original")
        if viewer_base is None:
            return None, None, f'<div class="hero-meta">{selected_image} — waiting…</div>'
        # Determine which detections to show (Final Best vs specific round)
        if not selected_round or selected_round == "Final Best":
            dets = img_data.get("detections") or []
            # fallback to best round if empty
            if not dets and img_data.get("rounds"):
                try:
                    best = max(img_data["rounds"], key=lambda r: r.get("score", -1))
                    dets = best.get("detections") or []
                    score = best.get("score", "-")
                except Exception:
                    score = "-"
            else:
                # compute best score for info
                try:
                    score = max(r.get("score", -1) for r in img_data.get("rounds", [])) if img_data.get("rounds") else "-"
                except Exception:
                    score = "-"
        else:
            try:
                r_idx = int(selected_round) - 1
                r = img_data["rounds"][r_idx]
                dets = r.get("detections") or []
                score = r.get("score", "-")
            except Exception:
                dets = img_data.get("detections") or []
                score = "-"
        anns = pipeline_detections_to_annotations(dets, viewer_base.size) if dets else []
        payload = _build_hero_payload(viewer_base, anns)
        # source image is raw_original (keep grid off for hero to avoid double grid)
        info = (
            f'<div class="hero-meta">'
            f'<span class="hero-title">{selected_image}</span> '
            f'<span class="score-badge">Score: {score}/10</span> '
            f'<span class="hero-count">{len(dets)} detections</span>'
            f'</div>'
        )
        return viewer_base, payload, info
    except Exception:
        return None, None, '<div class="hero-empty">Preview unavailable</div>'


def _on_endpoint_mode_change(mode: str):
    is_ext = mode == "External API"
    return (
        gr.update(visible=not is_ext),  # local_server_group
        gr.update(visible=is_ext),  # ext_api_group
        is_ext,  # use_external_api_chk (hidden bool)
        gr.update(interactive=not is_ext),  # start_server_btn
        gr.update(interactive=not is_ext),  # stop_server_btn
        gr.update(interactive=not is_ext),  # server_preset
        gr.update(interactive=not is_ext),  # server_backend
        gr.update(interactive=not is_ext),  # server_model_input
        gr.update(interactive=not is_ext),  # server_port_input
        gr.update(interactive=not is_ext),  # server_thinking_chk
        gr.update(interactive=not is_ext),  # server_mtp_chk
    )


def _wire_events(
    c_srv, c_bat, c_pmt, server_status_badge, batch_id_state, write_yolo_state
):
    """Wire all event handlers across server, batch, and prompt tabs."""

    # ── Server tab ────────────────────────────────────────────────────────
    c_srv["endpoint_mode"].change(
        _on_endpoint_mode_change,
        inputs=[c_srv["endpoint_mode"]],
        outputs=[
            c_srv["local_server_group"],
            c_srv["ext_api_group"],
            c_srv["use_external_api_chk"],
            c_srv["start_server_btn"],
            c_srv["stop_server_btn"],
            c_srv["server_preset"],
            c_srv["server_backend"],
            c_srv["server_model_input"],
            c_srv["server_port_input"],
            c_srv["server_thinking_chk"],
            c_srv["server_mtp_chk"],
        ],
    )

    c_srv["server_preset"].change(
        handle_preset_change,
        c_srv["server_preset"],
        c_srv["server_model_input"],
    )

    # Backend-aware advanced options: llama-only vs vLLM groups
    c_srv["server_backend"].change(
        on_server_backend_change,
        inputs=[c_srv["server_backend"]],
        outputs=[
            c_srv["llama_advanced_group"],
            c_srv["vllm_advanced_group"],
            c_srv["server_mtp_chk"],
        ],
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
            c_srv["server_backend"],
            c_srv["server_vllm_tp"],
            c_srv["server_vllm_gpu_util"],
            c_srv["server_vllm_max_seq"],
        ],
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )
    c_srv["stop_server_btn"].click(
        stop_server_wrapper,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )
    # Logs auto-refresh every 5s via gr.Timer below — no manual refresh button.

    # ── Batch tab — category strategy & presets ──────────────────────────
    c_bat["category_strategy"].change(
        fn=on_batch_strategy_change,
        inputs=[c_bat["category_strategy"]],
        outputs=[
            c_bat["categories_input"],
            c_bat["category_defs_input"],
            c_bat["category_preset_dropdown"],
        ],
    )

    c_bat["category_preset_dropdown"].change(
        fn=on_batch_preset_change,
        inputs=[c_bat["category_preset_dropdown"]],
        outputs=[c_bat["categories_input"], c_bat["category_defs_input"]],
    )

    # ── Batch tab — preprocessing toggles ────────────────────────────────
    c_bat["prep_enabled_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_enabled_chk"]],
        outputs=[c_bat["prep_options_group"]],
    )

    c_bat["prep_custom_resize_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_custom_resize_chk"]],
        outputs=[c_bat["prep_custom_resize_row"]],
    )

    for dd, custom_field in [
        (c_bat["prep_grid_line_color_dropdown"], c_bat["prep_grid_line_color_custom"]),
        (c_bat["prep_grid_text_color_dropdown"], c_bat["prep_grid_text_color_custom"]),
        (
            c_bat["prep_grid_backing_color_dropdown"],
            c_bat["prep_grid_backing_color_custom"],
        ),
    ]:
        dd.change(toggle_custom_color_field, inputs=[dd], outputs=[custom_field])

    c_bat["prep_send_pixel_bounds_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_send_pixel_bounds_chk"]],
        outputs=[c_bat["prep_pixel_bounds_row"]],
    )

    # ── Batch tab — Upload persistence (fixes UploadButton transient bug) ──
    c_bat["input_images"].upload(
        fn=handle_batch_upload,
        inputs=[c_bat["input_images"], c_bat["upload_state"]],
        outputs=[c_bat["upload_state"], c_bat["source_image_viewer"], c_bat["pipeline_status"]],
    )

    # ── Run / Cancel ──────────────────────────────────────────────────────
    c_bat["run_btn"].click(
        fn=lambda: toggle_run_btn(is_running=True),
        inputs=None,
        outputs=[c_bat["run_btn"], c_bat["stop_run_btn"]],
        queue=False,
    ).then(
        fn=run_batch_dispatcher,
        inputs=[
            c_bat["upload_state"],
            c_bat["categories_input"],
            c_bat["category_defs_input"],
            c_srv["server_port_input"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
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
            c_bat["prep_min_pixels_num"],
            c_bat["prep_max_pixels_num"],
            c_bat["prep_custom_resize_chk"],
            c_bat["prep_custom_resize_width"],
            c_bat["prep_custom_resize_height"],
            write_yolo_state,
            c_bat["category_strategy"],
        ],
        outputs=[
            c_bat["pipeline_status"],
            c_bat["progress_html"],
            c_bat["download_results_box"],
            batch_id_state,
            c_bat["explorer_image_select"],
            c_bat["pipeline_logs_viewer"],
            c_bat["batch_status_table"],
            c_bat["hero_source_image"],
            c_bat["hero_viewer"],
            c_bat["hero_info"],
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

    # ── Explorer interactions ─────────────────────────────────────────────
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

    _hero_outputs = [
        c_bat["hero_source_image"],
        c_bat["hero_viewer"],
        c_bat["hero_info"],
    ]
    _hero_inputs = [
        c_bat["explorer_image_select"],
        c_bat["explorer_round_select"],
        batch_id_state,
    ]

    c_bat["explorer_image_select"].change(
        on_explorer_image_change,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_round_select"]],
    ).then(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    ).then(
        _hero_view_for_selection,
        inputs=_hero_inputs,
        outputs=_hero_outputs,
    ).then(
        on_explorer_pos,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_pos_display"]],
        queue=False,
    )

    # ── Arrow / jump navigation for batch explorer ──
    # Each button just moves the image dropdown; the dropdown's .change chain
    # above refreshes rounds, viewer, hero and the position badge.
    c_bat["explorer_first_btn"].click(
        on_explorer_first,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_image_select"]],
        queue=False,
    )
    c_bat["explorer_prev_btn"].click(
        on_explorer_prev,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_image_select"]],
        queue=False,
    )
    c_bat["explorer_next_btn"].click(
        on_explorer_next,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_image_select"]],
        queue=False,
    )
    c_bat["explorer_last_btn"].click(
        on_explorer_last,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_image_select"]],
        queue=False,
    )

    # Round dropdown changed: refresh all explorer outputs
    c_bat["explorer_round_select"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    ).then(
        _hero_view_for_selection,
        inputs=_hero_inputs,
        outputs=_hero_outputs,
    )

    c_bat["show_grid_chk"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="LLM Object Detection Console") as app:
        # Gradio 6: <script> inside gr.HTML value never executes (innerHTML);
        # head= injects it into <head> where it actually runs.
        gr.HTML(value="", head=CONSOLE_JS)

        # ── Header with inline status badge ──────────────────────────────
        gr.HTML(
            """
        <div class="app-header">
            <div>
                <h1><span>🔍</span> LLM Object Detection Console</h1>
                <p>// vision-LLM detector/judge pipeline · local or external endpoint</p>
            </div>
            <div class="app-header-meta" id="header-status-meta">
            </div>
        </div>"""
        )

        server_status_badge = gr.HTML(
            value='<span class="status-badge badge-stopped">STOPPED</span>',
        )

        batch_id_state = gr.State("")
        write_yolo_state = gr.State(False)

        with gr.Tabs():
            with gr.TabItem("🧠 Model / Endpoint"):
                c_srv = _build_server_tab(server_status_badge)

            with gr.TabItem("🎨 Draw & Recognize"):
                c_draw = build_draw_tab()

            with gr.TabItem("🗂️ Batch Processing"):
                c_bat = _build_batch_tab()

            with gr.TabItem("✍️ Prompts"):
                c_pmt = _build_prompts_tab()

            with gr.TabItem("🎥 Real-Time Detection"):
                c_rt = _build_realtime_tab()

            with gr.TabItem("🎯 Real-Time Draw"):
                c_rt_interactive = build_realtime_interactive_tab()

        # ── Wire all events ───────────────────────────────────────────────
        _wire_events(
            c_srv, c_bat, c_pmt, server_status_badge, batch_id_state, write_yolo_state
        )
        wire_draw_events(c_draw, c_srv, c_bat)
        _wire_realtime_events(c_rt, c_srv, c_bat)
        wire_realtime_interactive_events(c_rt_interactive, c_srv)

        # ── Auto-refresh server status every 5 s ─────────────────────────
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
