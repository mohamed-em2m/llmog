"""
Gradio UI layout and event wiring for the Realtime streaming tab.
"""

from typing import Dict, Any
import gradio as gr

from detection_viewer import DetectionViewer
from free_detection.trackers import MultiAlgorithmTracker
from interface.realtime.state import (
    SessionDetector,
    new_session_detector,
    reset_session,
    DEFAULT_HUD,
)
from interface.realtime.handlers import process_single_frame, process_video_frames


def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR (MULTI-TRACKER INTEGRATED)
            </h2>
        </div>
        """
        )
        with gr.Row():
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
                )
                c["tracker_algorithm"] = gr.Dropdown(
                    choices=MultiAlgorithmTracker.SUPPORTED_ALGOS,
                    value="CSRT (TrackerCSRT)",
                    label="REAL-TIME TRACKING ALGORITHM",
                    info=(
                        "None = show raw VLM boxes. "
                        "MOSSE/KCF/CSRT/VitTracker = OpenCV single-object trackers "
                        "that propagate boxes between VLM calls. "
                        "ByteTrack = multi-object Kalman IoU tracker."
                    ),
                )
                c["categories_input"] = gr.Textbox(
                    value="person, car, dog, bottle, phone",
                    label="Target Categories (comma-separated)",
                    info="Leave empty or type * for free/open-vocabulary detection.",
                )
                c["category_defs_input"] = gr.Textbox(
                    label="Category Definitions",
                    placeholder="Write instructions for categories...",
                    lines=3,
                    value="",
                )

                # ── Motion Gate + Refresh ─────────────────────────────────────
                c["motion_gate_enabled"] = gr.Checkbox(
                    value=True,
                    label="⚡ MOTION GATE (Scene-Change Gating)",
                    info="ON: only re-detect when scene changes or stale timer fires. "
                         "OFF: re-detect as fast as GPU can respond.",
                )
                c["motion_sensitivity"] = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=1.5,
                    label="MOTION SENSITIVITY (% PIXELS CHANGED)",
                    info="Lower = more sensitive — more VLM calls.",
                )
                c["stale_refresh"] = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=3.0,
                    label="STALE REFRESH FALLBACK (SECONDS)",
                    info="Re-detect anyway after this long even with no motion.",
                )

                with gr.Accordion("Pipeline Parameters", open=False):
                    c["det_temp_slider"] = gr.Slider(
                        label="Detector Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=0.9,
                    )

                # ── Preprocessing Accordion (Identical to Batch Tab) ──────────
                with gr.Accordion("Image Preprocessing & Augmentation", open=False):
                    c["prep_enabled_chk"] = gr.Checkbox(
                        label="Enable Preprocessing",
                        value=False,
                        info="Master toggle for all preprocessing steps below.",
                    )

                    with gr.Group(visible=False) as prep_options_group:
                        c["prep_short_edge_slider"] = gr.Slider(
                            label="Target Short Edge (px)",
                            minimum=512,
                            maximum=2048,
                            step=128,
                            value=1024,
                        )
                        c["prep_pad_square_chk"] = gr.Checkbox(
                            label="Pad to Square",
                            value=False,
                        )

                        c["prep_custom_resize_chk"] = gr.Checkbox(
                            label="Enable Custom Resize (override short edge)",
                            value=False,
                        )
                        with gr.Row(visible=False) as prep_custom_resize_row:
                            c["prep_custom_resize_width"] = gr.Number(
                                label="Target Width (px)", value=1024, precision=0
                            )
                            c["prep_custom_resize_height"] = gr.Number(
                                label="Target Height (px)", value=1024, precision=0
                            )

                        c["prep_contrast_dropdown"] = gr.Dropdown(
                            label="Contrast Correction Method",
                            choices=["none", "clahe", "autocontrast"],
                            value="clahe",
                        )
                        c["prep_gamma_slider"] = gr.Slider(
                            label="Gamma Correction",
                            minimum=0.5, maximum=2.0, step=0.05, value=1.0
                        )
                        c["prep_wb_chk"] = gr.Checkbox(
                            label="Gray World White Balance Correction", value=False
                        )

                        c["prep_denoise_dropdown"] = gr.Dropdown(
                            label="Denoising Filter",
                            choices=["none", "bilateral", "nlm"],
                            value="none",
                        )
                        c["prep_sharpen_chk"] = gr.Checkbox(
                            label="Apply Unsharp Mask (Sharpen)", value=False
                        )

                        c["prep_grid_dropdown"] = gr.Dropdown(
                            label="Grid Style",
                            choices=["Standard Red", "transparent", "fine", "none"],
                            value="Standard Red",
                        )
                        c["prep_grid_step_slider"] = gr.Slider(
                            label="Grid Step Size (px)",
                            minimum=20, maximum=500, step=10, value=250
                        )
                        c["prep_grid_line_width_slider"] = gr.Slider(
                            label="Grid Line Thickness (px)",
                            minimum=1, maximum=10, step=1, value=1
                        )
                        c["prep_grid_font_size_slider"] = gr.Slider(
                            label="Grid Label Font Size (0 = Auto)",
                            minimum=0, maximum=48, step=1, value=0
                        )
                        with gr.Row():
                            c["prep_grid_line_color_dropdown"] = gr.Dropdown(
                                label="Grid Line Color",
                                choices=["red", "blue", "green", "white", "black", "yellow", "cyan", "magenta", "custom"],
                                value="red",
                            )
                            c["prep_grid_line_color_custom"] = gr.Textbox(
                                label="Custom Line Color", value="red", visible=False
                            )
                        with gr.Row():
                            c["prep_grid_text_color_dropdown"] = gr.Dropdown(
                                label="Grid Text Color",
                                choices=["white", "black", "red", "blue", "green", "yellow", "cyan", "magenta", "custom"],
                                value="white",
                            )
                            c["prep_grid_text_color_custom"] = gr.Textbox(
                                label="Custom Text Color", value="white", visible=False
                            )
                        with gr.Row():
                            c["prep_grid_backing_color_dropdown"] = gr.Dropdown(
                                label="Grid Text Backing Color",
                                choices=["black", "none", "white", "red", "blue", "green", "custom"],
                                value="black",
                            )
                            c["prep_grid_backing_color_custom"] = gr.Textbox(
                                label="Custom Backing", value="black", visible=False
                            )

                        c["prep_som_chk"] = gr.Checkbox(
                            label="Enable Set-of-Mark (SoM) Prompting", value=False
                        )
                        c["prep_tiling_chk"] = gr.Checkbox(
                            label="Enable Image Tiling", value=False
                        )
                        c["prep_tile_size_slider"] = gr.Slider(
                            label="Tile Size (px)", minimum=256, maximum=1024, step=128, value=512
                        )
                        c["prep_tile_overlap_slider"] = gr.Slider(
                            label="Tile Overlap (%)", minimum=0, maximum=50, step=5, value=20
                        )
                        c["prep_cv_chk"] = gr.Checkbox(
                            label="Enable Crop & Verify Validation", value=False
                        )
                        c["prep_cv_padding_slider"] = gr.Slider(
                            label="Crop Context Padding (%)", minimum=0, maximum=50, step=5, value=15
                        )
                        c["prep_send_pixel_bounds_chk"] = gr.Checkbox(
                            label="Send Pixel Bounds in API Request", value=False
                        )
                        with gr.Row(visible=False) as prep_pixel_bounds_row:
                            c["prep_min_pixels_num"] = gr.Number(label="min_pixels", value=200704, precision=0)
                            c["prep_max_pixels_num"] = gr.Number(label="max_pixels", value=4194304, precision=0)

                        c["prep_options_group"] = prep_options_group
                        c["prep_custom_resize_row"] = prep_custom_resize_row
                        c["prep_pixel_bounds_row"] = prep_pixel_bounds_row

                # ── Video / HUD ───────────────────────────────────────────────
                c["sample_interval"] = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="VIDEO FRAME SAMPLING INTERVAL (SECONDS)",
                )
                c["process_video_btn"] = gr.Button(
                    "⚡ PROCESS VIDEO FRAMES",
                    variant="primary",
                    elem_classes=["neo-retro-badge"],
                )
                c["hud_status"] = gr.HTML(value=DEFAULT_HUD)

            with gr.Column(scale=2):
                with gr.Group(elem_id="rt_webcam_wrap") as webcam_wrap:
                    c["webcam_input"] = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="LIVE WEBCAM STREAM",
                        type="numpy",
                        elem_id="rt_webcam_input",
                    )
                c["webcam_wrap_group"] = webcam_wrap
                # Replaced old floating-canvas + JSON overlay with DetectionViewer
                # (client-side canvas, 10× less websocket payload, no rAF loop).
                c["realtime_viewer"] = DetectionViewer(
                    label="Live Detections",
                    panel_title="Live Detections",
                    list_height=380,
                )
                # Keep hidden JSON for backward-compat (no longer wired)
                c["boxes_json_state"] = gr.JSON(visible=False)
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["video_gallery_output"] = DetectionViewer(
                    label="Sampled Frame Detections",
                    panel_title="Sampled Detections",
                    list_height=380,
                )
                # Legacy Gallery key retained as alias for old callers
                c["video_gallery_legacy"] = gr.Gallery(visible=False)
    return c


def _wire_realtime_events(
    c_real: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]
):
    c_real["prep_enabled_chk"].change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[c_real["prep_enabled_chk"]],
        outputs=[c_real["prep_options_group"]],
    )
    c_real["prep_custom_resize_chk"].change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[c_real["prep_custom_resize_chk"]],
        outputs=[c_real["prep_custom_resize_row"]],
    )
    c_real["prep_send_pixel_bounds_chk"].change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[c_real["prep_send_pixel_bounds_chk"]],
        outputs=[c_real["prep_pixel_bounds_row"]],
    )

    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=is_cam),  # realtime_viewer visible only in cam mode
            gr.update(visible=not is_cam),
            gr.update(visible=not is_cam),
            fresh_session,
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"], c_real["session_state"]],
        outputs=[
            c_real["webcam_wrap_group"],
            c_real["realtime_viewer"],
            c_real["video_input"],
            c_real["video_gallery_output"],
            c_real["session_state"],
        ],
    )

    c_real["webcam_input"].stream(
        fn=process_single_frame,
        inputs=[
            c_real["webcam_input"],
            c_real["categories_input"],
            c_real["category_defs_input"],
            c_srv["server_port_input"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
            c_real["motion_gate_enabled"],
            c_real["motion_sensitivity"],
            c_real["stale_refresh"],
            c_real["tracker_algorithm"],
            c_real["session_state"],
            c_real["prep_enabled_chk"],
            c_real["prep_short_edge_slider"],
            c_real["prep_pad_square_chk"],
            c_real["prep_contrast_dropdown"],
            c_real["prep_gamma_slider"],
            c_real["prep_denoise_dropdown"],
            c_real["prep_sharpen_chk"],
            c_real["prep_wb_chk"],
            c_real["prep_grid_dropdown"],
            c_real["prep_som_chk"],
            c_real["prep_tiling_chk"],
            c_real["prep_tile_size_slider"],
            c_real["prep_tile_overlap_slider"],
            c_real["prep_cv_chk"],
            c_real["prep_cv_padding_slider"],
            c_real["prep_grid_step_slider"],
            c_real["prep_grid_line_width_slider"],
            c_real["prep_grid_font_size_slider"],
            c_real["prep_grid_line_color_dropdown"],
            c_real["prep_grid_line_color_custom"],
            c_real["prep_grid_text_color_dropdown"],
            c_real["prep_grid_text_color_custom"],
            c_real["prep_grid_backing_color_dropdown"],
            c_real["prep_grid_backing_color_custom"],
            c_real["prep_send_pixel_bounds_chk"],
            c_real["prep_min_pixels_num"],
            c_real["prep_max_pixels_num"],
            c_real["prep_custom_resize_chk"],
            c_real["prep_custom_resize_width"],
            c_real["prep_custom_resize_height"],
            c_real["det_temp_slider"],
        ],
        outputs=[
            c_real["realtime_viewer"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.12,
        show_progress="hidden",
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_real["category_defs_input"],
            c_srv["server_port_input"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
            c_real["prep_enabled_chk"],
            c_real["prep_short_edge_slider"],
            c_real["prep_pad_square_chk"],
            c_real["prep_contrast_dropdown"],
            c_real["prep_gamma_slider"],
            c_real["prep_denoise_dropdown"],
            c_real["prep_sharpen_chk"],
            c_real["prep_wb_chk"],
            c_real["prep_grid_dropdown"],
            c_real["prep_som_chk"],
            c_real["prep_tiling_chk"],
            c_real["prep_tile_size_slider"],
            c_real["prep_tile_overlap_slider"],
            c_real["prep_cv_chk"],
            c_real["prep_cv_padding_slider"],
            c_real["prep_grid_step_slider"],
            c_real["prep_grid_line_width_slider"],
            c_real["prep_grid_font_size_slider"],
            c_real["prep_grid_line_color_dropdown"],
            c_real["prep_grid_line_color_custom"],
            c_real["prep_grid_text_color_dropdown"],
            c_real["prep_grid_text_color_custom"],
            c_real["prep_grid_backing_color_dropdown"],
            c_real["prep_grid_backing_color_custom"],
            c_real["prep_send_pixel_bounds_chk"],
            c_real["prep_min_pixels_num"],
            c_real["prep_max_pixels_num"],
            c_real["prep_custom_resize_chk"],
            c_real["prep_custom_resize_width"],
            c_real["prep_custom_resize_height"],
            c_real["det_temp_slider"],
            c_real["tracker_algorithm"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
