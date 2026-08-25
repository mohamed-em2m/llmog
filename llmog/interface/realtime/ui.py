"""
Gradio UI layout and event wiring for the Realtime streaming tab.
"""

from typing import Dict, Any
import gradio as gr

from detection_viewer import DetectionViewer
from free_detection.trackers import MultiAlgorithmTracker
from interface.batch.reclassification import CATEGORY_PRESETS
from interface.realtime.state import (
    SessionDetector,
    new_session_detector,
    reset_session,
    DEFAULT_HUD,
)
from interface.realtime.handlers import process_single_frame, process_video_frames


def _on_realtime_preset_change(preset_name: str):
    preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["Custom / Blank"])
    return gr.update(value=preset["classes"]), gr.update(value=preset["defs"])


def _on_realtime_strategy_change(strategy: str):
    if strategy == "free":
        return (
            gr.update(
                label="Domain / Focus Hint (Optional)",
                placeholder="e.g. Focus on industrial defects, wildlife, vehicles... (or leave blank)",
                info="Free Mode: Agent autonomously detects all salient objects/anomalies.",
            ),
            gr.update(
                label="Domain Guidance / Prompt Context (Optional)",
                placeholder="Optional domain context or special instructions...",
                info="Optional domain guidance.",
            ),
            gr.update(visible=False),
        )
    elif strategy == "hybrid":
        return (
            gr.update(
                label="Priority Target Categories (comma-separated)",
                placeholder="e.g. hole, stain, tear, cut",
                info="Hybrid Mode: Target categories are prioritized; agent can discover new anomaly classes.",
            ),
            gr.update(
                label="Category Definitions & Novel Discovery Guidelines",
                placeholder="Definitions for priority categories...",
                info="Definitions for priority categories.",
            ),
            gr.update(visible=True),
        )
    else:  # strict
        return (
            gr.update(
                label="Target Categories (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                info="Strict Mode: Agent is restricted strictly to listed categories.",
            ),
            gr.update(
                label="Category Definitions",
                placeholder="Write instructions for categories...",
                info="Category definitions.",
            ),
            gr.update(visible=True),
        )


def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    c["session_state"] = gr.State(new_session_detector)

    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML('<p class="section-label">🎥 Real-Time Detection — Webcam & Video (Multi-Tracker Integrated)</p>')

        # ── TWIN SCREENS: Input (left) | Output (right) — same line, same size (520px) ──
        with gr.Row(equal_height=True, elem_classes=["draw-tab-row", "twin-screens-row"]):
            with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
                gr.HTML('<p class="section-label">📥 Input — Live Webcam / Video</p>')
                with gr.Group(elem_id="rt_webcam_wrap", elem_classes=["rt-webcam-wrap"]) as webcam_wrap:
                    c["webcam_input"] = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="LIVE WEBCAM STREAM (free detection – boxes in same window when ⚡ enabled)",
                        type="numpy",
                        elem_id="rt_webcam_input",
                    )
                    c["same_window_html"] = gr.HTML(
                        value="""
                        <canvas id="rt_same_window_canvas" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:5;"></canvas>
                        <style>
                        #rt_webcam_wrap{position:relative;}
                        #rt_same_window_canvas{position:absolute; top:0; left:0; width:100%; height:100%;}
                        </style>
                        """,
                        visible=False,
                        elem_id="rt_same_window_html",
                    )
                c["webcam_wrap_group"] = webcam_wrap
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
            with gr.Column(scale=1, min_width=420, elem_classes=["batch-bottom-col"]):
                gr.HTML('<p class="section-label">👁️ Output — Live Detections</p>')
                c["realtime_viewer"] = DetectionViewer(
                    label="Live Detections (interactive viewer – enable ⚡ for same-window max FPS)",
                    panel_title="Live Detections",
                    list_height=300,
                    visible=True,
                    elem_id="rt-live-viewer",
                )
                c["hud_status"] = gr.HTML(value=DEFAULT_HUD)
                c["boxes_json_state"] = gr.JSON(visible=False)
                c["video_gallery_output"] = gr.Gallery(
                    label="Sampled Frame Detections (Gallery)",
                    columns=3,
                    visible=False,
                    elem_id="rt_video_gallery",
                )
                c["video_viewer"] = DetectionViewer(
                    label="Sampled Frame – Interactive Viewer (last frame)",
                    panel_title="Sampled Detections",
                    list_height=380,
                    elem_id="rt_video_viewer",
                )
                c["video_gallery_legacy"] = c["video_gallery_output"]

        # ── Visible primary input controls (always) ──
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
                )
            with gr.Column(scale=1):
                c["sample_interval"] = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="VIDEO FRAME SAMPLING INTERVAL (SECONDS)",
                )
                c["process_video_btn"] = gr.Button(
                    "▶  Process Video Frames",
                    variant="primary",
                )

        # ── DROPDOWN 1: Categories — focused ──
        with gr.Accordion("📥 Categories & Detection Mode — Strategy, Presets & Definitions", open=False):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    c["category_strategy"] = gr.Radio(
                        label="🎯 Class Expectation Mode",
                        choices=[
                            ("🔒 Strict (Closed-Set)", "strict"),
                            ("🔀 Hybrid (Extendable)", "hybrid"),
                            ("🌐 Free (Open-World)", "free"),
                        ],
                        value="free",
                        info="Free: detect all salient objects freely. Strict: only detect listed categories.",
                    )
                    c["category_preset_dropdown"] = gr.Dropdown(
                        label="📋 Category Domain Presets",
                        choices=list(CATEGORY_PRESETS.keys()),
                        value="General Objects (COCO)",
                        visible=False,
                        info="Quickly load target categories & definitions.",
                    )
                with gr.Column(scale=1):
                    c["categories_input"] = gr.Textbox(
                        value="",
                        label="Domain / Focus Hint (Optional)",
                        placeholder="e.g. Focus on industrial defects, wildlife, vehicles... (or leave blank)",
                        info="Free Mode: Agent autonomously detects all salient objects/anomalies.",
                    )
                    c["category_defs_input"] = gr.Textbox(
                        label="Domain Guidance / Prompt Context (Optional)",
                        placeholder="Optional domain context or special instructions...",
                        lines=3,
                        value="",
                        info="Optional domain guidance.",
                    )

        # ── DROPDOWN 2: Advanced Detection — focused ──
        with gr.Accordion("⚙️ Advanced Detection Settings — Tracker, Motion & Temperature", open=False):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
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
                    c["same_window_chk"] = gr.Checkbox(
                        value=False,
                        label="⚡ Same-window overlay (fastest – no WebP re-encode)",
                        info="ON: boxes drawn directly on video (max FPS). OFF: use interactive DetectionViewer (more features, adds WebP). Default OFF for reliability.",
                    )
                with gr.Column(scale=1):
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
                    c["det_temp_slider"] = gr.Slider(
                        label="Detector Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=0.9,
                    )

        # ── DROPDOWN 3: Preprocessing — focused ──
        with gr.Accordion("🎨 Image Preprocessing & Augmentation", open=False):
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

    # ── Category strategy & presets (Class Expectation Mode) ──────────────
    c_real["category_preset_dropdown"].change(
        fn=_on_realtime_preset_change,
        inputs=[c_real["category_preset_dropdown"]],
        outputs=[c_real["categories_input"], c_real["category_defs_input"]],
    )
    c_real["category_strategy"].change(
        fn=_on_realtime_strategy_change,
        inputs=[c_real["category_strategy"]],
        outputs=[
            c_real["categories_input"],
            c_real["category_defs_input"],
            c_real["category_preset_dropdown"],
        ],
    )
    # Custom grid colors (parity with Batch tab)
    from interface.state import toggle_custom_color_field

    for dd, custom_field in [
        (c_real["prep_grid_line_color_dropdown"], c_real["prep_grid_line_color_custom"]),
        (c_real["prep_grid_text_color_dropdown"], c_real["prep_grid_text_color_custom"]),
        (c_real["prep_grid_backing_color_dropdown"], c_real["prep_grid_backing_color_custom"]),
    ]:
        dd.change(toggle_custom_color_field, inputs=[dd], outputs=[custom_field])

    # Same-window overlay toggle – free detection max speed
    c_real["same_window_chk"].change(
        fn=lambda enabled: (gr.update(visible=enabled), gr.update(visible=not enabled)),
        inputs=[c_real["same_window_chk"]],
        outputs=[c_real["same_window_html"], c_real["realtime_viewer"]],
    )

    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=is_cam),  # realtime_viewer
            gr.update(visible=not is_cam),
            gr.update(visible=not is_cam),  # gallery
            gr.update(visible=not is_cam),  # viewer (last frame)
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
            c_real["video_viewer"],
            c_real["session_state"],
        ],
    )

    c_real["webcam_input"].stream(
        fn=process_single_frame,
        inputs=[
            c_real["webcam_input"],
            c_real["categories_input"],
            c_real["category_defs_input"],
            c_real["category_strategy"],
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
            c_real["boxes_json_state"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.12,
        show_progress="hidden",
    )

    # Same-window overlay – draws boxes_json_state directly on video (no WebP, max FPS)
    # Canvas is absolutely positioned over the video element; scale from video's displayed size
    c_real["boxes_json_state"].change(
        fn=None,
        inputs=[c_real["boxes_json_state"]],
        outputs=[],
        js="""(payload) => {
            const canvas = document.getElementById('rt_same_window_canvas');
            const wrap = document.getElementById('rt_webcam_wrap');
            if (!canvas || !wrap) return;
            const ctx = canvas.getContext('2d');
            const video = wrap.querySelector('video');
            if (!video || !video.videoWidth) {
                ctx.clearRect(0,0,canvas.width,canvas.height);
                return;
            }
            const rect = video.getBoundingClientRect();
            const wrapRect = wrap.getBoundingClientRect();
            // Position canvas exactly over the video (handles letterboxing)
            canvas.width = Math.round(rect.width);
            canvas.height = Math.round(rect.height);
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            canvas.style.left = (rect.left - wrapRect.left) + 'px';
            canvas.style.top = (rect.top - wrapRect.top) + 'px';
            ctx.clearRect(0,0,canvas.width,canvas.height);
            if (!payload || !payload.boxes || payload.boxes.length===0) return;
            const boxes = payload.boxes;
            const frameW = payload.frame_w || video.videoWidth;
            const frameH = payload.frame_h || video.videoHeight;
            if (!frameW || !frameH) return;
            const scaleX = rect.width / frameW;
            const scaleY = rect.height / frameH;
            ctx.lineWidth = 2;
            ctx.font = '12px \"JetBrains Mono\", monospace';
            for (let i=0;i<boxes.length;i++){
                const b=boxes[i]; if(!b||b.length<4) continue;
                const ymin=b[0], xmin=b[1], ymax=b[2], xmax=b[3];
                const label = b[4]!==undefined?String(b[4]):'';
                const tid = b[5]!==undefined?b[5]:null;
                const x=xmin*scaleX, y=ymin*scaleY, w=(xmax-xmin)*scaleX, h=(ymax-ymin)*scaleY;
                ctx.strokeStyle='#00ffcc'; ctx.shadowColor='#00ffcc'; ctx.shadowBlur=6;
                ctx.strokeRect(x,y,w,h); ctx.shadowBlur=0;
                const tag = tid!==null ? (label+' #'+tid) : label;
                if(tag.trim()){
                    const tw=ctx.measureText(tag).width; const bh=18;
                    const by = (y>bh)?(y-bh):(y+h);
                    ctx.fillStyle='rgba(0,255,204,0.85)'; ctx.fillRect(x,by,tw+8,bh);
                    ctx.fillStyle='#050811'; ctx.fillText(tag, x+4, by+bh-4);
                }
            }
        }""",
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_real["category_defs_input"],
            c_real["category_strategy"],
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
        outputs=[c_real["video_gallery_output"], c_real["video_viewer"], c_real["hud_status"]],
    )
