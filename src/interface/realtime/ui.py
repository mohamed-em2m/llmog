"""
Gradio UI layout and event wiring for the Realtime streaming tab.
"""

from typing import Dict, Any
import gradio as gr

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
                    label="TARGET CATEGORIES (comma-separated)",
                )
                c["max_resolution"] = gr.Slider(
                    minimum=384,
                    maximum=1084,
                    step=128,
                    value=640,
                    label="MAX FRAME RESOLUTION (PX)",
                    info="Lower resolution = faster real-time processing and lower latency.",
                )
                c["motion_sensitivity"] = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=1.5,
                    label="MOTION SENSITIVITY (% PIXELS CHANGED)",
                    info="Webcam mode: a new detection only runs once at least this % of the "
                    "frame changes (absdiff+threshold). Lower = more sensitive/more model calls.",
                )
                c["stale_refresh"] = gr.Slider(
                    minimum=2.0,
                    maximum=20.0,
                    step=1.0,
                    value=6.0,
                    label="STALE REFRESH FALLBACK (SECONDS)",
                    info="Webcam mode: re-detect anyway after this long, even with no motion.",
                )
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
                # NOTE: Gradio 6.x strips <script> from gr.HTML for security.
                # Only inject the CSS here; all canvas JS runs in the .change(js=) handler.
                gr.HTML(
                    """
                    <style>
                    #rt_float_canvas {
                        position: fixed;
                        pointer-events: none;
                        z-index: 9999;
                        box-sizing: border-box;
                    }
                    </style>
                    """
                )
                with gr.Group(elem_id="rt_webcam_wrap") as webcam_wrap:
                    c["webcam_input"] = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="LIVE WEBCAM STREAM",
                        type="numpy",
                        elem_id="rt_webcam_input",
                    )
                c["webcam_wrap_group"] = webcam_wrap
                c["boxes_json_state"] = gr.JSON(visible=False)
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["video_gallery_output"] = gr.Gallery(
                    label="SAMPLED FRAME DETECTIONS (EVERY 1 SEC)",
                    visible=False,
                    columns=3,
                )
    return c


def _wire_realtime_events(
    c_real: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]
):
    def toggle_mode(mode, session):
        is_cam = mode == "Webcam Stream"
        fresh_session = reset_session(session)
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=not is_cam),
            fresh_session,
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"], c_real["session_state"]],
        outputs=[
            c_real["webcam_wrap_group"],
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
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            gr.State(0.3),
            c_real["max_resolution"],
            c_real["motion_sensitivity"],
            c_real["stale_refresh"],
            c_real["tracker_algorithm"],
            c_real["session_state"],
        ],
        outputs=[
            c_real["boxes_json_state"],
            c_real["hud_status"],
            c_real["session_state"],
        ],
        stream_every=0.1,
    )

    c_real["boxes_json_state"].change(
        fn=None,
        inputs=[c_real["boxes_json_state"]],
        outputs=[],
        js="""
        (payload) => {
            if (!payload) return;

            // ── 1. Lazily create the floating canvas once (gr.HTML strips <script>) ──
            var canvas = document.getElementById('rt_float_canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.id = 'rt_float_canvas';
                canvas.style.position      = 'fixed';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex        = '9999';
                canvas.style.boxSizing     = 'border-box';
                document.body.appendChild(canvas);
            }

            // ── 2. Start the rAF position loop exactly once ───────────────────────
            if (!window._rtCanvasLoopRunning) {
                window._rtCanvasLoopRunning = true;
                (function loop() {
                    var cv = document.getElementById('rt_float_canvas');
                    if (!cv) { window._rtCanvasLoopRunning = false; return; }
                    var anchor = document.getElementById('rt_webcam_input');
                    if (anchor) {
                        var vid = anchor.querySelector('video');
                        var target = vid || anchor;
                        var r = target.getBoundingClientRect();
                        if (r.width > 0 && r.height > 0) {
                            cv.style.left   = r.left   + 'px';
                            cv.style.top    = r.top    + 'px';
                            cv.style.width  = r.width  + 'px';
                            cv.style.height = r.height + 'px';
                            var rw = Math.round(r.width),  rh = Math.round(r.height);
                            if (cv.width !== rw)  cv.width  = rw;
                            if (cv.height !== rh) cv.height = rh;
                        }
                    }
                    requestAnimationFrame(loop);
                })();
            }

            // ── 3. Draw boxes ─────────────────────────────────────────────────────
            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            var boxes  = payload.boxes  || [];
            var frameW = payload.frame_w || canvas.width;
            var frameH = payload.frame_h || canvas.height;
            if (!frameW || !frameH || boxes.length === 0) return;

            var scaleX = canvas.width  / frameW;
            var scaleY = canvas.height / frameH;

            ctx.lineWidth = 2;
            ctx.font = '12px "JetBrains Mono", monospace';

            for (var i = 0; i < boxes.length; i++) {
                var box = boxes[i];
                if (!box || box.length < 4) continue;
                var ymin = box[0], xmin = box[1], ymax = box[2], xmax = box[3];
                var label   = box[4] !== undefined ? String(box[4]) : '';
                var trackId = box[5] !== undefined ? box[5] : null;
                var x = xmin * scaleX,  y = ymin * scaleY;
                var w = (xmax - xmin) * scaleX,  h = (ymax - ymin) * scaleY;

                // Neon cyan box with glow
                ctx.strokeStyle = '#00ffcc';
                ctx.shadowColor = '#00ffcc';
                ctx.shadowBlur  = 6;
                ctx.strokeRect(x, y, w, h);
                ctx.shadowBlur  = 0;

                var tag = trackId !== null ? (label + ' #' + trackId) : label;
                if (tag.trim()) {
                    var tw = ctx.measureText(tag).width;
                    var bh = 18;
                    var by = (y > bh) ? (y - bh) : (y + h);
                    ctx.fillStyle = 'rgba(0,255,204,0.85)';
                    ctx.fillRect(x, by, tw + 8, bh);
                    ctx.fillStyle = '#050811';
                    ctx.fillText(tag, x + 4, by + bh - 4);
                }
            }
        }
        """,
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_real["max_resolution"],
            c_real["tracker_algorithm"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
