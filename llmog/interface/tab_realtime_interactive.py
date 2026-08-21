"""
Real-Time Interactive Draw tab – live capture + Draw-style region classification.

Uses the same CustomCanvasController as Draw & Recognize but the background
is a captured webcam frame (via Capture button). Regions are classified via
classify_regions_gui with global endpoint and request_mode parallel/batched.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, Any

import gradio as gr
import numpy as np
from PIL import Image

from detection_viewer import DetectionViewer
from interface.batch.reclassification import (
    classify_regions_gui,
    _RECLS_EMPTY_TABLE,
    CATEGORY_PRESETS,
)


def _frame_to_payload(frame: np.ndarray | Image.Image | None) -> str:
    if frame is None:
        return json.dumps({"background": None, "regions": [], "layers": [], "composite": None})
    try:
        if isinstance(frame, np.ndarray):
            pil = Image.fromarray(frame).convert("RGB")
        elif isinstance(frame, Image.Image):
            pil = frame.convert("RGB")
        else:
            return json.dumps({"background": None, "regions": [], "layers": [], "composite": None})
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = {"background": f"data:image/jpeg;base64,{b64}", "regions": [], "layers": [], "composite": None}
        return json.dumps(payload)
    except Exception:
        return json.dumps({"background": None, "regions": [], "layers": [], "composite": None})


def _on_realtime_interactive_preset_change(preset_name: str):
    preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["Custom / Blank"])
    return gr.update(value=preset["classes"]), gr.update(value=preset["defs"])


def _on_realtime_interactive_mode_change(mode: str):
    if mode == "free":
        return (
            gr.update(label="Domain / Focus Hint (Optional)", placeholder="e.g. defects, wildlife...", info="Free: names any object"),
            gr.update(label="Domain Guidance (Optional)", placeholder="Optional context...", info="Optional"),
            gr.update(visible=False),
        )
    elif mode == "hybrid":
        return (
            gr.update(label="Priority Target Classes", placeholder="e.g. hole, stain", info="Hybrid: prioritize + discover"),
            gr.update(label="Category Definitions & Discovery", placeholder="Definitions...", info="Definitions"),
            gr.update(visible=True),
        )
    else:
        return (
            gr.update(label="Target Classes (Strict)", placeholder="hole, stain, tear", info="Strict: locked to list"),
            gr.update(label="Class Definitions", placeholder="Write instructions...", info="Criteria"),
            gr.update(visible=True),
        )


# Reuse Draw's custom canvas but with RT-specific ids to avoid singleton clash
from interface.tab_draw import _CUSTOM_CANVAS_HTML as _DRAW_HTML, _CUSTOM_CANVAS_JS as _DRAW_JS

# Generate RT-specific HTML/JS by replacing ids – keeps Draw and RT canvases independent
# Must replace ALL toolbar ids that are duplicated across tabs (else Draw tab controls RT canvas)
_RT_CUSTOM_CANVAS_HTML = (
    _DRAW_HTML.replace('id="llmog-custom-canvas-app"', 'id="llmog-custom-canvas-app-rt"')
    .replace('id="custom-annotation-canvas"', 'id="rt-custom-annotation-canvas"')
    .replace('id="canvas-stage-wrapper"', 'id="rt-canvas-stage-wrapper"')
    .replace('id="canvas-empty-overlay"', 'id="rt-canvas-empty-overlay"')
    .replace('id="canvas-file-input"', 'id="rt-canvas-file-input"')
    .replace('id="tool-bbox"', 'id="rt-tool-bbox"')
    .replace('id="tool-brush"', 'id="rt-tool-brush"')
    .replace('id="tool-circle"', 'id="rt-tool-circle"')
    .replace('id="tool-eraser"', 'id="rt-tool-eraser"')
    .replace('id="palette-swatches"', 'id="rt-palette-swatches"')
    .replace('id="custom-color-picker"', 'id="rt-custom-color-picker"')
    .replace('id="brush-size-slider"', 'id="rt-brush-size-slider"')
    .replace('id="brush-size-val"', 'id="rt-brush-size-val"')
    .replace('id="btn-undo"', 'id="rt-btn-undo"')
    .replace('id="btn-redo"', 'id="rt-btn-redo"')
    .replace('id="btn-clear-drawings"', 'id="rt-btn-clear-drawings"')
    .replace('id="btn-clear-all"', 'id="rt-btn-clear-all"')
    .replace('id="btn-zoom-in"', 'id="rt-btn-zoom-in"')
    .replace('id="btn-zoom-out"', 'id="rt-btn-zoom-out"')
    .replace('id="btn-zoom-fit"', 'id="rt-btn-zoom-fit"')
    .replace('id="zoom-level-text"', 'id="rt-zoom-level-text"')
    .replace('id="btn-empty-upload"', 'id="rt-btn-empty-upload"')
    .replace('id="btn-empty-sample"', 'id="rt-btn-empty-sample"')
    .replace('id="regions-count-badge"', 'id="rt-regions-count-badge"')
    .replace('id="regions-chips-container"', 'id="rt-regions-chips-container"')
)
_RT_CUSTOM_CANVAS_JS = (
    _DRAW_JS.replace("window.CustomCanvasController", "window.CustomCanvasControllerRT")
    .replace("CustomCanvasController", "CustomCanvasControllerRT")
    .replace("getCustomDrawData", "getRtInteractiveDrawData")
    .replace("__llmog_custom_canvas_payload__", "__rt_interactive_payload__")
    .replace("llmog-custom-canvas-app", "llmog-custom-canvas-app-rt")
    .replace("custom-annotation-canvas", "rt-custom-annotation-canvas")
    .replace("canvas-stage-wrapper", "rt-canvas-stage-wrapper")
    .replace("canvas-empty-overlay", "rt-canvas-empty-overlay")
    .replace("canvas-file-input", "rt-canvas-file-input")
    .replace("custom_draw_payload_box", "rt_interactive_payload_box")
    .replace("recls_sample_bridge_btn", "rt_capture_bridge_btn")
    .replace("tool-bbox", "rt-tool-bbox")
    .replace("tool-brush", "rt-tool-brush")
    .replace("tool-circle", "rt-tool-circle")
    .replace("tool-eraser", "rt-tool-eraser")
    .replace("palette-swatches", "rt-palette-swatches")
    .replace("custom-color-picker", "rt-custom-color-picker")
    .replace("brush-size-slider", "rt-brush-size-slider")
    .replace("brush-size-val", "rt-brush-size-val")
    .replace("btn-undo", "rt-btn-undo")
    .replace("btn-redo", "rt-btn-redo")
    .replace("btn-clear-drawings", "rt-btn-clear-drawings")
    .replace("btn-clear-all", "rt-btn-clear-all")
    .replace("btn-zoom-in", "rt-btn-zoom-in")
    .replace("btn-zoom-out", "rt-btn-zoom-out")
    .replace("btn-zoom-fit", "rt-btn-zoom-fit")
    .replace("zoom-level-text", "rt-zoom-level-text")
    .replace("btn-empty-upload", "rt-btn-empty-upload")
    .replace("btn-empty-sample", "rt-btn-empty-sample")
    .replace("regions-count-badge", "rt-regions-count-badge")
    .replace("regions-chips-container", "rt-regions-chips-container")
)


def build_realtime_interactive_tab() -> Dict[str, Any]:
    """Build Real-Time Interactive Draw tab (live capture + Draw canvas)."""
    with gr.Row(equal_height=False, elem_classes=["draw-tab-row"]):
        # Left: Live preview + capture + interactive canvas
        with gr.Column(scale=3, min_width=520):
            gr.HTML('<p class="section-label">🎥 Live Capture & Draw</p>')
            with gr.Row():
                live_preview = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live Preview (for capture)",
                    type="numpy",
                    elem_id="rt_interactive_live_preview",
                )
            gr.HTML('<p style="color:#7d8590; font-size:0.7rem; margin-bottom:0.5rem;">Tip: Click <b>📸 Capture</b> to freeze the current frame into the draw canvas below, then draw boxes. Uses its own camera – keep <b>🎥 Real-Time Detection</b> tab\'s webcam running or not, both work.</p>')
            with gr.Row(elem_classes=["btn-group"]):
                capture_btn = gr.Button("📸 Capture Frame from Real-Time Tab", variant="secondary", elem_id="rt_capture_btn")
                clear_canvas_btn = gr.Button("🧹 Clear Canvas", variant="secondary")
            # Custom canvas – RT-specific ids to avoid Draw tab singleton clash
            gr.HTML('<p class="section-label">🎨 Draw on Captured Frame</p>')
            custom_canvas = gr.HTML(
                value=_RT_CUSTOM_CANVAS_HTML,
                js_on_load=_RT_CUSTOM_CANVAS_JS,
                elem_id="rt-interactive-canvas-html",
            )
            payload_box = gr.Textbox(
                value="{}",
                visible=False,
                elem_id="rt_interactive_payload_box",
            )
            # Hidden bridge for sample/capture
            capture_bridge_btn = gr.Button(visible=False, elem_id="rt_capture_bridge_btn")

            with gr.Row(elem_classes=["btn-group"]):
                run_btn = gr.Button("🔎  Recognize Drawn Regions", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear Results", variant="secondary", scale=1)

        # Right: config + viewer (mirrors Draw tab)
        with gr.Column(scale=2, min_width=380, elem_classes=["draw-right-panel"]):
            gr.HTML('<p class="section-label">⚙️ Detection Strategy & Classes</p>')
            class_mode = gr.Radio(
                label="🎯 Class Expectation Mode",
                choices=[
                    ("🔒 Strict (Closed-Set)", "strict"),
                    ("🔀 Hybrid (Extendable)", "hybrid"),
                    ("🌐 Free (Open-World)", "free"),
                ],
                value="strict",
                info="Control how VLM assigns classes to drawn regions.",
            )
            preset_dropdown = gr.Dropdown(
                label="📋 Category Domain Presets",
                choices=list(CATEGORY_PRESETS.keys()),
                value="Fabric & Surface Defects",
                info="Quickly load target classes & definitions.",
            )
            classes_input = gr.Textbox(
                label="Target Classes (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                value="hole, stain, tear, cut, knot, weaving_defect",
                lines=2,
                info="Strict Mode: Agent is locked to these classes (or 'none').",
            )
            defs_input = gr.Textbox(
                label="Class Definitions / Distinguishing Rules",
                lines=4,
                value=CATEGORY_PRESETS["Fabric & Surface Defects"]["defs"],
                info="Detailed criteria for distinguishing each class.",
            )
            with gr.Accordion("⚙️ Advanced Filter & Context Settings", open=False):
                conf_threshold = gr.Slider(
                    label="Minimum Confidence Threshold (%)",
                    minimum=0, maximum=100, step=5, value=20,
                    info="Omit or flag recognitions below this threshold in YOLO outputs.",
                )
                padding_slider = gr.Slider(
                    label="Region Context Padding (%)",
                    minimum=0, maximum=50, step=1, value=10,
                    info="Extra visual context around each drawn region sent to VLM.",
                )
                request_mode = gr.Radio(
                    label="⚡ Request Mode (optional)",
                    choices=[
                        ("Sequential – 1 request per region", "sequential"),
                        ("Parallel – asyncio.gather concurrent", "parallel"),
                        ("Batched – single request with N images", "batched"),
                    ],
                    value="parallel",
                    info="Sequential: simple. Parallel: ~N× faster. Batched: 1 round-trip.",
                )
            connect_btn = gr.Button("🔌 Check Connection", variant="secondary", elem_id="rt-interactive-connect-btn")
            status = gr.Markdown("**Status: Idle – capture a frame, draw boxes, then Recognize**")
            with gr.Group(elem_classes=["img-viewer-wrap"]):
                viewer = DetectionViewer(
                    label="Annotated Recognition Result",
                    panel_title="Recognized Regions (Live Capture)",
                    list_height=340,
                    elem_id="rt-interactive-viewer",
                )
            results = gr.HTML(value=_RECLS_EMPTY_TABLE)
            with gr.Accordion("YOLO Labels (<class_id> <xc> <yc> <w> <h>)", open=False):
                yolo = gr.Textbox(lines=8, interactive=False, label="Copy these lines into the image's .txt label file")

    return dict(
        live_preview=live_preview,
        capture_btn=capture_btn,
        clear_canvas_btn=clear_canvas_btn,
        custom_canvas=custom_canvas,
        payload_box=payload_box,
        capture_bridge_btn=capture_bridge_btn,
        class_mode=class_mode,
        preset_dropdown=preset_dropdown,
        classes_input=classes_input,
        defs_input=defs_input,
        conf_threshold=conf_threshold,
        padding_slider=padding_slider,
        request_mode=request_mode,
        connect_btn=connect_btn,
        run_btn=run_btn,
        clear_btn=clear_btn,
        status=status,
        viewer=viewer,
        results=results,
        yolo=yolo,
    )


def _check_rt_interactive_endpoint(use_external_api: bool, ext_api_url: str, ext_api_key: str, ext_model_name: str, server_port):
    try:
        from interface.realtime.state import resolve_endpoint
        from openai import OpenAI
        base_url, api_key, model_name = resolve_endpoint(int(server_port) if server_port else 8080, bool(use_external_api), ext_api_url or "", ext_api_key or "", ext_model_name or "")
        if use_external_api:
            if not ext_api_key or ext_api_key.strip() in ("", "your-key"):
                return "**Status: ⚠️ External API selected but no API key set – configure in 🧠 Model / Endpoint tab.**"
            client = OpenAI(base_url=base_url, api_key=api_key)
            try:
                client.models.list()
                return f"**Status: ✅ Connected to External API `{model_name}` at `{base_url}`**"
            except Exception as e:
                return f"**Status: ⚠️ External API reachable but ping failed: {e}**"
        else:
            from interface.state import state
            with state.server_lock:
                mgr = state.server_manager
                if mgr is None:
                    return "**Status: ❌ Local server not running – start it in 🧠 Model / Endpoint tab.**"
                if not mgr.is_healthy():
                    return "**Status: ⏳ Local server starting – check logs in 🧠 Model / Endpoint tab.**"
                return f"**Status: ✅ Local server healthy on port {mgr.port} (model `{mgr.model}`)**"
    except Exception as e:
        return f"**Status: ❌ Connection check failed: {e}**"


def wire_realtime_interactive_events(c_rt_interactive: Dict[str, Any], c_srv: Dict[str, Any]) -> None:
    # Preset / mode
    c_rt_interactive["preset_dropdown"].change(
        fn=_on_realtime_interactive_preset_change,
        inputs=[c_rt_interactive["preset_dropdown"]],
        outputs=[c_rt_interactive["classes_input"], c_rt_interactive["defs_input"]],
    )
    c_rt_interactive["class_mode"].change(
        fn=_on_realtime_interactive_mode_change,
        inputs=[c_rt_interactive["class_mode"]],
        outputs=[c_rt_interactive["classes_input"], c_rt_interactive["defs_input"], c_rt_interactive["preset_dropdown"]],
    )
    # Capture: take live_preview numpy frame → payload_box → canvas (RT controller)
    c_rt_interactive["capture_btn"].click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            const video = document.querySelector('#rt_interactive_live_preview video');
            const wrap = document.getElementById('rt_interactive_live_preview');
            const target = video || (wrap ? wrap.querySelector('video') : null);
            if (!target || !target.videoWidth || target.readyState < 2) {
                alert('No live video found - start the webcam in Real-Time Detection tab (Webcam Stream) first.');
                return;
            }
            const canvas = document.createElement('canvas');
            canvas.width = target.videoWidth;
            canvas.height = target.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(target, 0, 0, canvas.width, canvas.height);
            const dataUri = canvas.toDataURL('image/jpeg', 0.92);
            if (window.CustomCanvasControllerRT) {
                window.CustomCanvasControllerRT.loadImageFromDataUrl(dataUri);
            } else {
                const ta = document.querySelector('#rt_interactive_payload_box textarea');
                if (ta) {
                    ta.value = JSON.stringify({background: dataUri, regions: [], layers: [], composite: null});
                    ta.dispatchEvent(new Event('input', {bubbles: true}));
                }
            }
        }""",
    )
    # Clear canvas (RT controller)
    c_rt_interactive["clear_canvas_btn"].click(
        fn=None, inputs=None, outputs=None,
        js="() => { if(window.CustomCanvasControllerRT) window.CustomCanvasControllerRT.clearAll(); }",
    )
    # Check connection
    c_rt_interactive["connect_btn"].click(
        fn=_check_rt_interactive_endpoint,
        inputs=[c_srv["use_external_api_chk"], c_srv["ext_api_url"], c_srv["ext_api_key"], c_srv["ext_model_name"], c_srv["server_port_input"]],
        outputs=[c_rt_interactive["status"]],
    )
    # Run recognition – uses same payload as Draw tab (custom canvas)
    # Note: payload_box is updated continuously by CustomCanvasController.syncGradioPayload()
    # We need a JS to ensure latest payload is flushed before Python call (like Draw tab)
    c_rt_interactive["run_btn"].click(
        fn=classify_regions_gui,
        inputs=[
            c_rt_interactive["payload_box"],
            c_rt_interactive["classes_input"],
            c_rt_interactive["defs_input"],
            c_rt_interactive["padding_slider"],
            c_rt_interactive["class_mode"],
            c_rt_interactive["conf_threshold"],
            c_srv["use_external_api_chk"],
            c_srv["ext_api_url"],
            c_srv["ext_api_key"],
            c_srv["ext_model_name"],
            c_srv["server_port_input"],
            c_rt_interactive["request_mode"],
        ],
        outputs=[
            c_rt_interactive["status"],
            c_rt_interactive["viewer"],
            c_rt_interactive["results"],
            c_rt_interactive["yolo"],
        ],
        js="(p,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode)=>{ const fresh=(window.getRtInteractiveDrawData?window.getRtInteractiveDrawData(): (window.CustomCanvasControllerRT?window.CustomCanvasControllerRT.syncGradioPayload() : p))||p; if(window.CustomCanvasControllerRT) { try{ const ta=document.querySelector('#rt_interactive_payload_box textarea'); if(ta) fresh=ta.value||fresh; }catch(e){} } return [fresh,c,d,pad,mode,conf,useExt,url,key,model,port,reqMode]; }",
        concurrency_limit=1,
    )
    c_rt_interactive["clear_btn"].click(
        fn=lambda: ("**Status: Idle – capture a frame, draw boxes, then Recognize**", None, _RECLS_EMPTY_TABLE, ""),
        inputs=None,
        outputs=[c_rt_interactive["status"], c_rt_interactive["viewer"], c_rt_interactive["results"], c_rt_interactive["yolo"]],
    )
