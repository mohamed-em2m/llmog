"""
Real-time webcam streaming & video frame detection tab UI and processing functions.
"""

import io
import time
import html
import threading
from typing import Dict, Any, Optional, List
import gradio as gr
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from openai import OpenAI

from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
)
from free_detection.image_preprocessing import (
    preprocess_resolution,
    map_bbox_to_original,
)

# ---------------------------------------------------------------------------
# Global Lock & Cache for Real-Time Non-Blocking Frame Dropping
# ---------------------------------------------------------------------------
_realtime_lock = threading.Lock()
_detect_thread: Optional[threading.Thread] = None
_last_boxes: List[Any] = []                         # Latest bounding boxes in original pixel space
_last_hud_info: str = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'
_is_detecting: bool = False


def draw_boxes_opencv(image_np: np.ndarray, boxes: List[Any]) -> np.ndarray:
    """Fast OpenCV bounding box rendering (<1ms vs ~200ms for Matplotlib)."""
    if cv2 is None:
        return image_np

    img = image_np.copy()
    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""

            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))

            # Draw vibrant cyber cyan bounding box (#00ffcc -> BGR (204, 255, 0))
            cv2.rectangle(img, pt1, pt2, (204, 255, 0), 2)

            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                lbl_pt1 = (int(xmin), max(0, int(ymin) - text_h - 6))
                lbl_pt2 = (int(xmin) + text_w + 6, max(text_h + 6, int(ymin)))
                cv2.rectangle(img, lbl_pt1, lbl_pt2, (204, 255, 0), -1)

                cv2.putText(
                    img,
                    label,
                    (int(xmin) + 3, max(text_h + 2, int(ymin) - 3)),
                    font,
                    font_scale,
                    (17, 8, 5),
                    thickness,
                    cv2.LINE_AA,
                )
    return img


def _run_detection_bg(
    frame: np.ndarray,
    categories: list,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_info: dict,
):
    """Background thread: runs VLM detection and updates global boxes cache."""
    global _last_boxes, _last_hud_info, _is_detecting

    start_time = time.time()
    try:
        pil_img = Image.fromarray(frame).convert("RGB")
        client = OpenAI(base_url=base_url, api_key=api_key)
        pipeline = ObjectDetectionPipeline(client=client, detector_model=model_name)

        img_uri = pil_to_data_uri(pil_img)
        raw_output = pipeline.run_inference(
            image_uris=img_uri,
            categories=categories,
            category_definitions="",
        )
        parsed_dets = parse_detections(raw_output)
        valid_dets = validate_detections(parsed_dets, categories)

        orig_w = prep_info["orig_w"]
        orig_h = prep_info["orig_h"]

        boxes = []
        for d in valid_dets:
            bbox = d.get("bbox_2d", [])
            lbl = d.get("label", "")
            if len(bbox) == 4:
                x1, y1, x2, y2 = map_bbox_to_original(list(bbox), prep_info)
                ymin = y1 * orig_h / 1000.0
                xmin = x1 * orig_w / 1000.0
                ymax = y2 * orig_h / 1000.0
                xmax = x2 * orig_w / 1000.0
                boxes.append([ymin, xmin, ymax, xmax, lbl])

        elapsed = (time.time() - start_time) * 1000.0
        fps = 1000.0 / max(elapsed, 1.0)

        with _realtime_lock:
            _last_boxes = boxes
            _last_hud_info = (
                f'<div class="neo-retro-hud-stat">FPS: {fps:.1f} | '
                f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
            )
    except Exception as e:
        with _realtime_lock:
            _last_hud_info = (
                f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">'
                f"ERROR: {html.escape(str(e))}</div>"
            )
    finally:
        with _realtime_lock:
            _is_detecting = False


def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    confidence_thresh: float,
    max_resolution: int = 640,
) -> tuple[Optional[np.ndarray], str]:
    """
    Called on every Gradio webcam stream tick.

    Strategy:
    - Every tick receives the LIVE webcam frame from the browser.
    - If no background detection is running, launch detection asynchronously on a downscaled frame.
    - Render the latest known detected bounding boxes dynamically onto the CURRENT LIVE WEBCAM FRAME.
    - This ensures the video stream remains 100% smooth & live at 30 FPS while bounding boxes update asynchronously!
    """
    global _detect_thread, _is_detecting

    if frame is None:
        with _realtime_lock:
            hud = _last_hud_info
        return None, hud

    pil_img = Image.fromarray(frame).convert("RGB")
    max_res = int(max_resolution or 640)
    proc_img, prep_info = preprocess_resolution(
        pil_img,
        enabled=True,
        target_short_edge=max_res,
    )

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or ["object"]
    base_url = ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
    api_key = ext_api_key if use_external_api else "no-key"
    model_name = ext_model_name if use_external_api else "local-model"

    # Launch detection in background if model is free
    with _realtime_lock:
        should_start = not _is_detecting
        if should_start:
            _is_detecting = True

    if should_start:
        _detect_thread = threading.Thread(
            target=_run_detection_bg,
            args=(
                np.array(proc_img),   # downscaled for VLM speed
                categories,
                base_url,
                api_key,
                model_name,
                prep_info,            # resolution & coordinate mapping info
            ),
            daemon=True,
        )
        _detect_thread.start()

    # Draw the LATEST detected bounding boxes onto the CURRENT LIVE WEBCAM FRAME
    with _realtime_lock:
        boxes_to_draw = list(_last_boxes)
        hud = _last_hud_info

    annotated_live = draw_boxes_opencv(np.array(pil_img), boxes_to_draw)

    return annotated_live, hud


def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    max_resolution: int = 640,
    progress=gr.Progress(),
) -> tuple[List[np.ndarray], str]:
    if not video_path:
        return [], "No video file uploaded."

    if cv2 is None:
        return [], "OpenCV (cv2) is required for video processing."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Failed to open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, fps * sample_interval))
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        frame_count += 1
    cap.release()

    annotated_frames = []

    for idx, f in enumerate(frames):
        progress((idx + 1) / len(frames), desc=f"Detecting frame {idx+1}/{len(frames)}")
        ann, _ = process_single_frame(
            f,
            categories_str,
            server_port,
            use_external_api,
            ext_api_url,
            ext_api_key,
            ext_model_name,
            0.3,
            max_resolution,
        )
        if ann is not None:
            annotated_frames.append(ann)

    return annotated_frames, f"Successfully processed {len(annotated_frames)} frames from video!"


def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML("""
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR
            </h2>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
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
                c["hud_status"] = gr.HTML(
                    value='<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'
                )

            with gr.Column(scale=2):
                c["webcam_input"] = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="LIVE WEBCAM STREAM",
                    type="numpy",
                )
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["annotated_stream_output"] = gr.Image(
                    label="NEO-RETRO DETECTED STREAM / OVERLAY",
                    type="numpy",
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
    def toggle_mode(mode):
        is_cam = mode == "Webcam Stream"
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"]],
        outputs=[
            c_real["webcam_input"],
            c_real["video_input"],
            c_real["annotated_stream_output"],
            c_real["video_gallery_output"],
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
        ],
        outputs=[c_real["annotated_stream_output"], c_real["hud_status"]],
        stream_every=0.3,
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
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
