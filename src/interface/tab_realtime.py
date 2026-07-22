"""
Real-time webcam streaming & video 1-second frame detection tab UI and processing functions.
"""

import io
import time
import html
from typing import Dict, Any, Optional, List
import gradio as gr
from PIL import Image
import numpy as np
from openai import OpenAI

from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    draw_grid,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
)


def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    confidence_thresh: float,
) -> tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, '<div class="neo-retro-hud-stat">STATUS: READY</div>'

    start_time = time.time()
    try:
        pil_img = Image.fromarray(frame).convert("RGB")
        categories = [c.strip() for c in categories_str.split(",") if c.strip()]
        if not categories:
            categories = ["object"]

        base_url = (
            ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
        )
        api_key = ext_api_key if use_external_api else "no-key"
        model_name = ext_model_name if use_external_api else "local-model"

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

        boxes = []
        for d in valid_dets:
            bbox = d.get("bbox_2d", [])
            lbl = d.get("label", "")
            if len(bbox) == 4:
                # Convert 0-1000 scale [x1, y1, x2, y2] to image coordinates [ymin, xmin, ymax, xmax, label]
                x1, y1, x2, y2 = bbox
                ymin = y1 * pil_img.height / 1000.0
                xmin = x1 * pil_img.width / 1000.0
                ymax = y2 * pil_img.height / 1000.0
                xmax = x2 * pil_img.width / 1000.0
                boxes.append([ymin, xmin, ymax, xmax, lbl])

        annotated_img = draw_grid(pil_img, style="none")
        annotated_np = draw_boxes_on_image(annotated_img, boxes)

        elapsed = (time.time() - start_time) * 1000
        hud_info = (
            f'<div class="neo-retro-hud-stat">FPS: {1000/max(elapsed, 1):.1f} | '
            f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
        )
        return annotated_np, hud_info
    except Exception as e:
        return (
            frame,
            f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">ERROR: {html.escape(str(e))}</div>',
        )


def draw_boxes_on_image(image: Image.Image, boxes: List[Any]) -> np.ndarray:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="#00ffcc",
                facecolor="none",
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    xmin,
                    max(0, ymin - 5),
                    label,
                    color="#050811",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.2", facecolor="#00ffcc", edgecolor="none"
                    ),
                )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    res_img = Image.open(buf).convert("RGB")
    return np.array(res_img)


def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    progress=gr.Progress(),
) -> tuple[List[np.ndarray], str]:
    if not video_path:
        return [], "No video file uploaded."

    import cv2

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
        ],
        outputs=[c_real["annotated_stream_output"], c_real["hud_status"]],
        stream_every=0.5,
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
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )
