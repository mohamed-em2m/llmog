"""
Stream handlers for live webcam frames and video file processing.
"""

import time
from typing import List, Tuple, Any, Optional
import gradio as gr
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from free_detection.image_preprocessing import preprocess_resolution
from free_detection.trackers import MultiAlgorithmTracker
from interface.realtime.state import SessionDetector, new_session_detector, resolve_endpoint
from interface.realtime.utils import (
    to_small_gray,
    scene_has_changed,
    run_vlm_detect,
    draw_boxes_opencv,
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
    max_resolution: int,
    motion_sensitivity_pct: float,
    stale_refresh_seconds: float,
    tracker_algorithm: str,
    session: SessionDetector,
) -> Tuple[dict, str, SessionDetector]:
    """Continuous Live Streaming Processor called on stream ticks."""
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        return {"boxes": boxes, "frame_w": 0, "frame_h": 0}, hud, session

    pil_img = Image.fromarray(frame).convert("RGB")
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 1) Execute chosen tracker update for continuous frame tracking
    tracked_boxes = session.update_tracking_only(frame, tracker_algorithm)
    hud = session.last_hud

    # 2) Dispatch background detection when ready
    if not session.is_busy():
        gray_small = to_small_gray(frame)
        change_ratio_thresh = max(0.0, float(motion_sensitivity_pct or 1.5)) / 100.0
        now = time.time()
        stale = (now - session.last_detect_time) >= max(
            0.5, float(stale_refresh_seconds or 6.0)
        )
        changed = (
            scene_has_changed(
                gray_small,
                session.reference_gray,
                change_ratio_thresh=change_ratio_thresh,
            )
            if gray_small is not None
            else True
        )

        if changed or stale:
            max_res = int(max_resolution or 640)
            categories = [
                c.strip() for c in categories_str.split(",") if c.strip()
            ] or ["object"]
            base_url, api_key, model_name = resolve_endpoint(
                server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
            )
            prep_info = {"max_res": max_res, "orig_w": frame_w, "orig_h": frame_h}
            frame_id = session.next_frame_id()
            session.submit(
                frame_id,
                run_vlm_detect,
                frame,
                categories,
                base_url,
                api_key,
                model_name,
                prep_info,
                True,  # enable_grid
                100,   # grid_step
            )
            session.reference_gray = gray_small
            session.last_detect_time = now

    return (
        {"boxes": tracked_boxes, "frame_w": frame_w, "frame_h": frame_h},
        hud,
        session,
    )


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
    tracker_algorithm: str = "ByteTrack",
    progress=gr.Progress(),
) -> Tuple[List[np.ndarray], str]:
    """Synchronous video file sampling and detection processing."""
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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()

    if not frames:
        return [], "No frames could be sampled from this video."

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or [
        "object"
    ]
    base_url, api_key, model_name = resolve_endpoint(
        server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
    )
    max_res = int(max_resolution or 640)

    tracker = MultiAlgorithmTracker(tracker_algorithm)
    annotated_frames = []
    errors = 0
    for idx, f in enumerate(frames):
        progress(
            (idx + 1) / len(frames), desc=f"Detecting frame {idx + 1}/{len(frames)}"
        )
        prep_info = {"max_res": max_res, "orig_w": f.shape[1], "orig_h": f.shape[0]}
        try:
            boxes, _hud = run_vlm_detect(
                f, categories, base_url, api_key, model_name, prep_info, True, 100
            )
            tracked_boxes = boxes
        except Exception:
            tracked_boxes = []
            errors += 1
        annotated_frames.append(draw_boxes_opencv(f, tracked_boxes))

    status = f"Successfully processed {len(annotated_frames)} frames from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    return annotated_frames, status
