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

from free_detection.trackers import MultiAlgorithmTracker
from interface.realtime.state import (
    SessionDetector,
    new_session_detector,
    resolve_endpoint,
)
from interface.viewer_utils import realtime_boxes_to_annotations, build_viewer_payload
from interface.realtime.utils import (
    run_vlm_detect,
)


def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    category_definitions: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    motion_gate_enabled: bool,
    motion_sensitivity_pct: float,
    stale_refresh_seconds: float,
    tracker_algorithm: str,
    session: SessionDetector,
    prep_enabled: bool,
    prep_short_edge: float,
    prep_pad_square: bool,
    prep_contrast_method: str,
    prep_gamma: float,
    prep_denoise_method: str,
    prep_sharpen: bool,
    prep_white_balance: bool,
    prep_grid_style: str,
    prep_som_enabled: bool,
    prep_tiling_enabled: bool,
    prep_tile_size: float,
    prep_tile_overlap: float,
    prep_crop_verify_enabled: bool,
    prep_crop_padding: float,
    prep_grid_step: float,
    prep_grid_line_width: float,
    prep_grid_font_size: float,
    prep_grid_line_color: str,
    prep_grid_line_color_custom: str,
    prep_grid_text_color: str,
    prep_grid_text_color_custom: str,
    prep_grid_backing_color: str,
    prep_grid_backing_color_custom: str,
    prep_send_pixel_bounds: bool,
    prep_min_pixels: float,
    prep_max_pixels: float,
    prep_custom_resize_enabled: bool,
    prep_custom_resize_width: float,
    prep_custom_resize_height: float,
    detector_temp: float = 0.9,
) -> Tuple[Any, str, SessionDetector]:
    """Continuous Live Streaming Processor – now returns a DetectionViewer payload.

    Returns ``(viewer_payload, hud, session)`` where viewer_payload is
    ``(frame_numpy, [annotations])`` for ``DetectionViewer``.  Boxes are
    converted once via ``realtime_boxes_to_annotations`` so the browser draws
    them on the JS canvas – zero server-side box rasterisation.
    """
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        # No frame yet – return placeholder (viewer shows "No data")
        if not boxes:
            return None, hud, session
        # Need a dummy frame size; return None payload to avoid wrong scale
        return None, hud, session

    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 1) Execute chosen tracker update for continuous frame tracking
    tracked_boxes = session.update_tracking_only(frame, tracker_algorithm)
    hud = session.last_hud

    # 2) Dispatch background detection when ready
    if not session.is_busy():
        now = time.time()
        stale_secs = max(0.5, float(stale_refresh_seconds or 3.0))
        stale = (now - session.last_detect_time) >= stale_secs

        # Check if a fresh VLM result just arrived — re-detect immediately
        fresh_result = session.consume_force_redetect()

        if fresh_result or not motion_gate_enabled or stale:
            session._last_submitted_frame = frame.copy()
            categories = [
                c.strip() for c in categories_str.split(",") if c.strip()
            ] or ["object"]
            base_url, api_key, model_name = resolve_endpoint(
                server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
            )

            from interface.viewer_utils import build_prep_config as _build_prep

            prep_config = _build_prep(
                prep_enabled=prep_enabled,
                prep_short_edge=prep_short_edge,
                prep_pad_square=prep_pad_square,
                prep_contrast_method=prep_contrast_method,
                prep_gamma=prep_gamma,
                prep_denoise_method=prep_denoise_method,
                prep_sharpen=prep_sharpen,
                prep_white_balance=prep_white_balance,
                prep_grid_style=prep_grid_style,
                prep_som_enabled=prep_som_enabled,
                prep_tiling_enabled=prep_tiling_enabled,
                prep_tile_size=prep_tile_size,
                prep_tile_overlap=prep_tile_overlap,
                prep_crop_verify_enabled=prep_crop_verify_enabled,
                prep_crop_padding=prep_crop_padding,
                prep_grid_step=prep_grid_step,
                prep_grid_line_width=prep_grid_line_width,
                prep_grid_font_size=prep_grid_font_size,
                prep_grid_line_color=prep_grid_line_color,
                prep_grid_line_color_custom=prep_grid_line_color_custom,
                prep_grid_text_color=prep_grid_text_color,
                prep_grid_text_color_custom=prep_grid_text_color_custom,
                prep_grid_backing_color=prep_grid_backing_color,
                prep_grid_backing_color_custom=prep_grid_backing_color_custom,
                prep_send_pixel_bounds=prep_send_pixel_bounds,
                prep_min_pixels=prep_min_pixels,
                prep_max_pixels=prep_max_pixels,
                prep_custom_resize_enabled=prep_custom_resize_enabled,
                prep_custom_resize_width=prep_custom_resize_width,
                prep_custom_resize_height=prep_custom_resize_height,
            )

            pipeline_params = {"detector_temperature": float(detector_temp or 0.9)}

            frame_id = session.next_frame_id()
            session.submit(
                frame_id,
                run_vlm_detect,
                frame,
                categories,
                category_definitions,
                base_url,
                api_key,
                model_name,
                prep_config,
                pipeline_params,
            )
            session.last_detect_time = now

    # Build DetectionViewer payload: (RGB frame, viewer annotations)
    try:
        anns = realtime_boxes_to_annotations(tracked_boxes)
    except Exception:
        anns = []
    viewer_payload = build_viewer_payload(frame, anns)
    # Update hud with tracked count for immediate feedback
    if not hud or "DETECTED" not in hud:
        hud = f'<div class="neo-retro-hud-stat">FPS: -- | DETECTED: {len(anns)}</div>' if anns else hud
    return (
        viewer_payload,
        hud,
        session,
    )


def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    category_definitions: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    prep_enabled: bool,
    prep_short_edge: float,
    prep_pad_square: bool,
    prep_contrast_method: str,
    prep_gamma: float,
    prep_denoise_method: str,
    prep_sharpen: bool,
    prep_white_balance: bool,
    prep_grid_style: str,
    prep_som_enabled: bool,
    prep_tiling_enabled: bool,
    prep_tile_size: float,
    prep_tile_overlap: float,
    prep_crop_verify_enabled: bool,
    prep_crop_padding: float,
    prep_grid_step: float,
    prep_grid_line_width: float,
    prep_grid_font_size: float,
    prep_grid_line_color: str,
    prep_grid_line_color_custom: str,
    prep_grid_text_color: str,
    prep_grid_text_color_custom: str,
    prep_grid_backing_color: str,
    prep_grid_backing_color_custom: str,
    prep_send_pixel_bounds: bool,
    prep_min_pixels: float,
    prep_max_pixels: float,
    prep_custom_resize_enabled: bool,
    prep_custom_resize_width: float,
    prep_custom_resize_height: float,
    detector_temp: float = 0.9,
    tracker_algorithm: str = "ByteTrack",
    progress=gr.Progress(),
) -> Tuple[Any, str]:
    """Synchronous video file sampling – returns a DetectionViewer payload for the last sampled frame.

    Perf: server no longer draws boxes into NumPy via OpenCV; the browser
    renders them via DetectionViewer's JS canvas (avoids per-frame JPEG
    re-encode + copy).  Gallery replaced by a single optimised viewer.
    """
    if not video_path:
        return None, "No video file uploaded."
    if cv2 is None:
        return None, "OpenCV (cv2) is required for video processing."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Failed to open video file."

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
        return None, "No frames could be sampled from this video."

    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or [
        "object"
    ]
    base_url, api_key, model_name = resolve_endpoint(
        server_port, use_external_api, ext_api_url, ext_api_key, ext_model_name
    )

    from interface.viewer_utils import build_prep_config as _build_pre2

    prep_config = _build_pre2(
        prep_enabled=prep_enabled,
        prep_short_edge=prep_short_edge,
        prep_pad_square=prep_pad_square,
        prep_contrast_method=prep_contrast_method,
        prep_gamma=prep_gamma,
        prep_denoise_method=prep_denoise_method,
        prep_sharpen=prep_sharpen,
        prep_white_balance=prep_white_balance,
        prep_grid_style=prep_grid_style,
        prep_som_enabled=prep_som_enabled,
        prep_tiling_enabled=prep_tiling_enabled,
        prep_tile_size=prep_tile_size,
        prep_tile_overlap=prep_tile_overlap,
        prep_crop_verify_enabled=prep_crop_verify_enabled,
        prep_crop_padding=prep_crop_padding,
        prep_grid_step=prep_grid_step,
        prep_grid_line_width=prep_grid_line_width,
        prep_grid_font_size=prep_grid_font_size,
        prep_grid_line_color=prep_grid_line_color,
        prep_grid_line_color_custom=prep_grid_line_color_custom,
        prep_grid_text_color=prep_grid_text_color,
        prep_grid_text_color_custom=prep_grid_text_color_custom,
        prep_grid_backing_color=prep_grid_backing_color,
        prep_grid_backing_color_custom=prep_grid_backing_color_custom,
        prep_send_pixel_bounds=prep_send_pixel_bounds,
        prep_min_pixels=prep_min_pixels,
        prep_max_pixels=prep_max_pixels,
        prep_custom_resize_enabled=prep_custom_resize_enabled,
        prep_custom_resize_width=prep_custom_resize_width,
        prep_custom_resize_height=prep_custom_resize_height,
    )

    pipeline_params = {"detector_temperature": float(detector_temp or 0.9)}

    tracker = MultiAlgorithmTracker(tracker_algorithm)
    last_payload = None
    last_boxes: List[Any] = []
    errors = 0
    # Keep last raw frame for viewer base if all frames fail
    last_raw = frames[-1] if frames else None
    for idx, f in enumerate(frames):
        progress(
            (idx + 1) / len(frames), desc=f"Detecting frame {idx + 1}/{len(frames)}"
        )
        try:
            boxes, _hud = run_vlm_detect(
                f,
                categories,
                category_definitions,
                base_url,
                api_key,
                model_name,
                prep_config,
                pipeline_params,
            )
            tracked_boxes = tracker.update_with_detections(
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if cv2 is not None else f,
                boxes,
            ) if boxes else []
            last_boxes = tracked_boxes if tracked_boxes else boxes
            last_raw = f
        except Exception:
            last_boxes = []
            errors += 1
            last_raw = f
        # Build viewer payload for this frame – will be overwritten, keep last
        try:
            anns = realtime_boxes_to_annotations(last_boxes)
        except Exception:
            anns = []
        last_payload = build_viewer_payload(last_raw, anns)

    total = len(frames)
    status = f"Successfully processed {total} sampled frame(s) from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    # Fallback: if last_payload is None, return raw last frame with empty annotations
    if last_payload is None and last_raw is not None:
        last_payload = build_viewer_payload(last_raw, [])
    return last_payload, status
