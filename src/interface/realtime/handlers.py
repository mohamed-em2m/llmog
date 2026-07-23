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
from interface.realtime.utils import (
    run_vlm_detect,
    draw_boxes_opencv,
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
) -> Tuple[dict, str, SessionDetector]:
    """Continuous Live Streaming Processor called on stream ticks."""
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        return {"boxes": boxes, "frame_w": 0, "frame_h": 0}, hud, session

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

            # Build unified preprocessing config (identical schema to tab_batch.py)
            if not prep_enabled:
                prep_config = {
                    "resolution_enabled": False,
                    "contrast_method": "none",
                    "denoise_method": "none",
                    "som_enabled": False,
                    "tiling_enabled": False,
                    "crop_verify_enabled": False,
                    "grid_style": "standard",
                    "grid_step": 100,
                    "grid_line_width": 1,
                    "grid_font_size": 0,
                    "grid_line_color": "red",
                    "grid_text_color": "white",
                    "grid_backing_color": "black",
                    "send_pixel_bounds": False,
                    "min_pixels": 200704,
                    "max_pixels": 4194304,
                    "custom_resize": False,
                    "custom_resize_width": 1024,
                    "custom_resize_height": 1024,
                }
            else:
                use_custom_resize = bool(prep_custom_resize_enabled)
                prep_config = {
                    "resolution_enabled": not use_custom_resize,
                    "target_short_edge": int(prep_short_edge or 1024),
                    "pad_to_square": bool(prep_pad_square),
                    "contrast_method": prep_contrast_method,
                    "clip_limit": 2.0,
                    "gamma": float(prep_gamma or 1.0),
                    "denoise_method": prep_denoise_method,
                    "sharpen": bool(prep_sharpen),
                    "white_balance": bool(prep_white_balance),
                    "grid_style": prep_grid_style if prep_grid_style != "Standard Red" else "standard",
                    "som_enabled": bool(prep_som_enabled),
                    "tiling_enabled": bool(prep_tiling_enabled),
                    "tile_size": int(prep_tile_size or 512),
                    "tile_overlap": float(prep_tile_overlap or 20) / 100.0,
                    "crop_verify_enabled": bool(prep_crop_verify_enabled),
                    "crop_padding": float(prep_crop_padding or 15) / 100.0,
                    "grid_step": int(prep_grid_step or 250),
                    "grid_line_width": int(prep_grid_line_width or 1),
                    "grid_font_size": int(prep_grid_font_size or 0),
                    "grid_line_color": prep_grid_line_color if prep_grid_line_color != "custom" else prep_grid_line_color_custom,
                    "grid_text_color": prep_grid_text_color if prep_grid_text_color != "custom" else prep_grid_text_color_custom,
                    "grid_backing_color": prep_grid_backing_color if prep_grid_backing_color != "custom" else prep_grid_backing_color_custom,
                    "send_pixel_bounds": bool(prep_send_pixel_bounds),
                    "min_pixels": int(prep_min_pixels) if prep_min_pixels is not None else None,
                    "max_pixels": int(prep_max_pixels) if prep_max_pixels is not None else None,
                    "custom_resize": use_custom_resize,
                    "custom_resize_width": int(prep_custom_resize_width or 1024),
                    "custom_resize_height": int(prep_custom_resize_height or 1024),
                }

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

    return (
        {"boxes": tracked_boxes, "frame_w": frame_w, "frame_h": frame_h},
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

    if not prep_enabled:
        prep_config = {
            "resolution_enabled": False,
            "contrast_method": "none",
            "denoise_method": "none",
            "som_enabled": False,
            "tiling_enabled": False,
            "crop_verify_enabled": False,
            "grid_style": "standard",
            "grid_step": 100,
            "grid_line_width": 1,
            "grid_font_size": 0,
            "grid_line_color": "red",
            "grid_text_color": "white",
            "grid_backing_color": "black",
            "send_pixel_bounds": False,
            "min_pixels": 200704,
            "max_pixels": 4194304,
            "custom_resize": False,
            "custom_resize_width": 1024,
            "custom_resize_height": 1024,
        }
    else:
        use_custom_resize = bool(prep_custom_resize_enabled)
        prep_config = {
            "resolution_enabled": not use_custom_resize,
            "target_short_edge": int(prep_short_edge or 1024),
            "pad_to_square": bool(prep_pad_square),
            "contrast_method": prep_contrast_method,
            "clip_limit": 2.0,
            "gamma": float(prep_gamma or 1.0),
            "denoise_method": prep_denoise_method,
            "sharpen": bool(prep_sharpen),
            "white_balance": bool(prep_white_balance),
            "grid_style": prep_grid_style if prep_grid_style != "Standard Red" else "standard",
            "som_enabled": bool(prep_som_enabled),
            "tiling_enabled": bool(prep_tiling_enabled),
            "tile_size": int(prep_tile_size or 512),
            "tile_overlap": float(prep_tile_overlap or 20) / 100.0,
            "crop_verify_enabled": bool(prep_crop_verify_enabled),
            "crop_padding": float(prep_crop_padding or 15) / 100.0,
            "grid_step": int(prep_grid_step or 250),
            "grid_line_width": int(prep_grid_line_width or 1),
            "grid_font_size": int(prep_grid_font_size or 0),
            "grid_line_color": prep_grid_line_color if prep_grid_line_color != "custom" else prep_grid_line_color_custom,
            "grid_text_color": prep_grid_text_color if prep_grid_text_color != "custom" else prep_grid_text_color_custom,
            "grid_backing_color": prep_grid_backing_color if prep_grid_backing_color != "custom" else prep_grid_backing_color_custom,
            "send_pixel_bounds": bool(prep_send_pixel_bounds),
            "min_pixels": int(prep_min_pixels) if prep_min_pixels is not None else None,
            "max_pixels": int(prep_max_pixels) if prep_max_pixels is not None else None,
            "custom_resize": use_custom_resize,
            "custom_resize_width": int(prep_custom_resize_width or 1024),
            "custom_resize_height": int(prep_custom_resize_height or 1024),
        }

    pipeline_params = {"detector_temperature": float(detector_temp or 0.9)}

    tracker = MultiAlgorithmTracker(tracker_algorithm)
    annotated_frames = []
    errors = 0
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
            tracked_boxes = boxes
        except Exception:
            tracked_boxes = []
            errors += 1
        annotated_frames.append(draw_boxes_opencv(f, tracked_boxes))

    status = f"Successfully processed {len(annotated_frames)} frames from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    return annotated_frames, status
