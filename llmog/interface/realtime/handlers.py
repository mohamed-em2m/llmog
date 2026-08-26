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


def _frame_diff_percent(a: np.ndarray, b: np.ndarray) -> float:
    """Fast downscaled mean diff % for motion gate – 64×64, ~0.1ms."""
    try:
        if a is None or b is None or a.shape != b.shape:
            return 100.0
        if cv2 is not None:
            small_a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_NEAREST)
            small_b = cv2.resize(b, (64, 64), interpolation=cv2.INTER_NEAREST)
        else:
            # fallback: center crop
            small_a = a[:: max(1, a.shape[0] // 64), :: max(1, a.shape[1] // 64)]
            small_b = b[:: max(1, b.shape[0] // 64), :: max(1, b.shape[1] // 64)]
        diff = np.mean(np.abs(small_a.astype(np.int16) - small_b.astype(np.int16)))
        return float(diff / 255.0 * 100.0)
    except Exception:
        return 100.0


def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    category_definitions: str,
    category_strategy: str,
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
    stream_mode: str = "Webcam Stream",
    same_window_on: bool = False,
) -> Tuple[Any, Any, str, SessionDetector]:
    """Continuous Live Streaming Processor – now returns both DetectionViewer payload and same-window boxes.

    Returns ``(viewer_payload, boxes_json, hud, session)`` where viewer_payload is
    ``(frame_numpy, [annotations])`` for ``DetectionViewer`` and boxes_json is
    ``{"boxes": [[ymin,xmin,ymax,xmax,label,trackId],...], "frame_w":W, "frame_h":H}``
    for the same-window canvas overlay (max speed – no WebP re-encode).
    """
    if session is None:
        session = new_session_detector()

    if frame is None:
        boxes, hud = session.snapshot()
        # No frame yet – return placeholders for both viewer and overlay
        if not boxes:
            return None, {"boxes": [], "frame_w": 0, "frame_h": 0}, hud, session
        return None, {"boxes": boxes, "frame_w": 0, "frame_h": 0}, hud, session

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

        # Performance: motion gate now actually compares frames (fast 64×64 diff)
        # instead of just checking the flag – avoids VLM calls on static scenes.
        should_submit = False
        if fresh_result or stale:
            should_submit = True
        elif not motion_gate_enabled:
            should_submit = True
        else:
            if session._last_submitted_frame is None:
                should_submit = True
            else:
                diff_pct = _frame_diff_percent(frame, session._last_submitted_frame)
                if diff_pct >= float(motion_sensitivity_pct or 1.5):
                    should_submit = True

        if should_submit:
            # copy only when we will submit (saves 1× frame copy on skipped ticks)
            session._last_submitted_frame = frame.copy()
            mode_norm = (category_strategy or "strict").lower().strip()
            categories = [c.strip() for c in (categories_str or "").split(",") if c.strip()]
            if not categories:
                if "free" in mode_norm:
                    categories = ["*"]
                else:
                    categories = ["object"]
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

    # Build both outputs: DetectionViewer payload (for interactive viewer) + boxes JSON (for same-window overlay, max speed – no WebP)
    try:
        anns = realtime_boxes_to_annotations(tracked_boxes)
    except Exception:
        anns = []
    # Perf: when the interactive viewer is hidden (same-window ⚡ overlay ON),
    # skip the per-tick PIL→WebP encode entirely — the canvas overlay draws
    # boxes client-side and is the only visible output.
    viewer_visible = ("video" not in (stream_mode or "").lower()) and not bool(
        same_window_on
    )
    viewer_payload = build_viewer_payload(frame, anns) if viewer_visible else None
    boxes_json = {"boxes": tracked_boxes, "frame_w": frame_w, "frame_h": frame_h}
    if not hud or "DETECTED" not in hud:
        hud = f'<div class="neo-retro-hud-stat">FPS: -- | DETECTED: {len(anns)}</div>' if anns else hud
    return (
        viewer_payload,
        boxes_json,
        hud,
        session,
    )


def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    category_definitions: str,
    category_strategy: str,
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
) -> Tuple[Any, Any, str]:
    """Synchronous video file sampling – returns Gallery (all sampled frames) + DetectionViewer payload (last frame).

    Gallery gives quick scan of sampled frames with OpenCV boxes; viewer gives
    interactive last-frame with DetectionViewer. Capped to 60 frames for long videos.
    """
    # Gradio Video may return dict with 'video' key or filepath string
    if isinstance(video_path, dict):
        video_path = video_path.get("video") or video_path.get("path") or video_path.get("name")
    if not video_path:
        return [], None, "No video file uploaded."
    if cv2 is None:
        return [], None, "OpenCV (cv2) is required for video processing."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], None, "Failed to open video file."

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
        return [], None, "No frames could be sampled from this video."

    # Performance: cap sampled frames to 60 to avoid 100+ VLM calls on long videos
    if len(frames) > 60:
        step = len(frames) / 60
        frames = [frames[int(i * step)] for i in range(60)]

    mode_norm = (category_strategy or "strict").lower().strip()
    categories = [c.strip() for c in (categories_str or "").split(",") if c.strip()]
    if not categories:
        categories = ["*"] if "free" in mode_norm else ["object"]
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
    # For Gallery (sample frame detection) we keep annotated frames
    from interface.realtime.utils import draw_boxes_opencv

    gallery_frames: List[np.ndarray] = []
    last_payload = None
    last_boxes: List[Any] = []
    errors = 0
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
        # Gallery keeps OpenCV-annotated frames for quick scan (restores sample frame detection)
        # Perf: downscale gallery copies to ≤960px — 60 full-res 1080p frames
        # otherwise hold ~370MB RAM and serialize slowly to the browser.
        try:
            gal = draw_boxes_opencv(f, last_boxes)
            if max(gal.shape[0], gal.shape[1]) > 960:
                scale = 960.0 / max(gal.shape[0], gal.shape[1])
                gal = cv2.resize(
                    gal,
                    (int(gal.shape[1] * scale), int(gal.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                ) if cv2 is not None else gal
            gallery_frames.append(gal)
        except Exception:
            gallery_frames.append(f)
        # Viewer keeps last frame's interactive DetectionViewer payload
        try:
            anns = realtime_boxes_to_annotations(last_boxes)
        except Exception:
            anns = []
        last_payload = build_viewer_payload(last_raw, anns)

    total = len(frames)
    status = f"Successfully processed {total} sampled frame(s) from video!"
    if errors:
        status += f" ({errors} frame(s) failed detection and were shown unannotated.)"
    if last_payload is None and last_raw is not None:
        last_payload = build_viewer_payload(last_raw, [])
    # Return both Gallery (all sampled frames) and viewer (last frame interactive)
    return gallery_frames, last_payload, status
