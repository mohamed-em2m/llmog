"""
Realtime image processing, motion gating, OpenCV drawing, grid drawing, and VLM detection utilities.
"""

import time
from typing import List, Tuple, Any, Optional
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from free_detection.detection_pipeline import (
    pil_to_data_uri,
    parse_detections,
    validate_detections,
    draw_grid,
    get_realtime_prompt,
)
from free_detection.image_preprocessing import (
    preprocess_resolution,
    map_bbox_to_original,
)
from interface.realtime.state import get_pipeline


def to_small_gray(
    frame_rgb: np.ndarray, size: Tuple[int, int] = (160, 120)
) -> Optional[np.ndarray]:
    """Downscaled, blurred grayscale version of a frame, cheap to diff."""
    if cv2 is None:
        return None
    small = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def scene_has_changed(
    current_gray: np.ndarray,
    reference_gray: Optional[np.ndarray],
    pixel_diff_thresh: int = 25,
    change_ratio_thresh: float = 0.015,
) -> bool:
    """Compares current frame against last detected frame for motion gating."""
    if reference_gray is None:
        return True
    if cv2 is None:
        return True
    diff = cv2.absdiff(current_gray, reference_gray)
    _, thresholded = cv2.threshold(diff, pixel_diff_thresh, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresholded)
    total_pixels = thresholded.shape[0] * thresholded.shape[1]
    return (changed_pixels / total_pixels) > change_ratio_thresh


def draw_boxes_opencv(image_np: np.ndarray, boxes: List[Any]) -> np.ndarray:
    """Fast OpenCV bounding box rendering for video frame exports."""
    if cv2 is None or not boxes:
        return image_np
    img = image_np.copy()
    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            track_id = f" #{box[5]}" if len(box) >= 6 else ""
            full_label = f"{label}{track_id}"
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(img, pt1, pt2, (204, 255, 0), 2)
            if full_label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    full_label, font, font_scale, thickness
                )
                lbl_pt1 = (int(xmin), max(0, int(ymin) - text_h - 6))
                lbl_pt2 = (int(xmin) + text_w + 6, max(text_h + 6, int(ymin)))
                cv2.rectangle(img, lbl_pt1, lbl_pt2, (204, 255, 0), -1)
                cv2.putText(
                    img,
                    full_label,
                    (int(xmin) + 3, max(text_h + 2, int(ymin) - 3)),
                    font,
                    font_scale,
                    (17, 8, 5),
                    thickness,
                    cv2.LINE_AA,
                )
    return img


def run_vlm_detect(
    frame: np.ndarray,
    categories: list,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_info: dict,
    enable_grid: bool = True,
    grid_step: int = 250,
    grid_style: str = "standard",
    grid_line_color: str = "red",
    grid_line_width: int = 1,
    grid_font_size: int = 0,
    grid_text_color: str = "white",
    grid_backing_color: str = "black",
    free_detection: bool = False,
) -> Tuple[List[Any], str]:
    """
    Runs VLM detection on a single frame using the detection pipeline preprocessing,
    optional coordinate grid overlay, and bounding box mapping.

    When ``free_detection=True`` (or ``categories`` is empty / ``["*"]``), uses the
    dedicated ``realtime_detector`` open-vocabulary prompt so the model can name
    any object it sees without a predefined category list.
    """
    start_time = time.time()
    pil_img = Image.fromarray(frame).convert("RGB")

    # Preprocess image resolution
    proc_img, prep_info = preprocess_resolution(
        pil_img, enabled=True, target_short_edge=prep_info.get("max_res", 640)
    )

    # Apply 0-1000 scale coordinate grid overlay if enabled (matching detection_pipeline)
    if enable_grid:
        input_img = draw_grid(
            proc_img,
            step=grid_step,
            style=grid_style,
            line_color=grid_line_color,
            line_width=grid_line_width,
            font_size=grid_font_size,
            text_color=grid_text_color,
            backing_color=grid_backing_color,
        )
    else:
        input_img = proc_img

    pipeline = get_pipeline(base_url, api_key, model_name)
    img_uri = pil_to_data_uri(input_img)

    # Determine whether to use free (open-vocabulary) or targeted detection
    is_free = free_detection or not categories or categories == ["*"]
    if is_free:
        # Build the realtime open-vocabulary prompt via DynaPrompt
        realtime_cats = None if is_free and (not categories or categories == ["*"]) else categories
        prompt_text = get_realtime_prompt(realtime_cats)
        raw_output = pipeline.run_inference(
            image_uris=img_uri,
            categories=categories or ["object"],
            category_definitions="",
            custom_prompt=prompt_text,
        )
    else:
        raw_output = pipeline.run_inference(
            image_uris=img_uri,
            categories=categories,
            category_definitions="",
        )
    parsed_dets = parse_detections(raw_output)
    # In free-detection mode there's no fixed category filter — accept all labels
    # the model emits. Pass the actual detected labels so bbox validation still runs.
    if is_free:
        detected_labels = list({d.get("label", "") for d in parsed_dets if d.get("label")})
        valid_dets = validate_detections(parsed_dets, detected_labels)
    else:
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
    hud = (
        f'<div class="neo-retro-hud-stat">FPS: {fps:.1f} | '
        f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
    )
    return boxes, hud
