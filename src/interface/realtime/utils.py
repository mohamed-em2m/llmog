"""
Realtime image processing, OpenCV drawing, and VLM detection utilities.
"""

import time
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional
from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
    draw_grid,
    get_realtime_prompt,
)
from free_detection.image_preprocessing import map_bbox_to_original
from interface.realtime.state import get_pipeline


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
    category_definitions: str,
    base_url: str,
    api_key: str,
    model_name: str,
    prep_config: dict,
    pipeline_params: dict = None,
    free_detection: bool = False,
) -> Tuple[List[Any], str]:
    """
    Runs VLM detection on a single frame by instantiating/retrieving ObjectDetectionPipeline
    and running its full preprocessing pipeline & inference.
    """
    start_time = time.time()
    orig_h, orig_w = frame.shape[0], frame.shape[1]

    pipeline_params = pipeline_params or {}

    # Get pipeline instance configured with model and preprocessing_config
    pipeline = get_pipeline(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        detector_temperature=pipeline_params.get("detector_temperature", 0.9),
        detector_max_tokens=pipeline_params.get("detector_max_tokens", 4096),
        preprocessing_config=prep_config,
    )

    # Save numpy frame temporarily to pass to pipeline.run or run preprocessing directly
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray(frame).save(tmp_path)

    try:
        # Run detection using ObjectDetectionPipeline
        best, _ = pipeline.run(
            image_path=tmp_path,
            categories=categories or ["object"],
            category_definitions=category_definitions or "",
            show_plot=False,
            output_dir=None,
        )
        valid_dets = best.get("detections") or []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    boxes = []
    for d in valid_dets:
        bbox = d.get("bbox_2d", [])
        lbl = d.get("label", "")
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
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
