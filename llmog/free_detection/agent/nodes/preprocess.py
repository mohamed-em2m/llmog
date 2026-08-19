"""Preprocessing node for image enhancement, scaling, and grid config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
from PIL import Image

from free_detection.image_preprocessing import (
    preprocess_color_space,
    preprocess_custom_resize,
    preprocess_resolution,
    preprocess_contrast,
    preprocess_noise_sharpness,
)
from free_detection.agent.state import DetectionState

logger = logging.getLogger("detection_pipeline")


def node_preprocess(state: DetectionState) -> Dict[str, Any]:
    """Load original image, apply color space corrections, resize, contrast, denoise & sharpening."""
    pipeline = state["pipeline"]
    prep_cfg = pipeline.preprocessing_config
    image_path = state["image_path"]

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Load original image and correct color space / EXIF rotation
    base_image_raw = Image.open(path)
    base_image_raw = preprocess_color_space(
        base_image_raw,
        white_balance=prep_cfg.get("white_balance", False),
    )

    # 2. Apply custom resize OR resolution scaling and padding
    use_custom_resize = prep_cfg.get("custom_resize", False)
    if use_custom_resize:
        custom_width = prep_cfg.get("custom_resize_width", 1024)
        custom_height = prep_cfg.get("custom_resize_height", 1024)
        preprocessed_image, prep_info = preprocess_custom_resize(
            base_image_raw, target_width=custom_width, target_height=custom_height
        )
    else:
        preprocessed_image, prep_info = preprocess_resolution(
            base_image_raw,
            enabled=prep_cfg.get("resolution_enabled", False),
            target_short_edge=prep_cfg.get("target_short_edge", 1024),
            pad_to_square=prep_cfg.get("pad_to_square", False),
        )
    prep_w, prep_h = preprocessed_image.size

    # 3. Apply contrast enhancement
    preprocessed_image = preprocess_contrast(
        preprocessed_image,
        method=prep_cfg.get("contrast_method", "none"),
        clip_limit=prep_cfg.get("clip_limit", 2.0),
        gamma=prep_cfg.get("gamma", 1.0),
    )

    # 4. Apply noise filtering and sharpening
    preprocessed_image = preprocess_noise_sharpness(
        preprocessed_image,
        method=prep_cfg.get("denoise_method", "none"),
        sharpen=prep_cfg.get("sharpen", False),
    )

    return {
        "base_image_raw": base_image_raw,
        "preprocessed_image": preprocessed_image,
        "prep_info": prep_info,
        "prep_w": prep_w,
        "prep_h": prep_h,
        "grid_style": prep_cfg.get("grid_style", "standard"),
        "grid_step": prep_cfg.get("grid_step", 100),
        "grid_line_color": prep_cfg.get("grid_line_color", "red"),
        "grid_line_width": prep_cfg.get("grid_line_width", 1),
        "grid_font_size": prep_cfg.get("grid_font_size", 0),
        "grid_text_color": prep_cfg.get("grid_text_color", "white"),
        "grid_backing_color": prep_cfg.get("grid_backing_color", "black"),
        "current_round": 1,
        "history": [],
        "best": {"score": -1, "annotated": None, "detections": None, "round": 0},
        "feedback": None,
        "judge_actions": None,
        "previous_detections_prep": None,
        "is_finished": False,
    }
