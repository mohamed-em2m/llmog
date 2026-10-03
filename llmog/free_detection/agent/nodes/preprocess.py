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
from esr.manager import ESRConfig, upscale_pil
from esr.project import esr_growth_factor, scaled_pixel_param

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
    true_orig_w, true_orig_h = base_image_raw.size

    # 1b. Optional Real-ESRGAN stage 0: super-resolve BEFORE anything else
    # so the VLM sees a high-resolution working image. Normalized (0-1000)
    # coordinates are identical for the original and its upscaled twin
    # (pure scale), so all downstream mapping stays valid; final outputs
    # are projected back onto the true original dims (see node_finalize).
    # Disabled configs return the input untouched (no torch needed).
    esr_cfg = ESRConfig.from_config(prep_cfg)
    working_image, esr_info = upscale_pil(base_image_raw, esr_cfg)
    esr_applied = bool(esr_info.get("applied", False))

    # 2. Apply custom resize OR resolution scaling and padding
    use_custom_resize = prep_cfg.get("custom_resize", False)
    if use_custom_resize:
        custom_width = prep_cfg.get("custom_resize_width", 1024)
        custom_height = prep_cfg.get("custom_resize_height", 1024)
        preprocessed_image, prep_info = preprocess_custom_resize(
            working_image, target_width=custom_width, target_height=custom_height
        )
    else:
        preprocessed_image, prep_info = preprocess_resolution(
            working_image,
            enabled=prep_cfg.get("resolution_enabled", False),
            target_short_edge=prep_cfg.get("target_short_edge", 1024),
            pad_to_square=prep_cfg.get("pad_to_square", False),
        )
    prep_w, prep_h = preprocessed_image.size
    prep_info = {**prep_info, "esr": esr_info}

    # 2b. Keep pixel-space cosmetics relative under ESR growth. Grid `step`
    # is already 0-1000-relative (untouched); line width / explicit font
    # size / tile size are pixels and would otherwise change meaning on the
    # bigger working image (the "red lines change measures" problem).
    # Factor is total working-vs-true-original growth, applied ONLY when
    # ESR ran so non-ESR behavior stays byte-identical.
    growth = (
        esr_growth_factor(prep_w, prep_h, true_orig_w, true_orig_h)
        if esr_applied
        else 1.0
    )
    if growth > 1.0:
        logger.info(
            "ESR growth factor %.2f (%dx%d -> %dx%d): scaling grid line "
            "width / font size / tile size to preserve relative measures.",
            growth,
            true_orig_w,
            true_orig_h,
            prep_w,
            prep_h,
        )
    grid_line_width = scaled_pixel_param(
        int(prep_cfg.get("grid_line_width", 1)), growth
    )
    _font_cfg = int(prep_cfg.get("grid_font_size", 0))
    grid_font_size = (
        _font_cfg if _font_cfg <= 0 else scaled_pixel_param(_font_cfg, growth)
    )
    tile_size_eff = scaled_pixel_param(int(prep_cfg.get("tile_size", 512)), growth)

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
        "esr_info": esr_info,
        "tile_size_eff": tile_size_eff,
        "tile_overlap": prep_cfg.get("tile_overlap", 0.2),
        "grid_style": prep_cfg.get("grid_style", "standard"),
        "grid_step": prep_cfg.get("grid_step", 100),
        "grid_line_color": prep_cfg.get("grid_line_color", "red"),
        "grid_line_width": grid_line_width,
        "grid_font_size": grid_font_size,
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
