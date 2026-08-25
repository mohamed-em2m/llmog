"""Crop & Verify second-pass validation node."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from free_detection.agent.state import DetectionState

logger = logging.getLogger("detection_pipeline")


def node_crop_verify(state: DetectionState) -> Dict[str, Any]:
    """Perform second-pass Crop & Verify validation if enabled."""
    pipeline = state["pipeline"]
    prep_cfg = pipeline.preprocessing_config
    detections_prep = state.get("detections_prep", [])

    if not detections_prep or not prep_cfg.get("crop_verify_enabled", False):
        return {"detections_prep": detections_prep}

    preprocessed_image = state["preprocessed_image"]
    prep_w, prep_h = state["prep_w"], state["prep_h"]
    crop_padding = prep_cfg.get("crop_padding", 0.15)
    logger.info(
        "Crop & Verify validation enabled. Validating %d detections...",
        len(detections_prep),
    )

    verified_detections = []

    def verify_single(det):
        x1, y1, x2, y2 = det["bbox_2d"]
        px1 = x1 * prep_w / 1000
        py1 = y1 * prep_h / 1000
        px2 = x2 * prep_w / 1000
        py2 = y2 * prep_h / 1000

        pw = px2 - px1
        ph = py2 - py1
        pad_w = pw * crop_padding
        pad_h = ph * crop_padding

        cx1 = max(0, int(px1 - pad_w))
        cy1 = max(0, int(py1 - pad_h))
        cx2 = min(prep_w, int(px2 + pad_w))
        cy2 = min(prep_h, int(py2 + pad_h))

        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            return det, True

        crop_img = preprocessed_image.crop((cx1, cy1, cx2, cy2))
        is_valid = pipeline.verify_crop(crop_img, det["label"])
        return det, is_valid

    with ThreadPoolExecutor(max_workers=4) as executor:
        verification_results = list(executor.map(verify_single, detections_prep))

    for det, is_valid in verification_results:
        if is_valid:
            verified_detections.append(det)
        else:
            logger.info(
                "Crop & Verify: discarded detection box %s for label '%s'",
                det["bbox_2d"],
                det["label"],
            )

    return {"detections_prep": verified_detections}
