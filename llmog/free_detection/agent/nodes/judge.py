"""Judge critique node evaluating bounding box detections."""

from __future__ import annotations

import logging
from typing import Any, Dict

from free_detection.image_preprocessing import (
    draw_premium_grid,
    map_bbox_to_original,
)
from free_detection.agent.state import DetectionState, RoundResult
from free_detection.agent.visuals import render_detections, pil_to_data_uri

logger = logging.getLogger("detection_pipeline")


def node_judge(state: DetectionState) -> Dict[str, Any]:
    """Critique current detections against the original image and calculate quality score."""
    pipeline = state["pipeline"]
    base_image_raw = state["base_image_raw"]
    preprocessed_image = state["preprocessed_image"]
    prep_info = state["prep_info"]
    detections_prep = state.get("detections_prep", [])
    category_definitions = state["category_definitions"]
    round_num = state["current_round"]
    progress_callback = state.get("progress_callback")
    history = list(state.get("history", []))
    best = dict(
        state.get(
            "best",
            {"score": -1, "annotated": None, "detections": None, "round": 0},
        )
        or {}
    )

    grid_style = state["grid_style"]
    grid_step = state["grid_step"]
    grid_line_color = state["grid_line_color"]
    grid_line_width = state["grid_line_width"]
    grid_font_size = state["grid_font_size"]
    grid_text_color = state["grid_text_color"]
    grid_backing_color = state["grid_backing_color"]

    # 1. Map coordinates from preprocessed scale back to original scale
    detections_orig = []
    for det in detections_prep:
        mapped_box = map_bbox_to_original(det["bbox_2d"], prep_info)
        detections_orig.append({"label": det["label"], "bbox_2d": mapped_box})

    annotated_orig = render_detections(base_image_raw, detections_orig)

    # 2. Draw preprocessed annotated view with grid for the judge
    annotated_prep = render_detections(preprocessed_image, detections_prep)
    annotated_prep_with_grid = draw_premium_grid(
        annotated_prep,
        style=grid_style,
        step=grid_step,
        line_color=grid_line_color,
        line_width=grid_line_width,
        font_size=grid_font_size,
        text_color=grid_text_color,
        backing_color=grid_backing_color,
    )
    annotated_prep_uri = pil_to_data_uri(annotated_prep_with_grid)

    # 3. Setup original scale background with grid for the judge
    grid_original_prep = draw_premium_grid(
        preprocessed_image,
        style=grid_style,
        step=grid_step,
        line_color=grid_line_color,
        line_width=grid_line_width,
        font_size=grid_font_size,
        text_color=grid_text_color,
        backing_color=grid_backing_color,
    )
    grid_original_prep_uri = pil_to_data_uri(grid_original_prep)

    # 4. Request judge critique
    score, judge_feedback, judge_actions = pipeline.judge_detections(
        original_grid_uri=grid_original_prep_uri,
        annotated_grid_uri=annotated_prep_uri,
        detections=detections_prep,
        category_definitions=category_definitions,
    )

    logger.info("Judge score: %d/10", score)
    logger.info("Judge feedback:\n%s", judge_feedback)

    round_result = RoundResult(
        round=round_num,
        detections=detections_orig,
        score=score,
        feedback=judge_feedback,
        raw_detector_output=state.get("raw_detector_output", ""),
        parse_error=state.get("parse_error"),
        actions=judge_actions,
    )
    history.append(round_result)

    if progress_callback:
        try:
            progress_callback(round_result, annotated_orig)
        except Exception:
            logger.warning("progress_callback raised an exception", exc_info=True)

    if score > best["score"]:
        best = {
            "score": score,
            "annotated": annotated_orig,
            "detections": detections_orig,
            "round": round_num,
        }

    return {
        "detections_orig": detections_orig,
        "score": score,
        "judge_feedback": judge_feedback,
        "judge_actions": judge_actions,
        "history": history,
        "best": best,
    }
