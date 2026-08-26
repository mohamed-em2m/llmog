"""
Batch results explorer tab callbacks and viewer logic.
"""

from __future__ import annotations

import json
import logging
from typing import Tuple, Any, Optional

import gradio as gr
from PIL import Image

from interface.state import _cache_get
from interface.viewer_utils import (
    build_viewer_payload,
    pipeline_detections_to_annotations,
)

logger = logging.getLogger("detection_pipeline.explorer")


def on_explorer_image_change(selected_image: str, batch_id: str):
    """Update round dropdown choices when selected image changes."""
    batch_results = _cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return gr.update(choices=[], value=None)
    rounds = batch_results[selected_image].get("rounds", [])
    choices = ["Final Best"] + [str(r["round"]) for r in rounds]
    return gr.update(choices=choices, value="Final Best")


def on_explorer_prev(current: str, batch_id: str):
    """Navigate to previous image — arrow navigation for batch results."""
    batch_results = _cache_get(batch_id)
    if not batch_results:
        return gr.update()
    choices = list(batch_results.keys())
    if not choices:
        return gr.update()
    if not current or current not in choices:
        return gr.update(value=choices[0])
    idx = choices.index(current)
    new_idx = max(0, idx - 1)
    return gr.update(value=choices[new_idx])


def on_explorer_next(current: str, batch_id: str):
    """Navigate to next image — arrow navigation for batch results."""
    batch_results = _cache_get(batch_id)
    if not batch_results:
        return gr.update()
    choices = list(batch_results.keys())
    if not choices:
        return gr.update()
    if not current or current not in choices:
        return gr.update(value=choices[0])
    idx = choices.index(current)
    new_idx = min(len(choices) - 1, idx + 1)
    return gr.update(value=choices[new_idx])


def _viewer_payload_for(
    base_img: Optional[Image.Image],
    detections: list | None,
) -> Any:
    """Build DetectionViewer payload (image, annotations) – fast & no server drawing."""
    if base_img is None:
        return None
    if not detections:
        return build_viewer_payload(base_img, [])
    try:
        anns = pipeline_detections_to_annotations(detections, base_img.size)
    except Exception:
        anns = []
    return build_viewer_payload(base_img, anns)


def on_explorer_round_change(
    selected_image: str,
    selected_round: str,
    batch_id: str,
    show_grid: bool,
) -> Tuple[
    Optional[Image.Image],
    Any,
    str,
    str,
    str,
    str,
    str,
]:
    """Load and return round details, viewer payload, score badge, feedback, and JSON.

    Second element is a DetectionViewer tuple ``(image, annotations)`` – the
    viewer draws boxes client-side so the server avoids re-encoding a second
    annotated PIL per round (major perf win for large batches).
    """
    batch_results = _cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return (
            None,
            None,
            '<span class="score-badge">Score: -/10</span>',
            "",
            "",
            "",
            "[]",
        )

    img_data = batch_results[selected_image]

    # Lazy grid generation – only pay draw_grid cost if user toggles Show Grid
    if show_grid:
        grid_img = img_data.get("grid_original")
        if grid_img is None and img_data.get("raw_original") is not None:
            try:
                from free_detection.agent.visuals import draw_grid as _draw_grid_lazy

                cfg = img_data.get("_grid_config") or {}
                grid_img = _draw_grid_lazy(
                    img_data["raw_original"],
                    step=cfg.get("step", 250),
                    style=cfg.get("style", "standard"),
                    line_color=cfg.get("line_color", "red"),
                    line_width=cfg.get("line_width", 1),
                    font_size=cfg.get("font_size", 0),
                    text_color=cfg.get("text_color", "white"),
                    backing_color=cfg.get("backing_color", "black"),
                )
                img_data["grid_original"] = grid_img  # memoize
            except Exception as e:
                logger.warning(f"Lazy grid generation failed: {e}")
                grid_img = img_data.get("raw_original")
        src_img = grid_img if grid_img is not None else img_data.get("raw_original")
    else:
        src_img = img_data.get("raw_original")

    # Viewer base always uses the (non-grid) raw image for accurate bbox placement
    viewer_base: Optional[Image.Image] = img_data.get("raw_original") or src_img

    if not selected_round or selected_round == "Final Best":
        best_score, best_round_num, best_feedback, best_raw, best_err = (
            -1,
            -1,
            "No detections found.",
            "",
            "",
        )
        best_detections = img_data.get("detections") or []
        for r in img_data["rounds"]:
            if r["score"] > best_score:
                best_score = r["score"]
                best_round_num = r["round"]
                best_feedback = r["feedback"]
                best_raw = r["raw_text"]
                best_err = r["parse_error"]

        # Build viewer payload client-side instead of serving a pre-rendered annotated JPEG
        viewer_payload = _viewer_payload_for(viewer_base, best_detections)

        if best_score >= 0:
            score_text = f'<span class="score-badge">Best Score: {best_score}/10 (Round {best_round_num})</span>'
        else:
            score_text = '<span class="score-badge">Score: -/10</span>'
        return (
            src_img,
            viewer_payload,
            score_text,
            best_feedback,
            best_raw,
            best_err or "None",
            (
                json.dumps(img_data["detections"], indent=2)
                if img_data["detections"]
                else "[]"
            ),
        )

    try:
        round_idx = int(selected_round) - 1
        rounds = img_data["rounds"]
        if 0 <= round_idx < len(rounds):
            r = rounds[round_idx]
            round_detections = r.get("detections") or []
            viewer_payload = _viewer_payload_for(viewer_base, round_detections)
            score_text = f'<span class="score-badge">Score: {r["score"]}/10</span>'
            return (
                src_img,
                viewer_payload,
                score_text,
                r["feedback"],
                r["raw_text"],
                r["parse_error"] or "None",
                json.dumps(r["detections"], indent=2) if r["detections"] else "[]",
            )
    except Exception as e:
        logger.error(f"Error loading round details: {e}")

    return (
        src_img,
        None,
        '<span class="score-badge">Score: -/10</span>',
        "",
        "",
        "",
        "[]",
    )
