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

logger = logging.getLogger("detection_pipeline.explorer")


def on_explorer_image_change(selected_image: str, batch_id: str):
    """Update round dropdown choices when selected image changes."""
    batch_results = _cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return gr.update(choices=[], value=None)
    rounds = batch_results[selected_image].get("rounds", [])
    choices = ["Final Best"] + [str(r["round"]) for r in rounds]
    return gr.update(choices=choices, value="Final Best")


def on_explorer_round_change(
    selected_image: str,
    selected_round: str,
    batch_id: str,
    show_grid: bool,
) -> Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    str,
    str,
    str,
    str,
    str,
]:
    """Load and return round details, images, score badge, feedback, and JSON for selected round."""
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
    src_img = img_data["grid_original"] if show_grid else img_data["raw_original"]

    if not selected_round or selected_round == "Final Best":
        best_annotated = img_data["best_annotated"]
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

        display_img = best_annotated if best_detections else src_img

        if best_score >= 0:
            score_text = f'<span class="score-badge">Best Score: {best_score}/10 (Round {best_round_num})</span>'
        else:
            score_text = '<span class="score-badge">Score: -/10</span>'
        return (
            src_img,
            display_img,
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
            display_img = r["image"] if round_detections else src_img
            score_text = f'<span class="score-badge">Score: {r["score"]}/10</span>'
            return (
                src_img,
                display_img,
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
