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
from interface.viewer_utils import build_viewer_payload, pipeline_detections_to_annotations

logger = logging.getLogger("detection_pipeline.explorer")

_EMPTY_SCORE = '<span class="score-badge">Score: -/10</span>'
_EMPTY_POS = '<span class="xp-pos">–&hairsp;/&hairsp;–</span>'


def _round_choices(batch_results: Optional[dict], image_key: Optional[str]) -> list[str]:
    """Dropdown choices for the round selector: 'Final Best' + each round number."""
    if not batch_results or not image_key or image_key not in batch_results:
        return []
    rounds = batch_results[image_key].get("rounds", [])
    return ["Final Best"] + [str(r["round"]) for r in rounds]


def _pos_badge(current: Optional[str], batch_results: Optional[dict]) -> str:
    """Position badge for the results explorer — '3 / 10'."""
    choices = list(batch_results.keys()) if batch_results else []
    if not choices:
        return _EMPTY_POS
    idx = choices.index(current) if current in choices else 0
    return f'<span class="xp-pos">{idx + 1}&hairsp;/&hairsp;{len(choices)}</span>'


def _step_image_key(current: Optional[str], batch_results: Optional[dict], direction: int) -> str:
    """Return the image key ``direction`` steps away from ``current`` (clamped, not wrapped)."""
    choices = list(batch_results.keys()) if batch_results else []
    if not choices:
        return current or ""
    if not current or current not in choices:
        return choices[0]
    idx = choices.index(current)
    new_idx = max(0, idx + direction) if direction < 0 else min(len(choices) - 1, idx + direction)
    return choices[new_idx]


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


def _lazy_grid(img_data: dict, show_grid: bool) -> Optional[Image.Image]:
    """Return the grid-overlay image for display, generating + memoizing it on first use."""
    if not show_grid:
        return img_data.get("raw_original")
    grid_img = img_data.get("grid_original")
    if grid_img is not None:
        return grid_img
    raw = img_data.get("raw_original")
    if raw is None:
        return None
    try:
        from free_detection.agent.visuals import draw_grid as _draw_grid_lazy

        cfg = img_data.get("_grid_config") or {}
        grid_img = _draw_grid_lazy(
            raw,
            step=cfg.get("step", 250),
            style=cfg.get("style", "standard"),
            line_color=cfg.get("line_color", "red"),
            line_width=cfg.get("line_width", 1),
            font_size=cfg.get("font_size", 0),
            text_color=cfg.get("text_color", "white"),
            backing_color=cfg.get("backing_color", "black"),
        )
        img_data["grid_original"] = grid_img  # memoize
        return grid_img
    except Exception as e:
        logger.warning(f"Lazy grid generation failed: {e}")
        return raw


def _round_result(img_data: dict, selected_round: Optional[str]) -> Tuple:
    """Build (viewer_payload, score_html, feedback, raw_text, parse_error, json) for one round."""
    viewer_base: Optional[Image.Image] = img_data.get("raw_original")

    if not selected_round or selected_round == "Final Best":
        best_score, best_round_num, best_feedback, best_raw, best_err = -1, -1, "No detections found.", "", ""
        best_detections = img_data.get("detections") or []
        for r in img_data.get("rounds", []):
            if r.get("score", -1) > best_score:
                best_score, best_round_num = r["score"], r["round"]
                best_feedback, best_raw, best_err = r.get("feedback", ""), r.get("raw_text", ""), r.get("parse_error", "")

        score_html = (
            f'<span class="score-badge">Best Score: {best_score}/10 (Round {best_round_num})</span>'
            if best_score >= 0
            else _EMPTY_SCORE
        )
        return (
            _viewer_payload_for(viewer_base, best_detections),
            score_html,
            best_feedback,
            best_raw,
            best_err or "None",
            json.dumps(img_data.get("detections", []), indent=2) if img_data.get("detections") else "[]",
        )

    try:
        round_idx = int(selected_round) - 1
        rounds = img_data.get("rounds", [])
        if 0 <= round_idx < len(rounds):
            r = rounds[round_idx]
            return (
                _viewer_payload_for(viewer_base, r.get("detections") or []),
                f'<span class="score-badge">Score: {r.get("score", "-")}/10</span>',
                r.get("feedback", ""),
                r.get("raw_text", ""),
                r.get("parse_error") or "None",
                json.dumps(r.get("detections", []), indent=2) if r.get("detections") else "[]",
            )
    except Exception as e:
        logger.error(f"Error loading round details: {e}")

    return (None, _EMPTY_SCORE, "", "", "", "[]")


def _load_round_data(
    selected_image: Optional[str],
    selected_round: Optional[str],
    batch_id: str,
    show_grid: bool,
) -> Tuple[Optional[Image.Image], Any, str, str, str, str, str]:
    """Load display image, viewer payload, score, feedback, raw, parse error, JSON."""
    batch_results = _cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return (None, None, _EMPTY_SCORE, "", "", "", "[]")

    img_data = batch_results[selected_image]
    src_img = _lazy_grid(img_data, show_grid) or img_data.get("raw_original")
    payload, score_html, feedback, raw, err, dets_json = _round_result(img_data, selected_round)
    return (src_img, payload, score_html, feedback, raw, err, dets_json)


# ── Unified Atomic Navigation ───────────────────────────────────────────────

def navigate_batch_explorer(
    action: str,
    current_image: Optional[str],
    current_round: Optional[str],
    batch_id: str,
    show_grid: bool = True,
) -> Tuple[Any, str, Any, Optional[Image.Image], Any, str, str, str, str, str]:
    """Single unified, ultra-fast dispatcher for all explorer navigation events.

    Actions:
      - 'first': Jump to the first batch image.
      - 'prev': Step to the previous image.
      - 'next': Step to the next image.
      - 'last': Jump to the last batch image.
      - 'image_select': User selected a new image from the dropdown.
      - 'round_select': User selected a different round for the current image.
      - 'grid_toggle': User toggled the coordinate grid.

    Returns 10 outputs in a single atomic server round-trip:
      1. explorer_image_select (gr.update value)
      2. explorer_pos_display (HTML position chip '3 / 10')
      3. explorer_round_select (gr.update with choices and selected round)
      4. source_image_viewer (PIL Image)
      5. best_annotated_viewer (DetectionViewer payload tuple)
      6. round_score_display (HTML score badge)
      7. round_feedback_display (str)
      8. round_raw_response_display (str)
      9. round_parse_error_display (str)
      10. detections_json_box (JSON str)
    """
    batch_results = _cache_get(batch_id)
    choices = list(batch_results.keys()) if batch_results else []

    if not choices:
        return (
            gr.update(choices=[], value=None),
            _EMPTY_POS,
            gr.update(choices=[], value=None),
            None,
            None,
            _EMPTY_SCORE,
            "",
            "",
            "",
            "[]",
        )

    # Determine target image key
    if action == "first":
        target_image = choices[0]
        target_round = "Final Best"
    elif action == "last":
        target_image = choices[-1]
        target_round = "Final Best"
    elif action == "prev":
        target_image = _step_image_key(current_image, batch_results, -1)
        target_round = "Final Best"
    elif action == "next":
        target_image = _step_image_key(current_image, batch_results, +1)
        target_round = "Final Best"
    elif action == "image_select":
        target_image = current_image if (current_image and current_image in choices) else choices[0]
        target_round = "Final Best"
    elif action in ("round_select", "grid_toggle"):
        target_image = current_image if (current_image and current_image in choices) else choices[0]
        target_round = current_round or "Final Best"
    else:
        target_image = current_image if (current_image and current_image in choices) else choices[0]
        target_round = "Final Best"

    # Compute choices & badge
    pos_html = _pos_badge(target_image, batch_results)
    round_choices_list = _round_choices(batch_results, target_image)
    if target_round not in round_choices_list and round_choices_list:
        target_round = "Final Best"

    round_update = gr.update(choices=round_choices_list, value=target_round)
    image_update = gr.update(value=target_image, choices=choices)

    # Load round data
    src_img, payload, score_html, feedback, raw, err, dets_json = _load_round_data(
        target_image, target_round, batch_id, show_grid
    )

    return (
        image_update,
        pos_html,
        round_update,
        src_img,
        payload,
        score_html,
        feedback,
        raw,
        err,
        dets_json,
    )


# ── Backward-Compatible Wrappers ───────────────────────────────────────────

def on_explorer_image_change(selected_image: str, batch_id: str, show_grid: bool = True):
    return navigate_batch_explorer("image_select", selected_image, "Final Best", batch_id, show_grid)


def on_explorer_round_change(selected_image: str, selected_round: str, batch_id: str, show_grid: bool):
    return _load_round_data(selected_image, selected_round, batch_id, show_grid)


def on_explorer_prev(current: str, batch_id: str, show_grid: bool = True):
    return navigate_batch_explorer("prev", current, "Final Best", batch_id, show_grid)


def on_explorer_next(current: str, batch_id: str, show_grid: bool = True):
    return navigate_batch_explorer("next", current, "Final Best", batch_id, show_grid)


def on_explorer_first(current: str, batch_id: str, show_grid: bool = True):
    return navigate_batch_explorer("first", current, "Final Best", batch_id, show_grid)


def on_explorer_last(current: str, batch_id: str, show_grid: bool = True):
    return navigate_batch_explorer("last", current, "Final Best", batch_id, show_grid)


def on_explorer_pos(current: str, batch_id: str) -> str:
    return _pos_badge(current, _cache_get(batch_id))
