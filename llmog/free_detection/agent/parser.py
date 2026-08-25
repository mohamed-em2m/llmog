"""Parsing and validation utilities for detector and judge outputs."""

from __future__ import annotations

import json
import logging
import re
import json_repair
from typing import Any, Dict, List

logger = logging.getLogger("detection_pipeline")


def _strip_think_blocks(text: str) -> str:
    """Remove thinking-mode artifacts from model output.

    Handles both the ``<thinking>…</thinking>`` tags some backends wrap around
    reasoning and the `` thinking…response `` suffix form emitted by others.
    """
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r" thinking.*? response", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_code_fences(text: str) -> str:
    """Recursively strip markdown code fences (``` … ```) from text."""
    text = text.strip()
    changed = True
    while changed:
        new = re.sub(r"^```[a-zA-Z]*\r?\n?(.*?)```\s*$", r"\1", text, flags=re.DOTALL)
        new = new.strip()
        changed = new != text
        text = new
    return text


def _extract_balanced_array(text: str) -> str:
    """
    Find the outermost JSON array in *text* by scanning for balanced brackets.
    Returns the matched substring, or *text* unchanged if no array is found.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return text


def extract_json_block(text: str) -> str:
    """Best-effort extraction of a JSON array from free-form model text."""
    text = _strip_code_fences(text)
    if "[" in text:
        return _extract_balanced_array(text)
    return text


def parse_detections(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse the model's raw response into a list of detection dicts.
    Raises ValueError (with the offending text attached) on failure so callers
    can log/inspect it instead of silently losing the round's output.
    """
    cleaned = _strip_think_blocks(raw_text)

    # Prefer the content inside <answer>…</answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", cleaned, re.DOTALL)
    candidate = answer_match.group(1).strip() if answer_match else cleaned

    json_block = extract_json_block(candidate)

    try:
        # Use json_repair to repair and parse JSON directly into Python objects
        parsed = json_repair.loads(json_block)
    except Exception:
        try:
            parsed = json_repair.loads(candidate)
        except Exception as exc:
            raise ValueError(
                f"Could not parse detections JSON: {exc}\nRaw text was:\n{raw_text}"
            ) from exc

    if not isinstance(parsed, (dict, list)):
        try:
            parsed = json_repair.loads(candidate)
        except Exception:
            pass

    if isinstance(parsed, dict):
        for key in ("detections", "objects", "results", "items", "data"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON array of detections, got: {type(parsed).__name__}"
        )
    return parsed


def validate_detections(
    detections: List[Dict[str, Any]], categories: List[str]
) -> List[Dict[str, Any]]:
    """
    Drop malformed entries (bad label, bad/degenerate bbox) instead of letting
    them silently corrupt rendering and the judge prompt. Logs what it drops.
    """
    valid_labels = set(categories)
    cleaned = []
    for i, item in enumerate(detections):
        if not isinstance(item, dict):
            logger.warning("Dropping detection #%d: not an object (%r)", i, item)
            continue

        label = item.get("label")
        if label not in valid_labels:
            logger.warning("Dropping detection #%d: unknown label %r", i, label)
            continue

        bbox = item.get("bbox_2d")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            logger.warning(
                "Dropping detection #%d (%s): malformed bbox %r", i, label, bbox
            )
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            logger.warning(
                "Dropping detection #%d (%s): non-numeric bbox %r", i, label, bbox
            )
            continue

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0, min(1000, x1)), max(0, min(1000, x2))
        y1, y2 = max(0, min(1000, y1)), max(0, min(1000, y2))

        if x2 - x1 < 1 or y2 - y1 < 1:
            logger.warning(
                "Dropping detection #%d (%s): degenerate bbox after clamping %r",
                i,
                label,
                bbox,
            )
            continue

        cleaned.append(
            {"label": label, "bbox_2d": [int(x1), int(y1), int(x2), int(y2)]}
        )
    return cleaned
