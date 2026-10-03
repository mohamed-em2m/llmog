"""Finalization and disk persistence node."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from PIL import Image

from free_detection.agent.state import DetectionState, RoundResult
from free_detection.agent.visuals import render_detections

logger = logging.getLogger("detection_pipeline")


def persist_results(
    output_dir: str,
    base_image: Image.Image,
    best: dict,
    history: List[RoundResult],
) -> None:
    """Persist best annotated image, detections JSON, and round history to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if best["annotated"] is not None:
        best["annotated"].save(out / "best_annotated.jpg")

    detections_payload = best.get("detections") or []
    (out / "best_detections.json").write_text(json.dumps(detections_payload, indent=2))

    history_payload = [
        {
            "round": r.round,
            "score": r.score,
            "detections": r.detections,
            "feedback": r.feedback,
            "actions": r.actions,
            "parse_error": r.parse_error,
            "raw_detector_output": r.raw_detector_output,
        }
        for r in history
    ]
    (out / "history.json").write_text(json.dumps(history_payload, indent=2))
    logger.info("Persisted results to %s", out.resolve())


def node_finalize(state: DetectionState) -> Dict[str, Any]:
    """Persist final outputs, plot if requested, and mark finished."""
    base_image_raw = state["base_image_raw"]
    best = state["best"]
    history = state["history"]
    output_dir = state.get("output_dir")
    show_plot = state.get("show_plot", False)

    logger.info("Best result: round %d with score %d/10", best["round"], best["score"])

    # ESR projection: the round loop annotates the high-res WORKING image
    # (what the VLM saw). Final outputs must be the ORIGINAL image with
    # boxes fitted to its width/height -- normalized (0-1000) detections are
    # resolution-independent, so re-render them on base_image_raw. The JSON
    # payload is untouched (already resolution-free).
    esr_info = state.get("esr_info") or {}
    if esr_info.get("applied") and best.get("detections"):
        try:
            best = {
                **best,
                "annotated": render_detections(base_image_raw, best["detections"]),
            }
            logger.info(
                "ESR: final annotated image re-rendered on original dims %dx%d.",
                base_image_raw.width,
                base_image_raw.height,
            )
        except Exception as exc:
            logger.warning("ESR final re-render failed, keeping working image: %s", exc)

    if output_dir:
        persist_results(output_dir, base_image_raw, best, history)

    if show_plot and best["annotated"] is not None:
        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(best["annotated"])
            plt.axis("off")
            plt.title(
                f"Best detections (round {best['round']}, score {best['score']}/10)"
            )
            plt.show()
        except Exception as exc:
            logger.debug("Failed to display plot: %s", exc)

    return {"is_finished": True}
