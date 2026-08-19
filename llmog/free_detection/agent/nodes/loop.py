"""Loop iteration and conditional branching nodes."""

from __future__ import annotations

import logging
from typing import Any, Dict

from free_detection.agent.state import DetectionState

logger = logging.getLogger("detection_pipeline")


def route_judge_decision(state: DetectionState) -> str:
    """Conditional router: check if stopping condition reached or continue to next round."""
    pipeline = state["pipeline"]
    score = state.get("score", 0)
    round_num = state["current_round"]
    max_rounds = state.get("max_rounds", pipeline.max_rounds)
    score_threshold = state.get("score_threshold", pipeline.score_threshold)

    if score >= score_threshold:
        logger.info(
            "Score threshold (%d) reached at round %d, stopping.",
            score_threshold,
            round_num,
        )
        return "finalize"

    if round_num >= max_rounds:
        logger.info(
            "Max rounds (%d) reached at round %d, stopping.",
            max_rounds,
            round_num,
        )
        return "finalize"

    return "prepare_next_round"


def node_prepare_next_round(state: DetectionState) -> Dict[str, Any]:
    """Carry forward feedback and detections for next round."""
    return {
        "current_round": state["current_round"] + 1,
        "feedback": state["judge_feedback"],
        "judge_actions": state["judge_actions"],
        "previous_detections_prep": state["detections_prep"],
    }
