"""Graph node implementations for Object Detection agent."""

from free_detection.agent.nodes.preprocess import node_preprocess
from free_detection.agent.nodes.detector import node_detector
from free_detection.agent.nodes.crop_verify import node_crop_verify
from free_detection.agent.nodes.judge import node_judge
from free_detection.agent.nodes.loop import (
    node_prepare_next_round,
    route_judge_decision,
)
from free_detection.agent.nodes.finalize import node_finalize, persist_results

__all__ = [
    "node_preprocess",
    "node_detector",
    "node_crop_verify",
    "node_judge",
    "node_prepare_next_round",
    "route_judge_decision",
    "node_finalize",
    "persist_results",
]
