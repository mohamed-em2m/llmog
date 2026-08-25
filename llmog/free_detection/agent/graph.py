"""LangGraph StateGraph builder for the Detector-Judge object detection pipeline."""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from free_detection.agent.state import DetectionState
from free_detection.agent.nodes import (
    node_preprocess,
    node_detector,
    node_crop_verify,
    node_judge,
    node_prepare_next_round,
    route_judge_decision,
    node_finalize,
)


def build_detection_graph() -> StateGraph:
    """Build and return the uncompiled LangGraph StateGraph for detection pipeline."""
    workflow = StateGraph(DetectionState)

    # Add Nodes
    workflow.add_node("preprocess", node_preprocess)
    workflow.add_node("detector", node_detector)
    workflow.add_node("crop_verify", node_crop_verify)
    workflow.add_node("judge", node_judge)
    workflow.add_node("prepare_next_round", node_prepare_next_round)
    workflow.add_node("finalize", node_finalize)

    # Add Edges
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "detector")
    workflow.add_edge("detector", "crop_verify")
    workflow.add_edge("crop_verify", "judge")

    # Conditional branching after Judge
    workflow.add_conditional_edges(
        "judge",
        route_judge_decision,
        {
            "finalize": "finalize",
            "prepare_next_round": "prepare_next_round",
        },
    )

    workflow.add_edge("prepare_next_round", "detector")
    workflow.add_edge("finalize", END)

    return workflow
