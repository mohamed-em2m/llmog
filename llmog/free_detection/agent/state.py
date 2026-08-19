"""State definition and data structures for the Object Detection agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict
from PIL import Image


@dataclass
class RoundResult:
    round: int
    detections: list
    score: int
    feedback: str
    raw_detector_output: str
    parse_error: Optional[str] = None
    actions: Optional[str] = None  # Structured action list from judge


class DetectionState(TypedDict, total=False):
    # Pipeline Reference / Context
    pipeline: Any

    # Input Parameters
    image_path: str
    categories: List[str]
    category_definitions: str
    output_dir: Optional[str]
    show_plot: bool
    progress_callback: Optional[Callable]

    # Preprocessed Image Artifacts
    base_image_raw: Image.Image
    preprocessed_image: Image.Image
    prep_info: Dict[str, Any]
    prep_w: int
    prep_h: int

    # Grid / Preprocessing Settings
    grid_style: str
    grid_step: int
    grid_line_color: str
    grid_line_width: int
    grid_font_size: int
    grid_text_color: str
    grid_backing_color: str

    # Iteration & Convergence State
    current_round: int
    max_rounds: int
    score_threshold: int

    # Feedback / Memory for next round
    feedback: Optional[str]
    judge_actions: Optional[str]
    previous_detections_prep: Optional[List[Dict[str, Any]]]

    # Current Round Outputs
    detections_prep: List[Dict[str, Any]]
    detections_orig: List[Dict[str, Any]]
    raw_detector_output: str
    parse_error: Optional[str]

    # Judge Evaluation Outputs
    score: int
    judge_feedback: str

    # Aggregated Results
    history: List[RoundResult]
    best: Dict[str, Any]
    is_finished: bool
