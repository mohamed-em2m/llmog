"""LLM Object Detection Agent Package.

Provides a modular LangGraph-based iterative Detector-Judge object detection
workflow.
"""

from free_detection.agent.state import DetectionState, RoundResult
from free_detection.agent.client_utils import _call_with_retries
from free_detection.agent.parser import (
    parse_detections,
    validate_detections,
    extract_json_block,
    _strip_think_blocks,
    _strip_code_fences,
    _extract_balanced_array,
)
from free_detection.agent.visuals import (
    draw_grid,
    render_detections,
    pil_to_data_uri,
    _load_font,
    _text_with_backing,
)
from free_detection.agent.prompts import (
    PROMPTS_DIR,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
    DEFAULT_REALTIME_TEMPLATE,
    DEFAULT_CROP_VERIFY_TEMPLATE,
    DEFAULT_AUTO_LABEL_TEMPLATE,
    get_realtime_prompt,
    render_detector_prompt,
    render_judge_prompt,
    render_crop_verify_prompt,
    render_auto_label_prompt,
    _load_prompt_template,
)
from free_detection.agent.graph import build_detection_graph
from free_detection.agent.pipeline import (
    ObjectDetectionPipeline,
    FabricDefectPipeline,
)

__all__ = [
    "DetectionState",
    "RoundResult",
    "ObjectDetectionPipeline",
    "FabricDefectPipeline",
    "build_detection_graph",
    "parse_detections",
    "validate_detections",
    "extract_json_block",
    "draw_grid",
    "render_detections",
    "pil_to_data_uri",
    "DEFAULT_DETECTOR_TEMPLATE",
    "DEFAULT_JUDGE_TEMPLATE",
    "DEFAULT_REALTIME_TEMPLATE",
    "DEFAULT_CROP_VERIFY_TEMPLATE",
    "DEFAULT_AUTO_LABEL_TEMPLATE",
    "get_realtime_prompt",
    "render_detector_prompt",
    "render_judge_prompt",
    "render_crop_verify_prompt",
    "render_auto_label_prompt",
    "_call_with_retries",
    "_load_prompt_template",
    "_load_font",
    "_text_with_backing",
    "_strip_think_blocks",
    "_strip_code_fences",
    "_extract_balanced_array",
    "PROMPTS_DIR",
]
