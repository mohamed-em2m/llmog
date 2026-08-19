"""Object detection pipeline: backward-compatibility module re-exporting from

`free_detection.agent`.
"""

from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"

from free_detection.agent import (
    DetectionState,
    RoundResult,
    ObjectDetectionPipeline,
    FabricDefectPipeline,
    build_detection_graph,
    parse_detections,
    validate_detections,
    extract_json_block,
    draw_grid,
    render_detections,
    pil_to_data_uri,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
    DEFAULT_REALTIME_TEMPLATE,
    get_realtime_prompt,
    render_detector_prompt,
    render_judge_prompt,
    _call_with_retries,
    _load_prompt_template,
    _load_font,
    _text_with_backing,
    _strip_think_blocks,
    _strip_code_fences,
    _extract_balanced_array,
    PROMPTS_DIR,
)
from free_detection.agent.nodes.detector import _filter_and_translate_feedback_for_tile

__all__ = [
    "ObjectDetectionPipeline",
    "FabricDefectPipeline",
    "DetectionState",
    "RoundResult",
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
    "get_realtime_prompt",
    "render_detector_prompt",
    "render_judge_prompt",
    "_call_with_retries",
    "_load_prompt_template",
    "_load_font",
    "_text_with_backing",
    "_strip_think_blocks",
    "_strip_code_fences",
    "_extract_balanced_array",
    "_filter_and_translate_feedback_for_tile",
    "PROMPTS_DIR",
]

if __name__ == "__main__":
    from openai import OpenAI

    local_client = OpenAI(
        api_key="not-needed",
        base_url="http://localhost:8080/v1",
    )

    categories = ["person", "car", "bicycle", "dog", "cat"]
    definitions = """
- person: a human being
- car: a 4-wheeled motor vehicle
- bicycle: a 2-wheeled human-powered vehicle
- dog: a domestic canine
- cat: a domestic feline
"""
    image_path = "/path/to/your/image.jpg"

    local_pipeline = ObjectDetectionPipeline(
        client=local_client,
        detector_model="local-model",
        judge_model="local-model",
        max_rounds=2,
        score_threshold=8,
        external_api=False,
    )
