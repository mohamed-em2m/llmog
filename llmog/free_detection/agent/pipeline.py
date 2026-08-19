"""Object detection pipeline: a VLM "detector" agent proposes bounding
boxes for objects in an image, a VLM "judge" agent critiques them against
the original image, and the loop repeats with feedback until a score
threshold is hit or rounds run out.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from PIL import Image
from openai import OpenAI

from free_detection.agent.client_utils import _call_with_retries
from free_detection.agent.graph import build_detection_graph
from free_detection.agent.parser import _strip_think_blocks
from free_detection.agent.prompts import (
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
    render_detector_prompt,
    render_judge_prompt,
)
from free_detection.agent.state import DetectionState, RoundResult
from free_detection.agent.visuals import pil_to_data_uri
from free_detection.agent.nodes.finalize import persist_results

logger = logging.getLogger("detection_pipeline")


class ObjectDetectionPipeline:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        detector_client: Optional[OpenAI] = None,
        judge_client: Optional[OpenAI] = None,
        detector_model: str = "gpt-4.1",
        judge_model: str = "gpt-4.1",
        max_rounds: int = 1,
        score_threshold: int = 8,
        detector_template: str = DEFAULT_DETECTOR_TEMPLATE,
        judge_template: str = DEFAULT_JUDGE_TEMPLATE,
        detector_max_tokens: int = 4096,
        judge_max_tokens: int = 1024,
        api_retries: int = 3,
        detector_temperature: float = 0.9,
        detector_top_p: float = 0.95,
        judge_temperature: float = 0.2,
        preprocessing_config: Optional[dict] = None,
        judge_enable_thinking: bool = False,
        feedback_image_mode: str = "original",
        external_api: bool = False,
        sampling_params_supported: bool = True,
    ):
        self.detector_client = detector_client or client
        self.judge_client = judge_client or client
        if self.detector_client is None or self.judge_client is None:
            raise ValueError(
                "Provide either `client` (used for both roles) or both "
                "`detector_client` and `judge_client`."
            )
        self.detector_model = detector_model
        self.judge_model = judge_model
        self.max_rounds = max_rounds
        self.score_threshold = score_threshold
        self.detector_template = detector_template
        self.judge_template = judge_template
        self.detector_max_tokens = detector_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.api_retries = api_retries
        self.detector_temperature = detector_temperature
        self.detector_top_p = detector_top_p
        self.judge_temperature = judge_temperature
        self.preprocessing_config = preprocessing_config or {}
        self.judge_enable_thinking = judge_enable_thinking
        self.feedback_image_mode = feedback_image_mode
        self.external_api = external_api
        self.sampling_params_supported = sampling_params_supported
        self.graph = build_detection_graph().compile()

        if self.external_api:
            self._warn_ignored_local_only_settings()

    def _warn_ignored_local_only_settings(self) -> None:
        """Log once, at construction time, if local-backend-only settings will be silently ignored."""
        ignored = []
        if self.preprocessing_config.get("send_pixel_bounds"):
            ignored.append(
                "preprocessing_config['send_pixel_bounds'] (min_pixels/max_pixels)"
            )
        if self.judge_enable_thinking:
            ignored.append("judge_enable_thinking")
        if ignored:
            logger.warning(
                "external_api=True: the following local-backend-only settings will be "
                "ignored since the official OpenAI API rejects unrecognized request "
                "fields: %s",
                ", ".join(ignored),
            )

    def _pixel_bounds_extra_args(self) -> dict:
        """Build extra_body kwargs for Qwen-VL/vLLM-style min_pixels/max_pixels hints."""
        if self.external_api:
            return {}
        if not self.preprocessing_config.get("send_pixel_bounds"):
            return {}
        extra_body = {}
        if self.preprocessing_config.get("min_pixels") is not None:
            extra_body["min_pixels"] = int(self.preprocessing_config["min_pixels"])
        if self.preprocessing_config.get("max_pixels") is not None:
            extra_body["max_pixels"] = int(self.preprocessing_config["max_pixels"])
        return {"extra_body": extra_body} if extra_body else {}

    def get_detector_prompt(
        self,
        categories: List[str],
        category_definitions: str,
        feedback: Optional[str] = None,
        actions: Optional[str] = None,
        previous_detections: Optional[str] = None,
        som_proposals: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return render_detector_prompt(
            template=self.detector_template,
            categories=categories,
            category_definitions=category_definitions,
            feedback=feedback,
            actions=actions,
            som_proposals=som_proposals,
        )

    def get_judge_prompt(
        self,
        category_definitions: str,
        detections: List[Dict[str, Any]],
    ) -> str:
        return render_judge_prompt(
            template=self.judge_template,
            category_definitions=category_definitions,
            detections=detections,
        )

    def run_inference(
        self,
        image_uris: str | list[str],
        categories: List[str],
        category_definitions: str,
        feedback: Optional[str] = None,
        actions: Optional[str] = None,
        previous_detections: Optional[str] = None,
        som_proposals: Optional[List[Dict[str, Any]]] = None,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Run a single VLM inference call."""
        if custom_prompt is not None:
            prompt = custom_prompt
        else:
            prompt = self.get_detector_prompt(
                categories,
                category_definitions,
                feedback=feedback,
                actions=actions,
                previous_detections="",
                som_proposals=som_proposals,
            )

        extra_args = self._pixel_bounds_extra_args()

        content_list = [{"type": "text", "text": prompt}]
        if isinstance(image_uris, list):
            for i, uri in enumerate(image_uris):
                if len(image_uris) > 1:
                    lbl = (
                        "Original image with grid:"
                        if i == 0
                        else "Previous annotated image with grid (for visual feedback of last round):"
                    )
                    content_list.append({"type": "text", "text": lbl})
                content_list.append({"type": "image_url", "image_url": {"url": uri}})
        else:
            content_list.append({"type": "image_url", "image_url": {"url": image_uris}})

        def _do_call():
            kwargs = dict(
                model=self.detector_model,
                max_tokens=self.detector_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": content_list,
                    }
                ],
                **extra_args,
            )
            if self.sampling_params_supported:
                kwargs["temperature"] = self.detector_temperature
                kwargs["top_p"] = self.detector_top_p
            return self.detector_client.chat.completions.create(**kwargs)

        response = _call_with_retries(
            _do_call, retries=self.api_retries, what="Detector call"
        )
        return response.choices[0].message.content

    def verify_crop(self, crop_image: Image.Image, label: str) -> bool:
        """Verify if target label is present in cropped image."""
        crop_uri = pil_to_data_uri(crop_image)
        prompt = f"Analyze this image crop carefully. Is there a visible '{label}' present inside this crop? You must respond in exactly this format, with nothing else: <present>YES</present> or <present>NO</present>."

        extra_args = self._pixel_bounds_extra_args()

        def _do_call():
            kwargs = dict(
                model=self.detector_model,
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": crop_uri}},
                        ],
                    }
                ],
                **extra_args,
            )
            if self.sampling_params_supported:
                kwargs["temperature"] = 0.1
            return self.detector_client.chat.completions.create(**kwargs)

        try:
            response = _call_with_retries(
                _do_call, retries=self.api_retries, what="Crop verification call"
            )
            text = response.choices[0].message.content.strip()
            logger.info("Verification response for label '%s': %s", label, text)
            match = re.search(r"<present>\s*(YES|NO)\s*</present>", text, re.IGNORECASE)
            if match:
                return match.group(1).upper() == "YES"
            return "YES" in text.upper()
        except Exception as e:
            logger.warning(
                "Crop verification failed for label '%s', keeping detection: %s",
                label,
                e,
            )
            return True

    def judge_detections(
        self,
        original_grid_uri: str,
        annotated_grid_uri: str,
        detections: List[Dict[str, Any]],
        category_definitions: str,
    ):
        prompt = self.get_judge_prompt(category_definitions, detections)

        extra_args = {}
        if self.judge_enable_thinking and not self.external_api:
            extra_args["extra_body"] = {"enable_thinking": True}

        def _do_call():
            kwargs = dict(
                model=self.judge_model,
                max_tokens=self.judge_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Original image (grid, no boxes):",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": original_grid_uri},
                            },
                            {
                                "type": "text",
                                "text": "Annotated image (grid + detected boxes):",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": annotated_grid_uri},
                            },
                        ],
                    }
                ],
                **extra_args,
            )
            if self.sampling_params_supported:
                kwargs["temperature"] = self.judge_temperature
            return self.judge_client.chat.completions.create(**kwargs)

        response = _call_with_retries(
            _do_call, retries=self.api_retries, what="Judge call"
        )
        text = _strip_think_blocks(response.choices[0].message.content)

        score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
        feedback_match = re.search(r"<feedback>(.*?)</feedback>", text, re.DOTALL)
        actions_match = re.search(r"<actions>(.*?)</actions>", text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(10, score))
        feedback_text = (
            feedback_match.group(1).strip() if feedback_match else text.strip()
        )
        actions_text = actions_match.group(1).strip() if actions_match else ""

        if actions_text:
            logger.info("Judge structured actions:\n%s", actions_text)
        else:
            logger.info(
                "Judge produced no structured <actions> block; falling back to text-only feedback."
            )

        return score, feedback_text, actions_text

    def run(
        self,
        image_path: str,
        categories: list[str],
        category_definitions: str,
        show_plot: bool = True,
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        Runs the object detection pipeline with custom preprocessing, tiling, NMS, SoM,
        and Crop & Verify validation using a LangGraph workflow.
        """
        if not categories:
            raise ValueError("`categories` must be a non-empty list.")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        initial_state: DetectionState = {
            "pipeline": self,
            "image_path": str(path.resolve()),
            "categories": categories,
            "category_definitions": category_definitions,
            "output_dir": output_dir,
            "show_plot": show_plot,
            "progress_callback": progress_callback,
            "max_rounds": self.max_rounds,
            "score_threshold": self.score_threshold,
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["best"], final_state["history"]

    @staticmethod
    def _persist(
        output_dir: str,
        base_image: Image.Image,
        best: dict,
        history: list[RoundResult],
    ):
        persist_results(output_dir, base_image, best, history)


# Backward-compat alias
FabricDefectPipeline = ObjectDetectionPipeline
