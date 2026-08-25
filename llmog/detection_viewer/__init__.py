from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from gradio import processing_utils
from PIL import Image

_STATIC_DIR = Path(__file__).parent / "static"

_COLOR_PALETTE = [
    "#FF0000",
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#E91E63",
    "#8BC34A",
]


def _load_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


_WEBP_URL_CACHE: dict[int, str] = {}


def _save_image_to_cache(img: Image.Image, cache_dir: str) -> str:
    # Dedup: same PIL object (id) reused across explorer round switches → reuse URL
    # without re-encoding WebP.  Also handle identical bytes via hash for realtime frames.
    try:
        img_id = id(img)
        if img_id in _WEBP_URL_CACHE:
            return _WEBP_URL_CACHE[img_id]
        # For realtime numpy frames that are new objects but identical bytes,
        # hash a thumbnail to avoid O(W*H) hash on 4K frames.
        thumb = img.copy()
        thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
        h = hash(thumb.tobytes())
        h_key = hash((h, img.size))
        if h_key in _WEBP_URL_CACHE:
            return _WEBP_URL_CACHE[h_key]
    except Exception:
        h_key = None
        img_id = None
    cached_path = processing_utils.save_pil_to_cache(img, cache_dir, format="webp")
    url = f"/gradio_api/file={cached_path}"
    try:
        if img_id is not None:
            _WEBP_URL_CACHE[img_id] = url
            if len(_WEBP_URL_CACHE) > 128:
                _WEBP_URL_CACHE.pop(next(iter(_WEBP_URL_CACHE)))
        if h_key is not None:
            _WEBP_URL_CACHE[h_key] = url
    except Exception:
        pass
    return url


class DetectionViewer(gr.HTML):
    def __init__(
        self,
        value: tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]]]
        | tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]], dict[str, Any]]
        | None = None,
        *,
        label: str | None = None,
        panel_title: str = "Detections",
        list_height: int = 300,
        score_threshold: tuple[float, float] = (0.0, 1.0),
        keypoint_threshold: float = 0.0,
        keypoint_radius: int = 3,
        **kwargs: object,
    ) -> None:
        html_template = (_STATIC_DIR / "template.html").read_text(encoding="utf-8")
        css_template = (_STATIC_DIR / "style.css").read_text(encoding="utf-8")
        js_on_load = (_STATIC_DIR / "script.js").read_text(encoding="utf-8")
        # Pre-interpolate panel_title / list_height to avoid ReferenceError:
        # Gradio's HTML templating expects JS `${props.x}` or Handlebars `{{x}}`,
        # but the static file ships `${panel_title}` which evaluates as bare JS var.
        # Doing Python replacement makes it work on all Gradio versions.
        try:
            html_template = html_template.replace("${panel_title}", str(panel_title))
        except Exception:
            pass
        try:
            css_template = css_template.replace("${list_height}", str(list_height))
        except Exception:
            pass

        has_label = label is not None
        # Gradio 5+ supports html_template/css_template/js_on_load on gr.HTML,
        # Gradio 4 does not — fall back to placeholder so build_app still works.
        try:
            super().__init__(
                value=value,
                label=label,
                show_label=has_label,
                container=has_label,
                html_template=html_template,
                css_template=css_template,
                js_on_load=js_on_load,
                panel_title=panel_title,
                list_height=list_height,
                score_threshold_min=score_threshold[0],
                score_threshold_max=score_threshold[1],
                keypoint_threshold=keypoint_threshold,
                keypoint_radius=keypoint_radius,
                **kwargs,
            )
        except TypeError:
            # Fallback for Gradio 4: Bauhaus soft placeholder
            fallback_kwargs = {k: v for k, v in kwargs.items() if k in ("elem_id", "elem_classes", "visible", "render", "key")}
            placeholder = (
                f"<div style='border:1px solid #E9E0CC; background:#FFD6E0; color:#1A1145; padding:14px; "
                f"border-radius:16px; font-family:Inter, sans-serif; font-weight:700;'>"
                f"{panel_title} — viewer requires Gradio 5+ (current {gr.__version__})</div>"
            )
            super().__init__(
                value=placeholder,
                label=label,
                show_label=has_label,
                container=has_label,
                **fallback_kwargs,
            )

    def postprocess(self, value: Any) -> str | None:  # noqa: ANN401
        if isinstance(value, str):
            return value
        return self._process(value)

    def _process(
        self,
        value: tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]]]
        | tuple[str | Path | Image.Image | np.ndarray, list[dict[str, Any]], dict[str, Any]]
        | None,
    ) -> str | None:
        if value is None:
            return None

        if len(value) == 3:  # noqa: PLR2004 - tuple length check
            image_src, annotations, config = value
        else:
            image_src, annotations = value
            config = {}

        img = _load_image(image_src)
        image_url = _save_image_to_cache(img, self.GRADIO_CACHE)

        processed: list[dict[str, Any]] = []
        for i, ann in enumerate(annotations):
            has_kps = "keypoints" in ann and len(ann["keypoints"]) > 0
            default_label = f"Person {i + 1}" if has_kps else f"Detection {i + 1}"

            entry: dict[str, Any] = {
                "keypoints": ann.get("keypoints", []),
                "connections": ann.get("connections", []),
                "color": ann.get("color", _COLOR_PALETTE[i % len(_COLOR_PALETTE)]),
                "label": ann.get("label", default_label),
            }
            if "bbox" in ann:
                entry["bbox"] = ann["bbox"]
            if "score" in ann:
                entry["score"] = ann["score"]
            if "mask" in ann:
                entry["mask"] = ann["mask"]
            processed.append(entry)

        result: dict[str, Any] = {"image": image_url, "annotations": processed}
        if "score_threshold" in config:
            result["scoreThresholdMin"] = config["score_threshold"][0]
            result["scoreThresholdMax"] = config["score_threshold"][1]

        return json.dumps(result)

    def process_example(self, value: Any) -> str | None:  # noqa: ANN401
        if value is None:
            return None
        image_source = value[0] if isinstance(value, tuple) else value
        img = _load_image(image_source)
        url = _save_image_to_cache(img, self.GRADIO_CACHE)
        return (
            f'<img src="{url}" alt="example" '
            f'style="max-width:100%;max-height:5rem;object-fit:contain;'
            f'display:block;border-radius:4px;">'
        )

    def api_info(self) -> dict[str, Any]:
        return {
            "type": "string",
            "description": (
                "JSON string containing detection visualization data. "
                "Structure: {image: string (URL), annotations: ["
                "{color: string, label: string, "
                "bbox?: {x: float, y: float, width: float, height: float}, "
                "score?: float, "
                "mask?: {counts: [int], size: [int, int]}, "
                "keypoints?: [{x: float, y: float, name: string, confidence?: float}], "
                "connections?: [[int, int]]}], "
                "scoreThresholdMin?: float, scoreThresholdMax?: float}"
            ),
        }
