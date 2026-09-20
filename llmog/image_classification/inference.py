"""Single-image VLM classification + response normalization."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List

import json_repair
from PIL import Image

logger = logging.getLogger("llmog.classify.inference")


def encode_image_to_data_uri(image_path: str | Path) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _coerce_confidence(raw: Any) -> float:
    try:
        conf = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # Accept 1-5 scale (auto_label) and normalize to 0-100
    if 1.0 <= conf <= 5.0:
        return round(conf / 5.0 * 100.0, 1)
    return max(0.0, min(100.0, conf))


def _clean_prediction(item: Any) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    cls = str(item.get("class") or item.get("label") or "").strip().lower()
    if not cls:
        return None
    return {
        "class": cls,
        "confidence": _coerce_confidence(item.get("confidence", 0)),
        "reasoning": str(item.get("reasoning") or "").strip()[:300],
        "is_new_class": bool(item.get("is_new_class", False)),
    }


def normalize_classify_response(
    parsed: Any,
    classification_mode: str,
    top_k: int = 3,
    multi_threshold: float = 50.0,
) -> List[Dict[str, Any]]:
    """Normalize any model JSON shape into an ordered prediction list."""
    shape = (classification_mode or "single").lower().strip()
    if (
        isinstance(parsed, dict)
        and "predictions" in parsed
        and isinstance(parsed["predictions"], list)
    ):
        items = parsed["predictions"]
        fallback_reason = str(parsed.get("reasoning") or "")
    elif isinstance(parsed, list):
        items = parsed
        fallback_reason = ""
    elif isinstance(parsed, dict) and "class" in parsed:
        items = [parsed]
        fallback_reason = ""
    else:
        return []

    preds: List[Dict[str, Any]] = []
    for item in items:
        clean = _clean_prediction(item)
        if clean is None:
            continue
        if not clean["reasoning"] and fallback_reason:
            clean["reasoning"] = fallback_reason[:300]
        preds.append(clean)

    preds.sort(key=lambda p: p["confidence"], reverse=True)
    if shape == "single":
        return preds[:1]
    if shape == "top_k":
        return preds[: max(1, int(top_k or 3))]
    if shape == "multi":
        kept = [p for p in preds if p["confidence"] >= float(multi_threshold)]
        return kept or preds[:1]
    return preds[:1]


def classify_single_image(
    image_path: str | Path,
    client: Any,
    model_name: str,
    prompt: str,
    classification_mode: str = "single",
    top_k: int = 3,
    multi_threshold: float = 50.0,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    retries: int = 3,
) -> List[Dict[str, Any]]:
    """Classify one full image; retries via 429-aware helper when available."""
    data_uri = encode_image_to_data_uri(image_path)

    def _call() -> Any:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        # max_tokens is supported by OpenAI + most compatibles; drop on error
        try:
            return client.chat.completions.create(max_tokens=max_tokens, **kwargs)
        except TypeError:
            return client.chat.completions.create(**kwargs)

    try:
        from free_detection.agent.client_utils import _call_with_retries

        response = _call_with_retries(
            _call, retries=max(1, int(retries or 3)), what="classify image"
        )
    except Exception:
        # Fallback when client_utils is unavailable (or raises ImportError)
        last_exc: Exception | None = None
        for _ in range(max(1, int(retries or 3))):
            try:
                response = _call()
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        else:
            raise last_exc or RuntimeError("classify call failed")

    if not response.choices or not response.choices[0].message:
        raise ValueError("No choices returned from the VLM API call.")
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Model returned an empty text response.")
    parsed = json_repair.loads(raw)
    preds = normalize_classify_response(
        parsed, classification_mode, top_k, multi_threshold
    )
    if not preds:
        raise ValueError(f"Could not parse predictions from response: {raw[:200]!r}")
    return preds
