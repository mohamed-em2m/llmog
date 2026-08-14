"""
Interactive Reclassification & Region Recognition Engine.
Extracts user drawn strokes/shapes from Gradio ImageEditor, crops regions with context,
classifies them using the VLM, and draws clear, high-contrast bounding boxes with labels.
"""

from __future__ import annotations

import io
import html
import base64
import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import json_repair
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

from interface.state import state
from free_detection.detection_pipeline import _load_font

logger = logging.getLogger("detection_pipeline.reclassification")

_RECLS_PALETTE = [
    (255, 60, 60),    # vibrant red
    (0, 150, 255),    # bright blue
    (0, 210, 80),     # vivid green
    (255, 160, 20),   # orange
    (220, 60, 255),   # magenta
    (0, 215, 215),    # cyan
    (150, 60, 255),   # purple
    (255, 210, 20),   # yellow-gold
    (255, 105, 180),  # hot pink
    (0, 255, 150),    # mint
]

_RECLS_EMPTY_TABLE = (
    '<div class="output-panel" style="margin-top:0.75rem">'
    '<div class="out-header"><div class="out-header-left">'
    '<span class="out-header-dot"></span>'
    '<span class="out-header-title">Recognition Results</span>'
    "</div></div>"
    '<div style="color:#7d8590;text-align:center;padding:1rem;">No results yet.</div></div>'
)


def make_recls_client(
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    local_server_port: int | str | None,
) -> Tuple[OpenAI, str]:
    """Build an OpenAI-compatible client + model name for the region classifier.

    Raises ValueError with a user-facing message when no usable backend is up.
    """
    if use_external_api:
        if not ext_api_key or ext_api_key == "your-key":
            raise ValueError(
                "External API selected but no API key provided. "
                "Set one in the External API section."
            )
        api_url, api_key, model_name = ext_api_url, ext_api_key, ext_model_name
    else:
        with state.server_lock:
            if state.server_manager is None or not state.server_manager.is_healthy():
                raise ValueError(
                    "Local server not running. Start it on the Server tab "
                    "or enable External API."
                )
            local_port = state.server_manager.port
            model_name = state.server_manager.model
        api_url = f"http://localhost:{local_port}/v1"
        api_key = "not-needed"
    return OpenAI(base_url=api_url, api_key=api_key), model_name


def draw_recls_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int] | List[int],
    label: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    img_size: Tuple[int, int],
) -> None:
    """Draw a crisp, high-visibility bounding box with a contrasting outline and solid labeled badge.

    Ensures the label box is never clipped by top/left/right/bottom edges of the image.
    """
    w_img, h_img = img_size
    x1, y1, x2, y2 = bbox

    # Clamp coordinates to image boundaries
    x1 = max(0, min(w_img - 1, int(x1)))
    y1 = max(0, min(h_img - 1, int(y1)))
    x2 = max(0, min(w_img - 1, int(x2)))
    y2 = max(0, min(h_img - 1, int(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Adaptive line width based on image resolution
    line_w = max(2, min(w_img, h_img) // 300)

    # 1. Outer black shadow stroke for universal contrast
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 0, 0), width=line_w + 2)
    # 2. Main colored stroke
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_w)

    # Calculate text badge bounds
    try:
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)

    padding = max(3, line_w + 1)
    badge_w = text_w + 2 * padding
    badge_h = text_h + 2 * padding

    # Position label above the box if room, else inside the box
    text_y = y1 - badge_h
    if text_y < 0:
        text_y = y1 + line_w

    # Prevent badge from overflowing right boundary
    text_x = x1
    if text_x + badge_w > w_img:
        text_x = max(0, w_img - badge_w)

    # Draw solid badge background with black border
    badge_rect = [
        (text_x, text_y),
        (text_x + badge_w, text_y + badge_h),
    ]
    draw.rectangle(badge_rect, fill=color, outline=(0, 0, 0), width=1)

    # Draw text inside badge (white text with slight drop shadow for readability)
    draw.text(
        (text_x + padding, text_y + padding),
        label,
        fill=(255, 255, 255),
        font=font,
    )


def extract_regions(
    editor_value: Any,
    min_area_ratio: float = 0.0005,
    max_regions: int = 50,
) -> List[Dict[str, int]]:
    """Turn the user's drawn strokes/shapes from Gradio ImageEditor into a list of pixel-space bounding regions.

    Handles:
      - Multi-layer stroke masks (with dimension scaling if resized by browser)
      - Direct alpha channel extraction
      - Composite vs Background pixel difference fallback
      - Stroke expansion / dilation to merge nearby disconnected strokes belonging to the same object
    """
    if not isinstance(editor_value, dict):
        return []

    background = editor_value.get("background")
    if background is None:
        return []

    # Standardize background as PIL Image to get authoritative dimensions
    if isinstance(background, np.ndarray):
        bg_pil = Image.fromarray(background).convert("RGB")
    elif isinstance(background, str):
        bg_pil = Image.open(background).convert("RGB")
    elif isinstance(background, Image.Image):
        bg_pil = background.convert("RGB")
    else:
        return []

    target_w, target_h = bg_pil.size
    layers = editor_value.get("layers") or []
    composite = editor_value.get("composite")

    combined_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    found_layer_strokes = False

    for layer in layers:
        if layer is None:
            continue
        try:
            if isinstance(layer, np.ndarray):
                layer_pil = Image.fromarray(layer)
            elif isinstance(layer, str):
                layer_pil = Image.open(layer)
            else:
                layer_pil = layer

            # Ensure layer is resized to background size in case Gradio canvas scaled it
            if layer_pil.size != (target_w, target_h):
                layer_pil = layer_pil.resize((target_w, target_h), Image.Resampling.NEAREST)

            layer_rgba = layer_pil.convert("RGBA")
            alpha = np.asarray(layer_rgba)[..., 3].astype(np.uint8)

            # Some layers might have strokes painted directly on RGB with full opacity
            if np.any(alpha > 10):
                combined_mask = np.maximum(combined_mask, (alpha > 10).astype(np.uint8) * 255)
                found_layer_strokes = True
            else:
                # Check RGB difference from blank/transparent
                rgb = np.asarray(layer_rgba.convert("RGB")).astype(np.int16)
                if np.any(rgb > 20):
                    layer_mask = (np.max(rgb, axis=2) > 20).astype(np.uint8) * 255
                    combined_mask = np.maximum(combined_mask, layer_mask)
                    found_layer_strokes = True
        except Exception as e:
            logger.warning(f"Error parsing editor layer: {e}")
            continue

    # Fallback to composite vs background difference if layers were empty
    if not found_layer_strokes and composite is not None:
        try:
            if isinstance(composite, np.ndarray):
                comp_pil = Image.fromarray(composite).convert("RGB")
            elif isinstance(composite, str):
                comp_pil = Image.open(composite).convert("RGB")
            else:
                comp_pil = composite.convert("RGB")

            if comp_pil.size != (target_w, target_h):
                comp_pil = comp_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)

            comp_arr = np.asarray(comp_pil).astype(np.int16)
            bg_arr = np.asarray(bg_pil).astype(np.int16)
            diff = np.abs(comp_arr - bg_arr).sum(axis=2)
            diff_mask = (diff > 25).astype(np.uint8) * 255
            if np.any(diff_mask > 0):
                combined_mask = np.maximum(combined_mask, diff_mask)
        except Exception as e:
            logger.warning(f"Error computing composite difference: {e}")

    if not np.any(combined_mask > 0):
        return []

    # Morphological closing & dilation to ensure hollow drawn shapes (circles, rectangles, loops)
    # are completely closed and treated as single solid bounded regions.
    # Adaptive kernel size relative to image resolution
    kernel_size = max(9, min(target_w, target_h) // 60)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 1. Close gaps in loops or rectangles
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # 2. Dilate slightly
    dilated = cv2.dilate(closed_mask, kernel, iterations=1)

    # 3. Find external contours to directly identify any drawn closed loop or outline shape
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(16, int(target_w * target_h * min_area_ratio))
    regions: List[Dict[str, int]] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0 or (w * h) < min_area:
            continue

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(target_w, x + w), min(target_h, y + h)

        if x2 - x1 >= 4 and y2 - y1 >= 4:
            regions.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "area": int((x2 - x1) * (y2 - y1)),
            })

    # Sort regions reading top-to-bottom, left-to-right
    regions.sort(key=lambda r: (r["y1"], r["x1"]))
    return regions[:max_regions]


def crop_with_padding(
    background: Image.Image,
    region: Dict[str, int],
    pad_pct: float | int = 10,
) -> Image.Image:
    """Crop a region from the background image, adding percentage context padding."""
    w, h = background.size
    pad = int(max(w, h) * (float(pad_pct or 0) / 100.0))
    x1 = max(0, region["x1"] - pad)
    y1 = max(0, region["y1"] - pad)
    x2 = min(w, region["x2"] + pad)
    y2 = min(h, region["y2"] + pad)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    return background.crop((x1, y1, x2, y2))


def classify_region(
    crop_pil: Image.Image,
    client: OpenAI,
    model_name: str,
    classes: List[str],
    class_definitions: str = "",
) -> Dict[str, Any]:
    """Ask the VLM to pick exactly one class for a cropped region."""
    buf = io.BytesIO()
    crop_pil.convert("RGB").save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    prompt = (
        "You are an expert visual inspector. A user drew a region on an image "
        "and you must recognize what object or defect is inside it.\n"
        f"Available classes: {classes}.\n"
    )
    if class_definitions and class_definitions.strip():
        prompt += f"Class definitions:\n{class_definitions}\n"
    prompt += (
        "Look carefully at the cropped region and choose EXACTLY ONE class from "
        "the available classes that best matches what is inside. If none of the "
        'classes clearly apply, respond with class "none".\n'
        "Respond with ONLY valid JSON in exactly this format: "
        '{"class":"<class_name>","confidence":<0-100>,"reasoning":"<short reason>"}. '
        "Do not include explanations, markdown, extra text, comments, or additional fields."
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    if not response.choices or not response.choices[0].message:
        raise ValueError("No choices returned from the VLM API call.")
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Model returned an empty text content response.")
    output = json_repair.loads(raw)
    return output if isinstance(output, dict) else {}


def render_recls_table(rows: List[str]) -> str:
    """Render the HTML table showing region classifications."""
    body = (
        "".join(rows)
        if rows
        else (
            '<tr><td colspan="4" style="color:#7d8590;text-align:center;padding:1rem;">'
            "No regions detected.</td></tr>"
        )
    )
    return (
        '<div class="output-panel" style="margin-top:0.75rem">'
        '<div class="out-header"><div class="out-header-left">'
        '<span class="out-header-dot"></span>'
        '<span class="out-header-title">Recognition Results</span>'
        "</div></div>"
        '<div style="max-height:320px; overflow-y:auto;">'
        '<table class="batch-status-table"><thead><tr>'
        "<th>Region</th><th>Class</th><th>Confidence</th><th>Reasoning</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></div></div>"
    )


def classify_regions_gui(
    editor_value: Any,
    classes_str: str,
    class_definitions: str,
    pad_pct: float | int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    local_server_port: int | str | None,
) -> Tuple[str, Optional[Image.Image], str, str]:
    """Interactive Draw-and-Recognize handler for Gradio UI."""
    classes = [c.strip().lower() for c in (classes_str or "").split(",") if c.strip()]
    if not classes:
        return (
            "Error: provide at least one class.",
            None,
            _RECLS_EMPTY_TABLE,
            "",
        )
    if not editor_value or not editor_value.get("background"):
        return (
            "Error: upload an image and draw strokes over each object first.",
            None,
            _RECLS_EMPTY_TABLE,
            "",
        )

    try:
        client, model_name = make_recls_client(
            use_external_api,
            ext_api_url,
            ext_api_key,
            ext_model_name,
            local_server_port,
        )
    except ValueError as e:
        return f"Error: {e}", None, _RECLS_EMPTY_TABLE, ""

    background = editor_value["background"]
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background).convert("RGB")
    elif not isinstance(background, Image.Image):
        background = Image.open(background).convert("RGB")
    else:
        background = background.convert("RGB")

    regions = extract_regions(editor_value)
    if not regions:
        return (
            "No drawn regions detected. Draw strokes or boxes over objects, then run again.",
            background,
            _RECLS_EMPTY_TABLE,
            "",
        )

    annotated = background.copy()
    draw = ImageDraw.Draw(annotated)
    w_img, h_img = background.size

    # Auto-scale font size for crisp text
    font_size = max(14, min(w_img, h_img) // 40)
    font = _load_font(font_size)

    class_ids = {c: i for i, c in enumerate(classes)}

    rows = []
    yolo_lines = []
    for idx, reg in enumerate(regions):
        crop = crop_with_padding(background, reg, pad_pct)
        try:
            result = classify_region(
                crop, client, model_name, classes, class_definitions
            )
        except Exception as e:
            result = {"class": "error", "confidence": 0, "reasoning": f"API error: {e}"}

        cls = (str(result.get("class") or "").strip()).lower()
        confidence = result.get("confidence", 0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        reasoning = str(result.get("reasoning") or "").strip()

        color = _RECLS_PALETTE[idx % len(_RECLS_PALETTE)]
        label_text = f"#{idx + 1} {cls} ({confidence:.0f}%)"

        # Draw the bounding box using the enhanced helper
        draw_recls_bbox(
            draw,
            (reg["x1"], reg["y1"], reg["x2"], reg["y2"]),
            label_text,
            color,
            font,
            (w_img, h_img),
        )

        if cls in class_ids and cls != "none":
            w_px = reg["x2"] - reg["x1"]
            h_px = reg["y2"] - reg["y1"]
            if w_px > 0 and h_px > 0:
                xc = (reg["x1"] + reg["x2"]) / 2.0 / w_img
                yc = (reg["y1"] + reg["y2"]) / 2.0 / h_img
                nw = w_px / w_img
                nh = h_px / h_img
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                yolo_lines.append(
                    f"{class_ids[cls]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                )

        conf_txt = f"{confidence:.0f}%" if cls != "error" else "—"
        cls_disp = html.escape(cls) if cls != "error" else "⚠ error"
        reason_esc = html.escape(reasoning[:160])
        rows.append(
            f"<tr><td>#{idx + 1}</td><td>{cls_disp}</td>"
            f"<td>{conf_txt}</td><td>{reason_esc}</td></tr>"
        )

    status = f"Recognized {len(regions)} region(s) -> YOLO labels: {len(yolo_lines)}"
    return status, annotated, render_recls_table(rows), "\n".join(yolo_lines)
