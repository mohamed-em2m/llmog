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
from pathlib import Path

logger = logging.getLogger("detection_pipeline.reclassification")

_RECLS_PALETTE = [
    (255, 60, 60),  # vibrant red
    (0, 150, 255),  # bright blue
    (0, 210, 80),  # vivid green
    (255, 160, 20),  # orange
    (220, 60, 255),  # magenta
    (0, 215, 215),  # cyan
    (150, 60, 255),  # purple
    (255, 210, 20),  # yellow-gold
    (255, 105, 180),  # hot pink
    (0, 255, 150),  # mint
]

_RECLS_EMPTY_TABLE = (
    '<div class="output-panel" style="margin-top:0.75rem">'
    '<div class="out-header"><div class="out-header-left">'
    '<span class="out-header-dot"></span>'
    '<span class="out-header-title">Recognition Results</span>'
    "</div></div>"
    '<div style="color:#7d8590;text-align:center;padding:1rem;">No results yet.</div></div>'
)

# Preset category libraries for common inspection and detection domains
CATEGORY_PRESETS: Dict[str, Dict[str, str]] = {
    "Fabric & Surface Defects": {
        "classes": "hole, stain, tear, cut, knot, weaving_defect",
        "defs": (
            "- hole: missing fabric or puncture\n"
            "- stain: discoloration or surface contaminant\n"
            "- tear: frayed, uneven physical separation\n"
            "- cut: clean sharp slice or incision\n"
            "- knot: raised thread lump or snarl\n"
            "- weaving_defect: uneven thread density or missing yarn"
        ),
    },
    "General Objects (COCO)": {
        "classes": "person, car, bicycle, dog, cat, chair, bottle, laptop, cell_phone, book",
        "defs": (
            "- person: human body\n"
            "- car: passenger automobile\n"
            "- bicycle: two-wheeled pedal bike\n"
            "- dog: canine domestic animal\n"
            "- cat: feline domestic animal\n"
            "- chair: seating furniture\n"
            "- bottle: liquid beverage container\n"
            "- laptop: portable notebook computer\n"
            "- cell_phone: handheld smartphone\n"
            "- book: bound printed volume"
        ),
    },
    "Road & Traffic": {
        "classes": "car, truck, pedestrian, cyclist, traffic_light, traffic_sign, bus, motorcycle",
        "defs": (
            "- car: passenger sedan, coupe, or SUV\n"
            "- truck: heavy transport or cargo vehicle\n"
            "- pedestrian: person on foot\n"
            "- cyclist: person riding a bicycle\n"
            "- traffic_light: signal light lamp\n"
            "- traffic_sign: road regulatory or warning signboard\n"
            "- bus: public transit passenger bus\n"
            "- motorcycle: motorized two-wheeled vehicle"
        ),
    },
    "Retail & Packaging": {
        "classes": "box, barcode, product_label, bottle, can, pouch, blister_pack",
        "defs": (
            "- box: cardboard or corrugated carton\n"
            "- barcode: 1D or 2D scanner code\n"
            "- product_label: brand packaging label\n"
            "- bottle: glass or plastic container\n"
            "- can: aluminum or tin can\n"
            "- pouch: flexible plastic packaging\n"
            "- blister_pack: clear molded plastic bubble packaging"
        ),
    },
    "PCB & Electronics Defects": {
        "classes": "short_circuit, missing_component, solder_bridge, broken_trace, scratch, misalignment",
        "defs": (
            "- short_circuit: unintended electrical contact\n"
            "- missing_component: empty pad where SMD/component should be\n"
            "- solder_bridge: solder connecting adjacent pins\n"
            "- broken_trace: severed copper circuit trace\n"
            "- scratch: surface gouge across the solder mask\n"
            "- misalignment: component rotated or shifted off pad"
        ),
    },
    "Custom / Blank": {
        "classes": "",
        "defs": "",
    },
}


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
        # Ensure port is a plain int/str (guard against None slipping through)
        if local_port is None:
            raise ValueError(
                "Local server port is not set. Restart the server on the Server tab."
            )
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
    except Exception:
        # Fallback: estimate size via font metrics (Pillow >=10 removed textsize)
        try:
            fb = font.getbbox(label)
            text_w = fb[2] - fb[0]
            text_h = fb[3] - fb[1]
        except Exception:
            # Last-resort pixel estimate (~8px per char, 14px tall)
            text_w = max(8, len(label) * 8)
            text_h = 14

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


def _to_pil_rgb(img_obj: Any) -> Optional[Image.Image]:
    """Convert any image representation into a PIL RGB Image."""
    if img_obj is None:
        return None
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, np.ndarray):
        return Image.fromarray(img_obj).convert("RGB")
    if isinstance(img_obj, (str, Path)):
        s = str(img_obj)
        if s.startswith("data:image"):
            _, _, b64_data = s.partition(",")
            return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        if Path(s).is_file():
            return Image.open(s).convert("RGB")
    if isinstance(img_obj, dict):
        p_val = img_obj.get("path") or img_obj.get("name")
        if p_val and Path(p_val).is_file():
            return Image.open(p_val).convert("RGB")
    if hasattr(img_obj, "path") and img_obj.path and Path(img_obj.path).is_file():
        return Image.open(img_obj.path).convert("RGB")
    if hasattr(img_obj, "name") and img_obj.name and Path(img_obj.name).is_file():
        return Image.open(img_obj.name).convert("RGB")
    return None


def _to_pil_rgba(img_obj: Any) -> Optional[Image.Image]:
    """Convert any layer representation into a PIL RGBA Image."""
    if img_obj is None:
        return None
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGBA")
    if isinstance(img_obj, np.ndarray):
        return Image.fromarray(img_obj).convert("RGBA")
    if isinstance(img_obj, (str, Path)):
        s = str(img_obj)
        if s.startswith("data:image"):
            _, _, b64_data = s.partition(",")
            return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGBA")
        if Path(s).is_file():
            return Image.open(s).convert("RGBA")
    if isinstance(img_obj, dict):
        p_val = img_obj.get("path") or img_obj.get("name")
        if p_val and Path(p_val).is_file():
            return Image.open(p_val).convert("RGBA")
    if hasattr(img_obj, "path") and img_obj.path and Path(img_obj.path).is_file():
        return Image.open(img_obj.path).convert("RGBA")
    if hasattr(img_obj, "name") and img_obj.name and Path(img_obj.name).is_file():
        return Image.open(img_obj.name).convert("RGBA")
    return None


def extract_regions(
    editor_value: Any,
    min_area_ratio: float = 0.0005,
    max_regions: int = 50,
) -> List[Dict[str, int]]:
    """Turn the user's drawn strokes/shapes into a list of pixel-space bounding regions.

    Handles:
      - Explicit pixel-space bounding boxes from Custom Frontend Box / Region tools
      - Multi-layer stroke masks (with dimension scaling if resized by browser)
      - Direct alpha channel extraction
      - Composite vs Background pixel difference fallback
      - Stroke expansion / dilation to merge nearby disconnected strokes belonging to the same object
    """
    if isinstance(editor_value, str):
        try:
            editor_value = json_repair.loads(editor_value)
        except Exception:
            return []

    if not isinstance(editor_value, dict):
        return []

    background = editor_value.get("background")
    bg_pil = _to_pil_rgb(background)
    if bg_pil is None:
        return []

    target_w, target_h = bg_pil.size

    # 1. First priority: Check if explicit region boxes were sent by the custom frontend
    explicit_regions = editor_value.get("regions")
    if explicit_regions and isinstance(explicit_regions, list):
        parsed_regions: List[Dict[str, int]] = []
        for reg in explicit_regions:
            if not isinstance(reg, dict):
                continue
            x1 = int(reg.get("x1", reg.get("xmin", 0)))
            y1 = int(reg.get("y1", reg.get("ymin", 0)))
            x2 = int(reg.get("x2", reg.get("xmax", 0)))
            y2 = int(reg.get("y2", reg.get("ymax", 0)))

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            x1 = max(0, min(target_w - 1, x1))
            y1 = max(0, min(target_h - 1, y1))
            x2 = max(0, min(target_w, x2))
            y2 = max(0, min(target_h, y2))

            if x2 - x1 >= 4 and y2 - y1 >= 4:
                parsed_regions.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "area": (x2 - x1) * (y2 - y1),
                })
        if parsed_regions:
            parsed_regions.sort(key=lambda r: (r["y1"], r["x1"]))
            return parsed_regions[:max_regions]

    layers = editor_value.get("layers") or []
    composite = editor_value.get("composite")

    combined_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    found_layer_strokes = False

    for layer in layers:
        if layer is None:
            continue
        try:
            layer_rgba = _to_pil_rgba(layer)
            if layer_rgba is None:
                continue

            # Ensure layer is resized to background size in case Gradio canvas scaled it
            if layer_rgba.size != (target_w, target_h):
                layer_rgba = layer_rgba.resize(
                    (target_w, target_h), Image.Resampling.NEAREST
                )

            alpha = np.asarray(layer_rgba)[..., 3].astype(np.uint8)

            # Guard: Gradio initialises the default drawing layer as a fully white-opaque
            # RGBA canvas (alpha=255 everywhere). Treating that as "drawn strokes" would
            # mark the entire image as one giant region. Skip any layer where ≥90% of
            # pixels are opaque — that pattern only appears on a blank default canvas, not
            # on sparse user-drawn strokes.
            opaque_ratio = np.count_nonzero(alpha > 10) / float(alpha.size)
            if opaque_ratio >= 0.90:
                logger.debug(
                    "Skipping layer with %.0f%% opaque pixels (blank default canvas).",
                    opaque_ratio * 100,
                )
                continue

            if np.any(alpha > 10):
                # Layer has sparse transparent strokes — use alpha as the stroke mask
                combined_mask = np.maximum(
                    combined_mask, (alpha > 10).astype(np.uint8) * 255
                )
                found_layer_strokes = True
            else:
                # Alpha is all zero; check RGB brightness in case strokes are fully opaque RGB
                rgb = np.asarray(layer_rgba.convert("RGB")).astype(np.int16)
                rgb_mask = np.max(rgb, axis=2) > 30
                rgb_ratio = np.count_nonzero(rgb_mask) / float(rgb_mask.size)
                if rgb_ratio > 0.0 and rgb_ratio < 0.90:
                    combined_mask = np.maximum(
                        combined_mask, rgb_mask.astype(np.uint8) * 255
                    )
                    found_layer_strokes = True
        except Exception as e:
            logger.warning(f"Error parsing editor layer: {e}")
            continue

    # Fallback to composite vs background difference if layers were empty
    if not found_layer_strokes and composite is not None:
        try:
            comp_pil = _to_pil_rgb(composite)
            if comp_pil is not None:
                if comp_pil.size != (target_w, target_h):
                    comp_pil = comp_pil.resize(
                        (target_w, target_h), Image.Resampling.BILINEAR
                    )

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
            regions.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "area": int((x2 - x1) * (y2 - y1)),
                }
            )

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
    class_mode: str = "strict",
) -> Dict[str, Any]:
    """Ask the VLM to classify a cropped region under the selected class expectation mode."""
    buf = io.BytesIO()
    crop_pil.convert("RGB").save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    mode_norm = (class_mode or "strict").lower().strip()
    if "free" in mode_norm:
        prompt = (
            "You are an expert visual inspector. A user marked a region on an image "
            "and you must autonomously identify and name whatever object, feature, defect, "
            "or anomaly is visible inside it without any fixed class constraint.\n"
        )
        if class_definitions and class_definitions.strip():
            prompt += f"Optional context/domain guidance:\n{class_definitions}\n"
        prompt += (
            "Look carefully at the cropped region and provide a concise, accurate, lowercase "
            "descriptor (1-3 words, e.g. 'hole', 'stain', 'person', 'car', 'zipper', 'scratch') "
            "as the class name. If the region is completely blank, empty background, or unidentifiable noise, "
            "respond with class \"none\".\n"
            "Respond with ONLY valid JSON in exactly this format:\n"
            '{"class":"<descriptive_class_name>","confidence":<0-100>,"reasoning":"<short reason>"}. '
            "Do not include markdown code fences, comments, or extra text."
        )
    elif "hybrid" in mode_norm:
        prompt = (
            "You are an expert visual inspector. A user marked a region on an image "
            "and you must recognize what object or defect is inside it.\n"
            f"Priority target classes: {classes}.\n"
        )
        if class_definitions and class_definitions.strip():
            prompt += f"Category definitions:\n{class_definitions}\n"
        prompt += (
            "Detection instructions:\n"
            "1. First evaluate if the cropped region matches any of the priority target classes. "
            "If it matches, use that exact class name.\n"
            "2. If the object/defect is clearly a distinct real-world object or defect that does NOT "
            "fit any priority class, assign a NEW concise, specific lowercase category name (e.g. 'zipper', 'button', 'scratch', 'rust').\n"
            "3. If the region is purely empty background with no meaningful object, respond with class \"none\".\n"
            "Respond with ONLY valid JSON in exactly this format:\n"
            '{"class":"<class_name>","is_new_class":<true_or_false>,"confidence":<0-100>,"reasoning":"<short reason>"}. '
            "Do not include markdown code fences, comments, or extra text."
        )
    else:  # strict (default)
        prompt = (
            "You are an expert visual inspector. A user marked a region on an image "
            "and you must recognize what object or defect is inside it.\n"
            f"Available target classes: {classes}.\n"
        )
        if class_definitions and class_definitions.strip():
            prompt += f"Class definitions:\n{class_definitions}\n"
        prompt += (
            "Look carefully at the cropped region and choose EXACTLY ONE class from "
            "the available target classes that best matches what is inside. "
            "You MUST select strictly from the provided list — do not invent new classes, synonyms, or variations. "
            'If none of the available classes clearly apply, respond with class "none".\n'
            "Respond with ONLY valid JSON in exactly this format:\n"
            '{"class":"<class_name>","confidence":<0-100>,"reasoning":"<short reason>"}. '
            "Do not include markdown code fences, comments, or extra text."
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


def render_recls_table(rows: List[str], class_mode: str = "strict") -> str:
    """Render the HTML table showing region classifications with mode badges."""
    body = (
        "".join(rows)
        if rows
        else (
            '<tr><td colspan="5" style="color:#7d8590;text-align:center;padding:1rem;">'
            "No regions detected.</td></tr>"
        )
    )
    mode_label = class_mode.capitalize()
    return (
        '<div class="output-panel" style="margin-top:0.75rem">'
        '<div class="out-header"><div class="out-header-left">'
        '<span class="out-header-dot"></span>'
        f'<span class="out-header-title">Recognition Results ({mode_label} Mode)</span>'
        "</div></div>"
        '<div style="max-height:320px; overflow-y:auto;">'
        '<table class="batch-status-table"><thead><tr>'
        "<th>Region</th><th>Class</th><th>Origin</th><th>Confidence</th><th>Reasoning</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></div></div>"
    )


def classify_regions_gui(
    editor_value: Any,
    classes_str: str,
    class_definitions: str,
    pad_pct: float | int,
    class_mode: str = "strict",
    conf_threshold: float | int = 20.0,
    use_external_api: bool = False,
    ext_api_url: str = "",
    ext_api_key: str = "",
    ext_model_name: str = "",
    local_server_port: int | str | None = None,
) -> Tuple[str, Optional[Image.Image], str, str]:
    """Interactive Draw-and-Recognize handler for Gradio UI supporting Strict, Hybrid, and Free modes.

    Returns:
        (status_str, annotated_pil_or_None, results_html, yolo_txt)
    """
    mode_norm = (class_mode or "strict").lower().strip()
    classes = [c.strip().lower() for c in (classes_str or "").split(",") if c.strip()]
    
    if not classes and "free" not in mode_norm:
        return (
            f"Error: provide at least one target class for {class_mode.capitalize()} mode (or switch to Free Classify).",
            None,
            _RECLS_EMPTY_TABLE,
            "",
        )

    if isinstance(editor_value, str):
        try:
            editor_value = json_repair.loads(editor_value)
        except Exception:
            editor_value = {}

    if not editor_value or not isinstance(editor_value, dict) or not editor_value.get("background"):
        return (
            "Error: upload an image and draw boxes or strokes over each object first.",
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

    background_raw = editor_value.get("background")
    background = _to_pil_rgb(background_raw)
    if background is None:
        return (
            "Error: could not load background image from editor.",
            None,
            _RECLS_EMPTY_TABLE,
            "",
        )

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

    font_size = max(14, min(w_img, h_img) // 40)
    font = _load_font(font_size)

    class_ids = {c: i for i, c in enumerate(classes)}
    discovered_new_classes = []

    rows = []
    yolo_lines = []
    conf_min = float(conf_threshold or 0)

    for idx, reg in enumerate(regions):
        crop = crop_with_padding(background, reg, pad_pct)
        try:
            result = classify_region(
                crop, client, model_name, classes, class_definitions, class_mode
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

        is_new = bool(result.get("is_new_class")) or (cls not in classes and cls not in ("none", "error"))

        if is_new and cls not in ("none", "error"):
            if cls not in class_ids:
                class_ids[cls] = len(class_ids)
                if cls not in discovered_new_classes:
                    discovered_new_classes.append(cls)

        color = _RECLS_PALETTE[idx % len(_RECLS_PALETTE)]
        
        # Tag formatting
        if confidence < conf_min:
            label_text = f"#{idx + 1} {cls} (LOW {confidence:.0f}%)"
            origin_badge = '<span class="status-badge badge-stopped">Low Conf</span>'
        elif is_new:
            label_text = f"#{idx + 1} ✨{cls} ({confidence:.0f}%)"
            origin_badge = '<span class="status-badge" style="background:#818cf8;color:#000;font-weight:700">NEW</span>'
        else:
            label_text = f"#{idx + 1} {cls} ({confidence:.0f}%)"
            origin_badge = '<span class="status-badge badge-running">Target</span>'

        # Draw bounding box
        draw_recls_bbox(
            draw,
            (reg["x1"], reg["y1"], reg["x2"], reg["y2"]),
            label_text,
            color,
            font,
            (w_img, h_img),
        )

        # Generate YOLO line if confidence meets threshold and not none/error
        if cls in class_ids and cls != "none" and confidence >= conf_min:
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
                    f"{class_ids[cls]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}  # {cls}"
                )

        conf_txt = f"{confidence:.0f}%" if cls != "error" else "—"
        cls_disp = html.escape(cls) if cls != "error" else "⚠ error"
        reason_esc = html.escape(reasoning[:160])
        rows.append(
            f"<tr><td>#{idx + 1}</td><td><b>{cls_disp}</b></td>"
            f"<td>{origin_badge}</td><td>{conf_txt}</td><td>{reason_esc}</td></tr>"
        )

    # Build YOLO output header with class mapping
    yolo_header_lines = [f"# Class Index Mapping:"]
    for c_name, c_idx in class_ids.items():
        tag = " (new)" if c_name in discovered_new_classes else ""
        yolo_header_lines.append(f"# {c_idx}: {c_name}{tag}")
    yolo_header_lines.append("")
    full_yolo_text = "\n".join(yolo_header_lines + yolo_lines) if yolo_lines else "\n".join(yolo_header_lines)

    new_info = f" (discovered {len(discovered_new_classes)} new class(es): {', '.join(discovered_new_classes)})" if discovered_new_classes else ""
    status = f"Mode: {class_mode.capitalize()} | Recognized {len(regions)} region(s) -> {len(yolo_lines)} YOLO label(s){new_info}"
    return status, annotated, render_recls_table(rows, class_mode), full_yolo_text
