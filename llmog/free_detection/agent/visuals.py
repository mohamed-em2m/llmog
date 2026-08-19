"""Image rendering, grid drawing, and visual overlay utilities."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List
from PIL import Image, ImageDraw, ImageFont

from free_detection.image_preprocessing import draw_premium_grid


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try a few common truetype fonts, fall back to PIL's default bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _text_with_backing(
    draw: ImageDraw.ImageDraw, xy, text, font, fill, backing="black", pad=2
):
    """Draw text with a solid backing rectangle so it stays legible over photos."""
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=backing,
    )
    draw.text((x, y), text, fill=fill, font=font)


def draw_grid(
    image: Image.Image,
    step: int = 100,
    style: str = "standard",
    line_color: str = "red",
    line_width: int = 1,
    font_size: int = 0,
    text_color: str = "white",
    backing_color: str = "black",
) -> Image.Image:
    """Overlay a 0-1000 scale coordinate grid with readable axis labels and custom colors/sizes."""
    return draw_premium_grid(
        image,
        style=style,
        step=step,
        line_color=line_color,
        line_width=line_width,
        font_size=font_size,
        text_color=text_color,
        backing_color=backing_color,
    )


def render_detections(base_image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Draw lime bounding boxes and indexed labels onto the base image."""
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(max(12, min(w, h) // 50))

    for idx, item in enumerate(detections, 1):
        bbox = item.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        left = xmin * w / 1000
        top = ymin * h / 1000
        right = xmax * w / 1000
        bottom = ymax * h / 1000
        draw.rectangle([left, top, right, bottom], outline="lime", width=4)
        label_y = max(0, top - 18)
        label_text = f"#{idx}: {item.get('label', 'object')}"
        _text_with_backing(draw, (left + 2, label_y), label_text, font, fill="lime")
    return img


def pil_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode a PIL image as a base64 data URI.
    Converts alpha/palette/CMYK to RGB if saving as JPEG.
    """
    fmt_norm = "JPEG" if fmt.upper() in ("JPEG", "JPG") else fmt.upper()
    save_img = img
    if fmt_norm == "JPEG" and img.mode != "RGB":
        save_img = img.convert("RGB")

    buffer = io.BytesIO()
    save_img.save(buffer, format=fmt_norm)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt_norm.lower()};base64,{encoded}"
