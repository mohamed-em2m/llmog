"""
Utility adapters to convert legacy detection outputs into DetectionViewer annotations.

Centralises bbox conversions and palette selection so every tab shares the same
fast, allocation-light path and no tab re-implements box drawing in Python.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from PIL import Image

# Mirror palette defined in detection_viewer/__init__.py
_VIEWER_PALETTE = [
    "#FF0000",
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#E91E63",
    "#8BC34A",
]


def _color(idx: int) -> str:
    return _VIEWER_PALETTE[idx % len(_VIEWER_PALETTE)]


def pipeline_detections_to_annotations(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Convert pipeline `bbox_2d` detections (0-1000 normalized, [x1,y1,x2,y2])
    into DetectionViewer annotations with pixel bbox.
    Optimised: local var binding, single pass, no per-item copy of image.
    """
    if not detections:
        return []
    w, h = image_size
    if w <= 0 or h <= 0:
        return []
    out: List[Dict[str, Any]] = []
    # local bind for speed
    palette = _VIEWER_PALETTE
    plen = len(palette)
    for i, det in enumerate(detections):
        bbox = det.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except Exception:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # Clamp to 0-1000 then scale
        # (pipeline guarantees 0-1000 but we clamp defensively)
        x1 = max(0.0, min(1000.0, x1))
        y1 = max(0.0, min(1000.0, y1))
        x2 = max(0.0, min(1000.0, x2))
        y2 = max(0.0, min(1000.0, y2))
        x = x1 * w / 1000.0
        y = y1 * h / 1000.0
        bw = (x2 - x1) * w / 1000.0
        bh = (y2 - y1) * h / 1000.0
        if bw <= 0 or bh <= 0:
            continue
        ann: Dict[str, Any] = {
            "bbox": {"x": x, "y": y, "width": bw, "height": bh},
            "label": str(det.get("label") or f"Detection {i + 1}"),
            "color": det.get("color") or palette[i % plen],
        }
        # score may be 0-10 (judge) or 0-1 or 0-100; normalise to 0-1 for viewer
        sc = det.get("score")
        if sc is not None:
            try:
                sc_f = float(sc)
                if sc_f > 1.0:
                    # handle 0-10 or 0-100
                    if sc_f <= 10:
                        sc_f = sc_f / 10.0
                    elif sc_f <= 100:
                        sc_f = sc_f / 100.0
                ann["score"] = max(0.0, min(1.0, sc_f))
            except Exception:
                pass
        out.append(ann)
    return out


def region_results_to_annotations(
    regions: List[Dict[str, int]],
    labels: List[str],
    confidences: List[float] | None = None,
    base_palette: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Convert Draw tab regions (pixel x1,y1,x2,y2) + labels into viewer annotations.
    """
    if not regions:
        return []
    palette = base_palette or _VIEWER_PALETTE
    plen = len(palette)
    out: List[Dict[str, Any]] = []
    for i, reg in enumerate(regions):
        try:
            x1 = int(reg["x1"])
            y1 = int(reg["y1"])
            x2 = int(reg["x2"])
            y2 = int(reg["y2"])
        except Exception:
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        label = labels[i] if i < len(labels) else f"Region {i + 1}"
        ann: Dict[str, Any] = {
            "bbox": {
                "x": float(x1),
                "y": float(y1),
                "width": float(w),
                "height": float(h),
            },
            "label": str(label),
            "color": palette[i % plen],
        }
        if confidences and i < len(confidences):
            try:
                sc = float(confidences[i])
                if sc > 1:
                    sc = sc / 100.0
                ann["score"] = max(0.0, min(1.0, sc))
            except Exception:
                pass
        out.append(ann)
    return out


def realtime_boxes_to_annotations(
    boxes: List[Any],
    palette: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Convert realtime tracker boxes [ymin,xmin,ymax,xmax,label,trackId?] to viewer annotations.
    """
    if not boxes:
        return []
    pal = palette or _VIEWER_PALETTE
    plen = len(pal)
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(boxes):
        if not b or len(b) < 4:
            continue
        try:
            ymin, xmin, ymax, xmax = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception:
            continue
        x = float(xmin)
        y = float(ymin)
        w = float(xmax - xmin)
        h = float(ymax - ymin)
        if w <= 0 or h <= 0:
            continue
        label = str(b[4]) if len(b) >= 5 and b[4] else f"obj {i + 1}"
        if len(b) >= 6 and b[5] is not None:
            label = f"{label} #{b[5]}"
        ann: Dict[str, Any] = {
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "label": label,
            "color": pal[i % plen],
        }
        out.append(ann)
    return out


def build_viewer_payload(
    image: Image.Image | Any,
    annotations: List[Dict[str, Any]],
) -> Tuple[Any, List[Dict[str, Any]]] | None:
    """
    Build the tuple expected by DetectionViewer.postprocess: (image, annotations).
    Returns None if image is missing so viewer shows placeholder.
    """
    if image is None:
        return None
    # Ensure annotations is list (not None)
    return (image, annotations or [])


def detections_to_viewer_payload(
    image: Image.Image,
    detections: List[Dict[str, Any]],
) -> Tuple[Any, List[Dict[str, Any]]] | None:
    """One-shot helper: pipeline detections + image -> viewer payload."""
    if image is None:
        return None
    anns = pipeline_detections_to_annotations(detections, image.size)
    return (image, anns)


# ── Shared preprocessing config builder (deduplicates runner + realtime) ──


def build_prep_config(
    prep_enabled: bool,
    prep_short_edge: float | int | None = 1024,
    prep_pad_square: bool = False,
    prep_contrast_method: str = "none",
    prep_gamma: float | int | None = 1.0,
    prep_denoise_method: str = "none",
    prep_sharpen: bool = False,
    prep_white_balance: bool = False,
    prep_grid_style: str = "standard",
    prep_som_enabled: bool = False,
    prep_tiling_enabled: bool = False,
    prep_tile_size: float | int | None = 512,
    prep_tile_overlap: float | int | None = 20,
    prep_crop_verify_enabled: bool = False,
    prep_crop_padding: float | int | None = 15,
    prep_grid_step: float | int | None = 250,
    prep_grid_line_width: float | int | None = 1,
    prep_grid_font_size: float | int | None = 0,
    prep_grid_line_color: str = "red",
    prep_grid_line_color_custom: str = "red",
    prep_grid_text_color: str = "white",
    prep_grid_text_color_custom: str = "white",
    prep_grid_backing_color: str = "black",
    prep_grid_backing_color_custom: str = "black",
    prep_send_pixel_bounds: bool = False,
    prep_min_pixels: float | int | None = 200704,
    prep_max_pixels: float | int | None = 4194304,
    prep_custom_resize_enabled: bool = False,
    prep_custom_resize_width: float | int | None = 1024,
    prep_custom_resize_height: float | int | None = 1024,
) -> dict:
    """Build unified pipeline preprocessing_config from raw Gradio inputs.

    Single source of truth for Batch + Realtime tabs – keeps tiling, SoM,
    grid, and contrast settings in sync.  Fast path for disabled case.
    """
    if not prep_enabled:
        return {
            "resolution_enabled": False,
            "contrast_method": "none",
            "denoise_method": "none",
            "som_enabled": False,
            "tiling_enabled": False,
            "crop_verify_enabled": False,
            "grid_style": "standard",
            "grid_step": 100,
            "grid_line_width": 1,
            "grid_font_size": 0,
            "grid_line_color": "red",
            "grid_text_color": "white",
            "grid_backing_color": "black",
            "send_pixel_bounds": False,
            "min_pixels": 200704,
            "max_pixels": 4194304,
            "custom_resize": False,
            "custom_resize_width": 1024,
            "custom_resize_height": 1024,
        }
    use_custom_resize = bool(prep_custom_resize_enabled)
    return {
        "resolution_enabled": not use_custom_resize,
        "target_short_edge": int(prep_short_edge or 1024),
        "pad_to_square": bool(prep_pad_square),
        "contrast_method": prep_contrast_method or "none",
        "clip_limit": 2.0,
        "gamma": float(prep_gamma or 1.0),
        "denoise_method": prep_denoise_method or "none",
        "sharpen": bool(prep_sharpen),
        "white_balance": bool(prep_white_balance),
        "grid_style": (
            "standard"
            if prep_grid_style == "Standard Red"
            else (prep_grid_style or "standard")
        ),
        "som_enabled": bool(prep_som_enabled),
        "tiling_enabled": bool(prep_tiling_enabled),
        "tile_size": int(prep_tile_size or 512),
        "tile_overlap": float(prep_tile_overlap or 20) / 100.0,
        "crop_verify_enabled": bool(prep_crop_verify_enabled),
        "crop_padding": float(prep_crop_padding or 15) / 100.0,
        "grid_step": int(prep_grid_step or 250),
        "grid_line_width": int(prep_grid_line_width or 1),
        "grid_font_size": int(prep_grid_font_size or 0),
        "grid_line_color": (
            prep_grid_line_color
            if prep_grid_line_color != "custom"
            else prep_grid_line_color_custom
        ),
        "grid_text_color": (
            prep_grid_text_color
            if prep_grid_text_color != "custom"
            else prep_grid_text_color_custom
        ),
        "grid_backing_color": (
            prep_grid_backing_color
            if prep_grid_backing_color != "custom"
            else prep_grid_backing_color_custom
        ),
        "send_pixel_bounds": bool(prep_send_pixel_bounds),
        "min_pixels": int(prep_min_pixels) if prep_min_pixels is not None else None,
        "max_pixels": int(prep_max_pixels) if prep_max_pixels is not None else None,
        "custom_resize": use_custom_resize,
        "custom_resize_width": int(prep_custom_resize_width or 1024),
        "custom_resize_height": int(prep_custom_resize_height or 1024),
    }
