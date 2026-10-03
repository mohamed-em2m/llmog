"""Resolution-independent projection math for ESR-upscaled detection.

The detector speaks 0-1000 normalized coordinates, which are identical for
an image and its ESR-upscaled twin (upscaling is a pure scale, aspect and
content preserved). These helpers cover the two places where pixels leak in:

* final rendering (annotated JPG, YOLO lines) must land on the TRUE ORIGINAL
  dims: ``x_orig_px = x_work_px * orig_w / work_w`` (single ratio per axis);
* pixel-space cosmetics (grid line width / font size, tile size) must keep
  their *relative* meaning on the bigger working image, or the red grid the
  VLM sees changes density and the "projection/measures" shift the user saw.
"""

from __future__ import annotations

import math


def norm_to_orig_pixels(
    bbox_0_1000: list[int] | tuple[int, ...], orig_w: int, orig_h: int
) -> list[int]:
    """Project a 0-1000 box onto original-image pixels (round outward, clamp).

    This is the "map the new prediction box onto the old image" step: the
    VLM predicts in normalized space on the high-res working image, and this
    lands those coordinates on the original dims for final outputs.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_0_1000)
    left = math.floor(min(x1, x2) * orig_w / 1000)
    top = math.floor(min(y1, y2) * orig_h / 1000)
    right = math.ceil(max(x1, x2) * orig_w / 1000)
    bottom = math.ceil(max(y1, y2) * orig_h / 1000)
    left = max(0, min(orig_w, left))
    top = max(0, min(orig_h, top))
    right = max(0, min(orig_w, right))
    bottom = max(0, min(orig_h, bottom))
    if right <= left:
        right = min(orig_w, left + 1)
    if bottom <= top:
        bottom = min(orig_h, top + 1)
    return [left, top, right, bottom]


def work_pixels_to_orig_pixels(
    bbox_work_px: list[float] | tuple[float, ...],
    work_w: int,
    work_h: int,
    orig_w: int,
    orig_h: int,
) -> list[int]:
    """Project a working-image pixel box onto original-image pixels.

    Same ratio rule per axis (``orig/work``), round outward + clamp. Used for
    tile-local or crop-local pixel boxes before they are normalized.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_work_px)
    left = math.floor(min(x1, x2) * orig_w / work_w)
    top = math.floor(min(y1, y2) * orig_h / work_h)
    right = math.ceil(max(x1, x2) * orig_w / work_w)
    bottom = math.ceil(max(y1, y2) * orig_h / work_h)
    left = max(0, min(orig_w, left))
    top = max(0, min(orig_h, top))
    right = max(0, min(orig_w, right))
    bottom = max(0, min(orig_h, bottom))
    if right <= left:
        right = min(orig_w, left + 1)
    if bottom <= top:
        bottom = min(orig_h, top + 1)
    return [left, top, right, bottom]


def esr_growth_factor(work_w: int, work_h: int, orig_w: int, orig_h: int) -> float:
    """Relative growth of the working image vs the original (>= ~1.0).

    ``1.0`` (or below, when ESR is off) means "no scaling needed" -- every
    scaler below is a no-op at factor <= 1 so behavior stays byte-identical
    with ESR disabled.
    """
    if orig_w <= 0 or orig_h <= 0:
        return 1.0
    return max(work_w / orig_w, work_h / orig_h)


def scaled_pixel_param(value: int, factor: float, minimum: int = 1) -> int:
    """Scale a pixel-space cosmetic (line width, tile size) by ``factor``.

    No-op at factor <= 1 (ESR off). Rounds to int and clamps to ``minimum``
    so a 1px grid line never vanishes.
    """
    if factor <= 1.0:
        return int(value)
    return max(int(minimum), int(round(value * factor)))


def fit_long_edge(w: int, h: int, target: int) -> tuple[int, int]:
    """Fit ``(w, h)`` into ``target`` long edge, aspect preserved (LANCZOS-side).

    ``target <= 0`` returns the size unchanged (means "keep native").
    """
    if target <= 0:
        return w, h
    long_edge = max(w, h)
    if long_edge == target or long_edge <= 0:
        return w, h
    scale = target / long_edge
    return max(1, round(w * scale)), max(1, round(h * scale))


def tile_geometry(
    h: int, w: int, tile: int, overlap: int
) -> tuple[list[tuple[int, int]], int, int]:
    """Tiling coordinates for an ``h`` x ``w`` image.

    Returns ``(coords, eff_th, eff_tw)`` where the effective tile dims are
    clamped to the image (``min(tile, h/w)``). Clamping matters: without it
    a narrow image (e.g. 600x100 with tile 512) yields short tiles that no
    longer match a square feather mask and crash blending. Tiles are always
    full ``eff_th`` x ``eff_tw`` (edge tiles shift, never shrink), so every
    stacked batch is shape-consistent.
    """
    tile = max(1, int(tile))
    overlap = max(0, int(overlap))
    th, tw = min(tile, max(1, h)), min(tile, max(1, w))
    step_y = max(1, th - overlap)
    step_x = max(1, tw - overlap)
    ys = list(range(0, h - th + 1, step_y))
    xs = list(range(0, w - tw + 1, step_x))
    if not ys or ys[-1] != h - th:
        ys.append(max(0, h - th))
    if not xs or xs[-1] != w - tw:
        xs.append(max(0, w - tw))
    coords = [(y, x) for y in ys for x in xs]
    # De-duplicate (tiny images can repeat the origin on both axes).
    seen: set[tuple[int, int]] = set()
    unique = [c for c in coords if not (c in seen or seen.add(c))]
    return unique, th, tw
