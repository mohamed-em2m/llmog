"""Detector node executing VLM inference (tiled or full image) and NMS."""

from __future__ import annotations

import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from free_detection.image_preprocessing import (
    draw_premium_grid,
    generate_som_proposals,
    get_image_tiles,
    map_tile_detection_to_original,
    apply_nms,
)
from free_detection.agent.state import DetectionState
from free_detection.agent.visuals import render_detections, pil_to_data_uri
from free_detection.agent.parser import parse_detections, validate_detections

logger = logging.getLogger("detection_pipeline")


def _filter_and_translate_feedback_for_tile(
    feedback: Optional[str],
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    orig_w: int,
    orig_h: int,
) -> Optional[str]:
    """Filter judge feedback coordinates into the tile's local 0-1000 space."""
    if not feedback:
        return None
    lines = feedback.split("\n")
    new_lines = []
    for line in lines:
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", line)
        if not matches:
            new_lines.append(line)
            continue

        keep_line = False
        translated_line = line
        for x_str, y_str in matches:
            x_val = int(x_str)
            y_val = int(y_str)
            px = x_val * orig_w / 1000
            py = y_val * orig_h / 1000

            if tile_x <= px <= tile_x + tile_w and tile_y <= py <= tile_y + tile_h:
                keep_line = True
                tx = int(round((px - tile_x) * 1000 / tile_w))
                ty = int(round((py - tile_y) * 1000 / tile_h))
                translated_line = translated_line.replace(
                    f"({x_str},{y_str})", f"({tx},{ty})"
                )
                translated_line = translated_line.replace(
                    f"({x_str}, {y_str})", f"({tx},{ty})"
                )

        if keep_line:
            new_lines.append(translated_line)

    return "\n".join(new_lines) if new_lines else None


def _prepare_annotated_feedback_image(
    pipeline: Any,
    preprocessed_image: Image.Image,
    previous_detections_prep: Optional[List[Dict[str, Any]]],
    round_num: int,
    grid_kwargs: Dict[str, Any],
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Pre-generate full-size annotated image and URI for feedback modes if in round > 1."""
    if round_num <= 1 or previous_detections_prep is None:
        return None, None

    if pipeline.feedback_image_mode not in ("annotated", "both"):
        return None, None

    annotated_prep_image = render_detections(
        preprocessed_image, previous_detections_prep
    )
    annotated_prep_with_grid = draw_premium_grid(
        annotated_prep_image, **grid_kwargs
    )
    annotated_prep_uri = pil_to_data_uri(annotated_prep_with_grid)
    return annotated_prep_image, annotated_prep_uri


def _process_single_tile(
    tile_item: Tuple[int, Dict[str, Any]],
    pipeline: Any,
    categories: List[str],
    category_definitions: str,
    feedback: Optional[str],
    judge_actions: Optional[str],
    prep_w: int,
    prep_h: int,
    annotated_prep_image: Optional[Image.Image],
    round_num: int,
    grid_kwargs: Dict[str, Any],
) -> Tuple[int, str, List[Dict[str, Any]], Optional[str]]:
    """Process a single image tile with grid overlay, feedback crops, and VLM inference."""
    idx, tile = tile_item
    tile_feedback = _filter_and_translate_feedback_for_tile(
        feedback,
        tile_x=tile["tile_x"],
        tile_y=tile["tile_y"],
        tile_w=tile["tile_w"],
        tile_h=tile["tile_h"],
        orig_w=prep_w,
        orig_h=prep_h,
    )

    tile_img_with_grid = draw_premium_grid(tile["tile_image"], **grid_kwargs)
    tile_uri = pil_to_data_uri(tile_img_with_grid)

    detector_images = [tile_uri]
    if round_num > 1 and annotated_prep_image is not None:
        annotated_tile_crop = annotated_prep_image.crop(
            (
                tile["tile_x"],
                tile["tile_y"],
                tile["tile_x"] + tile["tile_w"],
                tile["tile_y"] + tile["tile_h"],
            )
        )
        annotated_tile_with_grid = draw_premium_grid(
            annotated_tile_crop, **grid_kwargs
        )
        annotated_tile_uri = pil_to_data_uri(annotated_tile_with_grid)

        if pipeline.feedback_image_mode == "annotated":
            detector_images = [annotated_tile_uri]
        elif pipeline.feedback_image_mode == "both":
            detector_images = [tile_uri, annotated_tile_uri]

    logger.info(
        "Running parallel detection on Tile %d (at x=%d, y=%d)...",
        idx,
        tile["tile_x"],
        tile["tile_y"],
    )
    try:
        tile_raw_text = pipeline.run_inference(
            image_uris=detector_images,
            categories=categories,
            category_definitions=category_definitions,
            feedback=tile_feedback,
            actions=judge_actions,
            som_proposals=None,
        )
        tile_dets = validate_detections(
            parse_detections(tile_raw_text), categories
        )

        mapped_dets = []
        for det in tile_dets:
            mapped = map_tile_detection_to_original(
                det["bbox_2d"],
                tile_x=tile["tile_x"],
                tile_y=tile["tile_y"],
                tile_w=tile["tile_w"],
                tile_h=tile["tile_h"],
                orig_w=prep_w,
                orig_h=prep_h,
            )
            det["bbox_2d"] = mapped
            mapped_dets.append(det)
        return (idx, tile_raw_text, mapped_dets, None)
    except Exception as exc:
        logger.error("Failed detection on tile %d: %s", idx, exc)
        return (idx, "", [], str(exc))


def _run_tiled_detection(
    pipeline: Any,
    preprocessed_image: Image.Image,
    prep_w: int,
    prep_h: int,
    categories: List[str],
    category_definitions: str,
    feedback: Optional[str],
    judge_actions: Optional[str],
    annotated_prep_image: Optional[Image.Image],
    round_num: int,
    tile_size: int,
    tile_overlap: float,
    grid_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Partition the image into tiles, run parallel detection, and merge results with NMS."""
    logger.info(
        "Tiling enabled: dividing image of size %dx%d into tiles of size %d",
        prep_w,
        prep_h,
        tile_size,
    )
    tiles = get_image_tiles(
        preprocessed_image, tile_size=tile_size, overlap_pct=tile_overlap
    )
    logger.info("Generated %d tiles", len(tiles))

    def process_item(item):
        return _process_single_tile(
            tile_item=item,
            pipeline=pipeline,
            categories=categories,
            category_definitions=category_definitions,
            feedback=feedback,
            judge_actions=judge_actions,
            prep_w=prep_w,
            prep_h=prep_h,
            annotated_prep_image=annotated_prep_image,
            round_num=round_num,
            grid_kwargs=grid_kwargs,
        )

    # Cap tiling concurrency for Gemini free tier (15 RPM) – 4 parallel tiles would burst quota
    is_gemini = "gemini" in (getattr(pipeline, "detector_model", "") or "").lower()
    max_tile_workers = 1 if is_gemini else min(4, len(tiles))
    with ThreadPoolExecutor(max_workers=max_tile_workers) as pool:
        tile_results = list(pool.map(process_item, enumerate(tiles, 1)))

    tile_results.sort(key=lambda x: x[0])
    all_tile_detections = []
    raw_outputs_collected = []
    parse_error = None

    for idx, tile_raw_text, mapped_dets, err in tile_results:
        if tile_raw_text:
            raw_outputs_collected.append(
                f"Tile {idx} (x={tiles[idx-1]['tile_x']}, y={tiles[idx-1]['tile_y']}):\n{tile_raw_text}"
            )
        if mapped_dets:
            all_tile_detections.extend(mapped_dets)
        if err:
            parse_error = (
                str(err)
                if not parse_error
                else parse_error + f"; Tile {idx}: {err}"
            )

    merged_detections = apply_nms(all_tile_detections, iou_threshold=0.5)
    merged_raw_text = "\n\n".join(raw_outputs_collected)
    return merged_detections, merged_raw_text, parse_error


def _run_full_image_detection(
    pipeline: Any,
    preprocessed_image: Image.Image,
    categories: List[str],
    category_definitions: str,
    feedback: Optional[str],
    judge_actions: Optional[str],
    annotated_prep_uri: Optional[str],
    round_num: int,
    som_enabled: bool,
    grid_kwargs: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Run full-frame detection inference with grid and optional Set-of-Mark proposals."""
    grid_img = draw_premium_grid(preprocessed_image, **grid_kwargs)

    som_proposals = None
    if som_enabled:
        logger.info(
            "Set-of-Mark (SoM) prompting enabled. Generating candidate regions..."
        )
        grid_img, som_proposals = generate_som_proposals(grid_img)
        logger.info("Generated %d candidate proposal regions", len(som_proposals))

    grid_uri = pil_to_data_uri(grid_img)

    detector_images = [grid_uri]
    if round_num > 1 and annotated_prep_uri is not None:
        if pipeline.feedback_image_mode == "annotated":
            detector_images = [annotated_prep_uri]
        elif pipeline.feedback_image_mode == "both":
            detector_images = [grid_uri, annotated_prep_uri]

    raw_text = pipeline.run_inference(
        image_uris=detector_images,
        categories=categories,
        category_definitions=category_definitions,
        feedback=feedback,
        actions=judge_actions,
        som_proposals=som_proposals,
    )

    parse_error = None
    try:
        detections = validate_detections(parse_detections(raw_text), categories)
    except ValueError as exc:
        logger.error("Detector output parsing failed: %s", exc)
        logger.debug(traceback.format_exc())
        detections = []
        parse_error = str(exc)

    return detections, raw_text, parse_error


def node_detector(state: DetectionState) -> Dict[str, Any]:
    """Top-level detector node delegating to either tiled or full-image detection."""
    pipeline = state["pipeline"]
    prep_cfg = pipeline.preprocessing_config
    round_num = state["current_round"]
    max_rounds = state.get("max_rounds", pipeline.max_rounds)
    categories = state["categories"]
    category_definitions = state["category_definitions"]
    preprocessed_image = state["preprocessed_image"]
    prep_w, prep_h = state["prep_w"], state["prep_h"]
    previous_detections_prep = state.get("previous_detections_prep")
    feedback = state.get("feedback")
    judge_actions = state.get("judge_actions")

    grid_kwargs = {
        "style": state["grid_style"],
        "step": state["grid_step"],
        "line_color": state["grid_line_color"],
        "line_width": state["grid_line_width"],
        "font_size": state["grid_font_size"],
        "text_color": state["grid_text_color"],
        "backing_color": state["grid_backing_color"],
    }

    logger.info("=== Round %d/%d (LangGraph) ===", round_num, max_rounds)

    annotated_prep_image, annotated_prep_uri = _prepare_annotated_feedback_image(
        pipeline=pipeline,
        preprocessed_image=preprocessed_image,
        previous_detections_prep=previous_detections_prep,
        round_num=round_num,
        grid_kwargs=grid_kwargs,
    )

    tiling_enabled = prep_cfg.get("tiling_enabled", False)
    if tiling_enabled:
        detections_prep, raw_text, parse_error = _run_tiled_detection(
            pipeline=pipeline,
            preprocessed_image=preprocessed_image,
            prep_w=prep_w,
            prep_h=prep_h,
            categories=categories,
            category_definitions=category_definitions,
            feedback=feedback,
            judge_actions=judge_actions,
            annotated_prep_image=annotated_prep_image,
            round_num=round_num,
            tile_size=prep_cfg.get("tile_size", 512),
            tile_overlap=prep_cfg.get("tile_overlap", 0.2),
            grid_kwargs=grid_kwargs,
        )
    else:
        detections_prep, raw_text, parse_error = _run_full_image_detection(
            pipeline=pipeline,
            preprocessed_image=preprocessed_image,
            categories=categories,
            category_definitions=category_definitions,
            feedback=feedback,
            judge_actions=judge_actions,
            annotated_prep_uri=annotated_prep_uri,
            round_num=round_num,
            som_enabled=prep_cfg.get("som_enabled", False),
            grid_kwargs=grid_kwargs,
        )

    return {
        "detections_prep": detections_prep,
        "raw_detector_output": raw_text,
        "parse_error": parse_error,
    }
