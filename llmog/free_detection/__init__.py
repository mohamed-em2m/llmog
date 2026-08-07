"""LLM Object Detection package.

This package contains the detector/judge pipeline, the image preprocessing
helpers, and the Gradio web UI.

Top-level public surface:

  * :func:`main`  -- CLI entry point. Accepts a validated
    :class:`~schemes.PipelineConfig` (preferred) or ``None`` to parse
    ``sys.argv`` directly via :func:`build_parser`.
  * :func:`build_parser` -- the argparse parser mirroring every relevant
    :class:`PipelineConfig` field.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from openai import OpenAI

from free_detection.detection_pipeline import ObjectDetectionPipeline
from schemes import PipelineConfig

logger = logging.getLogger("free_detection")

__all__ = ["main", "build_parser", "ObjectDetectionPipeline"]


# ---------------------------------------------------------------------------
# Argparse parser -- mirrors every PipelineConfig field relevant to the
# free-detection task. Kept here so the standalone ``detection-cli`` entry
# point can run with --help without importing heavy modules.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detection-cli",
        description="Run the LLM object-detection detector/judge pipeline on one or more images from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Images ---
    p.add_argument(
        "--image",
        "-i",
        metavar="PATH",
        action="append",
        required=True,
        dest="images",
        help="Path to an input image. Can be specified multiple times.",
    )

    # --- Categories ---
    p.add_argument(
        "--categories",
        "-c",
        metavar="LIST",
        default="person, car, bicycle, dog, cat",
        help="Comma-separated list of object categories to detect.",
    )
    p.add_argument(
        "--definitions",
        "-d",
        metavar="TEXT",
        default="",
        help="Optional category definitions (plain text, one per line).",
    )

    # --- Server / model ---
    p.add_argument(
        "--base_url",
        metavar="URL",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL of the inference server.",
    )
    p.add_argument(
        "--api_key",
        metavar="KEY",
        default="not-needed",
        help="API key (use 'not-needed' for local servers).",
    )
    p.add_argument(
        "--detector_model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the detector API request.",
    )
    p.add_argument(
        "--judge_model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the judge API request.",
    )
    p.add_argument(
        "--judge_url",
        metavar="URL",
        default=None,
        help="Separate base URL for the judge model (defaults to --base_url).",
    )

    # --- Pipeline params ---
    p.add_argument(
        "--max_rounds",
        type=int,
        default=2,
        help="Maximum number of detector/judge rounds per image.",
    )
    p.add_argument(
        "--score_threshold",
        type=int,
        default=8,
        help="Stop early when judge score reaches this value (0-10).",
    )
    p.add_argument("--detector_temperature", type=float, default=0.9)
    p.add_argument("--detector_top_p", type=float, default=0.95)
    p.add_argument("--judge_temperature", type=float, default=0.2)
    p.add_argument("--detector_max_tokens", type=int, default=4096)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--api_retries", type=int, default=3)

    # --- Preprocessing ---
    p.add_argument(
        "--prep_enabled", action="store_true", help="Enable image preprocessing."
    )
    p.add_argument(
        "--prep_short_edge",
        type=int,
        default=1024,
        help="Target size for short edge of the image.",
    )
    p.add_argument(
        "--prep_pad_square",
        action="store_true",
        help="Pad preprocessed image to square with neutral gray.",
    )
    p.add_argument(
        "--prep_contrast_method",
        choices=["none", "clahe", "autocontrast"],
        default="none",
        help="Contrast enhancement method.",
    )
    p.add_argument(
        "--prep_gamma", type=float, default=1.0, help="Gamma correction factor."
    )
    p.add_argument(
        "--prep_denoise_method",
        choices=["none", "bilateral", "nlm"],
        default="none",
        help="Denoising method.",
    )
    p.add_argument(
        "--prep_sharpen", action="store_true", help="Apply unsharp mask sharpening."
    )
    p.add_argument(
        "--prep_white_balance",
        action="store_true",
        help="Apply white balance correction.",
    )
    p.add_argument(
        "--prep_grid_style",
        choices=["standard", "transparent", "fine", "none"],
        default="standard",
        help="Visual grid overlay style.",
    )
    p.add_argument(
        "--prep_som_enabled",
        action="store_true",
        help="Enable Set-of-Mark visual prompting overlay.",
    )
    p.add_argument(
        "--prep_tiling_enabled",
        action="store_true",
        help="Enable image tiling for small object detection.",
    )
    p.add_argument(
        "--prep_tile_size", type=int, default=512, help="Tile size in pixels."
    )
    p.add_argument(
        "--prep_tile_overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between tiles (0.0 to 0.5).",
    )
    p.add_argument(
        "--prep_crop_verify_enabled",
        action="store_true",
        help="Enable multi-pass Crop & Verify validation pipeline.",
    )
    p.add_argument(
        "--prep_crop_padding",
        type=float,
        default=0.15,
        help="Context padding ratio for cropped patches.",
    )

    # Custom Grid overlays and VLM processor parameters
    p.add_argument(
        "--prep_grid_step",
        type=int,
        default=100,
        help="Grid line separation (0-1000 scale).",
    )
    p.add_argument(
        "--prep_grid_line_width",
        type=int,
        default=1,
        help="Grid line thickness in pixels.",
    )
    p.add_argument(
        "--prep_grid_font_size",
        type=int,
        default=0,
        help="Grid text label font size (0 for auto).",
    )
    p.add_argument(
        "--prep_grid_line_color",
        type=str,
        default="red",
        help="Grid line color (Hex or CSS name).",
    )
    p.add_argument(
        "--prep_grid_text_color",
        type=str,
        default="white",
        help="Grid text label color (Hex or CSS name).",
    )
    p.add_argument(
        "--prep_grid_backing_color",
        type=str,
        default="black",
        help="Grid text label backing color (Hex or CSS name or 'none').",
    )

    p.add_argument(
        "--prep_send_pixel_bounds",
        action="store_true",
        help="Send min_pixels and max_pixels in API request.",
    )
    p.add_argument(
        "--prep_min_pixels", type=int, default=200704, help="VLM min_pixels parameter."
    )
    p.add_argument(
        "--prep_max_pixels", type=int, default=4194304, help="VLM max_pixels parameter."
    )

    # --- Output ---
    p.add_argument(
        "--output_dir",
        "-o",
        metavar="DIR",
        default="./detection_results",
        help="Base output directory. Each image gets its own sub-folder.",
    )
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip the matplotlib preview window.",
    )

    return p


# ---------------------------------------------------------------------------
# Preprocessing config assembly
# ---------------------------------------------------------------------------
def _build_prep_config(args) -> dict:
    """Translate the prep_* fields on either a PipelineConfig or Namespace
    into the dict shape expected by :class:`ObjectDetectionPipeline`."""
    if not args.prep_enabled:
        return {
            "resolution_enabled": False,
            "contrast_method": "none",
            "denoise_method": "none",
            "som_enabled": False,
            "tiling_enabled": False,
            "crop_verify_enabled": False,
            "grid_style": args.prep_grid_style,
            "grid_step": args.prep_grid_step,
            "grid_line_width": args.prep_grid_line_width,
            "grid_font_size": args.prep_grid_font_size,
            "grid_line_color": args.prep_grid_line_color,
            "grid_text_color": args.prep_grid_text_color,
            "grid_backing_color": args.prep_grid_backing_color,
            "send_pixel_bounds": args.prep_send_pixel_bounds,
            "min_pixels": args.prep_min_pixels,
            "max_pixels": args.prep_max_pixels,
        }
    return {
        "resolution_enabled": True,
        "target_short_edge": args.prep_short_edge,
        "pad_to_square": args.prep_pad_square,
        "contrast_method": args.prep_contrast_method,
        "clip_limit": 2.0,
        "gamma": args.prep_gamma,
        "denoise_method": args.prep_denoise_method,
        "sharpen": args.prep_sharpen,
        "white_balance": args.prep_white_balance,
        "grid_style": args.prep_grid_style,
        "som_enabled": args.prep_som_enabled,
        "tiling_enabled": args.prep_tiling_enabled,
        "tile_size": args.prep_tile_size,
        "tile_overlap": args.prep_tile_overlap,
        "crop_verify_enabled": args.prep_crop_verify_enabled,
        "crop_padding": args.prep_crop_padding,
        # Custom grid layout parameters
        "grid_step": args.prep_grid_step,
        "grid_line_width": args.prep_grid_line_width,
        "grid_font_size": args.prep_grid_font_size,
        "grid_line_color": args.prep_grid_line_color,
        "grid_text_color": args.prep_grid_text_color,
        "grid_backing_color": args.prep_grid_backing_color,
        # VLM Processor bounds parameters
        "send_pixel_bounds": args.prep_send_pixel_bounds,
        "min_pixels": args.prep_min_pixels,
        "max_pixels": args.prep_max_pixels,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(args=None) -> None:
    """Run the free-detection pipeline on the configured images.

    ``args`` may be:
      * a :class:`~schemes.PipelineConfig` (preferred -- passed in by
        :mod:`main` after a single centralized parse),
      * a legacy argparse ``Namespace`` from :func:`build_parser` (the
        standalone ``detection-cli`` entry point),
      * ``None`` -- in which case :func:`build_parser` parses ``sys.argv``.
    """
    if args is None:
        # Standalone invocation: parse argv via the local parser, then upgrade
        # to a PipelineConfig so the rest of the pipeline sees a unified
        # config object regardless of entry point.
        ns = build_parser().parse_args()
        raw = {k: v for k, v in vars(ns).items() if v is not None}
        # Force the task so downstream code can read it consistently.
        raw["task"] = "free_detection"
        try:
            args = PipelineConfig(**raw)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    # PipelineConfig is a pydantic BaseModel; both it and argparse Namespace
    # expose the same dotted attribute API.
    base_url = args.base_url
    if not base_url:
        base_url = "http://localhost:8080/v1"
    detector_client = OpenAI(api_key=args.api_key, base_url=base_url)
    judge_url = args.judge_url or base_url
    judge_client = (
        OpenAI(api_key=args.api_key, base_url=judge_url)
        if judge_url != base_url
        else detector_client
    )

    # Parse categories
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        print("ERROR: --categories must contain at least one entry.", file=sys.stderr)
        sys.exit(1)

    prep_config = _build_prep_config(args)

    pipeline = ObjectDetectionPipeline(
        detector_client=detector_client,
        judge_client=judge_client,
        detector_model=args.detector_model,
        judge_model=args.judge_model,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold,
        detector_temperature=args.detector_temperature,
        detector_top_p=args.detector_top_p,
        judge_temperature=args.judge_temperature,
        detector_max_tokens=args.detector_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        api_retries=args.api_retries,
        preprocessing_config=prep_config,
    )

    from pathlib import Path

    out_dir = getattr(args, "output_dir", None) or args.output_folder
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    all_results = []

    for image_path in args.images:
        p = Path(image_path)
        if not p.is_file():
            print(f"WARNING: image not found, skipping: {image_path}", file=sys.stderr)
            continue

        image_out_dir = out_base / p.stem
        print(f"\n{'='*60}")
        print(f"Processing: {p.name}  →  {image_out_dir}")
        print(f"{'='*60}")

        def on_round(round_result, _annotated):
            print(
                f"  Round {round_result.round}: "
                f"score {round_result.score}/10, "
                f"{len(round_result.detections)} detection(s)"
                + (f"  [parse error]" if round_result.parse_error else "")
            )

        try:
            best, history = pipeline.run(
                image_path=str(p),
                categories=categories,
                category_definitions=args.definitions,
                show_plot=not args.no_plot,
                output_dir=str(image_out_dir),
                progress_callback=on_round,
            )
            print(
                f"  ✅ Best: round {best['round']}, score {best['score']}/10, "
                f"{len(best['detections'] or [])} detection(s)"
            )
            all_results.append(
                {
                    "image": str(p),
                    "status": "ok",
                    "best_round": best["round"],
                    "best_score": best["score"],
                    "n_detections": len(best["detections"] or []),
                    "output_dir": str(image_out_dir),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ ERROR: {exc}", file=sys.stderr)
            all_results.append({"image": str(p), "status": f"error: {exc}"})

    # Write summary
    summary_path = out_base / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
