"""Unified CLI for the LLM object-detection / auto-annotation framework.

This module is the single public entry point for running the project from the
command line. It exposes:

  * :class:`PipelineConfig`   -- a pydantic model that validates every CLI option
    (re-exported from :mod:`schemes.argument`).
  * :func:`build_parser`      -- an :class:`argparse.ArgumentParser` mirror of
    every field on :class:`PipelineConfig`.
  * :func:`parse_args`        -- parses ``sys.argv`` (or an explicit argv list),
    overlays an optional YAML config file, and returns a validated
    :class:`PipelineConfig` instance.
  * :func:`main`              -- dispatches to either :func:`free_detection.main`
    or :func:`auto_annotation.main` based on ``config.task``.

Usage::

    python -m main --task free_detection -i img.jpg -c "person, car"
    python -m main --task auto_label    --train_image imgs/ --train_label lbls/ \\
        --yaml_path data.yaml -o ./out --model local-model
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from typing import Any, List, Optional

from schemes import PipelineConfig

logger = logging.getLogger("llmog.main")


# ---------------------------------------------------------------------------
# Argparse parser (mirrors PipelineConfig exactly)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Return an :class:`argparse.ArgumentParser` whose dest names exactly match
    :class:`PipelineConfig` field names, so a plain ``vars(namespace)`` can be
    fed straight into the pydantic model."""

    p = argparse.ArgumentParser(
        prog="llmog",
        description=(
            "Unified CLI for the LLM object-detection framework. Two tasks are "
            "supported via --task: 'free_detection' runs the detector/judge "
            "pipeline on explicit images; 'auto_label' relabels binary "
            "defect/no-defect YOLO annotations into multi-class labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Task selection -----------------------------------------------------
    p.add_argument(
        "--task",
        choices=["free_detection", "auto_label"],
        default="free_detection",
        help="Which pipeline to run. 'free_detection' -> detector/judge loop on "
        "explicit --image paths. 'auto_label' -> batch YOLO relabeling from a "
        "--train_image folder + --yaml_path.",
    )

    # --- Logging -----------------------------------------------------------
    p.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    p.add_argument("--log_file", default=None, help="Optional log file path.")

    # --- Input images ------------------------------------------------------
    p.add_argument(
        "--image",
        "-i",
        metavar="PATH",
        action="append",
        dest="images",
        help="Path to an input image (free_detection task). Repeatable.",
    )
    p.add_argument(
        "--train_image",
        default=None,
        help="Folder of training images (auto_label task).",
    )
    p.add_argument(
        "--train_label",
        default=None,
        help="Folder of YOLO training labels (auto_label task).",
    )
    p.add_argument(
        "--yaml_path", default=None, help="Path to dataset YAML (data.yaml)."
    )
    p.add_argument(
        "--image_extensions",
        default=".jpg,.jpeg,.png",
        help="Comma-separated image extensions to process.",
    )

    # --- Categories --------------------------------------------------------
    p.add_argument(
        "--categories",
        "-c",
        default="person, car, bicycle, dog, cat",
        help="Comma-separated object categories to detect (free_detection).",
    )
    p.add_argument(
        "--definitions",
        "-d",
        default="",
        help="Optional category definitions (plain text, one per line).",
    )
    p.add_argument(
        "--init_class_map",
        action="store_true",
        help="Initialize class map from the YAML file (auto_label).",
    )
    p.add_argument(
        "--conf_threshold",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Confidence threshold (1-5) for low-confidence review log (auto_label).",
    )

    # --- Output ------------------------------------------------------------
    p.add_argument(
        "--output_folder",
        "-o",
        default="./detection_results",
        help="Output directory (results or relabeled annotations).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Alias for --output_folder (free_detection). Hidden to keep the "
        "schema clean; if both are passed, --output_folder wins.",
    )
    p.add_argument(
        "--inplace_saving",
        action="store_true",
        help="Save relabeled annotations in the original label folder (auto_label).",
    )
    p.add_argument(
        "--no_plot", action="store_true", help="Skip the matplotlib preview window."
    )

    # --- Sampling / slicing ------------------------------------------------
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process this many images (sanity check).",
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle image order (seeded by --seed)."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index (0-based, inclusive).",
    )
    p.add_argument(
        "--end_index", type=int, default=None, help="End index (0-based, exclusive)."
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Images per batch (0 disables batching).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't call model or write files; just print plan.",
    )

    # --- Resume ------------------------------------------------------------
    p.add_argument(
        "--resume", action="store_true", help="Legacy per-file resume check."
    )
    p.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="Resume from <output_folder>/.checkpoint.json (default on).",
    )
    p.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Disable auto-resume and start fresh.",
    )

    # --- Server / model ----------------------------------------------------
    p.add_argument(
        "--model",
        default="local-model",
        help="Model name (relabeler / single-model mode).",
    )
    p.add_argument(
        "--detector_model",
        default="local-model",
        help="Detector model name sent in API request.",
    )
    p.add_argument(
        "--judge_model",
        default="local-model",
        help="Judge model name sent in API request.",
    )
    p.add_argument(
        "--judge_url", default=None, help="Separate base URL for the judge model."
    )
    p.add_argument("--api_key", default="not-needed", help="API key.")
    p.add_argument(
        "--base_url",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL of the inference server.",
    )
    p.add_argument(
        "--server_type",
        default="llama_cpp",
        choices=["llama_cpp", "vllm", "external"],
        help="Server backend to use.",
    )
    p.add_argument(
        "--max_workers", type=int, default=1, help="Concurrent image workers."
    )

    # llama.cpp
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable model thinking/reasoning on llama.cpp.",
    )
    p.add_argument(
        "--use_mtp",
        action="store_true",
        default=True,
        help="Enable draft-MTP speculative decoding (llama.cpp).",
    )
    p.add_argument(
        "--no_mtp",
        action="store_false",
        dest="use_mtp",
        help="Disable draft-MTP speculative decoding.",
    )
    p.add_argument(
        "--ctx_size", type=int, default=20000, help="Context size for llama.cpp."
    )
    p.add_argument("--port", type=int, default=8080, help="Port for llama.cpp server.")
    p.add_argument(
        "--parallel_slots",
        type=int,
        default=1,
        help="Parallel inference slots for llama.cpp.",
    )

    # --- Detection pipeline tuning ----------------------------------------
    p.add_argument(
        "--max_rounds", type=int, default=2, help="Max detector/judge rounds per image."
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

    # --- vLLM configuration ------------------------------------------------
    p.add_argument("--max_model_len", type=int, default=20000)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--pipeline_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--quantization", default=None)
    p.add_argument("--kv_cache_dtype", default="auto")
    p.add_argument("--max_num_seqs", type=int, default=2)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--enable_chunked_prefill", action="store_true", default=True)
    p.add_argument(
        "--no_chunked_prefill", action="store_false", dest="enable_chunked_prefill"
    )
    p.add_argument("--enable_prefix_caching", action="store_true", default=True)
    p.add_argument(
        "--no_prefix_caching", action="store_false", dest="enable_prefix_caching"
    )
    p.add_argument("--speculative_model", default=None)
    p.add_argument("--num_speculative_tokens", type=int, default=None)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument(
        "--no_trust_remote_code", action="store_false", dest="trust_remote_code"
    )
    p.add_argument("--download_dir", default=None)
    p.add_argument("--limit_mm_per_prompt", default=None)
    p.add_argument("--chat_template", default=None)
    p.add_argument(
        "--extra_args",
        action="append",
        default=None,
        help="Extra raw tokens forwarded to vLLM server (repeatable).",
    )

    # --- VLM image encoding -----------------------------------------------
    p.add_argument("--image_min_tokens", type=int, default=1024)
    p.add_argument("--image_max_tokens", type=int, default=4096)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    # --- Preprocessing -----------------------------------------------------
    p.add_argument(
        "--prep_enabled", action="store_true", help="Enable image preprocessing."
    )
    p.add_argument("--prep_short_edge", type=int, default=1024)
    p.add_argument("--prep_pad_square", action="store_true")
    p.add_argument(
        "--prep_contrast_method",
        choices=["none", "clahe", "autocontrast"],
        default="none",
    )
    p.add_argument("--prep_gamma", type=float, default=1.0)
    p.add_argument(
        "--prep_denoise_method", choices=["none", "bilateral", "nlm"], default="none"
    )
    p.add_argument("--prep_sharpen", action="store_true")
    p.add_argument("--prep_white_balance", action="store_true")
    p.add_argument(
        "--prep_grid_style",
        choices=["standard", "transparent", "fine", "none"],
        default="standard",
    )
    p.add_argument(
        "--prep_som_enabled",
        action="store_true",
        help="Enable Set-of-Mark visual prompting overlay.",
    )
    p.add_argument("--prep_tiling_enabled", action="store_true")
    p.add_argument("--prep_tile_size", type=int, default=512)
    p.add_argument("--prep_tile_overlap", type=float, default=0.2)
    p.add_argument("--prep_crop_verify_enabled", action="store_true")
    p.add_argument("--prep_crop_padding", type=float, default=0.15)

    # Custom grid overlays
    p.add_argument("--prep_grid_step", type=int, default=100)
    p.add_argument("--prep_grid_line_width", type=int, default=1)
    p.add_argument("--prep_grid_font_size", type=int, default=0)
    p.add_argument("--prep_grid_line_color", default="red")
    p.add_argument("--prep_grid_text_color", default="white")
    p.add_argument("--prep_grid_backing_color", default="black")

    # VLM processor pixels
    p.add_argument("--prep_send_pixel_bounds", action="store_true")
    p.add_argument("--prep_min_pixels", type=int, default=200_704)
    p.add_argument("--prep_max_pixels", type=int, default=4_194_304)

    # --- Serving extras ---------------------------------------------------
    p.add_argument(
        "--serving_extra",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra override tokens for the underlying server-manager kwargs, "
        "in 'key=value' form (repeatable). Parsed into a dict on the config.",
    )

    # --- Config file ------------------------------------------------------
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a PipelineConfig YAML file. Values are loaded first and "
        "overridden by explicit CLI flags.",
    )

    return p


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def _load_yaml_config(path: str) -> dict:
    """Read a YAML config file into a flat dict of field-name -> value."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"--config file '{path}' must contain a YAML mapping.")
    return data


def _coerce_serving_extra(raw: Optional[List[str]]) -> dict:
    """Convert repeated ``key=value`` CLI tokens into a dict."""
    if raw is None:
        return {}
    out: dict = {}
    for token in raw:
        if "=" not in token:
            raise ValueError(
                f"--serving_extra must be 'key=value', got: {token!r}"
            )
        key, _, value = token.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--serving_extra has empty key in {token!r}")
        # Try to coerce numeric / bool literals so pydantic stays happy.
        v: Any = value.strip()
        if v.lower() in {"true", "false"}:
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# parse_args helper
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    """Parse CLI args (and optional ``--config`` YAML) -> validated
    :class:`PipelineConfig`.

    Precedence (lowest to highest):
      1. PipelineConfig field defaults
      2. YAML key/values from ``--config``
      3. Explicit CLI flags
    """
    parser = build_parser()

    # Use argparse.SUPPRESS for every optional arg's default so that unset
    # flags don't appear in the parsed namespace at all. This makes it
    # possible to layer YAML (base) < CLI (explicit) < pydantic defaults
    # (anything still missing) -- which is exactly the precedence users
    # expect from a config-file system: explicit CLI flags always win over
    # YAML, and YAML always wins over framework defaults.
    for action in parser._actions:
        if action.dest in (argparse.SUPPRESS, "help"):
            continue
        # store_true / store_false flags keep their False/True literal default
        # (pydantic still wants a real bool rather than SUPPRESS), but explicit
        # CLI passing sets them to the opposite so the precedence still works
        # because the YAML layer only fills values that are missing entirely.
        action.default = argparse.SUPPRESS

    # First-pass: parse only enough to discover --config.
    ns_peek, _ = parser.parse_known_args(argv)
    overrides: dict = {}
    if getattr(ns_peek, "config", None):
        overrides = _load_yaml_config(ns_peek.config)

    # With all defaults SUPPRESSED, parse_args only puts explicitly-passed
    # flags into the namespace -- exactly what we want for the precedence:
    #   pydantic defaults  <  YAML  <  CLI
    ns = parser.parse_args(argv)
    raw = vars(ns)

    # base layer: YAML overrides pydantic defaults
    # CLI explicit flags override YAML
    merged: dict = {**overrides, **raw}

    # --- output_dir alias --------------------------------------------------
    yaml_outdir = overrides.get("output_dir") or overrides.get("output_folder")
    cli_outdir = raw.get("output_dir")
    if cli_outdir is not None:
        merged["output_folder"] = cli_outdir
    elif yaml_outdir is not None and "output_folder" not in raw:
        merged["output_folder"] = yaml_outdir
    merged.pop("output_dir", None)

    # --- serving_extra -----------------------------------------------------
    cli_extra = _coerce_serving_extra(raw.get("serving_extra"))
    yaml_extra = overrides.get("serving_extra", {}) or {}
    # CLI wins; YAML fills the gaps.
    merged["serving_extra"] = {**yaml_extra, **cli_extra}

    # config is consumed, not a PipelineConfig field
    merged.pop("config", None)

    # Drop any None values so pydantic defaults apply where neither CLI nor
    # YAML supplied a value. (PipelineConfig forbids extras, so we must not
    # leave stray Nones for fields the model has no default-on-None policy
    # for -- in practice every field has a real default, so this is a no-op
    # belt-and-braces guard.)
    merged = {k: v for k, v in merged.items() if v is not None}

    try:
        return PipelineConfig(**merged)
    except Exception as exc:  # surface validation errors nicely
        parser.error(str(exc))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch to the selected task's entry point based on ``config.task``."""
    config = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if config.task == "free_detection":
        from free_detection import main as free_detection_main

        free_detection_main(config)
    elif config.task == "auto_label":
        from auto_annotation import main as auto_annotation_main

        auto_annotation_main(config)
    else:  # pragma: no cover -- guarded by Literal + argparse choices
        raise ValueError(f"Invalid task: {config.task!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
