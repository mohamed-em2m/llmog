"""CLI argument parsing for the auto-annotation pipeline.

This parser is intentionally a thin standalone layer kept for the
``auto-annotation`` console script and ``python -m auto_annotation``.

When invoked through the unified entry point in :mod:`main` (with
``--task auto_label``), :class:`PipelineConfig` is already validated and
:this parser is bypassed entirely. Both flavours accept the exact same flags
:though, so behaviour is identical.
"""

import argparse


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to a file where logs should also be written.",
    )
    parser.add_argument(
        "--output_folder",
        "--output-folder",
        dest="output_folder",
        type=str,
        required=True,
        help="Where to save the relabeled output.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use for classification."
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="API key for the external/hosted model."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base URL for the external/hosted model.",
    )
    parser.add_argument(
        "--server_type",
        type=str,
        default="llama_cpp",
        choices=["llama_cpp", "llama_cpp_python", "vllm", "external"],
        help="Use a local llama.cpp server ('llama_cpp' = native binary, "
        "'llama_cpp_python' = bundled in the llama-cpp-python package) "
        "instead of an external API.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable the model's thinking/reasoning mode on the local llama.cpp server "
        "(--use_local_model only). Off by default: faster and usually unnecessary for a "
        "single classify-this-crop call.",
    )
    parser.add_argument(
        "--use_mtp",
        action="store_true",
        default=True,
        help="Enable draft-MTP speculative decoding on the local llama.cpp server "
        "(--use_local_model only). On by default for speed; pass --no_mtp to disable it "
        "if you hit compatibility issues with a given model/build.",
    )
    parser.add_argument(
        "--no_mtp",
        action="store_false",
        dest="use_mtp",
        help="Disable draft-MTP speculative decoding (--use_local_model only).",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=20000,
        help="Context size for the local llama.cpp server (--use_local_model only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the local llama.cpp server (--use_local_model only).",
    )
    parser.add_argument(
        "--parallel_slots",
        type=int,
        default=1,
        help="Number of parallel inference slots on the local llama.cpp server "
        "(--use_local_model only). If you raise this, --max_workers can be raised to match "
        "so multiple images are in flight at once.",
    )
    parser.add_argument(
        "--train_image",
        type=str,
        required=True,
        help="Path to the folder of training images.",
    )
    parser.add_argument(
        "--train_label",
        type=str,
        required=True,
        help="Path to the folder of YOLO training labels.",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the dataset yaml file (data.yaml).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Confidence (1-5) at/below which a box is ALSO logged to a *_low_confidence.json for manual review.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process this many images (quick sanity-check run instead of the full dataset). "
        "Applied after --start_index/--end_index.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order (seeded by --seed) before applying --start_index/--end_index/"
        "--num_samples, so a sample isn't just the first N images alphabetically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set, for reproducible samples.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index (0-based, inclusive) of the image range to process, applied after "
        "--shuffle and before --num_samples. Useful for splitting a large dataset across "
        "multiple runs/machines, e.g. --start_index 0 --end_index 1000 on one machine and "
        "--start_index 1000 --end_index 2000 on another.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index (0-based, exclusive) of the image range to process. See --start_index.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't call the model and don't write any files; just print what would happen.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Legacy resume check: skip images whose output label file already exists on disk. "
        "Has no effect when --inplace_saving is set (there is no separate output file to check "
        "in that mode) -- use --auto_resume instead, which is on by default. Auto-resume is the "
        "more robust option in general since it only counts an image as done once it's confirmed "
        "in the checkpoint; you can leave --resume off in normal usage.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        default=True,
        help="Automatically resume from '<output-folder>/.checkpoint.json' if one exists "
        "(on by default). This is what lets an interrupted/crashed run be continued by simply "
        "re-running the exact same command -- already-finished images and batches are skipped "
        "and the accumulated class_map is restored.",
    )
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Disable auto-resume and ignore/clear any existing checkpoint, starting completely fresh.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of images per batch (default: 50). When not using --inplace_saving, each "
        "batch's relabeled annotations are written to their own 'batch_XXXX' subfolder under "
        "--output-folder, and a checkpoint marks each batch done as soon as it finishes, so a "
        "resumed run can skip whole finished batches quickly. Pass 0 to disable batching "
        "(single flat output folder, same as before).",
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of image file extensions to process.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of images to process concurrently (thread pool). Keep at 1 for a local "
        "llama.cpp server with parallel_slots=1; raise it for a remote API that supports "
        "concurrent requests.",
    )
    # vLLM Configuration Options
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=20000,
        help="vLLM maximum model length.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="vLLM pipeline parallel size.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="vLLM data type.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="vLLM quantization method.",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        default="auto",
        help="vLLM KV cache data type.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=2,
        help="vLLM maximum number of sequences.",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Enforce eager execution in vLLM.",
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        default=True,
        help="Enable chunked prefill in vLLM.",
    )
    parser.add_argument(
        "--no_chunked_prefill",
        action="store_false",
        dest="enable_chunked_prefill",
        help="Disable chunked prefill in vLLM.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=True,
        help="Enable prefix caching in vLLM.",
    )
    parser.add_argument(
        "--no_prefix_caching",
        action="store_false",
        dest="enable_prefix_caching",
        help="Disable prefix caching in vLLM.",
    )
    parser.add_argument(
        "--speculative_model",
        type=str,
        default=None,
        help="vLLM speculative model name.",
    )
    parser.add_argument(
        "--num_speculative_tokens",
        type=int,
        default=None,
        help="Number of speculative tokens.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code in vLLM.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="Do not trust remote code in vLLM.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="vLLM model download directory.",
    )
    parser.add_argument(
        "--limit_mm_per_prompt",
        type=str,
        default=None,
        help="vLLM limit multimodal items per prompt.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="vLLM chat template.",
    )
    parser.add_argument(
        "--extra_args",
        action="append",
        default=None,
        help="Extra arguments to pass to vLLM server.",
    )
    parser.add_argument(
        "--serving_extra",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra override tokens for the underlying server-manager kwargs, "
        "in 'key=value' form (repeatable).",
    )
    parser.add_argument(
        "--image_min_tokens",
        type=int,
        default=1024,
        help="Minimum number of tokens to use for image encoding.",
    )
    parser.add_argument(
        "--image_max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to use for image encoding.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the input image for the model.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the input image for the model.",
    )
    parser.add_argument(
        "--init_class_map",
        action="store_true",
        help="Initialize the class map from the YAML file.",
    )
    parser.add_argument(
        "--inplace_saving",
        action="store_true",
        help="save inplace the relabeled annotations in the original label folder instead of a separate output folder.",
    )

    # ── Class Expectation Mode ──────────────────────────────────────────────
    parser.add_argument(
        "--class_mode",
        "--class-mode",
        dest="class_mode",
        type=str,
        default="hybrid",
        choices=["strict", "hybrid", "free"],
        help=(
            "How the VLM should handle target class expectations when classifying crops:\n"
            "  strict  — Agent is locked to the class list from data.yaml (or --class_definitions).\n"
            "            If the crop does not match any known class it is labelled 'none'.\n"
            "  hybrid  — Agent prioritises known classes but can discover and name brand-new\n"
            "            classes when the crop clearly does not match any existing label.\n"
            "            This is the default and best for growing datasets.\n"
            "  free    — Agent freely names whatever defect/object it sees, ignoring the\n"
            "            existing class list entirely. Use for fully open-vocabulary runs.\n"
        ),
    )
    parser.add_argument(
        "--class_definitions",
        "--class-definitions",
        dest="class_definitions",
        type=str,
        default="",
        help=(
            "Optional multiline string (or path to a .txt/.md file) with per-class descriptions "
            "that help the model distinguish between classes.  "
            "Example: '- hole: missing fabric\\n- stain: discoloration'.  "
            "If a file path is given and exists, its content is read; otherwise the string is used verbatim."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[
            "fabric_defects",
            "coco",
            "road_traffic",
            "retail_packaging",
            "pcb_defects",
        ],
        help=(
            "Quick-load a built-in category preset to populate --class_definitions.  "
            "Ignored when --class_definitions is already set.  "
            "Choices:\n"
            "  fabric_defects   — hole, stain, tear, cut, knot, weaving_defect\n"
            "  coco             — person, car, bicycle, dog, cat, chair, bottle, laptop, cell_phone, book\n"
            "  road_traffic     — car, truck, pedestrian, cyclist, traffic_light, traffic_sign, bus, motorcycle\n"
            "  retail_packaging — box, barcode, product_label, bottle, can, pouch, blister_pack\n"
            "  pcb_defects      — short_circuit, missing_component, solder_bridge, broken_trace, scratch, misalignment\n"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone argparse parser for the auto-annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Relabel binary defect/no-defect YOLO annotations into multi-class defect labels using a VLM."
    )
    _add_arguments(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.start_index is not None and args.start_index < 0:
        parser.error("--start_index must be >= 0")
    if (
        args.end_index is not None
        and args.start_index is not None
        and args.end_index <= args.start_index
    ):
        parser.error("--end_index must be greater than --start_index")

    # Normalize serving_extra from list[str] -> dict for downstream consumers.
    if getattr(args, "serving_extra", None):
        coerced = {}
        for token in args.serving_extra:
            if "=" not in token:
                parser.error(f"--serving_extra must be 'key=value', got: {token!r}")
            k, _, v = token.partition("=")
            k = k.strip()
            v = v.strip()
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
            coerced[k] = v
        args.serving_extra = coerced
    else:
        args.serving_extra = {}

    # ── Resolve --class_definitions (file path or inline text) ─────────────
    import os as _os

    raw_defs = getattr(args, "class_definitions", "") or ""
    if raw_defs and _os.path.isfile(raw_defs):
        try:
            with open(raw_defs, "r", encoding="utf-8") as _f:
                args.class_definitions = _f.read().strip()
        except Exception as _e:
            parser.error(f"--class_definitions: could not read file {raw_defs!r}: {_e}")

    # ── Resolve --preset -> populate class_definitions if not already set ───
    _PRESET_DEFS = {
        "fabric_defects": (
            "- hole: missing fabric or puncture\n"
            "- stain: discoloration or surface contaminant\n"
            "- tear: frayed, uneven physical separation\n"
            "- cut: clean sharp slice or incision\n"
            "- knot: raised thread lump or snarl\n"
            "- weaving_defect: uneven thread density or missing yarn"
        ),
        "coco": (
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
        "road_traffic": (
            "- car: passenger sedan, coupe, or SUV\n"
            "- truck: heavy transport or cargo vehicle\n"
            "- pedestrian: person on foot\n"
            "- cyclist: person riding a bicycle\n"
            "- traffic_light: signal light lamp\n"
            "- traffic_sign: road regulatory or warning signboard\n"
            "- bus: public transit passenger bus\n"
            "- motorcycle: motorized two-wheeled vehicle"
        ),
        "retail_packaging": (
            "- box: cardboard or corrugated carton\n"
            "- barcode: 1D or 2D scanner code\n"
            "- product_label: brand packaging label\n"
            "- bottle: glass or plastic container\n"
            "- can: aluminum or tin can\n"
            "- pouch: flexible plastic packaging\n"
            "- blister_pack: clear molded plastic bubble packaging"
        ),
        "pcb_defects": (
            "- short_circuit: unintended electrical contact\n"
            "- missing_component: empty pad where SMD/component should be\n"
            "- solder_bridge: solder connecting adjacent pins\n"
            "- broken_trace: severed copper circuit trace\n"
            "- scratch: surface gouge across the solder mask\n"
            "- misalignment: component rotated or shifted off pad"
        ),
    }
    preset = getattr(args, "preset", None)
    if preset and not args.class_definitions:
        args.class_definitions = _PRESET_DEFS.get(preset, "")

    # Ensure class_mode has a sensible default even when PipelineConfig skips parse_args
    if not getattr(args, "class_mode", None):
        args.class_mode = "hybrid"

    return args
