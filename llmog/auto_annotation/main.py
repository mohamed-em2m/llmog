"""Entry point: parse args, load the dataset yaml, drive the batch runner,
persist the updated yaml, and stop the local server cleanly.

This module accepts either:
  * a validated :class:`schemes.PipelineConfig` (the recommended path going
    forward -- passed in by :mod:`main` after a single shared parse), or
  * ``None`` -- in which case :func:`auto_annotation.cli.parse_args` is run
    to read sys.argv directly, for the standalone ``auto-annotation`` entry
    point or anyone running ``python -m auto_annotation``.
"""

import os

import yaml

from auto_annotation.logging_utils import logger, setup_logging
from auto_annotation.stats import RunStats
from auto_annotation.checkpoint import CheckpointManager
from auto_annotation.image_io import load_or_init_class_map
from auto_annotation.server_init import build_client
from auto_annotation.batch_runner import read_images_with_labels
from auto_annotation.yaml_utils import save_updated_yaml
from auto_annotation.cli import parse_args as aa_parse_args


def parse_categories_list(raw) -> list:
    """Split a comma-separated --categories string into a clean ordered list."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = str(raw).split(",")
    seen, out = set(), []
    for item in items:
        name = str(item).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _normalize_yaml_descriptions(value) -> str:
    """Normalize a yaml description field (dict / list / str) to text."""
    if not value:
        return ""
    if isinstance(value, dict):
        lines = [f"- {k}: {v}" for k, v in value.items()]
        return "\n".join(lines).strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x).strip() for x in value if str(x).strip()).strip()
    text = str(value)
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    return text.strip()


def resolve_auto_label_definitions(args, yaml_data=None) -> str:
    """Resolve effective class definitions for auto_label.

    Layers (merged, yaml first so CLI text wins on repeat reads):
      1. data.yaml description fields (``descriptions`` / ``definitions`` /
         ``class_definitions`` / ``class_descriptions`` as dict, list or str)
      2. --class_definitions, falling back to -d/--definitions (file path
         is read when it exists)
      3. --preset (only when layers 1+2 are both empty)

    Escaped ``\\n`` literals (PowerShell) become real newlines.
    """
    # Layer 1: data.yaml description fields (dataset-side documentation).
    yaml_text = ""
    if isinstance(yaml_data, dict):
        for _key in (
            "descriptions",
            "definitions",
            "class_definitions",
            "class_descriptions",
        ):
            yaml_text = _normalize_yaml_descriptions(yaml_data.get(_key))
            if yaml_text:
                logger.info(
                    f"Found class descriptions in data.yaml field '{_key}' "
                    f"({len(yaml_text)} chars)."
                )
                break

    # Layer 2: explicit CLI text (--class_definitions > -d/--definitions).
    raw = (
        getattr(args, "class_definitions", "") or getattr(args, "definitions", "") or ""
    )
    if isinstance(raw, str) and "\\n" in raw:
        raw = raw.replace("\\n", "\n")
    raw = raw.strip() if isinstance(raw, str) else ""
    cli_text = ""
    if raw:
        import os as _os

        if _os.path.isfile(raw):
            try:
                with open(raw, "r", encoding="utf-8") as _f:
                    cli_text = _f.read().strip()
            except Exception as exc:
                logger.warning(f"Could not read definitions file {raw!r}: {exc}")
        else:
            cli_text = raw

    # Merge: yaml base + CLI appended (CLI wins on repeat reads by the model).
    if yaml_text and cli_text and cli_text not in yaml_text:
        logger.info("Merging data.yaml descriptions with CLI definitions.")
        return f"{yaml_text}\n{cli_text}"
    if cli_text:
        return cli_text
    if yaml_text:
        return yaml_text

    # Layer 3: preset fills the gap only.
    preset = getattr(args, "preset", None)
    if preset:
        try:
            from image_classification.prompt import _PRESET_DEFS as _CLS_PRESETS

            return _CLS_PRESETS.get(preset, "")
        except Exception as exc:
            logger.warning(f"Could not resolve --preset {preset!r}: {exc}")
    return ""


def main(args=None):
    """Run the auto-annotation pipeline.

    Args must be a :class:`PipelineConfig` (preferred) OR an argparse.Namespace
    with the legacy auto_annotation fields. Passing ``None`` triggers a fresh
    ``parse_args()`` on ``sys.argv`` for the standalone entry-point case.
    """
    if args is None:
        # Standalone invocation (e.g. ``python -m auto_annotation ...`` or
        # the ``auto-annotation`` console script). Parse from sys.argv.
        args = aa_parse_args()

    # PipelineConfig is a pydantic BaseModel: pydantic v2 prints a friendly
    # repr, so callers can pass either flavor transparently. We normalize to
    # attribute access in either case -- both pydantic BaseModel and argparse
    # Namespace expose the same dotted API.
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    yaml_path = args.yaml_path
    if not yaml_path:
        logger.error("task='auto_label' requires --yaml_path.")
        exit(1)

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read dataset yaml file at {yaml_path}: {e}")
        exit(1)

    class_map = (
        load_or_init_class_map(data.get("names", [])) if args.init_class_map else {}
    )
    # ---- Track yaml names explicitly -----------------------------------------
    # data.yaml is always read, but its `names` only seed the prompt when
    # --init_class_map is set. Log both cases so a silently-ignored yaml is
    # visible instead of a mystery.
    yaml_names = data.get("names", []) or []
    if isinstance(yaml_names, dict):
        yaml_names = [yaml_names[k] for k in sorted(yaml_names, key=int)]
    if yaml_names:
        if args.init_class_map:
            logger.info(
                f"Loaded {len(yaml_names)} class(es) from data.yaml "
                f"(--init_class_map): {list(yaml_names)}"
            )
        else:
            logger.warning(
                f"data.yaml contains {len(yaml_names)} name(s) "
                f"{list(yaml_names)} but --init_class_map was NOT passed, "
                "so they are ignored and the run starts from an empty class map "
                "(pass --init_class_map to reuse them, or --categories to seed "
                "a list explicitly)."
            )
    else:
        logger.info("data.yaml contains no names; starting from an empty class map.")

    # ---- Seed known classes from --categories --------------------------------
    # --categories/-c was previously silently ignored by auto_label. Merge it
    # into the class_map so strict mode locks to it and hybrid reuses it.
    # Yaml ids (via --init_class_map) win on name conflict to keep old label
    # files valid.
    cli_categories = parse_categories_list(getattr(args, "categories", ""))
    if cli_categories:
        for name in cli_categories:
            if name not in class_map:
                class_map[name] = len(class_map)
        logger.info(
            f"Seeded {len(cli_categories)} class(es) from --categories: {cli_categories}"
        )

    # ---- Resolve effective class definitions ---------------------------------
    # Layers: data.yaml description fields + CLI (-d/--class_definitions),
    # then --preset. Logged so the user can verify what the
    # model actually receives.
    effective_definitions = resolve_auto_label_definitions(args, yaml_data=data)
    if effective_definitions:
        preview = (
            effective_definitions[:300] + "..."
            if len(effective_definitions) > 300
            else effective_definitions
        )
        logger.info(f"Using class definitions:\n{preview}")
    else:
        logger.info("No class definitions provided (prompt will list classes only).")
    logger.info(
        f"Effective known classes ({len(class_map)}): {sorted(class_map, key=class_map.get)}"
    )
    _class_mode = str(getattr(args, "class_mode", "hybrid") or "hybrid").lower()
    if _class_mode == "strict" and not class_map:
        logger.warning(
            "Strict class_mode with an EMPTY known-class list: every box will be "
            "discarded as unknown (empty YOLO files). Provide --categories "
            "and/or --init_class_map, or switch to --class_mode hybrid."
        )

    # ---- Auto-resume: pick the checkpoint back up if one exists ----
    os.makedirs(args.output_folder, exist_ok=True)
    checkpoint = CheckpointManager(args.output_folder)
    completed_images = set()
    batches_done = set()

    if not args.auto_resume:
        logger.info(
            "--no_auto_resume set: ignoring/clearing any existing checkpoint, starting fresh."
        )
        checkpoint.clear()
    else:
        checkpoint_data = checkpoint.load()
        if checkpoint_data:
            completed_images = set(checkpoint_data.get("completed_images", []))
            batches_done = set(checkpoint_data.get("batches_done", []))
            # Merge checkpointed classes into class_map, keeping their original
            # ids so previously-written label files (which already reference
            # those ids) stay valid.
            checkpoint_class_map = checkpoint_data.get("class_map", {})
            for name, idx in sorted(checkpoint_class_map.items(), key=lambda kv: kv[1]):
                if name not in class_map:
                    class_map[name] = idx
            logger.info(
                f"Auto-resume: found checkpoint with {len(completed_images)} completed image(s), "
                f"{len(batches_done)} finished batch(es), and {len(class_map)} known class(es). "
                "Continuing from where the previous run left off."
            )
        else:
            logger.info(
                "Auto-resume: no existing checkpoint found, starting a new run."
            )

    if args.resume and args.inplace_saving:
        logger.warning(
            "--resume has no effect together with --inplace_saving (there's no separate output "
            "file to check); relying on --auto_resume's checkpoint instead."
        )

    client = None
    llama_manager = None
    try:
        client, llama_manager = build_client(args)
    except Exception as e:
        logger.error(f"Failed to initialize server or client: {e}")
        exit(1)

    image_extensions = tuple(
        ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
        for ext in args.image_extensions.split(",")
        if ext.strip()
    )

    stats = RunStats()

    def auto_save():
        save_updated_yaml(yaml_path, args.output_folder, data, class_map)

    # --image_size square override (PipelineConfig already applies it via a
    # validator; the standalone Namespace path via cli.parse_args does too --
    # this is belt-and-braces for any hand-built Namespace).
    _image_size = getattr(args, "image_size", None)
    if _image_size:
        target_height = target_width = int(_image_size)
    else:
        target_height = args.height
        target_width = args.width

    try:
        read_images_with_labels(
            args.train_image,
            args.train_label,
            class_map,
            client,
            args.model,
            args.output_folder,
            conf_threshold=args.conf_threshold,
            num_samples=args.num_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            start_index=args.start_index,
            end_index=args.end_index,
            dry_run=args.dry_run,
            resume=args.resume,
            image_extensions=image_extensions,
            max_workers=args.max_workers,
            target_height=target_height,
            target_width=target_width,
            stats=stats,
            inplace_saving=args.inplace_saving,
            batch_size=args.batch_size,
            checkpoint=checkpoint,
            completed_images=completed_images,
            batches_done=batches_done,
            auto_save=auto_save,
            class_mode=getattr(args, "class_mode", "hybrid"),
            class_definitions=effective_definitions,
            none_labels=getattr(
                args,
                "none_labels",
                "none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
            ),
            drop_none=getattr(args, "drop_none", True),
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred during image processing: {e}")
    finally:
        if llama_manager is not None:
            logger.info("Stopping local llama.cpp server...")
            try:
                llama_manager.stop_llama_server()
            except Exception as e:
                logger.error(f"Error occurred while stopping local server: {e}")

    # Output detailed runtime metrics through configured logger channels
    for line in stats.summary_lines():
        logger.info(line)

    if args.dry_run:
        logger.info(
            "Dry run complete \u2014 no files were written, yaml was not updated."
        )
    else:
        try:
            save_updated_yaml(yaml_path, args.output_folder, data, class_map)

            logger.info(f"Done. Final classes: {class_map}")
        except Exception as e:
            logger.error(f"Failed to save updated dataset yaml file: {e}")


if __name__ == "__main__":
    main()
