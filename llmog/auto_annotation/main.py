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
from auto_annotation.server_guard import ServerDownError
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

    # ---- Normalize yaml names (ordered list + yaml-side ids) -------------------
    # Needed both for the fresh-run seed and for the resume consistency check
    # against the checkpoint (checkpoint ids always win on conflict).
    yaml_names = data.get("names", []) or []
    if isinstance(yaml_names, dict):
        yaml_names = [yaml_names[k] for k in sorted(yaml_names, key=int)]
    yaml_names = [str(n) for n in list(yaml_names)]
    yaml_id_by_name = load_or_init_class_map(data.get("names", []))
    if yaml_names:
        logger.info(f"data.yaml contains {len(yaml_names)} name(s): {yaml_names}")
    else:
        logger.info("data.yaml contains no names.")

    cli_categories = parse_categories_list(getattr(args, "categories", ""))

    # ---- Load checkpoint FIRST so resume is checkpoint-authoritative ---------
    # The checkpoint's class_map preserves the exact ids already burned into
    # previously-written label files. On resume it is the source of truth:
    # yaml/--categories can only ADD brand-new names (at max(id)+1), never
    # reassign an existing id (which would corrupt the finished labels).
    os.makedirs(args.output_folder, exist_ok=True)
    checkpoint = CheckpointManager(args.output_folder)
    completed_images = set()
    batches_done = set()
    checkpoint_data = None
    if not args.auto_resume:
        logger.info(
            "--no_auto_resume set: ignoring/clearing any existing checkpoint, starting fresh."
        )
        checkpoint.clear()
    else:
        checkpoint_data = checkpoint.load()
        if checkpoint_data is None:
            logger.info(
                "Auto-resume: no existing checkpoint found, starting a new run."
            )

    if checkpoint_data:
        from auto_annotation.checkpoint import next_free_id as _next_free_id

        checkpoint_class_map = {
            str(k): int(v)
            for k, v in dict(checkpoint_data.get("class_map", {})).items()
        }
        completed_images = set(checkpoint_data.get("completed_images", []))
        batches_done = set(checkpoint_data.get("batches_done", []))
        # Start from the checkpoint verbatim -- ids stay exactly as written.
        class_map = dict(sorted(checkpoint_class_map.items(), key=lambda kv: kv[1]))
        # ---- Consistency check: data.yaml vs checkpoint ----------------------
        for name in yaml_names:
            if name not in class_map:
                logger.warning(
                    f"Resume check: data.yaml class {name!r} (yaml id "
                    f"{yaml_id_by_name.get(name, '?')}) is NOT in the checkpoint; "
                    "it will be appended as a NEW class so finished labels keep "
                    "their ids."
                )
        for name, idx in checkpoint_class_map.items():
            if name not in yaml_id_by_name:
                logger.warning(
                    f"Resume check: checkpoint class {name!r} (id {idx}) is missing "
                    f"from {yaml_path}; data.yaml will be re-synced from the "
                    "checkpoint so the run does not start from scratch."
                )
            elif yaml_id_by_name[name] != idx:
                logger.warning(
                    f"Resume check: class {name!r} has yaml id "
                    f"{yaml_id_by_name[name]} but checkpoint id {idx}. "
                    "CHECKPOINT WINS (finished label files already use it); "
                    "data.yaml will be re-synced. Continuing -- no labels harmed."
                )
        if cli_categories:
            known = [n for n in cli_categories if n in class_map]
            if known:
                logger.info(
                    f"Resume check: {len(known)} --categories name(s) already known "
                    f"from checkpoint (ids kept): {known}"
                )
        # Merge yaml + CLI names as new-only additions (append-only, max+1).
        for name in list(yaml_names) + cli_categories:
            if name not in class_map:
                class_map[name] = _next_free_id(class_map)
                logger.info(
                    f"Resume check: appended new class {name!r} "
                    f"as id {class_map[name]} (from "
                    f"{'data.yaml' if name in yaml_names else '--categories'})."
                )
        logger.info(
            f"Auto-resume: found checkpoint with {len(completed_images)} completed image(s), "
            f"{len(batches_done)} finished batch(es), and {len(class_map)} known class(es). "
            "Continuing from where the previous run left off (classes from checkpoint, "
            "NOT from scratch)."
        )
        # Re-sync data.yaml files from the checkpoint-backed map so the prompt
        # and the next resume see the same classes from image 1. Append-only:
        # never shrinks/renumbers checkpoint ids (see yaml_utils guard).
        try:
            save_updated_yaml(yaml_path, args.output_folder, data, class_map)
            data = dict(data)
            data["names"] = [
                name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])
            ]
            data["nc"] = len(class_map)
        except Exception as e:
            logger.error(f"Resume check: failed to re-sync data.yaml: {e}")
    else:
        from auto_annotation.checkpoint import next_free_id as _next_free_id

        # ---- Fresh run: yaml (--init_class_map) + --categories ----------------
        class_map = (
            load_or_init_class_map(data.get("names", [])) if args.init_class_map else {}
        )
        if yaml_names and not args.init_class_map:
            logger.warning(
                f"data.yaml contains {len(yaml_names)} name(s) {yaml_names} but "
                "--init_class_map was NOT passed, so they are ignored and the run "
                "starts from an empty class map (pass --init_class_map to reuse them, "
                "or --categories to seed a list explicitly)."
            )
        if cli_categories:
            for name in cli_categories:
                if name not in class_map:
                    class_map[name] = _next_free_id(class_map)
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
            max_consecutive_failures=getattr(args, "max_consecutive_failures", 20),
            abort_on_server_down=getattr(args, "abort_on_server_down", True),
        )
    except ServerDownError as e:
        # The inference server died/OOMed mid-run. Progress up to the failure
        # is already checkpointed (failed images were NOT marked completed),
        # so just save the yaml and exit non-zero: schedulers/Kaggle must see
        # this as a failure, not a successful run, and a resume with the same
        # command will retry exactly the unfinished images.
        logger.error(f"Run aborted: {e}")
        try:
            save_updated_yaml(yaml_path, args.output_folder, data, class_map)
        except Exception as save_e:
            logger.error(f"Failed to save updated dataset yaml file: {save_e}")
        for line in stats.summary_lines():
            logger.info(line)
        exit(1)
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
        # Flatten staged batches/batch_XXXX/ labels to the top level of
        # <output>/labels/ so the output is directly YOLO-trainable.
        # labels/ is populated ONLY here -- never during batching -- so it is
        # always final output, never half-finished staging state.
        # Copy-only (staging folders and checkpoint untouched); runs only
        # after ALL batches finished, never on abort paths above.
        # Disable with --no_flatten.
        if getattr(args, "flatten", True) and not args.inplace_saving:
            try:
                from auto_annotation.reverse_batches import (
                    flatten_batches_to_labels,
                )

                flatten_batches_to_labels(args.output_folder)
            except Exception as e:
                logger.error(f"Failed to flatten batch labels: {e}")


if __name__ == "__main__":
    main()
