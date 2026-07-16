"""Entry point: parse args, load the dataset yaml, drive the batch runner,
persist the updated yaml, and stop the local server cleanly.
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
from auto_annotation.cli import parse_args


def main():
    args = parse_args()
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    try:
        with open(args.yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read dataset yaml file at {args.yaml_path}: {e}")
        exit(1)

    class_map = (
        load_or_init_class_map(data.get("names", [])) if args.init_class_map else {}
    )

    # ---- Auto-resume: pick the checkpoint back up if one exists ----
    os.makedirs(getattr(args, "output_folder"), exist_ok=True)
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
            target_height=args.height,
            target_width=args.width,
            stats=stats,
            inplace_saving=args.inplace_saving,
            batch_size=args.batch_size,
            checkpoint=checkpoint,
            completed_images=completed_images,
            batches_done=batches_done,
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
            save_updated_yaml(args.yaml_path, args.output_folder, data, class_map)
            logger.info(f"Done. Final classes: {class_map}")
        except Exception as e:
            logger.error(f"Failed to save updated dataset yaml file: {e}")
