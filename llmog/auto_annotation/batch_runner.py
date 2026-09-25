"""Batch orchestration: slice the dataset, drive single-image relabeling,
optionally across a thread pool, and track per-batch checkpoint progress.

The class_map is mutated in place and shared across all images (protected by
a lock when max_workers > 1), so a class discovered on image 3 is available
(and reused) for image 300.
"""

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from auto_annotation.logging_utils import logger
from auto_annotation.stats import RunStats
from auto_annotation.image_io import find_labeled_images, chunk_list
from auto_annotation.server_guard import (
    ServerDownError,
    FailureTracker,
    probe_server,
)
from auto_annotation.single_image import process_one_image
from pathlib import Path


def read_images_with_labels(
    train_image,
    train_label,
    class_map,
    client,
    model_name,
    output_folder,
    conf_threshold=2,
    num_samples=None,
    shuffle=False,
    seed=42,
    start_index=None,
    end_index=None,
    dry_run=False,
    resume=False,
    image_extensions=(".jpg", ".jpeg", ".png"),
    max_workers=1,
    target_height=1024,
    target_width=1024,
    stats=None,
    inplace_saving=False,
    batch_size=0,
    checkpoint=None,
    completed_images=None,
    batches_done=None,
    auto_save=None,
    class_mode: str = "hybrid",
    class_definitions: str = "",
    # Comma-separated string or list of names (YAML --config list form).
    none_labels="none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    drop_none: bool = True,
    failure_tracker=None,
    max_consecutive_failures: int = 20,
    abort_on_server_down: bool = True,
):
    """
    Re-label every bounding box in every image with a model-predicted class.

    class_map is mutated in place and shared across all images (protected by a
    lock when max_workers > 1), so a class discovered on image 3 is available
    (and reused) for image 300.

    Images are processed in fixed-size batches (batch_size). When not saving
    in-place, each batch's relabeled annotations land in their own
    'batch_XXXX' subfolder under output_folder. If a checkpoint says a whole
    batch is already done (batches_done), that batch is skipped without even
    looking at the individual images inside it.

    Server safety: model calls that fail at the *server* level (dead process,
    timeout, 5xx, OOM) are counted on a shared :class:`FailureTracker`. When
    ``max_consecutive_failures`` such failures happen in a row, a
    :class:`ServerDownError` is raised and the run aborts instead of writing
    fake-empty labels for every remaining image. Pass
    ``abort_on_server_down=False`` (``--no_abort_on_server_down``) to restore
    the legacy keep-going behavior -- failed images are then left out of the
    checkpoint so a resumed run retries them.

    start_index / end_index select a [start, end) slice of the image list
    (after --shuffle, before --num_samples), so a large dataset can be split
    across multiple runs/machines by index range.
    """
    if stats is None:
        stats = RunStats()
    if completed_images is None:
        completed_images = set()
    if batches_done is None:
        batches_done = set()
    labels_folder = Path(output_folder) / "labels"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    image_extensions = tuple(ext.lower() for ext in image_extensions)

    # Read the labels folder first, and only keep images whose label file
    # actually has lines to process. --num_samples / --shuffle / the index
    # range are applied AFTER this filter, so a requested slice is filled
    # with real work instead of images that have nothing to classify.
    image_names = find_labeled_images(train_image, train_label, image_extensions)

    if not image_names:
        logger.warning(
            f"No images with non-empty label files found (labels: '{train_label}', "
            f"images: '{train_image}', extensions: {image_extensions})."
        )
        return None

    if shuffle:
        random.seed(seed)
        random.shuffle(image_names)

    if start_index is not None or end_index is not None:
        start = start_index or 0
        end = end_index if end_index is not None else len(image_names)
        if start < 0 or end < start:
            logger.error(
                f"Invalid --start_index/--end_index range ({start}, {end}) "
                f"for {len(image_names)} image(s); ignoring the range."
            )
        else:
            sliced = image_names[start:end]
            logger.info(
                f"Applying index range [{start}, {end}) -> {len(sliced)} of "
                f"{len(image_names)} image(s)."
            )
            image_names = sliced

    if num_samples is not None:
        image_names = image_names[:num_samples]

    stats.images_total = len(image_names)

    batches = chunk_list(image_names, batch_size)
    logger.info(
        f"Processing {len(image_names)} image(s) in {len(batches)} batch(es) "
        f"of up to {batch_size or len(image_names)}"
        + (" [dry run]" if dry_run else "")
        + "."
    )

    class_map_lock = threading.Lock()
    completed_lock = threading.Lock()
    last_img = None

    # Shared circuit breaker against a dead/OOM server. Always created (a
    # caller-supplied tracker wins) so failed batches are never marked done;
    # only the *abort* is gated by abort_on_server_down.
    if failure_tracker is None:
        failure_tracker = FailureTracker(
            max_consecutive_failures=max_consecutive_failures
        )
    if not abort_on_server_down:
        logger.info(
            "--no_abort_on_server_down set: the run will keep going without "
            "a server, but images with server failures are left out of the "
            "checkpoint so a resumed run retries them."
        )

    def _check_server_before_batch(batch_idx: int) -> None:
        """Abort early if the breaker tripped or a probe shows a dead server."""
        if failure_tracker is None or not abort_on_server_down:
            return
        try:
            failure_tracker.check_and_raise(f"batch {batch_idx}")
        except ServerDownError:
            raise
        if failure_tracker.consecutive > 0 and not dry_run:
            # Some server failures were seen but not enough to trip: verify
            # the server is actually alive before burning through a batch.
            if not probe_server(client):
                logger.error(
                    f"Server liveness probe failed before batch {batch_idx} "
                    f"({failure_tracker.consecutive} recent server failure(s)); "
                    "aborting run."
                )
                raise ServerDownError(
                    f"Server liveness probe failed before batch {batch_idx} "
                    f"({failure_tracker.consecutive} recent server failure(s)). "
                    "Aborting so no fake-empty labels are written -- fix the "
                    "server and resume with the same command."
                )

    for batch_idx, batch_images in enumerate(batches):
        if batch_idx in batches_done:
            already = len(batch_images)
            stats.incr("images_skipped_resume", already)
            with stats._lock:
                stats.images_done += already
            logger.info(
                f"Skipping batch {batch_idx} ({already} image(s)) "
                "-- fully completed per checkpoint, auto-resume."
            )
            continue

        if inplace_saving or batch_size <= 0:
            batch_output_folder = labels_folder
        else:
            batch_output_folder = os.path.join(labels_folder, f"batch_{batch_idx:04d}")
            os.makedirs(batch_output_folder, exist_ok=True)

        # Fail fast before burning through a batch when the server is gone.
        _check_server_before_batch(batch_idx)
        failures_before_batch = failure_tracker.total

        logger.info(
            f"--- Batch {batch_idx + 1}/{len(batches)} ({len(batch_images)} image(s)) ---"
        )

        if max_workers <= 1:
            for img_file in batch_images:
                try:
                    img = process_one_image(
                        img_file,
                        train_image,
                        train_label,
                        batch_output_folder,
                        class_map,
                        class_map_lock,
                        client,
                        model_name,
                        conf_threshold,
                        dry_run,
                        resume,
                        target_height,
                        target_width,
                        stats,
                        inplace_saving,
                        checkpoint=checkpoint,
                        completed_images=completed_images,
                        completed_lock=completed_lock,
                        batches_done=batches_done,
                        class_mode=class_mode,
                        class_definitions=class_definitions,
                        none_labels=none_labels,
                        drop_none=drop_none,
                        failure_tracker=failure_tracker,
                        abort_on_server_down=abort_on_server_down,
                    )
                    if img is not None:
                        last_img = img
                    if not dry_run:
                        auto_save()

                except ServerDownError:
                    # Never swallow: abort the whole run immediately so no
                    # further images are (mis)processed without a server.
                    logger.error(
                        f"Aborting run during batch {batch_idx} "
                        f"({img_file}): inference server is down."
                    )
                    raise
                except Exception as e:
                    logger.exception(f"Unexpected error processing {img_file}: {e}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_one_image,
                        img_file,
                        train_image,
                        train_label,
                        batch_output_folder,
                        class_map,
                        class_map_lock,
                        client,
                        model_name,
                        conf_threshold,
                        dry_run,
                        resume,
                        target_height,
                        target_width,
                        stats,
                        inplace_saving,
                        checkpoint,
                        completed_images,
                        completed_lock,
                        batches_done,
                        class_mode,
                        class_definitions,
                        none_labels,
                        drop_none,
                        failure_tracker,
                        abort_on_server_down,
                    ): img_file
                    for img_file in batch_images
                }
                for future in as_completed(futures):
                    img_file = futures[future]
                    try:
                        img = future.result()
                        if img is not None:
                            last_img = img
                    except ServerDownError:
                        # Abort the whole run: cancel work still queued so a
                        # dead server doesn't fake-process the rest of the
                        # batch, then propagate.
                        logger.error(
                            f"Aborting run during batch {batch_idx} "
                            f"({img_file}): inference server is down; "
                            "cancelling the rest of the batch."
                        )
                        for f in futures:
                            f.cancel()
                        raise
                    except Exception as e:
                        logger.exception(
                            f"Processing {img_file} raised an exception: {e}"
                        )
                if not dry_run:
                    auto_save()

        # Whole batch finished (every image in it either processed just now
        # or already marked completed earlier) -> record it so a resumed run
        # can skip this batch's folder entirely without re-checking images.
        # Never mark a batch done when the server breaker tripped or any
        # server-class failure happened inside it: some of its images were
        # never classified, and per-image checkpointing already covers the
        # ones that were, so a resumed run re-enters this batch safely.
        if abort_on_server_down:
            failure_tracker.check_and_raise(f"batch {batch_idx}")
        if not dry_run and checkpoint is not None:
            if failure_tracker.total > failures_before_batch:
                logger.warning(
                    f"Batch {batch_idx}: {failure_tracker.total - failures_before_batch} "
                    "server failure(s) seen -- checkpointing finished images "
                    "but NOT marking the batch done (will be re-entered on resume)."
                )
                with completed_lock:
                    completed_snapshot = set(completed_images)
                with class_map_lock:
                    class_map_snapshot = dict(class_map)
                checkpoint.save(
                    completed_snapshot, class_map_snapshot, set(batches_done)
                )
            else:
                batches_done.add(batch_idx)
                with completed_lock:
                    completed_snapshot = set(completed_images)
                with class_map_lock:
                    class_map_snapshot = dict(class_map)
                checkpoint.save(
                    completed_snapshot, class_map_snapshot, set(batches_done)
                )

    return last_img
