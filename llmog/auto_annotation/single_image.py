"""Relabel every box in a single image. Thread-safe w.r.t. class_map and stats."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from free_detection.image_preprocessing import preprocess_custom_resize
from auto_annotation.logging_utils import logger
from auto_annotation.image_io import detect_defect
from auto_annotation.server_guard import (
    ServerDownError,
    is_server_error,
)


def _normalize_label(name: str) -> str:
    """Normalize a model class label for none-like comparison."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _parse_none_labels(none_labels) -> set:
    """Parse comma-separated none-labels into a normalized set."""
    if not none_labels:
        return set()
    if isinstance(none_labels, (list, tuple, set)):
        raw = list(none_labels)
    else:
        raw = str(none_labels).split(",")
    return {_normalize_label(x) for x in raw if str(x).strip()}


def process_one_image(
    img_file,
    train_image,
    train_label,
    output_folder,
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
    checkpoint=None,
    completed_images=None,
    completed_lock=None,
    batches_done=None,
    class_mode: str = "hybrid",
    class_definitions: str = "",
    # Comma-separated string or list of names (YAML --config list form).
    none_labels="none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    drop_none: bool = True,
    failure_tracker=None,
    abort_on_server_down: bool = True,
):
    """Relabel every box in a single image. Thread-safe w.r.t. class_map and stats.

    ``failure_tracker`` (a :class:`FailureTracker`, may be None) is the shared
    circuit breaker against a dead/OOM inference server: server-class model
    errors are recorded on it, successes reset it, and when it trips a
    :class:`ServerDownError` is raised (if ``abort_on_server_down``) so the
    run aborts instead of writing fake-empty labels for every remaining
    image. Images with server failures are never marked completed, so a
    resumed run always retries them.
    """
    img_path = os.path.join(train_image, img_file)
    img_stem = Path(img_file).stem
    label_path = os.path.join(train_label, img_stem + ".txt")

    # Check existence FIRST (before touching the file at all). Previously the
    # code tried to open() the label file before this check, so a genuinely
    # missing label file raised inside the try/except and got miscounted as
    # a "failed read" instead of "no label file" -- and the exists() check
    # below was dead code that could never fire for that case.
    if not os.path.exists(label_path):
        logger.warning(f"Label file not found for {img_file}: {label_path}")
        stats.incr("images_skipped_no_label")
        stats.log_progress(img_file)
        return None

    if inplace_saving:
        label_out_path = Path(train_label) / (img_stem + ".txt")
    else:
        label_out_path = Path(output_folder) / (img_stem + ".txt")

    # Auto-resume: this image was already finished in a previous (interrupted)
    # run, per the checkpoint. This is a stronger guarantee than checking
    # whether the output file merely exists (--resume below), since the
    # checkpoint is only updated *after* a label file is fully written.
    if completed_images is not None and img_stem in completed_images:
        logger.info(
            f"Skipping {img_file} (already completed per checkpoint, auto-resume)."
        )
        stats.incr("images_skipped_resume")
        stats.log_progress(img_file)
        return None

    # --resume (legacy): skip if the output label file already exists.
    # This check is meaningless (and was previously always true) when
    # --inplace_saving is set, because label_out_path == label_path in that
    # mode -- the "output" file is the very input file we just confirmed
    # exists, so every image would be skipped. Auto-resume (via the
    # checkpoint, above) is what actually tracks completion in that mode.
    if resume and not inplace_saving and label_out_path.exists():
        logger.info(f"Skipping {img_file} (already relabeled, --resume).")
        stats.incr("images_skipped_resume")
        stats.log_progress(img_file)
        return None

    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Could not read image {img_path}, skipping.")
        stats.incr("images_failed_read")
        stats.log_progress(img_file)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read label file {label_path}: {e}")
        stats.incr("images_failed_read")
        stats.log_progress(img_file)
        return None

    logger.debug(f"{img_file}: label file has {len(lines)} line(s) -> {label_path}")

    new_label_lines = []
    low_confidence_records = []
    # Boxes whose model call failed at the *server* level (dead process,
    # timeout, 5xx, OOM) as opposed to per-box content problems. If any such
    # failure happened on this image, the result is untrustworthy: the image
    # must NOT be marked completed, and a fully-empty result must NOT be
    # written (that would forge a "no objects" label for an image the server
    # never actually looked at).
    server_failures_this_image = 0

    for line in lines:
        stats.incr("boxes_seen")
        values = line.strip().split()
        if len(values) != 5:
            logger.warning(f"Malformed label line in {label_path}: '{line.strip()}'")
            stats.incr("boxes_malformed_line")
            continue

        _old_cls, x, y, bw, bh = map(float, values)

        # YOLO format is normalized cx/cy/bw/bh in [0, 1]. Anything outside
        # usually means a misformatted file (pixel coords, 0-1000 scale, or
        # xyxy) that would otherwise clamp into a degenerate box and be
        # skipped silently below -- flag it explicitly.
        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 <= bw <= 1.0
            and 0.0 <= bh <= 1.0
        ):
            logger.warning(
                f"Out-of-range normalized coords in {label_path}: "
                f"'{line.strip()}' (expected cx/cy/bw/bh in [0, 1]). "
                "Clamping to image bounds; check the label format."
            )

        x1 = max(0, min(w, round((x - bw / 2) * w)))
        y1 = max(0, min(h, round((y - bh / 2) * h)))
        x2 = max(0, min(w, round((x + bw / 2) * w)))
        y2 = max(0, min(h, round((y + bh / 2) * h)))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid box in {img_file}: {values}")
            continue

        crop_image = img[y1:y2, x1:x2]
        if crop_image.size == 0:
            logger.warning(
                f"Empty crop in {img_file} for box ({x}, {y}, {bw}, {bh}), skipping box."
            )
            stats.incr("boxes_empty_crop")
            continue

        if dry_run:
            logger.info(
                f"[dry run] {img_file}: would classify box at ({x}, {y}, {bw}, {bh})."
            )
            stats.incr("boxes_dry_run")
            continue

        # preprocess_custom_resize works on PIL.Image, not numpy arrays
        pil_crop = Image.fromarray(crop_image)
        try:
            pil_crop, _ = preprocess_custom_resize(
                pil_crop, target_height=target_height, target_width=target_width
            )
            crop_image = np.array(pil_crop)
        except Exception as e:
            logger.error(f"Error resizing crop in {img_file} for box ({x}, {y}): {e}")
            stats.incr("boxes_empty_crop")
            continue

        # class_map is read here for the prompt before we know if this call
        # will add a new class. Take the snapshot under the lock so a
        # concurrent writer (another thread inserting a new class) can't
        # mutate the dict mid-iteration.
        with class_map_lock:
            known_names = list(class_map.keys())

        try:
            result = detect_defect(
                crop_image,
                client,
                model_name,
                known_names,
                class_mode=class_mode,
                class_definitions=class_definitions,
                none_labels=none_labels,
                drop_none=drop_none,
            )
        except Exception as e:
            logger.error(f"Model call failed for {img_file}: {e}")
            stats.incr("boxes_model_call_failed")
            if is_server_error(e):
                # Server-class failure (dead process, timeout, 5xx, OOM):
                # counted locally even without a shared tracker so the
                # image is never marked completed on server trouble.
                server_failures_this_image += 1
                if failure_tracker is not None:
                    tripped = failure_tracker.record_failure()
                    logger.warning(
                        f"Server-class failure {failure_tracker.consecutive} in a row "
                        f"({img_file})."
                    )
                    if tripped and abort_on_server_down:
                        # Abort immediately: no label file, no checkpoint update.
                        # The batch runner catches this and stops the whole run.
                        logger.error(
                            f"Server appears dead/OOM during {img_file}; aborting run."
                        )
                        raise ServerDownError(
                            f"Inference server appears dead/OOM during {img_file} "
                            f"({failure_tracker.consecutive} consecutive server "
                            "failures). Aborting so no fake-empty labels are "
                            "written -- fix the server and resume with the same "
                            "command (auto-resume skips finished images)."
                        ) from e
            continue

        if failure_tracker is not None:
            failure_tracker.record_success()

        if not isinstance(result, dict):
            logger.warning(
                f"Unexpected (non-dict) model response for {img_file}: "
                f"{result!r}, skipping box."
            )
            stats.incr("boxes_bad_response")
            continue

        class_name = result.get("class")
        if not class_name or not isinstance(class_name, str):
            logger.warning(
                f"Invalid or missing class name in response for {img_file}: {result}"
            )
            stats.incr("boxes_bad_response")
            continue
        class_name = class_name.strip().lower()
        if not class_name:
            logger.warning(f"Empty class name in response for {img_file}: {result}")
            stats.incr("boxes_bad_response")
            continue

        # --- None / no-detection handling -----------------------------------
        # If the model says this crop is "none" (or any alias in --none_labels
        # like "no_detection", "background", "unknown"), treat it as an empty
        # prediction: write NO YOLO line for this box. An image whose every
        # box is none-like therefore gets an empty (0-byte) .txt file, which
        # is exactly YOLO's "no objects" format.
        none_set = _parse_none_labels(none_labels)
        if _normalize_label(class_name) in none_set:
            if drop_none:
                logger.info(
                    f"{img_file}: dropping none-like box "
                    f"(class={class_name!r}) -> empty YOLO prediction for this box."
                )
                stats.incr("boxes_dropped_none")
                continue
            # drop_none=False: fall through and keep it as a regular class
            # (legacy behavior).

        raw_confidence = result.get("confidence", 0)
        try:
            confidence = int(raw_confidence)
        except (TypeError, ValueError):
            logger.warning(
                f"Non-numeric confidence {raw_confidence!r} for {img_file}, "
                "defaulting to 0."
            )
            confidence = 0

        with class_map_lock:
            if class_name not in class_map:
                mode_norm = (class_mode or "hybrid").lower().strip()
                if mode_norm == "strict":
                    # In strict mode: skip any class not already in the map
                    logger.warning(
                        f"[strict mode] Model returned unknown class {class_name!r} for {img_file}; "
                        "discarding box (strict mode disallows new classes)."
                    )
                    stats.incr("boxes_bad_response")
                    continue
                # max()+1 (not len()) so resumed maps with gaps/merges never
                # collide with an id already written to finished label files.
                try:
                    from auto_annotation.checkpoint import next_free_id as _next_free_id

                    class_map[class_name] = _next_free_id(class_map)
                except Exception:
                    class_map[class_name] = len(class_map)
                stats.note_new_class(class_name)
            new_cls_id = class_map[class_name]

        new_label_lines.append(f"{new_cls_id} {x} {y} {bw} {bh}")
        stats.incr("boxes_classified")

        if confidence <= conf_threshold:
            stats.incr("boxes_low_confidence")
            low_confidence_records.append(
                {"class": class_name, "confidence": confidence, "bbox": [x, y, bw, bh]}
            )

    if dry_run:
        stats.log_progress(img_file)
        return img

    had_server_failure = server_failures_this_image > 0

    if had_server_failure and not new_label_lines:
        # Every box on this image failed at the server level: writing an
        # empty file here would forge a "no objects" label for an image the
        # server never looked at. Leave disk and checkpoint untouched so a
        # resumed run retries this image from scratch.
        logger.warning(
            f"{img_file}: {server_failures_this_image} server failure(s) and "
            "no boxes classified -> NOT writing a label file and NOT marking "
            "as completed (will be retried on resume)."
        )
        stats.incr("images_failed_server")
        stats.log_progress(img_file)
        if failure_tracker is not None and abort_on_server_down:
            # Another thread may have tripped the breaker meanwhile.
            failure_tracker.check_and_raise(img_file)
        return None

    write_ok = True
    try:
        # Always (over)write the label file: when new_label_lines is empty
        # (e.g. every box was none-like and dropped) this intentionally
        # produces an empty (0-byte) .txt file = YOLO empty prediction.
        with open(label_out_path, "w") as f:
            if new_label_lines:
                f.write("\n".join(new_label_lines) + "\n")
            else:
                logger.info(
                    f"{img_file}: no boxes to write "
                    "(all dropped/empty) -> writing empty YOLO label file."
                )
    except Exception as e:
        write_ok = False
        logger.error(f"Failed to write relabeled annotations to {label_out_path}: {e}")

    if low_confidence_records:
        debug_out_path = Path(output_folder) / (img_stem + "_low_confidence.json")
        try:
            with open(debug_out_path, "w") as f:
                json.dump(low_confidence_records, f, indent=4)
        except Exception as e:
            logger.error(
                f"Failed to write low confidence records to {debug_out_path}: {e}"
            )

    # Only mark the image "done" in the checkpoint once its label file has
    # actually landed on disk. This is what lets a killed/crashed run resume
    # exactly at the first unfinished image instead of redoing work or
    # silently losing an image that never got written.
    #
    # Additionally, an image that suffered ANY server-class failure is never
    # marked completed, even if a partial label file was written: some of its
    # boxes were never classified, so a resumed run must retry it (and
    # overwrite the partial file with the full result).
    if had_server_failure:
        logger.warning(
            f"{img_file}: label write finished with "
            f"{server_failures_this_image} server failure(s); partial result "
            "kept on disk but NOT marking as completed in checkpoint "
            "(will be retried on resume)."
        )
        stats.incr("images_failed_server")
        stats.log_progress(img_file)
        if failure_tracker is not None and abort_on_server_down:
            failure_tracker.check_and_raise(img_file)
        return img

    if write_ok and checkpoint is not None and completed_images is not None:
        with completed_lock:
            completed_images.add(img_stem)
            completed_snapshot = set(completed_images)
        with class_map_lock:
            class_map_snapshot = dict(class_map)
        batches_snapshot = set(batches_done) if batches_done is not None else set()
        checkpoint.save(completed_snapshot, class_map_snapshot, batches_snapshot)
    elif not write_ok:
        # Don't silently mark progress for an image whose label file failed
        # to write -- otherwise a resumed run would skip it forever even
        # though nothing was actually persisted.
        logger.warning(
            f"{img_file}: label write failed, NOT marking as completed in checkpoint."
        )

    stats.log_progress(img_file)
    return img
