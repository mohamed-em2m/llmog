"""Relabel every box in a single image. Thread-safe w.r.t. class_map and stats."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from free_detection.image_preprocessing import preprocess_custom_resize
from auto_annotation.logging_utils import logger
from auto_annotation.image_io import (
    build_esr_settings,
    detect_defect,
    draw_som_context,
    dump_vlm_crop,
    maybe_esr_upscale_pil,
    pad_box,
    resize_crop_ratio,
    upscale_scene_for_som,
)
from auto_annotation.yaml_utils import resolve_kept_class
from auto_annotation.checkpoint import build_run_settings
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
    extra_body=None,
    min_box_size: int = 0,
    small_box_action: str = "keep",
    drop_small_images: bool = True,
    crop_padding_pct: float = 0.0,
    recls_context: str = "crop",
    crop_resize_ratio=None,
    esr_settings=None,
    dump_vlm_crops=None,
    # Ordered data.yaml names (list) or id->name mapping, used ONLY to
    # recover a kept box's original label name (true keep, no model call).
    # None (tests/legacy callers) keeps the previous synthetic-name behavior.
    orig_names=None,
    # Shared stem -> {"reason": str} map of failed images, guarded by
    # completed_lock (same convention as completed_images). A resumed run
    # ALWAYS retries these stems -- even when a stale output file exists
    # (the legacy --resume file check is bypassed for them). Entries are
    # cleared the moment the image completes. None disables tracking
    # (tests/legacy callers).
    failed_images=None,
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

    def _failed_contains():
        """Whether this stem is a recorded failure (lock-guarded read)."""
        if failed_images is None:
            return False
        if completed_lock is not None:
            with completed_lock:
                return img_stem in failed_images
        return img_stem in failed_images

    def _persist_failed_state(run_settings=None):
        """Persist progress incl. the shared failed set (no-op on dry run).

        Thread-safety comes from save_under_locks (snapshots under the
        fixed lock order); callers must NOT hold locks when calling.
        """
        if checkpoint is None or dry_run:
            return
        checkpoint.save_under_locks(
            completed_images,
            completed_lock,
            class_map,
            class_map_lock,
            batches_done,
            run_settings,
            failed_images,
        )

    def _record_failure(reason):
        """Mark this image failed (retried on resume) and persist.

        No-op when tracking is disabled (failed_images=None, tests/legacy
        callers) or on dry runs: those paths keep the historical
        write-nothing-on-failure behavior exactly.
        """
        if failed_images is None or dry_run:
            return
        if completed_lock is not None:
            with completed_lock:
                failed_images[img_stem] = {"reason": str(reason)[:300]}
        else:
            failed_images[img_stem] = {"reason": str(reason)[:300]}
        logger.warning(
            f"{img_file}: recorded as failed ({reason}) -- will be retried on resume."
        )
        _persist_failed_state()

    def _clear_failure():
        """Drop this image from the failed set (it completed)."""
        if failed_images is None:
            return
        if completed_lock is not None:
            with completed_lock:
                failed_images.pop(img_stem, None)
        else:
            failed_images.pop(img_stem, None)

    # Check existence FIRST (before touching the file at all). Previously the
    # code tried to open() the label file before this check, so a genuinely
    # missing label file raised inside the try/except and got miscounted as
    # a "failed read" instead of "no label file" -- and the exists() check
    # below was dead code that could never fire for that case.
    if not os.path.exists(label_path):
        logger.warning(f"Label file not found for {img_file}: {label_path}")
        stats.incr("images_skipped_no_label")
        stats.log_progress(img_file)
        _record_failure(f"label file not found: {label_path}")
        return None

    if inplace_saving:
        label_out_path = Path(train_label) / (img_stem + ".txt")
    else:
        label_out_path = Path(output_folder) / (img_stem + ".txt")

    # Auto-resume: this image was already finished in a previous (interrupted)
    # run, per the checkpoint. This is a stronger guarantee than checking
    # whether the output file merely exists (--resume below), since the
    # checkpoint is only updated *after* a label file is fully written.
    # The membership read takes the lock when available: concurrent workers
    # mutate this set, and an unlocked read only risks duplicate work, but
    # the lock is free here.
    if completed_lock is not None:
        with completed_lock:
            _already_done = (
                completed_images is not None and img_stem in completed_images
            )
    else:
        _already_done = completed_images is not None and img_stem in completed_images
    if _already_done:
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
    # Recorded failures ALWAYS bypass this skip: a stale/partial output file
    # from a failed run must never count as done.
    if resume and not inplace_saving and label_out_path.exists():
        if _failed_contains():
            logger.info(
                f"Retrying {img_file} (failed on a previous run; ignoring "
                "the stale output file)."
            )
        else:
            logger.info(f"Skipping {img_file} (already relabeled, --resume).")
            stats.incr("images_skipped_resume")
            stats.log_progress(img_file)
            return None

    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Could not read image {img_path}, skipping.")
        stats.incr("images_failed_read")
        stats.log_progress(img_file)
        _record_failure(f"could not read image {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Real-ESRGAN settings for VLM-bound pixels ({} when disabled; the
    # helper centralizes getattr defaults so sync == batch by construction).
    # Never loads a model on dry runs (no-op runs stay model-free).
    _esr = build_esr_settings(esr_settings)
    if _esr and not dry_run:
        logger.info(
            "ESR upscaling ON for VLM pixels "
            f"(model={_esr.get('esr_model')}, target_long_edge="
            f"{_esr.get('esr_target_long_edge')}). Final YOLO coords stay in "
            "original-image space."
        )

    # Stage 0 (ESR-then-crop): super-resolve the WHOLE image ONCE, then cut
    # every box crop out of the upscaled image. `_fx`/`_fy` map working
    # pixels back to original pixels; the small-box filter below always
    # measures ORIGINAL pixels (working dims divided by f). Output YOLO
    # coords use the original normalized label values, untouched.
    _fx = _fy = 1.0
    _whole_applied = False
    if _esr and not dry_run:
        _full, _info = maybe_esr_upscale_pil(
            Image.fromarray(img), _esr, purpose="scene"
        )
        if _info.get("applied"):
            _fx = _info["work_w"] / max(1, _info["orig_w"])
            _fy = _info["work_h"] / max(1, _info["orig_h"])
            img = np.array(_full)
            h, w, _ = img.shape
            _whole_applied = True
            logger.info(
                f"{img_file}: ESR whole-image "
                f"{_info['orig_w']}x{_info['orig_h']} -> working "
                f"{_info['work_w']}x{_info['work_h']} "
                f"(x{_info.get('scale', 1)}); boxes projected "
                f"x{_fx:.2f}/x{_fy:.2f}; crops cut from the upscaled image."
            )

    # Context padding for the VLM crop (box-relative %, clamped to the image;
    # 0 = legacy exact-box crop). Applied AFTER the small-box filter, which
    # always measures the original box, and never touches output YOLO coords.
    try:
        _crop_pad = float(crop_padding_pct or 0.0)
    except (TypeError, ValueError):
        _crop_pad = 0.0
    _som_mode = str(recls_context or "crop").lower().strip() == "full_som"
    if _som_mode:
        logger.info(
            f"{img_file}: reclassification context=full_som -- sending the "
            "full scene with the box highlighted instead of the crop."
        )
    if _crop_pad > 0:
        logger.info(
            f"{img_file}: crop padding={_crop_pad}% of box dims per side "
            "(clamped to image) before the VLM crop."
        )
    try:
        _ratio = float(crop_resize_ratio) if crop_resize_ratio is not None else None
        if _ratio is not None and _ratio <= 0:
            _ratio = None
    except (TypeError, ValueError):
        _ratio = None
    if _ratio is not None and not _som_mode:
        logger.info(
            f"{img_file}: crop resize ratio={_ratio} (long edge capped at "
            f"{max(target_height, target_width)}px) instead of the fixed "
            f"{target_width}x{target_height} letterbox."
        )
    # Real-ESRGAN settings for VLM-bound pixels ({} when disabled; the
    # helper centralizes getattr defaults so sync == batch by construction).
    # Never loads a model on dry runs (no-op runs stay model-free).
    # NOTE: _esr was already built above (stage 0 needs it right after load).
    _dump_dir = str(dump_vlm_crops) if dump_vlm_crops else None
    if _dump_dir:
        # Debug dumps run on dry runs too: they cost no model calls and no
        # label writes, and previewing the payload is the point.
        logger.info(f"{img_file}: dumping final VLM crops to {_dump_dir}.")
    # Fingerprint of the label-affecting settings, stored in the checkpoint
    # with every save so a resume with changed flags warns (main.py) instead
    # of silently mixing label vintages.
    _run_settings = build_run_settings(
        crop_padding_pct=_crop_pad,
        recls_context=recls_context,
        crop_resize_ratio=crop_resize_ratio,
        min_box_size=min_box_size,
        small_box_action=small_box_action,
        model=model_name,
        height=target_height,
        width=target_width,
        class_mode=class_mode,
        none_labels=none_labels,
        drop_none=drop_none,
        esr_enabled=bool(_esr),
        esr_model=_esr.get("esr_model") if _esr else None,
        esr_model_path=_esr.get("esr_model_path") if _esr else None,
        esr_scale=_esr.get("esr_scale") if _esr else None,
        esr_target_long_edge=_esr.get("esr_target_long_edge") if _esr else None,
        esr_for_crops=_esr.get("esr_for_crops") if _esr else None,
    )

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read label file {label_path}: {e}")
        stats.incr("images_failed_read")
        stats.log_progress(img_file)
        _record_failure(f"failed to read label file {label_path}: {e}")
        return None

    logger.debug(f"{img_file}: label file has {len(lines)} line(s) -> {label_path}")

    new_label_lines = []
    low_confidence_records = []
    # Boxes removed by the small-box filter on THIS image (never sent to the
    # LLM). Used by the drop_small_images guard below to tell "genuinely
    # empty" apart from "emptied by the size filter".
    small_skipped_this_image = 0
    # Boxes whose model call failed at the *server* level (dead process,
    # timeout, 5xx, OOM) as opposed to per-box content problems. If any such
    # failure happened on this image, the result is untrustworthy: the image
    # must NOT be marked completed, and a fully-empty result must NOT be
    # written (that would forge a "no objects" label for an image the server
    # never actually looked at).
    server_failures_this_image = 0
    # Boxes whose model call failed for NON-server reasons (auth, bad model
    # name, 4xx validation, unparseable/empty responses). Like server
    # failures, these must never forge an empty "no objects" label: if no
    # box on the image produced a usable line, the image is left
    # un-checkpointed so a resume retries it. Policy outcomes (none-drop,
    # strict-discard) do NOT count here -- those are deterministic model
    # decisions, not errors.
    failed_boxes_this_image = 0

    for line_no, line in enumerate(lines):
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

        # --- Small-box filter (measured on ORIGINAL image pixels) -----------
        # Tiny boxes produce crops the VLM cannot classify reliably, so they
        # are never sent to the model. "keep" preserves the original YOLO
        # line verbatim; "drop" omits the box from the output entirely.
        # Under whole-image ESR, x1..y2 above are WORKING pixels: divide by
        # the stage-0 factors to recover original-pixel dims for the test.
        try:
            _min_side = int(min_box_size or 0)
        except (TypeError, ValueError):
            _min_side = 0
        _action = str(small_box_action or "keep").lower().strip()
        if _action not in ("keep", "drop"):
            _action = "keep"
        _ow, _oh = (x2 - x1) / _fx, (y2 - y1) / _fy
        if _min_side > 0 and (_ow < _min_side or _oh < _min_side):
            stats.incr("boxes_skipped_small")
            small_skipped_this_image += 1
            # INFO (not DEBUG): the user needs to see the filter firing to trust
            # it, and the box dimensions to calibrate min_box_size.
            logger.info(
                f"{img_file}: small box filtered "
                f"({int(round(_ow))}x{int(round(_oh))}px < "
                f"min_box_size={_min_side}px, action={_action})."
            )
            if dry_run:
                logger.info(
                    f"[dry run] {img_file}: would skip small box "
                    f"({int(round(_ow))}x{int(round(_oh))}px < "
                    f"min_box_size={_min_side}px, action={_action})."
                )
                continue
            if _action == "keep":
                # Kept boxes are never classified: their class must not
                # silently merge into an unrelated map entry (e.g. an old
                # binary id reinterpreted under a new multi-class map).
                # With orig_names (data.yaml), a box whose original name
                # still holds its id is kept verbatim ("original"); free ids
                # slot in place, taken ids quarantine once via
                # resolve_kept_class; the written line uses the RESOLVED id
                # (coords always verbatim).
                with class_map_lock:
                    _kept_name, _kept_id, _kept_how = resolve_kept_class(
                        class_map, values[0], orig_names=orig_names
                    )
                if _kept_how == "invalid":
                    logger.warning(
                        f"{img_file}: kept small box has non-numeric class "
                        f"{values[0]!r}; writing the line verbatim."
                    )
                    new_label_lines.append(line.strip())
                else:
                    # Rewrite only the class token; coordinate text stays
                    # byte-identical to the input line.
                    new_label_lines.append(
                        " ".join([str(_kept_id)] + [v for v in values[1:]])
                    )
                stats.incr("boxes_kept_small")
                if _kept_how == "remapped":
                    logger.warning(
                        f"{img_file}: kept small box original id {values[0]} is "
                        f"taken in the current map -- re-registered as id "
                        f"{_kept_id} ('{_kept_name}') instead of merging into "
                        "an unrelated class."
                    )
                    stats.note_new_class(_kept_name)
                elif _kept_how == "slot":
                    logger.warning(
                        f"{img_file}: kept small box references class id "
                        f"{values[0]} missing from the class map; registered "
                        f"as {_kept_name!r} so data.yaml stays trainable."
                    )
                    stats.note_new_class(_kept_name)
                elif _kept_how in ("reused", "original"):
                    logger.info(
                        f"{img_file}: keeping small box "
                        f"({int(round(_ow))}x{int(round(_oh))}px) as id {_kept_id} "
                        f"('{_kept_name}') without LLM call."
                    )
            else:
                stats.incr("boxes_dropped_small")
                logger.debug(
                    f"{img_file}: dropping small box "
                    f"({int(round(_ow))}x{int(round(_oh))}px < {_min_side}px)."
                )
            continue

        if _som_mode:
            # Full-scene context: highlight the box (padding is a crop-mode
            # concept) and send the whole annotated image, fitted to the
            # target size so batch payloads stay bounded. Under whole-image
            # ESR the scene is ALREADY super-resolved (coords already in
            # working pixels), so it is marked directly; otherwise the
            # scene is super-resolved first with the box scaled by the TRUE
            # working/original ratio (never assumed) before marking.
            try:
                _scene = Image.fromarray(img)
                if _esr and not dry_run and not _whole_applied:
                    _scene, (x1, y1, x2, y2), _ = upscale_scene_for_som(
                        _scene, (x1, y1, x2, y2), _esr
                    )
                som_view = draw_som_context(_scene, x1, y1, x2, y2)
                som_view, _ = preprocess_custom_resize(
                    som_view,
                    target_height=target_height,
                    target_width=target_width,
                )
                crop_image = np.array(som_view)
            except Exception as e:
                logger.error(
                    f"Error building SoM context in {img_file} for box ({x}, {y}): {e}"
                )
                stats.incr("boxes_empty_crop")
                continue
        else:
            if _crop_pad > 0:
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, w, h, _crop_pad)
            crop_image = img[y1:y2, x1:x2]
            if crop_image.size == 0:
                logger.warning(
                    f"Empty crop in {img_file} for box ({x}, {y}, {bw}, {bh}), skipping box."
                )
                stats.incr("boxes_empty_crop")
                continue

            # preprocess_custom_resize works on PIL.Image, not numpy arrays.
            # The crop is cut from the (possibly whole-image-upscaled) img
            # above, so its pixels already carry ESR detail. A per-crop ESR
            # pass runs ONLY when the whole-image stage did not (opt-in
            # esr_for_crops without stage 0 has no other path); never both.
            pil_crop = Image.fromarray(crop_image)
            try:
                if _esr and not dry_run and not _whole_applied:
                    pil_crop, _ = maybe_esr_upscale_pil(
                        pil_crop,
                        _esr,
                        purpose="crop",
                        long_edge_cap=max(target_height, target_width),
                    )
                if _ratio is not None:
                    pil_crop = resize_crop_ratio(
                        pil_crop,
                        _ratio,
                        max_long_edge=max(target_height, target_width),
                    )
                else:
                    pil_crop, _ = preprocess_custom_resize(
                        pil_crop,
                        target_height=target_height,
                        target_width=target_width,
                    )
                crop_image = np.array(pil_crop)
            except Exception as e:
                logger.error(
                    f"Error resizing crop in {img_file} for box ({x}, {y}): {e}"
                )
                stats.incr("boxes_empty_crop")
                continue

        if _dump_dir:
            dump_vlm_crop(crop_image, _dump_dir, img_stem, line_no)

        if dry_run:
            logger.info(
                f"[dry run] {img_file}: would classify box at ({x}, {y}, {bw}, {bh})."
            )
            stats.incr("boxes_dry_run")
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
                extra_body=extra_body,
                region_context="full_som" if _som_mode else "crop",
            )
        except Exception as e:
            logger.error(f"Model call failed for {img_file}: {e}")
            stats.incr("boxes_model_call_failed")
            # Server-class failure (dead process, timeout, 5xx, OOM):
            # counted locally even without a shared tracker so the
            # image is never marked completed on server trouble.
            if is_server_error(e):
                server_failures_this_image += 1
                if failure_tracker is not None:
                    tripped = failure_tracker.record_failure()
                    logger.warning(
                        f"Server-class failure {failure_tracker.consecutive} in a row "
                        f"({img_file})."
                    )
                    if tripped and abort_on_server_down:
                        # Abort immediately: no label file, no checkpoint
                        # update. Record the failure FIRST so a resumed run
                        # retries this image (the batch runner catches the
                        # raise below and stops the whole run).
                        _record_failure(
                            "inference server dead/OOM during "
                            f"{img_file} -- run aborted"
                        )
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
            else:
                # Non-server failure (auth, bad model name, 4xx validation):
                # counts toward the no-usable-lines guard below.
                failed_boxes_this_image += 1
            continue

        if failure_tracker is not None:
            failure_tracker.record_success()

        if not isinstance(result, dict):
            logger.warning(
                f"Unexpected (non-dict) model response for {img_file}: "
                f"{result!r}, skipping box."
            )
            stats.incr("boxes_bad_response")
            failed_boxes_this_image += 1
            continue

        class_name = result.get("class")
        if not class_name or not isinstance(class_name, str):
            logger.warning(
                f"Invalid or missing class name in response for {img_file}: {result}"
            )
            stats.incr("boxes_bad_response")
            failed_boxes_this_image += 1
            continue
        class_name = class_name.strip().lower()
        if not class_name:
            logger.warning(f"Empty class name in response for {img_file}: {result}")
            stats.incr("boxes_bad_response")
            failed_boxes_this_image += 1
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
        _record_failure(
            f"{server_failures_this_image} server failure(s), nothing classified"
        )
        if failure_tracker is not None and abort_on_server_down:
            # Another thread may have tripped the breaker meanwhile.
            failure_tracker.check_and_raise(img_file)
        return None

    if failed_boxes_this_image > 0 and not new_label_lines:
        # Every box on this image failed for NON-server reasons (auth, bad
        # model name, 4xx validation, unusable responses): writing an empty
        # file here would forge a "no objects" label for an image whose
        # boxes were never classified (e.g. one wrong flag wipes the whole
        # dataset). Leave disk and checkpoint untouched so a resumed run
        # retries this image from scratch. Policy outcomes (none-drop,
        # strict-discard) don't land here -- those are deterministic model
        # decisions, and kept small boxes count as usable lines above.
        logger.warning(
            f"{img_file}: {failed_boxes_this_image} box(es) failed with model "
            "errors and no boxes classified -> NOT writing a label file and "
            "NOT marking as completed (fix the cause and resume to retry)."
        )
        stats.incr("images_failed_unclassified")
        stats.log_progress(img_file)
        _record_failure(f"{failed_boxes_this_image} box(es) failed with model errors")
        return None

    if not new_label_lines and small_skipped_this_image > 0 and drop_small_images:
        # Every writable box was removed by the small-box filter: writing an
        # empty .txt here would teach the detector "no objects" for an image
        # that DOES contain (tiny) defects -- a baked-in false negative. So
        # write nothing at all (inplace mode: original labels left untouched)
        # and record the stem in skipped_small_images.txt so the image can be
        # excluded from the training set. Still marked completed: re-running
        # with the same flags would deterministically repeat this, so resume
        # must not retry it -- re-run with a smaller --min_box_size (and
        # --no_auto_resume) to reconsider these images.
        manifest_path = Path(output_folder) / "skipped_small_images.txt"
        try:
            if completed_lock is not None:
                with completed_lock:
                    with open(manifest_path, "a") as mf:
                        mf.write(img_stem + "\n")
            else:
                with open(manifest_path, "a") as mf:
                    mf.write(img_stem + "\n")
        except Exception as e:
            logger.error(
                f"Failed to append to small-image manifest {manifest_path}: {e}"
            )
        logger.warning(
            f"{img_file}: all {small_skipped_this_image} box(es) removed by "
            f"the small-box filter (min_box_size={_min_side}px) -> NOT writing a label file "
            f"(no false-negative empty label; listed in {manifest_path}). "
            "Exclude this image from training or re-run with a smaller --min_box_size."
        )
        stats.incr("images_skipped_all_small")
        if checkpoint is not None and completed_images is not None:
            if completed_lock is not None:
                with completed_lock:
                    completed_images.add(img_stem)
            else:
                completed_images.add(img_stem)
            _clear_failure()
            # Snapshot + write atomically (see save_under_locks): a bare
            # save() here could overwrite a newer worker's progress.
            checkpoint.save_under_locks(
                completed_images,
                completed_lock,
                class_map,
                class_map_lock,
                batches_done,
                _run_settings,
                failed_images,
            )
        stats.log_progress(img_file)
        return img

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
        _record_failure(
            f"{server_failures_this_image} server failure(s); partial result kept"
        )
        if failure_tracker is not None and abort_on_server_down:
            failure_tracker.check_and_raise(img_file)
        return img

    if write_ok and checkpoint is not None and completed_images is not None:
        if completed_lock is not None:
            with completed_lock:
                completed_images.add(img_stem)
        else:
            completed_images.add(img_stem)
        _clear_failure()
        checkpoint.save_under_locks(
            completed_images,
            completed_lock,
            class_map,
            class_map_lock,
            batches_done,
            _run_settings,
            failed_images,
        )
    elif not write_ok:
        # Don't silently mark progress for an image whose label file failed
        # to write -- otherwise a resumed run would skip it forever even
        # though nothing was actually persisted.
        logger.warning(
            f"{img_file}: label write failed, NOT marking as completed in checkpoint."
        )
        _record_failure("label file write failed")

    stats.log_progress(img_file)
    return img
