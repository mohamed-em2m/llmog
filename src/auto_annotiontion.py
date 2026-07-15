"""
Relabel binary defect / no-defect YOLO annotations into multi-class defect
labels (e.g. "cut", "spot", ...) using a vision-language model.

Supports two backends:
  --use_llama_model            -> spins up a local llama.cpp server (LlamaServerManager)
  (default, no flag)           -> talks to an external OpenAI-compatible API via
                                   --base_url / --api_key

New class names discovered by the model are accumulated across the whole
dataset (not reset per-image) and written back into --yaml_path at the end.

Batching / auto-resume
-----------------------
Images are split into fixed-size batches (--batch_size). When not using
--inplace_saving, each batch gets its own subfolder under --output-folder
(batch_0000/, batch_0001/, ...), so a run's output is naturally chunked and
it's easy to see how far a run got just by looking at the folders on disk.

A checkpoint file (<output-folder>/.checkpoint.json) is written after every
image is classified. It records which images are fully done and the current
class_map. If the process is killed (crash, OOM, ctrl-C, machine reboot...),
simply re-running the exact same command will pick the checkpoint back up
automatically (--auto_resume is on by default) and continue from the first
unfinished image, without re-spending API calls on work already done. Use
--no_auto_resume to force a from-scratch run.

Selecting a subset of the dataset
----------------------------------
--start_index / --end_index select a [start, end) slice of the (optionally
shuffled) image list, e.g. so a dataset can be split across several machines
or runs. This is applied before --num_samples, and it plays nicely with
resume: the checkpoint is keyed by image stem, so overlapping ranges won't
double-count work but non-overlapping ranges can safely share one
--output-folder if you want a single combined checkpoint/class_map.
"""

import argparse
import base64
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.error
import urllib.request
import cv2
import json_repair
import numpy as np
import yaml
from openai import OpenAI
from PIL import Image

from llama_server_manager import LlamaServerManager
from image_preprocessing import preprocess_custom_resize

logger = logging.getLogger(__name__)


def setup_logging(log_level="DEBUG", log_file=None):
    """
    Configure logging once, at process start. Always logs to the console;
    additionally logs to --log_file if one is given, so a long unattended
    run can be tailed / grepped / diffed after the fact.
    """
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicate handlers if called more than once

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)
        logger.info(f"Logging to console and to file: {log_file}")

    # Quiet down noisy third-party loggers (httpx/openai log every request at INFO).
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))


class RunStats:
    """Thread-safe counters so a concurrent run can print an accurate summary."""

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.monotonic()
        self.images_total = 0
        self.images_done = 0
        self.images_skipped_no_label = 0
        self.images_skipped_resume = 0
        self.images_failed_read = 0
        self.boxes_seen = 0
        self.boxes_malformed_line = 0
        self.boxes_empty_crop = 0
        self.boxes_dry_run = 0
        self.boxes_model_call_failed = 0
        self.boxes_bad_response = 0
        self.boxes_classified = 0
        self.boxes_low_confidence = 0
        self.classes_discovered = []

    def incr(self, field, n=1):
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def note_new_class(self, class_name):
        with self._lock:
            self.classes_discovered.append(class_name)

    def log_progress(self, img_file):
        with self._lock:
            self.images_done += 1
            done, total = self.images_done, self.images_total
            elapsed = time.monotonic() - self.start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_s = (total - done) / rate if rate > 0 else float("inf")
        eta_str = "unknown" if eta_s == float("inf") else f"{eta_s:,.0f}s"
        logger.info(
            f"[{done}/{total}] finished {img_file} "
            f"({rate:.2f} img/s, ETA {eta_str})"
        )

    def summary_lines(self):
        with self._lock:
            elapsed = time.monotonic() - self.start_time
        lines = [
            "===== Run summary =====",
            f"Elapsed: {elapsed:,.1f}s",
            f"Images: {self.images_total} total | "
            f"{self.images_done} processed | "
            f"{self.images_skipped_resume} skipped (--resume / auto-resume) | "
            f"{self.images_skipped_no_label} skipped (no label file) | "
            f"{self.images_failed_read} failed to read",
            f"Boxes: {self.boxes_seen} seen | "
            f"{self.boxes_classified} classified | "
            f"{self.boxes_malformed_line} malformed label lines | "
            f"{self.boxes_empty_crop} empty crops | "
            f"{self.boxes_dry_run} dry-run (not sent to model)",
            f"Model issues: {self.boxes_model_call_failed} call failures | "
            f"{self.boxes_bad_response} unusable responses",
            f"Low confidence boxes flagged for review: {self.boxes_low_confidence}",
        ]
        if self.classes_discovered:
            lines.append(f"New classes discovered this run: {self.classes_discovered}")
        else:
            lines.append("New classes discovered this run: none")
        return lines


class CheckpointManager:
    """
    Persists run progress to '<output_folder>/.checkpoint.json' so an
    interrupted run (crash, OOM, ctrl-C, preemption, ...) can be resumed
    automatically by just re-running the same command.

    The checkpoint records:
      - completed_images: image stems that are fully relabeled and written
      - class_map:        name -> id, so newly-discovered classes survive
                           a restart with the same ids (doesn't reshuffle
                           ids already burned into previously-written label
                           files)
      - batches_done:     batch indices that are fully finished, so a
                           resumed run can skip a whole batch folder without
                           even checking each image inside it individually

    Writes are atomic (write to a temp file, then os.replace) so a crash
    mid-write can never leave a corrupt/partial checkpoint behind.
    """

    def __init__(self, output_folder):
        self.path = Path(output_folder) / ".checkpoint.json"
        self._lock = threading.Lock()

    def load(self):
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            data.setdefault("completed_images", [])
            data.setdefault("class_map", {})
            data.setdefault("batches_done", [])
            return data
        except Exception as e:
            logger.warning(
                f"Could not read checkpoint at {self.path} ({e}). "
                "Ignoring it and starting fresh."
            )
            return None

    def save(self, completed_images, class_map, batches_done):
        """completed_images / batches_done: iterables (sets are fine)."""
        with self._lock:
            payload = {
                "completed_images": sorted(completed_images),
                "class_map": dict(class_map),
                "batches_done": sorted(batches_done),
            }
            tmp_path = self.path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, self.path)  # atomic on POSIX
            except Exception as e:
                logger.error(f"Failed to write checkpoint at {self.path}: {e}")

    def clear(self):
        """Remove the checkpoint (used for a deliberate --no_auto_resume fresh run)."""
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove checkpoint at {self.path}: {e}")


def wait_for_server_health(port, timeout=1200, poll_interval=2.0):
    """
    Poll the llama.cpp /health endpoint until it returns a status of 200 ('ok')
    or we hit the timeout threshold.
    """
    url = f"http://localhost:{port}/health"
    start_time = time.time()
    logger.info(f"Probing server health at {url} (max timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    try:
                        data = json.loads(response.read().decode())
                        if data.get("status") == "ok":
                            logger.info(
                                "Server is healthy, model is loaded, and ready to process requests."
                            )
                            return True
                    except Exception:
                        logger.info("Server responded with 200. Proceeding.")
                        return True
        except urllib.error.HTTPError as e:
            # HTTP 503 means the server is online but still loading the model weights
            if e.code == 503:
                try:
                    err_data = json.loads(e.read().decode())
                    msg = err_data.get("error", {}).get("message", "Loading model")
                    logger.info(
                        f"Server is online but model is still loading: '{msg}'..."
                    )
                except Exception:
                    logger.info("Server is online but still loading the model (503)...")
            else:
                logger.warning(f"Server returned unexpected HTTP status: {e.code}")
        except Exception as e:
            # Quietly wait if connection is refused (server process hasn't fully bound to the port yet)
            logger.debug(f"Could not connect to server port yet: {e}")

        time.sleep(poll_interval)

    logger.error(
        f"Timed out waiting for server to become healthy after {timeout} seconds."
    )
    return False


def init_llama_server(args):
    llama_manager = LlamaServerManager(
        model=args.model,
        host="localhost",
        port=args.port,
        ctx_size=args.ctx_size,
        parallel_slots=args.parallel_slots,
        n_threads=-1,
        gpu_layers=-1,
        tensor_split="1,1",
        main_gpu=0,
        temp=0.4,
        top_p=0.95,
        top_k=64,
        spec_type="draft-mtp" if args.use_mtp else "none",
        spec_draft_n_max=4 if args.use_mtp else 0,
        fa="auto",
        enable_thinking=args.enable_thinking,
        batch_size=1024,
        ubatch_size=1024,
        kv_cache_type="q4_0",
        image_min_tokens=args.image_min_tokens,
        image_max_tokens=args.image_max_tokens,
    )
    llama_manager.start_llama_server()

    # Active HTTP polling replaces the static event wait logic
    server_ready = wait_for_server_health(args.port, timeout=1200, poll_interval=20.0)
    if not server_ready:
        logger.warning(
            "Proceeding, but server health checks did not pass successfully."
        )

    return llama_manager


def encode_crop_to_data_uri(crop_rgb):
    """Encode an RGB numpy crop (as produced by cv2 after BGR2RGB) into a base64 JPEG data URI."""
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        raise ValueError("Could not JPEG-encode crop.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def detect_defect(crop_image, client, model_name, known_class_names):
    """
    Ask the model to classify a cropped defect region.

    known_class_names: list[str] of classes already known, so the model reuses
    an existing name instead of inventing near-duplicates.

    Returns dict like {"class": "spot", "confidence": 4}
    """
    data_uri = encode_crop_to_data_uri(crop_image)
    prompt = (
        "You are an expert textile quality inspector. "
        "Analyze the cropped fabric image and identify the primary visible defect. "
        f"Existing defect classes discovered so far in this dataset: {known_class_names}. "
        "First determine whether the defect matches one of the existing classes. "
        "If it does, use the exact existing class name. "
        "Only create a new class if the defect is clearly and meaningfully different "
        "from every existing class above. "
        "A new class name must be lowercase, a single word, concise, and descriptive. "
        "Do not create synonyms or variations of existing classes. "
        "Rate the defect severity based on its visible size and extent: "
        "1 = very small, 2 = small, 3 = medium, 4 = large, 5 = very large. "
        "Respond with ONLY valid JSON in exactly this format: "
        '{"reasoning":"<reasoning>","class":"<class_name>","confidence":<1-5>}. '
        "Do not include explanations, markdown, extra text, comments, or additional fields."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    if not response.choices or not response.choices[0].message:
        raise ValueError("No choices returned from the VLM API call.")

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Model returned an empty text content response.")

    output = json_repair.loads(raw)
    logger.info(f"Model response: {output}")
    return output


def load_or_init_class_map(names_from_yaml):
    """Normalize yaml `names` (list or {id: name} dict) into a name -> id dict."""
    class_map = {}
    if isinstance(names_from_yaml, dict):
        for idx, name in names_from_yaml.items():
            class_map[name] = int(idx)
    elif isinstance(names_from_yaml, list):
        for idx, name in enumerate(names_from_yaml):
            class_map[name] = idx
    return class_map


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
):
    """Relabel every box in a single image. Thread-safe w.r.t. class_map and stats."""
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

    for line in lines:
        stats.incr("boxes_seen")
        values = line.strip().split()
        if len(values) != 5:
            logger.warning(f"Malformed label line in {label_path}: '{line.strip()}'")
            stats.incr("boxes_malformed_line")
            continue

        _old_cls, x, y, bw, bh = map(float, values)

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
            result = detect_defect(crop_image, client, model_name, known_names)
        except Exception as e:
            logger.error(f"Model call failed for {img_file}: {e}")
            stats.incr("boxes_model_call_failed")
            continue

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

    write_ok = True
    try:
        with open(label_out_path, "w") as f:
            if new_label_lines:
                f.write("\n".join(new_label_lines) + "\n")
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


def find_labeled_images(train_image, train_label, image_extensions):
    """
    Walk the labels folder (not the images folder) and keep only label files
    that have at least one non-blank line, then resolve each to its matching
    image file. Returns a sorted list of image filenames.

    This is the source of truth for "has something to process" -- an image
    whose label file is empty or missing is never a candidate, even before
    --num_samples / --shuffle / --start_index / --end_index are applied.
    """
    image_names = []
    skipped_empty = 0
    skipped_no_image = 0

    label_files = sorted(
        f for f in os.listdir(train_label) if f.lower().endswith(".txt")
    )

    for label_file in label_files:
        label_path = os.path.join(train_label, label_file)
        try:
            with open(label_path, "r") as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
        except Exception as e:
            logger.error(f"Failed to read label file {label_path}: {e}")
            continue

        if not lines:
            skipped_empty += 1
            continue

        stem = Path(label_file).stem
        matched_image = None
        # Compare case-insensitively so a file like "img1.JPG" still matches
        # an --image_extensions entry of ".jpg".
        try:
            dir_entries_lower = {
                entry.lower(): entry for entry in os.listdir(train_image)
            }
        except Exception:
            dir_entries_lower = {}
        for ext in image_extensions:
            candidate = stem + ext
            if os.path.exists(os.path.join(train_image, candidate)):
                matched_image = candidate
                break
            candidate_lower = candidate.lower()
            if candidate_lower in dir_entries_lower:
                matched_image = dir_entries_lower[candidate_lower]
                break

        if matched_image is None:
            logger.warning(
                f"No matching image for label '{label_file}' in {train_image}"
            )
            skipped_no_image += 1
            continue

        image_names.append(matched_image)

    logger.info(
        f"Found {len(image_names)} image(s) with non-empty labels "
        f"({skipped_empty} label file(s) empty, {skipped_no_image} with no matching image)."
    )
    return sorted(image_names)


def chunk_list(items, batch_size):
    """Split `items` into consecutive chunks of at most `batch_size` each."""
    if not batch_size or batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


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

    os.makedirs(output_folder, exist_ok=True)
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
            batch_output_folder = output_folder
        else:
            batch_output_folder = os.path.join(output_folder, f"batch_{batch_idx:04d}")
            os.makedirs(batch_output_folder, exist_ok=True)

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
                    )
                    if img is not None:
                        last_img = img
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
                    ): img_file
                    for img_file in batch_images
                }
                for future in as_completed(futures):
                    img_file = futures[future]
                    try:
                        img = future.result()
                        if img is not None:
                            last_img = img
                    except Exception as e:
                        logger.exception(
                            f"Processing {img_file} raised an exception: {e}"
                        )

        # Whole batch finished (every image in it either processed just now
        # or already marked completed earlier) -> record it so a resumed run
        # can skip this batch's folder entirely without re-checking images.
        if not dry_run and checkpoint is not None:
            batches_done.add(batch_idx)
            with completed_lock:
                completed_snapshot = set(completed_images)
            with class_map_lock:
                class_map_snapshot = dict(class_map)
            checkpoint.save(completed_snapshot, class_map_snapshot, set(batches_done))

    return last_img


def save_updated_yaml(yaml_path, original_data, class_map):
    if not class_map:
        logger.warning(
            "Class map is empty. Skipping YAML update to prevent erasing existing names."
        )
        return
    updated = dict(original_data)
    sorted_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    updated["names"] = sorted_names
    updated["nc"] = len(sorted_names)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)


def build_client(args):
    """--use_llama_model True -> local llama.cpp server. Otherwise -> external API."""
    if args.use_llama_model:
        llama_manager = init_llama_server(args)
        client = OpenAI(
            base_url=f"http://localhost:{args.port}/v1", api_key="not-needed"
        )
        return client, llama_manager
    else:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        return client, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relabel binary defect/no-defect YOLO annotations into multi-class defect labels using a VLM."
    )
    # Logging Configuration
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
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
        "--output-folder",
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
        "--use_llama_model",
        action="store_true",
        help="Use a local llama.cpp server instead of an external API.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable the model's thinking/reasoning mode on the local llama.cpp server "
        "(--use_llama_model only). Off by default: faster and usually unnecessary for a "
        "single classify-this-crop call.",
    )
    parser.add_argument(
        "--use_mtp",
        action="store_true",
        default=True,
        help="Enable draft-MTP speculative decoding on the local llama.cpp server "
        "(--use_llama_model only). On by default for speed; pass --no_mtp to disable it "
        "if you hit compatibility issues with a given model/build.",
    )
    parser.add_argument(
        "--no_mtp",
        action="store_false",
        dest="use_mtp",
        help="Disable draft-MTP speculative decoding (--use_llama_model only).",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=20000,
        help="Context size for the local llama.cpp server (--use_llama_model only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the local llama.cpp server (--use_llama_model only).",
    )
    parser.add_argument(
        "--parallel_slots",
        type=int,
        default=1,
        help="Number of parallel inference slots on the local llama.cpp server "
        "(--use_llama_model only). If you raise this, --max_workers can be raised to match "
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
    args = parser.parse_args()

    if args.start_index is not None and args.start_index < 0:
        parser.error("--start_index must be >= 0")
    if (
        args.end_index is not None
        and args.start_index is not None
        and args.end_index <= args.start_index
    ):
        parser.error("--end_index must be greater than --start_index")

    return args


if __name__ == "__main__":
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
        logger.info("Dry run complete — no files were written, yaml was not updated.")
    else:
        try:
            save_updated_yaml(args.yaml_path, data, class_map)
            logger.info(f"Done. Final classes: {class_map}")
        except Exception as e:
            logger.error(f"Failed to save updated dataset yaml file: {e}")
