"""
Relabel binary defect / no-defect YOLO annotations into multi-class defect
labels (e.g. "cut", "spot", ...) using a vision-language model.

Supports two backends:
  --use_llama_model            -> spins up a local llama.cpp server (LlamaServerManager)
  (default, no flag)           -> talks to an external OpenAI-compatible API via
                                   --base_url / --api_key

New class names discovered by the model are accumulated across the whole
dataset (not reset per-image) and written back into --yaml_path at the end.
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

import cv2
import json_repair
import numpy as np
import yaml
from openai import OpenAI
from PIL import Image

from llama_server_manager import LlamaServerManager
from image_preprocessing import preprocess_custom_resize

logger = logging.getLogger(__name__)


def setup_logging(log_level="INFO", log_file=None):
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
        logger.info(f"[{done}/{total}] finished {img_file}")

    def summary_lines(self):
        lines = [
            "===== Run summary =====",
            f"Images: {self.images_total} total | "
            f"{self.images_done} processed | "
            f"{self.images_skipped_resume} skipped (--resume) | "
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
    llama_manager.server_ready_event.wait(timeout=1200)
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
        f"Existing defect classes: {known_class_names}. "
        "First determine whether the defect matches one of the existing classes. "
        "If it does, use the exact existing class name. "
        "Only create a new class if the defect is clearly different from every existing class. "
        "A new class name must be lowercase, a single word, concise, and descriptive. "
        "Do not create synonyms or variations of existing classes. "
        "Rate the defect severity based on its visible size and extent: "
        "1 = very small, 2 = small, 3 = medium, 4 = large, 5 = very large. "
        "Respond with ONLY valid JSON in exactly this format: "
        '{"class":"<class_name>","confidence":<1-5>}. '
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
    raw = response.choices[0].message.content
    return json_repair.loads(raw)


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
):
    """Relabel every box in a single image. Thread-safe w.r.t. class_map and stats."""
    img_path = os.path.join(train_image, img_file)
    img_stem = Path(img_file).stem
    label_path = os.path.join(train_label, img_stem + ".txt")
    label_out_path = Path(output_folder) / (img_stem + ".txt")

    if not os.path.exists(label_path):
        logger.warning(f"Label file not found for {img_file}: {label_path}")
        stats.incr("images_skipped_no_label")
        return None

    if resume and label_out_path.exists():
        logger.info(f"Skipping {img_file} (already relabeled, --resume).")
        stats.incr("images_skipped_resume")
        return None

    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"Could not read image {img_path}, skipping.")
        stats.incr("images_failed_read")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read label file {label_path}: {e}")
        stats.incr("images_failed_read")
        return None

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
        x1 = max(int((x - bw / 2) * w), 0)
        y1 = max(int((y - bh / 2) * h), 0)
        x2 = min(int((x + bw / 2) * w), w)
        y2 = min(int((y + bh / 2) * h), h)

        crop_image = img[y1:y2, x1:x2]
        if crop_image.size == 0:
            logger.warning(f"Empty crop in {img_file} for box ({x}, {y}, {bw}, {bh}), skipping box.")
            stats.incr("boxes_empty_crop")
            continue

        if dry_run:
            logger.info(f"[dry run] {img_file}: would classify box at ({x}, {y}, {bw}, {bh}).")
            stats.incr("boxes_dry_run")
            continue

        # preprocess_custom_resize works on PIL.Image, not numpy arrays -
        # round-trip through PIL and back so encode_crop_to_data_uri still gets what it needs.
        pil_crop = Image.fromarray(crop_image)
        try:
            pil_crop, _ = preprocess_custom_resize(
                pil_crop, target_height=target_height, target_width=target_width
            )
            crop_image = np.array(pil_crop)
        except Exception as e:
            logger.error(f"Error preprocessing crop in {img_file}: {e}")
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
            logger.warning(f"Invalid or missing class name in response for {img_file}: {result}")
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

    try:
        with open(label_out_path, "w") as f:
            if new_label_lines:
                f.write("\n".join(new_label_lines) + "\n")
    except Exception as e:
        logger.error(f"Failed to write relabeled annotations to {label_out_path}: {e}")

    if low_confidence_records:
        debug_out_path = Path(output_folder) / (img_stem + "_low_confidence.json")
        try:
            with open(debug_out_path, "w") as f:
                json.dump(low_confidence_records, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write low confidence records to {debug_out_path}: {e}")

    stats.log_progress(img_file)
    return img


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
    dry_run=False,
    resume=False,
    image_extensions=(".jpg", ".jpeg", ".png"),
    max_workers=1,
    target_height=1024,
    target_width=1024,
    stats=None,
):
    """
    Re-label every bounding box in every image with a model-predicted class.

    class_map is mutated in place and shared across all images (protected by a
    lock when max_workers > 1), so a class discovered on image 3 is available
    (and reused) for image 300.
    """
    if stats is None:
        stats = RunStats()

    os.makedirs(output_folder, exist_ok=True)
    image_extensions = tuple(ext.lower() for ext in image_extensions)
    image_names = sorted(
        f for f in os.listdir(train_image) if f.lower().endswith(image_extensions)
    )

    if shuffle:
        random.seed(seed)
        random.shuffle(image_names)

    if num_samples is not None:
        image_names = image_names[:num_samples]

    stats.images_total = len(image_names)
    logger.info(f"Processing {len(image_names)} image(s)" + (" [dry run]" if dry_run else "") + ".")

    class_map_lock = threading.Lock()
    last_img = None

    if max_workers <= 1:
        for img_file in image_names:
            try:
                img = process_one_image(
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
                ): img_file
                for img_file in image_names
            }
            for future in as_completed(futures):
                img_file = futures[future]
                try:
                    img = future.result()
                    if img is not None:
                        last_img = img
                except Exception as e:
                    logger.exception(f"Processing {img_file} raised an exception: {e}")

    return last_img


def save_updated_yaml(yaml_path, original_data, class_map):
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
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="not-needed")
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
    parser.add_argument("--output-folder", type=str, required=True, help="Where to save the relabeled output.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for classification.")
    parser.add_argument("--api_key", type=str, default="", help="API key for the external/hosted model.")
    parser.add_argument("--base_url", type=str, default="", help="Base URL for the external/hosted model.")
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
    parser.add_argument("--train_image", type=str, required=True, help="Path to the folder of training images.")
    parser.add_argument("--train_label", type=str, required=True, help="Path to the folder of YOLO training labels.")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to the dataset yaml file (data.yaml).")
    parser.add_argument(
        "--conf_threshold",
        type=int,
        default=2,
        help="Confidence (1-5) at/below which a box is ALSO logged to a *_low_confidence.json for manual review.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only process this many images (quick sanity-check run instead of the full dataset).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order (seeded by --seed) before applying --num_samples, "
        "so a sample isn't just the first N images alphabetically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set, for reproducible samples.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't call the model and don't write any files; just print what would happen.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have an output label file (resume an interrupted run "
        "without re-spending API calls).",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    try:
        with open(args.yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read dataset yaml file at {args.yaml_path}: {e}")
        exit(1)

    class_map = load_or_init_class_map(data.get("names", [])) if args.init_class_map else {}
    
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
            dry_run=args.dry_run,
            resume=args.resume,
            image_extensions=image_extensions,
            max_workers=args.max_workers,
            target_height=args.height,
            target_width=args.width,
            stats=stats,
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