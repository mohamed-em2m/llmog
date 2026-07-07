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
import os
import random
import threading
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
    llama_manager.server_ready_event.wait(timeout=120)
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
):
    """Relabel every box in a single image. Thread-safe w.r.t. class_map."""
    img_path = os.path.join(train_image, img_file)
    img_stem = Path(img_file).stem
    label_path = os.path.join(train_label, img_stem + ".txt")
    label_out_path = Path(output_folder) / (img_stem + ".txt")

    if not os.path.exists(label_path):
        return None

    if resume and label_out_path.exists():
        print(f"Skipping {img_file} (already relabeled, --resume).")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not read image {img_path}, skipping.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    new_label_lines = []
    low_confidence_records = []

    for line in lines:
        values = line.strip().split()
        if len(values) != 5:
            continue

        _old_cls, x, y, bw, bh = map(float, values)
        x1 = max(int((x - bw / 2) * w), 0)
        y1 = max(int((y - bh / 2) * h), 0)
        x2 = min(int((x + bw / 2) * w), w)
        y2 = min(int((y + bh / 2) * h), h)

        crop_image = img[y1:y2, x1:x2]
        if crop_image.size == 0:
            print(f"Warning: empty crop in {img_file}, skipping box.")
            continue

        if dry_run:
            print(f"[dry run] {img_file}: would classify box at ({x}, {y}, {bw}, {bh}).")
            continue

        # preprocess_custom_resize works on PIL.Image, not numpy arrays -
        # round-trip through PIL and back so encode_crop_to_data_uri (which
        # expects an RGB numpy array for cv2) still gets what it needs.
        pil_crop = Image.fromarray(crop_image)
        pil_crop, _ = preprocess_custom_resize(
            pil_crop, target_height=target_height, target_width=target_width
        )
        crop_image = np.array(pil_crop)

        # class_map is read here for the prompt before we know if this call
        # will add a new class. Take the snapshot under the lock so a
        # concurrent writer (another thread inserting a new class) can't
        # mutate the dict mid-iteration and blow up list(...).
        with class_map_lock:
            known_names = list(class_map.keys())

        try:
            result = detect_defect(crop_image, client, model_name, known_names)
        except Exception as e:
            print(f"Warning: model call failed for {img_file}: {e}")
            continue

        if not isinstance(result, dict):
            print(
                f"Warning: unexpected (non-dict) model response for {img_file}: "
                f"{result!r}, skipping box."
            )
            continue

        class_name = result.get("class")
        if not class_name or not isinstance(class_name, str):
            continue
        class_name = class_name.strip().lower()
        if not class_name:
            continue

        raw_confidence = result.get("confidence", 0)
        try:
            confidence = int(raw_confidence)
        except (TypeError, ValueError):
            print(
                f"Warning: non-numeric confidence {raw_confidence!r} for {img_file}, "
                "defaulting to 0."
            )
            confidence = 0

        with class_map_lock:
            if class_name not in class_map:
                class_map[class_name] = len(class_map)
            new_cls_id = class_map[class_name]

        new_label_lines.append(f"{new_cls_id} {x} {y} {bw} {bh}")

        if confidence <= conf_threshold:
            low_confidence_records.append(
                {"class": class_name, "confidence": confidence, "bbox": [x, y, bw, bh]}
            )

    if dry_run:
        return img

    with open(label_out_path, "w") as f:
        if new_label_lines:
            f.write("\n".join(new_label_lines) + "\n")

    if low_confidence_records:
        debug_out_path = Path(output_folder) / (img_stem + "_low_confidence.json")
        with open(debug_out_path, "w") as f:
            json.dump(low_confidence_records, f, indent=4)

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
):
    """
    Re-label every bounding box in every image with a model-predicted class.

    class_map is mutated in place and shared across all images (protected by a
    lock when max_workers > 1), so a class discovered on image 3 is available
    (and reused) for image 300.

    num_samples: if set, only process this many images (great for quickly
        sanity-checking a prompt change before a full run).
    shuffle: shuffle image order (seeded) before applying num_samples, so a
        "sample" isn't just the first N alphabetically.
    dry_run: don't call the model and don't write any files, just print what
        would happen for each image/box.
    resume: skip images that already have an output label file, so an
        interrupted run can be restarted without redoing work (and without
        burning API calls again).
    image_extensions: only files with these extensions are treated as images.
    max_workers: number of images processed concurrently. Keep at 1 when using
        a local llama.cpp server with parallel_slots=1.
    """
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

    print(f"Processing {len(image_names)} image(s)" + (" [dry run]" if dry_run else "") + ".")

    class_map_lock = threading.Lock()
    last_img = None

    if max_workers <= 1:
        for img_file in image_names:
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
            )
            if img is not None:
                last_img = img
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
                    print(f"Warning: processing {img_file} raised an exception: {e}")

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

    with open(args.yaml_path, "r") as f:
        data = yaml.safe_load(f)

    class_map = load_or_init_class_map(data.get("names", [])) if args.init_class_map else {}
    client, llama_manager = build_client(args)

    image_extensions = tuple(
        ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
        for ext in args.image_extensions.split(",")
        if ext.strip()
    )


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
        )
    finally:
        if llama_manager is not None:
            try:
                llama_manager.stop_llama_server()
            except Exception:
                pass

    if args.dry_run:
        print("Dry run complete — no files were written, yaml was not updated.")
    else:
        save_updated_yaml(args.yaml_path, data, class_map)
        print(f"Done. Final classes: {class_map}")