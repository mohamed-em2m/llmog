"""VLM classification call and dataset-IO helpers used by the pipeline."""

import base64
import os
from pathlib import Path

import cv2
import json_repair

from auto_annotation.logging_utils import logger


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
