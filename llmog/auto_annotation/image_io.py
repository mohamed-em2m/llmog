"""VLM classification call and dataset-IO helpers used by the pipeline."""

from rich import traceback
import base64
import os
from pathlib import Path
import cv2
import json_repair

from auto_annotation.logging_utils import logger
from free_detection.agent.prompts import render_auto_label_prompt


def encode_crop_to_data_uri(crop_rgb):
    """Encode an RGB numpy crop (as produced by cv2 after BGR2RGB) into a base64 JPEG data URI."""
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        raise ValueError("Could not JPEG-encode crop.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pad_box(x1, y1, x2, y2, img_w, img_h, pad_pct=0.0):
    """Expand a pixel box by pct% of its own width/height per side.

    E.g. ``pad_pct=50`` adds half a box-width to the left AND right (and
    half a box-height above AND below). The result is clamped to
    ``[0, img_w] x [0, img_h]``; ``pad_pct<=0`` returns the box unchanged,
    and a degenerate result falls back to the clamped original box.
    Returns ``(x1, y1, x2, y2)`` ints.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    try:
        pct = float(pad_pct or 0.0)
    except (TypeError, ValueError):
        pct = 0.0
    if pct <= 0.0:
        return x1, y1, x2, y2
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return x1, y1, x2, y2
    dx = bw * pct / 100.0
    dy = bh * pct / 100.0
    nx1 = max(0, int(round(x1 - dx)))
    ny1 = max(0, int(round(y1 - dy)))
    nx2 = min(int(img_w), int(round(x2 + dx)))
    ny2 = min(int(img_h), int(round(y2 + dy)))
    if nx2 <= nx1 or ny2 <= ny1:
        return (
            max(0, min(int(img_w), x1)),
            max(0, min(int(img_h), y1)),
            max(0, min(int(img_w), x2)),
            max(0, min(int(img_h), y2)),
        )
    return nx1, ny1, nx2, ny2


def _coerce_extra_body(extra_body):
    """Normalize an extra_body value to a dict ({} when unset).

    Accepts a mapping (YAML form) or a JSON-object string (CLI form) so every
    entry point converges here; raises ValueError on anything else.
    """
    if extra_body is None:
        return {}
    if isinstance(extra_body, dict):
        return dict(extra_body)
    if isinstance(extra_body, str):
        import json

        text = extra_body.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except ValueError as e:
            raise ValueError(
                f"extra_body must be a JSON object, got: {extra_body!r} ({e})"
            )
        if not isinstance(parsed, dict):
            raise ValueError(f"extra_body must be a JSON object, got: {extra_body!r}")
        return parsed
    raise ValueError(
        f"extra_body must be a mapping or JSON object, got: {extra_body!r}"
    )


def build_classify_body(
    crop_image,
    model_name,
    known_class_names,
    class_mode: str = "hybrid",
    class_definitions: str = "",
    # Comma-separated string or list of names (YAML --config list form).
    none_labels="none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    drop_none: bool = True,
    extra_body=None,
):
    """Build the chat-completions request body for one crop.

    Shared by the online path (:func:`detect_defect`) and the Batch API path
    (:mod:`auto_annotation.batch_api`), so both send byte-identical prompts.
    """
    data_uri = encode_crop_to_data_uri(crop_image)
    prompt = render_auto_label_prompt(
        known_class_names,
        class_mode=class_mode,
        class_definitions=class_definitions,
        none_labels=none_labels,
        drop_none=drop_none,
    )
    body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    coerced = _coerce_extra_body(extra_body)
    if coerced:
        body["extra_body"] = coerced
    return body


def detect_defect(
    crop_image,
    client,
    model_name,
    known_class_names,
    class_mode: str = "hybrid",
    class_definitions: str = "",
    # Comma-separated string or list of names (YAML --config list form).
    none_labels="none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    drop_none: bool = True,
    extra_body=None,
):
    """
    Ask the model to classify a cropped defect region.

    known_class_names: list[str] of classes already known, so the model reuses
    an existing name instead of inventing near-duplicates.

    class_mode: "strict" | "hybrid" | "free" — controls how the prompt
    is written w.r.t. the known class list.

    class_definitions: Optional per-class description block injected into prompt.

    none_labels / drop_none: forwarded to the prompt so the model knows it may
    answer 'none' for clean crops (which the caller then drops -> empty YOLO).

    extra_body: optional mapping forwarded verbatim as the `extra_body=`
    kwarg of the chat-completions call (provider-specific params such as
    OpenRouter `provider` routing or reasoning controls). A JSON string is
    accepted and parsed; None/empty means "send nothing extra".

    Returns dict like {"class": "spot", "confidence": 4}
    """
    create_kwargs = build_classify_body(
        crop_image,
        model_name,
        known_class_names,
        class_mode=class_mode,
        class_definitions=class_definitions,
        none_labels=none_labels,
        drop_none=drop_none,
        extra_body=extra_body,
    )
    response = client.chat.completions.create(**create_kwargs)
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


def chunk_list(items, batch_size=0):
    """Split `items` into consecutive chunks of at most `batch_size` each."""
    if not items:
        return []

    if not batch_size or batch_size <= 0:
        if type(items) is list:
            return [items]
        elif type(dict):
            return list(items.values())
        else:
            try:
                return list(items)
            except Exception:
                traceback.print_exc()
                raise Exception(f"images list names can't listed {Exception}")

    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
