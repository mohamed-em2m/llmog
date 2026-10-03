"""VLM classification call and dataset-IO helpers used by the pipeline."""

from rich import traceback
import base64
import os
from pathlib import Path
import cv2
import json_repair
from PIL import Image, ImageDraw, ImageFont

from auto_annotation.logging_utils import logger
from free_detection.agent.prompts import render_auto_label_prompt
from esr.manager import ESRConfig, upscale_pil
from esr.project import fit_long_edge


def build_esr_settings(obj) -> dict:
    """Collect the ``esr_*`` fields off a config/Namespace/dict ({} when off).

    Centralizes the getattr-with-defaults dance so the sync path, the Batch
    API path, and the checkpoint fingerprint all see identical settings.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        raw = dict(obj)
    else:
        raw = {
            k: getattr(obj, k, None)
            for k in (
                "esr_enabled",
                "esr_model",
                "esr_model_path",
                "esr_model_repo",
                "esr_cache_dir",
                "esr_scale",
                "esr_target_long_edge",
                "esr_max_long_edge",
                "esr_tile_size",
                "esr_overlap",
                "esr_batch_size",
                "esr_for_crops",
                "esr_compile",
                "esr_channels_last",
                "esr_device",
            )
        }
    if not raw.get("esr_enabled"):
        return {}
    cfg = ESRConfig.from_config(raw)
    if not cfg.enabled:
        return {}
    return raw


def maybe_esr_upscale_pil(
    pil_image, esr_settings, *, purpose="crop", long_edge_cap=None
):
    """Upscale a VLM-bound PIL image with Real-ESRGAN when enabled.

    ``purpose`` is ``"crop"`` (gated by ``esr_for_crops``) or ``"scene"``
    (full_som / classify full images -- always applied when enabled).
    ``long_edge_cap`` (pixels, e.g. the downstream letterbox size) trims the
    ESR working image when it is larger than what the caller keeps anyway --
    a 100px crop need not swell to the 2048px scene target before being
    letterboxed back to 1024. Returns ``(pil_image, esr_info)``; disabled
    configs return the input with ``applied: False`` (no torch import, no
    model load). Callers must skip this entirely on dry runs (no model load
    for a no-op run).
    """
    if not esr_settings:
        w, h = pil_image.size
        return pil_image, {
            "applied": False,
            "orig_w": w,
            "orig_h": h,
            "pre_w": w,
            "pre_h": h,
            "up_w": w,
            "up_h": h,
            "work_w": w,
            "work_h": h,
            "scale": 1,
        }
    cfg = ESRConfig.from_config(esr_settings)
    if not cfg.enabled:
        w, h = pil_image.size
        return pil_image, {
            "applied": False,
            "orig_w": w,
            "orig_h": h,
            "pre_w": w,
            "pre_h": h,
            "up_w": w,
            "up_h": h,
            "work_w": w,
            "work_h": h,
            "scale": 1,
        }
    if purpose == "crop" and not cfg.for_crops:
        logger.info("ESR: esr_for_crops is off -- sending the native crop.")
        w, h = pil_image.size
        return pil_image, {
            "applied": False,
            "orig_w": w,
            "orig_h": h,
            "pre_w": w,
            "pre_h": h,
            "up_w": w,
            "up_h": h,
            "work_w": w,
            "work_h": h,
            "scale": 1,
        }
    upscaled, info = upscale_pil(pil_image.convert("RGB"), cfg)
    if info.get("applied") and long_edge_cap:
        cw, ch = fit_long_edge(info["work_w"], info["work_h"], long_edge_cap)
        if (cw, ch) != (info["work_w"], info["work_h"]):
            logger.debug(
                "ESR crop cap: trimming working %dx%d to %dx%d "
                "(downstream keeps <= %d).",
                info["work_w"],
                info["work_h"],
                cw,
                ch,
                long_edge_cap,
            )
            upscaled = upscaled.resize((cw, ch), Image.Resampling.LANCZOS)
            info = {**info, "work_w": cw, "work_h": ch}
    if info.get("applied"):
        logger.info(
            "ESR upscale (%s): %dx%d -> working %dx%d (x%d).",
            purpose,
            info["orig_w"],
            info["orig_h"],
            info["work_w"],
            info["work_h"],
            info.get("scale", 1),
        )
    return upscaled, info


def upscale_scene_for_som(pil_image, box_xyxy, esr_settings):
    """Upscale a full scene for SoM marking; returns ``(scene, scaled_box)``.

    Box coords are in the input image's pixels; after ESR they are
    multiplied by the actual working/original ratio (never assumed from
    config -- the true factor comes back in ``esr_info``).
    """
    upscaled, info = maybe_esr_upscale_pil(pil_image, esr_settings, purpose="scene")
    if not info.get("applied"):
        return upscaled, tuple(int(v) for v in box_xyxy), info
    sx = info["work_w"] / max(1, info["orig_w"])
    sy = info["work_h"] / max(1, info["orig_h"])
    x1, y1, x2, y2 = box_xyxy
    w, h = upscaled.size
    scaled = (
        max(0, min(w, int(round(x1 * sx)))),
        max(0, min(h, int(round(y1 * sy)))),
        max(0, min(w, int(round(x2 * sx)))),
        max(0, min(h, int(round(y2 * sy)))),
    )
    return upscaled, scaled, info


def encode_crop_to_data_uri(crop_rgb):
    """Encode an RGB numpy crop (as produced by cv2 after BGR2RGB) into a base64 JPEG data URI."""
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        raise ValueError("Could not JPEG-encode crop.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def dump_vlm_crop(crop_rgb, dump_dir, stem, line_no):
    """Save the exact final VLM-bound RGB numpy crop for debugging.

    Writes ``<stem>_box<line_no>.jpg`` under ``dump_dir`` (created on
    demand) -- precisely the pixels the model receives, letterbox bars
    included. Never raises: failures are WARNING-logged and return None.
    """
    try:
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}_box{line_no}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        logger.debug("Dumped VLM crop %s", path)
        return path
    except Exception as e:
        logger.warning(f"Could not dump VLM crop for {stem} box {line_no}: {e}")
        return None


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


def draw_som_context(pil_image, x1, y1, x2, y2, label="1"):
    """Copy a full-scene image with one region highlighted, SoM-style.

    Draws a thick lime box plus a filled number badge at the top-left corner
    so a VLM asked to "classify the marked box" can locate it unambiguously.
    The input image is never mutated; a new ``PIL.Image`` is returned.
    """
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    line = max(2, min(w, h) // 300)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    for off in range(line):
        draw.rectangle([x1 + off, y1 + off, x2 - off, y2 - off], outline=(57, 255, 20))
    try:
        font = ImageFont.load_default(size=max(14, min(w, h) // 40))
    except Exception:
        font = ImageFont.load_default()
    text = str(label)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = max(2, line)
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    draw.rectangle(
        [bx1, by1, bx1 + tw + 2 * pad, by1 + th + 2 * pad], fill=(57, 255, 20)
    )
    draw.text((bx1 + pad, by1 + pad), text, font=font, fill=(0, 0, 0))
    return annotated


def resize_crop_ratio(pil_image, ratio, max_long_edge=None):
    """Scale a crop by ``ratio`` (aspect preserved, LANCZOS, no padding bars).

    E.g. ``ratio=1.5`` turns a 100x80 crop into 150x120. The long edge is
    capped at ``max_long_edge`` when given (protects batch request sizes).
    Raises ``ValueError`` for non-positive ratios. Returns a ``PIL.Image``.
    """
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        raise ValueError(f"crop_resize_ratio must be a number, got {ratio!r}.")
    if r <= 0:
        raise ValueError(f"crop_resize_ratio must be > 0, got {ratio!r}.")
    w, h = pil_image.size
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    if max_long_edge and max(nw, nh) > max_long_edge:
        s = float(max_long_edge) / max(nw, nh)
        nw, nh = max(1, round(nw * s)), max(1, round(nh * s))
    if (nw, nh) == (w, h):
        return pil_image.copy()
    return pil_image.resize((nw, nh), Image.Resampling.LANCZOS)


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
    region_context: str = "crop",
):
    """Build the chat-completions request body for one crop.

    Shared by the online path (:func:`detect_defect`) and the Batch API path
    (:mod:`auto_annotation.batch_api`), so both send byte-identical prompts.

    ``region_context="full_som"`` means the image is the FULL scene with the
    candidate region marked by a highlighted box labeled "1" (see
    :func:`draw_som_context`); a directive is prepended so the model
    classifies only the marked box and reads "crop" below as that region.
    """
    data_uri = encode_crop_to_data_uri(crop_image)
    prompt = render_auto_label_prompt(
        known_class_names,
        class_mode=class_mode,
        class_definitions=class_definitions,
        none_labels=none_labels,
        drop_none=drop_none,
    )
    if str(region_context or "crop").lower().strip() == "full_som":
        prompt = (
            "CONTEXT MODE: the attached image is the FULL scene, not a crop. "
            "The candidate region is marked with a highlighted bounding box "
            'labeled "1". Classify ONLY the object/defect inside that marked '
            "box; treat the rest of the scene as surrounding context. Where "
            'the instructions below say "crop", read it as "the marked box '
            'region".\n\n' + prompt
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
    region_context: str = "crop",
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
        region_context=region_context,
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
