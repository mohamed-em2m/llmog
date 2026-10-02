"""OpenAI Batch API flow for the auto_label task (~50% cheaper than sync).

Three phases (a batch job can outlive a single run -- OpenAI allows up to a
24h completion window, so submit and finalize are separable):

  1. ``collect_batch_requests`` -- walk the dataset exactly like the online
     path (label filter -> shuffle/seed -> index slice -> num_samples),
     apply the small-box filter at build time, and stage one
     ``/v1/chat/completions`` request per remaining box.
  2. ``submit_batch_job`` -- write the ``.jsonl``, upload it, create the
     batch, and persist everything finalize needs to ``.batch_job.json``.
  3. ``poll_batch_job`` + ``finalize_batch_job`` -- wait for completion,
     download the output file, and run the SAME post-processing as the online
     path (none-handling, strict-mode guard, class_map append, confidence
     review, drop_small_images manifest) into the normal staging layout so
     checkpointing, resume and the end-of-run flatten keep working.

CLI modes (``--batch_mode``): ``auto`` (resume a saved job or submit+poll+finalize
in one run), ``submit`` (build+submit only, exit), ``poll`` (poll a saved job
-- or ``--batch_job_id`` -- and finalize).
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import httpx
import json_repair
import numpy as np
from PIL import Image

from auto_annotation.logging_utils import logger
from auto_annotation.checkpoint import build_run_settings
from auto_annotation.image_io import (
    build_classify_body,
    draw_som_context,
    find_labeled_images,
    pad_box,
    resize_crop_ratio,
)
from auto_annotation.single_image import _normalize_label, _parse_none_labels
from free_detection.image_preprocessing import preprocess_custom_resize

JOB_FILENAME = ".batch_job.json"
SKIPPED_MANIFEST = "skipped_small_images.txt"

_TERMINAL_OK = "completed"
_TERMINAL_BAD = ("failed", "expired", "cancelled")
_IN_PROGRESS = ("validating", "in_progress", "finalizing", "cancelling")
# A freshly submitted batch may 404 on GET for a while (provider-side
# registration lag -- seen on OpenRouter: create returns the id, the next
# retrieve 1s later is "not found"). Tolerate consecutive 404s this long
# before treating the batch id as genuinely wrong.
_NOT_FOUND_GRACE_S = 900


def _is_transient_retrieve_error(err) -> bool:
    """True when a retrieve failure likely means 'still initializing'.

    Covers the provider's registration lag right after submit: 404s, empty
    or non-JSON bodies, and payloads that are not a batch object (yet).
    Anything else (auth, real 4xx/5xx with a body) still raises immediately.
    """
    if isinstance(err, ValueError):
        # json.JSONDecodeError from resp.json() subclasses ValueError.
        return True
    msg = str(err).lower()
    return (
        "not found" in msg
        or " 404" in msg
        or msg.startswith("404")
        or "unexpected batch payload" in msg
    )


def batch_job_path(output_folder) -> Path:
    return Path(output_folder) / JOB_FILENAME


def load_job(output_folder):
    """Return the saved batch job dict, or None when no job was submitted."""
    p = batch_job_path(output_folder)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error(f"Could not read batch job file {p}: {e}")
        return None


def save_job(output_folder, job: dict) -> Path:
    p = batch_job_path(output_folder)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    return p


def make_custom_id(stem: str, line_no: int) -> str:
    """Stable per-box id (OpenAI caps custom_id at 64 chars)."""
    base = f"{stem}-{line_no}"
    return base if len(base) <= 64 else f"{stem[:56]}-{line_no}"[-64:]


def collect_batch_requests(
    train_image,
    train_label,
    image_extensions=(".jpg", ".jpeg", ".png"),
    num_samples=None,
    shuffle=False,
    seed=42,
    start_index=None,
    end_index=None,
    target_height=1024,
    target_width=1024,
    model_name="local-model",
    class_mode="hybrid",
    class_definitions="",
    none_labels="none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    drop_none=True,
    extra_body=None,
    min_box_size=0,
    small_box_action="keep",
    known_names=(),
    crop_padding_pct=0.0,
    recls_context="crop",
    crop_resize_ratio=None,
):
    """Build one batch request per classifiable box.

    Returns ``(requests, stems)`` where ``requests`` is a list of
    ``{"custom_id", "body", "meta"}`` and ``stems`` maps
    ``stem -> {"img_file", "kept": [orig YOLO lines], "skipped_small": int,
    "sent": {custom_id: meta}}``. Small boxes never become requests: ``keep``
    lines are staged in ``kept`` for finalize to merge, ``drop`` lines are
    counted in ``skipped_small`` for the drop_small_images guard.
    """
    try:
        min_side = int(min_box_size or 0)
    except (TypeError, ValueError):
        min_side = 0
    action = str(small_box_action or "keep").lower().strip()
    if action not in ("keep", "drop"):
        action = "keep"
    try:
        _crop_pad = float(crop_padding_pct or 0.0)
    except (TypeError, ValueError):
        _crop_pad = 0.0
    logger.info(
        f"Small-box filter: min_box_size={min_side}px "
        f"({'DISABLED' if min_side <= 0 else 'boxes with w<MIN or h<MIN are filtered'}), "
        f"small_box_action={action}, drop_small_images will be applied at finalize."
    )
    if _crop_pad > 0:
        logger.info(
            f"Crop padding: boxes expanded by {_crop_pad}% of their own "
            "width/height per side (clamped to image) before the VLM crop."
        )
    _som_mode = str(recls_context or "crop").lower().strip() == "full_som"
    if _som_mode:
        logger.info(
            "Reclassification context=full_som: sending the full scene with "
            "the box highlighted instead of the crop (more tokens/request)."
        )
    try:
        _ratio = float(crop_resize_ratio) if crop_resize_ratio is not None else None
        if _ratio is not None and _ratio <= 0:
            _ratio = None
    except (TypeError, ValueError):
        _ratio = None
    if _ratio is not None and not _som_mode:
        logger.info(
            f"Crop resize ratio={_ratio} (long edge capped at "
            f"{max(target_height, target_width)}px) instead of the fixed "
            f"{target_width}x{target_height} letterbox."
        )

    image_names = find_labeled_images(train_image, train_label, image_extensions)
    eligible_total = len(image_names)
    if shuffle:
        random.seed(seed)
        random.shuffle(image_names)
        logger.info(
            f"Sample selection: shuffled ALL {eligible_total} eligible image(s) "
            f"with seed={seed}, then slicing (the seed decides WHICH images "
            "are picked, not just their order)."
        )
    elif seed is not None and int(seed) != 42:
        logger.warning(
            f"--seed {seed} was given but --shuffle is OFF: images stay in label "
            "file order, the seed is ignored, and the same first N images are "
            "picked every run. Pass --shuffle (or set shuffle: true in --config) "
            "to let the seed choose a random subset."
        )
    else:
        logger.info(
            "Sample selection: --shuffle is off -- images are processed in "
            "label file order (--seed has no effect)."
        )
    if start_index is not None or end_index is not None:
        start = start_index or 0
        end = end_index if end_index is not None else len(image_names)
        if start < 0 or end < start:
            logger.error(
                f"Invalid --start_index/--end_index range ({start}, {end}) "
                f"for {len(image_names)} image(s); ignoring the range."
            )
        else:
            logger.info(
                f"Applying index range [{start}, {end}) -> "
                f"{len(image_names[start:end])} of {len(image_names)} image(s)."
            )
            image_names = image_names[start:end]
    if num_samples is not None:
        if num_samples >= eligible_total:
            logger.warning(
                f"--num_samples {num_samples} >= {eligible_total} eligible "
                "image(s): EVERY eligible image is selected, so no seed/shuffle "
                "can change the set. Lower --num_samples or add more labeled "
                "images to get a varying sample."
            )
        image_names = image_names[:num_samples]
    if eligible_total:
        preview = ", ".join(Path(n).stem for n in image_names[:5])
        logger.info(
            f"Selected {len(image_names)} of {eligible_total} eligible image(s) "
            f"(shuffle={'on, seed=' + str(seed) if shuffle else 'off'}); "
            f"first: {preview}{'...' if len(image_names) > 5 else ''}"
        )

    requests = []
    stems = {}
    for img_file in image_names:
        stem = Path(img_file).stem
        entry = {"img_file": img_file, "kept": [], "skipped_small": 0, "sent": {}}
        stems[stem] = entry

        img_path = os.path.join(train_image, img_file)
        label_path = os.path.join(train_label, stem + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not read image {img_path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read label file {label_path}: {e}")
            continue

        for line_no, line in enumerate(lines):
            values = line.strip().split()
            if len(values) != 5:
                logger.warning(
                    f"Malformed label line in {label_path}: '{line.strip()}'"
                )
                continue
            _old_cls, x, y, bw, bh = map(float, values)
            x1 = max(0, min(w, round((x - bw / 2) * w)))
            y1 = max(0, min(h, round((y - bh / 2) * h)))
            x2 = max(0, min(w, round((x + bw / 2) * w)))
            y2 = max(0, min(h, round((y + bh / 2) * h)))
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid box in {img_file}: {values}")
                continue
            if min_side > 0 and ((x2 - x1) < min_side or (y2 - y1) < min_side):
                if action == "keep":
                    entry["kept"].append(line.strip())
                else:
                    entry["skipped_small"] += 1
                continue

            if _som_mode:
                try:
                    som_view = draw_som_context(Image.fromarray(img), x1, y1, x2, y2)
                    som_view, _ = preprocess_custom_resize(
                        som_view,
                        target_height=target_height,
                        target_width=target_width,
                    )
                    crop = np.array(som_view)
                except Exception as e:
                    logger.error(
                        f"Error building SoM context in {img_file} for box "
                        f"({x}, {y}): {e}"
                    )
                    continue
            else:
                if _crop_pad > 0:
                    x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, w, h, _crop_pad)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.warning(
                        f"Empty crop in {img_file} for box ({x}, {y}), skipping."
                    )
                    continue
                try:
                    pil_crop = Image.fromarray(crop)
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
                    crop = np.array(pil_crop)
                except Exception as e:
                    logger.error(
                        f"Error resizing crop in {img_file} for box ({x}, {y}): {e}"
                    )
                    continue

            custom_id = make_custom_id(stem, line_no)
            body = build_classify_body(
                crop,
                model_name,
                list(known_names),
                class_mode=class_mode,
                class_definitions=class_definitions,
                none_labels=none_labels,
                drop_none=drop_none,
                extra_body=extra_body,
                region_context="full_som" if _som_mode else "crop",
            )
            meta = {
                "stem": stem,
                "img_file": img_file,
                "line_no": line_no,
                "x": x,
                "y": y,
                "bw": bw,
                "bh": bh,
            }
            entry["sent"][custom_id] = meta
            requests.append({"custom_id": custom_id, "body": body, "meta": meta})

    return requests, stems


def _batches_url(client) -> str:
    base = str(getattr(client, "base_url", "") or "").rstrip("/")
    return f"{base}/batches"


def _batches_headers(client) -> dict:
    key = getattr(client, "api_key", None)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _inline_requests(requests):
    """Shape one batch request list for providers that reject ``input_file_id``.

    Hosts like OpenRouter implement ``/batches`` but read the requests straight
    out of the create body and ignore an uploaded file reference (they answer
    ``Batch body ended before a `requests` array was found.``). Their per-item
    shape is just ``custom_id`` + ``body`` -- no method/url envelope.
    """
    return [{"custom_id": req["custom_id"], "body": req["body"]} for req in requests]


def _has_data_uri_images(requests) -> bool:
    """True when any request embeds a base64/data: URI image part."""
    for req in requests or []:
        body = (req or {}).get("body") or {}
        for msg in body.get("messages", []) or []:
            content = (msg or {}).get("content")
            parts = content if isinstance(content, list) else []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                iu = part.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if isinstance(url, str) and url.startswith("data:"):
                    return True
    return False


def _create_batch_inline(client, requests, model_name, timeout=120.0):
    """POST /batches with the requests inline instead of an uploaded file id.

    Uses httpx directly rather than ``client.batches.create``: the SDK casts the
    response into its strict ``Batch`` model, which rejects the non-standard
    status enums and shapes these hosts return. Plain JSON keeps them working.
    """
    if _has_data_uri_images(requests):
        # OpenRouter-style batch is URL-only for multimodal input: base64 /
        # data: URI images are rejected on every provider, so every request
        # would fail after submit. Fail here instead of billing a dead batch.
        raise RuntimeError(
            "Inline batch hosts (e.g. OpenRouter) accept images as public "
            "http(s) URLs only -- base64 / data: URI images are rejected on "
            "every provider, so this batch would fail 100% of its requests. "
            "Either serve the crops at public URLs, or run without "
            "--use_batch_api (sync chat/completions accepts base64 data URIs, "
            "at full price instead of the ~50% batch discount)."
        )
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": model_name,
        "requests": _inline_requests(requests),
    }
    timeout = httpx.Timeout(timeout, connect=30.0)
    with httpx.Client(timeout=timeout) as http:
        resp = http.post(
            _batches_url(client), headers=_batches_headers(client), json=payload
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"POST {_batches_url(client)} failed with {resp.status_code}: {resp.text[:400]}"
        )
    data = resp.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise RuntimeError(f"Batch create returned no id: {resp.text[:400]}")
    return data


def _retrieve_batch_inline(client, batch_id, timeout=60.0):
    """GET /batches/{id} as plain JSON (same tolerance reason as create)."""
    url = f"{_batches_url(client)}/{batch_id}"
    timeout = httpx.Timeout(timeout, connect=30.0)
    with httpx.Client(timeout=timeout) as http:
        resp = http.get(url, headers=_batches_headers(client))
    if resp.status_code == 404:
        raise RuntimeError(
            f"Batch {batch_id} not found at {url}. Batch ids are provider-scoped; "
            "pass the right --batch_job_id or re-run with --batch_mode submit."
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"GET {url} failed with {resp.status_code}: {resp.text[:400]}"
        )
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected batch payload from {url}: {resp.text[:400]}")
    err = data.get("error")
    if err is not None:
        # OpenRouter answers "not found" as HTTP 200 + {"error": {...}}, not
        # as HTTP 404 -- surface it as an error so the poll loop can treat
        # "not found" as still-initializing while other errors fail fast.
        if isinstance(err, dict):
            detail = err.get("message") or err.get("code") or str(err)[:300]
        else:
            detail = str(err)[:300]
        raise RuntimeError(f"Batch {batch_id} error at {url}: {detail}")
    return data


def _batch_field(batch, name, default=None):
    """Read a field from either an SDK Batch object or a plain JSON dict."""
    if isinstance(batch, dict):
        return batch.get(name, default)
    value = getattr(batch, name, None)
    if value is None:
        extra = getattr(batch, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return default if value is None else value


def _inline_results(batch):
    """Return the inlined ``results`` list from a provider batch payload, if any.

    OpenAI writes results to an output file; hosts like OpenRouter inline them
    in the retrieve response instead (retained ~30 days).
    """
    results = _batch_field(batch, "results")
    return results if isinstance(results, list) else None


def _result_content_text(item):
    """Pull the assistant text out of one batch result, tolerating shape drift.

    Handles OpenAI's ``{"response": {"body": {"choices": [...]}}}`` and inline
    hosts' ``{"result": {"body": {"choices": [...]}}}`` / flatter variants.
    """
    node = item.get("result")
    if not isinstance(node, dict):
        node = item.get("response")
    if not isinstance(node, dict):
        return None
    if node.get("status_code") not in (None, 200):
        return None
    body = node.get("body")
    if not isinstance(body, dict):
        body = node
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = choices[0].get("text")
    return content if isinstance(content, str) else None


def _inline_error_summary(batch, limit=3):
    """Summarize per-request errors from inlined results (failed batches).

    Shapes vary: ``{"custom_id", "error": {"message": ...}}``,
    ``{"custom_id", "error": "..."}``, or a non-200 ``result.status_code``.
    Returns up to ``limit`` ``"custom_id: message"`` strings.
    """
    results = _inline_results(batch) or []
    problems = []
    for item in results:
        if not isinstance(item, dict):
            continue
        cid = item.get("custom_id", "?")
        err = item.get("error")
        msg = None
        if isinstance(err, dict):
            msg = (
                err.get("message")
                or err.get("code")
                or err.get("type")
                or str(err)[:200]
            )
        elif isinstance(err, str):
            msg = err[:200]
        if msg is None:
            node = item.get("result")
            if isinstance(node, dict) and node.get("status_code") not in (
                None,
                200,
            ):
                msg = f"status_code={node.get('status_code')}"
        if msg:
            problems.append(f"{cid}: {msg}")
        if len(problems) >= limit:
            break
    return problems


def _is_inline_required_error(err) -> bool:
    """True when a 400 means the provider wants an inline ``requests`` array."""
    text = f"{getattr(err, 'message', '') or ''} {err}".lower()
    return "requests" in text and (
        "before a" in text or "not found" in text or "missing" in text
    )


def submit_batch_job(
    client,
    output_folder,
    requests,
    stems,
    model_name,
    class_map,
    params,
    completion_window="24h",
    submit_style="auto",
    public_images=False,
    image_host="catbox",
    run_settings=None,
):
    """Write the JSONL, upload it, create the batch, persist the job file.

    ``params`` is a JSON-serializable dict of everything finalize needs
    (class_mode, class_definitions, none_labels, drop_none, conf_threshold,
    min_box_size, small_box_action, drop_small_images, extra_body,
    inplace_saving). Returns the job dict.

    ``submit_style``: ``file`` (OpenAI's uploaded-JSONL reference), ``inline``
    (requests embedded in the create body -- for hosts whose /v1/batches
    ignores ``input_file_id``), or ``auto`` (try file, fall back to inline).

    ``public_images``: when the submit resolves to inline, upload crop JPEGs
    to ``image_host`` and rewrite data-URI parts to public URLs first
    (inline hosts reject base64 images). Without it, data-URI bodies fail
    fast inside ``_create_batch_inline`` instead of billing a dead batch.

    ``run_settings``: optional fingerprint dict (see
    ``auto_annotation.checkpoint.build_run_settings``) describing the
    label-affecting settings that built these requests; stored in the job
    file so finalize writes it into the checkpoint.
    """
    # OpenRouter-style hosts resolve the ``:batch`` catalog entry themselves:
    # the Batch API takes the BASE model slug (their own example submits
    # "openai/gpt-6-luna", never "openai/gpt-6-luna:batch"). A suffixed slug
    # passes submit-time validation but fails every request at execution.
    # Normalize both the batch-level model and any per-request body model.
    batch_model = str(model_name or "")
    if batch_model.endswith(":batch"):
        batch_model = batch_model[: -len(":batch")]
        logger.info(
            f"Batch model {model_name!r} -> {batch_model!r}: :batch is a "
            "catalog/pricing suffix, the Batch API wants the base slug."
        )
        for req in requests:
            body = req.get("body")
            if isinstance(body, dict) and body.get("model") == model_name:
                body["model"] = batch_model
    model_name = batch_model
    os.makedirs(output_folder, exist_ok=True)
    jsonl_path = Path(output_folder) / "batch_requests.jsonl"

    def _write_jsonl():
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for req in requests:
                f.write(
                    json.dumps(
                        {
                            "custom_id": req["custom_id"],
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": req["body"],
                        }
                    )
                    + "\n"
                )
        logger.info(f"Wrote {len(requests)} batch request(s) to {jsonl_path}.")

    _write_jsonl()

    style = (submit_style or "auto").strip().lower()
    if style not in ("auto", "file", "inline"):
        raise ValueError(
            f"--batch_submit_style must be auto|file|inline, got {style!r}"
        )

    batch_client = getattr(client, "batches", None)
    # The file path needs the SDK namespace; the inline path needs a base_url
    # to POST to. Neither present means this is not a provider client at all.
    if batch_client is None and not getattr(client, "base_url", None):
        raise RuntimeError(
            "This endpoint does not expose the Batches API (client.batches is "
            "missing). Batch mode needs an OpenAI-compatible provider with "
            "/v1/batches support -- local llama.cpp/vLLM servers do not have "
            "it. Use --server_type external against such a provider, or run "
            "without --use_batch_api."
        )
    if batch_client is None and style != "inline":
        style = "inline"

    uploaded_id = None
    used_inline = style == "inline"
    batch = None
    if style in ("auto", "file"):
        try:
            with open(jsonl_path, "rb") as f:
                uploaded = client.files.create(file=f, purpose="batch")
            uploaded_id = uploaded.id
            logger.info(f"Uploaded batch input file: {uploaded.id}")
            batch = batch_client.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata={"description": f"llmog auto_label: {len(requests)} box(es)"},
            )
        except Exception as e:
            if style == "file" or not _is_inline_required_error(e):
                raise
            # The provider ignored input_file_id and wants the requests in the
            # body. Drop the file reference so the job is recorded as inline.
            logger.warning(
                "Provider rejected the uploaded-file reference "
                f"({getattr(e, 'message', None) or e}); retrying with the requests "
                "inlined in the create body (batch_submit_style=inline)."
            )
            batch = None
    if batch is None:
        used_inline = True
        if public_images:
            if image_host != "catbox":
                raise ValueError(f"--image_host must be catbox, got {image_host!r}.")
            from auto_annotation.image_hosting import (
                rewrite_bodies_to_public_urls,
            )

            rewrite_bodies_to_public_urls(requests, output_folder)
            # Re-write so the on-disk JSONL matches exactly what is submitted
            # (bodies were mutated from data URIs to public URLs above).
            _write_jsonl()
        batch = _create_batch_inline(client, requests, model_name)
    batch_id = _batch_field(batch, "id")
    batch_status = _batch_field(batch, "status", "?")
    logger.info(
        f"Batch submitted: id={batch_id} status={batch_status} "
        f"({len(requests)} request(s), window={completion_window}). "
        "Batch API bills at ~50% of sync chat-completions rates on OpenAI."
    )
    job = {
        "phase": "submitted",
        "batch_id": batch_id,
        "input_file_id": uploaded_id,
        "output_file_id": None,
        "submit_style": "inline" if used_inline else "file",
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_window": completion_window,
        "n_requests": len(requests),
        "class_map": dict(class_map),
        "params": dict(params or {}),
        "stems": stems,
    }
    if run_settings is not None:
        job["run_settings"] = dict(run_settings)
    save_job(output_folder, job)
    return job


def poll_batch_job(client, job, poll_interval=60, poll_timeout=0):
    """Block until the batch reaches a terminal state (Ctrl+C resumes later).

    Returns the refreshed batch object. ``poll_timeout=0`` polls forever.
    """
    batch_id = job.get("batch_id") if isinstance(job, dict) else None
    if not batch_id:
        raise RuntimeError(
            "Saved batch job is missing its batch_id (corrupt or hand-edited "
            f"{JOB_FILENAME}). Remove it and re-run with --batch_mode submit."
        )
    interval = max(5, int(poll_interval or 60))
    timeout = float(poll_timeout or 0)
    start = time.monotonic()
    inline_mode = job.get("submit_style") == "inline"
    logger.info(
        f"Polling batch {batch_id} every {interval}s "
        + ("(no timeout)." if not timeout else f"(timeout {timeout:.0f}s).")
    )
    try:
        not_found_since = None
        while True:
            try:
                if inline_mode:
                    batch = _retrieve_batch_inline(client, batch_id)
                else:
                    batch = client.batches.retrieve(batch_id)
            except (RuntimeError, ValueError, httpx.HTTPError) as e:
                # Network blips (connection/timeout, no HTTP response) are
                # retried like registration lag; real HTTP errors (4xx/5xx
                # responses) fail fast with the provider's message.
                transient = inline_mode and (
                    _is_transient_retrieve_error(e)
                    or isinstance(e, (httpx.ConnectError, httpx.TimeoutException))
                )
                if transient:
                    if not_found_since is None:
                        not_found_since = time.monotonic()
                    waited = time.monotonic() - not_found_since
                    if waited > _NOT_FOUND_GRACE_S:
                        raise RuntimeError(
                            f"Batch {batch_id} still not readable after "
                            f"{waited:.0f}s of polling ({e}). The id is probably "
                            "wrong (provider-scoped) -- pass the right "
                            "--batch_job_id or re-run with --batch_mode submit."
                        ) from e
                    logger.warning(
                        f"Batch {batch_id} not yet readable at the provider "
                        f"({e}; {waited:.0f}s) -- still initializing; will keep "
                        "polling."
                    )
                    if timeout and (time.monotonic() - start) > timeout:
                        raise TimeoutError(
                            f"Batch {batch_id} still not visible after "
                            f"{timeout:.0f}s. Re-run with --batch_mode poll to "
                            "resume waiting later."
                        )
                    time.sleep(interval)
                    continue
                raise
            not_found_since = None
            if isinstance(batch, dict) and "status" not in batch:
                # Initializing (or wrapped) payload with no status yet: we
                # hold a batch id, so keep polling instead of failing.
                keys = sorted(batch.keys())
                logger.warning(
                    f"Batch {batch_id} returned no status yet (keys: {keys}) "
                    "-- still initializing; will keep polling."
                )
                if timeout and (time.monotonic() - start) > timeout:
                    raise TimeoutError(
                        f"Batch {batch_id} still has no status after "
                        f"{timeout:.0f}s. Re-run with --batch_mode poll to "
                        "resume waiting later."
                    )
                time.sleep(interval)
                continue
            status = _batch_field(batch, "status", "?")
            counts = _batch_field(batch, "request_counts")
            if inline_mode and not isinstance(counts, dict):
                raw_counts = _batch_field(batch, "request_counts", {})
                counts = (
                    {k: v for k, v in raw_counts.items() if v is not None}
                    if isinstance(raw_counts, dict)
                    else None
                ) or None
            logger.info(f"Batch {batch_id}: status={status} counts={counts}")
            if status == _TERMINAL_OK:
                return batch
            if status in _TERMINAL_BAD:
                detail = ""
                if inline_mode:
                    problems = _inline_error_summary(batch, limit=3)
                    if problems:
                        detail = " Per-request errors: " + " | ".join(problems)
                    else:
                        top = _batch_field(batch, "error")
                        top_msg = None
                        if isinstance(top, dict):
                            top_msg = top.get("message") or top.get("code")
                        elif isinstance(top, str):
                            top_msg = top
                        if top_msg:
                            # Providers report batch-level failures here
                            # (e.g. rejected multimodal content).
                            detail = f" Batch error: {str(top_msg)[:300]}"
                        else:
                            # Failed but no parseable errors at all: dump the
                            # raw payload shape (truncated) so the provider's
                            # actual schema can be mapped.
                            try:
                                raw = json.dumps(batch, default=str)
                            except Exception:
                                raw = str(batch)
                            keys = (
                                sorted(batch.keys())
                                if isinstance(batch, dict)
                                else type(batch).__name__
                            )
                            logger.error(
                                f"Batch {batch_id} payload keys: {keys}. Raw "
                                f"payload (truncated): {raw[:2000]}"
                            )
                raise RuntimeError(
                    f"Batch {batch_id} ended with status={status} "
                    f"(counts={counts}).{detail} Check the provider dashboard; "
                    "re-run with --batch_mode submit to build a fresh job."
                )
            if status not in _IN_PROGRESS:
                logger.warning(
                    f"Batch {batch_id}: unexpected status={status}; keep waiting."
                )
            if timeout and (time.monotonic() - start) > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} still {status} after {timeout:.0f}s. "
                    "Re-run with --batch_mode poll to resume waiting later."
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted while batch {batch_id} is still running. The job is "
            "safe on the provider side -- re-run with --batch_mode poll to "
            "resume waiting and finalize."
        )
        raise


def _download_text(client, file_id) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        text = content.text
        if text is None:
            return ""
        return (
            text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
        )
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if hasattr(content, "read"):
        data = content.read()
        if data is None:
            return ""
        return (
            data.decode("utf-8", errors="replace")
            if isinstance(data, bytes)
            else str(data)
        )
    if content is None:
        return ""
    return str(content)


def finalize_batch_job(
    client,
    job,
    output_folder,
    stats,
    checkpoint=None,
    completed_images=None,
    batches_done=None,
):
    """Download the batch output and write YOLO labels (online-path semantics).

    Returns the number of stems finalized. Images whose every box failed at
    the provider level are left out of the checkpoint (like server-failure
    images online) so a fresh ``--batch_mode submit`` can retry them.
    """
    from auto_annotation.checkpoint import next_free_id as _next_free_id

    batch_id = job.get("batch_id") if isinstance(job, dict) else None
    if not batch_id:
        raise RuntimeError(
            "Saved batch job is missing its batch_id (corrupt or hand-edited "
            f"{JOB_FILENAME}). Remove it and re-run with --batch_mode submit."
        )
    if job.get("submit_style") == "inline":
        batch = _retrieve_batch_inline(client, batch_id)
    else:
        batch = client.batches.retrieve(batch_id)
    if _batch_field(batch, "status") != _TERMINAL_OK:
        raise RuntimeError(
            f"Cannot finalize batch {batch_id}: status={_batch_field(batch, 'status', '?')} "
            "(expected 'completed')."
        )
    error_file_id = _batch_field(batch, "error_file_id")
    if error_file_id:
        try:
            err_text = _download_text(client, error_file_id)
            err_lines = [ln for ln in err_text.splitlines() if ln.strip()]
            logger.warning(
                f"Batch {batch_id} reported {len(err_lines)} error line(s); first few: "
                f"{err_lines[:3]}"
            )
        except Exception as e:
            logger.warning(f"Could not download batch error file: {e}")

    inline = _inline_results(batch)
    output_file_id = None
    if inline is not None:
        # OpenRouter-style: results ride along in the retrieve response.
        result_items = [item for item in inline if isinstance(item, dict)]
        logger.info(f"Batch {batch_id} returned {len(result_items)} inlined result(s).")
    else:
        output_file_id = _batch_field(batch, "output_file_id")
        if not output_file_id:
            raise RuntimeError(
                f"Batch {batch_id} is completed but has neither an output file "
                "nor inlined results. Check the provider dashboard."
            )
        text = _download_text(client, output_file_id)
        result_items = [ln for ln in text.splitlines() if ln.strip()]
        logger.info(
            f"Downloaded {len(result_items)} batch result line(s) for batch {batch_id}."
        )

    params = job.get("params", {})
    class_mode = str(params.get("class_mode", "hybrid") or "hybrid")
    none_labels = params.get(
        "none_labels",
        "none,no_detection,nodetection,no_defect,background,unknown,negative,normal",
    )
    drop_none = params.get("drop_none", True)
    conf_threshold = params.get("conf_threshold", 2)
    drop_small_images = params.get("drop_small_images", True)
    inplace_saving = params.get("inplace_saving", False)
    train_label = params.get("train_label", "")
    none_set = _parse_none_labels(none_labels)

    class_map = dict(job.get("class_map", {}))
    stems = job.get("stems", {})

    # sent custom_id -> parsed model result (or failure marker)
    results = {}
    n_failed = 0
    for item in result_items:
        if isinstance(item, str):
            try:
                row = json.loads(item)
            except ValueError:
                try:
                    row = json_repair.loads(item)
                except Exception:
                    logger.warning(
                        f"Unparseable batch output line, skipping: {item[:160]}"
                    )
                    continue
        else:
            row = item
        if not isinstance(row, dict) or "custom_id" not in row:
            continue
        cid = row["custom_id"]
        content = _result_content_text(row)
        if not content:
            err = row.get("error") or row.get("result") or row.get("response") or row
            logger.warning(f"Batch request {cid} failed at provider: {str(err)[:200]}")
            results[cid] = None
            n_failed += 1
            continue
        try:
            parsed = json_repair.loads(content)
            results[cid] = parsed if isinstance(parsed, dict) else None
        except Exception as e:
            logger.warning(
                f"Batch request {cid}: bad response body ({e}), skipping box."
            )
            results[cid] = None
            n_failed += 1
    if n_failed:
        stats.incr("boxes_model_call_failed", n_failed)

    # Sanity: every submitted request must come back with a result row. A
    # foreign --batch_job_id (or a provider that dropped the output) shows
    # up here as zero overlap -- fail loudly instead of marking every image
    # failed and locking the job as done with nothing produced.
    sent_ids = {
        cid for entry in (stems or {}).values() for cid in (entry.get("sent") or {})
    }
    if sent_ids and not (sent_ids & set(results)):
        raise RuntimeError(
            f"Batch {batch_id} returned {len(result_items)} result(s) but none "
            f"match the {len(sent_ids)} submitted request(s) -- wrong batch id "
            "(provider-scoped) or a dropped provider output. Not marking "
            "anything done; re-run with the right --batch_job_id or resubmit."
        )

    if inplace_saving:
        batch_output_folder = Path(train_label)
    else:
        # Stage under <output>/batches/ (NOT labels/): this is the layout
        # the end-of-run flatten scans. Staging inside labels/ would be
        # invisible to flatten (labels/batch_* is legacy-only, labels/*.txt
        # is non-recursive) and the labels would be silently lost.
        batch_output_folder = Path(output_folder) / "batches" / "batch_0000"
    os.makedirs(batch_output_folder, exist_ok=True)
    manifest_path = Path(output_folder) / SKIPPED_MANIFEST

    finalized = 0
    for stem, entry in stems.items():
        img_file = entry.get("img_file", stem)
        sent = entry.get("sent", {})
        kept = entry.get("kept", [])
        skipped_small = int(entry.get("skipped_small", 0) or 0)
        stats.incr("boxes_seen", len(sent) + len(kept) + skipped_small)

        new_label_lines = []
        if kept:
            from auto_annotation.yaml_utils import resolve_kept_class

            stats.incr("boxes_kept_small", len(kept))
            # Kept boxes are never classified: resolve each original id so a
            # taken id mints a fresh one instead of merging into an unrelated
            # class (finalize is single-threaded, no lock needed). Only the
            # class token is rewritten; coordinate text stays verbatim.
            for _kept_line in kept:
                _parts = _kept_line.split()
                if not _parts:
                    continue
                _kept_name, _kept_id, _kept_how = resolve_kept_class(
                    class_map, _parts[0]
                )
                if _kept_how == "invalid":
                    logger.warning(
                        f"{img_file}: kept small box has non-numeric class "
                        f"{_parts[0]!r}; writing the line verbatim."
                    )
                    new_label_lines.append(_kept_line)
                else:
                    new_label_lines.append(" ".join([str(_kept_id)] + _parts[1:]))
                if _kept_how == "remapped":
                    logger.warning(
                        f"{img_file}: kept small box original id {_parts[0]} is "
                        f"taken in the current map -- re-registered as id "
                        f"{_kept_id} ('{_kept_name}') instead of merging into "
                        "an unrelated class."
                    )
                    stats.note_new_class(_kept_name)
                elif _kept_how == "slot":
                    logger.warning(
                        f"{img_file}: kept small box references class id "
                        f"{_parts[0]} missing from the class map; registered "
                        f"as {_kept_name!r} so data.yaml stays trainable."
                    )
                    stats.note_new_class(_kept_name)
                elif _kept_how == "reused":
                    logger.info(
                        f"{img_file}: keeping small box as id {_kept_id} "
                        f"('{_kept_name}') without LLM call."
                    )
        if skipped_small:
            stats.incr("boxes_skipped_small", skipped_small)
        low_conf = []
        resolved = bool(kept)
        failed_this_image = 0

        for cid, meta in sent.items():
            if cid not in results or not isinstance(results[cid], dict):
                failed_this_image += 1
                continue
            result = results[cid]
            class_name = result.get("class")
            if not class_name or not isinstance(class_name, str):
                logger.warning(f"Batch {cid}: invalid class in {result}, skipping box.")
                stats.incr("boxes_bad_response")
                failed_this_image += 1
                continue
            class_name = class_name.strip().lower()
            if not class_name:
                stats.incr("boxes_bad_response")
                failed_this_image += 1
                continue
            if _normalize_label(class_name) in none_set:
                if drop_none:
                    stats.incr("boxes_dropped_none")
                    continue
            mode_norm = (class_mode or "hybrid").lower().strip()
            if class_name not in class_map:
                if mode_norm == "strict":
                    logger.warning(
                        f"[strict mode] Batch returned unknown class {class_name!r} "
                        f"for {img_file}; discarding box."
                    )
                    stats.incr("boxes_bad_response")
                    continue
                class_map[class_name] = _next_free_id(class_map)
                stats.note_new_class(class_name)
            try:
                confidence = int(result.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0
            new_label_lines.append(
                f"{class_map[class_name]} {meta['x']} {meta['y']} {meta['bw']} {meta['bh']}"
            )
            stats.incr("boxes_classified")
            resolved = True
            try:
                if confidence <= int(conf_threshold):
                    stats.incr("boxes_low_confidence")
                    low_conf.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [meta["x"], meta["y"], meta["bw"], meta["bh"]],
                        }
                    )
            except (TypeError, ValueError):
                pass

        if not resolved and failed_this_image > 0:
            # Nothing usable for this image (every sent box failed at the
            # provider): mirror the online server-failure rule -- write
            # nothing and stay out of the checkpoint so a fresh submit retries.
            logger.warning(
                f"{img_file}: all {failed_this_image} batch box(es) failed -> "
                "NOT writing a label file and NOT marking as completed."
            )
            stats.incr("images_failed_server")
            stats.log_progress(img_file)
            continue

        if not new_label_lines and skipped_small > 0 and drop_small_images:
            try:
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(stem + "\n")
            except Exception as e:
                logger.error(f"Failed to append to {manifest_path}: {e}")
            logger.warning(
                f"{img_file}: all {skipped_small} box(es) removed by the small-box "
                f"filter -> NOT writing a label file (listed in {manifest_path})."
            )
            stats.incr("images_skipped_all_small")
        else:
            if inplace_saving:
                out_path = Path(train_label) / (stem + ".txt")
            else:
                out_path = batch_output_folder / (stem + ".txt")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    if new_label_lines:
                        f.write("\n".join(new_label_lines) + "\n")
                    else:
                        logger.info(
                            f"{img_file}: no boxes to write -> empty YOLO label file."
                        )
            except Exception as e:
                logger.error(f"Failed to write {out_path}: {e}")
            if low_conf:
                try:
                    with open(
                        Path(output_folder) / (stem + "_low_confidence.json"), "w"
                    ) as f:
                        json.dump(low_conf, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to write low-confidence file for {stem}: {e}")

        if checkpoint is not None and completed_images is not None:
            completed_images.add(stem)
        finalized += 1
        stats.log_progress(img_file)

    if checkpoint is not None and completed_images is not None:
        if batches_done is not None and not inplace_saving:
            batches_done.add(0)
        checkpoint.save(
            set(completed_images),
            dict(class_map),
            set(batches_done or set()),
            job.get("run_settings"),
        )

    job["phase"] = "done"
    job["finalized"] = finalized
    job["output_file_id"] = output_file_id
    job["class_map"] = dict(class_map)
    save_job(output_folder, job)
    logger.info(
        f"Batch {batch_id} finalized: {finalized}/{len(stems)} image(s) written. "
        "Run the normal end-of-run steps (yaml sync + flatten) to finish."
    )
    return finalized


def run_batch_api_flow(
    args,
    client,
    class_map,
    checkpoint,
    completed_images,
    batches_done,
    stats,
    image_extensions,
    target_height,
    target_width,
    effective_definitions,
):
    """Driver for ``--use_batch_api``: submit and/or poll+finalize per mode."""
    from auto_annotation.stats import RunStats  # noqa: F401  (docs: stats type)

    if getattr(args, "server_type", "external") != "external":
        logger.error(
            "--use_batch_api needs an external OpenAI-compatible provider with "
            f"/v1/batches support, but --server_type is {getattr(args, 'server_type')!r} "
            "(local llama.cpp/vLLM servers do not implement the Batches API). "
            "Re-run with --server_type external (see examples/auto_label_external.example.yaml) "
            "or without --use_batch_api."
        )
        exit(1)
    base_url = str(getattr(args, "base_url", "") or "")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        logger.warning(
            f"--base_url looks local ({base_url}) -- most local servers do not "
            "implement /v1/batches and submit will fail. This is only a warning."
        )

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    mode = str(getattr(args, "batch_mode", "auto") or "auto").lower()
    override_id = getattr(args, "batch_job_id", None) or None

    job = load_job(output_folder)
    if getattr(args, "dry_run", False) and job is not None:
        # A saved job short-circuits the build (and its detailed dry-run
        # message), so guard here: polling/finalizing writes labels and
        # checkpoints, which a dry run must never do.
        logger.info(
            f"[dry run] saved batch job {job.get('batch_id')} "
            f"(phase={job.get('phase')}) exists; nothing polled, finalized, "
            "or written. Re-run without --dry_run to proceed."
        )
        return 0
    if job is not None and mode == "submit":
        # Guard BEFORE the --batch_job_id override below mutates anything: a
        # rejected command must not rewrite the saved job file.
        logger.error(
            f"A batch job ({job.get('batch_id')}, phase={job.get('phase')}) is already "
            f"saved in {output_folder}. Poll/finalize it first (--batch_mode poll) "
            f"or remove {JOB_FILENAME} to submit a fresh one."
        )
        exit(1)
    if override_id:
        if job is None:
            logger.error(
                f"--batch_job_id {override_id} was given but there is no saved job in "
                f"{output_folder}/{JOB_FILENAME} (it holds the request mapping needed "
                "to finalize). Submit first, then poll with --batch_job_id."
            )
            exit(1)
        if job.get("batch_id") != override_id:
            logger.info(
                f"Overriding saved batch {job.get('batch_id')} with --batch_job_id {override_id}."
            )
            job["batch_id"] = override_id
            job["phase"] = "submitted"
            save_job(output_folder, job)

    if mode == "poll" and job is None:
        logger.error(
            f"--batch_mode poll needs a saved job in {output_folder}/{JOB_FILENAME} "
            "(or --batch_job_id with one). Nothing to poll -- submit first."
        )
        exit(1)
    if job is not None and job.get("phase") == "done" and mode in ("auto", "poll"):
        _done_n = job.get("finalized", None)
        _done_note = (
            f" ({_done_n} image(s) written)"
            if isinstance(_done_n, int)
            else " (finalized before result counts were recorded)"
        )
        if _done_n == 0:
            logger.warning(
                f"Saved batch {job.get('batch_id')} is finalized BUT produced "
                "0 images -- likely every box failed at the provider. Nothing "
                "to resume; remove "
                f"{output_folder}/{JOB_FILENAME} (and use --no_auto_resume if "
                "you also want to redo finished images) to submit a fresh job."
            )
        else:
            logger.info(
                f"Saved batch {job.get('batch_id')} is already finalized -- "
                f"nothing to do{_done_note}. "
                "To start a fresh batch job, remove "
                f"{output_folder}/{JOB_FILENAME} (and use --no_auto_resume if "
                "you also want to redo finished images)."
            )
        return 0

    if job is not None:
        logger.info(
            f"Resuming saved batch job {job.get('batch_id')} (phase="
            f"{job.get('phase')}) -- requests were built earlier, so the current "
            "--shuffle/--seed/--num_samples flags do NOT re-select images. To "
            f"build a fresh sample, remove {output_folder}/{JOB_FILENAME} first."
        )
    if job is None:
        # ---- Build ------------------------------------------------------
        with_class_map_lock = list(class_map.keys())
        requests, stems = collect_batch_requests(
            args.train_image,
            args.train_label,
            image_extensions=image_extensions,
            num_samples=args.num_samples,
            shuffle=args.shuffle,
            seed=args.seed,
            start_index=args.start_index,
            end_index=args.end_index,
            target_height=target_height,
            target_width=target_width,
            model_name=args.model,
            class_mode=getattr(args, "class_mode", "hybrid"),
            class_definitions=effective_definitions,
            none_labels=getattr(args, "none_labels", ""),
            drop_none=getattr(args, "drop_none", True),
            extra_body=getattr(args, "extra_body", None),
            min_box_size=getattr(args, "min_box_size", 0) or 0,
            small_box_action=getattr(args, "small_box_action", "keep") or "keep",
            known_names=with_class_map_lock,
            crop_padding_pct=getattr(args, "crop_padding_pct", 0.0) or 0.0,
            recls_context=getattr(args, "recls_context", "crop") or "crop",
            crop_resize_ratio=getattr(args, "crop_resize_ratio", None),
        )
        # Auto-resume: never rebuild/resubmit images the checkpoint says are
        # finished. Without this, deleting .batch_job.json for a fresh sample
        # (as the resume log suggests) would rebill already-done images.
        done = set(completed_images or ())
        if done:
            skipped = sorted(s for s in stems if s in done)
            if skipped:
                logger.info(
                    f"Auto-resume: skipping {len(skipped)} completed image(s) "
                    f"from the fresh build ({', '.join(skipped[:5])}"
                    f"{'...' if len(skipped) > 5 else ''})."
                )
                for s in skipped:
                    del stems[s]
                requests = [
                    r for r in requests if (r.get("meta") or {}).get("stem") not in done
                ]
        stats.images_total = len(stems)
        n_kept = sum(len(e["kept"]) for e in stems.values())
        n_small = sum(e["skipped_small"] for e in stems.values())
        all_small_stems = sum(
            1
            for e in stems.values()
            if not e["sent"] and (e["kept"] or e["skipped_small"])
        )
        logger.info(
            f"Batch build: {len(requests)} request(s) across {len(stems)} image(s) "
            f"({n_kept} small box(es) kept as-is, {n_small} small box(es) filtered, "
            f"{all_small_stems} image(s) with nothing to send)."
        )
        _min_side_cfg = int(getattr(args, "min_box_size", 0) or 0)
        if _min_side_cfg > 0 and n_small == 0 and n_kept == 0:
            logger.warning(
                f"min_box_size={_min_side_cfg}px is set but no box fell below it in "
                "this run. The filter IS active -- the boxes are simply all larger "
                f"than {_min_side_cfg}px. Lower min_box_size (or check it against your "
                "image resolution: the threshold is in ORIGINAL pixel units, so a "
                "value tuned for 1024px images will rarely trigger on 4000px ones)."
            )
        if args.dry_run:
            logger.info(
                "[dry run] batch YOU would submit "
                f"{len(requests)} request(s) for {len(stems)} image(s); no upload, no labels written."
            )
            return 0
        if not requests:
            logger.warning(
                "Batch build produced 0 requests (every box filtered or no inputs). "
                "Nothing to submit."
            )
            return 0
        params = {
            "class_mode": getattr(args, "class_mode", "hybrid"),
            "class_definitions": effective_definitions,
            "none_labels": getattr(args, "none_labels", ""),
            "drop_none": getattr(args, "drop_none", True),
            "conf_threshold": args.conf_threshold,
            "min_box_size": getattr(args, "min_box_size", 0) or 0,
            "small_box_action": getattr(args, "small_box_action", "keep") or "keep",
            "drop_small_images": getattr(args, "drop_small_images", True),
            "extra_body": getattr(args, "extra_body", None),
            "inplace_saving": args.inplace_saving,
            "train_label": args.train_label,
        }
        job = submit_batch_job(
            client,
            output_folder,
            requests,
            stems,
            args.model,
            class_map,
            params,
            completion_window=getattr(args, "batch_completion_window", "24h") or "24h",
            submit_style=getattr(args, "batch_submit_style", "auto") or "auto",
            public_images=getattr(args, "batch_public_images", False),
            image_host=getattr(args, "image_host", "catbox") or "catbox",
            run_settings=build_run_settings(
                crop_padding_pct=getattr(args, "crop_padding_pct", 0.0),
                recls_context=getattr(args, "recls_context", "crop"),
                crop_resize_ratio=getattr(args, "crop_resize_ratio", None),
                min_box_size=getattr(args, "min_box_size", 0),
                small_box_action=getattr(args, "small_box_action", "keep"),
                model=args.model,
                height=target_height,
                width=target_width,
                class_mode=getattr(args, "class_mode", "hybrid"),
                none_labels=getattr(args, "none_labels", ""),
                drop_none=getattr(args, "drop_none", True),
                batch_size=getattr(args, "batch_size", 0),
            ),
        )
        if mode == "submit":
            logger.info(
                f"Submit-only mode: job {job['batch_id']} saved. Come back later with "
                "--batch_mode poll (same command) to finalize it into YOLO labels."
            )
            return 0

    # ---- Poll + finalize -------------------------------------------------
    batch = poll_batch_job(
        client,
        job,
        poll_interval=getattr(args, "batch_poll_interval", 60),
        poll_timeout=getattr(args, "batch_poll_timeout", 0),
    )
    _ = batch
    finalized = finalize_batch_job(
        client,
        job,
        output_folder,
        stats,
        checkpoint=checkpoint,
        completed_images=completed_images,
        batches_done=batches_done,
    )
    # Sync newly discovered classes back so the yaml sync + final log see them.
    try:
        fresh = load_job(output_folder) or {}
        for k, v in dict(fresh.get("class_map", {})).items():
            if k not in class_map:
                class_map[k] = int(v)
    except Exception as e:
        logger.warning(f"Could not sync class_map from finalized batch job: {e}")
    return finalized
