"""Public image hosting for inline batch requests.

Inline batch hosts (e.g. OpenRouter) accept images as public http(s) URLs
only -- base64 / data: URI parts are rejected on every provider. This module
uploads JPEG crop bytes to a free anonymous host and rewrites request bodies
to reference the public URLs, with a content-hash cache so resumes never
re-upload.

WARNING: uploaded crops are world-readable to anyone with the URL. Never
enable this for sensitive or private datasets; prefer sync mode
(``--use_batch_api`` off), which sends base64 privately per request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time

import httpx

from auto_annotation.logging_utils import logger

CACHE_FILENAME = ".uploaded_images.json"
_CATBOX_API = "https://catbox.moe/user/api.php"


def upload_jpeg_to_catbox(
    jpeg_bytes: bytes, timeout: float = 60.0, retries: int = 3
) -> str:
    """Upload JPEG bytes to catbox.moe, return the public URL. Retries."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0)) as http:
                resp = http.post(
                    _CATBOX_API,
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": ("crop.jpg", jpeg_bytes, "image/jpeg")},
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"catbox upload failed with {resp.status_code}: {resp.text[:200]}"
                )
            url = resp.text.strip()
            if not url.startswith("http"):
                raise RuntimeError(f"catbox returned no URL: {resp.text[:200]}")
            return url
        except Exception as e:
            last_err = e
            logger.warning(
                f"Image upload attempt {attempt}/{retries} failed ({e}); retrying."
            )
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Image upload failed after {retries} attempts: {last_err}")


def _load_cache(output_folder) -> dict:
    p = os.path.join(str(output_folder), CACHE_FILENAME)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(output_folder, cache: dict) -> None:
    p = os.path.join(str(output_folder), CACHE_FILENAME)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, p)


def rewrite_bodies_to_public_urls(
    requests,
    output_folder,
    uploader=upload_jpeg_to_catbox,
) -> dict:
    """Replace data-URI image parts with public URLs, using a hash cache.

    Returns ``{"uploaded": int, "reused": int}``. Bodies without data URIs
    are left untouched. Upload failures raise (fail fast: a partial rewrite
    would submit a guaranteed-dead batch).
    """
    from auto_annotation.logging_utils import logger as _log

    _log.warning(
        "Uploading crops to a PUBLIC third-party host: anyone with a URL can "
        "view them. Do not use --batch_public_images for sensitive data."
    )
    cache = _load_cache(output_folder)
    uploaded = reused = 0
    dirty = False
    for req in requests or []:
        body = (req or {}).get("body") or {}
        for msg in body.get("messages", []) or []:
            content = (msg or {}).get("content")
            parts = content if isinstance(content, list) else []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                iu = part.get("image_url")
                if not isinstance(iu, dict):
                    continue
                url = iu.get("url")
                if not isinstance(url, str) or not url.startswith("data:"):
                    continue
                try:
                    _, b64 = url.split(",", 1)
                    jpeg_bytes = base64.b64decode(b64)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not decode data-URI image for "
                        f"{(req or {}).get('custom_id', '?')}: {e}"
                    )
                digest = hashlib.sha256(jpeg_bytes).hexdigest()
                public = cache.get(digest)
                if public:
                    reused += 1
                else:
                    public = uploader(jpeg_bytes)
                    cache[digest] = public
                    dirty = True
                    uploaded += 1
                iu["url"] = public
    if dirty:
        _save_cache(output_folder, cache)
    _log.info(
        f"Image hosting: {uploaded} crop(s) uploaded, {reused} reused from "
        f"cache ({CACHE_FILENAME})."
    )
    return {"uploaded": uploaded, "reused": reused}
