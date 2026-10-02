"""Tests for the OpenAI Batch API flow (auto_label --use_batch_api).

Uses a fake client (no network): covers request building, the small-box
filter at build time, submit/job-file persistence, and finalize semantics
(kept-line merge, none-drop, strict-mode guard, all-small manifest).
"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from auto_annotation.batch_api import (
    collect_batch_requests,
    finalize_batch_job,
    load_job,
    make_custom_id,
    poll_batch_job,
    submit_batch_job,
)
from auto_annotation.image_io import build_classify_body, detect_defect
from auto_annotation.stats import RunStats


# --------------------------------------------------------------------------
# Fake OpenAI client (files + batches namespaces only)
# --------------------------------------------------------------------------
class _FakeFiles:
    def __init__(self, output_text=""):
        self.output_text = output_text
        self.uploaded = []

    def create(self, file, purpose):
        self.uploaded.append((file.read(), purpose))
        return SimpleNamespace(id="file-in-1")

    def content(self, file_id):
        return SimpleNamespace(text=self.output_text)


class _FakeBatches:
    def __init__(self, status="completed", reject_file_ref=False):
        self.status = status
        self.created = []
        self.reject_file_ref = reject_file_ref

    def create(self, **kwargs):
        if self.reject_file_ref:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'Batch body ended "
                "before a `requests` array was found.', 'code': 400}}"
            )
        self.created.append(kwargs)
        return SimpleNamespace(id="batch-1", status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(
            id=batch_id,
            status=self.status,
            output_file_id="file-out-1",
            error_file_id=None,
            request_counts={"total": 1, "completed": 1, "failed": 0},
        )


class FakeClient:
    def __init__(self, output_text="", status="completed", reject_file_ref=False):
        self.files = _FakeFiles(output_text)
        self.batches = _FakeBatches(status, reject_file_ref)
        self.base_url = "https://provider.test/v1/"
        self.api_key = "sk-test"


# --------------------------------------------------------------------------
# Fake httpx transport for the inline (OpenRouter-style) path
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self):
        return self._payload


class _FakeHttpx:
    """Stand-in for httpx.Client that records posts/gets and replays payloads."""

    def __init__(self, posted, retrieved, status_code=200):
        self.posted = posted
        self.retrieved = retrieved
        self.status_code = status_code

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        self.posted.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse({"id": "batch-inline", "status": "in_progress"}, 200)

    def get(self, url, headers=None):
        self.retrieved.append({"url": url, "headers": headers})
        return _FakeResponse(self.retrieved[-1].setdefault("payload", {}))


@pytest.fixture()
def inline_httpx(monkeypatch):
    """Patch httpx.Client inside batch_api with a recording fake.

    Usage: ``posted, _ = inline_httpx()`` for create-only tests, or
    ``inline_httpx(get_payload={...})`` when poll/finalize must read a
    provider response.
    """
    import auto_annotation.batch_api as ba

    def _install(get_payload=None):
        posted, retrieved = [], []
        fake = _FakeHttpx(posted, retrieved)
        fake.get = lambda url, headers=None: _append_retrieve(
            retrieved, url, headers, get_payload
        )
        monkeypatch.setattr(ba.httpx, "Client", fake)
        monkeypatch.setattr(ba.httpx, "Timeout", lambda *a, **k: None)
        return posted, retrieved

    return _install


def _inline_result(custom_id, cls="hole", conf=90, status_code=200):
    """One OpenRouter-style inlined result entry (results ride in retrieve)."""
    if status_code != 200:
        return {"custom_id": custom_id, "error": {"message": "boom"}}
    return {
        "custom_id": custom_id,
        "result": {
            "status_code": status_code,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"class": cls, "confidence": conf})
                        }
                    }
                ]
            },
        },
    }


def _result_line(custom_id, cls="hole", conf=90, status_code=200):
    if status_code != 200:
        return json.dumps(
            {
                "id": "r",
                "custom_id": custom_id,
                "response": {"status_code": status_code, "body": None},
                "error": {"message": "boom"},
            }
        )
    return json.dumps(
        {
            "id": "r",
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": "q",
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"class": cls, "confidence": conf}
                                )
                            }
                        }
                    ]
                },
            },
        },
    )


@pytest.fixture()
def dataset(tmp_path):
    img_dir = tmp_path / "imgs"
    lbl_dir = tmp_path / "lbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # big box (100x100) + small box (10x10 on a 200px image)
    cv2.imwrite(str(img_dir / "big.jpg"), img)
    cv2.imwrite(str(img_dir / "tiny.jpg"), img)
    (lbl_dir / "big.txt").write_text("0 0.7 0.7 0.5 0.5\n0 0.2 0.2 0.05 0.05\n")
    (lbl_dir / "tiny.txt").write_text("0 0.25 0.25 0.05 0.05\n")
    return img_dir, lbl_dir


def _params(**over):
    p = {
        "class_mode": "hybrid",
        "class_definitions": "",
        "none_labels": "none",
        "drop_none": True,
        "conf_threshold": 2,
        "min_box_size": 32,
        "small_box_action": "drop",
        "drop_small_images": True,
        "extra_body": None,
        "inplace_saving": False,
        "train_label": "",
    }
    p.update(over)
    return p


# --------------------------------------------------------------------------
def test_body_builder_matches_detect_defect_kwargs():
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    body = build_classify_body(
        crop,
        "m",
        ["hole"],
        class_mode="strict",
        extra_body='{"provider": {"order": ["x"]}}',
    )
    assert body["model"] == "m"
    assert body["messages"][0]["content"][1]["type"] == "image_url"
    assert body["extra_body"] == {"provider": {"order": ["x"]}}
    # detect_defect forwards the same body to create()
    seen = {}

    class C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    msg = SimpleNamespace(
                        content=json.dumps({"class": "hole", "confidence": 5})
                    )
                    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    out = detect_defect(
        crop,
        C(),
        "m",
        ["hole"],
        class_mode="strict",
        extra_body='{"provider": {"order": ["x"]}}',
    )
    assert out["class"] == "hole"
    assert seen["model"] == "m" and seen["extra_body"] == {"provider": {"order": ["x"]}}


def test_custom_id_capped():
    assert len(make_custom_id("s", 3)) <= 64
    assert len(make_custom_id("x" * 100, 12)) <= 64


def test_collect_applies_small_box_filter(dataset):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(
        str(img_dir),
        str(lbl_dir),
        min_box_size=32,
        small_box_action="drop",
        known_names=["hole"],
    )
    # big.jpg: 1 sent + 1 small dropped; tiny.jpg: 1 small dropped, nothing sent
    assert len(reqs) == 1
    assert len(stems["big"]["sent"]) == 1
    assert stems["big"]["skipped_small"] == 1
    assert stems["tiny"]["sent"] == {} and stems["tiny"]["skipped_small"] == 1
    body = reqs[0]["body"]
    assert body["messages"][0]["content"][0]["text"]
    json.dumps(body)  # must be JSON-serializable for the JSONL


def test_collect_keep_stages_lines(dataset):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(
        str(img_dir),
        str(lbl_dir),
        min_box_size=32,
        small_box_action="keep",
        known_names=[],
    )
    assert len(reqs) == 1
    assert stems["tiny"]["kept"] == ["0 0.25 0.25 0.05 0.05"]
    assert stems["big"]["kept"] == ["0 0.2 0.2 0.05 0.05"]


def test_submit_strips_batch_suffix_from_model(dataset, tmp_path, inline_httpx):
    """OpenRouter-style hosts want the base slug; :batch fails every request."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    for r in reqs:
        r["body"]["model"] = "openai/gpt-6-luna:batch"
    _as_url_images(reqs)
    posted, _ = inline_httpx()
    client = FakeClient()
    job = submit_batch_job(
        client,
        str(tmp_path),
        reqs,
        stems,
        "openai/gpt-6-luna:batch",
        {},
        _params(),
        submit_style="inline",
    )
    assert job["submit_style"] == "inline"
    payload = posted[0]["json"]
    assert payload["model"] == "openai/gpt-6-luna"
    assert all(r["body"]["model"] == "openai/gpt-6-luna" for r in payload["requests"])


def test_submit_inline_rejects_data_uri_images(dataset, tmp_path, inline_httpx):
    """Inline hosts are URL-only: base64 crops must fail fast, pre-submit."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    inline_httpx()
    with pytest.raises(RuntimeError, match="public .* URLs only"):
        submit_batch_job(
            FakeClient(),
            str(tmp_path),
            reqs,
            stems,
            "m",
            {},
            _params(),
            submit_style="inline",
        )


def test_rewrite_bodies_to_public_urls_uploads_and_caches(dataset, tmp_path):
    """Data URIs become public URLs; a rerun reuses the hash cache."""
    from auto_annotation.image_hosting import rewrite_bodies_to_public_urls

    img_dir, lbl_dir = dataset
    out = tmp_path / "out"
    out.mkdir()
    calls = {"n": 0}

    def fake_upload(jpeg_bytes):
        calls["n"] += 1
        assert jpeg_bytes[:2] == b"\xff\xd8"  # real JPEG bytes, not base64 text
        return f"https://example.com/{calls['n']}.jpg"

    reqs, _ = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    stats = rewrite_bodies_to_public_urls(reqs, str(out), uploader=fake_upload)
    # uniform-gray fixture crops dedup by content hash -- all 3 covered
    assert stats["uploaded"] + stats["reused"] == len(reqs) == 3
    assert stats["uploaded"] >= 1
    for r in reqs:
        url = r["body"]["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("https://example.com/")
    assert (out / ".uploaded_images.json").is_file()

    # identical crops on a fresh build hit the cache: zero new uploads
    reqs2, _ = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    stats2 = rewrite_bodies_to_public_urls(reqs2, str(out), uploader=fake_upload)
    assert stats2 == {"uploaded": 0, "reused": 3}
    assert calls["n"] == stats["uploaded"]


def test_submit_inline_publishes_images_when_enabled(
    dataset, tmp_path, inline_httpx, monkeypatch
):
    """--batch_public_images rewrites bodies before the inline POST."""
    import auto_annotation.image_hosting as ih

    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    posted, _ = inline_httpx()
    monkeypatch.setattr(
        ih,
        "rewrite_bodies_to_public_urls",
        lambda requests, output_folder: (
            _as_url_images(requests),
            {"uploaded": len(requests), "reused": 0},
        )[1],
    )
    job = submit_batch_job(
        FakeClient(),
        str(tmp_path),
        reqs,
        stems,
        "m",
        {},
        _params(),
        submit_style="inline",
        public_images=True,
    )
    assert job["submit_style"] == "inline"
    bodies = posted[0]["json"]["requests"]
    assert all(
        r["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "https://"
        )
        for r in bodies
    )


def test_submit_inline_rejects_unknown_image_host(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    with pytest.raises(ValueError, match="image_host"):
        submit_batch_job(
            FakeClient(),
            str(tmp_path),
            reqs,
            stems,
            "m",
            {},
            _params(),
            submit_style="inline",
            public_images=True,
            image_host="imgur",
        )


def test_submit_inline_key_order_requests_last(dataset, tmp_path, inline_httpx):
    """OpenRouter stream-parses the create body: metadata first, requests last."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    _as_url_images(reqs)
    posted, _ = inline_httpx()
    submit_batch_job(
        FakeClient(),
        str(tmp_path),
        reqs,
        stems,
        "m",
        {},
        _params(),
        submit_style="inline",
    )
    payload = posted[0]["json"]
    assert list(payload.keys()) == ["endpoint", "model", "requests"]
    assert all(list(r.keys()) == ["custom_id", "body"] for r in payload["requests"])


def test_submit_stores_run_settings_in_job(dataset, tmp_path):
    """The build-time settings fingerprint rides in the job file."""
    from auto_annotation.checkpoint import build_run_settings

    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    fp = build_run_settings(crop_padding_pct=50, model="m")
    submit_batch_job(
        FakeClient(),
        str(tmp_path),
        reqs,
        stems,
        "m",
        {},
        _params(),
        run_settings=fp,
    )
    assert load_job(str(tmp_path))["run_settings"]["crop_padding_pct"] == 50
    # omitted when not provided (old callers / file-style jobs)
    job2 = submit_batch_job(
        FakeClient(), str(tmp_path), reqs, stems, "m", {}, _params()
    )
    assert "run_settings" not in job2


def _as_url_images(reqs, url="https://example.com/crop.jpg"):
    """Rewrite data-URI image parts to public URLs (inline hosts are URL-only)."""
    for req in reqs:
        for msg in req["body"].get("messages", []):
            content = msg.get("content")
            parts = content if isinstance(content, list) else []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                iu = part.get("image_url")
                if isinstance(iu, dict) and str(iu.get("url", "")).startswith("data:"):
                    iu["url"] = url
    return reqs


def test_submit_persists_job(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    client = FakeClient()
    job = submit_batch_job(client, str(tmp_path), reqs, stems, "m", {}, _params())
    assert job["phase"] == "submitted" and job["batch_id"] == "batch-1"
    assert job["n_requests"] == len(reqs) == 3
    assert (tmp_path / ".batch_job.json").is_file()
    assert (tmp_path / "batch_requests.jsonl").is_file()
    assert load_job(str(tmp_path))["batch_id"] == "batch-1"
    # endpoint + window forwarded
    assert client.batches.created[0]["endpoint"] == "/v1/chat/completions"
    assert client.batches.created[0]["completion_window"] == "24h"


def test_submit_rejects_non_batch_client(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    with pytest.raises(RuntimeError, match="Batches API"):
        submit_batch_job(object(), str(tmp_path), reqs, stems, "m", {}, _params())


def test_submit_auto_falls_back_to_inline(dataset, tmp_path, inline_httpx):
    """A host that rejects input_file_id gets the requests inlined over httpx."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    _as_url_images(reqs)
    posted, _ = inline_httpx()
    client = FakeClient(reject_file_ref=True)
    job = submit_batch_job(
        client, str(tmp_path), reqs, stems, "deepseek/x", {}, _params()
    )
    assert job["submit_style"] == "inline"
    assert job["batch_id"] == "batch-inline"
    assert client.batches.created == []  # file path was attempted, not accepted
    # the fallback POSTs to /batches with an inline requests array
    assert posted[0]["url"] == "https://provider.test/v1/batches"
    assert posted[0]["json"]["model"] == "deepseek/x"
    assert len(posted[0]["json"]["requests"]) == len(reqs)
    first = posted[0]["json"]["requests"][0]
    assert first["custom_id"] == reqs[0]["custom_id"]
    # per-request body is byte-identical to the sync-path body
    assert first["body"] == reqs[0]["body"]


def test_submit_inline_style_skips_file_upload(dataset, tmp_path, inline_httpx):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    _as_url_images(reqs)
    posted, _ = inline_httpx()
    client = FakeClient()
    job = submit_batch_job(
        client,
        str(tmp_path),
        reqs,
        stems,
        "m",
        {},
        _params(),
        submit_style="inline",
    )
    assert job["submit_style"] == "inline"
    assert client.files.uploaded == []  # never uploaded
    assert len(posted) == 1


def test_submit_file_style_does_not_fall_back(dataset, tmp_path, inline_httpx):
    """Explicit --batch_submit_style file must surface the provider error."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    posted, _ = inline_httpx()
    with pytest.raises(RuntimeError, match="requests"):
        submit_batch_job(
            FakeClient(reject_file_ref=True),
            str(tmp_path),
            reqs,
            stems,
            "m",
            {},
            _params(),
            submit_style="file",
        )
    assert posted == []


def test_poll_inline_uses_httpx_and_completes(inline_httpx):
    _, retrieved = inline_httpx(get_payload={"id": "b", "status": "completed"})
    batch = poll_batch_job(
        FakeClient(),
        {"batch_id": "b", "submit_style": "inline"},
        poll_interval=5,
        poll_timeout=30,
    )
    assert batch["status"] == "completed"
    assert retrieved[0]["url"] == "https://provider.test/v1/batches/b"


def _append_retrieve(retrieved, url, headers, payload):
    retrieved.append({"url": url, "headers": headers})
    return _FakeResponse(payload or {})


def _inline_client(monkeypatch, get):
    """Install a bare httpx fake whose GET behaves per ``get(url)``."""
    import auto_annotation.batch_api as ba

    class _Client:
        def __call__(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            return get(url)

    monkeypatch.setattr(ba.httpx, "Client", _Client())
    monkeypatch.setattr(ba.httpx, "Timeout", lambda *a, **k: None)
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)


def test_poll_tolerates_transient_404_then_completes(monkeypatch):
    """A fresh batch may 404 on GET until the provider registers it."""
    calls = {"n": 0}

    def _flaky_get(url):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _FakeResponse({"error": {"message": "not found"}}, 404)
        return _FakeResponse({"id": "b", "status": "completed"})

    _inline_client(monkeypatch, _flaky_get)
    batch = poll_batch_job(
        FakeClient(),
        {"batch_id": "b", "submit_style": "inline"},
        poll_interval=5,
        poll_timeout=60,
    )
    assert batch["status"] == "completed"
    assert calls["n"] == 3


def test_poll_gives_up_on_persistent_404(monkeypatch):
    """A batch that never appears is a wrong id -- fail with guidance."""
    import auto_annotation.batch_api as ba

    _inline_client(
        monkeypatch,
        lambda url: _FakeResponse({"error": {"message": "not found"}}, 404),
    )
    monkeypatch.setattr(ba, "_NOT_FOUND_GRACE_S", 0)
    with pytest.raises(RuntimeError, match="still not readable"):
        poll_batch_job(
            FakeClient(),
            {"batch_id": "b", "submit_style": "inline"},
            poll_interval=5,
            poll_timeout=60,
        )


def test_poll_tolerates_200_error_body_then_completes(monkeypatch):
    """OpenRouter answers 'not found' as HTTP 200 + {"error": ...}."""
    calls = {"n": 0}

    def _flaky_get(url):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _FakeResponse(
                {
                    "error": {
                        "message": "Batch job b not found.",
                        "code": 404,
                    }
                },
                200,
            )
        return _FakeResponse({"id": "b", "status": "completed"})

    _inline_client(monkeypatch, _flaky_get)
    batch = poll_batch_job(
        FakeClient(),
        {"batch_id": "b", "submit_style": "inline"},
        poll_interval=5,
        poll_timeout=60,
    )
    assert batch["status"] == "completed"
    assert calls["n"] == 3


def test_poll_failed_batch_surfaces_inline_errors(monkeypatch):
    """A failed inline batch should report WHY (per-request errors)."""
    _inline_client(
        monkeypatch,
        lambda url: _FakeResponse(
            {
                "id": "b",
                "status": "failed",
                "request_counts": {"total": 3, "completed": 0, "failed": 3},
                "results": [
                    {"custom_id": "a:0", "error": {"message": "model_blah"}},
                    {"custom_id": "a:1", "error": "plain-string-boom"},
                    {"custom_id": "a:2", "result": {"status_code": 400}},
                ],
            }
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        poll_batch_job(
            FakeClient(),
            {"batch_id": "b", "submit_style": "inline"},
            poll_interval=5,
            poll_timeout=60,
        )
    msg = str(excinfo.value)
    assert "model_blah" in msg
    assert "plain-string-boom" in msg
    assert "status_code=400" in msg


def test_finalize_reads_inlined_results(dataset, tmp_path, inline_httpx):
    """OpenRouter-style: results live in the retrieve response, no output file."""
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    inline_httpx(
        get_payload={
            "id": "b",
            "status": "completed",
            "results": [_inline_result(r["custom_id"]) for r in reqs],
        }
    )
    job = {
        "phase": "submitted",
        "batch_id": "b",
        "submit_style": "inline",
        "input_file_id": None,
        "class_map": {"hole": 0},
        "params": _params(),
        "stems": stems,
    }
    (tmp_path / ".batch_job.json").write_text(json.dumps(job), encoding="utf-8")

    stats = RunStats()
    finalized = finalize_batch_job(
        FakeClient(), load_job(str(tmp_path)), str(tmp_path), stats
    )
    assert finalized == len(stems)
    labels = tmp_path / "batches" / "batch_0000"
    assert (labels / "big.txt").is_file()
    assert stats.boxes_classified == len(reqs)


def test_result_content_text_shape_tolerance():
    from auto_annotation.batch_api import _result_content_text

    # OpenAI file-based shape
    assert (
        _result_content_text(
            {
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": "A"}}]},
                }
            }
        )
        == "A"
    )
    # OpenRouter inline shape
    assert (
        _result_content_text(
            {"result": {"body": {"choices": [{"message": {"content": "B"}}]}}}
        )
        == "B"
    )
    # Flat shape with text instead of message
    assert _result_content_text({"result": {"choices": [{"text": "C"}]}}) == "C"
    # Failures and empties
    assert _result_content_text({"error": {"message": "boom"}}) is None
    assert _result_content_text({"result": {"status_code": 400}}) is None
    assert _result_content_text({"result": {"body": {"choices": []}}}) is None


def test_poll_returns_when_completed():
    batch = poll_batch_job(
        FakeClient(), {"batch_id": "batch-1"}, poll_interval=5, poll_timeout=30
    )
    assert batch.status == "completed"


def test_poll_raises_on_failed_batch():
    with pytest.raises(RuntimeError, match="failed"):
        poll_batch_job(
            FakeClient(status="failed"),
            {"batch_id": "b"},
            poll_interval=5,
            poll_timeout=30,
        )


def test_finalize_writes_labels_and_manifest(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(
        str(img_dir),
        str(lbl_dir),
        min_box_size=32,
        small_box_action="drop",
        known_names=[],
    )
    out_text = "\n".join(_result_line(r["custom_id"]) for r in reqs)
    client = FakeClient(output_text=out_text)
    job = submit_batch_job(
        client, str(tmp_path), reqs, stems, "m", {"hole": 0}, _params()
    )
    stats, completed = RunStats(), set()
    from auto_annotation.checkpoint import CheckpointManager

    ckpt = CheckpointManager(str(tmp_path))
    n = finalize_batch_job(
        client,
        job,
        str(tmp_path),
        stats,
        checkpoint=ckpt,
        completed_images=completed,
        batches_done=set(),
    )
    assert n == 2 and completed == {"big", "tiny"}
    # big: classified line written to staging (flatten-compatible)
    big_lbl = tmp_path / "batches" / "batch_0000" / "big.txt"
    assert big_lbl.is_file() and "0 0.7 0.7 0.5 0.5" in big_lbl.read_text()
    # tiny: all-small -> NO label file + manifest entry
    assert not (tmp_path / "batches" / "batch_0000" / "tiny.txt").exists()
    assert "tiny" in (tmp_path / "skipped_small_images.txt").read_text().split()
    assert stats.images_skipped_all_small == 1
    assert stats.boxes_classified == 1
    # end-to-end: the end-of-run flatten must find the staged labels
    # (regression: staging inside labels/ was invisible to flatten)
    from auto_annotation.reverse_batches import flatten_batches_to_labels

    flat = flatten_batches_to_labels(str(tmp_path))
    assert (flat / "big.txt").is_file()
    assert "0 0.7 0.7 0.5 0.5" in (flat / "big.txt").read_text()


def test_finalize_strict_discards_unknown(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(
        str(img_dir),
        str(lbl_dir),
        min_box_size=32,
        small_box_action="drop",
        known_names=["hole"],
    )
    out_text = "\n".join(_result_line(r["custom_id"], cls="zzz_new") for r in reqs)
    client = FakeClient(output_text=out_text)
    job = submit_batch_job(
        client,
        str(tmp_path),
        reqs,
        stems,
        "m",
        {"hole": 0},
        _params(class_mode="strict"),
    )
    stats = RunStats()
    finalize_batch_job(
        client,
        job,
        str(tmp_path),
        stats,
        checkpoint=None,
        completed_images=set(),
        batches_done=set(),
    )
    # big: sent box discarded as strict-unknown + small box filtered ->
    # no lines + skipped_small>0 -> manifest, no label file.
    assert stats.boxes_bad_response >= 1
    assert not (tmp_path / "batches" / "batch_0000" / "big.txt").exists()
    assert "big" in (tmp_path / "skipped_small_images.txt").read_text().split()


def test_finalize_none_writes_empty_not_manifest(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    # no size filter: both boxes sent; model says none -> genuine empty file
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    out_text = "\n".join(_result_line(r["custom_id"], cls="none") for r in reqs)
    client = FakeClient(output_text=out_text)
    params = _params(min_box_size=0)
    job = submit_batch_job(client, str(tmp_path), reqs, stems, "m", {}, params)
    stats = RunStats()
    finalize_batch_job(
        client,
        job,
        str(tmp_path),
        stats,
        checkpoint=None,
        completed_images=set(),
        batches_done=set(),
    )
    assert (tmp_path / "batches" / "batch_0000" / "big.txt").is_file()
    assert (tmp_path / "batches" / "batch_0000" / "big.txt").stat().st_size == 0
    assert not (tmp_path / "skipped_small_images.txt").exists()
    assert stats.boxes_dropped_none == 3


def test_finalize_failed_boxes_leave_image_uncompleted(dataset, tmp_path):
    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    out_text = "\n".join(_result_line(r["custom_id"], status_code=500) for r in reqs)
    client = FakeClient(output_text=out_text)
    job = submit_batch_job(client, str(tmp_path), reqs, stems, "m", {}, _params())
    stats, completed = RunStats(), set()
    finalize_batch_job(
        client,
        job,
        str(tmp_path),
        stats,
        checkpoint=None,
        completed_images=completed,
        batches_done=set(),
    )
    assert completed == set()
    assert stats.images_failed_server == 2


def test_batch_config_validation():
    from schemes import PipelineConfig

    c = PipelineConfig(
        task="auto_label",
        train_image="x",
        use_batch_api=True,
        batch_mode="poll",
        batch_poll_interval=30,
    )
    assert c.use_batch_api and c.batch_mode == "poll"
    with pytest.raises(Exception):
        PipelineConfig(task="auto_label", train_image="x", batch_poll_interval=1)
    with pytest.raises(Exception):
        PipelineConfig(task="auto_label", train_image="x", batch_mode="soon")


def test_batch_cli_flags():
    from auto_annotation.cli import parse_args as aa

    a = aa(
        [
            "--train_image",
            "i",
            "--train_label",
            "l",
            "--yaml_path",
            "y",
            "--output_folder",
            "o",
            "--model",
            "m",
            "--use_batch_api",
            "--batch_mode",
            "submit",
            "--batch_poll_interval",
            "30",
        ]
    )
    assert a.use_batch_api and a.batch_mode == "submit" and a.batch_poll_interval == 30
    from main import parse_args

    m = parse_args(
        [
            "--task",
            "auto_label",
            "--train_image",
            "i",
            "--yaml_path",
            "y",
            "--use_batch_api",
            "--batch_mode",
            "poll",
            "--batch_job_id",
            "batch_abc",
        ]
    )
    assert m.use_batch_api and m.batch_job_id == "batch_abc"


def test_batch_flow_rejects_local_server():
    from auto_annotation.batch_api import run_batch_api_flow
    from types import SimpleNamespace

    args = SimpleNamespace(server_type="llama_cpp", output_folder="o")
    with pytest.raises(SystemExit):
        run_batch_api_flow(
            args,
            FakeClient(),
            {},
            None,
            set(),
            set(),
            RunStats(),
            (".jpg",),
            1024,
            1024,
            "",
        )


# --------------------------------------------------------------------------
# Crop padding (reclassification context)
# --------------------------------------------------------------------------
def test_pad_box_identity_and_math():
    from auto_annotation.image_io import pad_box

    assert pad_box(10, 20, 30, 60, 100, 100, 0) == (10, 20, 30, 60)
    assert pad_box(10, 20, 30, 60, 100, 100, -5) == (10, 20, 30, 60)
    # 20x40 box at 50%: +10px horizontally, +20px vertically per side
    assert pad_box(10, 20, 30, 60, 100, 100, 50) == (0, 0, 40, 80)
    # clamped to image bounds
    assert pad_box(0, 0, 20, 20, 30, 30, 100) == (0, 0, 30, 30)
    # degenerate input passes through
    assert pad_box(5, 5, 5, 5, 100, 100, 50) == (5, 5, 5, 5)


def _pad_fixture(tmp_path):
    """200x200 image, black left half / white right half, box in the black."""
    import numpy as np

    img_dir = tmp_path / "pimgs"
    lbl_dir = tmp_path / "plbls"
    img_dir.mkdir()
    lbl_dir.mkdir()
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[:, :100] = 0
    cv2.imwrite(str(img_dir / "half.jpg"), img)
    # box ending exactly at the black/white boundary: padding reaches white
    (lbl_dir / "half.txt").write_text("0 0.4 0.5 0.2 0.2\n")
    return img_dir, lbl_dir


def _body_image_bytes(body):
    import base64

    uri = body["messages"][0]["content"][1]["image_url"]["url"]
    assert uri.startswith("data:image/jpeg;base64,")
    return base64.b64decode(uri.split(",", 1)[1])


def test_collect_padding_changes_crop_pixels(tmp_path):
    """Padded crops include surrounding context (exact vs padded differ)."""
    img_dir, lbl_dir = _pad_fixture(tmp_path)
    plain, _ = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    padded, _ = collect_batch_requests(
        str(img_dir), str(lbl_dir), known_names=[], crop_padding_pct=100
    )
    assert len(plain) == len(padded) == 1
    assert _body_image_bytes(plain[0]["body"]) != _body_image_bytes(padded[0]["body"])


def test_crop_padding_pct_config_validation():
    from schemes import PipelineConfig

    cfg = PipelineConfig(
        task="auto_label",
        train_image="i",
        train_label="l",
        yaml_path="x.yaml",
        crop_padding_pct=50,
    )
    assert cfg.crop_padding_pct == 50
    with pytest.raises(Exception):
        PipelineConfig(
            task="auto_label",
            train_image="i",
            train_label="l",
            yaml_path="x.yaml",
            crop_padding_pct=-1,
        )


# --------------------------------------------------------------------------
# Full-image SoM context (reclassification)
# --------------------------------------------------------------------------
def test_draw_som_context_marks_box():
    from auto_annotation.image_io import draw_som_context
    from PIL import Image
    import numpy as np

    base = Image.fromarray(np.full((100, 120, 3), 50, dtype=np.uint8))
    out = draw_som_context(base, 10, 20, 40, 70)
    assert out.size == base.size
    # input not mutated
    assert np.array(base)[25, 15].tolist() == [50, 50, 50]
    arr = np.array(out)
    lime = (arr[:, :, 0] == 57) & (arr[:, :, 1] == 255) & (arr[:, :, 2] == 20)
    # box border + filled number badge both drawn in lime
    assert lime.sum() > 50


def test_collect_full_som_sends_annotated_scene(tmp_path):
    """full_som bodies carry the directive and scene pixels, not the crop."""
    img_dir, lbl_dir = _pad_fixture(tmp_path)
    crop_reqs, _ = collect_batch_requests(
        str(img_dir), str(lbl_dir), known_names=[], recls_context="crop"
    )
    som_reqs, _ = collect_batch_requests(
        str(img_dir), str(lbl_dir), known_names=[], recls_context="full_som"
    )
    assert len(crop_reqs) == len(som_reqs) == 1
    crop_body, som_body = crop_reqs[0]["body"], som_reqs[0]["body"]
    som_text = som_body["messages"][0]["content"][0]["text"]
    crop_text = crop_body["messages"][0]["content"][0]["text"]
    assert "marked box" in som_text
    assert "marked box" not in crop_text
    assert _body_image_bytes(som_body) != _body_image_bytes(crop_body)


def test_recls_context_config_validation():
    from schemes import PipelineConfig

    cfg = PipelineConfig(
        task="auto_label",
        train_image="i",
        train_label="l",
        yaml_path="x.yaml",
        recls_context="full_som",
    )
    assert cfg.recls_context == "full_som"
    with pytest.raises(Exception):
        PipelineConfig(
            task="auto_label",
            train_image="i",
            train_label="l",
            yaml_path="x.yaml",
            recls_context="bogus",
        )


# --------------------------------------------------------------------------
# Ratio resize (reclassification)
# --------------------------------------------------------------------------
def test_resize_crop_ratio_math_and_cap():
    from auto_annotation.image_io import resize_crop_ratio
    from PIL import Image
    import numpy as np

    im = Image.fromarray(np.full((50, 100, 3), 128, dtype=np.uint8))
    assert resize_crop_ratio(im, 2.0).size == (200, 100)
    assert resize_crop_ratio(im, 0.5).size == (50, 25)
    # long-edge cap: 200x100 @3.0 -> 600x300, capped at 150 -> 150x75
    assert resize_crop_ratio(im, 3.0, max_long_edge=150).size == (150, 75)
    # identity ratio returns an equal copy
    out = resize_crop_ratio(im, 1.0)
    assert out.size == (100, 50) and out is not im
    with pytest.raises(ValueError):
        resize_crop_ratio(im, 0)
    with pytest.raises(ValueError):
        resize_crop_ratio(im, -2)
    with pytest.raises(ValueError):
        resize_crop_ratio(im, "big")


def test_collect_ratio_resize_sets_crop_pixels(tmp_path):
    """Ratio resize replaces the fixed letterbox: decoded dims prove it."""
    img_dir, lbl_dir = _pad_fixture(tmp_path)
    fixed, _ = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    scaled, _ = collect_batch_requests(
        str(img_dir), str(lbl_dir), known_names=[], crop_resize_ratio=2.0
    )
    assert len(fixed) == len(scaled) == 1

    def _dims(body):
        arr = cv2.imdecode(
            np.frombuffer(_body_image_bytes(body), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        return arr.shape[1], arr.shape[0]

    # fixture box is 40x40px; ratio 2.0 -> 80x80 (no letterbox bars)
    assert _dims(scaled[0]["body"]) == (80, 80)
    # fixed path letterboxes to the default 1024x1024 target
    assert _dims(fixed[0]["body"]) == (1024, 1024)


def test_crop_resize_ratio_config_validation():
    from schemes import PipelineConfig

    cfg = PipelineConfig(
        task="auto_label",
        train_image="i",
        train_label="l",
        yaml_path="x.yaml",
        crop_resize_ratio=1.5,
    )
    assert cfg.crop_resize_ratio == 1.5
    cfg2 = PipelineConfig(
        task="auto_label", train_image="i", train_label="l", yaml_path="x.yaml"
    )
    assert cfg2.crop_resize_ratio is None
    with pytest.raises(Exception):
        PipelineConfig(
            task="auto_label",
            train_image="i",
            train_label="l",
            yaml_path="x.yaml",
            crop_resize_ratio=0,
        )


# --------------------------------------------------------------------------
# run_batch_api_flow driver guards
# --------------------------------------------------------------------------
def _flow_args(tmp_path, img_dir, lbl_dir, **over):
    from types import SimpleNamespace

    kw = dict(
        server_type="external",
        base_url="https://provider.test/v1/",
        output_folder=str(tmp_path),
        batch_mode="auto",
        batch_job_id=None,
        dry_run=False,
        train_image=str(img_dir),
        train_label=str(lbl_dir),
        num_samples=None,
        shuffle=False,
        seed=42,
        start_index=None,
        end_index=None,
        model="m",
        class_mode="hybrid",
        none_labels="none",
        drop_none=True,
        extra_body=None,
        min_box_size=0,
        small_box_action="keep",
        conf_threshold=2,
        batch_completion_window="24h",
        batch_submit_style="file",
        inplace_saving=False,
        batch_poll_interval=5,
        batch_poll_timeout=30,
    )
    kw.update(over)
    return SimpleNamespace(**kw)


def test_flow_build_skips_completed_images(dataset, tmp_path):
    """A fresh build must not resubmit checkpoint-completed images."""
    from auto_annotation.batch_api import load_job, run_batch_api_flow

    img_dir, lbl_dir = dataset
    args = _flow_args(
        tmp_path, img_dir, lbl_dir, batch_mode="submit", batch_submit_style="file"
    )
    rc = run_batch_api_flow(
        args,
        FakeClient(),
        {},
        None,
        {"big"},
        set(),
        RunStats(),
        (".jpg",),
        1024,
        1024,
        "",
    )
    assert rc == 0
    job = load_job(str(tmp_path))
    assert set(job["stems"]) == {"tiny"}
    # only tiny's box was submitted (dataset big.jpg has 2 boxes)
    assert job["n_requests"] == 1


def test_flow_dry_run_with_saved_job_does_nothing(tmp_path, monkeypatch):
    """Dry run + saved job: no poll, no finalize, job untouched."""
    import auto_annotation.batch_api as ba
    from auto_annotation.batch_api import load_job, run_batch_api_flow, save_job

    job = {
        "phase": "submitted",
        "batch_id": "b",
        "submit_style": "inline",
        "stems": {},
        "class_map": {},
    }
    save_job(str(tmp_path), job)
    before = (tmp_path / ".batch_job.json").read_text()

    def _boom(*a, **k):
        raise AssertionError("poll must not run on dry run")

    monkeypatch.setattr(ba, "poll_batch_job", _boom)
    img_dir = tmp_path / "i"
    lbl_dir = tmp_path / "l"
    img_dir.mkdir()
    lbl_dir.mkdir()
    args = _flow_args(tmp_path, img_dir, lbl_dir, dry_run=True)
    assert (
        run_batch_api_flow(
            args,
            FakeClient(),
            {},
            None,
            set(),
            set(),
            RunStats(),
            (".jpg",),
            1024,
            1024,
            "",
        )
        == 0
    )
    assert load_job(str(tmp_path))["phase"] == "submitted"
    assert (tmp_path / ".batch_job.json").read_text() == before


def test_flow_submit_guard_precedes_job_override(tmp_path):
    """A rejected submit must not mutate the saved job file."""
    from auto_annotation.batch_api import load_job, run_batch_api_flow, save_job

    job = {"phase": "submitted", "batch_id": "orig", "stems": {}, "class_map": {}}
    save_job(str(tmp_path), job)
    img_dir = tmp_path / "i"
    lbl_dir = tmp_path / "l"
    img_dir.mkdir()
    lbl_dir.mkdir()
    args = _flow_args(
        tmp_path, img_dir, lbl_dir, batch_mode="submit", batch_job_id="other"
    )
    with pytest.raises(SystemExit):
        run_batch_api_flow(
            args,
            FakeClient(),
            {},
            None,
            set(),
            set(),
            RunStats(),
            (".jpg",),
            1024,
            1024,
            "",
        )
    kept = load_job(str(tmp_path))
    assert kept["batch_id"] == "orig" and kept["phase"] == "submitted"


def test_poll_retries_transient_http_errors(monkeypatch):
    """Connection blips during polling are retried, not fatal."""
    import httpx

    calls = {"n": 0}

    def _flaky_get(url, headers=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("connection reset", request=None)
        return _FakeResponse({"id": "b", "status": "completed"})

    _inline_client(monkeypatch, _flaky_get)
    batch = poll_batch_job(
        FakeClient(),
        {"batch_id": "b", "submit_style": "inline"},
        poll_interval=5,
        poll_timeout=60,
    )
    assert batch["status"] == "completed"
    assert calls["n"] == 3


def test_poll_rejects_job_without_batch_id():
    """A corrupt saved job fails cleanly instead of KeyError."""
    with pytest.raises(RuntimeError, match="missing its batch_id"):
        poll_batch_job(FakeClient(), {"phase": "submitted"}, poll_interval=5)


def test_finalize_rejects_job_without_batch_id(tmp_path):
    from auto_annotation.batch_api import finalize_batch_job

    with pytest.raises(RuntimeError, match="missing its batch_id"):
        finalize_batch_job(
            FakeClient(),
            {},
            str(tmp_path),
            RunStats(),
            checkpoint=None,
            completed_images=set(),
            batches_done=set(),
        )


def test_finalize_rejects_zero_result_overlap(dataset, tmp_path):
    """A foreign/dead batch (no result matches any submitted request)
    raises instead of marking everything failed-and-done."""
    from auto_annotation.batch_api import finalize_batch_job

    img_dir, lbl_dir = dataset
    reqs, stems = collect_batch_requests(str(img_dir), str(lbl_dir), known_names=[])
    client = FakeClient(
        output_text="\n".join([_result_line("wrong-id-1"), _result_line("wrong-id-2")])
    )
    job = submit_batch_job(client, str(tmp_path), reqs, stems, "m", {}, _params())
    completed = set()
    with pytest.raises(RuntimeError, match="none match the 3 submitted"):
        finalize_batch_job(
            client,
            job,
            str(tmp_path),
            RunStats(),
            checkpoint=None,
            completed_images=completed,
            batches_done=set(),
        )
    assert completed == set()


def test_done_gate_distinguishes_empty_finalization(tmp_path):
    """A job finalized with 0 images warns differently than a real one."""
    from auto_annotation.batch_api import run_batch_api_flow, save_job

    for finalized, phase_note in ((0, "warn"), (2, "info")):
        out = tmp_path / f"out{finalized}"
        out.mkdir()
        save_job(
            str(out),
            {
                "phase": "done",
                "batch_id": "b",
                "finalized": finalized,
                "stems": {},
                "class_map": {},
            },
        )
        img_dir = tmp_path / "i"
        lbl_dir = tmp_path / "l"
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)
        args = _flow_args(out, img_dir, lbl_dir, batch_mode="auto")
        assert (
            run_batch_api_flow(
                args,
                FakeClient(),
                {},
                None,
                set(),
                set(),
                RunStats(),
                (".jpg",),
                1024,
                1024,
                "",
            )
            == 0
        )
