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
    labels = tmp_path / "labels" / "batches" / "batch_0000"
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
    big_lbl = tmp_path / "labels" / "batches" / "batch_0000" / "big.txt"
    assert big_lbl.is_file() and "0 0.7 0.7 0.5 0.5" in big_lbl.read_text()
    # tiny: all-small -> NO label file + manifest entry
    assert not (tmp_path / "labels" / "batches" / "batch_0000" / "tiny.txt").exists()
    assert "tiny" in (tmp_path / "skipped_small_images.txt").read_text().split()
    assert stats.images_skipped_all_small == 1
    assert stats.boxes_classified == 1


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
    assert not (tmp_path / "labels" / "batches" / "batch_0000" / "big.txt").exists()
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
    assert (tmp_path / "labels" / "batches" / "batch_0000" / "big.txt").is_file()
    assert (
        tmp_path / "labels" / "batches" / "batch_0000" / "big.txt"
    ).stat().st_size == 0
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
