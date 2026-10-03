"""Server-failure safety: no fake-empty labels, abort on dead/OOM server."""

import threading

import pytest
from PIL import Image

from auto_annotation.server_guard import (
    FailureTracker,
    ServerDownError,
    is_server_error,
)


# ---------------------------------------------------------------------------
# is_server_error classification
# ---------------------------------------------------------------------------
class TestIsServerError:
    def test_connection_errors(self):
        assert is_server_error(ConnectionError("connection refused"))
        assert is_server_error(ConnectionError("failed to connect"))
        assert is_server_error(TimeoutError("request timed out"))
        assert is_server_error(RuntimeError("read timeout after 30s"))

    def test_http_5xx(self):
        err = RuntimeError("request failed with status code: 503")
        assert is_server_error(err)
        err = RuntimeError("Internal server error (500) from vllm")
        assert is_server_error(err)

    def test_oom_flavours(self):
        assert is_server_error(RuntimeError("CUDA out of memory"))
        assert is_server_error(RuntimeError("vllm: out of memory, kv cache full"))
        assert is_server_error(RuntimeError("resource exhausted: OOM"))
        assert is_server_error(MemoryError("cannot allocate memory"))

    def test_openai_sdk_style_names(self):
        api_conn = type("APIConnectionError", (Exception,), {})("boom")
        assert is_server_error(api_conn)
        api_timeout = type("APITimeoutError", (Exception,), {})("slow")
        assert is_server_error(api_timeout)
        status_err = type("APIStatusError", (Exception,), {})("500 blah")
        status_err.status_code = 500
        assert is_server_error(status_err)

    def test_per_box_problems_are_not_server_errors(self):
        assert not is_server_error(ValueError("bad json in model response"))
        assert not is_server_error(KeyError("class"))
        assert not is_server_error(RuntimeError("weird crop, skipping"))
        assert not is_server_error(ValueError("No choices returned"))


# ---------------------------------------------------------------------------
# FailureTracker
# ---------------------------------------------------------------------------
class TestFailureTracker:
    def test_trips_after_threshold(self):
        t = FailureTracker(max_consecutive_failures=3)
        assert not t.record_failure()
        assert not t.record_failure()
        assert t.record_failure()
        assert t.tripped
        with pytest.raises(ServerDownError):
            t.check_and_raise("img.jpg")

    def test_success_resets_streak(self):
        t = FailureTracker(max_consecutive_failures=2)
        t.record_failure()
        t.record_success()
        assert t.consecutive == 0
        assert not t.record_failure()
        assert not t.tripped

    def test_min_one(self):
        t = FailureTracker(max_consecutive_failures=0)
        assert t.max_consecutive_failures == 1


# ---------------------------------------------------------------------------
# process_one_image with a dead server
# ---------------------------------------------------------------------------
def _make_dataset(tmp_path, n_images=1, boxes_per_image=1):
    train_image = tmp_path / "images"
    train_label = tmp_path / "labels"
    train_image.mkdir()
    train_label.mkdir()
    names = []
    for i in range(n_images):
        name = f"img{i}.jpg"
        Image.new("RGB", (100, 100), (128, 128, 128)).save(train_image / name)
        lines = "\n".join(["0 0.5 0.5 0.5 0.5"] * boxes_per_image) + "\n"
        (train_label / f"img{i}.txt").write_text(lines)
        names.append(name)
    return train_image, train_label, names


def _call_process_one_image(monkeypatch, tmp_path, img_file, fail=None, **kw):
    """Call process_one_image with detect_defect stubbed. Returns namespace."""
    from auto_annotation.single_image import process_one_image
    from auto_annotation.stats import RunStats

    calls = {"saves": []}

    class FakeCheckpoint:
        def save(self, completed, class_map, batches, run_settings=None):
            calls["saves"].append((set(completed), dict(class_map), set(batches)))

        def save_under_locks(
            self,
            completed_images,
            completed_lock,
            class_map,
            class_map_lock,
            batches_done,
            run_settings=None,
        ):
            self.save(
                set(completed_images or ()),
                dict(class_map or {}),
                set(batches_done or ()),
                run_settings,
            )

    if fail is not None:

        def _boom(*a, **k):
            raise fail

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _boom)
    else:
        monkeypatch.setattr(
            "auto_annotation.single_image.detect_defect",
            lambda *a, **k: {"class": "spot", "confidence": 5},
        )

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    stats = RunStats()
    class_map = {}
    completed = set()
    result = process_one_image(
        img_file,
        str(kw.pop("train_image")),
        str(kw.pop("train_label")),
        str(out),
        class_map,
        threading.Lock(),
        object(),  # client (unused, detect_defect is stubbed)
        "test-model",
        2,  # conf_threshold
        False,  # dry_run
        False,  # resume
        64,  # target_height
        64,  # target_width
        stats,
        False,  # inplace_saving
        checkpoint=FakeCheckpoint(),
        completed_images=completed,
        completed_lock=threading.Lock(),
        batches_done=set(),
        failure_tracker=kw.pop("failure_tracker", None),
    )
    return {
        "result": result,
        "out": out,
        "stats": stats,
        "class_map": class_map,
        "completed": completed,
        "saves": calls["saves"],
    }


class TestDeadServerSingleImage:
    def test_all_boxes_fail_server_writes_nothing(self, tmp_path, monkeypatch):
        train_image, train_label, _ = _make_dataset(tmp_path)
        ns = _call_process_one_image(
            monkeypatch,
            tmp_path,
            "img0.jpg",
            fail=ConnectionError("connection refused"),
            train_image=train_image,
            train_label=train_label,
        )
        assert ns["result"] is None
        # No fake-empty label file...
        assert not (ns["out"] / "img0.txt").exists()
        # ...and nothing checkpointed.
        assert ns["completed"] == set()
        assert ns["saves"] == []
        assert ns["stats"].images_failed_server == 1

    def test_trips_and_raises(self, tmp_path, monkeypatch):
        train_image, train_label, _ = _make_dataset(tmp_path, boxes_per_image=3)
        tracker = FailureTracker(max_consecutive_failures=2)
        with pytest.raises(ServerDownError):
            _call_process_one_image(
                monkeypatch,
                tmp_path,
                "img0.jpg",
                fail=ConnectionError("connection refused"),
                train_image=train_image,
                train_label=train_label,
                failure_tracker=tracker,
            )

    def test_success_still_works(self, tmp_path, monkeypatch):
        train_image, train_label, _ = _make_dataset(tmp_path)
        tracker = FailureTracker(max_consecutive_failures=2)
        ns = _call_process_one_image(
            monkeypatch,
            tmp_path,
            "img0.jpg",
            train_image=train_image,
            train_label=train_label,
            failure_tracker=tracker,
        )
        assert (ns["out"] / "img0.txt").exists()
        assert ns["completed"] == {"img0"}
        assert len(ns["saves"]) == 1
        assert tracker.consecutive == 0

    def test_non_server_errors_forge_nothing(self, tmp_path, monkeypatch):
        """Auth/config errors (non-server) must not write empty labels or
        checkpoint the image -- otherwise one wrong flag wipes the dataset
        and resume skips everything forever."""
        train_image, train_label, _ = _make_dataset(tmp_path)
        ns = _call_process_one_image(
            monkeypatch,
            tmp_path,
            "img0.jpg",
            fail=RuntimeError("401 unauthorized: bad api key"),
            train_image=train_image,
            train_label=train_label,
        )
        assert ns["result"] is None
        assert not (ns["out"] / "img0.txt").exists()
        assert ns["completed"] == set()
        assert ns["saves"] == []
        assert ns["stats"].images_failed_unclassified == 1
        assert ns["stats"].images_failed_server == 0


# ---------------------------------------------------------------------------
# small-box keep registers unknown original ids
# ---------------------------------------------------------------------------
class TestKeepRegistersUnknownId:
    def test_foreign_id_kept_and_registered(self, tmp_path, monkeypatch):
        import threading

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        Image.new("RGB", (100, 100), (128, 128, 128)).save(train_image / "img0.jpg")
        # 50px box on a 100px image, foreign class id 9
        (train_label / "img0.txt").write_text("9 0.5 0.5 0.5 0.5\n")

        def _must_not_run(*a, **k):
            raise AssertionError("kept boxes must never reach the model")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _must_not_run)

        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        stats = RunStats()
        class_map = {}
        completed = set()
        saves = []

        class FakeCheckpoint:
            def save(self, completed, class_map, batches, run_settings=None):
                saves.append(dict(run_settings or {}))

            def save_under_locks(
                self,
                completed_images,
                completed_lock,
                class_map,
                class_map_lock,
                batches_done,
                run_settings=None,
            ):
                self.save(
                    set(completed_images or ()),
                    dict(class_map or {}),
                    set(batches_done or ()),
                    run_settings,
                )

        process_one_image(
            "img0.jpg",
            str(train_image),
            str(train_label),
            str(out),
            class_map,
            threading.Lock(),
            object(),
            "test-model",
            2,
            False,
            False,
            64,
            64,
            stats,
            False,
            checkpoint=FakeCheckpoint(),
            completed_images=completed,
            completed_lock=threading.Lock(),
            batches_done=set(),
            min_box_size=60,
            small_box_action="keep",
        )
        # verbatim line kept AND the unknown id registered for data.yaml
        assert (out / "img0.txt").read_text() == "9 0.5 0.5 0.5 0.5\n"
        assert class_map == {"original_class_9": 9}
        # checkpoint carries the settings fingerprint
        assert saves and saves[0].get("model") == "test-model"

    def test_kept_box_keeps_original_name(self, tmp_path, monkeypatch):
        """With orig_names (data.yaml), a kept box whose original name still
        holds its id keeps the ORIGINAL name and id verbatim: no synthetic
        class, no map mutation, model never called."""
        import threading

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        Image.new("RGB", (100, 100), (128, 128, 128)).save(train_image / "img0.jpg")
        # 50px box on a 100px image, original class id 0 == 'hole'
        (train_label / "img0.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        def _must_not_run(*a, **k):
            raise AssertionError("kept boxes must never reach the model")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _must_not_run)

        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        stats = RunStats()
        class_map = {"hole": 0}
        process_one_image(
            "img0.jpg",
            str(train_image),
            str(train_label),
            str(out),
            class_map,
            threading.Lock(),
            object(),
            "test-model",
            2,
            False,
            False,
            64,
            64,
            stats,
            False,
            min_box_size=60,
            small_box_action="keep",
            orig_names=["hole"],
        )
        assert (out / "img0.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"
        assert class_map == {"hole": 0}

    def test_taken_id_remapped_to_fresh(self, tmp_path, monkeypatch):
        """A kept box whose old id is taken gets a FRESH id (no silent merge
        into an unrelated class): old binary id 1 meant 'defect', but the new
        map already has 1 -> 'dog', so the box becomes 2 ('original_class_1').
        Coordinates stay byte-identical; the model is never called."""
        import threading

        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        Image.new("RGB", (100, 100), (128, 128, 128)).save(train_image / "img0.jpg")
        (train_label / "img0.txt").write_text("1 0.5 0.5 0.5 0.5\n")

        def _must_not_run(*a, **k):
            raise AssertionError("kept boxes must never reach the model")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _must_not_run)

        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        class_map = {"cat": 0, "dog": 1}
        process_one_image(
            "img0.jpg",
            str(train_image),
            str(train_label),
            str(out),
            class_map,
            threading.Lock(),
            object(),
            "test-model",
            2,
            False,
            False,
            64,
            64,
            RunStats(),
            False,
            min_box_size=60,
            small_box_action="keep",
        )
        assert (out / "img0.txt").read_text() == "2 0.5 0.5 0.5 0.5\n"
        assert class_map == {"cat": 0, "dog": 1, "original_class_1": 2}


# ---------------------------------------------------------------------------
# batch_runner aborts the whole run
# ---------------------------------------------------------------------------
class TestBatchRunnerAbort:
    def test_run_aborts_on_dead_server(self, tmp_path, monkeypatch):
        from auto_annotation.batch_runner import read_images_with_labels
        from auto_annotation.checkpoint import CheckpointManager
        from auto_annotation.stats import RunStats

        train_image, train_label, _ = _make_dataset(tmp_path, n_images=3)

        def _boom(*a, **k):
            raise ConnectionError("connection refused")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _boom)

        out = tmp_path / "run_out"
        out.mkdir()
        stats = RunStats()
        completed, batches = set(), set()
        with pytest.raises(ServerDownError):
            read_images_with_labels(
                str(train_image),
                str(train_label),
                {},
                object(),
                "test-model",
                str(out),
                max_workers=1,
                batch_size=0,
                checkpoint=CheckpointManager(str(out)),
                completed_images=completed,
                batches_done=batches,
                stats=stats,
                auto_save=lambda: None,
                max_consecutive_failures=2,
                abort_on_server_down=True,
            )
        # Nothing completed, no batch marked done, no checkpoint file.
        assert completed == set()
        assert batches == set()
        assert not (out / ".checkpoint.json").exists()

    def test_abort_disabled_leaves_images_unmarked(self, tmp_path, monkeypatch):
        from auto_annotation.batch_runner import read_images_with_labels
        from auto_annotation.checkpoint import CheckpointManager
        from auto_annotation.stats import RunStats

        train_image, train_label, _ = _make_dataset(tmp_path, n_images=2)

        def _boom(*a, **k):
            raise ConnectionError("connection refused")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _boom)

        out = tmp_path / "run_out2"
        out.mkdir()
        completed, batches = set(), set()
        # Must NOT raise with abort disabled...
        read_images_with_labels(
            str(train_image),
            str(train_label),
            {},
            object(),
            "test-model",
            str(out),
            max_workers=1,
            batch_size=0,
            checkpoint=CheckpointManager(str(out)),
            completed_images=completed,
            batches_done=batches,
            stats=RunStats(),
            auto_save=lambda: None,
            abort_on_server_down=False,
        )
        # ...but failed images are still NOT marked completed (retried later).
        assert completed == set()
        # And no fake-empty label files were written.
        assert list((out / "labels").glob("*.txt")) == []
