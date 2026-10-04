"""Failed images are recorded in the checkpoint and retried on resume.

Covers: failure recording (sync path), legacy --resume bypass for failed
stems, success clearing the record, the whole-batch-done gate, and
fingerprint preservation across failure saves.
"""

import json
import threading

import cv2
import numpy as np
from PIL import Image


def _make_dataset(tmp_path, stems=("img0",)):
    train_image = tmp_path / "images"
    train_label = tmp_path / "labels"
    train_image.mkdir(exist_ok=True)
    train_label.mkdir(exist_ok=True)
    for stem in stems:
        Image.new("RGB", (100, 100), (128, 128, 128)).save(train_image / f"{stem}.jpg")
        (train_label / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    return train_image, train_label


class TestFailureRecording:
    def test_model_failure_recorded_and_checkpointed(self, tmp_path, monkeypatch):
        from auto_annotation.checkpoint import CheckpointManager
        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        train_image, train_label = _make_dataset(tmp_path)
        out = tmp_path / "out"
        out.mkdir()

        def _boom(*a, **k):
            raise ValueError("bad model name")

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _boom)

        checkpoint = CheckpointManager(str(out))
        completed, class_map, failed = set(), {}, {}
        stats = RunStats()
        result = process_one_image(
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
            checkpoint=checkpoint,
            completed_images=completed,
            completed_lock=threading.Lock(),
            batches_done=set(),
            failed_images=failed,
        )
        assert result is None
        assert completed == set()
        assert "img0" in failed
        assert "model errors" in failed["img0"]["reason"]
        on_disk = json.loads((out / ".checkpoint.json").read_text())
        assert "img0" in on_disk["failed_images"]
        assert on_disk["completed_images"] == []

    def test_failed_bypasses_legacy_resume_and_clears_on_success(
        self, tmp_path, monkeypatch
    ):
        """A stale output file from a failed run must not count as done;
        success clears the failure record."""
        from auto_annotation.single_image import process_one_image
        from auto_annotation.stats import RunStats

        train_image, train_label = _make_dataset(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        (out / "img0.txt").write_text("stale partial\n")

        monkeypatch.setattr(
            "auto_annotation.single_image.detect_defect",
            lambda *a, **k: {"class": "spot", "confidence": 5},
        )

        from auto_annotation.checkpoint import CheckpointManager

        completed, class_map = set(), {}
        failed = {"img0": {"reason": "previous boom"}}
        stats = RunStats()
        result = process_one_image(
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
            True,  # legacy --resume would skip on the stale file alone
            64,
            64,
            stats,
            False,
            checkpoint=CheckpointManager(str(out)),
            completed_images=completed,
            completed_lock=threading.Lock(),
            batches_done=set(),
            failed_images=failed,
        )
        assert result is not None
        assert (out / "img0.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"
        assert "img0" in completed
        assert "img0" not in failed
        assert class_map == {"spot": 0}

    def test_batch_with_failure_not_marked_done(self, tmp_path, monkeypatch):
        """A batch containing an uncompleted failure is re-entered on resume."""
        from auto_annotation.batch_runner import read_images_with_labels
        from auto_annotation.checkpoint import CheckpointManager
        from auto_annotation.stats import RunStats

        train_image = tmp_path / "images"
        train_label = tmp_path / "labels"
        train_image.mkdir()
        train_label.mkdir()
        for stem in ("img_a", "img_b"):
            cv2.imwrite(
                str(train_image / f"{stem}.jpg"),
                np.full((100, 100, 3), 128, dtype=np.uint8),
            )
            (train_label / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        out = tmp_path / "out"
        out.mkdir()

        calls = []

        def _flaky(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("first box fails")
            return {"class": "spot", "confidence": 5}

        monkeypatch.setattr("auto_annotation.single_image.detect_defect", _flaky)

        checkpoint = CheckpointManager(str(out))
        completed, class_map, batches_done, failed = set(), {}, set(), {}
        read_images_with_labels(
            str(train_image),
            str(train_label),
            class_map,
            object(),
            "test-model",
            str(out),
            stats=RunStats(),
            batch_size=0,
            checkpoint=checkpoint,
            completed_images=completed,
            batches_done=batches_done,
            max_workers=1,
            target_height=64,
            target_width=64,
            failed_images=failed,
        )
        assert "img_b" in completed
        assert "img_a" in failed and "img_a" not in completed
        assert batches_done == set()  # NOT done: failures remain
        on_disk = json.loads((out / ".checkpoint.json").read_text())
        assert "img_a" in on_disk["failed_images"]


class TestCheckpointFailedKeys:
    def test_roundtrip_preserves_failed(self, tmp_path):
        from auto_annotation.checkpoint import CheckpointManager

        mgr = CheckpointManager(str(tmp_path))
        mgr.save({"a"}, {"x": 0}, set(), {"model": "m"}, {"b": {"reason": "boom"}})
        data = mgr.load()
        assert data["failed_images"] == {"b": {"reason": "boom"}}
        assert data["run_settings"] == {"model": "m"}

    def test_partial_save_preserves_other_keys(self, tmp_path):
        from auto_annotation.checkpoint import CheckpointManager

        mgr = CheckpointManager(str(tmp_path))
        mgr.save({"a"}, {"x": 0}, set(), {"model": "m"}, {"b": {"reason": "x"}})
        # A save that knows nothing new carries the old keys over.
        mgr.save({"a", "c"}, {"x": 0}, set())
        data = mgr.load()
        assert data["run_settings"] == {"model": "m"}
        assert data["failed_images"] == {"b": {"reason": "x"}}

    def test_malformed_failed_tolerated(self):
        from auto_annotation.checkpoint import validate_checkpoint_data

        warnings, errors = validate_checkpoint_data(
            {
                "completed_images": [],
                "class_map": {},
                "batches_done": [],
                "failed_images": ["not", "a", "dict"],
            }
        )
        assert errors == []
        assert any("failed_images" in w for w in warnings)
