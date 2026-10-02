"""Resume integrity: checkpoint is authoritative for class ids on resume."""

import json

from auto_annotation.checkpoint import (
    CheckpointManager,
    build_run_settings,
    next_free_id,
    run_settings_mismatches,
    validate_checkpoint_data,
)
from auto_annotation.yaml_utils import find_renumbered, save_updated_yaml


class TestNextFreeId:
    def test_empty(self):
        assert next_free_id({}) == 0

    def test_contiguous(self):
        assert next_free_id({"a": 0, "b": 1}) == 2

    def test_gap_uses_max_plus_one_not_len(self):
        # len() would return 2 and collide with id 5; max()+1 gives 6.
        assert next_free_id({"a": 0, "b": 5}) == 6


class TestValidate:
    def test_valid(self):
        w, e = validate_checkpoint_data(
            {"completed_images": ["x"], "class_map": {"a": 0}, "batches_done": []}
        )
        assert e == []

    def test_duplicate_ids_rejected(self):
        _, errors = validate_checkpoint_data(
            {
                "completed_images": [],
                "class_map": {"a": 0, "b": 0},
                "batches_done": [],
            }
        )
        assert any("duplicate" in m for m in errors)

    def test_non_dict_rejected(self):
        _, errors = validate_checkpoint_data(
            {"completed_images": [], "class_map": [1, 2], "batches_done": []}
        )
        assert errors != []

    def test_gap_warns_but_valid(self):
        warnings, errors = validate_checkpoint_data(
            {
                "completed_images": [],
                "class_map": {"a": 0, "b": 5},
                "batches_done": [],
            }
        )
        assert errors == []
        assert any("gaps" in m for m in warnings)


class TestCheckpointRoundtrip:
    def test_save_load_normalizes_ids(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.save({"img1"}, {"a": 0, "b": 1}, {0})
        data = mgr.load()
        assert set(data["completed_images"]) == {"img1"}
        assert data["class_map"] == {"a": 0, "b": 1}
        assert data["batches_done"] == [0]

    def test_corrupt_checkpoint_returns_none(self, tmp_path):
        p = tmp_path / ".checkpoint.json"
        p.write_text("{not valid json")
        assert CheckpointManager(tmp_path).load() is None

    def test_duplicate_id_checkpoint_rejected(self, tmp_path):
        p = tmp_path / ".checkpoint.json"
        p.write_text(
            json.dumps(
                {
                    "completed_images": [],
                    "class_map": {"a": 0, "b": 0},
                    "batches_done": [],
                }
            )
        )
        assert CheckpointManager(tmp_path).load() is None


class TestRunSettingsFingerprint:
    def test_save_load_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        fp = build_run_settings(
            crop_padding_pct=50,
            recls_context="crop",
            crop_resize_ratio=None,
            min_box_size=0,
            small_box_action="keep",
            model="m",
            height=1024,
            width=1024,
            class_mode="hybrid",
        )
        mgr.save({"img1"}, {"a": 0}, set(), fp)
        data = mgr.load()
        assert data["run_settings"]["crop_padding_pct"] == 50
        assert data["run_settings"]["crop_resize_ratio"] is None

    def test_old_checkpoint_without_fingerprint_loads(self, tmp_path):
        p = tmp_path / ".checkpoint.json"
        p.write_text(
            json.dumps(
                {
                    "completed_images": ["x"],
                    "class_map": {"a": 0},
                    "batches_done": [],
                }
            )
        )
        data = CheckpointManager(tmp_path).load()
        assert data is not None
        assert (
            run_settings_mismatches(
                data.get("run_settings"), build_run_settings(crop_padding_pct=50)
            )
            == []
        )

    def test_detects_changed_keys(self):
        saved = build_run_settings(
            crop_padding_pct=0,
            recls_context="crop",
            min_box_size=0,
            model="m",
            class_mode="hybrid",
        )
        current = build_run_settings(
            crop_padding_pct=50,
            recls_context="full_som",
            min_box_size=0,
            model="m",
            class_mode="hybrid",
        )
        diffs = run_settings_mismatches(saved, current)
        assert any(d.startswith("crop_padding_pct:") for d in diffs)
        assert any(d.startswith("recls_context:") for d in diffs)
        assert not any(d.startswith("min_box_size:") for d in diffs)

    def test_numeric_string_equivalence(self):
        saved = {"crop_padding_pct": 0, "height": 1024}
        current = {"crop_padding_pct": 0.0, "height": "1024"}
        assert run_settings_mismatches(saved, current) == []

    def test_unset_to_set_detected(self):
        saved = build_run_settings(crop_resize_ratio=None, model="m")
        current = build_run_settings(crop_resize_ratio=1.5, model="m")
        diffs = run_settings_mismatches(saved, current)
        assert any(d.startswith("crop_resize_ratio:") for d in diffs)


class TestYamlGuard:
    def test_no_renumber(self):
        assert find_renumbered(["a", "b"], {"a": 0, "b": 1, "c": 2}) == []

    def test_detects_renumber(self):
        out = find_renumbered(["a", "b"], {"a": 1, "b": 0})
        assert ("a", 0, 1) in out and ("b", 1, 0) in out

    def test_save_keeps_checkpoint_order(self, tmp_path):
        import yaml

        yp = tmp_path / "data.yaml"
        yp.write_text(yaml.safe_dump({"names": ["b", "a"], "nc": 2}))
        # Checkpoint says a:0, b:1 (opposite order) -> file re-synced to
        # checkpoint order with a warning, not silently kept stale.
        save_updated_yaml(
            str(yp), str(tmp_path), {"names": ["b", "a"]}, {"a": 0, "b": 1}
        )
        data = yaml.safe_load(yp.read_text())
        assert data["names"] == ["a", "b"]
        assert data["nc"] == 2
        out = yaml.safe_load((tmp_path / "data.yaml").read_text())
        assert out["names"] == ["a", "b"]
