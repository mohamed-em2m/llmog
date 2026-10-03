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


class TestResolveKeptClass:
    def test_free_id_keeps_slot(self):
        from auto_annotation.yaml_utils import resolve_kept_class

        m = {"hole": 0}
        assert resolve_kept_class(m, 5) == ("original_class_5", 5, "slot")
        assert m == {"hole": 0, "original_class_5": 5}

    def test_own_registration_reused(self):
        from auto_annotation.yaml_utils import resolve_kept_class

        m = {"hole": 0, "original_class_5": 5}
        assert resolve_kept_class(m, 5) == ("original_class_5", 5, "reused")
        assert m == {"hole": 0, "original_class_5": 5}

    def test_taken_id_mints_fresh(self):
        from auto_annotation.yaml_utils import resolve_kept_class

        m = {"cat": 0, "dog": 1}
        name, new_id, how = resolve_kept_class(m, 1)
        assert how == "remapped"
        assert new_id == 2 and m[name] == 2 and name.startswith("original_class_1")

    def test_invalid_and_reserved_prefix(self):
        from auto_annotation.yaml_utils import resolve_kept_class

        assert resolve_kept_class({}, "abc") == (None, None, "invalid")
        # The original_class_<id> pattern is reserved for quarantines: a
        # pre-existing entry with that name is treated as ours and reused,
        # never suffixed into a second class.
        m = {"original_class_5": 0}
        assert resolve_kept_class(m, 5) == ("original_class_5", 0, "reused")
        assert m == {"original_class_5": 0}

    def test_repeated_remaps_share_one_quarantine(self):
        from auto_annotation.yaml_utils import resolve_kept_class

        # Regression: kept boxes whose old id is taken (e.g. id 0 already
        # means 'hole') must ALL land on one quarantine id -- previously
        # every box minted a fresh original_class_0_N id.
        m = {"hole": 0}
        first = resolve_kept_class(m, 0)
        assert first[2] == "remapped"
        assert first[0] == "original_class_0" and first[1] == 1
        for _ in range(5):
            assert resolve_kept_class(m, 0) == (first[0], first[1], "reused")
        assert m == {"hole": 0, "original_class_0": 1}


class TestSaveUnderLocks:
    def test_concurrent_saves_lose_nothing(self, tmp_path):
        """8 workers add disjoint stems, then all save: the final file must
        contain every stem (a stale snapshot overwriting newer progress
        would drop some). Join timeouts turn a deadlock into a failure."""
        import threading

        from auto_annotation.checkpoint import CheckpointManager

        mgr = CheckpointManager(tmp_path)
        completed, class_map = set(), {}
        completed_lock, class_map_lock = threading.Lock(), threading.Lock()
        barrier = threading.Barrier(8)
        errors = []

        def worker(w):
            try:
                with completed_lock:
                    for i in range(25):
                        completed.add(f"w{w}-img{i}")
                with class_map_lock:
                    class_map[f"cls{w}"] = w
                barrier.wait(timeout=30)
                mgr.save_under_locks(
                    completed,
                    completed_lock,
                    class_map,
                    class_map_lock,
                    set(),
                    {"model": "m"},
                )
            except Exception as e:  # noqa: BLE001 - collected across threads
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(w,), daemon=True) for w in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors
        assert all(not t.is_alive() for t in threads), "deadlock in save_under_locks"
        data = mgr.load()
        assert len(data["completed_images"]) == 8 * 25
        assert len(data["class_map"]) == 8
        assert data["run_settings"] == {"model": "m"}
