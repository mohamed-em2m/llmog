"""
Persists run progress to '<output_folder>/.checkpoint.json' so an
interrupted run (crash, OOM, ctrl-C, preemption, ...) can be resumed
automatically by just re-running the same command.

The checkpoint records:
  - completed_images: image stems that are fully relabeled and written
  - class_map:        name -> id, so newly-discovered classes survive
                       a restart with the same ids (doesn't reshuffle
                       ids already burned into previously-written label
                       files)
  - batches_done:     batch indices that are fully finished, so a
                       resumed run can skip a whole batch folder without
                       even checking each image inside it individually
  - failed_images:    stem -> {"reason": str} for images that failed
                       (read errors, server/model failures, write
                       failures). A resumed run ALWAYS retries these --
                       even when a stale output file exists (the legacy
                       --resume file check is bypassed for them) -- and a
                       batch containing one is never marked done. Entries
                       are cleared the moment the image completes.
  - run_settings:     fingerprint of the label-affecting settings that
                       produced the finished labels (crop padding/context/
                       resize, size filter, model, resolution, class mode).
                       On resume the current flags are compared against it
                       and every change is WARNING-logged, so a run that
                       tweaks e.g. --crop_padding_pct mid-dataset cannot
                       silently mix labels produced under different settings.

Writes are atomic (write to a temp file, then os.replace) so a crash
mid-write can never leave a corrupt/partial checkpoint behind.
"""

import json
import os
import threading
from pathlib import Path

from auto_annotation.logging_utils import logger


def next_free_id(class_map) -> int:
    """Return the next unused class id (max(ids) + 1, or 0 when empty).

    Uses max()+1 instead of len() so resumed maps with gaps or merges
    never collide with an id already burned into written label files.
    """
    if not class_map:
        return 0
    try:
        return max(int(v) for v in class_map.values()) + 1
    except Exception:
        return len(class_map)


def validate_checkpoint_data(data):
    """Validate a loaded checkpoint payload.

    Returns (warnings, errors) as lists of strings. Never raises.
    """
    warnings = []
    errors = []
    if not isinstance(data, dict):
        return warnings, ["checkpoint payload is not a JSON object"]
    completed = data.get("completed_images", [])
    class_map = data.get("class_map", {})
    batches = data.get("batches_done", [])
    if not isinstance(completed, list):
        errors.append("completed_images is not a list")
    if not isinstance(class_map, dict):
        errors.append("class_map is not an object")
    else:
        seen_ids = {}
        for name, idx in class_map.items():
            try:
                idx = int(idx)
            except Exception:
                errors.append(f"class_map[{name!r}] id is not an integer")
                continue
            if idx < 0:
                errors.append(f"class_map[{name!r}] id is negative ({idx})")
            if idx in seen_ids:
                errors.append(
                    f"duplicate class id {idx} for {seen_ids[idx]!r} and {name!r}"
                )
            else:
                seen_ids[idx] = name
        if seen_ids:
            ids = sorted(seen_ids)
            expected = list(range(max(ids) + 1))
            gaps = [i for i in expected if i not in seen_ids]
            if gaps:
                warnings.append(
                    f"class_map has id gaps {gaps} (harmless; new classes continue at {max(ids) + 1})"
                )
    if not isinstance(batches, list):
        errors.append("batches_done is not a list")
    failed = data.get("failed_images", {})
    if not isinstance(failed, dict):
        warnings.append("failed_images is not an object (failed retry list skipped)")
    rs = data.get("run_settings", None)
    if rs is not None and not isinstance(rs, dict):
        warnings.append("run_settings is not an object (settings check skipped)")
    return warnings, errors


# Label-affecting settings fingerprinted into the checkpoint so a resume
# with changed flags warns instead of silently mixing label vintages.
# batch_size is included because batches_done stores positional batch
# indices: changing it re-derives which images old indices point at.
RUN_SETTINGS_KEYS = (
    "crop_padding_pct",
    "recls_context",
    "crop_resize_ratio",
    "min_box_size",
    "small_box_action",
    "model",
    "height",
    "width",
    "class_mode",
    "none_labels",
    "drop_none",
    "batch_size",
    # Real-ESRGAN changes VLM-bound pixels (crop/scene resolution), so an
    # ESR flip mid-dataset must warn like any other label-affecting flag.
    "esr_enabled",
    "esr_model",
    "esr_model_path",
    "esr_scale",
    "esr_target_long_edge",
    "esr_for_crops",
)


def build_run_settings(**values) -> dict:
    """Build a fingerprint dict with ALL known keys (None when unset).

    Every key is always present so "unset -> set" changes across resumes
    are detected too; only a wholly-missing fingerprint (pre-upgrade
    checkpoints) skips the comparison.
    """
    return {k: values.get(k) for k in RUN_SETTINGS_KEYS}


def _norm_setting(v):
    """Normalize for comparison: 0 == 0.0 == "0"; comma lists and YAML
    lists compare by sorted token tuple ("a,b" == ["a", "b"])."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple, set)):
        return tuple(sorted(str(x).strip().lower() for x in v))
    if isinstance(v, str):
        s = v.strip()
        if "," in s:
            return tuple(sorted(p.strip().lower() for p in s.split(",") if p.strip()))
        try:
            return float(s)
        except ValueError:
            return s
    return v


def run_settings_mismatches(saved, current) -> list:
    """Compare a saved fingerprint against current settings.

    Returns ["key: was OLD, now NEW", ...] for keys present in the saved
    fingerprint whose normalized value differs. Keys absent from the saved
    fingerprint (e.g. pre-upgrade checkpoints) never mismatch.
    """
    saved = saved or {}
    current = current or {}
    diffs = []
    for k in RUN_SETTINGS_KEYS:
        if k not in saved:
            continue
        old, new = _norm_setting(saved[k]), _norm_setting(current.get(k))
        if old != new:
            diffs.append(f"{k}: checkpoint had {saved[k]!r}, now {current.get(k)!r}")
    return diffs


class CheckpointManager:
    def __init__(self, output_folder):
        self.path = Path(output_folder) / ".checkpoint.json"
        self._lock = threading.Lock()

    def load(self):
        if not self.path.exists():
            return None
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            data.setdefault("completed_images", [])
            data.setdefault("class_map", {})
            data.setdefault("batches_done", [])
            data.setdefault("failed_images", {})
            # Normalize class_map ids to int (JSON keeps them numeric, but
            # hand-edited checkpoints may store strings).
            try:
                data["class_map"] = {
                    str(k): int(v) for k, v in dict(data["class_map"]).items()
                }
            except Exception:
                pass
            warnings, errors = validate_checkpoint_data(data)
            for w in warnings:
                logger.warning(f"Checkpoint at {self.path}: {w}")
            if errors:
                for e in errors:
                    logger.error(f"Checkpoint at {self.path}: {e}")
                logger.warning(
                    f"Ignoring corrupt checkpoint at {self.path} and starting fresh. "
                    "Rename/remove it for a deliberate fresh run, or restore a backup."
                )
                return None
            return data
        except Exception as e:
            logger.warning(
                f"Could not read checkpoint at {self.path} ({e}). "
                "Ignoring it and starting fresh."
            )
            return None

    def _carried_keys(self):
        """Best-effort read of keys a None param must preserve.

        Returns ``(failed_images, run_settings)`` from the current file
        (``({}, None)`` when missing/unreadable). Used so a save that only
        knows part of the state (e.g. an early failure save without the
        fingerprint) never wipes keys written by an earlier save.
        """
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except Exception:
            return {}, None
        if not isinstance(data, dict):
            return {}, None
        failed = data.get("failed_images", {})
        if not isinstance(failed, dict):
            failed = {}
        run_settings = data.get("run_settings")
        if not isinstance(run_settings, dict):
            run_settings = None
        return failed, run_settings

    def save(
        self,
        completed_images,
        class_map,
        batches_done,
        run_settings=None,
        failed_images=None,
    ):
        """completed_images / batches_done: iterables (sets are fine).

        ``run_settings``: optional fingerprint dict (see
        :func:`build_run_settings`); ``failed_images``: optional
        stem -> {"reason": str} mapping. A None value carries over what the
        current file holds (so partial saves never wipe keys), instead of
        clearing it.

        NOTE: not atomic across threads by itself -- it only serializes the
        file write. Threaded callers must use :meth:`save_under_locks` so a
        stale snapshot cannot overwrite newer progress.
        """
        with self._lock:
            carried_failed, carried_settings = self._carried_keys()
            payload = {
                "completed_images": sorted(completed_images),
                "class_map": dict(class_map),
                "batches_done": sorted(batches_done),
                "failed_images": (
                    dict(failed_images) if failed_images is not None else carried_failed
                ),
            }
            if run_settings is not None:
                payload["run_settings"] = dict(run_settings)
            elif carried_settings is not None:
                payload["run_settings"] = carried_settings
            self._write_payload(payload)

    def _write_payload(self, payload):
        tmp_path = self.path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.path)  # atomic on POSIX
        except Exception as e:
            logger.error(f"Failed to write checkpoint at {self.path}: {e}")

    def save_under_locks(
        self,
        completed_images,
        completed_lock,
        class_map,
        class_map_lock,
        batches_done,
        run_settings=None,
        failed_images=None,
    ):
        """Snapshot shared run state and persist it atomically.

        Snapshot AND file write happen while holding (in fixed order)
        completed_lock, class_map_lock, then the manager lock, so a
        concurrent worker cannot slip a newer snapshot+write in between and
        get wiped by a stale one (which would lose progress and burn new
        class ids already written into label files).

        ``failed_images`` is a shared stem -> {"reason": str} dict guarded
        by ``completed_lock`` (same convention as ``completed_images``);
        it is snapshotted alongside everything else, so a recorded failure
        can never be lost by a concurrent success save. Pass None to carry
        over what the file holds.

        Callers must NOT hold completed_lock/class_map_lock when calling
        (they are acquired here; re-acquiring would deadlock since
        threading.Lock is not re-entrant). Locks may be None for
        single-threaded callers. ``batches_done`` mutations elsewhere must
        also hold completed_lock for the snapshot to be consistent.
        """
        if completed_lock is not None:
            completed_lock.acquire()
        try:
            if class_map_lock is not None:
                class_map_lock.acquire()
            try:
                completed_snapshot = set(completed_images or ())
                class_map_snapshot = dict(class_map or {})
                batches_snapshot = set(batches_done or ())
                failed_snapshot = (
                    dict(failed_images) if failed_images is not None else None
                )
                with self._lock:
                    carried_failed, carried_settings = self._carried_keys()
                    payload = {
                        "completed_images": sorted(completed_snapshot),
                        "class_map": class_map_snapshot,
                        "batches_done": sorted(batches_snapshot),
                        "failed_images": (
                            failed_snapshot
                            if failed_snapshot is not None
                            else carried_failed
                        ),
                    }
                    if run_settings is not None:
                        payload["run_settings"] = dict(run_settings)
                    elif carried_settings is not None:
                        payload["run_settings"] = carried_settings
                    self._write_payload(payload)
            finally:
                if class_map_lock is not None:
                    class_map_lock.release()
        finally:
            if completed_lock is not None:
                completed_lock.release()

    def clear(self):
        """Remove the checkpoint (used for a deliberate --no_auto_resume fresh run)."""
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove checkpoint at {self.path}: {e}")
