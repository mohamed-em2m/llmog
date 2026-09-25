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
    return warnings, errors


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

    def save(self, completed_images, class_map, batches_done):
        """completed_images / batches_done: iterables (sets are fine)."""
        with self._lock:
            payload = {
                "completed_images": sorted(completed_images),
                "class_map": dict(class_map),
                "batches_done": sorted(batches_done),
            }
            tmp_path = self.path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, self.path)  # atomic on POSIX
            except Exception as e:
                logger.error(f"Failed to write checkpoint at {self.path}: {e}")

    def clear(self):
        """Remove the checkpoint (used for a deliberate --no_auto_resume fresh run)."""
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as e:
            logger.warning(f"Could not remove checkpoint at {self.path}: {e}")
