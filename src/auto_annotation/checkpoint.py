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
