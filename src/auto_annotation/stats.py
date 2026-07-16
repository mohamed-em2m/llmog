"""Thread-safe counters so a concurrent run can print an accurate summary."""

import threading
import time

from auto_annotation.logging_utils import logger


class RunStats:
    """Thread-safe counters so a concurrent run can print an accurate summary."""

    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = time.monotonic()
        self.images_total = 0
        self.images_done = 0
        self.images_skipped_no_label = 0
        self.images_skipped_resume = 0
        self.images_failed_read = 0
        self.boxes_seen = 0
        self.boxes_malformed_line = 0
        self.boxes_empty_crop = 0
        self.boxes_dry_run = 0
        self.boxes_model_call_failed = 0
        self.boxes_bad_response = 0
        self.boxes_classified = 0
        self.boxes_low_confidence = 0
        self.classes_discovered = []

    def incr(self, field, n=1):
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def note_new_class(self, class_name):
        with self._lock:
            self.classes_discovered.append(class_name)

    def log_progress(self, img_file):
        with self._lock:
            self.images_done += 1
            done, total = self.images_done, self.images_total
            elapsed = time.monotonic() - self.start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_s = (total - done) / rate if rate > 0 else float("inf")
        eta_str = "unknown" if eta_s == float("inf") else f"{eta_s:,.0f}s"
        logger.info(
            f"[{done}/{total}] finished {img_file} "
            f"({rate:.2f} img/s, ETA {eta_str})"
        )

    def summary_lines(self):
        with self._lock:
            elapsed = time.monotonic() - self.start_time
        lines = [
            "===== Run summary =====",
            f"Elapsed: {elapsed:,.1f}s",
            f"Images: {self.images_total} total | "
            f"{self.images_done} processed | "
            f"{self.images_skipped_resume} skipped (--resume / auto-resume) | "
            f"{self.images_skipped_no_label} skipped (no label file) | "
            f"{self.images_failed_read} failed to read",
            f"Boxes: {self.boxes_seen} seen | "
            f"{self.boxes_classified} classified | "
            f"{self.boxes_malformed_line} malformed label lines | "
            f"{self.boxes_empty_crop} empty crops | "
            f"{self.boxes_dry_run} dry-run (not sent to model)",
            f"Model issues: {self.boxes_model_call_failed} call failures | "
            f"{self.boxes_bad_response} unusable responses",
            f"Low confidence boxes flagged for review: {self.boxes_low_confidence}",
        ]
        if self.classes_discovered:
            lines.append(f"New classes discovered this run: {self.classes_discovered}")
        else:
            lines.append("New classes discovered this run: none")
        return lines
