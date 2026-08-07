"""
Unified tracker manager for single and multi-object tracking.

Algorithm roster
----------------
None (Pass-through)    – Display raw VLM boxes; no inter-frame propagation.
ByteTrack              – Kalman + Hungarian IoU. Best for multi-object crowd scenes.
MOSSE (TrackerMOSSE)   – Ultra-fast ~450+ FPS. Good for fast-moving single targets.
KCF (TrackerKCF)       – Fast ~80-120 FPS. Solid single-object balance.
CSRT (TrackerCSRT)     – Most accurate ~25-40 FPS; handles scale/rotation changes.
VitTracker             – Vision-Transformer based; newest OpenCV tracker.
SiamONNX              – Siamese network; requires external ONNX model files.

Single-Object Tracking (SOT) notes
-----------------------------------
OpenCV trackers (MOSSE, KCF, CSRT, VitTracker) are *single-object* trackers —
each instance tracks exactly one target.  When multiple VLM detections arrive,
we spawn one tracker per box.  Between VLM rounds we call ``track_frame_only``
which drives each live tracker independently, providing smooth propagation without
drifting from old VLM predictions.

When new VLM detections arrive we *re-initialise only the trackers whose IoU
with any new box drops below a threshold*, preserving live state for stable boxes.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from free_detection.bytetrack import ByteTracker
from free_detection.siamonnx import SiamONNXTracker

logger = logging.getLogger(__name__)

# IoU below this triggers tracker re-init when new VLM boxes arrive
_REINIT_IOU_THRESHOLD = 0.35

# Minimum box area (px²) — tiny boxes cause OpenCV tracker instability
_MIN_AREA_PX = 16


def _iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two [ymin,xmin,ymax,xmax] boxes."""
    ay1, ax1, ay2, ax2 = a[0], a[1], a[2], a[3]
    by1, bx1, by2, bx2 = b[0], b[1], b[2], b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class _SOTTrackInstance:
    """
    Wraps one OpenCV single-object tracker for a single detected target.

    State machine
    -------------
    is_valid=True   → tracker is live and can be updated via ``update(frame)``.
    is_valid=False  → tracker failed/lost; caller should discard this instance.
    """

    def __init__(
        self,
        algo_name: str,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],  # (ymin, xmin, ymax, xmax)
        label: str,
        track_id: int,
    ) -> None:
        self.algo_name = algo_name
        self.label = label
        self.track_id = track_id
        self.bbox = list(bbox)  # [ymin, xmin, ymax, xmax] — updated each tick
        self.is_valid = False
        self.tracker = self._create_cv_tracker(algo_name)

        if self.tracker is not None and frame is not None and frame.size > 0:
            ymin, xmin, ymax, xmax = bbox
            w = max(1.0, xmax - xmin)
            h = max(1.0, ymax - ymin)
            if w * h >= _MIN_AREA_PX:
                roi = (float(xmin), float(ymin), float(w), float(h))
                try:
                    self.tracker.init(frame, roi)
                    self.is_valid = True
                except Exception as exc:
                    logger.debug("Tracker init failed (%s): %s", algo_name, exc)

    @staticmethod
    def _create_cv_tracker(algo_name: str) -> Optional[Any]:
        algo = algo_name.upper()
        try:
            if algo == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
                return cv2.TrackerMOSSE_create()
            if algo == "KCF" and hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if algo == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if algo in ("VIT", "VITTRACKER") and hasattr(cv2, "TrackerVit_create"):
                try:
                    return cv2.TrackerVit_create(cv2.TrackerVit_Params())
                except Exception:
                    return cv2.TrackerVit_create()
        except Exception as exc:
            logger.debug("Could not create OpenCV tracker %s: %s", algo_name, exc)
        return None

    def update(self, frame: np.ndarray) -> Optional[List[Any]]:
        """Advance tracker by one frame. Returns box or None on loss."""
        if not self.is_valid or self.tracker is None:
            return None
        try:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = box
                ymin, xmin = float(y), float(x)
                ymax, xmax = float(y + h), float(x + w)
                self.bbox = [ymin, xmin, ymax, xmax]
                return [ymin, xmin, ymax, xmax, self.label, self.track_id]
        except Exception as exc:
            logger.debug("Tracker update failed: %s", exc)
        self.is_valid = False
        return None

    def reinit(self, frame: np.ndarray, new_bbox: Tuple[float, float, float, float]) -> None:
        """Re-initialise this tracker with a new bounding box from VLM detection."""
        ymin, xmin, ymax, xmax = new_bbox
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)
        if w * h < _MIN_AREA_PX:
            self.is_valid = False
            return
        roi = (float(xmin), float(ymin), float(w), float(h))
        try:
            self.tracker = self._create_cv_tracker(self.algo_name)
            if self.tracker is not None:
                self.tracker.init(frame, roi)
                self.bbox = [ymin, xmin, ymax, xmax]
                self.is_valid = True
        except Exception as exc:
            logger.debug("Tracker reinit failed: %s", exc)
            self.is_valid = False


class MultiAlgorithmTracker:
    """
    Unified coordinator for all supported tracking algorithms.

    Usage
    -----
    1. Call ``update_with_detections(frame, vlm_boxes)`` whenever the VLM
       returns new boxes — this merges the new predictions into live tracker state.
    2. Call ``track_frame_only(frame)`` on every intermediate stream tick —
       this drives the live trackers without hitting the VLM.
    """

    SUPPORTED_ALGOS = [
        "None (Pass-through)",
        "ByteTrack",
        "MOSSE (TrackerMOSSE)",
        "KCF (TrackerKCF)",
        "CSRT (TrackerCSRT)",
        "VitTracker",
        "SiamONNX",
    ]

    def __init__(self, algorithm: str = "CSRT (TrackerCSRT)") -> None:
        self.algorithm = algorithm
        self._bytetracker = ByteTracker(high_thresh=0.4, low_thresh=0.1)
        self._siam = SiamONNXTracker()
        self._sot_instances: List[_SOTTrackInstance] = []
        self._id_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_algorithm(self, algorithm: str) -> None:
        if algorithm != self.algorithm:
            self.algorithm = algorithm
            self._reset_sot()

    def update_with_detections(
        self,
        frame: Optional[np.ndarray],
        detections: List[List[Any]],
    ) -> List[List[Any]]:
        """
        Integrate new VLM detections into tracker state.

        For pass-through / ByteTrack: straightforward update.
        For SOT trackers: match existing live instances to new boxes via IoU,
        re-init drifted/lost ones, and spawn instances for new boxes.

        Returns the updated list of tracked boxes.
        """
        algo = self.algorithm

        # ── Pass-through ──────────────────────────────────────────────
        if "NONE" in algo.upper() or "PASS" in algo.upper():
            return list(detections)

        # ── ByteTrack ─────────────────────────────────────────────────
        if "BYTETRACK" in algo.upper():
            formatted = []
            for b in detections:
                if len(b) >= 4:
                    lbl = b[4] if len(b) >= 5 else ""
                    formatted.append([b[0], b[1], b[2], b[3], lbl, 0.9])
            return self._bytetracker.update(formatted)

        # ── SiamONNX ─────────────────────────────────────────────────
        if "SIAM" in algo.upper():
            if frame is not None:
                return self._siam.init_tracks(frame, detections)
            return list(detections)

        # ── OpenCV SOT family (MOSSE, KCF, CSRT, VitTracker) ─────────
        return self._sot_update(frame, detections)

    def track_frame_only(
        self,
        frame: Optional[np.ndarray],
        last_boxes: List[List[Any]],
    ) -> List[List[Any]]:
        """
        Propagate tracking for one video frame *without* a VLM call.

        For pass-through: return last_boxes unchanged.
        For ByteTrack: return last_boxes (Kalman predicts internally on next update).
        For SiamONNX: run template matching on active tracks.
        For SOT: drive each live OpenCV tracker and drop lost ones.
        """
        algo = self.algorithm

        if "NONE" in algo.upper() or "PASS" in algo.upper():
            return list(last_boxes)

        if "BYTETRACK" in algo.upper():
            return list(last_boxes)

        if "SIAM" in algo.upper():
            if frame is not None and self._siam.active_tracks:
                result = self._siam.track_only(frame)
                if result:
                    return result
            return list(last_boxes)

        # OpenCV SOT
        return self._sot_track(frame, last_boxes)

    # ------------------------------------------------------------------
    # Internal SOT helpers
    # ------------------------------------------------------------------

    def _reset_sot(self) -> None:
        self._sot_instances.clear()

    def _sot_cv_algo_name(self) -> str:
        """Strip display suffix to get a clean algo keyword for _SOTTrackInstance."""
        algo = self.algorithm.upper()
        if "MOSSE" in algo:
            return "MOSSE"
        if "KCF" in algo:
            return "KCF"
        if "CSRT" in algo:
            return "CSRT"
        if "VIT" in algo:
            return "VIT"
        return "CSRT"

    def _sot_update(
        self,
        frame: Optional[np.ndarray],
        detections: List[List[Any]],
    ) -> List[List[Any]]:
        """
        Merge new VLM detections into live SOT tracker instances.

        Strategy
        --------
        - For each new detection box, find the best-matching live tracker by IoU.
        - If IoU >= threshold → re-init that tracker with the new precise box.
        - If no match → spawn a new tracker instance.
        - Existing trackers with no matching new detection are kept alive
          (they will propagate on their own until the next VLM round or loss).
        """
        if frame is None or frame.size == 0:
            return list(detections)

        cv_algo = self._sot_cv_algo_name()
        results: List[List[Any]] = []
        matched_inst_ids: set[int] = set()

        for det in detections:
            if len(det) < 4:
                continue
            ymin, xmin, ymax, xmax = [float(v) for v in det[:4]]
            label = str(det[4]) if len(det) >= 5 else ""
            det_box = [ymin, xmin, ymax, xmax]

            # Try to find best-matching live instance
            best_iou, best_inst = 0.0, None
            for inst in self._sot_instances:
                if id(inst) in matched_inst_ids:
                    continue
                iou = _iou(det_box, inst.bbox)
                if iou > best_iou:
                    best_iou, best_inst = iou, inst

            if best_inst is not None and best_iou >= _REINIT_IOU_THRESHOLD:
                # Re-init with refined VLM box
                best_inst.label = label
                best_inst.reinit(frame, (ymin, xmin, ymax, xmax))
                matched_inst_ids.add(id(best_inst))
                results.append([ymin, xmin, ymax, xmax, label, best_inst.track_id])
            else:
                # New detection — spawn fresh tracker
                self._id_counter += 1
                inst = _SOTTrackInstance(
                    algo_name=cv_algo,
                    frame=frame,
                    bbox=(ymin, xmin, ymax, xmax),
                    label=label,
                    track_id=self._id_counter,
                )
                if inst.is_valid:
                    self._sot_instances.append(inst)
                results.append([ymin, xmin, ymax, xmax, label, self._id_counter])

        # Prune any instances that haven't been matched and have lost tracking
        # (don't prune matched ones — they are still being managed)
        self._sot_instances = [
            inst for inst in self._sot_instances if inst.is_valid
        ]

        return results if results else list(detections)

    def _sot_track(
        self,
        frame: Optional[np.ndarray],
        last_boxes: List[List[Any]],
    ) -> List[List[Any]]:
        """Drive all live SOT instances for a single inter-VLM frame tick."""
        if frame is None or not self._sot_instances:
            return list(last_boxes)

        results: List[List[Any]] = []
        alive: List[_SOTTrackInstance] = []
        for inst in self._sot_instances:
            box = inst.update(frame)
            if box is not None:
                results.append(box)
                alive.append(inst)
            # Lost instances are simply not appended → pruned implicitly
        self._sot_instances = alive

        return results if results else list(last_boxes)
