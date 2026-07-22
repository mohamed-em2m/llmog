"""
ByteTrack tracking implementation (Kalman Filter + Hungarian Algorithm / IoU matching).
Lightweight, pure Python/NumPy implementation requiring no C++ bindings or extra heavy dependencies.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class KalmanFilterVector:
    """
    A simple 8D Kalman Filter for bounding box tracking in image space.
    State vector: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
    """

    def __init__(self, bbox: Tuple[float, float, float, float]):
        # bbox: [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = bbox
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        r = w / h

        # Mean state vector (8x1)
        self.mean = np.array([cx, cy, r, h, 0, 0, 0, 0], dtype=np.float32)

        # Covariance matrix (8x8)
        std_p = [2 * cx / 20.0, 2 * cy / 20.0, 1e-2, 2 * h / 20.0]
        std_v = [10 * cx / 160.0, 10 * cy / 160.0, 1e-5, 10 * h / 160.0]
        self.covariance = np.diag(np.square(np.r_[std_p, std_v]))

        # State transition matrix F
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        # Measurement projection matrix H
        self._H = np.eye(4, 8, dtype=np.float32)

    def predict(self) -> None:
        std_p = [2 * self.mean[0] / 20.0, 2 * self.mean[1] / 20.0, 1e-2, 2 * self.mean[3] / 20.0]
        std_v = [10 * self.mean[0] / 160.0, 10 * self.mean[1] / 160.0, 1e-5, 10 * self.mean[3] / 160.0]
        Q = np.diag(np.square(np.r_[std_p, std_v]))

        self.mean = np.dot(self._F, self.mean)
        self.covariance = np.dot(np.dot(self._F, self.covariance), self._F.T) + Q

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        ymin, xmin, ymax, xmax = bbox
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        r = w / h
        measurement = np.array([cx, cy, r, h], dtype=np.float32)

        std_m = [2 * measurement[0] / 20.0, 2 * measurement[1] / 20.0, 1e-2, 2 * measurement[3] / 20.0]
        R = np.diag(np.square(std_m))

        projected_mean = np.dot(self._H, self.mean)
        projected_cov = np.dot(np.dot(self._H, self.covariance), self._H.T) + R

        # Kalman gain K
        K = np.dot(np.dot(self.covariance, self._H.T), np.linalg.inv(projected_cov))
        innovation = measurement - projected_mean

        self.mean = self.mean + np.dot(K, innovation)
        self.covariance = self.covariance - np.dot(np.dot(K, projected_cov), K.T)

    def get_rect(self) -> Tuple[float, float, float, float]:
        cx, cy, r, h = self.mean[:4]
        w = r * h
        xmin = cx - w / 2.0
        ymin = cy - h / 2.0
        xmax = cx + w / 2.0
        ymax = cy + h / 2.0
        return (float(ymin), float(xmin), float(ymax), float(xmax))


class Track:
    """Represents a single tracked object across frames."""

    def __init__(self, bbox: Tuple[float, float, float, float], score: float, label: str, track_id: int):
        self.track_id = track_id
        self.kf = KalmanFilterVector(bbox)
        self.score = score
        self.label = label
        self.hits = 1
        self.time_since_update = 0
        self.state = "tracked"  # "tracked" or "lost"

    def predict(self) -> Tuple[float, float, float, float]:
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.get_rect()

    def update(self, bbox: Tuple[float, float, float, float], score: float, label: str) -> None:
        self.kf.update(bbox)
        self.score = score
        self.label = label
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"

    def get_box(self) -> List[Any]:
        ymin, xmin, ymax, xmax = self.kf.get_rect()
        return [ymin, xmin, ymax, xmax, self.label, self.track_id, self.score]


def compute_iou_matrix(boxes1: List[Tuple[float, float, float, float]], boxes2: List[Tuple[float, float, float, float]]) -> np.ndarray:
    if not boxes1 or not boxes2:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1 = np.array(boxes1, dtype=np.float32)  # [N, 4]
    b2 = np.array(boxes2, dtype=np.float32)  # [M, 4]

    # b1: [ymin, xmin, ymax, xmax]
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

    iou_matrix = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    for i, box1 in enumerate(b1):
        ymin1, xmin1, ymax1, xmax1 = box1
        for j, box2 in enumerate(b2):
            ymin2, xmin2, ymax2, xmax2 = box2
            inter_ymin = max(ymin1, ymin2)
            inter_xmin = max(xmin1, xmin2)
            inter_ymax = min(ymax1, ymax2)
            inter_xmax = min(xmax1, xmax2)

            inter_w = max(0.0, inter_xmax - inter_xmin)
            inter_h = max(0.0, inter_ymax - inter_ymin)
            inter_area = inter_w * inter_h

            union_area = area1[i] + area2[j] - inter_area
            if union_area > 0:
                iou_matrix[i, j] = inter_area / union_area

    return iou_matrix


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy matching algorithm for IoU matrix assignment."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))

    # Flatten and sort pairs by highest IoU / lowest cost
    rows, cols = np.unravel_index(np.argsort(-cost_matrix, axis=None), cost_matrix.shape)
    for r, c in zip(rows, cols):
        if r in unmatched_a and c in unmatched_b:
            if cost_matrix[r, c] >= thresh:
                matches.append((r, c))
                unmatched_a.remove(r)
                unmatched_b.remove(c)

    return matches, list(unmatched_a), list(unmatched_b)


class ByteTracker:
    """
    ByteTrack multi-object tracker.
    Splits detections into high-score and low-score sets to associate both
    confident detections and occluded/low-confidence bounding boxes.

    Lifecycle:
    - update(detections): call when new VLM detections arrive. Runs full
      Kalman predict + Hungarian IoU association + track init/kill.
    - predict_only(): call every frame tick BETWEEN detections. Advances
      Kalman state so tracks glide smoothly without re-running association.
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1, match_thresh: float = 0.8, max_time_lost: int = 30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self._id_counter = 0  # Per-instance counter so sessions don't share IDs

    def _new_track(self, bbox, score, label) -> Track:
        self._id_counter += 1
        return Track(bbox, score, label, self._id_counter)

    def predict_only(self) -> List[List[Any]]:
        """Advance Kalman filters one step and return current predicted box positions.
        Does NOT do any detection association — call this between VLM ticks so boxes
        glide smoothly rather than staying frozen until the next model response."""
        for t in self.tracked_tracks:
            t.kf.predict()
            t.time_since_update += 1
        return [t.get_box() for t in self.tracked_tracks]

    def update(self, detections: List[List[Any]]) -> List[List[Any]]:
        """
        Detections format: [[ymin, xmin, ymax, xmax, label, score], ...]
        or [[ymin, xmin, ymax, xmax, label], ...]
        Returns active tracked boxes: [[ymin, xmin, ymax, xmax, label, track_id], ...]
        """
        # Step 1: Predict positions for existing tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # Parse detections into high and low score sets
        det_high = []
        det_low = []
        for d in detections:
            if len(d) >= 4:
                ymin, xmin, ymax, xmax = [float(x) for x in d[:4]]
                label = str(d[4]) if len(d) >= 5 else ""
                score = float(d[5]) if len(d) >= 6 else 0.9
                box_tuple = (ymin, xmin, ymax, xmax)
                if score >= self.high_thresh:
                    det_high.append((box_tuple, score, label))
                elif score >= self.low_thresh:
                    det_low.append((box_tuple, score, label))

        # Step 2: First association with high-score detections
        track_boxes = [t.kf.get_rect() for t in self.tracked_tracks]
        high_boxes = [d[0] for d in det_high]
        iou_matrix = compute_iou_matrix(track_boxes, high_boxes)

        matches_1, u_track_1, u_det_high = linear_assignment(iou_matrix, thresh=1.0 - self.match_thresh)

        # Update matched tracks
        for trk_idx, det_idx in matches_1:
            box, score, label = det_high[det_idx]
            self.tracked_tracks[trk_idx].update(box, score, label)

        # Step 3: Second association with low-score detections and unmatched tracks
        unmatched_tracks = [self.tracked_tracks[i] for i in u_track_1]
        unmatched_track_boxes = [t.kf.get_rect() for t in unmatched_tracks]
        low_boxes = [d[0] for d in det_low]
        iou_matrix_low = compute_iou_matrix(unmatched_track_boxes, low_boxes)

        matches_2, u_track_2, u_det_low = linear_assignment(iou_matrix_low, thresh=0.5)

        for trk_idx, det_idx in matches_2:
            box, score, label = det_low[det_idx]
            unmatched_tracks[trk_idx].update(box, score, label)

        # Remaining unmatched tracks marked lost
        still_unmatched = [unmatched_tracks[i] for i in u_track_2]
        for t in still_unmatched:
            t.state = "lost"
            if t not in self.lost_tracks:
                self.lost_tracks.append(t)
            if t in self.tracked_tracks:
                self.tracked_tracks.remove(t)

        # Step 4: Try to match unmatched high-score detections with lost tracks
        lost_boxes = [t.kf.get_rect() for t in self.lost_tracks]
        u_high_boxes = [det_high[i][0] for i in u_det_high]
        iou_matrix_lost = compute_iou_matrix(lost_boxes, u_high_boxes)

        matches_3, u_lost, u_det_high_final = linear_assignment(iou_matrix_lost, thresh=1.0 - self.match_thresh)

        for lost_idx, det_idx in matches_3:
            orig_det_idx = u_det_high[det_idx]
            box, score, label = det_high[orig_det_idx]
            trk = self.lost_tracks[lost_idx]
            trk.update(box, score, label)
            self.tracked_tracks.append(trk)

        # Remove restored tracks from lost_tracks
        self.lost_tracks = [t for i, t in enumerate(self.lost_tracks) if i not in [m[0] for m in matches_3]]

        # Step 5: Init new tracks for unmatched high-score detections
        for det_idx in u_det_high_final:
            box, score, label = det_high[det_idx]
            new_trk = self._new_track(box, score, label)
            self.tracked_tracks.append(new_trk)

        # Step 6: Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_time_lost]

        # Combine active tracks to return
        active_boxes = []
        for t in self.tracked_tracks:
            box = t.get_box()
            # box: [ymin, xmin, ymax, xmax, label, track_id, score]
            active_boxes.append(box)

        return active_boxes
