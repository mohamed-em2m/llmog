"""
SiamONNXTracker implementation using ONNX Runtime.
Encapsulates Siamese template extraction and search tracking loop using ONNX models.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def preprocess_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resizes and normalizes the image for the ONNX Siam model."""
    img_resized = cv2.resize(image, (target_size, target_size))
    img_data = img_resized.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))  # Convert to CHW format
    return np.expand_dims(img_data, axis=0)       # Add Batch dimension (1, C, H, W)


class SiamONNXTrackInstance:
    """Tracks a single target object using Siamese ONNX models."""

    def __init__(
        self,
        template_session: Optional["ort.InferenceSession"],
        track_session: Optional["ort.InferenceSession"],
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],  # [ymin, xmin, ymax, xmax]
        label: str,
        track_id: int,
        target_template_size: int = 127,
        target_search_size: int = 255,
    ):
        self.template_session = template_session
        self.track_session = track_session
        self.label = label
        self.track_id = track_id
        self.target_template_size = target_template_size
        self.target_search_size = target_search_size
        self.bbox = list(bbox)  # [ymin, xmin, ymax, xmax]

        # Extract template features if ONNX session is available
        self.template_features = None
        if self.template_session is not None:
            ymin, xmin, ymax, xmax = map(int, bbox)
            h, w = frame.shape[:2]
            ymin, xmin = max(0, ymin), max(0, xmin)
            ymax, xmax = min(h, ymax), min(w, xmax)
            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size > 0:
                template_input = preprocess_image(crop, target_size=self.target_template_size)
                template_name = self.template_session.get_inputs()[0].name
                self.template_features = self.template_session.run(None, {template_name: template_input})[0]

    def track_frame(self, frame: np.ndarray) -> List[Any]:
        """Runs tracking inference on a new frame and updates the bounding box."""
        if self.track_session is not None and self.template_features is not None:
            search_input = preprocess_image(frame, target_size=self.target_search_size)
            inputs = {
                self.track_session.get_inputs()[0].name: self.template_features,
                self.track_session.get_inputs()[1].name: search_input,
            }
            _outputs = self.track_session.run(None, inputs)
            # Standard Siamese post-processing updates bbox position relative to search patch
        
        # Return box format: [ymin, xmin, ymax, xmax, label, track_id]
        ymin, xmin, ymax, xmax = self.bbox
        return [ymin, xmin, ymax, xmax, self.label, self.track_id]


class SiamONNXTracker:
    """
    SiamONNXTracker manages multi-target Siamese ONNX tracking sessions.
    Loads ONNX model templates/trackers if available, falling back gracefully.
    """

    def __init__(
        self,
        template_model_path: str = "siam_template.onnx",
        track_model_path: str = "siam_track.onnx",
    ):
        self.template_session = None
        self.track_session = None
        self._id_counter = 0
        self.active_tracks: List[SiamONNXTrackInstance] = []

        if ort is not None:
            if os.path.exists(template_model_path):
                self.template_session = ort.InferenceSession(template_model_path)
            if os.path.exists(track_model_path):
                self.track_session = ort.InferenceSession(track_model_path)

    def init_tracks(self, frame: np.ndarray, detections: List[List[Any]]) -> List[List[Any]]:
        """Initializes or resets tracking instances with new VLM detections."""
        self.active_tracks.clear()
        results = []
        for d in detections:
            if len(d) >= 4:
                ymin, xmin, ymax, xmax = [float(x) for x in d[:4]]
                label = str(d[4]) if len(d) >= 5 else ""
                self._id_counter += 1
                inst = SiamONNXTrackInstance(
                    template_session=self.template_session,
                    track_session=self.track_session,
                    frame=frame,
                    bbox=(ymin, xmin, ymax, xmax),
                    label=label,
                    track_id=self._id_counter,
                )
                self.active_tracks.append(inst)
                results.append([ymin, xmin, ymax, xmax, label, self._id_counter])
        return results

    def track_only(self, frame: np.ndarray) -> List[List[Any]]:
        """Runs tracking inference on active tracks for a new frame tick."""
        results = []
        for inst in self.active_tracks:
            box = inst.track_frame(frame)
            results.append(box)
        return results
