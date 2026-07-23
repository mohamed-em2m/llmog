import html
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from openai import OpenAI

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

from free_detection.detection_pipeline import ObjectDetectionPipeline
from free_detection.trackers import MultiAlgorithmTracker

DEFAULT_HUD = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'

# Client cache for ObjectDetectionPipeline
_client_cache: Dict[Tuple[str, str, str], ObjectDetectionPipeline] = {}
_client_cache_lock = Lock()


def _to_bgr(frame: "np.ndarray") -> "np.ndarray":
    """Convert an RGB numpy frame to BGR for OpenCV tracker calls."""
    if _cv2 is not None and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
        return _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
    return frame


def get_pipeline(
    base_url: str, api_key: str, model_name: str
) -> ObjectDetectionPipeline:
    key = (base_url, api_key, model_name)
    with _client_cache_lock:
        pipeline = _client_cache.get(key)
        if pipeline is None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            pipeline = ObjectDetectionPipeline(client=client, detector_model=model_name)
            _client_cache[key] = pipeline
        return pipeline


def resolve_endpoint(
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
) -> Tuple[str, str, str]:
    base_url = ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
    api_key = ext_api_key if use_external_api else "no-key"
    model_name = ext_model_name if use_external_api else "local-model"
    return base_url, api_key, model_name


@dataclass
class SessionDetector:
    """
    Per-browser-session detector and tracking state.

    The ``multi_tracker`` is kept live across the entire session.
    On each stream tick:
      - ``update_tracking_only`` drives the tracker with the current frame.
    When a background VLM inference completes:
      - ``_run_and_store`` feeds new detections into the tracker via
        ``update_with_detections``, merging and re-initialising instances.
    """

    lock: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="det"
        )
    )
    future: Optional[Future] = None
    multi_tracker: MultiAlgorithmTracker = field(
        default_factory=lambda: MultiAlgorithmTracker()
    )
    last_raw_boxes: List[Any] = field(default_factory=list)
    last_tracked_boxes: List[Any] = field(default_factory=list)
    last_hud: str = DEFAULT_HUD
    last_applied_frame_id: int = -1
    _frame_counter: "itertools.count" = field(
        default_factory=lambda: itertools.count(1)
    )

    reference_gray: Optional[Any] = None
    last_detect_time: float = 0.0
    _last_submitted_frame: Optional[Any] = None

    def next_frame_id(self) -> int:
        return next(self._frame_counter)

    def is_busy(self) -> bool:
        with self.lock:
            return self.future is not None and not self.future.done()

    def submit(self, frame_id: int, fn: Any, *args: Any) -> None:
        with self.lock:
            self.future = self.executor.submit(self._run_and_store, frame_id, fn, *args)

    def _run_and_store(self, frame_id: int, fn: Any, *args: Any) -> None:
        """Background VLM inference — stores raw boxes and drives tracker update."""
        # args[0] is the RGB frame passed from run_vlm_detect; convert to BGR
        # here so OpenCV tracker init always receives the correct color space.
        frame_rgb = args[0] if args else None
        frame_bgr = _to_bgr(frame_rgb) if frame_rgb is not None else None
        try:
            boxes, hud = fn(*args)
        except Exception as e:  # noqa: BLE001
            boxes, hud = None, (
                f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">'
                f"ERROR: {html.escape(str(e))}</div>"
            )
        with self.lock:
            # Use strict > to avoid an older queued frame overwriting a newer result
            if frame_id > self.last_applied_frame_id:
                self.last_applied_frame_id = frame_id
                if boxes is not None:
                    self.last_raw_boxes = boxes
                    # Integrate new VLM detections into tracker state (BGR frame)
                    self.last_tracked_boxes = self.multi_tracker.update_with_detections(
                        frame_bgr, boxes
                    )
                self.last_hud = hud

    def update_tracking_only(self, frame: Optional[Any], algorithm: str) -> List[Any]:
        """
        Drive the tracker for a single video frame tick (between VLM calls).

        Switches algorithm if the UI selection changed since last tick.
        Returns the current tracked boxes (propagated from last VLM result).

        Note: Gradio webcam delivers RGB frames; OpenCV SOT trackers expect BGR.
        The conversion is done here to avoid double-conversion on every tick.
        """
        with self.lock:
            self.multi_tracker.set_algorithm(algorithm)
            frame_bgr = _to_bgr(frame) if frame is not None else frame
            self.last_tracked_boxes = self.multi_tracker.track_frame_only(
                frame_bgr, self.last_tracked_boxes
            )
            return list(self.last_tracked_boxes)

    def snapshot(self) -> Tuple[List[Any], str]:
        with self.lock:
            return list(self.last_tracked_boxes), self.last_hud

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def new_session_detector() -> SessionDetector:
    return SessionDetector()


def reset_session(session: Optional[SessionDetector]) -> SessionDetector:
    if session is not None:
        session.shutdown()
    return new_session_detector()
