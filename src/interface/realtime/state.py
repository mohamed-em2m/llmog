"""
Realtime processing session state and pipeline helper functions.
"""

import html
import itertools
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Optional, List, Tuple

from openai import OpenAI
from free_detection.detection_pipeline import ObjectDetectionPipeline
from free_detection.trackers import MultiAlgorithmTracker

DEFAULT_HUD = '<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'

# Client cache for ObjectDetectionPipeline
_client_cache: Dict[Tuple[str, str, str], ObjectDetectionPipeline] = {}
_client_cache_lock = Lock()


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
    """

    lock: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="det"
        )
    )
    future: Optional[Future] = None
    multi_tracker: MultiAlgorithmTracker = field(default_factory=MultiAlgorithmTracker)
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
        try:
            boxes, hud = fn(*args)
        except Exception as e:  # noqa: BLE001
            boxes, hud = None, (
                f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">'
                f"ERROR: {html.escape(str(e))}</div>"
            )
        with self.lock:
            if frame_id >= self.last_applied_frame_id:
                self.last_applied_frame_id = frame_id
                if boxes is not None:
                    self.last_raw_boxes = boxes
                    self.last_tracked_boxes = boxes
                self.last_hud = hud

    def update_tracking_only(self, frame: Optional[Any], algorithm: str) -> List[Any]:
        with self.lock:
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
