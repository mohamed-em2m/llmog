"""
Real-time webcam streaming & video frame detection tab UI module.
Re-exports components from the subpackage `interface.realtime`.
"""

from interface.realtime import (
    _build_realtime_tab,
    _wire_realtime_events,
    SessionDetector,
    new_session_detector,
    reset_session,
    process_single_frame,
    process_video_frames,
)

__all__ = [
    "_build_realtime_tab",
    "_wire_realtime_events",
    "SessionDetector",
    "new_session_detector",
    "reset_session",
    "process_single_frame",
    "process_video_frames",
]
