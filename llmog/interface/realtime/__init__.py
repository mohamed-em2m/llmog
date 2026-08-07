"""
Realtime subpackage initialization re-exporting public tab components.
"""

from interface.realtime.ui import _build_realtime_tab, _wire_realtime_events
from interface.realtime.state import SessionDetector, new_session_detector, reset_session
from interface.realtime.handlers import process_single_frame, process_video_frames

__all__ = [
    "_build_realtime_tab",
    "_wire_realtime_events",
    "SessionDetector",
    "new_session_detector",
    "reset_session",
    "process_single_frame",
    "process_video_frames",
]
