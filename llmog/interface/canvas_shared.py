"""
Shared canvas logic for Draw & Real-Time Draw — deduplication placeholder.

Current state: _CUSTOM_CANVAS_JS (~1110 lines) in tab_draw.py and
_RT_DRAW_CANVAS_JS (~1006 lines) in tab_realtime_interactive.py share ~90%
logic (screenToImageCoords, render, drawSingleRegion, undo/redo, etc.)
with divergence only for video snapshot (getUserMedia, toDataURL).

Next step (WS-E): Extract common base to `canvas_base.js` and have both
controllers extend it via `Object.assign` or ES6 class inheritance.
This file documents the shared API to guide that refactor without breaking
current inline <script> injection (Gradio 4 compat).

Shared API (to be extracted):
- screenToImageCoords, fitToScreen, zoom, zoomAt, updateZoomIndicator
- onPointerDown/Move/Up, eraseAt, removeRegion, saveStateForUndo, undo/redo, clearDrawings/clearAll
- updateRegionsList, render, drawSingleRegion, drawRegionBadge, drawBackgroundPattern, syncGradioPayload
- Divergent: startCamera/stopCamera/toggleFreezeCamera/flipCamera/startRenderLoop (RT only)

Dead code note: free_detection/image_preprocessing.py Section B triage
(compute_blur_laplacian, compute_edge_density_canny, triage_frame_check etc.)
is not wired in batch/realtime and can be removed in a follow-up.
"""

# This file intentionally empty — serves as documentation for WS-E.
SHARED_CANVAS_API = [
    "screenToImageCoords",
    "fitToScreen",
    "zoom",
    "zoomAt",
    "onPointerDown",
    "onPointerMove",
    "onPointerUp",
    "eraseAt",
    "removeRegion",
    "saveStateForUndo",
    "undo",
    "redo",
    "clearDrawings",
    "clearAll",
    "updateRegionsList",
    "render",
    "drawSingleRegion",
]
