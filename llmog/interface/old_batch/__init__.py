"""
Batch Sandbox package entry point.
Exposes UI builders, runner routines, and reclassification modules for backward compatibility.
"""

from __future__ import annotations

from interface.batch.runner import (
    TASK_CHOICES,
    TASK_FREE_ANNOTATION,
    TASK_RECLASSIFICATION,
    PipelineCancelledException,
    run_batch_detection_gui,
    run_batch_dispatcher,
    cancel_pipeline,
)
from interface.batch.helpers import (
    render_status_table,
    detections_to_yolo,
)
from interface.batch.reclassification import (
    classify_regions_gui,
    classify_region,
    crop_with_padding,
    extract_regions,
    draw_recls_bbox,
    make_recls_client,
    render_recls_table,
    _RECLS_EMPTY_TABLE,
    _RECLS_PALETTE,
)
from interface.batch.explorer import (
    on_explorer_image_change,
    on_explorer_round_change,
    on_explorer_prev,
    on_explorer_next,
)
from interface.batch.components import (
    build_batch_tab,
    toggle_run_btn,
    toggle_external_api,
    on_batch_preset_change,
    on_batch_strategy_change,
)

__all__ = [
    "TASK_CHOICES",
    "TASK_FREE_ANNOTATION",
    "TASK_RECLASSIFICATION",
    "PipelineCancelledException",
    "run_batch_detection_gui",
    "run_batch_dispatcher",
    "cancel_pipeline",
    "render_status_table",
    "detections_to_yolo",
    "classify_regions_gui",
    "classify_region",
    "crop_with_padding",
    "extract_regions",
    "draw_recls_bbox",
    "make_recls_client",
    "render_recls_table",
    "_RECLS_EMPTY_TABLE",
    "_RECLS_PALETTE",
    "on_explorer_image_change",
    "on_explorer_round_change",
    "on_explorer_prev",
    "on_explorer_next",
    "build_batch_tab",
    "toggle_run_btn",
    "toggle_external_api",
    "on_batch_preset_change",
    "on_batch_strategy_change",
]
