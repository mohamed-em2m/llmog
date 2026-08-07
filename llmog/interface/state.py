"""
Shared state, cache management, constants, and assets for the Gradio Interface.
"""

import os
import time
import html
import zipfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional
import gradio as gr

from interface.console_theme import theme
from servers import LlamaServerManager

# ---------------------------------------------------------------------------
# Load Interface Assets (CSS & JS)
# ---------------------------------------------------------------------------
_iface_dir = Path(__file__).parent

with open(_iface_dir / "console.css", encoding="utf-8") as f:
    custom_css = f.read()
with open(_iface_dir / "console.js", encoding="utf-8") as f:
    CONSOLE_JS = f.read()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONCURRENCY = 16

MODEL_PRESETS = [
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
    "unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/gemma-4-31B-it-GGUF:UD-IQ2_M",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q3_K_M",
    "custom",
]

_STATUS_PILL = {
    "queued": '<span class="img-status-pill pill-queued">QUEUED</span>',
    "running": '<span class="img-status-pill pill-running">RUNNING</span>',
    "done": '<span class="img-status-pill pill-done">DONE</span>',
    "error": '<span class="img-status-pill pill-error">ERROR</span>',
    "cancelled": '<span class="img-status-pill pill-cancelled">CANCELLED</span>',
}

LOG_TAIL_BYTES = 8 * 1024
MAX_CACHED_BATCHES = 3

# ---------------------------------------------------------------------------
# Global Server & Pipeline State Container
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.server_manager: Optional[LlamaServerManager] = None
        self.server_lock = threading.Lock()
        self.pipeline_cancel_event = threading.Event()
        self.batch_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.batch_cache_lock = threading.Lock()

state = AppState()

# Backward-compatibility properties/aliases pointing directly to `state`
server_lock = state.server_lock
pipeline_cancel_event = state.pipeline_cancel_event
BATCH_CACHE = state.batch_cache
BATCH_CACHE_LOCK = state.batch_cache_lock

# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------
def _cache_put(batch_id: str, value: Dict[str, Any]) -> None:
    with state.batch_cache_lock:
        state.batch_cache[batch_id] = value
        state.batch_cache.move_to_end(batch_id)
        while len(state.batch_cache) > MAX_CACHED_BATCHES:
            state.batch_cache.popitem(last=False)


def _cache_get(batch_id: str) -> Dict[str, Any]:
    with state.batch_cache_lock:
        b = state.batch_cache.get(batch_id)
        if b is not None:
            state.batch_cache.move_to_end(batch_id)
        return b or {}


def _cache_drop(batch_id: str) -> None:
    with state.batch_cache_lock:
        state.batch_cache.pop(batch_id, None)

# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def zip_results_folder(folder_path: Path) -> Path:
    zip_path = folder_path.parent / f"batch_results_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file() and file.name != zip_path.name:
                zipf.write(file, file.relative_to(folder_path))
    return zip_path


def handle_preset_change(preset: str) -> gr.update:
    if preset == "custom":
        return gr.update(value="", visible=True)
    return gr.update(value=preset, visible=True)


def panel_header(title: str, raw_ta_id: str) -> str:
    return f"""
<div class="out-header">
  <div class="out-header-left">
    <span class="out-header-dot"></span>
    <span class="out-header-title">{title}</span>
  </div>
  <div class="out-header-right">
    <button class="copy-btn" onclick="copyOut('{raw_ta_id}')">&#9096; Copy Raw Text</button>
  </div>
</div>"""


def _tail(s: str, n: int = LOG_TAIL_BYTES) -> str:
    if len(s) <= n:
        return s
    return "...[log tail truncated]...\n" + s[-n:]


def _render_progress_bar(pct: int, status: str = "") -> str:
    pct = max(0, min(100, int(pct)))
    return f"""
    <div class="custom-progress-wrapper">
        <div class="custom-progress-track">
            <div class="custom-progress-fill" style="width:{pct}%;"></div>
        </div>
        <div class="custom-progress-text">{html.escape(status)} ({pct}%)</div>
    </div>
    """


def _section_title(icon: str, label: str) -> str:
    """Render a styled section-title divider inside an accordion."""
    return f'<div class="config-section-title">{icon} {label}</div>'


def toggle_custom_color_field(choice: str) -> gr.update:
    return gr.update(visible=(choice == "custom"))
