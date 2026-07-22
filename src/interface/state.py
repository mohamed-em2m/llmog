"""
Shared state, cache management, constants, and assets for the Gradio Interface.
"""

import os
import time
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
# Global Server & Pipeline State
# ---------------------------------------------------------------------------
server_manager: Optional[LlamaServerManager] = None
server_lock = threading.Lock()
pipeline_cancel_event = threading.Event()

BATCH_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
BATCH_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------
def _cache_put(batch_id: str, value: Dict[str, Any]) -> None:
    with BATCH_CACHE_LOCK:
        BATCH_CACHE[batch_id] = value
        BATCH_CACHE.move_to_end(batch_id)
        while len(BATCH_CACHE) > MAX_CACHED_BATCHES:
            BATCH_CACHE.popitem(last=False)


def _cache_get(batch_id: str) -> Dict[str, Any]:
    with BATCH_CACHE_LOCK:
        b = BATCH_CACHE.get(batch_id)
        if b is not None:
            BATCH_CACHE.move_to_end(batch_id)
        return b or {}


def _cache_drop(batch_id: str) -> None:
    with BATCH_CACHE_LOCK:
        BATCH_CACHE.pop(batch_id, None)


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
    <button class="out-copy-btn" onclick="consoleCopyText('{raw_ta_id}', this)">COPY</button>
  </div>
</div>"""
