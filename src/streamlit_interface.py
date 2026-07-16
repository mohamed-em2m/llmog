"""
LLM Object Detection Console — Streamlit Version.

Converted from the original Gradio console to Streamlit, preserving all
core functionality: local llama-server management, batch detection pipeline
with real-time progress, image/round explorer, and custom prompt templates.

Requirements:
    streamlit >= 1.33.0  (for st.fragment support)
    httpx
    Pillow
    openai
"""

import sys
import os
import time
import json
import queue
import shutil
import zipfile
import threading
import io
import logging
import html
import traceback
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import httpx
from PIL import Image
from openai import OpenAI

# Make src directory importable
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from detection_pipeline import (
    ObjectDetectionPipeline,
    RoundResult,
    draw_grid,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
)
from servers.llama_server_manager import LlamaServerManager
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONCURRENCY = 16
MAX_CACHED_BATCHES = 3
LOG_TAIL_BYTES = 8 * 1024
MODEL_PRESETS = [
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
    "unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL",
    "unsloth/gemma-4-31B-it-GGUF:UD-IQ2_M",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q3_K_M",
    "custom",
]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
.status-badge {
    display: inline-block; padding: 0.3rem 0.9rem; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.badge-running { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.badge-stopped { background: rgba(125,133,144,0.12); color: #7d8590; border: 1px solid rgba(125,133,144,0.3); }
.badge-starting { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-error { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

.score-badge {
    display: inline-block; padding: 0.4rem 1.1rem; border-radius: 8px;
    background: rgba(56,189,248,0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3);
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.85rem;
}

.batch-status-table {
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
}
.batch-status-table th {
    padding: 0.4rem 0.7rem; text-align: left;
    background: #161b22; color: #7d8590; position: sticky; top: 0;
}
.batch-status-table td { padding: 0.3rem 0.7rem; border-bottom: 1px solid #21262d; }
.batch-status-table tbody tr:hover { background: #161b22; }

.img-status-pill {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 10px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap;
}
.pill-queued { background: rgba(125,133,144,0.12); color: #7d8590; border: 1px solid rgba(125,133,144,0.3); }
.pill-running { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.pill-done { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.pill-error { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.pill-cancelled { background: rgba(125,133,144,0.12); color: #7d8590; border: 1px solid rgba(125,133,144,0.3); }

.app-header h1 { font-size: 1.8rem; margin-bottom: 0; }
.app-header p { color: #7d8590; font-size: 0.85rem; margin-top: 0.3rem; }
.section-label {
    color: #7d8590; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------


def init_session_state():
    """Initialize all session state variables on first run."""
    defaults = {
        # Server state
        "server_manager": None,
        "server_status": "stopped",
        "server_logs": "No server instance exists.",
        "server_lock": threading.Lock(),
        # Pipeline state
        "pipeline_cancel_event": threading.Event(),
        "pipeline_running": False,
        "pipeline_queue": None,
        "worker_done": None,
        "image_status": {},
        "stem_order": [],
        "pipeline_logs": "",
        "last_zip_path": None,
        "current_batch_id": "",
        "batch_cache": OrderedDict(),
        # Internal references for the worker
        "_batch_results": {},
        "_results_lock": threading.Lock(),
        "_log_capture": None,
        "_total_imgs": 0,
        # Explorer state
        "explorer_image": None,
        "explorer_round": "Final Best",
        # Prompt state
        "customize_prompts": False,
        # Flag to trigger full rerun from fragment
        "_need_rerun": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------


def cache_put(batch_id: str, value: Dict[str, Any]) -> None:
    cache = st.session_state.batch_cache
    cache[batch_id] = value
    cache.move_to_end(batch_id)
    while len(cache) > MAX_CACHED_BATCHES:
        cache.popitem(last=False)


def cache_get(batch_id: str) -> Dict[str, Any]:
    return st.session_state.batch_cache.get(batch_id, {})


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def zip_results_folder(folder_path: Path) -> Path:
    zip_path = folder_path.parent / f"batch_results_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file() and file.name != zip_path.name:
                zipf.write(file, file.relative_to(folder_path))
    return zip_path


def tail(s: str, n: int = LOG_TAIL_BYTES) -> str:
    if len(s) <= n:
        return s
    return "...[log tail truncated]...\n" + s[-n:]


def render_status_badge(status: str, port: Optional[int] = None) -> str:
    classes = {
        "running": "badge-running",
        "stopped": "badge-stopped",
        "starting": "badge-starting",
        "error": "badge-error",
    }
    labels = {
        "running": f"RUNNING (Port {port})" if port else "RUNNING",
        "stopped": "STOPPED",
        "starting": "STARTING...",
        "error": "ERROR",
    }
    cls = classes.get(status, "badge-stopped")
    label = labels.get(status, status.upper())
    return f'<span class="status-badge {cls}">{label}</span>'


_STATUS_PILL = {
    "queued": '<span class="img-status-pill pill-queued">QUEUED</span>',
    "running": '<span class="img-status-pill pill-running">RUNNING</span>',
    "done": '<span class="img-status-pill pill-done">DONE</span>',
    "error": '<span class="img-status-pill pill-error">ERROR</span>',
    "cancelled": '<span class="img-status-pill pill-cancelled">CANCELLED</span>',
}


def render_status_table(image_status: Dict[str, dict], order: List[str]) -> str:
    rows = []
    for stem in order:
        st_data = image_status.get(stem)
        if not st_data:
            continue
        pill = _STATUS_PILL.get(st_data["state"], _STATUS_PILL["queued"])
        score = st_data.get("score")
        score_txt = f"{score}/10" if score is not None else "\u2014"
        rounds_txt = str(st_data.get("rounds_done", 0))
        detail = st_data.get("detail", "") or ""
        name_esc = html.escape(st_data["name"])
        detail_short = html.escape(detail[:120])
        detail_attr = html.escape(detail)
        rows.append(
            f"<tr><td>{name_esc}</td><td>{pill}</td>"
            f"<td>{rounds_txt}</td><td>{score_txt}</td>"
            f'<td style="color:#7d8590;font-size:0.7rem" title="{detail_attr}">{detail_short}</td></tr>'
        )
    body = (
        "".join(rows)
        if rows
        else ('<tr><td colspan="5" style="color:#7d8590">No images yet.</td></tr>')
    )
    return f"""
<div style="margin-top: 0.75rem;">
  <div style="max-height: 260px; overflow-y: auto; border: 1px solid #30363d; border-radius: 6px;">
  <table class="batch-status-table">
    <thead><tr>
      <th>Image</th><th>Status</th><th>Rounds</th><th>Score</th><th>Detail</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Pipeline Cancelled Exception
# ---------------------------------------------------------------------------


class PipelineCancelledException(Exception):
    """Raised when a user cancels the pipeline mid-run."""

    pass


# ---------------------------------------------------------------------------
# Server Management Functions
# ---------------------------------------------------------------------------


def refresh_server_status():
    """Non-blocking refresh of server status into session state."""
    lock = st.session_state.server_lock
    if not lock.acquire(blocking=False):
        return  # Lock held by start/stop — skip this refresh
    try:
        sm = st.session_state.server_manager
        if sm is None:
            st.session_state.server_status = "stopped"
            if not st.session_state.server_logs:
                st.session_state.server_logs = "No server instance exists."
            return
        if sm.process and sm.process.poll() is not None:
            exit_code = sm.process.poll()
            st.session_state.server_status = "error"
            st.session_state.server_logs = (
                f"Server process is dead (Exit code: {exit_code}).\n\n"
                f"--- Logs ---\n{sm.get_logs()}"
            )
            return
        logs = sm.get_logs()
        st.session_state.server_logs = logs
        if sm.is_healthy():
            st.session_state.server_status = "running"
        else:
            st.session_state.server_status = "starting"
    finally:
        lock.release()


def start_server(
    model, port, host, enable_thinking, enable_mtp, ctx_size, gpu_layers, kv_cache_type
):
    """Start the llama-server synchronously with status updates."""
    lock = st.session_state.server_lock
    with lock:
        sm = st.session_state.server_manager
        if sm is not None and sm.is_healthy():
            st.info(f"Server is already running on port {sm.port}.")
            return

        if sm is not None:
            st.write("Stopping existing server instance...")
            try:
                sm.stop_llama_server()
            except Exception as e:
                st.warning(f"Error stopping old server: {e}")
            st.session_state.server_manager = None

        st.write("Configuring server...")
        spec_type = "draft-mtp" if enable_mtp else "none"
        sm = LlamaServerManager(
            model=model,
            host=host,
            port=int(port),
            ctx_size=int(ctx_size),
            parallel_slots=1,
            n_threads=-1,
            gpu_layers=int(gpu_layers),
            tensor_split="1,1",
            main_gpu=0,
            temp=0.4,
            top_p=0.95,
            top_k=64,
            spec_type=spec_type,
            spec_draft_n_max=4 if enable_mtp else 0,
            enable_thinking=enable_thinking,
            batch_size=1024,
            ubatch_size=512,
            kv_cache_type=kv_cache_type,
        )
        st.session_state.server_manager = sm
        st.session_state.server_status = "starting"

        st.write("Spawning llama-server process...")
        try:
            sm.start_llama_server()
        except Exception as e:
            st.session_state.server_manager = None
            st.session_state.server_status = "error"
            st.error(f"Failed to start server process: {e}")
            return

    # Wait for health (outside the lock so refresh_server_status can still run)
    start_time = time.time()
    timeout = 180
    healthy = False
    status_placeholder = st.empty()
    log_placeholder = st.empty()

    while time.time() - start_time < timeout:
        sm = st.session_state.server_manager
        if sm is None:
            status_placeholder.warning("Server initialization aborted.")
            st.session_state.server_status = "stopped"
            return
        if sm.process and sm.process.poll() is not None:
            exit_code = sm.process.poll()
            logs = sm.get_logs()
            st.session_state.server_logs = logs
            st.session_state.server_manager = None
            st.session_state.server_status = "error"
            status_placeholder.error(f"Server process exited with code {exit_code}.")
            log_placeholder.text(logs[-2000:])
            return
        if sm.is_healthy():
            healthy = True
            break
        logs = sm.get_logs()
        elapsed = int(time.time() - start_time)
        status_placeholder.info(
            f"Waiting for model to load into memory... ({elapsed}s elapsed)"
        )
        log_placeholder.text(logs[-1200:])
        st.session_state.server_logs = logs
        time.sleep(2)

    if healthy:
        status_placeholder.write("Server is up. Running warmup request...")
        try:
            sm = st.session_state.server_manager
            if sm:
                sm.warmup_model()
            st.session_state.server_status = "running"
            status_placeholder.success(
                "Server started and warmed up. Ready for detection tasks."
            )
        except Exception as e:
            st.session_state.server_status = "running"
            status_placeholder.warning(f"Server is healthy, but warmup failed: {e}")
    else:
        st.session_state.server_status = "error"
        status_placeholder.error(
            "Timed out waiting for the server to report healthy status."
        )


def stop_server():
    """Stop the running server."""
    lock = st.session_state.server_lock
    with lock:
        sm = st.session_state.server_manager
        if sm is None:
            st.info("No server running.")
            return
        try:
            sm.stop_llama_server()
            st.session_state.server_manager = None
            st.session_state.server_status = "stopped"
            st.session_state.server_logs = "Server stopped."
            st.success("Server stopped successfully.")
        except Exception as e:
            st.session_state.server_status = "error"
            st.error(f"Error stopping server: {e}")


# ---------------------------------------------------------------------------
# Pipeline Worker (Background Thread)
# ---------------------------------------------------------------------------


def start_pipeline_worker(
    image_paths: List[Path],
    categories: List[str],
    category_definitions: str,
    api_url: str,
    api_key: str,
    model_name: str,
    max_rounds: int,
    score_threshold: int,
    detector_temp: float,
    judge_temp: float,
    concurrency: int,
    det_tmpl: str,
    jdg_tmpl: str,
):
    """Validate inputs, create API client, and launch the worker thread."""
    # --- Create HTTP client ---
    try:
        http_client = httpx.Client(
            timeout=httpx.Timeout(None),
            limits=httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            ),
        )
        client = OpenAI(base_url=api_url, api_key=api_key, http_client=http_client)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return

    # --- Prepare batch ---
    st.session_state.pipeline_cancel_event.clear()
    batch_id = str(int(time.time()))
    st.session_state.current_batch_id = batch_id

    batch_logger = logging.getLogger(f"detection_pipeline.batch_{batch_id}")
    batch_logger.setLevel(logging.INFO)
    batch_logger.propagate = False

    log_capture = io.StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    batch_logger.addHandler(log_handler)

    run_dir = Path("./gui_runs") / f"run_{batch_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_results: Dict[str, Any] = {}
    results_lock = threading.Lock()
    cache_put(batch_id, batch_results)

    q: queue.Queue = queue.Queue()
    worker_done = threading.Event()

    # Build stem mapping (ensure unique stems)
    stem_order: List[str] = []
    stem_for_path: Dict[Path, str] = {}
    for img_path in image_paths:
        img_stem = img_path.stem
        uniq_stem = img_stem
        counter = 1
        while uniq_stem in stem_for_path.values():
            uniq_stem = f"{img_stem}_{counter}"
            counter += 1
        stem_for_path[img_path] = uniq_stem
        stem_order.append(uniq_stem)

    total_imgs = len(image_paths)

    # --- Worker functions (closures capturing client, batch_logger, etc.) ---

    def process_one_image(img_path: Path):
        stem = stem_for_path[img_path]
        if st.session_state.pipeline_cancel_event.is_set():
            q.put(("image_skipped", stem))
            return

        q.put(("start_image", img_path.name, stem))

        try:
            image_out_dir = run_dir / stem
            image_out_dir.mkdir(parents=True, exist_ok=True)

            target_suffix = img_path.suffix or ".jpg"
            shutil.copy(img_path, image_out_dir / f"original{target_suffix}")
            base_image = Image.open(img_path).convert("RGB")

            with results_lock:
                batch_results[stem] = {
                    "grid_original": draw_grid(base_image),
                    "raw_original": base_image,
                    "best_annotated": None,
                    "detections": [],
                    "rounds": [],
                }

            def progress_callback(
                round_result: RoundResult, annotated_image: Image.Image, _stem=stem
            ):
                if st.session_state.pipeline_cancel_event.is_set():
                    raise PipelineCancelledException("Pipeline cancelled by user.")
                q.put(("round", _stem, round_result, annotated_image))

            pipeline = ObjectDetectionPipeline(
                detector_client=client,
                judge_client=client,
                detector_model=model_name,
                judge_model=model_name,
                max_rounds=max_rounds,
                score_threshold=score_threshold,
                detector_template=det_tmpl,
                judge_template=jdg_tmpl,
                detector_max_tokens=4096,
                judge_max_tokens=1024,
                api_retries=3,
                detector_temperature=detector_temp,
                detector_top_p=0.95,
                judge_temperature=judge_temp,
            )

            best, _history = pipeline.run(
                image_path=str(img_path),
                categories=categories,
                category_definitions=category_definitions,
                show_plot=False,
                output_dir=str(image_out_dir),
                progress_callback=progress_callback,
            )

            detections = best.get("detections") or []
            with results_lock:
                batch_results[stem]["best_annotated"] = (
                    best.get("annotated") if detections else None
                )
                batch_results[stem]["detections"] = detections
            q.put(("finish_image", stem))

        except PipelineCancelledException:
            q.put(("image_cancelled", stem))
        except Exception as e:
            batch_logger.error(f"[{stem}] {e}\n{traceback.format_exc()}")
            q.put(("image_error", stem, str(e)))

    def worker():
        try:
            if not st.session_state.pipeline_cancel_event.is_set():
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(process_one_image, p) for p in image_paths]
                    for fut in as_completed(futures):
                        exc = fut.exception()
                        if exc is not None:
                            batch_logger.error(
                                f"Unhandled worker exception: {exc}\n"
                                f"{traceback.format_exc()}"
                            )
                            q.put(("image_error", "unknown", str(exc)))

            if st.session_state.pipeline_cancel_event.is_set():
                q.put(("cancelled",))
            else:
                try:
                    zip_path = zip_results_folder(run_dir)
                    q.put(("done", str(zip_path)))
                except Exception as e:
                    q.put(("error", str(e), traceback.format_exc()))
        except Exception as e:
            q.put(("error", str(e), traceback.format_exc()))
        finally:
            batch_logger.removeHandler(log_handler)
            log_handler.close()
            worker_done.set()

    # --- Initialize image status ---
    image_status = {
        stem: {
            "name": img_path.name,
            "state": "queued",
            "rounds_done": 0,
            "score": None,
            "detail": "",
        }
        for img_path, stem in stem_for_path.items()
    }

    # --- Store everything in session state ---
    st.session_state.pipeline_queue = q
    st.session_state.worker_done = worker_done
    st.session_state.image_status = image_status
    st.session_state.stem_order = stem_order
    st.session_state.pipeline_logs = ""
    st.session_state.last_zip_path = None
    st.session_state.pipeline_running = True
    st.session_state._log_capture = log_capture
    st.session_state._batch_results = batch_results
    st.session_state._results_lock = results_lock
    st.session_state._total_imgs = total_imgs

    # Launch worker thread
    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


def get_explorer_data(selected_image, selected_round, show_grid):
    """Retrieve image and metadata for the explorer view."""
    batch_id = st.session_state.current_batch_id
    batch_results = cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return (
            None,
            None,
            '<span class="score-badge">Score: -/10</span>',
            "",
            "",
            "",
            "[]",
        )

    img_data = batch_results[selected_image]
    src_img = img_data["grid_original"] if show_grid else img_data["raw_original"]

    if not selected_round or selected_round == "Final Best":
        best_annotated = img_data["best_annotated"]
        best_score = -1
        best_round_num = -1
        best_feedback = "No detections found."
        best_raw = ""
        best_err = ""
        best_detections = img_data.get("detections") or []

        for r in img_data["rounds"]:
            if r["score"] > best_score:
                best_score = r["score"]
                best_round_num = r["round"]
                best_feedback = r["feedback"]
                best_raw = r["raw_text"]
                best_err = r["parse_error"]

        display_img = best_annotated if best_detections else src_img

        if best_score >= 0:
            score_text = (
                f'<span class="score-badge">'
                f"Best Score: {best_score}/10 (Round {best_round_num})"
                f"</span>"
            )
        else:
            score_text = '<span class="score-badge">Score: -/10</span>'

        return (
            src_img,
            display_img,
            score_text,
            best_feedback,
            best_raw,
            best_err or "None",
            (
                json.dumps(img_data["detections"], indent=2)
                if img_data["detections"]
                else "[]"
            ),
        )

    try:
        round_idx = int(selected_round) - 1
        rounds = img_data["rounds"]
        if 0 <= round_idx < len(rounds):
            r = rounds[round_idx]
            round_detections = r.get("detections") or []
            display_img = r["image"] if round_detections else src_img
            score_text = f'<span class="score-badge">Score: {r["score"]}/10</span>'
            return (
                src_img,
                display_img,
                score_text,
                r["feedback"],
                r["raw_text"],
                r["parse_error"] or "None",
                json.dumps(r["detections"], indent=2) if r["detections"] else "[]",
            )
    except Exception as e:
        st.error(f"Error loading round details: {e}")

    return (
        src_img,
        None,
        '<span class="score-badge">Score: -/10</span>',
        "",
        "",
        "",
        "[]",
    )


# ---------------------------------------------------------------------------
# UI: Server Tab
# ---------------------------------------------------------------------------


def on_preset_change():
    """Callback when the model preset dropdown changes."""
    preset = st.session_state.server_preset_sel
    if preset == "custom":
        st.session_state.server_model_input = ""
    else:
        st.session_state.server_model_input = preset


def render_server_tab():
    st.markdown(
        '<p class="section-label">Model Server Configuration</p>',
        unsafe_allow_html=True,
    )

    col_config, col_output = st.columns([2, 3])

    # ---- Configuration column ----
    with col_config:
        st.selectbox(
            "Recommended Model Presets",
            MODEL_PRESETS,
            index=0,
            key="server_preset_sel",
            on_change=on_preset_change,
        )

        use_ext = st.session_state.get("use_external_chk", False)
        st.text_input(
            "Model GGUF Path or HF Repo ID",
            value=MODEL_PRESETS[0] if MODEL_PRESETS else "",
            placeholder="e.g. C:/models/qwen.gguf or HF ID",
            key="server_model_input",
            disabled=use_ext,
        )

        port_col, host_col = st.columns(2)
        with port_col:
            st.number_input(
                "Port Number",
                value=8080,
                step=1,
                key="server_port_input",
                disabled=use_ext,
            )
        with host_col:
            st.text_input(
                "Host Binding",
                value="0.0.0.0",
                key="server_host_input",
                disabled=use_ext,
            )

        chk_col1, chk_col2 = st.columns(2)
        with chk_col1:
            st.checkbox(
                "Thinking Mode",
                value=False,
                key="server_thinking_chk",
                disabled=use_ext,
            )
        with chk_col2:
            st.checkbox(
                "MTP Speculative Drafting",
                value=True,
                key="server_mtp_chk",
                disabled=use_ext,
            )

        with st.expander("Advanced Server Parameters"):
            st.number_input(
                "Context Size",
                value=20000,
                step=1,
                key="server_ctx_input",
                disabled=use_ext,
            )
            st.number_input(
                "GPU Layers (-ngl)",
                value=-1,
                step=1,
                key="server_gpu_layers_input",
                disabled=use_ext,
            )
            st.selectbox(
                "KV Cache Type",
                ["q4_0", "q8_0", "f16"],
                index=0,
                key="server_kv_cache_input",
                disabled=use_ext,
            )

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button(
                "\u25b6 Start Server",
                type="primary",
                use_container_width=True,
                disabled=use_ext,
            )
        with btn_col2:
            stop_btn = st.button(
                "\u23f9 Stop",
                use_container_width=True,
                disabled=use_ext,
            )
        with btn_col3:
            refresh_btn = st.button(
                "\U0001f504 Refresh",
                use_container_width=True,
            )

    # ---- Output column (auto-refreshing fragment) ----
    with col_output:
        render_server_output_fragment()

    # ---- Button handlers ----
    if start_btn:
        with st.status("Starting server...", expanded=True) as status:
            start_server(
                st.session_state.server_model_input,
                int(st.session_state.server_port_input),
                st.session_state.server_host_input,
                st.session_state.server_thinking_chk,
                st.session_state.server_mtp_chk,
                int(st.session_state.server_ctx_input),
                int(st.session_state.server_gpu_layers_input),
                st.session_state.server_kv_cache_input,
            )
            if st.session_state.server_status == "running":
                status.update(label="Server started!", state="complete")
            else:
                status.update(label="Server failed to start", state="error")
        st.rerun()

    if stop_btn:
        stop_server()
        st.rerun()

    if refresh_btn:
        refresh_server_status()
        st.rerun()


@st.fragment(run_every=5.0)
def render_server_output_fragment():
    """Auto-refresh server status badge and logs every 5 seconds."""
    refresh_server_status()
    sm = st.session_state.server_manager
    port = sm.port if sm else None
    st.markdown(
        render_status_badge(st.session_state.server_status, port),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-label">Server Output Console</p>',
        unsafe_allow_html=True,
    )
    st.text_area(
        "Live Logs",
        value=st.session_state.server_logs,
        height=400,
        key="server_logs_ta",
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# UI: Batch Sandbox Tab
# ---------------------------------------------------------------------------


def render_batch_sandbox_tab():
    col_config, col_results = st.columns([2, 3])

    # ---- Configuration column ----
    with col_config:
        st.markdown(
            '<p class="section-label">Configuration</p>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload Source Image(s)",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
            key="input_images_uploader",
        )

        st.text_input(
            "Target Categories (comma-separated)",
            value="hole, stain, tear, cut, knot, weaving_defect",
            key="categories_input",
        )

        st.text_area(
            "Category Definitions",
            value=(
                "- hole: missing fabric\n"
                "- stain: discoloration only\n"
                "- tear: frayed, uneven separation\n"
                "- cut: clean cut\n"
                "- knot: raise lump\n"
                "- weaving_defect: uneven thread density"
            ),
            height=100,
            key="category_defs_input",
        )

        with st.expander("Pipeline Parameters"):
            st.slider("Optimization Max Rounds", 1, 5, 1, key="max_rounds_slider")
            st.slider(
                "Stop Score Threshold (0-10)", 0, 10, 8, key="score_threshold_slider"
            )
            st.slider(
                "Detector Temperature", 0.0, 1.5, 0.9, 0.05, key="det_temp_slider"
            )
            st.slider("Judge Temperature", 0.0, 1.5, 0.2, 0.05, key="jdg_temp_slider")

        with st.expander("External API (Optional)"):
            use_ext = st.checkbox(
                "Use External API instead of Local Server",
                value=False,
                key="use_external_chk",
            )
            st.text_input(
                "Base URL", value="https://api.openai.com/v1", key="ext_api_url_input"
            )
            st.text_input("API Key", value="", type="password", key="ext_api_key_input")
            st.text_input("Model Name", value="gpt-4o", key="ext_model_input")

        with st.expander("Advanced Settings"):
            st.slider(
                "Concurrent Images",
                min_value=1,
                max_value=64,
                value=DEFAULT_CONCURRENCY,
                step=1,
                help=(
                    "Images processed in parallel via httpx. With a local "
                    "llama-server running parallel_slots=1, only one request "
                    "is served at a time — high values just queue at the "
                    "server. Set higher (8–32) only when targeting an "
                    "external API or a multi-slot local server."
                ),
                key="concurrency_slider",
            )

        # Get prompt settings (set in Prompts tab)
        customize_prompts = st.session_state.get("customize_prompts", False)
        custom_det_prompt = st.session_state.get(
            "custom_det_prompt_input", DEFAULT_DETECTOR_TEMPLATE
        )
        custom_jdg_prompt = st.session_state.get(
            "custom_jdg_prompt_input", DEFAULT_JUDGE_TEMPLATE
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_btn = st.button(
                "\u25b6 Run Batch Pipeline",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.pipeline_running,
            )
        with btn_col2:
            cancel_btn = st.button(
                "\u23f9 Cancel",
                use_container_width=True,
                disabled=not st.session_state.pipeline_running,
            )

    # ---- Results column (auto-refreshing fragment) ----
    with col_results:
        render_results_fragment()

    # ---- Button handlers ----
    if run_btn and not st.session_state.pipeline_running:
        _handle_run_button(
            uploaded_files,
            st.session_state.categories_input,
            st.session_state.category_defs_input,
            use_ext,
            st.session_state.ext_api_url_input,
            st.session_state.ext_api_key_input,
            st.session_state.ext_model_input,
            st.session_state.max_rounds_slider,
            st.session_state.score_threshold_slider,
            st.session_state.det_temp_slider,
            st.session_state.jdg_temp_slider,
            st.session_state.concurrency_slider,
            customize_prompts,
            custom_det_prompt,
            custom_jdg_prompt,
        )

    if cancel_btn:
        st.session_state.pipeline_cancel_event.set()
        st.info(
            "Cancellation requested. In-flight images will finish their "
            "current round and write results; queued images will be skipped."
        )
        st.rerun()


def _handle_run_button(
    uploaded_files,
    categories_str,
    category_definitions,
    use_external_api,
    ext_api_url,
    ext_api_key,
    ext_model_name,
    max_rounds,
    score_threshold,
    det_temp,
    jdg_temp,
    concurrency,
    customize_prompts,
    detector_template,
    judge_template,
):
    """Validate inputs and start the pipeline worker."""
    if not uploaded_files:
        st.error("Please upload at least one image.")
        return

    categories = [c.strip() for c in categories_str.split(",") if c.strip()]
    if not categories:
        st.error("Please list at least one category.")
        return

    # Save uploaded files to a temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="detection_batch_"))
    image_paths: List[Path] = []
    for f in uploaded_files:
        temp_path = temp_dir / f.name
        with open(temp_path, "wb") as out:
            out.write(f.getvalue())
        try:
            with Image.open(temp_path) as im:
                im.verify()
            image_paths.append(temp_path)
        except Exception as e:
            st.error(f"File '{f.name}' is not a valid image: {e}")
            return

    if not image_paths:
        st.error("No valid images found.")
        return

    # Determine API endpoint
    if use_external_api:
        api_url = ext_api_url
        api_key = ext_api_key
        model_name = ext_model_name
        if not api_key or api_key == "your-key":
            st.error(
                "External API selected but no API key provided. "
                "Set one in the External API section."
            )
            return
    else:
        with st.session_state.server_lock:
            sm = st.session_state.server_manager
            if sm is None or not sm.is_healthy():
                st.error(
                    "Local server not running. Start it on the Server tab "
                    "or enable External API."
                )
                return
            port = sm.port
            model_name = sm.model
        api_url = f"http://localhost:{port}/v1"
        api_key = "not-needed"

    det_tmpl = detector_template if customize_prompts else DEFAULT_DETECTOR_TEMPLATE
    jdg_tmpl = judge_template if customize_prompts else DEFAULT_JUDGE_TEMPLATE

    start_pipeline_worker(
        image_paths,
        categories,
        category_definitions,
        api_url,
        api_key,
        model_name,
        max_rounds,
        score_threshold,
        det_temp,
        jdg_temp,
        max(1, int(concurrency or DEFAULT_CONCURRENCY)),
        det_tmpl,
        jdg_tmpl,
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Results Fragment (auto-refreshing)
# ---------------------------------------------------------------------------


@st.fragment(run_every=0.5)
def render_results_fragment():
    """Poll the pipeline queue and render real-time results.

    This fragment reruns every 0.5 seconds to process pending queue messages,
    update session state, and re-render the results area. When the pipeline
    transitions from running to idle, a full script rerun is triggered to
    update button states.
    """
    q = st.session_state.pipeline_queue
    worker_done = st.session_state.worker_done
    was_running = st.session_state.pipeline_running

    # ---- Process all available queue messages ----
    if q is not None:
        while True:
            try:
                msg = q.get_nowait()
                tag = msg[0]

                if tag == "start_image":
                    stem = msg[2]
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["state"] = "running"

                elif tag == "round":
                    stem, r_res, r_img = msg[1], msg[2], msg[3]
                    batch_results = st.session_state._batch_results
                    with st.session_state._results_lock:
                        if stem in batch_results:
                            batch_results[stem]["rounds"].append(
                                {
                                    "round": r_res.round,
                                    "score": r_res.score,
                                    "feedback": r_res.feedback,
                                    "raw_text": r_res.raw_detector_output,
                                    "parse_error": r_res.parse_error,
                                    "image": r_img,
                                    "detections": r_res.detections,
                                }
                            )
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["rounds_done"] = r_res.round
                        st.session_state.image_status[stem]["score"] = r_res.score

                elif tag == "finish_image":
                    stem = msg[1]
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["state"] = "done"

                elif tag == "image_error":
                    stem, err = msg[1], msg[2]
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["state"] = "error"
                        st.session_state.image_status[stem]["detail"] = err[:200]

                elif tag == "image_cancelled":
                    stem = msg[1]
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["state"] = "cancelled"

                elif tag == "image_skipped":
                    stem = msg[1]
                    if stem in st.session_state.image_status:
                        st.session_state.image_status[stem]["state"] = "cancelled"

                elif tag == "done":
                    st.session_state.last_zip_path = msg[1]
                    st.session_state.pipeline_running = False

                elif tag == "cancelled":
                    st.session_state.pipeline_running = False

                elif tag == "error":
                    st.session_state.pipeline_running = False

            except queue.Empty:
                break

        # Update logs from the log capture buffer
        log_capture = st.session_state._log_capture
        if log_capture is not None:
            st.session_state.pipeline_logs = tail(log_capture.getvalue())

        # Check if worker has exited
        if (
            worker_done is not None
            and worker_done.is_set()
            and st.session_state.pipeline_running
        ):
            st.session_state.pipeline_running = False

    # ---- Trigger full rerun if pipeline just finished ----
    if was_running and not st.session_state.pipeline_running:
        st.session_state._need_rerun = True

    # ---- Render status and progress ----
    image_status = st.session_state.image_status
    stem_order = st.session_state.stem_order
    total = st.session_state._total_imgs

    st.markdown(
        '<p class="section-label">Results</p>',
        unsafe_allow_html=True,
    )

    if total > 0 and stem_order:
        done_n = sum(
            1
            for s in image_status.values()
            if s["state"] in ("done", "error", "cancelled")
        )
        running_n = sum(1 for s in image_status.values() if s["state"] == "running")
        errored_n = sum(1 for s in image_status.values() if s["state"] == "error")
        pct = int((done_n / total) * 100) if total else 0

        if st.session_state.pipeline_running:
            if running_n > 0:
                status_text = (
                    f"Processing... ({done_n}/{total} done, " f"{running_n} running)"
                )
            else:
                status_text = f"Starting... ({done_n}/{total} done)"
            st.markdown(f"**Status: {status_text}**")
        else:
            if done_n == total:
                st.success(
                    f"Batch complete: {done_n - errored_n} succeeded, "
                    f"{errored_n} failed."
                )
            else:
                st.warning("Pipeline stopped.")

        st.progress(pct / 100, text=f"{pct}%")

        # Status table
        st.markdown(
            render_status_table(image_status, stem_order),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("**Status: Idle**")

    # ---- Download button ----
    if st.session_state.last_zip_path:
        zip_path = Path(st.session_state.last_zip_path)
        if zip_path.exists():
            with open(zip_path, "rb") as f:
                st.download_button(
                    "\U0001f4e5 Download Processed Results (.zip)",
                    data=f.read(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )

    # ---- Sub-tabs: Explorer, JSON, Logs ----
    if stem_order:
        sub1, sub2, sub3 = st.tabs(
            [
                "\U0001f5bc\ufe0f Batch Explorer",
                "\U0001f4c4 Detections JSON",
                "\U0001f4cb Pipeline Logs",
            ]
        )

        with sub1:
            _render_explorer(stem_order)

        with sub2:
            batch_id = st.session_state.current_batch_id
            batch_results = cache_get(batch_id)
            sel_img = st.session_state.get("explorer_image_select")
            if sel_img and sel_img in batch_results:
                detections = batch_results[sel_img].get("detections") or []
                st.code(
                    json.dumps(detections, indent=2),
                    language="json",
                )
            else:
                st.code("[]", language="json")

        with sub3:
            st.text_area(
                "Execution Logs",
                value=st.session_state.pipeline_logs,
                height=400,
                key="pipeline_logs_ta",
                label_visibility="collapsed",
            )

    # ---- Trigger full rerun if needed ----
    if st.session_state.get("_need_rerun", False):
        st.session_state._need_rerun = False
        st.rerun()


def _render_explorer(stem_order):
    """Render the image/round explorer inside the results fragment."""
    batch_id = st.session_state.current_batch_id
    batch_results = cache_get(batch_id)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_image = st.selectbox(
            "Select Image",
            stem_order,
            key="explorer_image_select",
        )

    with col2:
        round_choices = ["Final Best"]
        if selected_image and selected_image in batch_results:
            rounds = batch_results[selected_image].get("rounds", [])
            round_choices += [str(r["round"]) for r in rounds]
        selected_round = st.selectbox(
            "Select Round",
            round_choices,
            key="explorer_round_select",
        )

    with col3:
        st.markdown("")  # spacer
        show_grid = st.checkbox(
            "Show grid overlay",
            value=True,
            key="show_grid_chk",
        )

    # Retrieve and display explorer data
    if selected_image:
        (
            src_img,
            display_img,
            score_html,
            feedback,
            raw_text,
            parse_err,
            detections_json,
        ) = get_explorer_data(selected_image, selected_round, show_grid)

        st.markdown(score_html, unsafe_allow_html=True)

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**Source Image**")
            if src_img is not None:
                st.image(src_img, use_container_width=True)
        with img_col2:
            st.markdown("**Annotated Image**")
            if display_img is not None:
                st.image(display_img, use_container_width=True)
            else:
                st.info("No detections to display.")

        st.text_area(
            "Judge's Feedback",
            value=feedback,
            height=100,
            key="feedback_ta",
            disabled=True,
        )

        with st.expander("Raw Response Details"):
            st.text_input(
                "Parsing Errors",
                value=parse_err,
                key="parse_err_ti",
                disabled=True,
            )
            st.text_area(
                "Raw Detector Text Response",
                value=raw_text,
                height=150,
                key="raw_response_ta",
                disabled=True,
            )


# ---------------------------------------------------------------------------
# UI: Prompts Tab
# ---------------------------------------------------------------------------


def render_prompts_tab():
    st.markdown(
        '<p class="section-label">Prompt Engineering</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Modify the custom instruction templates fed to the Detector and "
        "Judge agents."
    )

    customize = st.checkbox(
        "Enable Custom Prompt Templates",
        value=st.session_state.get("customize_prompts", False),
        key="customize_prompts_chk",
    )
    st.session_state.customize_prompts = customize

    if customize:
        st.text_area(
            "Detector Prompt Template",
            value=DEFAULT_DETECTOR_TEMPLATE,
            height=250,
            key="custom_det_prompt_input",
        )
        st.text_area(
            "Judge Prompt Template",
            value=DEFAULT_JUDGE_TEMPLATE,
            height=250,
            key="custom_jdg_prompt_input",
        )
    else:
        st.session_state.custom_det_prompt_input = DEFAULT_DETECTOR_TEMPLATE
        st.session_state.custom_jdg_prompt_input = DEFAULT_JUDGE_TEMPLATE
        st.info(
            "Custom prompts are disabled. Default templates will be used. "
            "Enable the checkbox above to customize."
        )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="LLM Object Detection Console",
        page_icon="\U0001f50d",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()

    # One-time refresh of server status on initial load
    if st.session_state.server_manager is not None:
        refresh_server_status()

    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>\U0001f50d LLM Object Detection Console</h1>
            <p>// vision-LLM detector/judge pipeline over a local or
               external endpoint</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Server status badge (static, updated by fragment)
    sm = st.session_state.server_manager
    port = sm.port if sm else None
    st.markdown(
        render_status_badge(st.session_state.server_status, port),
        unsafe_allow_html=True,
    )

    st.divider()

    # Main tabs
    tab_server, tab_batch, tab_prompts = st.tabs(
        [
            "\U0001f999 Llama Server",
            "\U0001f9ea Batch Sandbox",
            "\u270d\ufe0f Prompts",
        ]
    )

    with tab_server:
        render_server_tab()

    with tab_batch:
        render_batch_sandbox_tab()

    with tab_prompts:
        render_prompts_tab()


if __name__ == "__main__":
    main()
