"""
LLM Object Detection Console.

Styled with the dark "terminal" Gradio console look (console_theme.py +
console.css + console.js) — see references/patterns.md in the
gradio-api-console skill for the rationale behind each pattern reused here.
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
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import httpx
from PIL import Image
import numpy as np
from openai import OpenAI

from interface.console_theme import theme

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    RoundResult,
    draw_grid,
    pil_to_data_uri,
    parse_detections,
    validate_detections,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
)
from servers import LlamaServerManager

# Interface assets (console.css / console.js) live at src/interface/, the
# package root's sibling, NOT inside the free_detection package itself.
_iface_dir = os.path.join(os.path.dirname(__file__), os.pardir, "interface")

with open(os.path.join(_iface_dir, "console.css"), encoding="utf-8") as f:
    custom_css = f.read()
with open(os.path.join(_iface_dir, "console.js"), encoding="utf-8") as f:
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
# Global State & Caching
# ---------------------------------------------------------------------------

server_manager: Optional[LlamaServerManager] = None
server_lock = threading.Lock()
pipeline_cancel_event = threading.Event()

BATCH_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
BATCH_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Cache helpers
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
# Helpers
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


DISPLAY_MAX_PX = 1280  # max long-edge pixels sent to the browser


def _thumb(
    img: Optional[Image.Image], max_px: int = DISPLAY_MAX_PX
) -> Optional[Image.Image]:
    """Return a display-sized copy of *img* so large images don't stall the UI."""
    if img is None:
        return None
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


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


# ---------------------------------------------------------------------------
# Server Manager Wrappers
# ---------------------------------------------------------------------------


def start_server_wrapper(
    model,
    port,
    host,
    enable_thinking,
    enable_mtp,
    ctx_size,
    gpu_layers,
    kv_cache_type,
    image_min_tokens,
    image_max_tokens,
    parallel_slots,
    batch_size,
    ubatch_size,
):
    ctx_size = ctx_size * parallel_slots
    global server_manager

    with server_lock:
        if server_manager is not None and server_manager.is_healthy():
            yield "Server is already running and healthy.", f'<span class="status-badge badge-running">RUNNING (Port {server_manager.port})</span>'
            return

        yield "Stopping any existing server instance...", '<span class="status-badge badge-starting">CLEANING UP...</span>'
        if server_manager is not None:
            try:
                server_manager.stop_llama_server()
            except Exception as e:
                print(f"Error stopping old server: {e}")
            server_manager = None

        yield "Configuring server...", '<span class="status-badge badge-starting">INITIALIZING...</span>'

        spec_type = "draft-mtp" if enable_mtp else "none"
        server_manager = LlamaServerManager(
            model=model,
            host=host,
            port=int(port),
            ctx_size=int(ctx_size),
            parallel_slots=parallel_slots,
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
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            kv_cache_type=kv_cache_type,
            image_min_tokens=(
                int(image_min_tokens) if image_min_tokens is not None else 1024
            ),
            image_max_tokens=(
                int(image_max_tokens) if image_max_tokens is not None else 4096
            ),
        )

        yield "Spawning llama-server process...", '<span class="status-badge badge-starting">STARTING...</span>'
        try:
            server_manager.start_llama_server()
        except Exception as e:
            server_manager = None
            yield f"Failed to start server process: {e}", '<span class="status-badge badge-error">PROCESS ERROR</span>'
            return

    start_time = time.time()
    timeout = 180
    healthy = False

    while time.time() - start_time < timeout:
        with server_lock:
            if server_manager is None:
                yield "Server initialization aborted.", '<span class="status-badge badge-stopped">STOPPED</span>'
                return
            if server_manager.process and server_manager.process.poll() is not None:
                exit_code = server_manager.process.poll()
                logs = server_manager.get_logs()
                server_manager = None
                yield f"Server process exited with code {exit_code}.\n\n--- Logs ---\n{logs}", '<span class="status-badge badge-error">CRASHED</span>'
                return
            if server_manager.is_healthy():
                healthy = True
                break

            logs = server_manager.get_logs()
            elapsed = int(time.time() - start_time)
            yield f"Waiting for model to load into memory... ({elapsed}s elapsed)\n\n--- Latest Output ---\n{logs[-1200:]}", '<span class="status-badge badge-starting">STARTING...</span>'
        time.sleep(2)

    if healthy:
        yield "Server is up. Running warmup request...", '<span class="status-badge badge-starting">WARMING UP...</span>'
        try:
            with server_lock:
                if server_manager:
                    server_manager.warmup_model()
            yield "Server started and warmed up. Ready for detection tasks.", f'<span class="status-badge badge-running">RUNNING (Port {port})</span>'
        except Exception as e:
            yield f"Server is healthy, but warmup failed: {e}", f'<span class="status-badge badge-running">RUNNING (Port {port})</span>'
    else:
        yield "Timed out waiting for the server to report healthy status.", '<span class="status-badge badge-error">TIMEOUT</span>'


def stop_server_wrapper():
    global server_manager
    with server_lock:
        if server_manager is None:
            return (
                "No server running.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        try:
            server_manager.stop_llama_server()
            server_manager = None
            return (
                "Server stopped successfully.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        except Exception as e:
            return (
                f"Error stopping server: {e}",
                '<span class="status-badge badge-error">STOP ERROR</span>',
            )


def get_server_status_and_logs():
    global server_manager
    with server_lock:
        if server_manager is None:
            return (
                "No server instance exists.",
                '<span class="status-badge badge-stopped">STOPPED</span>',
            )
        if server_manager.process and server_manager.process.poll() is not None:
            exit_code = server_manager.process.poll()
            return (
                f"Server process is dead (Exit code: {exit_code}).\n\n--- Logs ---\n{server_manager.get_logs()}",
                '<span class="status-badge badge-error">CRASHED</span>',
            )
        logs = server_manager.get_logs()
        if server_manager.is_healthy():
            return (
                f"Server is healthy and running.\n\n--- Logs ---\n{logs[-2000:]}",
                f'<span class="status-badge badge-running">RUNNING (Port {server_manager.port})</span>',
            )
        return (
            f"Server is starting or unhealthy.\n\n--- Logs ---\n{logs[-2000:]}",
            '<span class="status-badge badge-starting">STARTING...</span>',
        )


# ---------------------------------------------------------------------------
# Pipeline Runner helpers
# ---------------------------------------------------------------------------


class PipelineCancelledException(Exception):
    """Raised when a user cancels the pipeline mid-run."""

    pass


def _render_status_table(image_status: Dict[str, dict], order: List[str]) -> str:
    rows = []
    for stem in order:
        st = image_status.get(stem)
        if not st:
            continue
        pill = _STATUS_PILL.get(st["state"], _STATUS_PILL["queued"])
        score = st.get("score")
        score_txt = f"{score}/10" if score is not None else "—"
        rounds_txt = str(st.get("rounds_done", 0))
        detail = st.get("detail", "") or ""
        name_esc = html.escape(st["name"])
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
        else '<tr><td colspan="5" style="color:#7d8590;text-align:center;padding:1rem;">No images yet.</td></tr>'
    )
    return f"""
<div class="output-panel" style="margin-top:0.75rem">
  <div class="out-header"><div class="out-header-left">
    <span class="out-header-dot"></span><span class="out-header-title">Batch Status ({len(order)} images)</span>
  </div></div>
  <div style="max-height:260px; overflow-y:auto;">
  <table class="batch-status-table">
    <thead><tr>
      <th>Image</th><th>Status</th>
      <th>Rounds</th><th>Score</th>
      <th>Detail</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_batch_detection_gui(
    image_files,
    categories_str,
    category_definitions,
    local_server_port,
    use_external_api,
    ext_api_url,
    ext_api_key,
    ext_model_name,
    max_rounds,
    score_threshold,
    detector_temp,
    judge_temp,
    concurrency,
    customize_prompts,
    detector_template,
    judge_template,
    prep_enabled,
    prep_short_edge,
    prep_pad_square,
    prep_contrast_method,
    prep_gamma,
    prep_denoise_method,
    prep_sharpen,
    prep_white_balance,
    prep_grid_style,
    prep_som_enabled,
    prep_tiling_enabled,
    prep_tile_size,
    prep_tile_overlap,
    prep_crop_verify_enabled,
    prep_crop_padding,
    prep_grid_step,
    prep_grid_line_width,
    prep_grid_font_size,
    prep_grid_line_color,
    prep_grid_line_color_custom,
    prep_grid_text_color,
    prep_grid_text_color_custom,
    prep_grid_backing_color,
    prep_grid_backing_color_custom,
    prep_send_pixel_bounds,
    prep_min_pixels,
    prep_max_pixels,
    prep_custom_resize_enabled,
    prep_custom_resize_width,
    prep_custom_resize_height,
    judge_thinking,
    feedback_image_mode,
):
    pipeline_cancel_event.clear()

    _empty_yield = (
        None,
        "",
        gr.update(choices=[]),
        "",
        _render_status_table({}, []),
    )

    if not image_files:
        yield "Error: Please upload at least one image.", _render_progress_bar(
            0
        ), *_empty_yield
        return

    categories = [c.strip() for c in categories_str.split(",") if c.strip()]
    if not categories:
        yield "Error: Please list at least one category.", _render_progress_bar(
            0
        ), *_empty_yield
        return

    image_paths: List[Path] = []
    for f in image_files:
        if isinstance(f, str):
            image_paths.append(Path(f))
        elif hasattr(f, "name"):
            image_paths.append(Path(f.name))
        elif isinstance(f, dict) and "name" in f:
            image_paths.append(Path(f["name"]))
    if not image_paths:
        yield "Error: Could not resolve uploaded files.", _render_progress_bar(
            0
        ), *_empty_yield
        return

    cleaned_paths: List[Path] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im.verify()
            cleaned_paths.append(p)
        except Exception as e:
            yield f"Error: file '{p.name}' is not a valid image ({e}).", _render_progress_bar(
                0
            ), *_empty_yield
            return
    image_paths = cleaned_paths

    concurrency = max(1, int(concurrency or DEFAULT_CONCURRENCY))

    yield "Initializing API clients...", _render_progress_bar(
        2, "Initializing..."
    ), None, "", gr.update(choices=[]), "", _render_status_table({}, [])

    if use_external_api:
        api_url, api_key, model_name = ext_api_url, ext_api_key, ext_model_name
        if not api_key or api_key == "your-key":
            yield (
                "Error: External API selected but no API key provided. "
                "Set one in the External API section."
            ), _render_progress_bar(0, "Error"), None, "", gr.update(
                choices=[]
            ), "", _render_status_table(
                {}, []
            )
            return
    else:
        with server_lock:
            if server_manager is None or not server_manager.is_healthy():
                yield "Error: Local server not running. Start it on the Server tab or enable External API.", _render_progress_bar(
                    0, "Error"
                ), None, "", gr.update(
                    choices=[]
                ), "", _render_status_table(
                    {}, []
                )
                return
            port = server_manager.port
            model_name = server_manager.model
        api_url = f"http://localhost:{port}/v1"
        api_key = "not-needed"

    try:
        http_client = httpx.Client(
            timeout=httpx.Timeout(None),
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
        )
        client = OpenAI(base_url=api_url, api_key=api_key, http_client=http_client)
    except Exception as e:
        yield f"Error initializing OpenAI client: {e}", _render_progress_bar(
            0, "Error"
        ), None, "", gr.update(choices=[]), "", _render_status_table({}, [])
        return

    batch_id = str(int(time.time()))
    batch_logger = logging.getLogger(f"detection_pipeline.batch_{batch_id}")
    batch_logger.setLevel(logging.INFO)
    batch_logger.propagate = False

    log_capture = io.StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    batch_logger.addHandler(log_handler)
    log_lock = threading.Lock()

    det_tmpl = detector_template if customize_prompts else DEFAULT_DETECTOR_TEMPLATE
    jdg_tmpl = judge_template if customize_prompts else DEFAULT_JUDGE_TEMPLATE

    run_dir = Path("./gui_runs") / f"run_{batch_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build preprocessing config
    if not prep_enabled:
        prep_config = {
            "resolution_enabled": False,
            "contrast_method": "none",
            "denoise_method": "none",
            "som_enabled": False,
            "tiling_enabled": False,
            "crop_verify_enabled": False,
            "grid_style": "standard",
            "grid_step": 100,
            "grid_line_width": 1,
            "grid_font_size": 0,
            "grid_line_color": "red",
            "grid_text_color": "white",
            "grid_backing_color": "black",
            "send_pixel_bounds": False,
            "min_pixels": 200704,
            "max_pixels": 4194304,
            "custom_resize": False,
            "custom_resize_width": 1024,
            "custom_resize_height": 1024,
        }
    else:
        use_custom_resize = (
            prep_custom_resize_enabled and prep_custom_resize_enabled is not None
        )
        prep_config = {
            "resolution_enabled": True if not use_custom_resize else False,
            "target_short_edge": int(prep_short_edge),
            "pad_to_square": prep_pad_square,
            "contrast_method": prep_contrast_method,
            "clip_limit": 2.0,
            "gamma": float(prep_gamma),
            "denoise_method": prep_denoise_method,
            "sharpen": prep_sharpen,
            "white_balance": prep_white_balance,
            "grid_style": (
                prep_grid_style if prep_grid_style != "Standard Red" else "standard"
            ),
            "som_enabled": prep_som_enabled,
            "tiling_enabled": prep_tiling_enabled,
            "tile_size": int(prep_tile_size),
            "tile_overlap": float(prep_tile_overlap) / 100.0,
            "crop_verify_enabled": prep_crop_verify_enabled,
            "crop_padding": float(prep_crop_padding) / 100.0,
            "grid_step": int(prep_grid_step),
            "grid_line_width": int(prep_grid_line_width),
            "grid_font_size": int(prep_grid_font_size),
            "grid_line_color": (
                prep_grid_line_color
                if prep_grid_line_color != "custom"
                else prep_grid_line_color_custom
            ),
            "grid_text_color": (
                prep_grid_text_color
                if prep_grid_text_color != "custom"
                else prep_grid_text_color_custom
            ),
            "grid_backing_color": (
                prep_grid_backing_color
                if prep_grid_backing_color != "custom"
                else prep_grid_backing_color_custom
            ),
            "send_pixel_bounds": prep_send_pixel_bounds,
            "min_pixels": int(prep_min_pixels) if prep_min_pixels is not None else None,
            "max_pixels": int(prep_max_pixels) if prep_max_pixels is not None else None,
            "custom_resize": use_custom_resize,
            "custom_resize_width": (
                int(prep_custom_resize_width)
                if prep_custom_resize_width is not None
                else 1024
            ),
            "custom_resize_height": (
                int(prep_custom_resize_height)
                if prep_custom_resize_height is not None
                else 1024
            ),
        }

    batch_results: Dict[str, Any] = {}
    _cache_put(batch_id, batch_results)
    results_lock = threading.Lock()

    q: queue.Queue = queue.Queue()
    worker_done = threading.Event()

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

    def process_one_image(img_path: Path):
        stem = stem_for_path[img_path]
        if pipeline_cancel_event.is_set():
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
                    "grid_original": draw_grid(
                        base_image,
                        step=prep_config.get("grid_step", 100),
                        style=prep_config.get("grid_style", "standard"),
                        line_color=prep_config.get("grid_line_color", "red"),
                        line_width=prep_config.get("grid_line_width", 1),
                        font_size=prep_config.get("grid_font_size", 0),
                        text_color=prep_config.get("grid_text_color", "white"),
                        backing_color=prep_config.get("grid_backing_color", "black"),
                    ),
                    "raw_original": base_image,
                    "best_annotated": None,
                    "detections": [],
                    "rounds": [],
                }

            def progress_callback(
                round_result: RoundResult, annotated_image: Image.Image, _stem=stem
            ):
                if pipeline_cancel_event.is_set():
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
                preprocessing_config=prep_config,
                judge_enable_thinking=judge_thinking,
                feedback_image_mode=feedback_image_mode,
                external_api=use_external_api,
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
            with log_lock:
                batch_logger.error(f"[{stem}] {e}\n{traceback.format_exc()}")
            q.put(("image_error", stem, str(e)))

    def worker():
        try:
            if not pipeline_cancel_event.is_set():
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(process_one_image, p) for p in image_paths]
                    for fut in as_completed(futures):
                        exc = fut.exception()
                        if exc is not None:
                            with log_lock:
                                batch_logger.error(
                                    f"Unhandled worker exception: {exc}\n{traceback.format_exc()}"
                                )
                            q.put(("image_error", "unknown", str(exc)))

            if pipeline_cancel_event.is_set():
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
            worker_done.set()

    threading.Thread(target=worker, daemon=True).start()

    image_status: Dict[str, dict] = {
        stem: {
            "name": img_path.name,
            "state": "queued",
            "rounds_done": 0,
            "score": None,
            "detail": "",
        }
        for img_path, stem in stem_for_path.items()
    }

    # Initial yield
    yield (
        f"Starting batch ({total_imgs} images, {concurrency} concurrent)...",
        _render_progress_bar(5, "Starting batch..."),
        None,
        batch_id,
        gr.update(choices=[]),
        "",
        _render_status_table(image_status, stem_order),
    )

    finished_count = 0
    errored_count = 0
    last_active_stem = ""
    last_yield_time = time.time()

    while True:
        try:
            msg = q.get(timeout=0.2)
            tag = msg[0]
            status_msg = "Processing..."
            is_terminal = False

            if tag == "start_image":
                stem = msg[2]
                last_active_stem = stem
                image_status[stem]["state"] = "running"
                running_n = sum(
                    1 for s in image_status.values() if s["state"] == "running"
                )
                status_msg = f"Processing ({finished_count}/{total_imgs} done) — {running_n} running concurrently..."

            elif tag == "round":
                stem, r_res, r_img = msg[1], msg[2], msg[3]
                with results_lock:
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
                image_status[stem]["rounds_done"] = r_res.round
                image_status[stem]["score"] = r_res.score
                status_msg = (
                    f"{stem}: round {r_res.round} done (score {r_res.score}/10)."
                )

            elif tag == "finish_image":
                stem = msg[1]
                finished_count += 1
                image_status[stem]["state"] = "done"
                status_msg = f"Finished {stem} ({finished_count}/{total_imgs})."

            elif tag == "image_error":
                stem, err = msg[1], msg[2]
                finished_count += 1
                errored_count += 1
                if stem in image_status:
                    image_status[stem]["state"] = "error"
                    image_status[stem]["detail"] = err[:200]
                status_msg = f"⚠ {stem} failed: {err[:160]}"

            elif tag == "image_cancelled":
                stem = msg[1]
                if stem in image_status:
                    image_status[stem]["state"] = "cancelled"
                status_msg = f"{stem} cancelled."

            elif tag == "image_skipped":
                stem = msg[1]
                if stem in image_status:
                    image_status[stem]["state"] = "cancelled"
                status_msg = "Batch cancelled — skipping remaining queued images."

            elif tag == "done":
                zip_path = msg[1]
                summary = f"Batch complete: {finished_count - errored_count} succeeded, {errored_count} failed."
                if not last_active_stem and stem_order:
                    last_active_stem = stem_order[0]
                yield (
                    summary,
                    _render_progress_bar(100, "Complete"),
                    zip_path,
                    batch_id,
                    gr.update(choices=stem_order, value=last_active_stem or None),
                    _tail(log_capture.getvalue()),
                    _render_status_table(image_status, stem_order),
                )
                is_terminal = True

            elif tag == "cancelled":
                yield (
                    "Pipeline execution cancelled by the user.",
                    _render_progress_bar(100, "Cancelled"),
                    None,
                    batch_id,
                    gr.update(
                        choices=stem_order,
                        value=last_active_stem
                        or (stem_order[0] if stem_order else None),
                    ),
                    _tail(log_capture.getvalue()),
                    _render_status_table(image_status, stem_order),
                )
                is_terminal = True

            elif tag == "error":
                err_msg, trace = msg[1], msg[2]
                yield (
                    f"Pipeline execution failed:\n{err_msg}",
                    _render_progress_bar(100, "Error"),
                    None,
                    batch_id,
                    gr.update(
                        choices=stem_order,
                        value=last_active_stem
                        or (stem_order[0] if stem_order else None),
                    ),
                    _tail(log_capture.getvalue())
                    + f"\n[CRITICAL ERROR] {err_msg}\n{trace}",
                    _render_status_table(image_status, stem_order),
                )
                is_terminal = True

            if is_terminal:
                break

            # Throttle non-terminal yields to ~3fps to prevent websocket overload
            now = time.time()
            if now - last_yield_time > 0.33:
                done_n = sum(
                    1
                    for s in image_status.values()
                    if s["state"] in ("done", "error", "cancelled")
                )
                pct = int((done_n / total_imgs) * 90) if total_imgs else 0
                yield (
                    status_msg,
                    _render_progress_bar(pct, status_msg),
                    None,
                    batch_id,
                    gr.update(choices=stem_order, value=last_active_stem or None),
                    _tail(log_capture.getvalue()),
                    _render_status_table(image_status, stem_order),
                )
                last_yield_time = now

        except queue.Empty:
            if worker_done.is_set():
                yield (
                    "Pipeline ended unexpectedly (worker exited).",
                    _render_progress_bar(100, "Aborted"),
                    None,
                    batch_id,
                    gr.update(
                        choices=stem_order,
                        value=last_active_stem
                        or (stem_order[0] if stem_order else None),
                    ),
                    _tail(log_capture.getvalue()),
                    _render_status_table(image_status, stem_order),
                )
                break

            done_n = sum(
                1
                for s in image_status.values()
                if s["state"] in ("done", "error", "cancelled")
            )
            pct = int((done_n / total_imgs) * 90) if total_imgs else 0
            running_n = sum(1 for s in image_status.values() if s["state"] == "running")

            now = time.time()
            if now - last_yield_time > 0.33:
                yield (
                    f"Processing... ({done_n}/{total_imgs} done, {running_n} running)",
                    _render_progress_bar(pct, "Processing..."),
                    None,
                    batch_id,
                    gr.update(choices=stem_order, value=last_active_stem or None),
                    _tail(log_capture.getvalue()),
                    _render_status_table(image_status, stem_order),
                )
                last_yield_time = now

            time.sleep(0.1)  # Prevent CPU spin

    batch_logger.removeHandler(log_handler)
    log_handler.close()


def cancel_pipeline():
    pipeline_cancel_event.set()
    return (
        "Cancellation requested. In-flight images will finish their current round "
        "and write results; queued images will be skipped. "
        "The Run button will re-enable once the worker drains."
    )


# ---------------------------------------------------------------------------
# Explorer Callbacks
# ---------------------------------------------------------------------------


def on_explorer_image_change(selected_image, batch_id):
    batch_results = _cache_get(batch_id)
    if not batch_results or not selected_image or selected_image not in batch_results:
        return gr.update(choices=[], value=None)
    rounds = batch_results[selected_image].get("rounds", [])
    choices = ["Final Best"] + [str(r["round"]) for r in rounds]
    return gr.update(choices=choices, value="Final Best")


def on_explorer_round_change(selected_image, selected_round, batch_id, show_grid):
    batch_results = _cache_get(batch_id)
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
    src_img = _thumb(
        img_data["grid_original"] if show_grid else img_data["raw_original"]
    )

    if not selected_round or selected_round == "Final Best":
        best_annotated = img_data["best_annotated"]
        best_score, best_round_num, best_feedback, best_raw, best_err = (
            -1,
            -1,
            "No detections found.",
            "",
            "",
        )
        best_detections = img_data.get("detections") or []
        for r in img_data["rounds"]:
            if r["score"] > best_score:
                best_score = r["score"]
                best_round_num = r["round"]
                best_feedback = r["feedback"]
                best_raw = r["raw_text"]
                best_err = r["parse_error"]

        display_img = _thumb(best_annotated) if best_detections else src_img

        if best_score >= 0:
            score_text = f'<span class="score-badge">Best Score: {best_score}/10 (Round {best_round_num})</span>'
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
            display_img = _thumb(r["image"]) if round_detections else src_img
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
        print(f"Error loading round details: {e}")

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
# UI Toggle Helpers
# ---------------------------------------------------------------------------


def toggle_run_btn(is_running):
    return gr.update(interactive=not is_running), gr.update(interactive=is_running)


def toggle_external_api(use_external):
    return (
        gr.update(interactive=not use_external),  # start_server_btn
        gr.update(interactive=not use_external),  # stop_server_btn
        gr.update(interactive=not use_external),  # server_preset
        gr.update(interactive=not use_external),  # server_model_input
        gr.update(interactive=not use_external),  # server_port_input
        gr.update(interactive=not use_external),  # server_thinking_chk
        gr.update(interactive=not use_external),  # server_mtp_chk
        gr.update(visible=use_external, open=use_external),  # ext_api_group accordion
    )


def toggle_custom_color_field(choice):
    return gr.update(visible=(choice == "custom"))


# ---------------------------------------------------------------------------
# UI Sub-Builders
# ---------------------------------------------------------------------------


def _build_server_tab(server_status_badge):
    """Build the Llama Server tab and return all interactive components."""

    gr.HTML('<p class="section-label">Model Server Configuration</p>')
    with gr.Row(equal_height=False):
        # ── Left: Config ──────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML(
                '<div class="config-card"><div class="config-card-title">🦙 Model Selection</div>'
            )
            server_preset = gr.Dropdown(
                label="Recommended Model Presets",
                choices=MODEL_PRESETS,
                value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                interactive=True,
            )
            server_model_input = gr.Textbox(
                label="Model GGUF Path or HF Repo ID",
                value="unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                placeholder="e.g. C:/models/qwen.gguf or HF ID",
                interactive=True,
            )
            gr.HTML("</div>")

            gr.HTML(
                '<div class="config-card"><div class="config-card-title">⚙️ Runtime Options</div>'
            )
            server_port_input = gr.Number(
                label="Port Number",
                value=8080,
                precision=0,
                interactive=True,
            )
            with gr.Row():
                server_thinking_chk = gr.Checkbox(
                    label="Thinking Mode", value=False, interactive=True
                )
                server_mtp_chk = gr.Checkbox(
                    label="MTP Speculative Drafting",
                    value=True,
                    interactive=True,
                )
            gr.HTML("</div>")

            with gr.Accordion("Advanced Server Parameters", open=False):
                gr.HTML(_section_title("🖧", "Network"))
                server_host_input = gr.Textbox(label="Host Binding", value="0.0.0.0")
                gr.HTML(_section_title("🎛️", "Compute"))
                server_ctx_input = gr.Number(
                    label="Context Size per Slot",
                    value=15000,
                    precision=0,
                )
                server_parallel_slots_input = gr.Number(
                    label="Parallel Slots", value=2, precision=0
                )
                server_batch_size = gr.Number(
                    label="Batch Size", value=1024, precision=0
                )
                server_ubatch_size = gr.Number(
                    label="Unbatched Size", value=512, precision=0
                )
                server_gpu_layers = gr.Number(
                    label="GPU Layers (-ngl)", value=-1, precision=0
                )
                server_kv_cache = gr.Dropdown(
                    label="KV Cache Type",
                    choices=[
                        "f32",
                        "f16",
                        "bf16",
                        "q8_0",
                        "q4_0",
                        "q4_1",
                        "iq4_nl",
                        "q5_0",
                        "q5_1",
                    ],
                    value="q4_0",
                )
                gr.HTML(_section_title("🖼️", "Vision / Image Tokens"))
                with gr.Row():
                    server_img_min_tokens = gr.Number(
                        label="Min Image Tokens (--image-min-tokens)",
                        value=1024,
                        precision=0,
                        info="Minimum tokens for image encoding. Lower = faster but lower quality.",
                    )
                    server_img_max_tokens = gr.Number(
                        label="Max Image Tokens (--image-max-tokens)",
                        value=4096,
                        precision=0,
                        info="Maximum tokens for image encoding. Higher = more detail but slower.",
                    )

            gr.HTML('<div class="btn-group" style="margin-top:0.75rem;">')
            with gr.Row():
                start_server_btn = gr.Button("▶  Start Server", variant="primary")
                stop_server_btn = gr.Button(
                    "⏹  Stop Server", variant="secondary", size="sm"
                )
                refresh_logs_btn = gr.Button(
                    "🔄 Refresh Logs",
                    variant="secondary",
                    size="sm",
                )
            gr.HTML("</div>")

        # ── Right: Logs ───────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<p class="section-label">Server Output Console</p>')
            gr.HTML(
                '<div class="output-panel" id="server-log-panel">'
                + panel_header("Live Logs", "server-log-ta")
            )
            with gr.Group(elem_classes=["out-md-wrap"]):
                server_logs_viewer = gr.Textbox(
                    lines=22,
                    max_lines=32,
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="server-log-ta",
                )
            gr.HTML("</div>")

    return dict(
        server_preset=server_preset,
        server_model_input=server_model_input,
        server_port_input=server_port_input,
        server_host_input=server_host_input,
        server_thinking_chk=server_thinking_chk,
        server_mtp_chk=server_mtp_chk,
        server_ctx_input=server_ctx_input,
        server_parallel_slots_input=server_parallel_slots_input,
        server_batch_size=server_batch_size,
        server_ubatch_size=server_ubatch_size,
        server_gpu_layers=server_gpu_layers,
        server_kv_cache=server_kv_cache,
        server_img_min_tokens=server_img_min_tokens,
        server_img_max_tokens=server_img_max_tokens,
        start_server_btn=start_server_btn,
        stop_server_btn=stop_server_btn,
        refresh_logs_btn=refresh_logs_btn,
        server_logs_viewer=server_logs_viewer,
    )


def _build_batch_tab():
    """Build the Batch Sandbox tab and return all interactive components."""

    with gr.Row(equal_height=False):
        # ── Left: Config ──────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=400):
            gr.HTML('<p class="section-label">Configuration</p>')

            input_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload Source Image(s)",
            )
            categories_input = gr.Textbox(
                label="Target Categories (comma-separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                value="hole, stain, tear, cut, knot, weaving_defect",
            )
            category_defs_input = gr.Textbox(
                label="Category Definitions",
                placeholder="Write instructions for categories...",
                lines=4,
                value=(
                    "- hole: missing fabric\n"
                    "- stain: discoloration only\n"
                    "- tear: frayed, uneven separation\n"
                    "- cut: clean cut\n"
                    "- knot: raise lump\n"
                    "- weaving_defect: uneven thread density"
                ),
            )

            with gr.Accordion("Pipeline Parameters", open=False):
                rounds_slider = gr.Slider(
                    label="Optimization Max Rounds",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                )
                score_threshold_slider = gr.Slider(
                    label="Stop Score Threshold (0-10)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=8,
                )
                det_temp_slider = gr.Slider(
                    label="Detector Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.9,
                )
                jdg_temp_slider = gr.Slider(
                    label="Judge Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.2,
                )
                judge_thinking_chk = gr.Checkbox(
                    label="Enable Judge Thinking Mode",
                    value=False,
                    info="Directs the VLM judge backend to use thinking/reasoning outputs (if supported).",
                )
                feedback_image_mode_dropdown = gr.Dropdown(
                    label="Feedback Image Mode (Round > 1)",
                    choices=["original", "annotated", "both"],
                    value="original",
                    info="Choose which images are sent to the detector in subsequent rounds (clean original, annotated with previous boxes, or both).",
                )

            with gr.Accordion("Image Preprocessing & Augmentation", open=False):
                prep_enabled_chk = gr.Checkbox(
                    label="Enable Preprocessing",
                    value=False,
                    info="Master toggle for all preprocessing steps below.",
                )

                with gr.Group(visible=False) as prep_options_group:
                    gr.HTML(_section_title("📐", "Resolution & Padding"))
                    prep_short_edge_slider = gr.Slider(
                        label="Target Short Edge (px)",
                        minimum=512,
                        maximum=2048,
                        step=128,
                        value=1024,
                        info="Upscale short edge to at least this value.",
                    )
                    prep_pad_square_chk = gr.Checkbox(
                        label="Pad to Square",
                        value=False,
                        info="Pad with neutral gray to maintain aspect ratio on square inputs.",
                    )

                    gr.HTML(_section_title("✂️", "Custom Resize"))
                    prep_custom_resize_chk = gr.Checkbox(
                        label="Enable Custom Resize (override short edge)",
                        value=False,
                        info="Resize all images to exact width × height. Overrides the short-edge target.",
                    )
                    with gr.Row(visible=False) as prep_custom_resize_row:
                        prep_custom_resize_width = gr.Number(
                            label="Target Width (px)",
                            value=1024,
                            precision=0,
                        )
                        prep_custom_resize_height = gr.Number(
                            label="Target Height (px)",
                            value=1024,
                            precision=0,
                        )

                    gr.HTML(_section_title("🎨", "Contrast & Color"))
                    prep_contrast_dropdown = gr.Dropdown(
                        label="Contrast Correction Method",
                        choices=["none", "clahe", "autocontrast"],
                        value="clahe",
                    )
                    prep_gamma_slider = gr.Slider(
                        label="Gamma Correction",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                    )
                    prep_wb_chk = gr.Checkbox(
                        label="Gray World White Balance Correction",
                        value=False,
                    )

                    gr.HTML(_section_title("🔇", "Noise & Sharpness"))
                    prep_denoise_dropdown = gr.Dropdown(
                        label="Denoising Filter",
                        choices=["none", "bilateral", "nlm"],
                        value="none",
                    )
                    prep_sharpen_chk = gr.Checkbox(
                        label="Apply Unsharp Mask (Sharpen)", value=False
                    )

                    gr.HTML(_section_title("🔲", "Coordinate Grid Overlay"))
                    prep_grid_dropdown = gr.Dropdown(
                        label="Grid Style",
                        choices=["Standard Red", "transparent", "fine", "none"],
                        value="Standard Red",
                        info="Select standard, semi-transparent, fine 10×10 grid, or disable.",
                    )
                    prep_grid_step_slider = gr.Slider(
                        label="Grid Step Size (px)",
                        minimum=20,
                        maximum=500,
                        step=10,
                        value=250,
                        info="Distance between grid lines.",
                    )
                    prep_grid_line_width_slider = gr.Slider(
                        label="Grid Line Thickness (px)",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                    )
                    prep_grid_font_size_slider = gr.Slider(
                        label="Grid Label Font Size (0 = Auto)",
                        minimum=0,
                        maximum=48,
                        step=1,
                        value=0,
                    )
                    with gr.Row():
                        prep_grid_line_color_dropdown = gr.Dropdown(
                            label="Grid Line Color",
                            choices=[
                                "red",
                                "blue",
                                "green",
                                "white",
                                "black",
                                "yellow",
                                "cyan",
                                "magenta",
                                "custom",
                            ],
                            value="red",
                        )
                        prep_grid_line_color_custom = gr.Textbox(
                            label="Custom Line Color (Hex/Name)",
                            value="red",
                            visible=False,
                        )
                    with gr.Row():
                        prep_grid_text_color_dropdown = gr.Dropdown(
                            label="Grid Text Color",
                            choices=[
                                "white",
                                "black",
                                "red",
                                "blue",
                                "green",
                                "yellow",
                                "cyan",
                                "magenta",
                                "custom",
                            ],
                            value="white",
                        )
                        prep_grid_text_color_custom = gr.Textbox(
                            label="Custom Text Color (Hex/Name)",
                            value="white",
                            visible=False,
                        )
                    with gr.Row():
                        prep_grid_backing_color_dropdown = gr.Dropdown(
                            label="Grid Text Backing Color",
                            choices=[
                                "black",
                                "none",
                                "white",
                                "red",
                                "blue",
                                "green",
                                "custom",
                            ],
                            value="black",
                        )
                        prep_grid_backing_color_custom = gr.Textbox(
                            label="Custom Backing (Hex/Name)",
                            value="black",
                            visible=False,
                        )

                    gr.HTML(_section_title("🎯", "Visual Prompting (SoM)"))
                    prep_som_chk = gr.Checkbox(
                        label="Enable Set-of-Mark (SoM) Prompting",
                        value=False,
                        info="Detect candidate regions and overlay numbered circles as hints.",
                    )

                    gr.HTML(_section_title("🧩", "Tiling (Small Objects)"))
                    prep_tiling_chk = gr.Checkbox(
                        label="Enable Image Tiling",
                        value=False,
                        info="Split image into overlapping tiles, detect independently, and merge via NMS.",
                    )
                    prep_tile_size_slider = gr.Slider(
                        label="Tile Size (px)",
                        minimum=256,
                        maximum=1024,
                        step=128,
                        value=512,
                    )
                    prep_tile_overlap_slider = gr.Slider(
                        label="Tile Overlap (%)",
                        minimum=0,
                        maximum=50,
                        step=5,
                        value=20,
                    )

                    gr.HTML(_section_title("🔍", "Multi-Pass Crop & Verify"))
                    prep_cv_chk = gr.Checkbox(
                        label="Enable Crop & Verify Validation",
                        value=False,
                        info="Perform a second VLM validation pass on cropped detections.",
                    )
                    prep_cv_padding_slider = gr.Slider(
                        label="Crop Context Padding (%)",
                        minimum=0,
                        maximum=50,
                        step=5,
                        value=15,
                    )

                    gr.HTML(_section_title("📡", "VLM Processor Pixel Bounds"))
                    prep_send_pixel_bounds_chk = gr.Checkbox(
                        label="Send Pixel Bounds in API Request",
                        value=False,
                        info="Pass min_pixels/max_pixels in extra_body (Qwen-VL / vLLM backends).",
                    )
                    with gr.Row(visible=False) as prep_pixel_bounds_row:
                        prep_min_pixels_num = gr.Number(
                            label="min_pixels",
                            value=200704,
                            precision=0,
                            info="Default: 256×28×28",
                        )
                        prep_max_pixels_num = gr.Number(
                            label="max_pixels",
                            value=4194304,
                            precision=0,
                            info="Default: 2048×2048",
                        )

            use_external_api_chk = gr.Checkbox(
                label="Use External API instead of Local Server",
                value=False,
                info="When enabled, all requests are sent to the external endpoint below instead of the local llama-server.",
            )
            with gr.Accordion(
                "External API Settings", open=False, visible=False
            ) as ext_api_group:
                ext_api_url = gr.Textbox(
                    label="Base URL", value="https://api.openai.com/v1"
                )
                ext_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="sk-...",
                    value="",
                    type="password",
                )
                ext_model_name = gr.Textbox(label="Model Name", value="gpt-4o")

            with gr.Accordion("Advanced Settings", open=False):
                concurrency_slider = gr.Slider(
                    label="Concurrent Images",
                    info=(
                        "Images processed in parallel. With a single-slot local server, "
                        "high values just queue at the server. Set 8–32 for external APIs "
                        "or multi-slot servers."
                    ),
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=DEFAULT_CONCURRENCY,
                )

            gr.HTML('<div class="btn-group" style="margin-top:0.75rem;">')
            with gr.Row():
                run_btn = gr.Button(
                    "▶  Run Batch Pipeline",
                    variant="primary",
                    interactive=True,
                )
                stop_run_btn = gr.Button(
                    "⏹  Cancel",
                    variant="secondary",
                    size="sm",
                    interactive=False,
                )
            gr.HTML("</div>")

        # ── Right: Results ────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=600):
            gr.HTML('<p class="section-label">Results</p>')

            with gr.Group():
                pipeline_status = gr.Markdown("**Status: Idle**")
                progress_html = gr.HTML(value=_render_progress_bar(0, "Idle"))

            batch_status_table = gr.HTML(value=_render_status_table({}, []))
            download_results_box = gr.File(
                label="📥 Download Processed Results (.zip)",
                interactive=False,
            )

            with gr.Tabs():
                with gr.TabItem("🖼️ Batch Explorer"):
                    with gr.Row():
                        explorer_image_select = gr.Dropdown(
                            label="Select Image",
                            choices=[],
                            interactive=True,
                            scale=2,
                        )
                        explorer_round_select = gr.Dropdown(
                            label="Select Round",
                            choices=[],
                            interactive=True,
                            scale=2,
                        )
                        round_score_display = gr.HTML(
                            value='<span class="score-badge">Score: -/10</span>',
                            elem_classes="score-display",
                            scale=1,
                        )

                    with gr.Row():
                        show_grid_chk = gr.Checkbox(
                            label="Show 0-1000 coordinate grid", value=True
                        )

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            gr.HTML('<div class="img-viewer-wrap">')
                            source_image_viewer = gr.Image(
                                label="Source Image", type="pil"
                            )
                            gr.HTML("</div>")
                        with gr.Column(scale=1):
                            gr.HTML('<div class="img-viewer-wrap">')
                            best_annotated_viewer = gr.Image(
                                label="Annotated Image", type="pil"
                            )
                            gr.HTML("</div>")

                    round_feedback_display = gr.Textbox(
                        label="Judge's Feedback", lines=4, interactive=False
                    )

                    with gr.Accordion("Raw Response Details", open=False):
                        round_parse_error_display = gr.Textbox(
                            label="Parsing Errors", interactive=False
                        )
                        round_raw_response_display = gr.Textbox(
                            label="Raw Detector Text Response",
                            lines=6,
                            interactive=False,
                        )

                with gr.TabItem("📄 Detections JSON"):
                    gr.HTML(
                        '<div class="json-panel">'
                        '<div class="json-panel-hdr"><span class="dot-amber"></span>'
                        "Detections (JSON List)</div>"
                    )
                    with gr.Group(elem_classes=["json-panel-body"]):
                        detections_json_box = gr.Code(
                            language="json",
                            show_label=False,
                            value="[]",
                        )
                    gr.HTML("</div>")

                with gr.TabItem("📋 Pipeline Logs"):
                    gr.HTML(
                        '<div class="output-panel" id="pipeline-log-panel">'
                        + panel_header("Execution Logs", "pipeline-log-ta")
                    )
                    with gr.Group(elem_classes=["out-md-wrap"]):
                        pipeline_logs_viewer = gr.Textbox(
                            lines=22,
                            max_lines=32,
                            interactive=False,
                            show_label=False,
                            container=False,
                            elem_id="pipeline-log-ta",
                        )
                    gr.HTML("</div>")

    return dict(
        input_images=input_images,
        categories_input=categories_input,
        category_defs_input=category_defs_input,
        rounds_slider=rounds_slider,
        score_threshold_slider=score_threshold_slider,
        det_temp_slider=det_temp_slider,
        jdg_temp_slider=jdg_temp_slider,
        judge_thinking_chk=judge_thinking_chk,
        feedback_image_mode_dropdown=feedback_image_mode_dropdown,
        prep_enabled_chk=prep_enabled_chk,
        prep_options_group=prep_options_group,
        prep_short_edge_slider=prep_short_edge_slider,
        prep_pad_square_chk=prep_pad_square_chk,
        prep_custom_resize_chk=prep_custom_resize_chk,
        prep_custom_resize_row=prep_custom_resize_row,
        prep_custom_resize_width=prep_custom_resize_width,
        prep_custom_resize_height=prep_custom_resize_height,
        prep_contrast_dropdown=prep_contrast_dropdown,
        prep_gamma_slider=prep_gamma_slider,
        prep_wb_chk=prep_wb_chk,
        prep_denoise_dropdown=prep_denoise_dropdown,
        prep_sharpen_chk=prep_sharpen_chk,
        prep_grid_dropdown=prep_grid_dropdown,
        prep_grid_step_slider=prep_grid_step_slider,
        prep_grid_line_width_slider=prep_grid_line_width_slider,
        prep_grid_font_size_slider=prep_grid_font_size_slider,
        prep_grid_line_color_dropdown=prep_grid_line_color_dropdown,
        prep_grid_line_color_custom=prep_grid_line_color_custom,
        prep_grid_text_color_dropdown=prep_grid_text_color_dropdown,
        prep_grid_text_color_custom=prep_grid_text_color_custom,
        prep_grid_backing_color_dropdown=prep_grid_backing_color_dropdown,
        prep_grid_backing_color_custom=prep_grid_backing_color_custom,
        prep_som_chk=prep_som_chk,
        prep_tiling_chk=prep_tiling_chk,
        prep_tile_size_slider=prep_tile_size_slider,
        prep_tile_overlap_slider=prep_tile_overlap_slider,
        prep_cv_chk=prep_cv_chk,
        prep_cv_padding_slider=prep_cv_padding_slider,
        prep_send_pixel_bounds_chk=prep_send_pixel_bounds_chk,
        prep_pixel_bounds_row=prep_pixel_bounds_row,
        prep_min_pixels_num=prep_min_pixels_num,
        prep_max_pixels_num=prep_max_pixels_num,
        ext_api_group=ext_api_group,
        use_external_api_chk=use_external_api_chk,
        ext_api_url=ext_api_url,
        ext_api_key=ext_api_key,
        ext_model_name=ext_model_name,
        concurrency_slider=concurrency_slider,
        run_btn=run_btn,
        stop_run_btn=stop_run_btn,
        pipeline_status=pipeline_status,
        progress_html=progress_html,
        batch_status_table=batch_status_table,
        download_results_box=download_results_box,
        explorer_image_select=explorer_image_select,
        explorer_round_select=explorer_round_select,
        round_score_display=round_score_display,
        show_grid_chk=show_grid_chk,
        source_image_viewer=source_image_viewer,
        best_annotated_viewer=best_annotated_viewer,
        round_feedback_display=round_feedback_display,
        round_parse_error_display=round_parse_error_display,
        round_raw_response_display=round_raw_response_display,
        detections_json_box=detections_json_box,
        pipeline_logs_viewer=pipeline_logs_viewer,
    )


def _build_prompts_tab():
    """Build the Prompts tab and return all interactive components."""

    gr.HTML('<p class="section-label">Prompt Engineering</p>')
    gr.Markdown(
        "Modify the custom instruction templates fed to the **Detector** and **Judge** agents.\n\n"
        "Available template variables: `{categories}`, `{category_definitions}`, `{feedback}`, `{detections_json}`"
    )
    gr.HTML(
        '<div class="input-hint">'
        "Template variables: "
        '<span class="hint-var">{categories}</span> '
        '<span class="hint-var">{category_definitions}</span> '
        '<span class="hint-var">{feedback}</span> '
        '<span class="hint-var">{detections_json}</span>'
        "</div>"
    )

    customize_prompts_chk = gr.Checkbox(
        label="Enable Custom Prompt Templates", value=False
    )

    with gr.Group(visible=False) as prompts_group:
        custom_det_prompt = gr.Textbox(
            label="Detector Prompt Template",
            lines=14,
            value=DEFAULT_DETECTOR_TEMPLATE,
        )
        custom_jdg_prompt = gr.Textbox(
            label="Judge Prompt Template",
            lines=14,
            value=DEFAULT_JUDGE_TEMPLATE,
        )

    customize_prompts_chk.change(
        lambda v: gr.update(visible=v),
        customize_prompts_chk,
        prompts_group,
    )

    return dict(
        customize_prompts_chk=customize_prompts_chk,
        custom_det_prompt=custom_det_prompt,
        custom_jdg_prompt=custom_jdg_prompt,
    )


def _wire_events(c_srv, c_bat, c_pmt, server_status_badge, batch_id_state):
    """Wire all event handlers. c_srv / c_bat / c_pmt are the component dicts."""

    # ── Server tab ────────────────────────────────────────────────────────
    c_srv["server_preset"].change(
        handle_preset_change,
        c_srv["server_preset"],
        c_srv["server_model_input"],
    )

    c_srv["start_server_btn"].click(
        start_server_wrapper,
        inputs=[
            c_srv["server_model_input"],
            c_srv["server_port_input"],
            c_srv["server_host_input"],
            c_srv["server_thinking_chk"],
            c_srv["server_mtp_chk"],
            c_srv["server_ctx_input"],
            c_srv["server_gpu_layers"],
            c_srv["server_kv_cache"],
            c_srv["server_img_min_tokens"],
            c_srv["server_img_max_tokens"],
            c_srv["server_parallel_slots_input"],
            c_srv["server_batch_size"],
            c_srv["server_ubatch_size"],
        ],
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )
    c_srv["stop_server_btn"].click(
        stop_server_wrapper,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )
    c_srv["refresh_logs_btn"].click(
        get_server_status_and_logs,
        outputs=[c_srv["server_logs_viewer"], server_status_badge],
    )

    # ── Batch tab — preprocessing toggles ────────────────────────────────
    c_bat["prep_enabled_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_enabled_chk"]],
        outputs=[c_bat["prep_options_group"]],
    )

    c_bat["prep_custom_resize_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_custom_resize_chk"]],
        outputs=[c_bat["prep_custom_resize_row"]],
    )

    for dd, custom_field in [
        (c_bat["prep_grid_line_color_dropdown"], c_bat["prep_grid_line_color_custom"]),
        (c_bat["prep_grid_text_color_dropdown"], c_bat["prep_grid_text_color_custom"]),
        (
            c_bat["prep_grid_backing_color_dropdown"],
            c_bat["prep_grid_backing_color_custom"],
        ),
    ]:
        dd.change(toggle_custom_color_field, inputs=[dd], outputs=[custom_field])

    c_bat["prep_send_pixel_bounds_chk"].change(
        lambda v: gr.update(visible=v),
        inputs=[c_bat["prep_send_pixel_bounds_chk"]],
        outputs=[c_bat["prep_pixel_bounds_row"]],
    )


# ---------------------------------------------------------------------------
# Real-Time Webcam & Video Frame Detection (Neo-Retro Style)
# ---------------------------------------------------------------------------


def process_single_frame(
    frame: np.ndarray,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    confidence_thresh: float,
) -> tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, '<div class="neo-retro-hud-stat">STATUS: READY</div>'

    start_time = time.time()
    try:
        pil_img = Image.fromarray(frame).convert("RGB")
        categories = [c.strip() for c in categories_str.split(",") if c.strip()]
        if not categories:
            categories = ["object"]

        base_url = (
            ext_api_url if use_external_api else f"http://127.0.0.1:{server_port}/v1"
        )
        api_key = ext_api_key if use_external_api else "no-key"
        model_name = ext_model_name if use_external_api else "local-model"

        client = OpenAI(base_url=base_url, api_key=api_key)
        pipeline = ObjectDetectionPipeline(client=client, detector_model=model_name)

        img_uri = pil_to_data_uri(pil_img)
        raw_output = pipeline.call_detector(
            image_uris=img_uri,
            categories=categories,
            category_definitions="",
        )
        parsed_dets = parse_detections(raw_output)
        valid_dets = validate_detections(parsed_dets, categories)

        boxes = []
        for d in valid_dets:
            bbox = d.get("bbox_2d", [])
            lbl = d.get("label", "")
            if len(bbox) == 4:
                # Convert 0-1000 scale [x1, y1, x2, y2] to image coordinates [ymin, xmin, ymax, xmax, label]
                x1, y1, x2, y2 = bbox
                ymin = y1 * pil_img.height / 1000.0
                xmin = x1 * pil_img.width / 1000.0
                ymax = y2 * pil_img.height / 1000.0
                xmax = x2 * pil_img.width / 1000.0
                boxes.append([ymin, xmin, ymax, xmax, lbl])

        annotated_img = draw_grid(pil_img, style="none")
        annotated_np = draw_boxes_on_image(annotated_img, boxes)

        elapsed = (time.time() - start_time) * 1000
        hud_info = (
            f'<div class="neo-retro-hud-stat">FPS: {1000/max(elapsed, 1):.1f} | '
            f"LATENCY: {elapsed:.0f}ms | DETECTED: {len(boxes)}</div>"
        )
        return annotated_np, hud_info
    except Exception as e:
        return (
            frame,
            f'<div class="neo-retro-hud-stat" style="color:#ff0055 !important;">ERROR: {html.escape(str(e))}</div>',
        )


def draw_boxes_on_image(image: Image.Image, boxes: List[Any]) -> np.ndarray:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for box in boxes:
        if len(box) >= 4:
            ymin, xmin, ymax, xmax = box[:4]
            label = str(box[4]) if len(box) >= 5 else ""
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="#00ffcc",
                facecolor="none",
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    xmin,
                    max(0, ymin - 5),
                    label,
                    color="#050811",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="square,pad=0.2", facecolor="#00ffcc", edgecolor="none"
                    ),
                )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    res_img = Image.open(buf).convert("RGB")
    return np.array(res_img)


def process_video_frames(
    video_path: str,
    sample_interval: float,
    categories_str: str,
    server_port: int,
    use_external_api: bool,
    ext_api_url: str,
    ext_api_key: str,
    ext_model_name: str,
    progress=gr.Progress(),
) -> tuple[List[np.ndarray], str]:
    if not video_path:
        return [], "No video file uploaded."

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Failed to open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(max(1, fps * sample_interval))
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        frame_count += 1
    cap.release()

    annotated_frames = []

    for idx, f in enumerate(frames):
        progress((idx + 1) / len(frames), desc=f"Detecting frame {idx+1}/{len(frames)}")
        ann, _ = process_single_frame(
            f,
            categories_str,
            server_port,
            use_external_api,
            ext_api_url,
            ext_api_key,
            ext_model_name,
            0.3,
        )
        if ann is not None:
            annotated_frames.append(ann)

    return (
        annotated_frames,
        f"Successfully processed {len(annotated_frames)} frames from video!",
    )


def _build_realtime_tab() -> Dict[str, Any]:
    c = {}
    with gr.Column(elem_classes=["neo-retro-card"]):
        gr.HTML(
            """
        <div style="padding: 10px; border-bottom: 2px solid #00ffcc; background: #050811;">
            <span class="neo-retro-badge">LIVE CYBER-STREAM</span>
            <h2 style="color: #00ffcc; font-family: 'JetBrains Mono', monospace; margin: 5px 0 0;">
                ⚡ REAL-TIME WEBCAM & VIDEO FRAME DETECTOR
            </h2>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                c["stream_mode"] = gr.Radio(
                    choices=["Webcam Stream", "Video Upload (1s Sampling)"],
                    value="Webcam Stream",
                    label="STREAM INPUT SOURCE",
                )
                c["categories_input"] = gr.Textbox(
                    value="person, car, dog, bottle, phone",
                    label="TARGET CATEGORIES (comma-separated)",
                )
                c["sample_interval"] = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    step=0.5,
                    value=1.0,
                    label="VIDEO FRAME SAMPLING INTERVAL (SECONDS)",
                )
                c["process_video_btn"] = gr.Button(
                    "⚡ PROCESS VIDEO FRAMES",
                    variant="primary",
                    elem_classes=["neo-retro-badge"],
                )
                c["hud_status"] = gr.HTML(
                    value='<div class="neo-retro-hud-stat">STATUS: INITIALIZED</div>'
                )

            with gr.Column(scale=2):
                c["webcam_input"] = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="LIVE WEBCAM STREAM",
                    type="numpy",
                )
                c["video_input"] = gr.Video(
                    label="INPUT VIDEO FILE",
                    visible=False,
                )
                c["annotated_stream_output"] = gr.Image(
                    label="NEO-RETRO DETECTED STREAM / OVERLAY",
                    type="numpy",
                )
                c["video_gallery_output"] = gr.Gallery(
                    label="SAMPLED FRAME DETECTIONS (EVERY 1 SEC)",
                    visible=False,
                    columns=3,
                )

    return c


def _wire_realtime_events(
    c_real: Dict[str, Any],
    c_srv: Dict[str, Any],
    c_bat: Dict[str, Any],
    c_pmt: Dict[str, Any],
    batch_id_state,
):
    def toggle_mode(mode):
        is_cam = mode == "Webcam Stream"
        return (
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
            gr.update(visible=is_cam),
            gr.update(visible=not is_cam),
        )

    c_real["stream_mode"].change(
        toggle_mode,
        inputs=[c_real["stream_mode"]],
        outputs=[
            c_real["webcam_input"],
            c_real["video_input"],
            c_real["annotated_stream_output"],
            c_real["video_gallery_output"],
        ],
    )

    c_real["webcam_input"].stream(
        fn=process_single_frame,
        inputs=[
            c_real["webcam_input"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            gr.State(0.3),
        ],
        outputs=[c_real["annotated_stream_output"], c_real["hud_status"]],
        stream_every=0.5,
    )

    c_real["process_video_btn"].click(
        fn=process_video_frames,
        inputs=[
            c_real["video_input"],
            c_real["sample_interval"],
            c_real["categories_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
        ],
        outputs=[c_real["video_gallery_output"], c_real["hud_status"]],
    )

    # ── External API toggle ───────────────────────────────────────────────
    c_bat["use_external_api_chk"].change(
        toggle_external_api,
        inputs=[c_bat["use_external_api_chk"]],
        outputs=[
            c_srv["start_server_btn"],
            c_srv["stop_server_btn"],
            c_srv["server_preset"],
            c_srv["server_model_input"],
            c_srv["server_port_input"],
            c_srv["server_thinking_chk"],
            c_srv["server_mtp_chk"],
            c_bat["ext_api_group"],
        ],
    )

    # ── Run / Cancel ──────────────────────────────────────────────────────
    c_bat["run_btn"].click(
        fn=lambda: toggle_run_btn(is_running=True),
        inputs=None,
        outputs=[c_bat["run_btn"], c_bat["stop_run_btn"]],
        queue=False,
    ).then(
        fn=run_batch_detection_gui,
        inputs=[
            c_bat["input_images"],
            c_bat["categories_input"],
            c_bat["category_defs_input"],
            c_srv["server_port_input"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_bat["rounds_slider"],
            c_bat["score_threshold_slider"],
            c_bat["det_temp_slider"],
            c_bat["jdg_temp_slider"],
            c_bat["concurrency_slider"],
            c_pmt["customize_prompts_chk"],
            c_pmt["custom_det_prompt"],
            c_pmt["custom_jdg_prompt"],
            c_bat["prep_enabled_chk"],
            c_bat["prep_short_edge_slider"],
            c_bat["prep_pad_square_chk"],
            c_bat["prep_contrast_dropdown"],
            c_bat["prep_gamma_slider"],
            c_bat["prep_denoise_dropdown"],
            c_bat["prep_sharpen_chk"],
            c_bat["prep_wb_chk"],
            c_bat["prep_grid_dropdown"],
            c_bat["prep_som_chk"],
            c_bat["prep_tiling_chk"],
            c_bat["prep_tile_size_slider"],
            c_bat["prep_tile_overlap_slider"],
            c_bat["prep_cv_chk"],
            c_bat["prep_cv_padding_slider"],
            c_bat["prep_grid_step_slider"],
            c_bat["prep_grid_line_width_slider"],
            c_bat["prep_grid_font_size_slider"],
            c_bat["prep_grid_line_color_dropdown"],
            c_bat["prep_grid_line_color_custom"],
            c_bat["prep_grid_text_color_dropdown"],
            c_bat["prep_grid_text_color_custom"],
            c_bat["prep_grid_backing_color_dropdown"],
            c_bat["prep_grid_backing_color_custom"],
            c_bat["prep_send_pixel_bounds_chk"],
            c_bat["prep_min_pixels_num"],
            c_bat["prep_max_pixels_num"],
            c_bat["prep_custom_resize_chk"],
            c_bat["prep_custom_resize_width"],
            c_bat["prep_custom_resize_height"],
            c_bat["judge_thinking_chk"],
            c_bat["feedback_image_mode_dropdown"],
        ],
        outputs=[
            c_bat["pipeline_status"],
            c_bat["progress_html"],
            c_bat["download_results_box"],
            batch_id_state,
            c_bat["explorer_image_select"],
            c_bat["pipeline_logs_viewer"],
            c_bat["batch_status_table"],
        ],
        concurrency_limit=1,
    ).then(
        fn=lambda: toggle_run_btn(is_running=False),
        inputs=None,
        outputs=[c_bat["run_btn"], c_bat["stop_run_btn"]],
        queue=False,
    )

    c_bat["stop_run_btn"].click(
        fn=cancel_pipeline,
        outputs=[c_bat["pipeline_status"]],
        queue=False,
    )

    # ── Explorer interactions ─────────────────────────────────────────────
    _explorer_outputs = [
        c_bat["source_image_viewer"],
        c_bat["best_annotated_viewer"],
        c_bat["round_score_display"],
        c_bat["round_feedback_display"],
        c_bat["round_raw_response_display"],
        c_bat["round_parse_error_display"],
        c_bat["detections_json_box"],
    ]
    _explorer_inputs = [
        c_bat["explorer_image_select"],
        c_bat["explorer_round_select"],
        batch_id_state,
        c_bat["show_grid_chk"],
    ]

    c_bat["explorer_image_select"].change(
        on_explorer_image_change,
        inputs=[c_bat["explorer_image_select"], batch_id_state],
        outputs=[c_bat["explorer_round_select"]],
    ).then(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )

    c_bat["explorer_round_select"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )

    c_bat["show_grid_chk"].change(
        on_explorer_round_change,
        inputs=_explorer_inputs,
        outputs=_explorer_outputs,
    )


# ---------------------------------------------------------------------------
# App Builder
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=theme, css=custom_css, title="LLM Object Detection Console"
    ) as app:
        gr.HTML(CONSOLE_JS)

        # ── Header with inline status badge ──────────────────────────────
        gr.HTML(
            """
        <div class="app-header">
            <div>
                <h1><span>🔍</span> LLM Object Detection Console</h1>
                <p>// vision-LLM detector/judge pipeline · local or external endpoint</p>
            </div>
            <div class="app-header-meta" id="header-status-meta">
            </div>
        </div>"""
        )

        server_status_badge = gr.HTML(
            value='<span class="status-badge badge-stopped">STOPPED</span>',
        )

        batch_id_state = gr.State("")

        with gr.Tabs():
            with gr.TabItem("⚡ Real-Time & Video"):
                c_real = _build_realtime_tab()

            with gr.TabItem("🦙 Llama Server"):
                c_srv = _build_server_tab(server_status_badge)

            with gr.TabItem("🧪 Batch Sandbox"):
                c_bat = _build_batch_tab()

            with gr.TabItem("✍️ Prompts"):
                c_pmt = _build_prompts_tab()

        # ── Wire all events ───────────────────────────────────────────────
        _wire_events(c_srv, c_bat, c_pmt, server_status_badge, batch_id_state)
        _wire_realtime_events(c_real, c_srv, c_bat, c_pmt, batch_id_state)

        # ── Auto-refresh server status every 5 s ─────────────────────────
        status_timer = gr.Timer(value=5.0)
        app.load(
            get_server_status_and_logs,
            outputs=[c_srv["server_logs_viewer"], server_status_badge],
        )
        status_timer.tick(
            get_server_status_and_logs,
            outputs=[c_srv["server_logs_viewer"], server_status_badge],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, inline=True)
