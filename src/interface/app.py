"""
LLM Object Detection Console.

Styled with the dark "terminal" Gradio console look (console_theme.py +
console.css + console.js) — see references/patterns.md in the
gradio-api-console skill for the rationale behind each pattern reused here:

  - console_theme.py / console.css   -> shared dark GitHub-style look
  - panel_header() + .output-panel   -> output panel anatomy (header with
    copy button, stats bar, scrollable body, hidden raw textbox for JS)
  - .section-label                   -> lightweight uppercase section dividers
  - copyOut() (console.js)           -> client-side copy-to-clipboard

This app drives a local process (a llama-server instance) and a multi-round
detection pipeline rather than a stateless REST API, so the skill's
bearer-token-auth and history/pagination patterns don't apply here and were
intentionally left out — see the accompanying note for what was/wasn't
adopted from the skill.
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
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import gradio as gr
from PIL import Image
from openai import OpenAI

# console_theme.py / console.css / console.js live alongside this file.
from console_theme import theme

# Ensure local imports resolve
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from detection_pipeline import (
    ObjectDetectionPipeline,
    RoundResult,
    draw_grid,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE
)
from llama_server_manager import LlamaServerManager

with open(os.path.join(os.path.dirname(__file__), 'console.css'), encoding='utf-8') as f:
    custom_css = f.read()
with open(os.path.join(os.path.dirname(__file__), 'console.js'), encoding='utf-8') as f:
    CONSOLE_JS = f.read()

# ---------------------------------------------------------------------------
# Global State & Caching
# ---------------------------------------------------------------------------

server_manager: Optional[LlamaServerManager] = None
server_lock = threading.Lock()
pipeline_cancel_event = threading.Event()

# Cache batch results in memory to avoid sending huge image payloads via gr.State
BATCH_CACHE: Dict[str, Dict[str, Any]] = {}
BATCH_CACHE_LOCK = threading.Lock()

MODEL_PRESETS = [
    "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
    "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    "unsloth/Qwen2.5-VL-72B-Instruct-GGUF",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "custom",
]

# Extra rules on top of console.css: a couple of status-badge variants and
# a score badge that the base stylesheet doesn't define, since this app's
# domain (server lifecycle, detection score) doesn't exist in the API
# console the base CSS was extracted from. Kept as a small appended block
# rather than forking console.css, per "Asset loading" in patterns.md.
EXTRA_CSS = """
.status-badge { display:inline-block; padding:0.3rem 0.9rem; border-radius:20px;
    font-family:'JetBrains Mono',monospace; font-weight:600; font-size:0.7rem;
    text-transform:uppercase; letter-spacing:0.06em; }
.badge-running { background:rgba(74,222,128,0.12); color:#4ade80; border:1px solid rgba(74,222,128,0.3); }
.badge-stopped { background:rgba(125,133,144,0.12); color:#7d8590; border:1px solid rgba(125,133,144,0.3); }
.badge-starting { background:rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); }
.badge-error { background:rgba(248,113,113,0.12); color:#f87171; border:1px solid rgba(248,113,113,0.3); }

.score-badge { display:inline-block; padding:0.4rem 1.1rem; border-radius:8px;
    background:rgba(56,189,248,0.1); color:#38bdf8; border:1px solid rgba(56,189,248,0.3);
    font-family:'JetBrains Mono',monospace; font-weight:600; font-size:0.85rem; }
"""
custom_css = custom_css + EXTRA_CSS


class PipelineCancelledException(Exception):
    """Raised when a user cancels the pipeline mid-run."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zip_results_folder(folder_path: Path) -> Path:
    zip_path = folder_path.parent / f"batch_results_{int(time.time())}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob('*'):
            if file.is_file() and file.name != zip_path.name:
                zipf.write(file, file.relative_to(folder_path))
    return zip_path


def handle_preset_change(preset: str) -> gr.update:
    if preset == "custom":
        return gr.update(value="", visible=True)
    return gr.update(value=preset, visible=True)


# Reused for every output panel (Server Logs, Pipeline Logs). `raw_ta_id`
# must match the elem_id given to the hidden Textbox below it — see
# references/patterns.md "Output panel anatomy" / "Common pitfalls".
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


# ---------------------------------------------------------------------------
# Server Manager Wrappers
# ---------------------------------------------------------------------------

def start_server_wrapper(model, port, host, enable_thinking, enable_mtp,
                         ctx_size, gpu_layers, kv_cache_type):
    global server_manager

    with server_lock:
        if server_manager is not None and server_manager.is_healthy():
            yield "Server is already running and healthy.", \
                  f'<span class="status-badge badge-running">RUNNING (Port {server_manager.port})</span>'
            return

        yield "Stopping any existing server instance...", \
              '<span class="status-badge badge-starting">CLEANING UP...</span>'
        if server_manager is not None:
            try:
                server_manager.stop_llama_server()
            except Exception as e:
                print(f"Error stopping old server: {e}")

        yield "Configuring server...", \
              '<span class="status-badge badge-starting">INITIALIZING...</span>'

        spec_type = "draft-mtp" if enable_mtp else "none"
        server_manager = LlamaServerManager(
            model=model, host=host, port=int(port),
            ctx_size=int(ctx_size), parallel_slots=1, n_threads=-1,
            gpu_layers=int(gpu_layers), tensor_split="1,1", main_gpu=0,
            temp=0.4, top_p=0.95, top_k=64,
            spec_type=spec_type,
            spec_draft_n_max=4 if enable_mtp else 0,
            enable_thinking=enable_thinking,
            batch_size=1024, ubatch_size=512,
            kv_cache_type=kv_cache_type,
        )

        yield "Spawning llama-server process...", \
              '<span class="status-badge badge-starting">STARTING...</span>'
        try:
            server_manager.start_llama_server()
        except Exception as e:
            server_manager = None
            yield f"Failed to start server process: {e}", \
                  '<span class="status-badge badge-error">PROCESS ERROR</span>'
            return

    start_time = time.time()
    timeout = 180
    healthy = False

    while time.time() - start_time < timeout:
        with server_lock:
            if server_manager is None:
                yield "Server initialization aborted.", \
                      '<span class="status-badge badge-stopped">STOPPED</span>'
                return
            if server_manager.process and server_manager.process.poll() is not None:
                exit_code = server_manager.process.poll()
                logs = server_manager.get_logs()
                server_manager = None
                yield f"Server process exited with code {exit_code}.\n\n--- Logs ---\n{logs}", \
                      '<span class="status-badge badge-error">CRASHED</span>'
                return
            if server_manager.is_healthy():
                healthy = True
                break

            logs = server_manager.get_logs()
            elapsed = int(time.time() - start_time)
            yield f"Waiting for model to load into memory... ({elapsed}s elapsed)\n\n--- Latest Output ---\n{logs[-1200:]}", \
                  '<span class="status-badge badge-starting">STARTING...</span>'
        time.sleep(2)

    if healthy:
        yield "Server is up. Running warmup request...", \
              '<span class="status-badge badge-starting">WARMING UP...</span>'
        try:
            with server_lock:
                if server_manager:
                    server_manager.warmup_model()
            yield "Server started and warmed up. Ready for detection tasks.", \
                  f'<span class="status-badge badge-running">RUNNING (Port {port})</span>'
        except Exception as e:
            yield f"Server is healthy, but warmup failed: {e}", \
                  f'<span class="status-badge badge-running">RUNNING (Port {port})</span>'
    else:
        yield "Timed out waiting for the server to report healthy status.", \
              '<span class="status-badge badge-error">TIMEOUT</span>'


def stop_server_wrapper():
    global server_manager
    with server_lock:
        if server_manager is None:
            return "No server running.", \
                   '<span class="status-badge badge-stopped">STOPPED</span>'
        try:
            server_manager.stop_llama_server()
            server_manager = None
            return "Server stopped successfully.", \
                   '<span class="status-badge badge-stopped">STOPPED</span>'
        except Exception as e:
            return f"Error stopping server: {e}", \
                   '<span class="status-badge badge-error">STOP ERROR</span>'


def get_server_status_and_logs():
    global server_manager
    with server_lock:
        if server_manager is None:
            return "No server instance exists.", \
                   '<span class="status-badge badge-stopped">STOPPED</span>'
        if server_manager.process and server_manager.process.poll() is not None:
            exit_code = server_manager.process.poll()
            return f"Server process is dead (Exit code: {exit_code}).\n\n--- Logs ---\n{server_manager.get_logs()}", \
                   '<span class="status-badge badge-error">CRASHED</span>'
        logs = server_manager.get_logs()
        if server_manager.is_healthy():
            return f"Server is healthy and running.\n\n--- Logs ---\n{logs[-2000:]}", \
                   f'<span class="status-badge badge-running">RUNNING (Port {server_manager.port})</span>'
        return f"Server is starting or unhealthy.\n\n--- Logs ---\n{logs[-2000:]}", \
               '<span class="status-badge badge-starting">STARTING...</span>'


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

def run_batch_detection_gui(image_files, categories_str, category_definitions,
                            local_server_port, use_external_api,
                            ext_api_url, ext_api_key, ext_model_name,
                            max_rounds, score_threshold,
                            detector_temp, judge_temp,
                            customize_prompts, detector_template, judge_template):
    global BATCH_CACHE
    pipeline_cancel_event.clear()

    # --- Validation ---
    if not image_files:
        yield "Error: Please upload at least one image.", 0, None, "", gr.update(choices=[]), ""
        return

    categories = [c.strip() for c in categories_str.split(",") if c.strip()]
    if not categories:
        yield "Error: Please list at least one category.", 0, None, "", gr.update(choices=[]), ""
        return

    image_paths: list[Path] = []
    for f in image_files:
        if isinstance(f, str):
            image_paths.append(Path(f))
        elif hasattr(f, "name"):
            image_paths.append(Path(f.name))
        elif isinstance(f, dict) and "name" in f:
            image_paths.append(Path(f["name"]))
    if not image_paths:
        yield "Error: Could not resolve uploaded files.", 0, None, "", gr.update(choices=[]), ""
        return

    # --- Client setup ---
    yield "Initializing API clients...", 2, None, "", gr.update(choices=[]), ""

    if use_external_api:
        api_url, api_key, model_name = ext_api_url, ext_api_key, ext_model_name
    else:
        with server_lock:
            if server_manager is None or not server_manager.is_healthy():
                yield "Error: Local server not running. Start it on the Server tab or enable External API.", 2, None, "", gr.update(choices=[]), ""
                return
            port = server_manager.port
            model_name = server_manager.model
        api_url = f"http://localhost:{port}/v1"
        api_key = "not-needed"

    try:
        client = OpenAI(base_url=api_url, api_key=api_key)
    except Exception as e:
        yield f"Error initializing OpenAI client: {e}", 2, None, "", gr.update(choices=[]), ""
        return

    # --- Logging capture ---
    log_capture = io.StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    pipeline_logger = logging.getLogger("detection_pipeline")
    pipeline_logger.addHandler(log_handler)
    pipeline_logger.setLevel(logging.INFO)

    det_tmpl = detector_template if customize_prompts else DEFAULT_DETECTOR_TEMPLATE
    jdg_tmpl = judge_template if customize_prompts else DEFAULT_JUDGE_TEMPLATE

    batch_id = str(int(time.time()))
    run_dir = Path("./gui_runs") / f"run_{batch_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize memory cache for this batch
    with BATCH_CACHE_LOCK:
        BATCH_CACHE[batch_id] = {}

    batch_results = BATCH_CACHE[batch_id]
    q: queue.Queue = queue.Queue()

    def progress_callback(round_result: RoundResult, annotated_image: Image.Image):
        if pipeline_cancel_event.is_set():
            raise PipelineCancelledException("Pipeline cancelled by user.")
        q.put(("round", round_result, annotated_image))

    def worker():
        try:
            for idx, img_path in enumerate(image_paths):
                if pipeline_cancel_event.is_set():
                    q.put(("cancelled",))
                    return

                img_name = img_path.name
                img_stem = img_path.stem
                uniq_stem = img_stem
                counter = 1
                while uniq_stem in batch_results:
                    uniq_stem = f"{img_stem}_{counter}"
                    counter += 1

                q.put(("start_image", img_name, uniq_stem, idx + 1, len(image_paths)))

                image_out_dir = run_dir / uniq_stem
                image_out_dir.mkdir(parents=True, exist_ok=True)

                target_suffix = img_path.suffix or ".jpg"
                shutil.copy(img_path, image_out_dir / f"original{target_suffix}")
                base_image = Image.open(img_path).convert("RGB")

                batch_results[uniq_stem] = {
                    "grid_original": draw_grid(base_image),
                    "raw_original": base_image,
                    "best_annotated": None,
                    "detections": [],
                    "rounds": [],
                }

                pipeline = ObjectDetectionPipeline(
                    detector_client=client, judge_client=client,
                    detector_model=model_name, judge_model=model_name,
                    max_rounds=max_rounds, score_threshold=score_threshold,
                    detector_template=det_tmpl, judge_template=jdg_tmpl,
                    detector_max_tokens=4096, judge_max_tokens=1024,
                    api_retries=3,
                    detector_temperature=detector_temp, detector_top_p=0.95,
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

                batch_results[uniq_stem]["best_annotated"] = best.get("annotated")
                batch_results[uniq_stem]["detections"] = best.get("detections") or []
                q.put(("finish_image", uniq_stem))

            zip_path = zip_results_folder(run_dir)
            q.put(("done", str(zip_path)))

        except PipelineCancelledException:
            q.put(("cancelled",))
        except Exception as e:
            q.put(("error", str(e), traceback.format_exc()))
        finally:
            pipeline_logger.removeHandler(log_handler)

    threading.Thread(target=worker, daemon=True).start()

    yield "Starting batch processing...", 5, None, batch_id, gr.update(choices=[]), ""

    current_stem = ""
    current_img = ""
    current_idx = 1
    total_imgs = len(image_paths)

    while True:
        try:
            msg = q.get(timeout=0.2)
            tag = msg[0]

            if tag == "start_image":
                current_img, current_stem, current_idx, total_imgs = msg[1], msg[2], msg[3], msg[4]
                pct = int(((current_idx - 1) / total_imgs) * 90)
                yield f"Processing image {current_idx}/{total_imgs}: {current_img}...", \
                      pct, None, batch_id, \
                      gr.update(choices=list(batch_results.keys()), value=current_stem), \
                      log_capture.getvalue()

            elif tag == "round":
                r_res, r_img = msg[1], msg[2]
                if current_stem in batch_results:
                    batch_results[current_stem]["rounds"].append({
                        "round": r_res.round, "score": r_res.score,
                        "feedback": r_res.feedback, "raw_text": r_res.raw_detector_output,
                        "parse_error": r_res.parse_error, "image": r_img,
                        "detections": r_res.detections,
                    })
                pct = int(((current_idx - 1) / total_imgs) * 90
                          + (r_res.round / max_rounds) * (90 / total_imgs))
                yield f"Processing image {current_idx}/{total_imgs} ({current_img}) - Round {r_res.round} done (Score: {r_res.score}/10).", \
                      pct, None, batch_id, \
                      gr.update(choices=list(batch_results.keys()), value=current_stem), \
                      log_capture.getvalue()

            elif tag == "finish_image":
                finished_stem = msg[1]
                pct = int((current_idx / total_imgs) * 90)
                yield f"Finished image {current_idx}/{total_imgs}: {finished_stem}.", \
                      pct, None, batch_id, \
                      gr.update(choices=list(batch_results.keys()), value=finished_stem), \
                      log_capture.getvalue()

            elif tag == "done":
                zip_path = msg[1]
                yield "Batch processing completed! Zip archive ready.", \
                      100, zip_path, batch_id, \
                      gr.update(choices=list(batch_results.keys())), \
                      log_capture.getvalue()
                break

            elif tag == "cancelled":
                yield "Pipeline execution cancelled by the user.", \
                      100, None, batch_id, \
                      gr.update(choices=list(batch_results.keys())), \
                      log_capture.getvalue()
                break

            elif tag == "error":
                err_msg, trace = msg[1], msg[2]
                yield f"Pipeline execution failed:\n{err_msg}", \
                      100, None, batch_id, \
                      gr.update(choices=list(batch_results.keys())), \
                      log_capture.getvalue() + f"\n[CRITICAL ERROR] {err_msg}\n{trace}"
                break

        except queue.Empty:
            if not threading.active_count() > 1:  # Simple check if worker died
                break
            yield (f"Processing batch ({current_idx}/{total_imgs})...",
                   int(((current_idx - 1) / total_imgs) * 90),
                   None, batch_id,
                   gr.update(choices=list(batch_results.keys()),
                             value=current_stem if current_stem else None),
                   log_capture.getvalue())
            time.sleep(0.5)


def cancel_pipeline():
    pipeline_cancel_event.set()
    return "Cancellation requested. Waiting for current round to stop..."


# ---------------------------------------------------------------------------
# Explorer Callbacks
# ---------------------------------------------------------------------------

def on_explorer_image_change(selected_image, batch_id):
    with BATCH_CACHE_LOCK:
        batch_results = BATCH_CACHE.get(batch_id, {})

    if not batch_results or not selected_image or selected_image not in batch_results:
        return gr.update(choices=[], value=None)

    rounds = batch_results[selected_image].get("rounds", [])
    choices = ["Final Best"] + [str(r["round"]) for r in rounds]
    return gr.update(choices=choices, value="Final Best")


def on_explorer_round_change(selected_image, selected_round, batch_id, show_grid):
    with BATCH_CACHE_LOCK:
        batch_results = BATCH_CACHE.get(batch_id, {})

    if not batch_results or not selected_image or selected_image not in batch_results:
        return None, None, '<span class="score-badge">Score: -/10</span>', "", "", "", "[]"

    img_data = batch_results[selected_image]
    src_img = img_data["grid_original"] if show_grid else img_data["raw_original"]

    if not selected_round or selected_round == "Final Best":
        best_annotated = img_data["best_annotated"]
        best_score, best_round_num, best_feedback, best_raw, best_err = -1, -1, "No detections found.", "", ""
        for r in img_data["rounds"]:
            if r["score"] > best_score:
                best_score = r["score"]
                best_round_num = r["round"]
                best_feedback = r["feedback"]
                best_raw = r["raw_text"]
                best_err = r["parse_error"]

        score_text = f'<span class="score-badge">Best Score: {best_score}/10 (Round {best_round_num})</span>' if best_score >= 0 else '<span class="score-badge">Score: -/10</span>'
        return (src_img, best_annotated, score_text, best_feedback,
                best_raw, best_err or "None",
                json.dumps(img_data["detections"], indent=2))
    else:
        try:
            round_idx = int(selected_round) - 1
            rounds = img_data["rounds"]
            if 0 <= round_idx < len(rounds):
                r = rounds[round_idx]
                score_text = f'<span class="score-badge">Score: {r["score"]}/10</span>'
                return (src_img, r["image"], score_text,
                        r["feedback"], r["raw_text"], r["parse_error"] or "None",
                        json.dumps(r["detections"], indent=2))
        except Exception as e:
            print(f"Error loading round details: {e}")

    return src_img, None, '<span class="score-badge">Score: -/10</span>', "", "", "", "[]"


# ---------------------------------------------------------------------------
# UI Toggle Helpers
# ---------------------------------------------------------------------------

def toggle_run_btn(is_running):
    """Toggles button interactivity based on pipeline state."""
    return gr.update(interactive=not is_running), gr.update(interactive=is_running)


# ---------------------------------------------------------------------------
# Gradio Layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(theme=theme, css=custom_css, title="LLM Object Detection Console") as app:
        gr.HTML(CONSOLE_JS)

        # --- Header ---
        gr.HTML("""
        <div class="app-header" style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <h1><span>&#128269;</span> LLM Object Detection Console</h1>
                <p>// vision-LLM detector/judge pipeline over a local or external endpoint</p>
            </div>
        </div>""")
        server_status_badge = gr.HTML(
            value='<span class="status-badge badge-stopped">STOPPED</span>',
        )

        batch_id_state = gr.State("")

        with gr.Tabs():

            # ============ TAB 1: SERVER ============
            with gr.TabItem("\U0001F999 Llama Server"):
                gr.HTML('<p class="section-label">Model Server Configuration</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        server_preset = gr.Dropdown(
                            label="Recommended Model Presets",
                            choices=MODEL_PRESETS,
                            value="unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
                        )
                        server_model_input = gr.Textbox(
                            label="Model GGUF Path or HF Repo ID",
                            value="unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q2_K_XL",
                            placeholder="e.g. C:/models/qwen.gguf or HF ID",
                        )
                        server_preset.change(handle_preset_change, server_preset, server_model_input)

                        server_port_input = gr.Number(label="Port Number", value=8080, precision=0)
                        with gr.Row():
                            server_thinking_chk = gr.Checkbox(label="Thinking Mode", value=False)
                            server_mtp_chk = gr.Checkbox(label="MTP Speculative Drafting", value=True)

                        with gr.Accordion("Advanced Server Parameters", open=False):
                            server_host_input = gr.Textbox(label="Host Binding", value="0.0.0.0")
                            server_ctx_input = gr.Number(label="Context Size", value=20000, precision=0)
                            server_gpu_layers = gr.Number(label="GPU Layers (-ngl)", value=-1, precision=0)
                            server_kv_cache = gr.Dropdown(
                                label="KV Cache Type",
                                choices=["q4_0", "q8_0", "f16"],
                                value="q4_0",
                            )

                        with gr.Row():
                            start_server_btn = gr.Button("\u25b6  Start Server", variant="primary")
                            stop_server_btn = gr.Button("\u23f9  Stop Server", variant="secondary", size="sm")

                    with gr.Column(scale=3):
                        gr.HTML('<p class="section-label">Server Output Console</p>')
                        gr.HTML('<div class="output-panel" id="server-log-panel">'
                                + panel_header('Live Logs', 'server-log-ta'))
                        with gr.Group(elem_classes=['out-md-wrap']):
                            server_logs_viewer = gr.Textbox(
                                lines=20, max_lines=30,
                                interactive=False,
                                show_label=False,
                                container=False,
                                elem_id="server-log-ta",
                            )
                        gr.HTML('</div>')

                start_server_btn.click(
                    start_server_wrapper,
                    inputs=[server_model_input, server_port_input, server_host_input,
                            server_thinking_chk, server_mtp_chk,
                            server_ctx_input, server_gpu_layers, server_kv_cache],
                    outputs=[server_logs_viewer, server_status_badge],
                )
                stop_server_btn.click(
                    stop_server_wrapper,
                    outputs=[server_logs_viewer, server_status_badge],
                )
                app.load(get_server_status_and_logs,
                         outputs=[server_logs_viewer, server_status_badge])

            # ============ TAB 2: BATCH SANDBOX ============
            with gr.TabItem("\U0001F9EA Batch Sandbox"):
                with gr.Row(equal_height=False):
                    # Left column — config
                    with gr.Column(scale=2, min_width=400):
                        gr.HTML('<p class="section-label">Configuration</p>')

                        input_images = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Upload Source Image(s)",
                        )
                        categories_input = gr.Textbox(
                            label="Target Categories (comma-separated)",
                            placeholder="person, car, dog",
                            value="person, car, bicycle, dog, cat",
                        )
                        category_defs_input = gr.Textbox(
                            label="Category Definitions",
                            placeholder="Write instructions for categories...",
                            lines=4,
                            value=("- person: a human being\n"
                                   "- car: a 4-wheeled motor vehicle\n"
                                   "- bicycle: a 2-wheeled human-powered vehicle\n"
                                   "- dog: a domestic canine\n"
                                   "- cat: a domestic feline"),
                        )

                        with gr.Accordion("Pipeline Parameters", open=False):
                            rounds_slider = gr.Slider(label="Max Rounds",
                                                      minimum=1, maximum=5, step=1, value=2)
                            score_threshold_slider = gr.Slider(
                                label="Stop Score Threshold (0-10)",
                                minimum=0, maximum=10, step=1, value=8)
                            det_temp_slider = gr.Slider(
                                label="Detector Temperature",
                                minimum=0.0, maximum=1.5, step=0.05, value=0.9)
                            jdg_temp_slider = gr.Slider(
                                label="Judge Temperature",
                                minimum=0.0, maximum=1.5, step=0.05, value=0.2)

                        with gr.Accordion("External API (Optional)", open=False):
                            use_external_api_chk = gr.Checkbox(
                                label="Use External API instead of Local Server",
                                value=False)
                            ext_api_url = gr.Textbox(label="Base URL", value="https://api.openai.com/v1")
                            ext_api_key = gr.Textbox(label="API Key", value="your-key", type="password")
                            ext_model_name = gr.Textbox(label="Model Name", value="gpt-4o")

                        with gr.Row():
                            run_btn = gr.Button("\u25b6  Run Batch Pipeline", variant="primary", interactive=True)
                            stop_run_btn = gr.Button("\u23f9  Cancel", variant="secondary", size="sm", interactive=False)

                    # Right column — results
                    with gr.Column(scale=3, min_width=600):
                        gr.HTML('<p class="section-label">Results</p>')

                        with gr.Group():
                            pipeline_status = gr.Markdown("**Status: Idle**")
                            progress_slider = gr.Slider(
                                label="Execution Progress",
                                minimum=0, maximum=100, step=1, value=0,
                                interactive=False,
                            )
                        download_results_box = gr.File(
                            label="\U0001F4E5 Download Processed Results (.zip)",
                            interactive=False,
                        )

                        with gr.Tabs():
                            with gr.TabItem("\U0001F5BC\uFE0F Batch Explorer"):
                                with gr.Row():
                                    explorer_image_select = gr.Dropdown(
                                        label="Select Image", choices=[], interactive=True, scale=2)
                                    explorer_round_select = gr.Dropdown(
                                        label="Select Round", choices=[], interactive=True, scale=2)
                                    round_score_display = gr.HTML(
                                        value='<span class="score-badge">Score: -/10</span>',
                                        elem_classes="score-display",
                                        scale=1
                                    )

                                with gr.Row(equal_height=True):
                                    with gr.Column(scale=1):
                                        show_grid_chk = gr.Checkbox(
                                            label="Show 0-1000 grid overlay", value=True)
                                        source_image_viewer = gr.Image(label="Source Image", type="pil")
                                    with gr.Column(scale=1):
                                        best_annotated_viewer = gr.Image(label="Annotated Image", type="pil")

                                round_feedback_display = gr.Textbox(
                                    label="Judge's Feedback", lines=4, interactive=False)

                                with gr.Accordion("Raw Response Details", open=False):
                                    round_parse_error_display = gr.Textbox(
                                        label="Parsing Errors", interactive=False)
                                    round_raw_response_display = gr.Textbox(
                                        label="Raw Detector Text Response",
                                        lines=6, interactive=False)

                            with gr.TabItem("\U0001F4C4 Detections JSON"):
                                gr.HTML('<div class="json-panel">'
                                        '<div class="json-panel-hdr"><span class="dot-amber"></span>'
                                        'Detections (JSON List)</div>')
                                with gr.Group(elem_classes=['json-panel-body']):
                                    detections_json_box = gr.Code(
                                        language="json",
                                        show_label=False,
                                    )
                                gr.HTML('</div>')

                            with gr.TabItem("\U0001F4CB Pipeline Logs"):
                                gr.HTML('<div class="output-panel" id="pipeline-log-panel">'
                                        + panel_header('Execution Logs', 'pipeline-log-ta'))
                                with gr.Group(elem_classes=['out-md-wrap']):
                                    pipeline_logs_viewer = gr.Textbox(
                                        lines=20, max_lines=30,
                                        interactive=False,
                                        show_label=False,
                                        container=False,
                                        elem_id="pipeline-log-ta",
                                    )
                                gr.HTML('</div>')

            # ============ TAB 3: PROMPTS ============
            with gr.TabItem("\u270D\uFE0F Prompts"):
                gr.HTML('<p class="section-label">Prompt Engineering</p>')
                gr.Markdown(
                    "Modify the custom instruction templates fed to the Detector and Judge agents."
                )
                customize_prompts_chk = gr.Checkbox(
                    label="Enable Custom Prompt Templates", value=False)

                with gr.Group(visible=False) as prompts_group:
                    custom_det_prompt = gr.Textbox(
                        label="Detector Prompt Template",
                        lines=12, value=DEFAULT_DETECTOR_TEMPLATE,
                    )
                    custom_jdg_prompt = gr.Textbox(
                        label="Judge Prompt Template",
                        lines=12, value=DEFAULT_JUDGE_TEMPLATE,
                    )

                customize_prompts_chk.change(
                    lambda v: gr.update(visible=v),
                    customize_prompts_chk, prompts_group,
                )

        # -------------------------------------------------------------------
        # Event Bindings (Moved to bottom to ensure all variables are in scope)
        # -------------------------------------------------------------------

        # 1. Disable Run button, Enable Stop button immediately on click
        run_btn.click(
            fn=lambda: toggle_run_btn(is_running=True),
            inputs=None,
            outputs=[run_btn, stop_run_btn],
            queue=False
        ).then(
            # 2. Execute the actual pipeline
            fn=run_batch_detection_gui,
            inputs=[
                input_images, categories_input, category_defs_input,
                server_port_input,
                use_external_api_chk, ext_api_url, ext_api_key, ext_model_name,
                rounds_slider, score_threshold_slider,
                det_temp_slider, jdg_temp_slider,
                customize_prompts_chk, custom_det_prompt, custom_jdg_prompt,
            ],
            outputs=[
                pipeline_status, progress_slider,
                download_results_box, batch_id_state,
                explorer_image_select, pipeline_logs_viewer,
            ],
            concurrency_limit=1  # Prevent overlapping batch runs
        ).then(
            # 3. Re-enable Run, Disable Stop when pipeline finishes/errors
            fn=lambda: toggle_run_btn(is_running=False),
            inputs=None,
            outputs=[run_btn, stop_run_btn],
            queue=False
        )

        # Cancel button
        stop_run_btn.click(
            fn=cancel_pipeline,
            outputs=[pipeline_status],
            queue=False
        )

        # Explorer Events
        explorer_image_select.change(
            on_explorer_image_change,
            inputs=[explorer_image_select, batch_id_state],
            outputs=[explorer_round_select],
        )

        explorer_round_select.change(
            on_explorer_round_change,
            inputs=[explorer_image_select, explorer_round_select,
                    batch_id_state, show_grid_chk],
            outputs=[source_image_viewer, best_annotated_viewer,
                     round_score_display, round_feedback_display,
                     round_raw_response_display, round_parse_error_display,
                     detections_json_box],
        )

        show_grid_chk.change(
            on_explorer_round_change,
            inputs=[explorer_image_select, explorer_round_select,
                    batch_id_state, show_grid_chk],
            outputs=[source_image_viewer, best_annotated_viewer,
                     round_score_display, round_feedback_display,
                     round_raw_response_display, round_parse_error_display,
                     detections_json_box],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
