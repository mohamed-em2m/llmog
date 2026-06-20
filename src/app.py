"""
Gradio front-end for the LLM object detection test console.

Three tabs:
  1. Llama Server — start/stop one or more `llama-server` instances, switch
     which one is "active", inspect logs.
  2. Detection Settings — fully customizable categories, definitions, prompt
     templates, per-role model/server selection, sampling params, rounds.
  3. Batch Test — upload multiple images, run them sequentially (one after
     another) through the pipeline with live per-round progress, a gallery
     of annotated results, and per-image outputs saved under a folder named
     after each image (best_annotated.jpg / best_detections.json /
     history.json), plus a single zip download of everything.

Run with: python app.py
"""

from __future__ import annotations
import os
os.environ['MPLBACKEND'] = 'Agg'

import json
import logging
import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

import gradio as gr
from openai import OpenAI

from detection_pipeline import (
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
    ObjectDetectionPipeline,
)
from llama_server_manager import LlamaServerManager

logger = logging.getLogger("detection_app")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

manager = LlamaServerManager(log_dir="./llama_logs")

DEFAULT_CATEGORIES = "person, car, bicycle, dog, cat"
DEFAULT_DEFINITIONS = """\
- person: a human being
- car: a 4-wheeled motor vehicle
- bicycle: a 2-wheeled human-powered vehicle
- dog: a domestic canine
- cat: a domestic feline
"""


# ---------------------------------------------------------------------------
# Llama Server tab — callbacks
# ---------------------------------------------------------------------------

def _ports_as_strings() -> list[str]:
    return [str(p) for p in manager.ports()]


def refresh_servers():
    rows = manager.list_rows()
    choices = _ports_as_strings()
    active = str(manager.active_port) if manager.active_port is not None else None
    active_dd = gr.update(choices=choices, value=active)
    generic_dd = gr.update(choices=choices, value=active)
    return rows, active_dd, generic_dd, generic_dd


def start_server_cb(
    model_path, llama_bin, host, port, ctx_size, parallel_slots, n_threads, gpu_layers,
    tensor_split, main_gpu, temp, top_p, top_k, spec_type, spec_draft_n_max, fa,
    enable_thinking, batch_size_str, ubatch_size_str, kv_cache_type_str, extra_args,
    startup_timeout,
):
    try:
        if not model_path or not model_path.strip():
            raise ValueError("Model path is required.")
        kwargs = dict(
            model=model_path.strip(),
            host=host.strip() or "0.0.0.0",
            port=int(port),
            ctx_size=int(ctx_size),
            parallel_slots=int(parallel_slots),
            n_threads=int(n_threads),
            gpu_layers=int(gpu_layers),
            tensor_split=tensor_split.strip() or "1,1",
            main_gpu=int(main_gpu),
            temp=float(temp),
            top_p=float(top_p),
            top_k=int(top_k),
            spec_type=spec_type.strip(),
            spec_draft_n_max=int(spec_draft_n_max),
            fa=fa,
            enable_thinking=bool(enable_thinking),
            batch_size=int(batch_size_str) if str(batch_size_str).strip() else None,
            ubatch_size=int(ubatch_size_str) if str(ubatch_size_str).strip() else None,
            kv_cache_type=(kv_cache_type_str or "").strip() or None,
            llama_server_bin=llama_bin.strip() or "llama-server",
            extra_args=extra_args or "",
            startup_timeout=float(startup_timeout),
        )
        info = manager.start(**kwargs)
        msg = f"✅ Started on port {info.port} (pid {info.process.pid})\nModel: {info.model}\nBase URL: {info.base_url}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start server")
        msg = f"❌ Failed to start server: {exc}"
    rows, active_dd, dd1, dd2 = refresh_servers()
    return msg, rows, active_dd, dd1, dd2


def stop_server_cb(port_str):
    if not port_str:
        msg = "No server selected."
    else:
        ok = manager.stop(int(port_str))
        msg = f"🛑 Stopped port {port_str}" if ok else f"⚠️ Could not stop port {port_str} (already gone?)"
    rows, active_dd, dd1, dd2 = refresh_servers()
    return msg, rows, active_dd, dd1, dd2


def stop_all_cb():
    manager.stop_all()
    rows, active_dd, dd1, dd2 = refresh_servers()
    return "🛑 Stopped all servers.", rows, active_dd, dd1, dd2


def set_active_cb(port_str):
    if not port_str:
        msg = "No server selected."
    else:
        try:
            manager.set_active(int(port_str))
            msg = f"✅ Active server set to port {port_str}"
        except ValueError as exc:
            msg = f"❌ {exc}"
    rows, active_dd, dd1, dd2 = refresh_servers()
    return msg, rows, active_dd, dd1, dd2


def view_log_cb(port_str):
    if not port_str:
        return ""
    return manager.log_tail(int(port_str)) or "(log empty)"


def refresh_cb():
    return refresh_servers()


# ---------------------------------------------------------------------------
# Batch Test tab — pipeline execution with live per-round streaming
# ---------------------------------------------------------------------------

def _parse_categories(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,\n]", text or "") if c.strip()]


def _make_client(port: Optional[str]) -> Optional[OpenAI]:
    if not port:
        return None
    url = manager.base_url(int(port))
    if not url:
        return None
    return OpenAI(api_key="not-needed", base_url=url)


def _safe_stem(file_path: str, used: set) -> str:
    stem = Path(file_path).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem) or "image"
    candidate, i = stem, 1
    while candidate in used:
        i += 1
        candidate = f"{stem}_{i}"
    used.add(candidate)
    return candidate


def _stream_one_image(pipeline: ObjectDetectionPipeline, image_path: str, categories: list[str],
                       definitions: str, output_dir: str):
    """
    Runs pipeline.run() in a background thread and yields ('round', round_result,
    annotated_image), then a final ('done', best, history) or ('error', exc, None),
    so the caller can stream live per-round updates into the UI.
    """
    q: "queue.Queue" = queue.Queue()

    def on_round(round_result, annotated_img):
        q.put(("round", round_result, annotated_img))

    def target():
        try:
            best, history = pipeline.run(
                image_path=image_path,
                categories=categories,
                category_definitions=definitions,
                show_plot=False,
                output_dir=output_dir,
                progress_callback=on_round,
            )
            q.put(("done", best, history))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", exc, None))

    threading.Thread(target=target, daemon=True).start()

    while True:
        kind, a, b = q.get()
        yield kind, a, b
        if kind in ("done", "error"):
            break


def run_batch(
    files,
    categories_text, definitions_text, detector_template, judge_template,
    detector_port, judge_port, detector_model_name, judge_model_name,
    max_rounds, score_threshold, det_temp, det_top_p, judge_temp,
    det_max_tokens, judge_max_tokens, api_retries,
    output_base_dir,
    progress=gr.Progress(track_tqdm=False),
):
    gallery_items: list = []
    summary_rows: list = []
    log_lines: list[str] = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines[-300:])

    if not files:
        yield gallery_items, summary_rows, log("No images uploaded."), None
        return

    categories = _parse_categories(categories_text)
    if not categories:
        yield gallery_items, summary_rows, log("Please provide at least one category."), None
        return

    if not detector_port or not judge_port:
        yield gallery_items, summary_rows, log(
            "Select a detector server and a judge server in the Detection Settings tab "
            "(start one in the Llama Server tab first)."
        ), None
        return

    detector_client = _make_client(detector_port)
    judge_client = _make_client(judge_port)
    if detector_client is None or judge_client is None:
        yield gallery_items, summary_rows, log(
            "The selected detector/judge server isn't running anymore. Check the Llama Server tab."
        ), None
        return

    try:
        pipeline = ObjectDetectionPipeline(
            detector_client=detector_client,
            judge_client=judge_client,
            detector_model=detector_model_name.strip() or "local-model",
            judge_model=judge_model_name.strip() or "local-model",
            max_rounds=int(max_rounds),
            score_threshold=int(score_threshold),
            detector_template=detector_template or DEFAULT_DETECTOR_TEMPLATE,
            judge_template=judge_template or DEFAULT_JUDGE_TEMPLATE,
            detector_max_tokens=int(det_max_tokens),
            judge_max_tokens=int(judge_max_tokens),
            api_retries=int(api_retries),
            detector_temperature=float(det_temp),
            detector_top_p=float(det_top_p),
            judge_temperature=float(judge_temp),
        )
    except Exception as exc:  # noqa: BLE001
        yield gallery_items, summary_rows, log(f"Failed to configure pipeline: {exc}"), None
        return

    out_base = Path(output_base_dir or "./test_results")
    out_base.mkdir(parents=True, exist_ok=True)
    used_stems: set = set()

    file_paths = [f.name if hasattr(f, "name") else f for f in files]
    total = len(file_paths)

    for idx, file_path in enumerate(file_paths, start=1):
        original_name = Path(file_path).name
        stem = _safe_stem(file_path, used_stems)
        image_out_dir = out_base / stem

        progress((idx - 1) / total, desc=f"[{idx}/{total}] {original_name}")
        log(f"=== [{idx}/{total}] {original_name} -> {image_out_dir} ===")
        yield gallery_items, summary_rows, log_lines and "\n".join(log_lines[-300:]), None

        final_status, final_score, final_round, n_det = "running", None, None, 0

        for kind, a, b in _stream_one_image(pipeline, file_path, categories, definitions_text, str(image_out_dir)):
            if kind == "round":
                round_result, annotated_img = a, b
                msg = f"  round {round_result.round}: score {round_result.score}/10, {len(round_result.detections)} detections"
                if round_result.parse_error:
                    msg += f"  (parse error — see history.json)"
                log(msg)
                yield gallery_items, summary_rows, "\n".join(log_lines[-300:]), None

            elif kind == "done":
                best, history = a, b
                final_status = "ok"
                final_score = best["score"]
                final_round = best["round"]
                n_det = len(best["detections"] or [])
                if best["annotated"] is not None:
                    gallery_items.append(
                        (best["annotated"], f"{original_name} — score {final_score}/10 (round {final_round})")
                    )
                log(f"  done: best score {final_score}/10 at round {final_round}")

            elif kind == "error":
                exc = a
                final_status = f"error: {exc}"
                log(f"  ERROR: {exc}")

        summary_rows.append([original_name, final_status, final_round, final_score, n_det, str(image_out_dir)])
        yield gallery_items, summary_rows, "\n".join(log_lines[-300:]), None

    summary_path = out_base / "summary.json"
    try:
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        zip_base = str(out_base) + "_results"
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(out_base))
    except Exception as exc:  # noqa: BLE001
        log(f"Could not create results zip: {exc}")
        zip_path = None

    progress(1.0, desc="Done")
    log(f"=== All {total} image(s) processed. Results saved under {out_base.resolve()} ===")
    yield gallery_items, summary_rows, "\n".join(log_lines[-300:]), zip_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="LLM Object Detection Tester") as demo:
        gr.Markdown("# 🔍 LLM Object Detection — Test Console")

        # ------------------------------------------------------------- Tab 1
        with gr.Tab("🦙 Llama Server"):
            gr.Markdown(
                "Start one or more `llama-server` instances on different ports, "
                "switch which one is **active**, and stop them individually. "
                "Note: some flag names (spec/draft, thinking) vary across llama.cpp "
                "builds — use **Extra CLI args** to override/add flags if your build differs."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    model_path = gr.Textbox(label="Model path (.gguf)", placeholder="/path/to/model.gguf")
                    llama_bin = gr.Textbox(label="llama-server binary", value="llama-server")
                    with gr.Row():
                        host = gr.Textbox(label="Host", value="0.0.0.0")
                        port = gr.Number(label="Port", value=8080, precision=0)
                    with gr.Row():
                        ctx_size = gr.Number(label="ctx_size", value=20000, precision=0)
                        parallel_slots = gr.Number(label="parallel_slots", value=1, precision=0)
                    with gr.Row():
                        n_threads = gr.Number(label="n_threads (-1=auto)", value=-1, precision=0)
                        gpu_layers = gr.Number(label="gpu_layers (-1=all)", value=-1, precision=0)
                    with gr.Row():
                        tensor_split = gr.Textbox(label="tensor_split", value="1,1")
                        main_gpu = gr.Number(label="main_gpu", value=0, precision=0)
                    with gr.Row():
                        temp = gr.Slider(0, 2, value=0.4, step=0.05, label="temp")
                        top_p = gr.Slider(0, 1, value=0.95, step=0.01, label="top_p")
                        top_k = gr.Number(label="top_k", value=64, precision=0)
                    with gr.Row():
                        spec_type = gr.Textbox(label="spec_type", value="draft-mtp")
                        spec_draft_n_max = gr.Number(label="spec_draft_n_max", value=4, precision=0)
                        fa = gr.Dropdown(["auto", "on", "off"], value="auto", label="flash attention (fa)")
                    enable_thinking = gr.Checkbox(label="enable_thinking", value=False)
                    with gr.Row():
                        batch_size = gr.Textbox(label="batch_size (blank = default)", value="")
                        ubatch_size = gr.Textbox(label="ubatch_size (blank = default)", value="")
                        kv_cache_type = gr.Textbox(label="kv_cache_type (blank = default)", value="")
                    extra_args = gr.Textbox(label="Extra / override CLI args", placeholder="--flag value --other-flag")
                    startup_timeout = gr.Number(label="startup_timeout (s)", value=120)
                    start_btn = gr.Button("🚀 Start server", variant="primary")

                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", lines=4, interactive=False)
                    servers_df = gr.Dataframe(
                        headers=["port", "model", "status", "active", "started"],
                        label="Running servers",
                        interactive=False,
                    )
                    refresh_btn = gr.Button("🔄 Refresh")
                    with gr.Row():
                        active_dropdown = gr.Dropdown(label="Set active server (port)", choices=[])
                        set_active_btn = gr.Button("✅ Set active")
                    with gr.Row():
                        stop_dropdown = gr.Dropdown(label="Stop server (port)", choices=[])
                        stop_btn = gr.Button("🛑 Stop")
                        stop_all_btn = gr.Button("🛑 Stop all")
                    with gr.Row():
                        log_dropdown = gr.Dropdown(label="View log for port", choices=[])
                        view_log_btn = gr.Button("📜 View log")
                    log_box = gr.Textbox(label="Server log (tail)", lines=10, interactive=False)

        # ------------------------------------------------------------- Tab 2
        with gr.Tab("⚙️ Detection Settings"):
            gr.Markdown("Everything here is editable — categories, definitions, both prompt templates, "
                        "which server handles detection vs. judging, sampling params, and round budget.")
            with gr.Row():
                categories_box = gr.Textbox(
                    label="Categories (comma or newline separated)", value=DEFAULT_CATEGORIES, lines=2
                )
            definitions_box = gr.Textbox(
                label="Category definitions", value=DEFAULT_DEFINITIONS, lines=6
            )

            with gr.Row():
                with gr.Column():
                    detector_template_box = gr.Textbox(
                        label="Detector prompt template", value=DEFAULT_DETECTOR_TEMPLATE, lines=22
                    )
                    reset_detector_btn = gr.Button("↩️ Reset detector template to default")
                with gr.Column():
                    judge_template_box = gr.Textbox(
                        label="Judge prompt template", value=DEFAULT_JUDGE_TEMPLATE, lines=22
                    )
                    reset_judge_btn = gr.Button("↩️ Reset judge template to default")

            gr.Markdown("### Server / model assignment")
            with gr.Row():
                detector_port_dd = gr.Dropdown(label="Detector server (port)", choices=[])
                judge_port_dd = gr.Dropdown(label="Judge server (port)", choices=[])
            with gr.Row():
                detector_model_name = gr.Textbox(label="Detector model name (sent in API request)", value="local-model")
                judge_model_name = gr.Textbox(label="Judge model name (sent in API request)", value="local-model")

            gr.Markdown("### Sampling & loop control")
            with gr.Row():
                max_rounds = gr.Slider(1, 10, value=2, step=1, label="max_rounds")
                score_threshold = gr.Slider(0, 10, value=8, step=1, label="score_threshold")
            with gr.Row():
                det_temp = gr.Slider(0, 2, value=0.9, step=0.05, label="detector temperature")
                det_top_p = gr.Slider(0, 1, value=0.95, step=0.01, label="detector top_p")
                judge_temp = gr.Slider(0, 2, value=0.2, step=0.05, label="judge temperature")
            with gr.Row():
                det_max_tokens = gr.Number(label="detector max_tokens", value=4096, precision=0)
                judge_max_tokens = gr.Number(label="judge max_tokens", value=1024, precision=0)
                api_retries = gr.Number(label="api_retries", value=3, precision=0)

        # ------------------------------------------------------------- Tab 3
        with gr.Tab("🧪 Batch Test"):
            gr.Markdown("Upload several images — they're processed **one after another**, with live "
                        "per-round progress. Each image's best result is saved to its own folder named "
                        "after the image.")
            files_box = gr.Files(label="Upload images", file_types=["image"])
            output_base_dir = gr.Textbox(label="Output base directory", value="./test_results")
            run_btn = gr.Button("▶️ Run batch test", variant="primary")

            with gr.Row():
                gallery = gr.Gallery(label="Annotated results (best round per image)", columns=3, height=500)
            summary_df = gr.Dataframe(
                headers=["image", "status", "round", "score", "#detections", "output_dir"],
                label="Per-image summary",
                interactive=False,
            )
            log_box2 = gr.Textbox(label="Live log", lines=14, interactive=False, autoscroll=True)
            download_zip = gr.File(label="Download all results (.zip)")

        # ----------------------------------------------------------- Wiring
        server_outputs = [status_box, servers_df, active_dropdown, detector_port_dd, judge_port_dd]
        start_btn.click(
            start_server_cb,
            inputs=[
                model_path, llama_bin, host, port, ctx_size, parallel_slots, n_threads, gpu_layers,
                tensor_split, main_gpu, temp, top_p, top_k, spec_type, spec_draft_n_max, fa,
                enable_thinking, batch_size, ubatch_size, kv_cache_type, extra_args, startup_timeout,
            ],
            outputs=server_outputs,
        )
        stop_btn.click(stop_server_cb, inputs=[stop_dropdown], outputs=server_outputs)
        stop_all_btn.click(stop_all_cb, outputs=server_outputs)
        set_active_btn.click(set_active_cb, inputs=[active_dropdown], outputs=server_outputs)
        refresh_btn.click(refresh_cb, outputs=server_outputs)
        view_log_btn.click(view_log_cb, inputs=[log_dropdown], outputs=[log_box])

        # Keep the "stop" and "log" dropdowns in sync with the same refreshes too
        for trigger in (start_btn, stop_btn, stop_all_btn, set_active_btn, refresh_btn):
            trigger.click(lambda: gr.update(choices=_ports_as_strings()), outputs=[stop_dropdown])
            trigger.click(lambda: gr.update(choices=_ports_as_strings()), outputs=[log_dropdown])

        reset_detector_btn.click(lambda: DEFAULT_DETECTOR_TEMPLATE, outputs=[detector_template_box])
        reset_judge_btn.click(lambda: DEFAULT_JUDGE_TEMPLATE, outputs=[judge_template_box])

        run_btn.click(
            run_batch,
            inputs=[
                files_box,
                categories_box, definitions_box, detector_template_box, judge_template_box,
                detector_port_dd, judge_port_dd, detector_model_name, judge_model_name,
                max_rounds, score_threshold, det_temp, det_top_p, judge_temp,
                det_max_tokens, judge_max_tokens, api_retries,
                output_base_dir,
            ],
            outputs=[gallery, summary_df, log_box2, download_zip],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch()
