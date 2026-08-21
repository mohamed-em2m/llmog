"""
Batch Sandbox execution engine & pipeline runner.
Manages threading, progress queue, logging, and YOLO label persistence.
"""

from __future__ import annotations

import io
import time
import json
import queue
import shutil
import logging
import traceback
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import httpx
from PIL import Image
from openai import OpenAI

from free_detection.detection_pipeline import (
    ObjectDetectionPipeline,
    RoundResult,
    draw_grid,
    DEFAULT_DETECTOR_TEMPLATE,
    DEFAULT_JUDGE_TEMPLATE,
)
from interface.state import (
    state,
    DEFAULT_CONCURRENCY,
    _cache_put,
    zip_results_folder,
    _render_progress_bar,
    _tail,
)
from interface.batch.helpers import render_status_table, detections_to_yolo

logger = logging.getLogger("detection_pipeline")

TASK_FREE_ANNOTATION = "Free Annotation (detections)"
TASK_RECLASSIFICATION = "Reclassification (draw & relabel)"
TASK_CHOICES = [TASK_FREE_ANNOTATION, TASK_RECLASSIFICATION]


class PipelineCancelledException(Exception):
    """Raised when a user cancels the pipeline mid-run."""
    pass


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
    write_yolo_labels: bool = False,
    category_strategy: str = "strict",
):
    state.pipeline_cancel_event.clear()

    _empty_yield = (
        None,
        "",
        gr.update(choices=[]),
        "",
        render_status_table({}, []),
    )

    if not image_files:
        yield "Error: Please upload at least one image.", _render_progress_bar(
            0
        ), *_empty_yield
        return

    mode_norm = (category_strategy or "strict").lower().strip()
    categories = [c.strip() for c in (categories_str or "").split(",") if c.strip()]
    if not categories:
        if "free" in mode_norm:
            categories = ["*"]
        else:
            yield f"Error: Please list at least one category for {category_strategy.capitalize()} mode (or choose Free mode).", _render_progress_bar(
                0
            ), *_empty_yield
            return

    # Normalize image_files into a list (prevent iterating character-by-character if a string is passed)
    raw_list: List[Any]
    if isinstance(image_files, (str, Path)):
        raw_list = [image_files]
    elif isinstance(image_files, (list, tuple, set)):
        raw_list = list(image_files)
    else:
        raw_list = [image_files]

    image_paths: List[Path] = []
    for f in raw_list:
        if f is None:
            continue
        p: Optional[Path] = None
        if isinstance(f, (str, Path)):
            p = Path(f)
        elif hasattr(f, "path") and f.path:
            p = Path(f.path)
        elif hasattr(f, "name") and f.name:
            p = Path(f.name)
        elif isinstance(f, dict):
            p_val = f.get("path") or f.get("name")
            if p_val:
                p = Path(p_val)

        if p is not None:
            if p.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
                for child in sorted(p.iterdir()):
                    if child.is_file() and child.suffix.lower() in valid_exts:
                        image_paths.append(child)
            else:
                image_paths.append(p)

    if not image_paths:
        yield "Error: Could not resolve uploaded files.", _render_progress_bar(
            0
        ), *_empty_yield
        return

    cleaned_paths: List[Path] = []
    for p in image_paths:
        if not p.exists():
            yield f"Error: Image file '{p}' not found.", _render_progress_bar(
                0
            ), *_empty_yield
            return
        if not p.is_file():
            yield f"Error: '{p}' is not a file.", _render_progress_bar(
                0
            ), *_empty_yield
            return
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
    ), None, "", gr.update(choices=[]), "", render_status_table({}, [])

    if use_external_api:
        api_url, api_key, model_name = ext_api_url, ext_api_key, ext_model_name
        if not api_key or api_key == "your-key":
            yield (
                "Error: External API selected but no API key provided. "
                "Set one in the External API section."
            ), _render_progress_bar(0, "Error"), None, "", gr.update(
                choices=[]
            ), "", render_status_table(
                {}, []
            )
            return
    else:
        with state.server_lock:
            if state.server_manager is None or not state.server_manager.is_healthy():
                yield "Error: Local server not running. Start it on the Server tab or enable External API.", _render_progress_bar(
                    0, "Error"
                ), None, "", gr.update(
                    choices=[]
                ), "", render_status_table(
                    {}, []
                )
                return
            port = state.server_manager.port
            model_name = state.server_manager.model
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
        ), None, "", gr.update(choices=[]), "", render_status_table({}, [])
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
        if state.pipeline_cancel_event.is_set():
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
                        step=prep_config.get("grid_step", 250),
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
                if state.pipeline_cancel_event.is_set():
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

            if write_yolo_labels:
                allow_dyn = (category_strategy != "strict")
                lines, unmapped = detections_to_yolo(detections, categories, allow_dynamic_classes=allow_dyn)
                labels_dir = run_dir / "labels"
                labels_dir.mkdir(parents=True, exist_ok=True)
                if lines:
                    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                if unmapped:
                    json.dump(
                        unmapped,
                        open(image_out_dir / f"{stem}_unmapped.json", "w"),
                        indent=2,
                    )
            q.put(("finish_image", stem))

        except PipelineCancelledException:
            q.put(("image_cancelled", stem))
        except Exception as e:
            with log_lock:
                batch_logger.error(f"[{stem}] {e}\n{traceback.format_exc()}")
            q.put(("image_error", stem, str(e)))

    def worker():
        try:
            if not state.pipeline_cancel_event.is_set():
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

            if state.pipeline_cancel_event.is_set():
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
        render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
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
                    render_status_table(image_status, stem_order),
                )
                last_yield_time = now

            time.sleep(0.1)

    batch_logger.removeHandler(log_handler)
    log_handler.close()


def cancel_pipeline():
    """Request graceful cancellation of running batch jobs."""
    state.pipeline_cancel_event.set()
    return (
        "Cancellation requested. In-flight images will finish their current round "
        "and write results; queued images will be skipped. "
        "The Run button will re-enable once the worker drains."
    )


def run_batch_dispatcher(
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
    write_yolo_labels: bool = False,
    category_strategy: str = "strict",
):
    """Run Batch detection pipeline across multiple images."""
    yield from run_batch_detection_gui(
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
        write_yolo_labels,
        category_strategy=category_strategy,
    )
