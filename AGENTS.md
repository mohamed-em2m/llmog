# AGENTS.md — LLM Object Detection Testing Console

## Project Overview
Interactive test console for assessing Vision-Language Models (VLMs) on object detection. Uses an iterative **Detector-Judge pipeline** (LangGraph) where a detector proposes bounding boxes, a judge critiques them, and the loop repeats with structured feedback. Gradio console provides **Batch Processing** (top Live Results Explorer + bottom Input/Configuration columns), **Draw & Recognize** (custom canvas + DetectionViewer), **Real-Time Detection** (webcam/video with tracker + same-window overlay) and **Real-Time Draw** (live capture + draw) tabs. All tabs share a unified **DetectionViewer** (`llmog/detection_viewer`) for client-side box rendering (WebP cache, no server PIL re-encode).

## Package Management
- **Tool**: `uv` (fast Python installer/resolver)
- **Python**: 3.12+ (see `.python-version`)
- **Install**: `uv sync` (after `scripts/install_llama_cpp.sh` for llama.cpp) – `pyproject.toml` now includes `detection_viewer` in `tool.setuptools.packages.find`

## Entry Points
| Command | Module | Description |
|---------|--------|-------------|
| `uv run llmog` | `main:main` | Unified CLI; dispatches by `--task` (`free_detection` / `auto_label`) |
| `uv run detection-cli` | `free_detection:main` | Shortcut for `llmog --task free_detection` (detector/judge loop on `--image` paths) |
| `uv run auto-annotation` | `auto_annotation:main` | Shortcut for `llmog --task auto_label` (batch YOLO relabeling from a `data.yaml`) |
| `uv run detection-gui` | `interface.gui:main` | Launch Gradio console (`app_builder.build_app()`) |

Single source of truth for CLI flags: `llmog/schemes/argument.py:PipelineConfig` (pydantic v2). `llmog/main.py:build_parser` mirrors every field onto `argparse`; `parse_args()` overlays optional `--config <yaml>` and constructs validated `PipelineConfig`.

## Key Directories
- `llmog/` — Package root (`tool.setuptools.package-dir = {"": "llmog"}`)
- `llmog/schemes/` — `PipelineConfig` + argparse mirror
- `llmog/main.py` — Unified CLI dispatcher (`--task free_detection | auto_label`)
- `llmog/free_detection/` — Detector/Judge pipeline package
- `llmog/free_detection/agent/` — LangGraph nodes (`preprocess`, `detector`, `crop_verify`, `judge`, `loop`, `finalize`), `pipeline.py`, `state.py`, `visuals.py`, `client_utils.py` (429-aware retry)
- `llmog/detection_viewer/` — New `DetectionViewer` (gr.HTML) – `__init__.py`, `static/template.html|style.css|script.js`, `py.typed` – client-side canvas, WebP cache with dedup (`_WEBP_URL_CACHE`)
- `llmog/auto_annotation/` — Batch YOLO relabeling
- `llmog/prompts/` — Markdown templates for detector/judge/realtime (`detector_agent.md`, `realtime_detector.md`, `auto_label_classifier.md`) loaded via DynaPrompt
- `llmog/servers/` — `LlamaServerManager`/`VllmServerManager` + `servers_factory`
- `llmog/interface/` — Gradio console
  - `app_builder.py` — Aggregator, builds 6 tabs, wires global endpoint
  - `viewer_utils.py` — Shared adapters (`pipeline_detections_to_annotations`, `region_results_to_annotations`, `realtime_boxes_to_annotations`, `build_prep_config`, `build_viewer_payload`) + palette/WebP helpers
  - `tab_server.py` — Unified **Model / Endpoint** tab (Local vs External API global state)
  - `tab_draw.py` — Draw & Recognize (custom HTML5 canvas `CustomCanvasController`, `gr.UploadButton` upload, `DetectionViewer` results)
  - `tab_realtime_interactive.py` — **Real-Time Draw** (live capture `CustomCanvasControllerRT` with RT-specific ids, `gr.Image` webcam → canvas)
  - `batch/` — `components.py` (Batch UI), `runner.py` (threaded batch, lazy grid, 1600px cache cap), `explorer.py` (lazy grid + viewer payload), `reclassification.py` (crop classify: sequential / `asyncio.gather` parallel / batched single-request)
  - `realtime/` — `ui.py` (stream + video + same-window overlay), `handlers.py` (motion-gate diff 64×64, frame hash dedup, `build_prep_config`), `utils.py` (downscale to 1280, JPEG q85), `state.py` (SessionDetector, `resolve_endpoint`, pipeline cache)
  - `console.css` / `console.js` / `console_theme.py` — Dark terminal theme, DetectionViewer dark overrides, draw-tab responsive
- `scripts/` — Linux install scripts

## Core Modules
- `free_detection/agent/pipeline.py` — `ObjectDetectionPipeline` (LangGraph `compile()`, `run()`, `run_inference()`, `judge_detections()`)
- `free_detection/agent/graph.py` — `build_detection_graph()`
- `free_detection/agent/client_utils.py` — `_call_with_retries()` (429-aware, parses `RetryInfo/retryDelay`, exponential backoff + jitter, global Gemini 4s interval, 5 retries)
- `free_detection/agent/nodes/detector.py` — `_run_tiled_detection()` caps `max_workers=1` for Gemini to respect 15 RPM
- `free_detection/agent/visuals.py` — `render_detections()`, `draw_grid()`, `pil_to_data_uri()`
- `free_detection/image_preprocessing.py` — CLAHE, gamma, denoise, unsharp, white balance, SoM, tiling, `draw_premium_grid()`
- `detection_viewer/__init__.py:DetectionViewer` — `gr.HTML` subclass, `html_template`/`css_template`/`js_on_load` from `static/`, `panel_title`/`list_height`/`score_threshold` props, `postprocess()` → JSON `{image, annotations}` with WebP cache dedup
- `interface/viewer_utils.py` — Bbox converters (0–1000→pixel), `build_prep_config()` single source for Batch/Realtime
- `interface/app_builder.py` — `build_app()` (6 tabs: Model/Endpoint, Draw, Batch, Prompts, Real-Time, Real-Time Draw), `_on_endpoint_mode_change()` global toggle, wires `toggle_run_btn`, `run_batch_dispatcher` (now via `c_srv` endpoint), explorer, `same_window` overlay
- `interface/tab_server.py:_build_server_tab()` — Radio `endpoint_mode` (Local/External), `local_server_group` + `ext_api_group` (global), hidden `use_external_api_chk` for compat
- `interface/tab_draw.py:build_draw_tab()` — Left custom canvas (`_CUSTOM_CANVAS_HTML/_JS`, `CustomCanvasController`), persistent `gr.UploadButton` (`draw-upload-btn`) → `_handle_draw_upload()` (handles `str|dict|list|file-like`, Windows lock retry, `clearAll()`+`loadImageFromDataUrl()` replace), right `DetectionViewer` (`draw-detection-viewer`) + `request_mode` Radio
- `interface/tab_realtime_interactive.py` — Live `gr.Image(sources=["webcam"])` capture via JS `canvas.toDataURL` → `CustomCanvasControllerRT` (RT ids `llmog-custom-canvas-app-rt`, `getRtInteractiveDrawData`), same `classify_regions_gui` with `request_mode`
- `interface/realtime/ui.py:_build_realtime_tab()` — `category_strategy` Radio + `category_preset_dropdown` (from `CATEGORY_PRESETS`), `same_window_chk` (default OFF, viewer primary), `rt_same_window_canvas` overlay (absolute over `rt_webcam_wrap`), `video_gallery_output` (Gallery) + `video_viewer` (DetectionViewer last frame)
- `interface/realtime/handlers.py:process_single_frame()` – returns `(viewer_payload, boxes_json, hud, session)` + `_frame_diff_percent()` 64×64 motion gate; `process_video_frames()` – caps sampled frames to 60, returns `(gallery_frames, last_payload, status)` with `draw_boxes_opencv` for Gallery
- `interface/batch/reclassification.py:classify_regions_gui()` – `async`, `request_mode` `sequential`/`parallel` (`asyncio.gather`+`to_thread`) / `batched` (`classify_regions_batched()` 1×N images), `_handle_draw_upload()`, `extract_regions()`, `crop_with_padding()`
- `interface/batch/runner.py:run_batch_detection_gui()` – uses `build_prep_config()`, lazy grid (`grid_original=None` until `explorer.py`), cache thumbnail ≤1600px, Gemini cap `concurrency→2` + 4s pacing
- `interface/state.py` — `AppState` (`server_manager`, `batch_cache` LRU 3), `_cache_put/get`, `toggle_custom_color_field()`, `zip_results_folder()`

## Common Commands
```bash
./scripts/install_llama_cpp.sh  # Linux only
uv sync
uv run detection-gui
uv run detection-gui --port 7861 --share
uv run llmog --task free_detection -i image.jpg -c "person, car, dog"
uv run llmog --task auto_label --train_image imgs/ --train_label lbls/ --yaml_path data.yaml --model local-model -o ./out
uv run detection-cli -i image.jpg -c "person, car, dog"
uv run auto-annotation --train_image imgs/ --train_label lbls/ --yaml_path data.yaml --model local-model -o ./out
uv run detection-cli -i img1.jpg -i img2.jpg -c "crack, scratch, dent" --prep_enabled --prep_contrast_method clahe --prep_tiling_enabled --prep_tile_size 512 -o ./results
uv run llmog --task free_detection --config pipeline.yaml -i img.jpg --max_rounds 3
uv run pytest -q
```

## Important Conventions
- **Prompt templates**: `llmog/prompts/*.md` via DynaPrompt, Jinja `{{ categories_list }}` etc., fallback hardcoded strings. `realtime_detector.md` is concise single-pass (4 steps) for low latency; `detector_agent.md` is full 6-step with `<analysis>`/`<answer>` JSON.
- **DetectionViewer**: `detection_viewer` is `gr.HTML` with `html_template`/`css_template`/`js_on_load` from `static/`; Python pre-interpolates `${panel_title}`/`${list_height}` to avoid `ReferenceError` on Gradio 6. `postprocess()` expects `(image, annotations)` or `(image, annotations, config)`.
- **Global endpoint**: `tab_server.py` Radio `endpoint_mode` is single source; `use_external_api_chk`/`ext_api_url`/`ext_api_key`/`ext_model_name` live in `c_srv` and are passed to Batch (`app_builder.py:167`), Draw (`tab_draw.py:1241`), Realtime (`realtime/ui.py:323`) and Real-Time Draw (`tab_realtime_interactive.py:266`). `interface/realtime/state.py:resolve_endpoint()` centralizes `base_url/api_key/model_name`.
- **Class Expectation Mode**: Strict/Hybrid/Free via `category_strategy` Radio – Batch (`batch/components.py:110`), Realtime (`realtime/ui.py:55`), Draw/Real-Time Draw (`tab_draw.py:1234`, `tab_realtime_interactive.py:116`) share `CATEGORY_PRESETS` and `on_*_strategy_change` helpers; free allows `["*"]` when categories empty.
- **Request modes (Draw)**: `recls_request_mode` Radio (`tab_draw.py:1288`) `sequential` (N reqs) / `parallel` (asyncio.gather) / `batched` (1×N images via `classify_regions_batched()`); handler is `async` and `js="(p,...,reqMode)=>[fresh,...,reqMode]"` preserves sliders.
- **Output structure**: Each image → subdir under `--output_folder` with `best_annotated.jpg`, `best_detections.json`, `history.json`; Batch zip via `state.zip_results_folder()`.
- **API compatibility**: OpenAI SDK, any OpenAI-compatible endpoint; `extra_body` `min_pixels`/`max_pixels` via `build_prep_config()`; external mode strips vLLM extras.
- **Config precedence**: `pydantic defaults < --config <yaml> < CLI flags` via `argparse.SUPPRESS`.
- **Flag style**: `--prep-tile-size` and `--prep_tile_size` equivalent.
- **Rate limiting**: `client_utils._call_with_retries()` handles 429 with `RetryInfo` delay + jitter, caps at 60s, global Gemini 4s interval; tiling `max_workers` capped to 1 for Gemini.
- **Starlette deprecation**: `gui.py`/`app_builder.py` filter `HTTP_422_UNPROCESSABLE_ENTITY` `StarletteDeprecationWarning` (Gradio 6 routes.py:1379).
- **Performance**: `viewer_utils.build_prep_config()` single source; lazy grid (`runner.py:333` stores `None` until `explorer.py:79`); 1600px cache cap; WebP dedup (`detection_viewer/__init__.py:34`); realtime downscale to 1280 (`realtime/utils.py:97` JPEG q85), `stream_every=0.12`, `motion_gate` 64×64 diff, video cap 60 frames.

## Development Notes
- Test suite: `pytest` 6 tests in `tests/test_detection_graph.py` (graph, json-repair, tiled, multi-round); `pytest.ini` filters `HTTP_422` deprecation.
- No lint/typecheck enforced but `py.typed` present for `detection_viewer`.
- Gradio theme: `console_theme.py` + `console.css`/`console.js` loaded in `app_builder.build_app()` via `Blocks(theme=theme, css=custom_css)`; `console.css` includes DetectionViewer dark overrides and `draw-tab-row` responsive.
- `detection_viewer/static/script.js` is `js_on_load` for `DetectionViewer`; draw canvases use `CustomCanvasController` / `CustomCanvasControllerRT` singletons with id-prefixed HTML for Draw vs Real-Time Draw isolation.
- Logging: standard `logging` with `[LEVEL] message`; `batch/runner.py` uses `log_capture` tail.
