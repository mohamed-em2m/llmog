# AGENTS.md — LLM Object Detection Testing Console

## Project Overview
Interactive test console for assessing Vision-Language Models (VLMs) on object detection. Uses an iterative **Detector-Judge pipeline** (LangGraph) where a detector proposes bounding boxes, a judge critiques them, and the loop repeats with structured feedback. Gradio console provides **Batch Processing** (top Live Results Explorer + bottom Input/Configuration columns), **Draw & Recognize** (custom canvas + DetectionViewer), **Real-Time Detection** (webcam/video with tracker + same-window overlay) and **Real-Time Draw** (live capture + draw) tabs. All tabs share a unified **DetectionViewer** (`llmog/detection_viewer`) for client-side box rendering (WebP cache, no server PIL re-encode).

## Package Management
- **Tool**: `uv` (fast Python installer/resolver)
- **Python**: 3.12+ (see `.python-version`)
- **Install**: `uv sync` (after `scripts/install_llama_cpp.sh` for llama.cpp) – `pyproject.toml` now includes `detection_viewer` in `tool.setuptools.packages.find`
- **Optional extras**: `uv sync --extra <name>` (pick serving backends) + `uv pip install -e .[esrgan]` for Real-ESRGAN (`torch` CUDA, CPU refused at runtime; `spandrel` is already core)

## Entry Points
| Command | Module | Description |
|---------|--------|-------------|
| `uv run llmog` | `main:main` | Unified CLI; dispatches by `--task` (`free_detection` / `auto_label` / `classify`) |
| `uv run detection-cli` | `free_detection:main` | Shortcut for `llmog --task free_detection` (detector/judge loop on `--image` paths) |
| `uv run auto-annotation` | `auto_annotation:main` | Shortcut for `llmog --task auto_label` (batch YOLO relabeling from a `data.yaml`) |
| `uv run classify-cli` | `image_classification:main` | Shortcut for `llmog --task classify` (whole-image single/multi/top_k) |
| `uv run detection-gui` | `interface.gui:main` | Launch Gradio console (`app_builder.build_app()`) |

Single source of truth for CLI flags: `llmog/schemes/argument.py:PipelineConfig` (pydantic v2). `llmog/main.py:build_parser` mirrors every field onto `argparse`; `parse_args()` overlays optional `--config <yaml>` and constructs validated `PipelineConfig`.

## Key Directories
- `llmog/` — Package root (`tool.setuptools.package-dir = {"": "llmog"}`)
- `llmog/schemes/` — `PipelineConfig` + argparse mirror
- `llmog/main.py` — Unified CLI dispatcher (`--task free_detection | auto_label | classify`)
- `llmog/free_detection/` — Detector/Judge pipeline package
- `llmog/free_detection/agent/` — LangGraph nodes (`preprocess`, `detector`, `crop_verify`, `judge`, `loop`, `finalize`), `pipeline.py`, `state.py`, `visuals.py`, `client_utils.py` (429-aware retry)
- `llmog/detection_viewer/` — New `DetectionViewer` (gr.HTML) – `__init__.py`, `static/template.html|style.css|script.js`, `py.typed` – client-side canvas, WebP cache with dedup (`_WEBP_URL_CACHE`)
- `llmog/auto_annotation/` — Batch YOLO relabeling
  - `batch_api.py` — OpenAI Batch API flow: `collect_batch_requests()` (shuffle→slice→`num_samples`, min-box filter, crop/pad/resize/SoM), `submit_batch_job()` (file/inline/auto styles, `:batch` strip, data-URI guard, public-URL rewrite), `poll_batch_job()` (404/transport tolerance, terminal states), `finalize_batch_job()` (online-path semantics, kept-merge, manifest, checkpoint)
  - `batch_runner.py` — Online path: `read_images_with_labels()` (selection, `ThreadPoolExecutor`, per-batch checkpoint) → `process_one_image()` per image
  - `single_image.py` — Per-image relabel: small-box filter → pad/resize/SoM crop → `detect_defect()` → none/strict/drop guards → write + checkpoint
  - `image_io.py` — Shared crop/body builders: `find_labeled_images()`, `pad_box()`, `draw_som_context()`, `resize_crop_ratio()`, `build_classify_body()` (byte-identical sync/batch), `detect_defect()`
  - `image_hosting.py` — Public crop upload (catbox, SHA-256 cache `.uploaded_images.json`) for inline hosts
  - `checkpoint.py` — `CheckpointManager` (atomic writes), `save_under_locks()` (fixed lock order), `build_run_settings()` fingerprint + `run_settings_mismatches()`
  - `reverse_batches.py` — `flatten_batches_to_labels()` (staging → flat `labels/`, creates missing dir), `rebuild_checkpoint()`, `drop_classes_and_compact()`
  - `cli.py` — Standalone `auto-annotation` parser (mirrors unified flags + hand-built Namespace defaults)
- `llmog/image_classification/` — Whole-image classify task
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
- `scripts/` — `install_llama_cpp.sh` (Linux), `download_real_esrgan.sh` (registry-mirrored 6-model table, idempotent, wget/curl), `run_spandrel.py` (CUDA tiled Real-ESRGAN: argparse, batch_size 4, channels-last, opt-in `--compile`, pinned H2D, fp16 canvas, cached feather mask, resume-skip)
- `llmog/esr/` — Real-ESRGAN stage-0 package: `registry.py` (6-model table, official GitHub release URLs), `download.py` (local→cache→download→HF-fallback resolution, torch-free), `manager.py` (`ESRConfig.from_config`, CUDA-only fingerprinted instances, pre-ESR input cap, tiled fp16 inference with rectangular tiles, `upscale_pil()`), `project.py` (original-size projection: `norm_to_orig_pixels`, `work_pixels_to_orig_pixels`, `esr_growth_factor`, `scaled_pixel_param`, `fit_long_edge`, `tile_geometry`)

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
- **auto_label sampling order**: `find_labeled_images()` (sorted, non-empty labels only) → `shuffle` all with `seed` → `[start:end)` slice → `[:num_samples]`. Seed is ignored without `--shuffle`; same seed = same sample (reproducibility).
- **Small-box filter**: `width < min_box_size OR height < min_box_size` in ORIGINAL pixels, before padding/resize/SoM. `small_box_action keep` preserves the box (coords byte-identical); its class id stays if free in the map, else quarantines ONCE under `original_class_<old>` at a fresh id (`resolve_kept_class()` reuses it for later boxes — never one id per box) — never silently merges. `drop` omits; `drop_small_images` (default ON) writes no file + manifests `<output>/skipped_small_images.txt` for filter-emptied images.
- **Reclassification context**: `--crop_padding_pct` (box-relative % per side, both paths), `--recls_context crop|full_som` (`draw_som_context()` full scene + directive), `--crop_resize_ratio` (scale factor replacing fixed letterbox, long-edge capped). Sync and batch bodies are byte-identical by construction (`build_classify_body()`).
- **Real-ESRGAN stage-0** (`--esr_enabled`, default off): `node_preprocess` upscales right after load (`esr_info` in state + `prep_info`), VLM sees the working image, `node_finalize` re-renders `best_annotated.jpg` on the TRUE original (JSON stays 0-1000). Pixel cosmetics scale by growth factor (`scaled_pixel_param`: grid line width/explicit font, `tile_size_eff` in state consumed by detector; grid `step` untouched). auto_label crops/scenes upscale inside the shared builders (`image_io.maybe_esr_upscale_pil`/`upscale_scene_for_som`, box scaled by TRUE ratio, dry runs never load a model); classify upscales in `encode_image_to_data_uri`. ESR keys are in the checkpoint fingerprint + saved-job reuse warning.
- **Batch API** (`--use_batch_api`, external server only): `batch_mode auto|submit|poll`, job in `<output>/.batch_job.json` (saved job short-circuits rebuild; delete for a fresh sample). `batch_submit_style auto|file|inline` — inline hosts (OpenRouter) need base model slug (`:batch` stripped), metadata-first key order, and reject base64 images (fail-fast guard; `--batch_public_images` uploads crops to catbox with hash cache, world-readable warning). Poll tolerates registration lag; finalize mirrors online semantics into `<output>/batches/batch_0000/` (the layout flatten scans).
- **Checkpoints**: `.checkpoint.json` holds `completed_images`/`class_map`/`batches_done` + `run_settings` fingerprint; resume WARNINGs on any changed label-affecting flag and drops stale `batches_done` when `--batch_size` changes. Writes go through `save_under_locks()` (fixed lock order) — never snapshot-then-`save()` across threads. Non-server model failures skip write+checkpoint (retry on resume) like server failures.
- **Packaging**: `setuptools.build_meta` backend (hatchling cannot build this layout); `[tool.setuptools.package-data]` ships `detection_viewer/static/*` + `interface/console.css|js` (both read at runtime); `twine check` must pass on sdist+wheel. `prompts/*.md` are NOT packaged (all loaders have hardcoded fallbacks).

## Development Notes
- Test suite: `pytest` ~160 tests across 9 files (`test_batch_api.py` is the largest: batch flow, selection, filters, poll/finalize, hosting). `pytest.ini` filters `HTTP_422` deprecation. 2 known pre-existing failures (`test_pipeline_execution_tiled`, `test_load_image_variants` — fail on clean tree too).
- Pre-commit hooks enforced (ruff, ruff-format, commitlint-conventional): keep diffs formatted or commits abort; multi-line `if` conditions get collapsed — check `git status` after a failed commit and recommit.
- No typecheck enforced but `py.typed` present for `detection_viewer`.
- Gradio theme: `console_theme.py` + `console.css`/`console.js` loaded in `app_builder.build_app()` via `Blocks(theme=theme, css=custom_css)`; `console.css` includes DetectionViewer dark overrides and `draw-tab-row` responsive.
- `detection_viewer/static/script.js` is `js_on_load` for `DetectionViewer`; draw canvases use `CustomCanvasController` / `CustomCanvasControllerRT` singletons with id-prefixed HTML for Draw vs Real-Time Draw isolation.
- Logging: standard `logging` with `[LEVEL] message`; `batch/runner.py` uses `log_capture` tail.
