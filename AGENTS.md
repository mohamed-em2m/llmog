# AGENTS.md — LLM Object Detection Testing Console

## Project Overview
Interactive test console for assessing Vision-Language Models (VLMs) on object detection tasks. Uses an iterative **Detector-Judge pipeline** where a detector proposes bounding boxes, a judge critiques them, and the loop repeats with structured feedback.

## Package Management
- **Tool**: `uv` (fast Python installer/resolver)
- **Python**: 3.12+ (see `.python-version`)
- **Install**: `uv sync` (after running `scripts/install_llama_cpp.sh` for llama.cpp)

## Entry Points
| Command | Module | Description |
|---------|--------|-------------|
| `uv run llmog` | `main:main` | Unified CLI; dispatches by `--task` (`free_detection` / `auto_label`) |
| `uv run detection-cli` | `free_detection:main` | Shortcut for `llmog --task free_detection` (detector/judge loop on `--image` paths) |
| `uv run auto-annotation` | `auto_annotation:main` | Shortcut for `llmog --task auto_label` (batch YOLO relabeling from a `data.yaml`) |
| `uv run detection-gui` | `interface.gui:main` | Launch the Gradio web interface |

Single source of truth for every CLI flag: `llmog/schemes/argument.py:PipelineConfig`
(a pydantic v2 model). `llmog/main.py:build_parser` mirrors every field of
`PipelineConfig` onto an `argparse.ArgumentParser`; `parse_args()` then
overlays an optional `--config <yaml>` file and finally constructs the
validated `PipelineConfig`.

## Key Directories
- `llmog/` — Main source code (package root via `tool.setuptools.package-dir`)
- `llmog/schemes/` — `PipelineConfig` pydantic model + argparse mirror (unified config)
- `llmog/main.py` — Unified CLI dispatcher (`--task free_detection | auto_label`)
- `llmog/free_detection/` — Detector/Judge pipeline package
- `llmog/free_detection/agent/` — LangGraph nodes, pipeline, state, prompts, parser, visuals
- `llmog/auto_annotation/` — Batch YOLO relabeling package
- `llmog/prompts/` — Markdown prompt templates for detector/judge agents (loaded by DynaPrompt)
- `llmog/servers/` — `LlamaServerManager` / `VllmServerManager` + `servers_factory` registry
- `llmog/interface/` — Gradio console theme (CSS/JS) and Streamlit interface
- `scripts/` — Installation scripts (Linux-focused)

## Core Modules
- `free_detection/agent/pipeline.py` — Core `ObjectDetectionPipeline` class; builds and invokes the LangGraph
- `free_detection/agent/graph.py` — `build_detection_graph()`: wires up all LangGraph nodes and conditional edges
- `free_detection/agent/nodes/` — Individual LangGraph nodes: `preprocess`, `detector`, `crop_verify`, `judge`, `loop`, `finalize`
- `free_detection/agent/prompts.py` — Prompt loading (DynaPrompt + Jinja2 fallback), `render_detector_prompt`, `render_judge_prompt`
- `free_detection/agent/parser.py` — JSON extraction, `parse_detections`, `validate_detections`
- `free_detection/detection_pipeline.py` — Backward-compat shim; re-exports everything from `free_detection.agent`
- `free_detection/image_preprocessing.py` — Resolution tuning, CLAHE/autocontrast, gamma, bilateral/NLM denoise, unsharp mask, white balance, SoM proposals, grid drawing, tiling
- `free_detection/gui.py` — `detection-gui` entry point (parses Gradio-specific `--host`/`--port`/`--share` flags, then launches `build_app()`)
- `free_detection/__init__.py:main` — `detection-cli` entry point (accepts a validated `PipelineConfig` or parses argv itself)
- `auto_annotation/main.py` — `auto-annotation` entry point (drives `read_images_with_labels`, checkpoint/yaml persistence, server lifecycle)
- `auto_annotation/cli.py` — Standalone argparse parser for the auto-annotation pipeline (a peer of `PipelineConfig` for users who skip `--task auto_label`)
- `auto_annotation/server_init.py` — `build_client()` chooses between local llama.cpp/vLLM servers and an external `--base_url`
- `schemes/argument.py` — `PipelineConfig` pydantic model + `vllm`/`llama_cpp` kwdict properties used by `servers_factory`
- `servers/llama_server_manager.py` — Local `llama-server` process management (start/stop/logs)

## Common Commands
```bash
# Install dependencies
./scripts/install_llama_cpp.sh  # Linux only; builds llama.cpp with CUDA
uv sync

# Run GUI (default: http://0.0.0.0:7860)
uv run detection-gui
uv run detection-gui --port 7861 --share

# Run unified CLI via --task
uv run llmog --task free_detection -i image.jpg -c "person, car, dog"
uv run llmog --task auto_label --train_image imgs/ --train_label lbls/ \
    --yaml_path data.yaml --model local-model -o ./out

# Run CLI (single image) -- shortcut aliases
uv run detection-cli -i image.jpg -c "person, car, dog"

# Run auto-annotation
uv run auto-annotation --train_image imgs/ --train_label lbls/ \
    --yaml_path data.yaml --model local-model -o ./out

# Run CLI (batch with preprocessing)
uv run detection-cli \
  -i img1.jpg -i img2.jpg \
  -c "crack, scratch, dent" \
  --prep_enabled --prep_contrast_method clahe \
  --prep_tiling_enabled --prep_tile_size 512 \
  --prep_grid_line_color blue --prep_grid_step 50 \
  -o ./results

# Mix a YAML config file with CLI overrides (CLI wins on conflicts)
uv run llmog --task free_detection --config pipeline.yaml -i img.jpg --max_rounds 3
```

## Important Conventions
- **Prompt templates**: Loaded from `llmog/prompts/*.md` via DynaPrompt; Jinja2 template rendering is the fallback. Hardcoded empty strings in `prompts.py` are the last resort.
- **Output structure**: Each image gets a subdir under `--output_folder` with `best_annotated.jpg`, `best_detections.json`, `history.json`
- **API compatibility**: Uses OpenAI Python SDK; works with any OpenAI-compatible endpoint (`llama-server`, vLLM, Ollama)
- **Extra body params**: `min_pixels`/`max_pixels` sent via `extra_body` for Qwen-VL/vLLM backends (controlled by `--prep-send-pixel-bounds`)
- **External API mode**: When enabled, only official OpenAI parameters are sent (no vLLM extensions, sampling params can be disabled)
- **Config precedence**: `pydantic defaults < --config <yaml> < explicit CLI flags`. Implemented by setting every optional argparse default to `argparse.SUPPRESS` so unset flags don't appear in the namespace; `parse_args()` then merges YAML into the parse result and lets explicit CLI flags win.
- **Flag name style**: Both `--prep-tile-size` (dashed) and `--prep_tile_size` (underscored) resolve to the same `PipelineConfig` field — `PipelineConfig` field names use underscores, the parser accepts either form for back-compat.
- **`serving_extra`**: A `Dict[str, Any]` for ad-hoc overrides passed to the underlying `LlamaServerManager`/`VllmServerManager` kwargs. CLI form is `--serving_extra key=value` (repeatable); YAML form is a mapping. Structured `PipelineConfig` fields always take precedence over keys in `serving_extra` when building the manager kwargs.
- **`vllm`/`llama_cpp` properties**: On `PipelineConfig` — return fresh dicts for `servers_factory`. Never mutate them; mutating `serving_extra` in place has no effect because the properties build a new dict each time.

## Development Notes
- No test suite currently exists
- No lint/typecheck configured in `pyproject.toml`
- Gradio UI loads custom CSS/JS from `llmog/interface/` at startup
- Matplotlib backend forced to 'Agg' in `detection_pipeline.py` (line 27)
- Logging uses standard `logging` module with `[LEVEL] message` format