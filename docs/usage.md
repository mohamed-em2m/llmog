# Usage

There are three primary ways to drive the project: the unified CLI, the
shortcut entry points, and the Gradio web interface. All configuration flows
through [`PipelineConfig`](api/schemes/argument.md), a pydantic model that is the
single source of truth for every flag.

## Unified CLI

```bash
# Free detection
uv run llmog --task free_detection -i image.jpg -c "person, car, dog"

# Auto-annotation (batch YOLO relabeling)
uv run llmog --task auto_label --train_image imgs/ --train_label lbls/ \
    --yaml_path data.yaml --model local-model -o ./out
```

## Shortcut entry points

The shortcut commands inject the task for you.

```bash
# Single-image detector/judge loop
uv run detection-cli -i image.jpg -c "person, car, dog"

# Batch YOLO relabeling
uv run auto-annotation --train_image imgs/ --train_label lbls/ \
    --yaml_path data.yaml --model local-model -o ./out

# Gradio web interface (default: http://0.0.0.0:7860)
uv run detection-gui
uv run detection-gui --port 7861 --share
```

## Batch detection with preprocessing

```bash
uv run detection-cli \
  -i img1.jpg -i img2.jpg \
  -c "crack, scratch, dent" \
  --prep_enabled --prep_contrast_method clahe \
  --prep_tiling_enabled --prep_tile_size 512 \
  --prep_grid_line_color blue --prep_grid_step 50 \
  -o ./results
```

## Mixing a YAML config with CLI overrides

A YAML config file provides base values, and explicit CLI flags always win on
conflict. Precedence is: **pydantic defaults < YAML config < explicit CLI flags**.

```bash
uv run llmog --task free_detection --config pipeline.yaml -i img.jpg --max_rounds 3
```

## Output structure

For each image, a subdirectory under `--output_folder` is created containing:

- `best_annotated.jpg` — the best-scoring round annotated image
- `best_detections.json` — the best round's detections
- `history.json` — the full per-round history

## Pointing at an external OpenAI-compatible endpoint

```bash
uv run detection-cli -i image.jpg -c "person" \
    --base_url http://my-server:8000/v1 --api_key sk-...
```

## Parameter reference

Every CLI flag mirrors a `PipelineConfig` field. See the
[`PipelineConfig` reference](api/schemes/argument.md) for the complete list of
fields, defaults, and validators.