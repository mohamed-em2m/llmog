# LLM Object Detection Testing Console

An interactive test console for assessing **Vision-Language Models (VLMs)** on
object detection tasks. It uses an iterative **Detector-Judge pipeline**: a
detector agent proposes bounding boxes for objects in an image, a judge agent
critiques them against the original image, and the loop repeats with structured
feedback until a quality score is hit or the round budget runs out.

## Features

- **Detector-Judge loop** — iteratively refine bounding boxes with LLM feedback.
- **Free detection** — detect any comma-separated set of categories on explicit images.
- **Auto-annotation** — batch relabel binary YOLO defect/no-defect boxes into
  multi-class labels.
- **Rich preprocessing** — resolution tuning, CLAHE/autocontrast, gamma,
  denoising, sharpening, white balance, Set-of-Mark (SoM) proposals, grid
  drawing, and image tiling.
- **Image preprocessing** helpers for resolution, contrast, denoise, sharpness,
  color space, and tile mapping back to the original coordinates.
- **Self-hosted serving** — launch and manage local `llama-server` or `vLLM`
  backends, or point at any OpenAI-compatible endpoint.
- **Gradio + Streamlit interfaces** and a realtime detection tab.

## Entry points

| Command | Task | Description |
|---------|------|-------------|
| `llmog` | any | Unified CLI; dispatches by `--task` |
| `detection-cli` | `free_detection` | Detector/judge loop on `--image` paths |
| `auto-annotation` | `auto_label` | Batch YOLO relabeling from a `data.yaml` |
| `detection-gui` | Gradio | Launch the Gradio web interface |

## Documentation

- [Installation](install.md)
- [Usage](usage.md)
- [API Reference](api/main.md)