# Installation

`uv` is the recommended package manager. The project requires **Python 3.12+**.

## Dependencies

- A local inference backend. Install exactly **one** serving backend.
- Python dependencies managed via `uv`.

## Setup

```bash
uv sync
```

## Choose a serving backend

Pick exactly **one** optional extra depending on how you want to run the model.
All three plug into the same OpenAI-compatible client.

```bash
# llama-cpp-python: Python binding; ships its own OpenAI-compatible server
# (`python -m llama_cpp.server`). Prebuilt wheels -- no native llama.cpp build
# needed. Select --server_type llama_cpp_python (or the "Server Backend"
# dropdown in the web UI)
uv pip install -e ".[llama-cpp-python]"

# llama.cpp (native): lightweight; no extra Python deps to install. Requires the
# standalone `llama-server` binary (build it with ./scripts/install_llama_cpp.sh,
# Linux). Select with --server_type llama_cpp (the default)
uv pip install -e ".[llama-cpp]"

# vLLM: high-throughput GPU serving on CUDA systems.
# Select with --server_type vllm
uv pip install -e ".[vllm]"
```

> `uv sync` alone installs the base package without a GPU serving backend (safe
> for CPU-only development and for pointing at an external `--base_url`).

## Install the documentation toolchain (optional)

To build and preview the API docs:

```bash
uv sync --extra docs
uv run mkdocs serve
```

## Verify

Check the installed entry points:

```bash
uv run llmog --help
uv run detection-gui --help
```
