# Installation

`uv` is the recommended package manager. The project requires **Python 3.12+**.

## Dependencies

- A local inference backend. On Linux, run
  `./scripts/install_llama_cpp.sh` first to build llama.cpp with CUDA.
- Python dependencies managed via `uv`.

## Setup

```bash
# Linux only — build llama.cpp with CUDA
./scripts/install_llama_cpp.sh

# Install the project and dependencies into a virtualenv
uv sync
```

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