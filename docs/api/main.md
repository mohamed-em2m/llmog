# `main` — Unified CLI

The single public entry point for running the project from the command line.
`llmog` console script dispatches to either `free_detection.main` or
`auto_annotation.main` based on `--task`.

::: main
