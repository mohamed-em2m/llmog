# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-10-01

### Added
- `auto_label` small-box reclassification filter: `--min_box_size` skips boxes
  smaller than N pixels (width < MIN or height < MIN) so unclassifiable tiny
  crops are never sent to the LLM. `--small_box_action keep` (default) writes
  the original YOLO line verbatim; `drop` omits the box. (YAML keys:
  `min_box_size`, `small_box_action`.)
- False-negative guard for the size filter: `--drop_small_images` (default ON)
  writes NO label file for images emptied purely by the small-box filter and
  lists them in `<output>/skipped_small_images.txt` for exclusion from
  training, instead of an empty `.txt` that would train as background.
  `--keep_small_images` restores legacy empty-file semantics.
- OpenAI Batch API flow for `auto_label` (~50% cheaper than sync):
  `--use_batch_api` submits one `/v1/chat/completions` request per box as a
  batch job and finalizes results into YOLO labels with online-path semantics
  (none-drop, strict-mode guard, class_map append, checkpoint/resume,
  flatten-compatible staging). `--batch_mode auto|submit|poll`,
  `--batch_poll_interval`, `--batch_poll_timeout`, `--batch_completion_window`,
  `--batch_job_id`. Requires `--server_type external` against a provider with
  `/v1/batches` support.

## [1.1.0]

### Added
- Whole-image `classify` task with CSV/YOLO output (`--task classify`).
- `auto_label` per-crop image sizing (`--image_size` / `--height` / `--width`).
- None-label handling (`--none_labels`, `--drop_none`) with YAML list support.
- Provider-specific `extra_body` forwarding on every classify call
  (CLI JSON string or YAML mapping).
- External API endpoint support through YAML config + vLLM example config.

### Fixed
- Auto-label batches stage under `<output>/batches/batch_XXXX/` and flatten
  to `<output>/labels/` only after all batches finish.
