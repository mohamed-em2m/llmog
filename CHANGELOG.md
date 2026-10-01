# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `--batch_public_images` (+ `--image_host`, currently `catbox`): upload
  crop JPEGs to a public host and rewrite inline batch request bodies to the
  URLs before submit, since inline hosts (e.g. OpenRouter) reject base64 /
  data-URI images on every provider. Hash-cached in
  `<output>/.uploaded_images.json` (resumes never re-upload); upload failures
  fail fast. WARNING: uploads are world-readable -- never use for sensitive
  data; sync mode keeps base64 private per request.

### Fixed
- Inline submit now fails fast with guidance when request bodies still embed
  `data:`-URI images instead of billing a batch that fails 100% of requests.
- Failed inline batches also report the batch-level `error.message`, and the
  poll loop tolerates malformed payloads / missing `status` during provider
  registration lag.

## [1.3.0] - 2026-10-01

### Added
- Seed sampling diagnostics for `auto_label`: every run logs whether the seed
  shuffled ALL eligible images before slicing, the eligible-vs-chosen counts
  and the first chosen stems. Warns when `--seed` is set but `--shuffle` is
  off (seed ignored, same first N picked every run), when `--num_samples >=`
  eligible images (no seed can change the set), and when a saved
  `.batch_job.json` is resumed (current shuffle/seed flags do not re-select).
- OpenRouter inline-batch robustness: tolerate transient retrieve failures
  during provider registration lag (HTTP 404, HTTP-200 `{"error": ...}`
  bodies, non-JSON/non-dict payloads, missing `status`) within a grace window
  instead of crashing on the first poll; fail with guidance afterwards.
- Per-request error surfacing: a failed inline batch appends up to 3
  `custom_id: message` pairs from the inlined results; batches with no
  parseable errors log payload keys + truncated raw JSON for schema mapping.
- `:batch` model-suffix normalization: the Batch API takes the base slug
  (OpenRouter resolves `:batch` itself), so a trailing `:batch` is stripped
  from the batch-level model and matching per-request bodies before submit.
- End-of-run flatten tolerates a missing `labels/` dir when nothing was
  staged (failed runs) instead of raising a second error.
- PyPI publishing: `setuptools.build_meta` backend (hatchling cannot build
  this layout), `[tool.setuptools.package-data]` for
  `detection_viewer/static/*` and `interface/console.css|js` (both are read
  at runtime; missing files break PyPI installs), and a fixed
  `.github/workflows/publish.yml` (trusted publishing on release).

### Fixed
- `TestTabServer` deadlock: tests held the non-reentrant
  `state.server_lock` across calls that re-acquire it; the calls now run
  outside the lock.
- Small-box per-image filter log raised from DEBUG to INFO so the
  `min_box_size` filter firing is visible in normal runs.

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
- `--batch_submit_style auto|file|inline` for the Batch API path. Hosts like
  OpenRouter implement `/batches` but ignore `input_file_id` and return results
  inlined in the retrieve response; `inline` submits the requests in the create
  body over plain HTTP and reads results from either shape. `auto` (default)
  tries the file reference and falls back to inline on that specific 400.
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
