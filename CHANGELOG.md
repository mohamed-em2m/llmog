# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Real-ESRGAN upscaling (`--esr_enabled`, CLI + YAML, default off, all
  tasks): super-resolves VLM-bound pixels with Real-ESRGAN BEFORE the model
  sees them -- full image in `free_detection` (stage 0, before
  resolution/grid/tiling) and `classify`, crops + full_som scenes in
  `auto_label` (gated by `--esr_for_crops`, default on). New `esr` package:
  weight registry (`general-x4v3` default, `x4plus`, `x2plus`, anime
  variants; official GitHub release assets, `~/.cache/llmog/esr`,
  `--esr_model_path` local override, optional HF repo fallback),
  CUDA-only torch-gated manager (fp16, channels-last, opt-in
  `--esr_compile`, tiled VRAM-safe inference, install via
  `uv pip install -e .[esrgan]`). Final outputs always map to ORIGINAL
  dims: `best_annotated.jpg` is re-rendered on the original image,
  detections JSON stays 0-1000, YOLO coords stay in original space; grid
  line width/font and tile size auto-scale by the growth factor so the red
  grid keeps its relative measures. `--esr_target_long_edge` (default
  2048, 0 = native), `--esr_max_long_edge` VRAM guard (default 4096).
  ESR keys join the checkpoint run-settings fingerprint, and saved batch
  jobs warn when ESR flags change after the build (requests are baked).
- ESR robustness fixes: rectangular tiling (narrow images such as 600x100
  crops no longer crash blending on a square feather mask -- effective
  tile dims clamp per axis in `esr.project.tile_geometry`, mirrored in
  `scripts/run_spandrel.py`); pre-ESR input cap (`max_long_edge /
  native_scale`) so the VRAM guard holds during inference, not just after
  (working target is also clamped to the cap); fingerprinted upscaler
  instances (distinct configs coexist instead of first-wins); crop
  `long_edge_cap` (crops stop swelling to the 2048 scene target before the
  downstream letterbox); dry-run batch builds never load the SR model.
- `--crop_padding_pct` (CLI + YAML, default 0): expand each auto_label box by
  this % of its own width/height per side (clamped to the image) before the
  VLM crop, giving the classifier surrounding context. Applies identically to
  the sync and batch-build paths; the small-box filter still measures the
  original box and output YOLO coords are never padded.
- `--recls_context crop|full_som` (CLI + YAML, default `crop`): `full_som`
  sends the FULL scene with the box highlighted/numbered (shared
  `draw_som_context` helper) plus a directive to classify only the marked
  box, instead of the crop. Full images cost more tokens per request.
- `--crop_resize_ratio` (CLI + YAML, unset by default): scale the (padded)
  crop by this factor (LANCZOS, aspect preserved, no letterbox bars) instead
  of the fixed `height` x `width` letterbox. Crop mode only; the long edge
  is capped at `max(height, width)` to bound batch payload sizes.
- Run-settings fingerprint: `.checkpoint.json` now records the label-affecting
  settings (`crop_padding_pct`, `recls_context`, `crop_resize_ratio`,
  `min_box_size`, `small_box_action`, `model`, `height`/`width`, `class_mode`,
  `none_labels`, `drop_none`, `batch_size`); resuming with changed flags
  WARNING-logs every old→new value instead of silently mixing label vintages.
  Changing `--batch_size` additionally drops stale `batches_done` indices
  (per-image resume still applies). Old checkpoints without a fingerprint skip
  the check; batch jobs carry the fingerprint from submit to finalize.
- `--batch_public_images` (+ `--image_host`, currently `catbox`): upload
  crop JPEGs to a public host and rewrite inline batch request bodies to the
  URLs before submit, since inline hosts (e.g. OpenRouter) reject base64 /
  data-URI images on every provider. Hash-cached in
  `<output>/.uploaded_images.json` (resumes never re-upload); upload failures
  fail fast. WARNING: uploads are world-readable -- never use for sensitive
  data; sync mode keeps base64 private per request.

### Fixed
- Small-box `keep` now registers kept boxes' original class ids in the class
  map (`original_class_<id>`, with a WARNING) when missing, so kept lines
  never reference a nameless id in data.yaml (previously untrainable output).
- Kept boxes can no longer silently merge into an unrelated class: a kept id
  already taken in the current map (e.g. an old binary id reinterpreted
  under a new multi-class map) mints a FRESH id (`original_class_<old>`)
  with a WARNING, while coordinates stay byte-identical and the model is
  never called. Free ids keep their original number. Kept boxes log their
  resolved `id ('name')` per box.
- Upload cache now persists incrementally, so an interrupted public-image
  upload run resumes without re-uploading; the on-disk `batch_requests.jsonl`
  is re-written after URL rewriting so the artifact matches what was sent.
- Inline submit now fails fast with guidance when request bodies still embed
  `data:`-URI images instead of billing a batch that fails 100% of requests.
- Failed inline batches also report the batch-level `error.message`, and the
  poll loop tolerates malformed payloads / missing `status` during provider
  registration lag.
- Inline create body key order locked by test (`endpoint`, `model`, then
  `requests` last) per OpenRouter's stream-parser requirement.
- Batch finalize now stages under `<output>/batches/batch_0000/` (the layout
  the end-of-run flatten scans); the previous `labels/batches/` location was
  invisible to flatten, silently losing every batch label. The flatten step
  also creates a missing `labels/` dir instead of erroring.
- Images whose every box fails for non-server reasons (auth, bad model,
  4xx validation, unusable responses) are no longer written as forged-empty
  labels nor checkpointed -- a resumed run retries them (mirrors the
  server-failure rule). New `images_failed_unclassified` summary counter.
- Fresh batch builds skip checkpoint-completed images (no more resubmitting
  and rebilling finished work); `--dry_run` with a saved job does nothing;
  a rejected `--batch_mode submit` no longer mutates the saved job file.
- Checkpoint saves are atomic under load (`save_under_locks` snapshots and
  writes under one mutex): concurrent workers can no longer wipe each
  other's progress or burn duplicate class ids. Legacy `--resume` is ignored
  (with a warning) together with `--no_auto_resume`.
- Poll tolerates transient connection/timeout blips, validates the saved
  job's `batch_id` cleanly, and finalize aborts (without marking done) when
  no result matches any submitted request; empty-result and zero-image
  finalizations are distinguished in the done-gate messaging.
- Track `cancelling` as an in-progress batch status; read per-request error
  `type` as a message fallback.
- README renders on PyPI: absolute asset URLs, corrected repo links, and a
  `twine check`-clean long description.

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
