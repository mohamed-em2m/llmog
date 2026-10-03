from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PipelineConfig(BaseModel):
    """Validated configuration for the unified relabeler / detector pipeline.

    A single config object drives every entry point in the project:

      * ``task="free_detection"`` -> :func:`free_detection.main` runs the
        detector/judge pipeline on explicit ``--image`` paths.
      * ``task="auto_label"`` -> :func:`auto_annotation.main` relabels
        binary YOLO defect/no-defect boxes into multi-class labels using a
        folder of images + labels described by a ``data.yaml``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- Task selection -----------------------------------------------------
    task: Literal["free_detection", "auto_label", "classify"] = "free_detection"

    # --- Logging -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[str] = None

    # --- Input images ------------------------------------------------------
    # Detection mode: explicit image list (may be empty when relabeling from folders).
    images: List[str] = Field(default_factory=list)
    train_image: Optional[str] = None
    train_label: Optional[str] = None
    yaml_path: Optional[str] = None
    input_folder: Optional[str] = None
    image_extensions: str = ".jpg,.jpeg,.png"

    # --- Categories --------------------------------------------------------
    categories: str = "person, car, bicycle, dog, cat"
    definitions: str = ""
    init_class_map: bool = False
    conf_threshold: Literal[1, 2, 3, 4, 5] = 2

    # --- Whole-image classification (task="classify") ----------------------
    classification_mode: Literal["single", "multi", "top_k"] = "single"
    top_k: int = 3
    multi_threshold: float = 50.0
    class_mode: Literal["strict", "hybrid", "free"] = "strict"
    class_definitions: str = ""
    preset: Optional[str] = None
    output_format: Literal["csv", "yolo", "both"] = "csv"
    classification_temperature: float = 0.2
    classification_max_tokens: int = 1024

    # --- Output ------------------------------------------------------------
    output_folder: str = "./detection_results"
    inplace_saving: bool = False
    no_plot: bool = False

    # --- Sampling / slicing ------------------------------------------------
    num_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    batch_size: int = 0
    dry_run: bool = False
    # After all batches finish, copy best-per-stem labels to the top level of
    # <output-folder>/labels/ (YOLO-trainable flat layout). In-progress batches
    # are staged under <output-folder>/batches/batch_XXXX/ so labels/ is only
    # ever final output. Copy-only: batch folders and the checkpoint are kept
    # so resume keeps working.
    flatten: bool = True

    # --- Resume ------------------------------------------------------------
    resume: bool = False
    auto_resume: bool = True

    # --- Server-failure safety (auto_label) ----------------------------------
    # Consecutive server-class model failures (dead process, timeout, 5xx,
    # OOM) after which the run aborts instead of writing fake-empty labels.
    max_consecutive_failures: int = 20
    # When False, the run keeps going image-by-image without a server
    # (legacy behavior); failed images are still left out of the checkpoint
    # so a resumed run retries them.
    abort_on_server_down: bool = True

    # --- Server / model ----------------------------------------------------
    model: str = "local-model"
    detector_model: str = "local-model"
    judge_model: str = "local-model"
    judge_url: Optional[str] = None
    api_key: str = "not-needed"
    base_url: str = "http://localhost:8080/v1"
    server_type: Literal["llama_cpp", "llama_cpp_python", "vllm", "external"] = (
        "llama_cpp"
    )
    max_workers: int = 1

    # llama.cpp-specific
    enable_thinking: bool = False
    use_mtp: bool = True
    ctx_size: int = 20000
    port: int = 8080
    parallel_slots: int = 1
    server_batch_size: int = 1024
    ubatch_size: int = 512

    # --- Detection pipeline tuning ----------------------------------------
    max_rounds: int = 2
    score_threshold: int = 8
    detector_temperature: float = 0.9
    detector_top_p: float = 0.95
    judge_temperature: float = 0.2
    detector_max_tokens: int = 4096
    judge_max_tokens: int = 1024
    api_retries: int = 3

    # --- vLLM configuration ------------------------------------------------
    max_model_len: int = 20000
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    kv_cache_dtype: str = "auto"
    max_num_seqs: int = 1
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    trust_remote_code: bool = True
    download_dir: Optional[str] = None
    limit_mm_per_prompt: Optional[str] = None
    chat_template: Optional[str] = None

    # --- VLM image encoding -----------------------------------------------
    image_min_tokens: int = 1024
    image_max_tokens: int = 4096
    height: int = 1024
    width: int = 1024
    # Convenience square size: when set (>0), overrides height/width so the
    # crop is resized to image_size x image_size (YOLO-style `imgsz`).
    image_size: Optional[int] = None

    # --- Auto-label none / no-detection handling ---------------------------
    # Model outputs whose class normalizes to one of these labels are treated
    # as "nothing detected" (case-insensitive; spaces and dashes are
    # normalized to underscores before comparison). Accepts a comma-separated
    # string (CLI) or a YAML list (``--config``); stored comma-separated.
    none_labels: str = (
        "none,no_detection,nodetection,no_defect,background,unknown,negative,normal"
    )

    @field_validator("none_labels", mode="before")
    @classmethod
    def _coerce_none_labels(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple, set)):
            return ",".join(str(x).strip() for x in v if str(x).strip())
        return v

    # When True (default), none-like boxes are DROPPED: no YOLO line is
    # written for them, so an image whose every box is none-like gets an
    # empty (0-byte) .txt file = empty YOLO prediction. When False, none-like
    # labels are kept as regular classes (legacy behavior).
    drop_none: bool = True

    # --- Small-box reclassification filter (auto_label) --------------------
    # Boxes whose pixel size on the ORIGINAL image is smaller than
    # min_box_size in either dimension (w < min OR h < min) are never sent
    # to the LLM -- tiny crops are usually unclassifiable noise. 0 disables.
    min_box_size: int = 0
    # What to do with small boxes: "keep" writes the original YOLO line
    # verbatim (original class id + coords, no LLM call, no class_map
    # mutation); "drop" skips the box entirely (no YOLO line).
    # NOTE on keep: coordinates stay byte-identical, but the class id is
    # resolved, not blindly copied: a free id is kept in place (registered
    # as original_class_<id> when nameless); an id already taken in the
    # current map mints a FRESH id so the box can never silently merge into
    # an unrelated class (e.g. an old binary id reinterpreted under a new
    # multi-class map). Unknown kept ids are registered as original_class_<id>
    # (with a WARNING) so data.yaml stays trainable.
    small_box_action: Literal["keep", "drop"] = "keep"
    # Image-level guard against false negatives: when True (default) and
    # small_box_action="drop", an image that ends up with ZERO writable
    # boxes only because the small-box filter removed them gets NO label
    # file at all (and its stem is appended to skipped_small_images.txt in
    # the output folder) instead of an empty .txt -- an empty file would
    # teach the detector "no objects here" for an image that does contain
    # defects. Remove those images from the training set (see the manifest)
    # so they don't train as background. When False, the empty file is
    # written (legacy drop semantics).
    drop_small_images: bool = True
    # Context padding for the crop sent to the VLM: each box is expanded by
    # this % of its own width/height per side (50 = half a box-width of
    # context on every side), clamped to the image. 0 (default) keeps the
    # legacy exact-box crop. The size filter above still measures the
    # ORIGINAL box, and output YOLO coords are never padded.
    crop_padding_pct: float = 0.0
    # What the VLM sees per box: "crop" sends the (optionally padded) crop;
    # "full_som" sends the FULL scene with the box highlighted/numbered and a
    # directive to classify only the marked box. Full images cost more tokens
    # per request -- relevant for batch payload sizes on metered providers.
    recls_context: Literal["crop", "full_som"] = "crop"
    # Ratio resize for the VLM crop (crop mode only): scale the (padded) crop
    # by this factor (1.5 = 150%, LANCZOS, aspect preserved, no letterbox
    # bars) instead of the fixed height x width letterbox. The long edge is
    # capped at max(height, width) to bound batch payload sizes. None
    # (default) keeps the fixed-size letterbox.
    crop_resize_ratio: Optional[float] = None

    # --- OpenAI Batch API (auto_label, ~50% cheaper than sync) -------------
    # Submit one /v1/chat/completions request per box as a batch job, then
    # poll to completion and finalize into YOLO labels. Requires an external
    # OpenAI-compatible provider with /v1/batches support (local llama.cpp /
    # vLLM servers do not have it).
    use_batch_api: bool = False
    # auto: resume a saved job or submit+poll+finalize in one run.
    # submit: build + submit only, exit (finalize later, 24h window).
    # poll: poll a saved job (or --batch_job_id) and finalize.
    batch_mode: Literal["auto", "submit", "poll"] = "auto"
    batch_poll_interval: int = 60
    # 0 = poll forever; otherwise give up waiting after this many seconds
    # (the job stays alive provider-side; re-run with --batch_mode poll).
    batch_poll_timeout: int = 0
    batch_completion_window: Literal["24h"] = "24h"
    # Poll/finalize a specific provider batch id instead of the saved job.
    batch_job_id: Optional[str] = None
    # How the batch requests reach the provider:
    #   file   -- OpenAI's way: upload the .jsonl, pass input_file_id.
    #   inline -- embed the requests in the create body, for hosts whose
    #             /v1/batches ignores input_file_id.
    #   auto   -- try file, fall back to inline on that specific 400.
    batch_submit_style: Literal["auto", "file", "inline"] = "auto"
    # Inline batch hosts (e.g. OpenRouter) accept images as public http(s)
    # URLs only. When True, crop JPEGs are uploaded to --image_host and the
    # request bodies rewritten to the public URLs before an inline submit
    # (hash-cached in <output>/.uploaded_images.json, never re-uploaded).
    # WARNING: uploads are world-readable -- never use for sensitive data;
    # leave off and either use sync mode or a file-style (OpenAI) batch.
    batch_public_images: bool = False
    image_host: Literal["catbox"] = "catbox"

    # --- Preprocessing -----------------------------------------------------
    prep_enabled: bool = False
    prep_short_edge: int = 1024
    prep_pad_square: bool = False
    prep_contrast_method: Literal["none", "clahe", "autocontrast"] = "none"
    prep_gamma: float = 1.0
    prep_denoise_method: Literal["none", "bilateral", "nlm"] = "none"
    prep_sharpen: bool = False
    prep_white_balance: bool = False
    prep_grid_style: Literal["standard", "transparent", "fine", "none"] = "standard"
    prep_som_enabled: bool = False
    prep_tiling_enabled: bool = False
    prep_tile_size: int = 512
    prep_tile_overlap: float = 0.2
    prep_crop_verify_enabled: bool = False
    prep_crop_padding: float = 0.15

    # Custom grid overlays
    prep_grid_step: int = 100
    prep_grid_line_width: int = 1
    prep_grid_font_size: int = 0
    prep_grid_line_color: str = "red"
    prep_grid_text_color: str = "white"
    prep_grid_backing_color: str = "black"

    # VLM processor pixels
    prep_send_pixel_bounds: bool = False
    prep_min_pixels: int = 200_704
    prep_max_pixels: int = 4_194_304

    # --- Real-ESRGAN upscaling (all tasks, opt-in) ---------------------------
    # When True, images are super-resolved with Real-ESRGAN BEFORE the VLM
    # sees them: free_detection upscales the full image (stage 0, before
    # resolution/grid/tiling), auto_label upscales each VLM crop (and the
    # full scene in full_som mode) when esr_for_crops is set, classify
    # upscales the whole image before encoding. Final boxes/outputs are
    # always projected back onto the ORIGINAL dims (see esr.project), and
    # pixel-space cosmetics (grid line width/font, tile size) are scaled by
    # the growth factor so the VLM sees the same relative grid. Needs the
    # [esrgan] extra (torch, CUDA-only; CPU is refused at runtime).
    esr_enabled: bool = False
    # Registry key from esr.registry.ESR_MODEL_CHOICES (default tiny/fast
    # general-x4v3). Any other spandrel-compatible checkpoint works via
    # esr_model_path, which always wins over the registry.
    esr_model: Literal[
        "general-x4v3",
        "general-wdn-x4v3",
        "animevideov3",
        "x4plus",
        "x4plus-anime-6B",
        "x2plus",
    ] = "general-x4v3"
    # Explicit local checkpoint file (wins over registry/download). Also
    # readable from the LLMOG_ESR_MODEL env var (see esr.download).
    esr_model_path: Optional[str] = None
    # Optional HuggingFace repo id holding the registry file name, used only
    # as a fallback when the official download fails (needs huggingface_hub).
    esr_model_repo: Optional[str] = None
    # Cache dir for downloaded weights (default ~/.cache/llmog/esr, or
    # LLMOG_CACHE_DIR / XDG layout; see esr.download.default_cache_dir).
    esr_cache_dir: Optional[str] = None
    # Upscale factor override. None (default) uses the checkpoint's native
    # scale; an explicit mismatch is an error, not a silent distortion.
    esr_scale: Optional[int] = None
    # Working size after ESR: fit the long edge to this (aspect preserved,
    # LANCZOS). 0 keeps the native ESR output. Default 2048.
    esr_target_long_edge: int = 2048
    # VRAM guard: ESR output past this long edge is Lanczos-downscaled first.
    esr_max_long_edge: int = 4096
    esr_tile_size: int = 512
    esr_overlap: int = 16
    esr_batch_size: int = 4
    # Apply ESR to auto_label VLM crops (default OFF -- the pipeline
    # upscales the WHOLE image/scene; per-crop ESR is opt-in for tiny
    # defects). Full scenes (full_som) and classify images always upscale
    # when esr_enabled is True. Only matters when esr_enabled is True.
    esr_for_crops: bool = False
    # torch.compile the SR model (faster batches, slow first run + warmup).
    esr_compile: bool = False
    # channels-last memory format (faster convs on CUDA, on by default).
    esr_channels_last: bool = True
    # CUDA device spec ("auto" = first CUDA device). Non-CUDA is refused.
    esr_device: str = "auto"

    serving_extra: Dict[str, Any] = Field(default_factory=dict)
    # Raw extra command-line tokens forwarded verbatim to vLLM (list[str]).
    extra_args: Optional[List[str]] = None
    # Provider-specific params forwarded verbatim as the `extra_body=` kwarg
    # of every auto_label chat-completions call (e.g. OpenRouter `provider`
    # routing, reasoning controls). Set it as a YAML mapping; None (default)
    # means "send nothing extra" so current behavior is unchanged.
    extra_body: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ validators
    @field_validator("extra_body", mode="before")
    @classmethod
    def _coerce_extra_body(cls, v: Any) -> Any:
        # YAML mapping passes through; a CLI JSON string is parsed here so
        # the unified entry point accepts --extra_body '{"k": v}' directly.
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except ValueError as e:
                raise ValueError(
                    f"--extra_body must be a JSON object, got: {v!r} ({e})"
                )
            if not isinstance(parsed, dict):
                raise ValueError(f"--extra_body must be a JSON object, got: {v!r}")
            return parsed
        raise ValueError(f"--extra_body must be a mapping or JSON object, got: {v!r}")

    @field_validator("images", mode="before")
    @classmethod
    def _coerce_images(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return v

    @field_validator("prep_tile_overlap")
    @classmethod
    def _check_overlap(cls, v: float) -> float:
        if not 0.0 <= v <= 0.5:
            raise ValueError("prep_tile_overlap must be between 0.0 and 0.5")
        return v

    @field_validator("max_consecutive_failures")
    @classmethod
    def _check_consecutive_failures(cls, v: int) -> int:
        if v < 1:
            raise ValueError("--max_consecutive_failures must be >= 1")
        return v

    @field_validator("min_box_size")
    @classmethod
    def _check_min_box_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError("--min_box_size must be >= 0 (0 disables the filter)")
        return v

    @field_validator("crop_padding_pct")
    @classmethod
    def _check_crop_padding_pct(cls, v: float) -> float:
        if v < 0:
            raise ValueError("--crop_padding_pct must be >= 0 (0 = exact-box crop)")
        return v

    @field_validator("crop_resize_ratio")
    @classmethod
    def _check_crop_resize_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("--crop_resize_ratio must be > 0 when set")
        return v

    @field_validator("esr_target_long_edge")
    @classmethod
    def _check_esr_target(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                "--esr_target_long_edge must be >= 0 (0 = keep native ESR output)"
            )
        return v

    @field_validator("esr_max_long_edge")
    @classmethod
    def _check_esr_max(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("--esr_max_long_edge must be > 0")
        return v

    @field_validator("esr_tile_size")
    @classmethod
    def _check_esr_tile(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("--esr_tile_size must be > 0")
        return v

    @field_validator("esr_overlap")
    @classmethod
    def _check_esr_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("--esr_overlap must be >= 0")
        return v

    @field_validator("esr_scale")
    @classmethod
    def _check_esr_scale(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("--esr_scale must be > 0 when set")
        return v

    @field_validator("batch_poll_interval")
    @classmethod
    def _check_batch_poll_interval(cls, v: int) -> int:
        if v < 5:
            raise ValueError("--batch_poll_interval must be >= 5 seconds")
        return v

    @field_validator("batch_poll_timeout")
    @classmethod
    def _check_batch_poll_timeout(cls, v: int) -> int:
        if v < 0:
            raise ValueError("--batch_poll_timeout must be >= 0 (0 = poll forever)")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _check_gpu_mem(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def _check_image_size(self) -> "PipelineConfig":
        """Apply --image_size square override onto height/width."""
        if self.image_size is not None:
            if self.image_size <= 0:
                raise ValueError("--image_size must be > 0")
            # Square override wins over individual height/width values.
            self.height = self.image_size
            self.width = self.image_size
        if self.height <= 0 or self.width <= 0:
            raise ValueError("--height/--width must be > 0")
        return self

    @model_validator(mode="after")
    def _normalize_definition_text(self) -> "PipelineConfig":
        """Normalize escaped ``\\n`` literals in definition text.

        Shells (notably PowerShell double-quoted strings) pass ``"a\\n b"``
        through as a literal backslash-n rather than a newline. The prompts
        expect real newlines (one definition per line), so convert them here
        once so every task (free_detection / auto_label / classify) benefits.
        """
        if isinstance(self.definitions, str) and "\\n" in self.definitions:
            self.definitions = self.definitions.replace("\\n", "\n")
        if isinstance(self.class_definitions, str) and "\\n" in self.class_definitions:
            self.class_definitions = self.class_definitions.replace("\\n", "\n")
        return self

    @model_validator(mode="after")
    def _check_indices(self) -> "PipelineConfig":
        if self.start_index is not None and self.start_index < 0:
            raise ValueError("--start_index must be >= 0")
        if (
            self.end_index is not None
            and self.start_index is not None
            and self.end_index <= self.start_index
        ):
            raise ValueError("--end_index must be greater than --start_index")
        return self

    @model_validator(mode="after")
    def _check_input_source(self) -> "PipelineConfig":
        """Input source depends on the selected task.

        ``free_detection`` requires one or more explicit ``--image`` paths.
        ``auto_label`` requires a ``--train_image`` folder (and typically also
        ``--train_label`` and ``--yaml_path``, but those are enforced by the
        sub-entry-point, not the schema, so users can stage experiments).
        ``classify`` requires ``--image`` paths and/or ``--input_folder``.
        """
        if self.task == "free_detection" and not self.images:
            raise ValueError(
                "task='free_detection' requires at least one --image/-i path."
            )
        if self.task == "auto_label" and not self.train_image:
            raise ValueError(
                "task='auto_label' requires --train_image (folder of images)."
            )
        if self.task == "classify" and not self.images and not self.input_folder:
            raise ValueError(
                "task='classify' requires at least one --image/-i path "
                "or --input_folder."
            )
        if self.top_k < 1:
            raise ValueError("--top_k must be >= 1")
        if not 0.0 <= self.multi_threshold <= 100.0:
            raise ValueError("--multi_threshold must be between 0 and 100")
        return self

    @property
    def vllm(self) -> Dict[str, Any]:
        """Build the kwarg dict consumed by :class:`VllmServerManager`.

        Always returns a freshly-built dict so the user-supplied
        ``serving_extra`` is never mutated in place (which would otherwise
        leak preview-specific overrides back into the config object).
        """
        args: Dict[str, Any] = dict(self.serving_extra or {})
        args["--tensor-parallel-size"] = self.tensor_parallel_size
        args["--pipeline_parallel_size"] = self.pipeline_parallel_size
        args["--dtype"] = self.dtype
        args["--kv_cache_dtype"] = self.kv_cache_dtype
        args["--max_num_seqs"] = self.max_num_seqs
        args["--enforce_eager"] = self.enforce_eager
        args["--enable_chunked_prefill"] = self.enable_chunked_prefill
        args["--enable_prefix_caching"] = self.enable_prefix_caching
        args["--speculative_model"] = self.speculative_model
        args["--num_speculative_tokens"] = self.num_speculative_tokens
        args["--trust_remote_code"] = self.trust_remote_code
        args["--download_dir"] = self.download_dir
        args["--limit_mm_per_prompt"] = self.limit_mm_per_prompt
        args["--chat_template"] = self.chat_template
        args["--quantization"] = self.quantization
        args["--model"] = self.model
        return args

    @property
    def llama_cpp(self) -> Dict[str, Any]:
        """Build the kwarg dict consumed by :class:`LlamaServerManager`."""
        args: Dict[str, Any] = dict(self.serving_extra or {})

        # 1. Base Model & Context Setup
        args["-m"] = self.model
        args["--ctx-size"] = self.ctx_size
        args["--port"] = self.port

        # 2. KV Cache Data Type Mapping
        # vLLM choice mapping (e.g., 'fp16', 'bf16', 'fp8', 'q8_0', 'q4_0')
        if self.kv_cache_dtype:
            # Standardize vLLM 'fp8' / 'auto' naming to typical llama.cpp cache types
            dtype_map = {
                "fp16": "f16",
                "bf16": "f16",
                "fp8": "q8_0",  # 'q8_0' is standard 8-bit cache quantization in llama.cpp
                "fp8_e5m2": "q8_0",
                "fp8_e4m3": "q8_0",
            }
            # Fallback directly to the string if user directly passed llama.cpp formats (e.g. 'q4_0')
            target_dtype = dtype_map.get(self.kv_cache_dtype, self.kv_cache_dtype)
            if target_dtype != "auto":
                args["--cache-type-k"] = target_dtype  # Key cache type
                args["--cache-type-v"] = target_dtype  # Value cache type

        # 3. Concurrency & Queue Capacity
        # vLLM max_num_seqs determines simultaneously active request slots
        args["--parallel"] = (
            self.parallel_slots if self.parallel_slots else self.max_num_seqs
        )

        # 4. Multi-GPU Splitting & Execution Controls
        args["--gpu-layers"] = 999  # Mandate all layer processing offloads to GPU
        if self.tensor_parallel_size > 1:
            args["--split-mode"] = "layer"

        # Flash Attention optimization toggle is disabled when eager is enforced
        args["--flash-attn"] = not self.enforce_eager

        # 5. Caching & Batch Strategies
        if self.enable_prefix_caching:
            args["--cont-batching"] = True  # Continuous batching handles reuse prompts

        if self.enable_chunked_prefill:
            args["--batch-size"] = 512  # Restricts chunk limits per step optimization

        # 6. Speculative Decoding Configurations
        if self.speculative_model:
            args["--model-draft"] = self.speculative_model  # Secondary draft model path
            if self.num_speculative_tokens:
                args["--n-predict"] = self.num_speculative_tokens

        # 7. Modern Toggle Overrides (Reasoning & MTP)
        args["--reasoning"] = "on" if self.enable_thinking else "off"
        if self.use_mtp:
            args["--spec-type"] = "draft-mtp"

        return args
