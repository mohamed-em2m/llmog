from __future__ import annotations

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
    task: Literal["free_detection", "auto_label"] = "free_detection"

    # --- Logging -----------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Optional[str] = None

    # --- Input images ------------------------------------------------------
    # Detection mode: explicit image list (may be empty when relabeling from folders).
    images: List[str] = Field(default_factory=list)
    train_image: Optional[str] = None
    train_label: Optional[str] = None
    yaml_path: Optional[str] = None
    image_extensions: str = ".jpg,.jpeg,.png"

    # --- Categories --------------------------------------------------------
    categories: str = "person, car, bicycle, dog, cat"
    definitions: str = ""
    init_class_map: bool = False
    conf_threshold: Literal[1, 2, 3, 4, 5] = 2

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

    # --- Resume ------------------------------------------------------------
    resume: bool = False
    auto_resume: bool = True

    # --- Server / model ----------------------------------------------------
    model: str = "local-model"
    detector_model: str = "local-model"
    judge_model: str = "local-model"
    judge_url: Optional[str] = None
    api_key: str = "not-needed"
    base_url: str = "http://localhost:8080/v1"
    server_type: Literal["llama_cpp", "vllm", "external"] = "llama_cpp"
    max_workers: int = 1

    # llama.cpp-specific
    enable_thinking: bool = False
    use_mtp: bool = True
    ctx_size: int = 10000
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

    serving_extra: Dict[str, Any] = Field(default_factory=dict)
    # Raw extra command-line tokens forwarded verbatim to vLLM (list[str]).
    extra_args: Optional[List[str]] = None

    # ------------------------------------------------------------------ validators
    @field_validator("prep_tile_overlap")
    @classmethod
    def _check_overlap(cls, v: float) -> float:
        if not 0.0 <= v <= 0.5:
            raise ValueError("prep_tile_overlap must be between 0.0 and 0.5")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _check_gpu_mem(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        return v

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
        """
        if self.task == "free_detection" and not self.images:
            raise ValueError(
                "task='free_detection' requires at least one --image/-i path."
            )
        if self.task == "auto_label" and not self.train_image:
            raise ValueError(
                "task='auto_label' requires --train_image (folder of images)."
            )
        return self

    @property
    def vllm(self) -> Dict[str, Any]:
        """Build the kwarg dict consumed by :class:`VllmServerManager`.

        Always returns a freshly-built dict so the user-supplied
        ``serving_extra`` is never mutated in place (which would otherwise
        leak preview-specific overrides back into the config object).
        """
        args: Dict[str, Any] = dict(self.serving_extra or {})
        args["--tensor_split"] = self.tensor_parallel_size
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
