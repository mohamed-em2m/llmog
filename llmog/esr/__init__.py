"""Real-ESRGAN upscaling for detection pipelines (torch-gated, CUDA-only).

Public surface:

* :class:`ESRConfig` -- runtime settings; build via ``ESRConfig.from_config``
  from a PipelineConfig, argparse Namespace, or plain dict.
* :func:`upscale_pil` -- upscale one PIL image per config; disabled configs
  return the input untouched (plus an ``applied: False`` info dict).
* :func:`torch_status` -- ``(ok, reason)`` for the GPU stack.
* :mod:`esr.registry` -- downloadable checkpoint table.
* :mod:`esr.project` -- original-size projection math.
"""

from esr.download import default_cache_dir, resolve_weights
from esr.manager import ESRConfig, ESRUpscaler, torch_status, upscale_pil
from esr.project import (
    esr_growth_factor,
    fit_long_edge,
    norm_to_orig_pixels,
    scaled_pixel_param,
    work_pixels_to_orig_pixels,
)
from esr.registry import (
    DEFAULT_ESR_MODEL,
    ESR_MODEL_CHOICES,
    ESR_MODELS,
    get_model_entry,
)

__all__ = [
    "DEFAULT_ESR_MODEL",
    "ESR_MODEL_CHOICES",
    "ESR_MODELS",
    "ESRConfig",
    "ESRUpscaler",
    "default_cache_dir",
    "esr_growth_factor",
    "fit_long_edge",
    "get_model_entry",
    "norm_to_orig_pixels",
    "resolve_weights",
    "scaled_pixel_param",
    "torch_status",
    "upscale_pil",
    "work_pixels_to_orig_pixels",
]
