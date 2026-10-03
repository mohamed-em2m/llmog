"""CUDA ESR upscaler: lazy singleton around a spandrel checkpoint.

Heavy stack (torch) is imported lazily so ``import esr.manager`` never fails
on machines without it -- only enabling ESR (``esr_enabled=True``) requires
``uv pip install -e .[esrgan]`` plus a CUDA device. CPU is refused on
purpose: ESRGAN on CPU is unusably slow and almost always a mistake.

Thread-safe: auto_label runs workers in a ThreadPoolExecutor, so GPU
inference is serialized behind a lock.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from esr.download import default_cache_dir, resolve_weights
from esr.project import fit_long_edge, tile_geometry
from esr.registry import DEFAULT_ESR_MODEL

logger = logging.getLogger("llmog.esr")


def torch_status() -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the ESR GPU stack."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return (
            False,
            "torch is not installed (install with: uv pip install -e .[esrgan])",
        )
    try:
        import spandrel  # noqa: F401
    except ImportError:
        return (
            False,
            "spandrel is not installed (it is a core dependency -- reinstall llmog)",
        )
    import torch

    if not torch.cuda.is_available():
        return False, "torch has no CUDA device available"
    return True, "ok"


@dataclass
class ESRConfig:
    """Runtime ESR settings (built from PipelineConfig/Namespace/dict)."""

    enabled: bool = False
    model: str = DEFAULT_ESR_MODEL
    model_path: Optional[str] = None
    model_repo: Optional[str] = None
    cache_dir: Optional[str] = None
    scale: Optional[int] = None  # None = checkpoint native scale
    target_long_edge: int = 2048  # 0 = keep native ESR output
    max_long_edge: int = 4096  # VRAM guard; Lanczos downscale past it
    tile_size: int = 512
    overlap: int = 16
    batch_size: int = 4
    for_crops: bool = False
    compile: bool = False
    channels_last: bool = True
    device: str = "auto"

    @classmethod
    def from_config(cls, obj: Any) -> "ESRConfig":
        """Build from a PipelineConfig, argparse Namespace, or plain dict."""

        def _get(name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        def _int(name: str, default: int) -> int:
            v = _get(name, default)
            # Explicit None falls back (unset); any other value is honored
            # verbatim (including 0, which disables the corresponding cap).
            return default if v is None else int(v)

        _scale_raw = _get("esr_scale")
        return cls(
            enabled=bool(_get("esr_enabled", False)),
            model=str(_get("esr_model", DEFAULT_ESR_MODEL) or DEFAULT_ESR_MODEL),
            model_path=_get("esr_model_path"),
            model_repo=_get("esr_model_repo"),
            cache_dir=_get("esr_cache_dir"),
            scale=None if _scale_raw is None else int(_scale_raw),
            target_long_edge=_int("esr_target_long_edge", 2048),
            max_long_edge=_int("esr_max_long_edge", 4096),
            tile_size=_int("esr_tile_size", 512),
            overlap=_int("esr_overlap", 16),
            batch_size=_int("esr_batch_size", 4),
            for_crops=bool(_get("esr_for_crops", False)),
            compile=bool(_get("esr_compile", False)),
            channels_last=bool(_get("esr_channels_last", True)),
            device=str(_get("esr_device", "auto") or "auto"),
        )

    def fingerprint(self) -> tuple:
        """Hashable identity of this config (singleton key)."""
        import dataclasses

        return dataclasses.astuple(self)


class ESRUpscaler:
    """Lazy, thread-safe Real-ESRGAN upscaler (one instance per config)."""

    _instances: dict[tuple, "ESRUpscaler"] = {}
    _instance_lock = threading.Lock()

    def __init__(self, cfg: ESRConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._model: Any = None
        self._device: Any = None
        self._native_scale: int = 0
        self._weights: Optional[Path] = None

    @classmethod
    def get(cls, cfg: ESRConfig) -> "ESRUpscaler":
        """Process-wide instances keyed by config fingerprint.

        Distinct configs coexist (each loads its own model); identical
        configs share one. Previously the first config silently won, which
        dropped setting changes between runs in the same process.
        """
        key = cfg.fingerprint()
        with cls._instance_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(cfg)
                cls._instances[key] = inst
                if len(cls._instances) > 1:
                    logger.warning(
                        "ESR: %d distinct upscaler configs live in this "
                        "process (each holds its own GPU model).",
                        len(cls._instances),
                    )
            return inst

    @classmethod
    def reset(cls) -> None:
        """Drop all cached instances (tests / config changes between runs)."""
        with cls._instance_lock:
            cls._instances.clear()

    # -- setup ------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ok, reason = torch_status()
        if not ok:
            raise RuntimeError(f"ESR upscaling is enabled but unusable: {reason}.")
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        cfg = self.cfg
        self._weights = resolve_weights(
            model_key=cfg.model,
            model_path=cfg.model_path,
            model_repo=cfg.model_repo,
            cache_dir=cfg.cache_dir or default_cache_dir(),
        )
        device = self._resolve_device(torch)
        self._device = device
        loader = ModelLoader(device=device)
        model = loader.load_from_file(str(self._weights))
        if not isinstance(model, ImageModelDescriptor):
            raise TypeError(
                f"ESR weights {self._weights} did not load as an image "
                f"super-resolution model (got {type(model).__name__})."
            )
        model.to(device)
        model.eval()
        self._native_scale = int(model.scale)
        if cfg.scale is not None and int(cfg.scale) != self._native_scale:
            raise ValueError(
                f"--esr_scale {cfg.scale} mismatches checkpoint native scale "
                f"{self._native_scale} ({self._weights}). Omit --esr_scale to "
                "use the native factor."
            )
        if cfg.channels_last:
            # NOTE: channels-last goes on the wrapped nn.Module, NOT the
            # spandrel descriptor: ModelDescriptor.to() only accepts the
            # plain device/dtype positionals and raises TypeError on
            # memory_format (seen live on Kaggle).
            model.model.to(memory_format=torch.channels_last)
        if cfg.compile:
            model = torch.compile(model)
            dummy = torch.zeros(
                max(1, cfg.batch_size),
                3,
                cfg.tile_size,
                cfg.tile_size,
                device=device,
            )
            if cfg.channels_last:
                dummy = dummy.to(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                model(dummy)
            torch.cuda.synchronize()
        n_params = sum(p.numel() for p in model.model.parameters())
        logger.info(
            "ESR model ready: %s scale=x%d params=%.1fM device=%s channels_last=%s compiled=%s",
            self._weights.name,
            self._native_scale,
            n_params / 1e6,
            device,
            cfg.channels_last,
            cfg.compile,
        )
        self._model = model

    def _resolve_device(self, torch: Any) -> Any:
        spec = (self.cfg.device or "auto").strip()
        if spec == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "ESR upscaling needs CUDA but torch reports no CUDA "
                    "device. CPU ESRGAN is refused (unusably slow)."
                )
            return torch.device("cuda:0")
        dev = torch.device(spec)
        if dev.type != "cuda":
            raise RuntimeError(
                f"ESR upscaling needs a CUDA device, got {spec!r}. "
                "CPU ESRGAN is refused (unusably slow)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("ESR upscaling needs CUDA but none is available.")
        return dev

    @property
    def native_scale(self) -> int:
        with self._lock:
            self._ensure_loaded()
            return self._native_scale

    # -- public API -------------------------------------------------------
    def upscale_pil(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """Upscale ``image`` per config; returns ``(working_image, esr_info)``.

        ``esr_info`` always carries ``applied`` plus the
        ``orig -> ESR -> cap -> target`` size chain so callers can project
        boxes back onto the original dims (see :mod:`esr.project`).
        """
        cfg = self.cfg
        orig_w, orig_h = image.size
        info: dict[str, Any] = {
            "applied": False,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "pre_w": orig_w,
            "pre_h": orig_h,
            "up_w": orig_w,
            "up_h": orig_h,
            "work_w": orig_w,
            "work_h": orig_h,
            "scale": 1,
        }
        if not cfg.enabled:
            return image, info
        with self._lock:
            self._ensure_loaded()
            import torch

            rgb = image.convert("RGB")
            w0, h0 = rgb.size
            # Pre-ESR input cap: the post-ESR long-edge guard alone cannot
            # prevent OOM, because the full-res blend canvas is allocated
            # DURING inference. Fitting the input to cap/scale first keeps
            # the ESR output <= cap by construction.
            native = self._native_scale or 1
            if cfg.max_long_edge > 0:
                in_cap = max(1, cfg.max_long_edge // max(1, native))
                if max(w0, h0) > in_cap:
                    nw, nh = fit_long_edge(w0, h0, in_cap)
                    logger.warning(
                        "ESR input %dx%d exceeds pre-cap %d "
                        "(max_long_edge %d / x%d); Lanczos downscaling to "
                        "%dx%d before ESR.",
                        w0,
                        h0,
                        in_cap,
                        cfg.max_long_edge,
                        native,
                        nw,
                        nh,
                    )
                    rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
                    w0, h0 = nw, nh
            arr = np.asarray(rgb).astype(np.float32) / 255.0
            # Fast path: fits in one native tile.
            if w0 <= cfg.tile_size and h0 <= cfg.tile_size:
                out = self._infer(arr)
            else:
                out = self._infer_tiled(arr)
            torch.cuda.synchronize()
        up = Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))
        up_w, up_h = up.size
        # Belt-and-braces post guard (normally a no-op after the pre-cap),
        # then the working target size (clamped to the cap so the guard is
        # a real upper bound even when target > cap).
        cap_w, cap_h = fit_long_edge(up_w, up_h, cfg.max_long_edge)
        if (cap_w, cap_h) != (up_w, up_h):
            logger.warning(
                "ESR output %dx%d exceeds --esr_max_long_edge %d; Lanczos "
                "downscaling to %dx%d.",
                up_w,
                up_h,
                cfg.max_long_edge,
                cap_w,
                cap_h,
            )
            up = up.resize((cap_w, cap_h), Image.Resampling.LANCZOS)
        target = cfg.target_long_edge
        if target > 0 and cfg.max_long_edge > 0:
            target = min(target, cfg.max_long_edge)
        work_w, work_h = fit_long_edge(*up.size, target)
        if (work_w, work_h) != up.size:
            up = up.resize((work_w, work_h), Image.Resampling.LANCZOS)
        info.update(
            applied=True,
            pre_w=w0,
            pre_h=h0,
            up_w=up_w,
            up_h=up_h,
            work_w=work_w,
            work_h=work_h,
            scale=native,
            weights=str(self._weights) if self._weights else "",
        )
        logger.debug(
            "ESR upscale: %dx%d -> pre %dx%d -> ESR x%d -> working %dx%d",
            orig_w,
            orig_h,
            w0,
            h0,
            native,
            work_w,
            work_h,
        )
        return up, info

    # -- inference core (adapted from scripts/run_spandrel.py) ------------
    def _infer(self, arr: np.ndarray) -> np.ndarray:
        import torch

        cfg = self.cfg
        assert self._model is not None and self._device is not None
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        if cfg.channels_last:
            t = t.to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            out = self._model(t)
        return out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()

    def _infer_tiled(self, arr: np.ndarray) -> np.ndarray:
        import torch

        cfg = self.cfg
        assert self._model is not None and self._device is not None
        h, w, _ = arr.shape
        # Effective tile dims clamp to the image: a narrow image (e.g.
        # 600x100 with tile 512) yields short-but-full tiles that match a
        # rectangular feather mask, instead of crashing on a square one.
        coords, th, tw = tile_geometry(h, w, cfg.tile_size, max(0, cfg.overlap))
        scale = self._native_scale
        canvas = torch.zeros(
            (3, h * scale, w * scale), dtype=torch.float16, device=self._device
        )
        weight = torch.zeros(
            (1, h * scale, w * scale), dtype=torch.float32, device=self._device
        )
        tiles = torch.stack(
            [
                torch.from_numpy(arr[y : y + th, x : x + tw]).permute(2, 0, 1)
                for (y, x) in coords
            ]
        ).pin_memory()
        mask = self._feather_mask(th * scale, tw * scale, max(0, cfg.overlap) * scale)
        mask = mask.to(self._device)
        bs = max(1, cfg.batch_size)
        for start in range(0, len(coords), bs):
            end = min(start + bs, len(coords))
            batch = tiles[start:end].to(self._device, non_blocking=True)
            if cfg.channels_last:
                batch = batch.to(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                out = self._model(batch)
            for i, (y, x) in enumerate(coords[start:end]):
                oy, ox = y * scale, x * scale
                patch = out[i].to(torch.float16)
                canvas[:, oy : oy + patch.shape[1], ox : ox + patch.shape[2]].addcmul_(
                    patch, mask
                )
                weight[:, oy : oy + patch.shape[1], ox : ox + patch.shape[2]] += mask
            del batch, out
        result = (canvas.float() / weight.clamp_min(1e-6)).clamp(0, 1)
        return result.permute(1, 2, 0).cpu().numpy()

    _mask_cache: dict[tuple[int, int, int], Any] = {}

    def _feather_mask(self, th_hr: int, tw_hr: int, overlap_hr: int) -> Any:
        """Rectangular feather mask (1.0 interior, ~0.5 at the rim)."""
        import torch

        key = (th_hr, tw_hr, overlap_hr)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = np.ones((th_hr, tw_hr), dtype=np.float32)
        o = max(0, min(overlap_hr, th_hr // 2, tw_hr // 2))
        for i in range(o):
            wgt = 0.5 + 0.5 * (i + 1) / o
            mask[i, :] = wgt
            mask[th_hr - 1 - i, :] = wgt
            mask[:, i] = wgt
            mask[tw_hr - 1 - i, :] = wgt
        t = torch.from_numpy(mask).unsqueeze(0)
        self._mask_cache[key] = t
        return t


def upscale_pil(
    image: Image.Image, cfg: ESRConfig
) -> tuple[Image.Image, dict[str, Any]]:
    """One-shot helper: disabled configs return the input untouched."""
    if not cfg.enabled:
        w, h = image.size
        return image, {
            "applied": False,
            "orig_w": w,
            "orig_h": h,
            "pre_w": w,
            "pre_h": h,
            "up_w": w,
            "up_h": h,
            "work_w": w,
            "work_h": h,
            "scale": 1,
        }
    return ESRUpscaler.get(cfg).upscale_pil(image)
