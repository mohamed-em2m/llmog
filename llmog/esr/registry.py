"""Real-ESRGAN weight registry.

Single source of truth for downloadable super-resolution checkpoints, shared
by the pipeline (``esr.download``) and ``scripts/download_real_esrgan.sh``.

Primary source is always the official xinntao/Real-ESRGAN GitHub release
assets (stable, versioned, no auth) -- the same URLs upstream docs,
SD-WebUI and chaiNNer use. HuggingFace mirrors exist but are unofficial /
unpinned, so HF is only a fallback via an explicit ``--esr_model_repo``.
"""

from __future__ import annotations

# Key -> (file name, download URL, native scale, size MB, description).
# Tags verified against the upstream release pages:
#   v0.2.5.0 carries general-x4v3 / general-wdn-x4v3 / animevideov3,
#   v0.1.0 carries RealESRGAN_x4plus, v0.2.2.4 carries the anime 6B,
#   v0.2.1 carries RealESRGAN_x2plus.
ESR_MODELS: dict[str, dict[str, object]] = {
    "general-x4v3": {
        "file": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale": 4,
        "size_mb": 17,
        "description": (
            "Tiny SRVGGNetCompact for general scenes; variable 1-4x output "
            "scale; lowest VRAM/time. Default."
        ),
    },
    "general-wdn-x4v3": {
        "file": "realesr-general-wdn-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "scale": 4,
        "size_mb": 17,
        "description": (
            "Denoise-strength companion to general-x4v3 (needs the -dn "
            "control, not exposed by this pipeline; listed for manual use)."
        ),
    },
    "animevideov3": {
        "file": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "size_mb": 17,
        "description": "Anime-video v3 model; use for animation/clean-line art.",
    },
    "x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "size_mb": 64,
        "description": ("Full RRDBNet; best still-image quality, slower and heavier."),
    },
    "x4plus-anime-6B": {
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "size_mb": 18,
        "description": "Anime stills (6-block RRDBNet).",
    },
    "x2plus": {
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "size_mb": 64,
        "description": "Native 2x; cheaper than 4x + downscale when 2x suffices.",
    },
}

DEFAULT_ESR_MODEL = "general-x4v3"

#: Valid values for the ``esr_model`` config field / ``--esr_model`` flag.
ESR_MODEL_CHOICES = tuple(ESR_MODELS)


def get_model_entry(key: str) -> dict[str, object]:
    """Return the registry entry for ``key`` or raise a helpful ``KeyError``."""
    try:
        return ESR_MODELS[key]
    except KeyError:
        valid = ", ".join(ESR_MODEL_CHOICES)
        raise KeyError(
            f"Unknown ESR model {key!r} (expected one of: {valid}). "
            "For any other spandrel-compatible checkpoint, pass "
            "--esr_model_path instead."
        ) from None
