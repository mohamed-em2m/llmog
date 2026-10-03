"""Weight resolution for the ESR upscaler: local path -> cache -> download.

Resolution order for a checkpoint:

  1. ``--esr_model_path`` (explicit local file) -- always wins.
  2. ``LLMOG_ESR_MODEL`` env var (explicit local file, CI/Kaggle friendly).
  3. Cache hit under the ESR cache dir (``~/.cache/llmog/esr/``,
     overridable via ``LLMOG_CACHE_DIR`` or ``--esr_cache_dir``).
  4. Auto-download from the official GitHub release asset
     (see :mod:`esr.registry`); skips when the file already exists.
  5. Optional HuggingFace fallback: only when ``--esr_model_repo`` names a
     repo AND ``huggingface_hub`` is importable (never a hard dependency).

Everything here is torch-free so weight handling stays unit-testable on
machines without a GPU stack.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from esr.registry import get_model_entry

logger = logging.getLogger("llmog.esr")


def default_cache_dir() -> Path:
    """Return the ESR weight cache dir (``~/.cache/llmog/esr`` by default)."""
    base = os.environ.get("LLMOG_CACHE_DIR")
    if base:
        return Path(base) / "esr"
    # XDG_CACHE_HOME on Linux, ~/.cache fallback elsewhere (incl. Windows).
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "llmog" / "esr"


def _existing_file(path: str | Path | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser()
    return p if p.is_file() and p.stat().st_size > 0 else None


def resolve_weights(
    model_key: str = "general-x4v3",
    model_path: str | None = None,
    model_repo: str | None = None,
    cache_dir: str | Path | None = None,
    auto_download: bool = True,
) -> Path:
    """Resolve a local checkpoint file, downloading it when needed.

    Returns the checkpoint :class:`Path`. Raises ``FileNotFoundError`` with
    an actionable message when nothing resolves and ``auto_download`` is
    off (or the download fails).
    """
    # 1-2. Explicit local files always win (flag beats env).
    explicit = _existing_file(model_path) or _existing_file(
        os.environ.get("LLMOG_ESR_MODEL")
    )
    if explicit is not None:
        logger.info("ESR weights: using local file %s", explicit)
        return explicit
    if model_path or os.environ.get("LLMOG_ESR_MODEL"):
        wanted = model_path or os.environ.get("LLMOG_ESR_MODEL")
        raise FileNotFoundError(
            f"ESR weights not found: {wanted!r} (via "
            f"{'--esr_model_path' if model_path else 'LLMOG_ESR_MODEL'}). "
            "Place the checkpoint there or unset it to use the cache/download."
        )

    entry = get_model_entry(model_key)
    filename = str(entry["file"])
    cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    dest = cache / filename

    # 3. Cache hit.
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("ESR weights: cache hit %s", dest)
        return dest

    # 4. Official download.
    if auto_download:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            _download(str(entry["url"]), dest)
            return dest
        except Exception as exc:
            logger.warning("ESR weights: official download failed: %s", exc)

    # 5. HuggingFace fallback (opt-in, optional dependency).
    if model_repo:
        hf_path = _download_hf(model_repo, filename, cache)
        if hf_path is not None:
            return hf_path

    raise FileNotFoundError(
        f"ESR weights for {model_key!r} are not available locally ({dest}) "
        "and the download failed. Options: re-run with network access, "
        "place the file manually and pass --esr_model_path, or set "
        "--esr_model_repo to a HuggingFace repo holding "
        f"{filename} (needs huggingface_hub installed)."
    )


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` atomically (temp file + os.replace)."""
    logger.info("ESR weights: downloading %s -> %s", url, dest)
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "ESR weights: downloading needs httpx (a core dependency) -- "
            f"it is not importable: {exc}"
        ) from exc

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0) or 0)
            done = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
            if total and done != total:
                raise IOError(f"short download: got {done} of {total} bytes for {url}")
        os.replace(tmp, dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    logger.info("ESR weights: saved %s (%d bytes)", dest, dest.stat().st_size)


def _download_hf(repo: str, filename: str, cache: Path) -> Path | None:
    """Try a HuggingFace repo download; ``None`` when unavailable/failed."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "ESR weights: --esr_model_repo was given but huggingface_hub is "
            "not installed; skipping the HF fallback."
        )
        return None
    try:
        snap = snapshot_download(
            repo_id=repo,
            allow_patterns=[filename],
            cache_dir=str(cache / "_hf"),
        )
        candidate = Path(snap) / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            logger.info("ESR weights: HF fallback resolved %s", candidate)
            return candidate
    except Exception as exc:
        logger.warning("ESR weights: HF fallback failed: %s", exc)
    return None
