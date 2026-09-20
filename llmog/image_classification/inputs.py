"""Collect whole-image classification inputs from --image and/or --input_folder."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple


def _parse_extensions(raw: str) -> Tuple[str, ...]:
    exts = []
    for token in (raw or "").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        exts.append(token)
    return tuple(exts) or (".jpg", ".jpeg", ".png")


def collect_classify_images(
    images: List[str] | None,
    input_folder: str | None,
    image_extensions: str = ".jpg,.jpeg,.png",
    shuffle: bool = False,
    seed: int = 42,
    start_index: int | None = None,
    end_index: int | None = None,
    num_samples: int | None = None,
) -> List[Path]:
    """Merge explicit -i paths and --input_folder scans into an ordered list.

    Order: explicit images first (in CLI order), then folder scan (sorted).
    Sampling (shuffle -> slice -> num_samples) mirrors auto_label so dataset
    splits stay reproducible.
    """
    collected: List[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path.absolute())
        if key not in seen:
            seen.add(key)
            collected.append(path)

    for raw in images or []:
        p = Path(raw)
        if not p.is_file():
            continue
        _add(p)

    folder_paths: List[Path] = []
    if input_folder:
        folder = Path(input_folder)
        if folder.is_dir():
            exts = _parse_extensions(image_extensions)
            for entry in sorted(folder.iterdir()):
                if entry.is_file() and entry.suffix.lower() in exts:
                    folder_paths.append(entry)
        # Shuffle/slice/sample apply to the COMBINED list below, so a folder
        # run and a mixed -i+folder run behave identically.

    for p in folder_paths:
        _add(p)

    ordered = list(collected)
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(ordered)

    if start_index is not None or end_index is not None:
        start = start_index or 0
        end = end_index if end_index is not None else len(ordered)
        ordered = ordered[start:end]

    if num_samples is not None:
        ordered = ordered[:num_samples]

    return ordered
