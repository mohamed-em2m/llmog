"""Client retry and network utility helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger("detection_pipeline")


def _call_with_retries(
    fn: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 1.5,
    what: str = "API call",
) -> Any:
    """Execute an API call with exponential backoff retry logic."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("%s failed (attempt %d/%d): %s", what, attempt, retries, exc)
            if attempt < retries:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"{what} failed after {retries} attempts") from last_exc
