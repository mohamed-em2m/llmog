"""Client retry and network utility helpers with 429-aware backoff."""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Callable

logger = logging.getLogger("detection_pipeline")

# Global rate-limit for Gemini free tier: 15 RPM => at least 4s between calls
_LAST_GEMINI_CALL: float = 0.0
_GEMINI_MIN_INTERVAL = 4.0


def _extract_retry_delay(exc: Exception) -> float | None:
    """Parse Retry-After / RetryInfo delay from 429 error message."""
    msg = str(exc)
    # e.g. 'retryDelay': '1s'  or  'Please retry in 1.911154742s.'
    m = re.search(r"retryDelay[^0-9]*(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Try response headers
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers:
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra:
                return float(str(ra).strip().rstrip("s"))
    except Exception:
        pass
    return None


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "quota" in msg
        or "resource_exhausted" in msg
        or "rate limit" in msg
    )


def _call_with_retries(
    fn: Callable[[], Any],
    *,
    retries: int = 5,
    base_delay: float = 2.0,
    what: str = "API call",
) -> Any:
    """Execute an API call with 429-aware exponential backoff + jitter.

    - Respects RetryInfo/retryDelay from Gemini 429 details.
    - Adds global Gemini rate-limit (4s interval) to avoid 15 RPM burst.
    - Uses jitter and cap to avoid thundering herd.
    """
    global _LAST_GEMINI_CALL
    last_exc = None
    for attempt in range(1, retries + 1):
        # Pre-call global rate-limit for Gemini free tier
        # Cheap heuristic: if error message will contain gemini, we rate-limit anyway
        # to avoid burst; 4s interval ensures <=15 RPM even with concurrency.
        now = time.time()
        # Always enforce a tiny gap to smooth bursts, but only strict for quota errors
        # we do it pre-emptively for all calls to reduce 429 likelihood
        wait_needed = _LAST_GEMINI_CALL + _GEMINI_MIN_INTERVAL - now
        if wait_needed > 0 and attempt == 1:
            # Only sleep on first attempt if we are too close to last call
            # Subsequent retries already have backoff, so don't double-sleep
            logger.debug(
                "Gemini rate-limit: sleeping %.2fs before %s", wait_needed, what
            )
            time.sleep(wait_needed + random.uniform(0, 0.3))

        try:
            result = fn()
            _LAST_GEMINI_CALL = time.time()
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            is_quota = _is_quota_error(exc)
            # Determine delay
            retry_delay = _extract_retry_delay(exc)
            if retry_delay is not None:
                delay = retry_delay + random.uniform(0.1, 0.5)
                # Cap at 60s
                delay = min(delay, 60.0)
                logger.warning(
                    "%s failed (attempt %d/%d) with quota 429 – retrying in %.1fs: %s",
                    what,
                    attempt,
                    retries,
                    delay,
                    exc,
                )
            elif is_quota:
                # No explicit delay, use exponential backoff with longer base for quota
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                delay = min(delay, 60.0)
                logger.warning(
                    "%s failed (attempt %d/%d) with quota error – backoff %.1fs: %s",
                    what,
                    attempt,
                    retries,
                    delay,
                    exc,
                )
            else:
                logger.warning(
                    "%s failed (attempt %d/%d): %s", what, attempt, retries, exc
                )
                delay = base_delay * attempt + random.uniform(0, 0.5)

            if attempt < retries:
                time.sleep(delay)
            else:
                # Update last call time even on failure to avoid immediate retry storm
                _LAST_GEMINI_CALL = time.time()
    raise RuntimeError(f"{what} failed after {retries} attempts") from last_exc
