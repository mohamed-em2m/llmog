"""Circuit breaker against a dead / OOM inference server.

Problem this solves: when the vLLM / llama.cpp server dies mid-run (crash,
OOM, connection reset), every model call raises. The old code caught each
failure per-box with ``continue`` and then wrote an *empty* YOLO label file
for the image and marked it completed in the checkpoint. A resumed run would
skip those images forever, silently baking fake "no objects" labels into the
dataset.

This module provides:

* :class:`ServerDownError` -- raised to abort the whole run as soon as the
  server is judged dead, so nothing more is written or checkpointed.
* :func:`is_server_error` -- classifies an exception as "server is sick"
  (connection / timeout / 5xx / OOM / CUDA) vs. a per-box content problem
  (bad JSON for one crop). Only server-class errors trip the breaker.
* :class:`FailureTracker` -- thread-safe consecutive-server-failure counter
  shared across worker threads. Any success resets it.
* :func:`probe_server` -- best-effort liveness check (``models.list()``)
  used between batches when failures were seen.
"""

from __future__ import annotations

import threading

from auto_annotation.logging_utils import logger


class ServerDownError(RuntimeError):
    """Raised when the inference server is judged dead/OOM: abort the run."""


# Substrings (lowercased) in an exception message / status that mean the
# *server* is sick rather than one crop being unclassifiable. Covers:
# - dead process / network: connection refused/reset/aborted, unreachable
# - overloaded / gone: 500/502/503/504, service unavailable, overload
# - OOM (vLLM + llama.cpp + CUDA flavours): out of memory, oom, kv cache,
#   cuda oom, allocation failed, resource exhausted
_SERVER_ERROR_HINTS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection error",
    "failed to connect",
    "connecterror",
    "connectionerror",
    "connection timed out",
    "timed out",
    "timeout",
    "server disconnected",
    "remote disconnected",
    "broken pipe",
    "network is unreachable",
    "no route to host",
    "name resolution",
    "temporary failure in name resolution",
    "service unavailable",
    "server error",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "overloaded",
    "server is busy",
    "too many requests",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    " 500 ",
    " 502 ",
    " 503 ",
    " 504 ",
    "out of memory",
    "out-of-memory",
    "oom",
    "kv cache",
    "kv-cache",
    "kvcache",
    "cuda out of memory",
    "cuda error",
    "allocation failed",
    "resource exhausted",
    "memory limit",
    "cannot allocate memory",
    "no memory",
    "model is still loading",
    "model not loaded",
    "no model loaded",
    "server is not ready",
    "llama server",
)

# OpenAI-SDK style error class names that always mean transport/server trouble
# (matched by class name so no SDK import is needed here).
_SERVER_ERROR_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "NewConnectionError",
        "MaxRetryError",
    }
)


def is_server_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a dead/overloaded/OOM server, not a bad box."""
    if exc is None:
        return False
    # Walk the MRO names: openai.APIConnectionError subclasses
    # httpx/connect errors, so a name match at any level counts.
    for klass in type(exc).__mro__:
        if klass.__name__ in _SERVER_ERROR_TYPES:
            return True
    # HTTP status carriers (openai.APIStatusError, httpx.HTTPStatusError, ...).
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    response = getattr(exc, "response", None)
    resp_status = getattr(response, "status_code", None)
    if isinstance(resp_status, int) and resp_status >= 500:
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(hint in haystack for hint in _SERVER_ERROR_HINTS)


class FailureTracker:
    """Thread-safe consecutive-server-failure counter (circuit breaker).

    Any successful model call resets the streak. When the streak reaches
    ``max_consecutive_failures`` the breaker trips and stays tripped --
    callers raise :class:`ServerDownError` to abort the run instead of
    writing fake-empty labels for every remaining image.
    """

    def __init__(self, max_consecutive_failures: int = 20):
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._lock = threading.Lock()
        self._consecutive = 0
        self._total = 0
        self._tripped = False

    @property
    def consecutive(self) -> int:
        with self._lock:
            return self._consecutive

    @property
    def total(self) -> int:
        """Total server-class failures ever recorded (never reset)."""
        with self._lock:
            return self._total

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def record_failure(self) -> bool:
        """Record one server-class failure. Returns True if now tripped."""
        with self._lock:
            self._consecutive += 1
            self._total += 1
            if self._consecutive >= self.max_consecutive_failures:
                self._tripped = True
            return self._tripped

    def check_and_raise(self, context: str = "") -> None:
        """Raise :class:`ServerDownError` if the breaker has tripped."""
        with self._lock:
            tripped = self._tripped
            consecutive = self._consecutive
        if tripped:
            where = f" while {context}" if context else ""
            raise ServerDownError(
                f"Inference server appears dead/OOM ({consecutive} consecutive "
                f"server failures{where}). Aborting the run so no fake-empty "
                "labels are written -- fix the server and resume with the "
                "same command (auto-resume skips finished images)."
            )


def probe_server(client, timeout: float = 10.0) -> bool:
    """Best-effort liveness check: True if the server answers ``models.list``."""
    try:
        client.models.list(timeout=timeout)
        return True
    except TypeError:
        # Older openai SDK without per-call timeout kwarg.
        try:
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Server liveness probe failed: {e}")
            return False
    except Exception as e:
        logger.warning(f"Server liveness probe failed: {e}")
        return False
