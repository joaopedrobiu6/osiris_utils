"""Lightweight, opt-in profiling utilities for osiris_utils.

All profiling is routed through the ``osiris_utils.profile`` logger at DEBUG
level, so it is **completely zero-overhead when not enabled** — no timing calls,
no memory queries, no string formatting.

HPC notes
---------
* Uses only :func:`time.perf_counter` (nanosecond resolution, stdlib, available
  on all POSIX and Windows systems including Marenostrum 5 / Deucalion).
* Memory tracking is optional and requires ``psutil``. If ``psutil`` is absent
  the context manager still works; it just omits the Δmem column.
* No MPI dependency — each rank profiles independently. Redirect logs to
  separate per-rank files with :func:`enable_profiling(logfile=...)` if needed.
* Thread-safe: :func:`time.perf_counter` is thread-safe; the logger is
  thread-safe by default.

Typical usage
-------------
::

    import osiris_utils as ou
    ou.enable_profiling()           # print timings to stderr

    sim = ou.Simulation("path/to/sim")
    with ou.profile_block("load e1"):
        sim["e1"].load_all()

    # Or instrument automatically via the built-in hooks:
    # Every load_all() / _read_index() already calls profile_block internally.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["enable_profiling", "disable_profiling", "profile_block"]

_profile_log = logging.getLogger("osiris_utils.profile")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enable_profiling(
    level: int = logging.DEBUG,
    logfile: str | None = None,
    fmt: str = "%(asctime)s [PROFILE] %(message)s",
) -> None:
    """Enable profiling output for all osiris_utils operations.

    Parameters
    ----------
    level : int
        Logging level for the profile logger (default ``logging.DEBUG``).
    logfile : str or None
        If given, write profiling output to this file instead of stderr.
        Useful on HPC to keep per-rank logs separate::

            enable_profiling(logfile=f"profile_rank{rank}.log")

    fmt : str
        Log record format string.

    Notes
    -----
    Calling this once at the start of your script is sufficient. The logger
    is process-global; call :func:`disable_profiling` to turn it off again.
    """
    _profile_log.setLevel(level)
    if _profile_log.handlers:
        return  # already configured
    if logfile:
        handler: logging.Handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    _profile_log.addHandler(handler)


def disable_profiling() -> None:
    """Disable profiling output (restore zero-overhead state)."""
    _profile_log.setLevel(logging.CRITICAL + 1)
    for h in list(_profile_log.handlers):
        _profile_log.removeHandler(h)
        h.close()


@contextmanager
def profile_block(label: str) -> Generator[None, None, None]:
    """Context manager that logs wall-clock time and optional memory delta.

    Zero overhead when the ``osiris_utils.profile`` logger is not at DEBUG
    level — no timing or memory calls are made at all.

    Parameters
    ----------
    label : str
        Human-readable label for this block, e.g. ``"load_all e1"``.

    Examples
    --------
    ::

        with profile_block("FFT e2"):
            fft_sim["e2"].load_all()
    """
    if not _profile_log.isEnabledFor(logging.DEBUG):
        yield
        return

    mem_before = _rss_mb()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0

    mem_after = _rss_mb()
    if mem_before is not None and mem_after is not None:
        mem_str = f"  Δmem={mem_after - mem_before:+.1f} MB  RSS={mem_after:.0f} MB"
    else:
        mem_str = ""

    _profile_log.debug("%s: %.3fs%s", label, elapsed, mem_str)


# ---------------------------------------------------------------------------
# Internal helpers (used by diagnostic.py instrumentation)
# ---------------------------------------------------------------------------


def _start_timer(label: str) -> tuple[str, float] | None:
    """Return ``(label, t0)`` if profiling is enabled, else ``None``.

    Use this together with :func:`_stop_timer` when you cannot use the
    ``with profile_block(...)`` form (e.g. the timed region spans an early
    return or a pre-existing ``with`` block).
    """
    if _profile_log.isEnabledFor(logging.DEBUG):
        return label, time.perf_counter()
    return None


def _stop_timer(token: tuple[str, float] | None) -> None:
    """Emit a DEBUG timing log if *token* was returned by :func:`_start_timer`."""
    if token is not None:
        label, t0 = token
        elapsed = time.perf_counter() - t0
        mem = _rss_mb()
        mem_str = f"  RSS={mem:.0f} MB" if mem is not None else ""
        _profile_log.debug("%s: %.3fs%s", label, elapsed, mem_str)


def _rss_mb() -> float | None:
    """Return current process RSS in MB, or None if psutil is unavailable."""
    try:
        import psutil  # optional; not required on HPC

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None
