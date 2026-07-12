"""Fast parallel export of OSIRIS diagnostics to NumPy .npy files.

Design principles
-----------------
* **Works for any number of timesteps** — output is a memory-mapped
  ``np.lib.format.open_memmap`` array written directly to disk; peak RAM
  is ``chunk_size x frame_size``, not ``n_frames x frame_size``.  5 000
  timesteps uses the same RAM as 50.

* **Parallel I/O** via ``ThreadPoolExecutor``.  h5py releases the GIL
  during reads, so threads genuinely overlap I/O latency — essential on
  HPC parallel filesystems (Lustre / GPFS).

* **Streaming reductions** — spatial and time averages are computed
  on-the-fly without ever holding the full dataset in memory.

Reduction modes (``reduce_axis`` / ``time_average``)
-----------------------------------------------------
For a 2-D field of shape ``(nx1, nx2)`` per frame:

=============================  ======================  ================
``reduce_axis``                ``time_average``        Output shape
=============================  ======================  ================
``None``                       ``False`` (default)     ``(n_t, nx1, nx2)``
``0``  (average over x1)       ``False``               ``(n_t, nx2)``
``1``  (average over x2)       ``False``               ``(n_t, nx1)``
``(0, 1)`` (average both)      ``False``               ``(n_t,)``
``None``                       ``True``                ``(nx1, nx2)``
``0``                          ``True``                ``(nx2,)``
``1``                          ``True``                ``(nx1,)``
``(0, 1)``                     ``True``                scalar ``()``
=============================  ======================  ================

``reduce_axis`` indices are relative to the *spatial* axes of a single
frame (i.e. axis 0 = x1, axis 1 = x2 for a 2-D simulation).

Typical usage
-------------
Full export (scales to any n_timesteps)::

    import osiris_utils as ou
    sim = ou.Simulation("path/to/os-stdin")
    ou.export_to_npy(sim["e1"], "output/e1.npy")

Average over x1 for every timestep (shape ``(n_t, nx2)``)::

    ou.export_to_npy(sim["e1"], "output/e1_x1avg.npy", reduce_axis=0)

Time-average over all timesteps (shape ``(nx1, nx2)``)::

    ou.export_to_npy(sim["e1"], "output/e1_tavg.npy", time_average=True)

Time-average + spatial average (shape ``(nx2,)``)::

    ou.export_to_npy(sim["e1"], "output/e1_x1avg_tavg.npy",
                     reduce_axis=0, time_average=True)

Export multiple quantities at once::

    ou.export_simulation_to_npy(
        sim,
        quantities=["e1", "e2", "b3"],
        output_folder="output/fields/",
    )
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tqdm

if TYPE_CHECKING:
    from ..data.diagnostic import Diagnostic
    from ..data.simulation import Simulation

# Beyond ~32 threads, Lustre MDS contention dominates on HPC clusters.
_MAX_EXPORT_WORKERS = 32

# WeakValueDictionary so locks are GC'd when their diagnostic is GC'd,
# preventing unbounded growth for long-running scripts with many diagnostics.
import weakref
_cache_lock_registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_registry_lock = threading.Lock()


def _get_cache_lock(diagnostic) -> threading.Lock:
    """Return (creating if needed) a per-diagnostic-identity Lock."""
    key = id(diagnostic)
    with _registry_lock:
        lock = _cache_lock_registry.get(key)
        if lock is None:
            lock = threading.Lock()
            _cache_lock_registry[key] = lock
        return lock


def _read_frame(args: tuple) -> tuple[int, np.ndarray]:
    """Worker: read one frame and optionally reduce spatial axes.

    Uses ``_frame()`` — the universal interface for all diagnostic types
    (base file-backed and derived via arithmetic / post-processing).

    Locking strategy
    ----------------
    * No lock when ``_frame_cache`` is absent (the common case for base
      diagnostics): each thread opens its own HDF5 handle with no shared
      state, so reads are fully concurrent.
    * Per-diagnostic lock only when ``_frame_cache`` is present: guards the
      check-compute-insert triplet to prevent duplicate work and dict races
      under concurrent access.

    Spatial reduction is done inside the thread so that only the
    (much smaller) reduced array crosses the thread boundary.
    """
    diagnostic, index, reduce_axis = args

    cache = getattr(diagnostic, "_frame_cache", None)
    if cache is not None:
        lock = _get_cache_lock(diagnostic)
        with lock:
            arr_found = index in cache
            if arr_found:
                arr = cache[index]
        if not arr_found:
            computed = diagnostic._frame(index)  # heavy work outside the lock
            with lock:
                if index not in cache:            # another thread may have inserted it
                    cache[index] = computed
                arr = cache[index]
    else:
        arr = diagnostic._frame(index)

    if reduce_axis is not None:
        arr = arr.mean(axis=reduce_axis)
    return index, arr


# -----------------------------------------------------------------------
# Derived-diagnostic helpers
# -----------------------------------------------------------------------

def _is_derived(diagnostic) -> bool:
    """Return True when frames come from computation rather than direct HDF5 reads.

    Covers:
    * Arithmetic results: ``_frame`` set as an instance attribute by ``_binary_op``
    * Post-processing subclasses: ``Derivative_Diagnostic``, ``FFT_Diagnostic``, etc.
    """
    from ..data.diagnostic import Diagnostic
    return '_frame' in diagnostic.__dict__ or type(diagnostic) is not Diagnostic


def _collect_chain(diagnostic, _seen: set | None = None) -> list:
    """Recursively collect every Diagnostic in the computation chain.

    Used to clear unbounded ``_frame_cache`` dicts after each chunk,
    preventing memory explosion for long time-series exports.

    Note: *only* ``_frame_cache`` is cleared — the bounded LRU ``_cache``
    inside ``Derivative_Diagnostic`` is intentionally preserved because it
    provides the sliding-window caching that makes sequential d/dt efficient.
    """
    from ..data.diagnostic import Diagnostic
    if _seen is None:
        _seen = set()
    uid = id(diagnostic)
    if uid in _seen:
        return []
    _seen.add(uid)

    chain = [diagnostic]

    # Arithmetic-derived: _frame is a closure capturing 'self' and 'other'
    instance_frame = diagnostic.__dict__.get('_frame')
    if instance_frame is not None and hasattr(instance_frame, '__closure__') and instance_frame.__closure__:
        for name, cell in zip(instance_frame.__code__.co_freevars, instance_frame.__closure__):
            try:
                val = cell.cell_contents
                if isinstance(val, Diagnostic):
                    chain.extend(_collect_chain(val, _seen))
            except ValueError:
                pass  # empty cell

    # Post-processing subclasses store their parent in self._diag
    diag_attr = getattr(diagnostic, '_diag', None)
    if isinstance(diag_attr, Diagnostic):
        chain.extend(_collect_chain(diag_attr, _seen))

    return chain


def _resolve_workers(n_workers: int | None, n_frames: int) -> int:
    cpu = os.cpu_count() or 4
    if n_workers is None:
        return min(_MAX_EXPORT_WORKERS, n_frames, cpu)
    return max(1, min(n_workers, _MAX_EXPORT_WORKERS, n_frames))


def export_to_npy(
    diagnostic: Diagnostic,
    output_path: str | Path,
    n_workers: int | None = None,
    chunk_size: int = 64,
    show_progress: bool = True,
    overwrite: bool = False,
    reduce_axis: int | tuple[int, ...] | None = None,
    time_average: bool = False,
    checkpoint: bool = True,
) -> Path:
    """Export all timesteps of a Diagnostic to a single ``.npy`` file.

    Parameters
    ----------
    diagnostic : Diagnostic
        A fully initialised ``Diagnostic`` (e.g. ``sim["e1"]``).
    output_path : str or Path
        Destination ``.npy`` file.  Parent directories are created automatically.
    n_workers : int, optional
        Number of parallel reader threads.  Defaults to
        ``min(32, n_timesteps, os.cpu_count())``.
    chunk_size : int, optional
        Frames submitted to the thread pool at a time (default 64).
        Peak in-flight RAM = ``chunk_size x frame_bytes``.
    show_progress : bool, optional
        Show a ``tqdm`` progress bar (default ``True``).
    overwrite : bool, optional
        Overwrite an existing file (default ``False``).
    reduce_axis : int or tuple of int, optional
        Spatial axis/axes to average *within each frame* before saving.
        Indices are relative to a single frame (0 = x1, 1 = x2, …).
        The reduction is done in the reader thread — only the small
        reduced array crosses the thread boundary.
    time_average : bool, optional
        If ``True``, stream a running mean over all timesteps and save a
        single averaged frame.  Combines naturally with ``reduce_axis``.
        Uses a ``float64`` accumulator for numerical stability.
    checkpoint : bool, optional
        If ``True`` (default), flush the output to disk and save a
        ``.progress`` sidecar after every chunk so an interrupted run can
        resume.  Set to ``False`` to skip both — saves one syscall per
        chunk and is faster on high-latency parallel filesystems when
        you do not need crash-safety.

    Returns
    -------
    Path
        Absolute path of the written ``.npy`` file.

    Raises
    ------
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    RuntimeError
        If the Diagnostic has no timesteps or the first frame cannot be read.

    Examples
    --------
    >>> sim = ou.Simulation("path/to/os-stdin")

    Full time series — works for 100 or 100 000 timesteps:

    >>> ou.export_to_npy(sim["e1"], "e1.npy")
    >>> np.load("e1.npy", mmap_mode="r").shape   # (n_t, nx1, nx2)

    Average over x1 at every timestep:

    >>> ou.export_to_npy(sim["e1"], "e1_x1avg.npy", reduce_axis=0)
    >>> np.load("e1_x1avg.npy").shape            # (n_t, nx2)

    Time-average over all timesteps:

    >>> ou.export_to_npy(sim["e1"], "e1_tavg.npy", time_average=True)
    >>> np.load("e1_tavg.npy").shape             # (nx1, nx2)

    Both — x1-average then time-average:

    >>> ou.export_to_npy(sim["e1"], "e1_x1avg_tavg.npy",
    ...                  reduce_axis=0, time_average=True)
    >>> np.load("e1_x1avg_tavg.npy").shape       # (nx2,)
    """
    output_path = Path(output_path).resolve()

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists.  Pass overwrite=True to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(diagnostic)
    if n_frames == 0:
        raise RuntimeError("Diagnostic has no timesteps.")

    # ------------------------------------------------------------------
    # Probe shape / dtype from first frame (do not keep in memory).
    # ------------------------------------------------------------------
    try:
        first_raw = diagnostic._frame(0)
    except Exception as exc:
        raise RuntimeError("Failed to read the first timestep.") from exc

    # Shape after optional spatial reduction
    if reduce_axis is not None:
        first_reduced = first_raw.mean(axis=reduce_axis)
    else:
        first_reduced = first_raw
    frame_shape = first_reduced.shape
    raw_dtype = first_raw.dtype
    del first_raw, first_reduced

    # ------------------------------------------------------------------
    # Derived-diagnostic detection
    # ------------------------------------------------------------------
    # Derived diagnostics (arithmetic results, Derivative_Diagnostic, FFT, …)
    # have two properties that make parallel I/O counter-productive:
    #
    # 1. Their internal caches (especially Derivative_Diagnostic._cache with
    #    maxsize=6) are designed for sequential sliding-window access.
    #    With N workers, N×stencil_width frames are needed simultaneously,
    #    causing constant LRU evictions and redundant re-reads.
    #    With n_workers=1 sequential, only 1 new file read is needed per
    #    frame after the stencil warmup (vs 10 with parallel workers).
    #
    # 2. The general _frame_cache on every Diagnostic in the chain is
    #    unbounded. Without periodic clearing, it balloons to GB/TB over
    #    thousands of frames, causing swap thrashing.
    #
    # Fix: force n_workers=1, collect the full chain of Diagnostics, and
    # clear their _frame_cache after every chunk.
    derived = _is_derived(diagnostic)
    if derived:
        chain = _collect_chain(diagnostic)
        if n_workers is None:
            n_workers = 1
        elif n_workers > 1:
            import warnings
            warnings.warn(
                f"Derived diagnostic detected — forcing n_workers=1 for memory and cache efficiency. "
                f"(You requested n_workers={n_workers}.) "
                f"See export_to_npy docstring for details.",
                stacklevel=3,
            )
            n_workers = 1
    else:
        chain = [diagnostic]
        n_workers = _resolve_workers(n_workers, n_frames)

    # ------------------------------------------------------------------
    # Branch 1 — time_average=True: streaming accumulator, O(frame) RAM.
    # Output: a single frame (spatially averaged if reduce_axis is set).
    # ------------------------------------------------------------------
    if time_average:
        return _export_time_average(
            diagnostic=diagnostic,
            output_path=output_path,
            n_frames=n_frames,
            frame_shape=frame_shape,
            raw_dtype=raw_dtype,
            reduce_axis=reduce_axis,
            n_workers=n_workers,
            chunk_size=chunk_size,
            show_progress=show_progress,
            overwrite=overwrite,
            checkpoint=checkpoint,
            chain=chain,
        )

    # ------------------------------------------------------------------
    # Branch 2 — full export (with optional per-frame spatial reduction).
    # Output: (n_frames, *frame_shape), memory-mapped → bounded RAM.
    # ------------------------------------------------------------------
    return _export_frames(
        diagnostic=diagnostic,
        output_path=output_path,
        n_frames=n_frames,
        frame_shape=frame_shape,
        raw_dtype=raw_dtype,
        reduce_axis=reduce_axis,
        n_workers=n_workers,
        chunk_size=chunk_size,
        show_progress=show_progress,
        overwrite=overwrite,
        checkpoint=checkpoint,
        chain=chain,
    )


# ======================================================================
# Internal helpers
# ======================================================================


def _make_pbar(desc: str, total: int, n_workers: int, chunk_mb: float, show: bool) -> tqdm.tqdm:
    return tqdm.tqdm(
        total=total,
        desc=desc,
        unit="frame",
        disable=not show,
        postfix={"workers": n_workers, "chunk": f"{chunk_mb:.0f} MB"},
    )


# ------------------------------------------------------------------
# Progress / checkpoint helpers
# ------------------------------------------------------------------

def _progress_path(output_path: Path) -> Path:
    """Sidecar file that tracks which frames have been written."""
    return output_path.with_suffix(".progress")


def _load_progress(
    progress_path: Path,
    mode: str,
    n_frames: int,
    reduce_axis,
) -> set[int] | None:
    """Load a saved progress file and validate it against current run parameters.

    Returns the set of already-completed frame indices, or None if the
    progress file is absent, corrupt, or belongs to a different run.
    """
    if not progress_path.exists():
        return None
    try:
        with open(progress_path) as f:
            data = json.load(f)
    except Exception:
        return None  # corrupted — start fresh

    # Validate that the saved run matches current parameters exactly.
    # reduce_axis is stored as a list in JSON; normalise for comparison.
    saved_reduce = data.get("reduce_axis")
    if isinstance(saved_reduce, list):
        saved_reduce = tuple(saved_reduce) if len(saved_reduce) > 1 else (saved_reduce[0] if saved_reduce else None)
    current_reduce = tuple(reduce_axis) if isinstance(reduce_axis, (list, tuple)) else reduce_axis

    if (
        data.get("mode") != mode
        or data.get("n_frames") != n_frames
        or saved_reduce != current_reduce
    ):
        return None  # stale / mismatched — start fresh

    return set(data.get("completed", []))


def _save_progress(
    progress_path: Path,
    mode: str,
    n_frames: int,
    reduce_axis,
    completed: set[int],
) -> None:
    """Atomically write the progress sidecar file."""
    # Normalise reduce_axis so JSON round-trips cleanly
    if isinstance(reduce_axis, (list, tuple)):
        ra_serial = list(reduce_axis)
    elif reduce_axis is None:
        ra_serial = None
    else:
        ra_serial = reduce_axis  # plain int

    data = {
        "mode": mode,
        "n_frames": n_frames,
        "reduce_axis": ra_serial,
        "completed": sorted(completed),
    }
    # Write to a temp file then rename for atomicity (avoids a corrupt
    # progress file if the process is killed mid-write).
    tmp = progress_path.with_suffix(".progress.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(progress_path)


# ------------------------------------------------------------------
# Frame export helpers
# ------------------------------------------------------------------

def _find_base_diagnostics(chain: list) -> list:
    """Return only the file-backed base diagnostics from the chain.

    These are the leaf nodes of the computation graph — they hold a
    ``_file_list`` and can call ``_read_index`` directly.  Prefetching
    their files in parallel is safe because each call opens an independent
    HDF5 handle.
    """
    return [d for d in chain if getattr(d, '_file_list', None) is not None]


def _get_stencil_range(chain: list) -> tuple[int, int]:
    """Return ``(max_backward, max_forward)`` time-frame offsets across all temporal derivatives.

    Only considers diagnostics with ``_deriv_type == 't'`` — spatial derivatives
    (x1, x2, x3) use the same ``_stencil`` attribute for spatial grid offsets, which
    must not be confused with time-frame offsets when managing ``_frame_cache``.
    Zero when no temporal ``Derivative_Diagnostic`` is present in the chain.
    """
    max_back = max_fwd = 0
    for d in chain:
        if getattr(d, '_deriv_type', None) != 't':
            continue
        stencil = getattr(d, '_stencil', None)
        if stencil is not None:
            for s in stencil:
                s = int(s)
                if s < 0:
                    max_back = max(max_back, -s)
                elif s > 0:
                    max_fwd = max(max_fwd, s)
    return max_back, max_fwd


def _prefetch_into_cache(diag, idx: int) -> None:
    """Read one frame from a base diagnostic into its ``_frame_cache``.

    Thread-safe in CPython: individual dict ``__setitem__`` calls are
    atomic under the GIL.  Duplicate concurrent writes of the same key
    are benign (same value).
    """
    key = (idx, None)
    fc = diag._frame_cache
    if key not in fc:
        fc[key] = diag._read_index(idx)


def _clear_frame_caches_selective(chain: list, keep_from: int, keep_to: int) -> None:
    """Evict ``_frame_cache`` entries outside ``[keep_from, keep_to]``.

    Keeps the stencil window warm while preventing unbounded growth.
    The bounded LRU ``_cache`` inside ``Derivative_Diagnostic`` is never
    touched here — it manages its own eviction.
    """
    for diag in chain:
        fc = diag.__dict__.get('_frame_cache')
        if fc is None:
            continue
        stale = [k for k in list(fc) if not (keep_from <= k[0] <= keep_to)]
        for k in stale:
            del fc[k]


def _clear_frame_caches(chain: list) -> None:
    """Clear ALL ``_frame_cache`` entries on every diagnostic in the chain.

    Used by the base (non-derived) path where there is no stencil window
    to preserve.
    """
    for diag in chain:
        fc = diag.__dict__.get('_frame_cache')
        if fc is not None:
            fc.clear()


def _export_frames(
    diagnostic,
    output_path,
    n_frames,
    frame_shape,
    raw_dtype,
    reduce_axis,
    n_workers,
    chunk_size,
    show_progress,
    overwrite,
    checkpoint,
    chain,
) -> Path:
    """Write every frame to a memory-mapped npy file (bounded RAM).

    Two internal paths:

    **Base diagnostics** (direct HDF5 reads, n_workers > 1):
        Frames are read in parallel into a contiguous chunk buffer, then
        written to the memmap in one sequential block.  Peak extra RAM =
        ``chunk_size × frame_bytes``.

    **Derived diagnostics** (arithmetic / post-processing, n_workers = 1):
        Frames are computed and written one at a time directly into the
        memmap slot — no chunk buffer.  ``_frame_cache`` is cleared after
        *every frame* so intermediate results from the computation chain
        never accumulate.  The derivative LRU cache (maxsize=6) is
        intentionally preserved for stencil reuse across consecutive frames.
    """
    derived = len(chain) > 1 or n_workers == 1
    progress_path = _progress_path(output_path)
    full_shape = (n_frames, *frame_shape)
    frame_bytes = int(np.prod(frame_shape)) * raw_dtype.itemsize
    chunk_mb = chunk_size * frame_bytes / 1024**2

    # ---- resume detection ----
    completed: set[int] = set()
    if not overwrite:
        prev = _load_progress(progress_path, "frames", n_frames, reduce_axis)
        if prev is not None and output_path.exists():
            completed = prev

    memmap_mode = "r+" if completed else "w+"
    out = np.lib.format.open_memmap(
        output_path, mode=memmap_mode, dtype=raw_dtype, shape=full_shape,
    )

    pending = [i for i in range(n_frames) if i not in completed]
    if not pending:
        return output_path  # already complete

    n_done_already = len(completed)

    if derived:
        # ------------------------------------------------------------------
        # Derived path: prefetch pipeline + selective cache eviction.
        #
        # Problem: for each output frame the lazy eval graph triggers ~18
        # sequential HDF5 reads (stencil frames + all "current-only" terms
        # like n, P12, vfl2, b3 …).  At 7 ms/read that is ~126 ms of pure
        # I/O per frame, on top of ~50-100 ms of numpy computation.
        #
        # Fix: a dedicated I/O thread pool pre-reads every base-diagnostic
        # file for the NEXT frame while the main thread computes the current
        # one.  By the time computation finishes the next frame's data is
        # already in _frame_cache — the lazy eval finds it without touching
        # disk.
        #
        # Memory: selective cache eviction keeps only the stencil window
        # [idx - max_back .. idx + max_fwd] alive, bounding RAM to
        # (stencil_width) × (n_base_diags) × frame_size regardless of n_frames.
        # ------------------------------------------------------------------
        base_diags = _find_base_diagnostics(chain)
        max_back, max_fwd = _get_stencil_range(chain)
        n_io = max(1, min(_MAX_EXPORT_WORKERS, len(base_diags) * (max_fwd + 1)))

        pbar = tqdm.tqdm(
            total=n_frames,
            desc=f"Exporting → {output_path.name}",
            unit="frame",
            disable=not show_progress,
            postfix={"mode": "prefetch", "io_workers": n_io,
                     "frame_MB": f"{frame_bytes / 1024**2:.0f}"},
        )
        pbar.update(n_done_already)

        with ThreadPoolExecutor(max_workers=n_io) as io_pool:
            # Warmup: pre-read the initial forward stencil window so the
            # very first frames don't pay full cold-cache I/O cost.
            warmup = [
                io_pool.submit(_prefetch_into_cache, d, j)
                for d in base_diags
                for j in range(min(max_fwd + 1, n_frames))
            ]
            for f in warmup:
                f.result()

            prefetch_futures: list = []
            for idx in pending:
                # Wait for the prefetch that was launched for *this* frame
                # (submitted at the end of the previous iteration).
                # In practice these are already done — computation takes
                # longer than 18 parallel HDF5 reads.
                for f in prefetch_futures:
                    f.result()

                # Compute current frame — most reads hit _frame_cache.
                arr = diagnostic._frame(idx)
                if reduce_axis is not None:
                    arr = arr.mean(axis=reduce_axis)

                # Launch prefetch for the next forward stencil frame of
                # every base diagnostic while we write the current frame.
                next_pf = idx + max_fwd + 1
                if next_pf < n_frames:
                    prefetch_futures = [
                        io_pool.submit(_prefetch_into_cache, d, next_pf)
                        for d in base_diags
                    ]
                else:
                    prefetch_futures = []

                # Write directly into the memmap slot (no chunk buffer).
                out[idx] = arr
                del arr

                # Selective eviction: keep only the current stencil window.
                # Entries outside [idx-max_back .. idx+max_fwd] are stale.
                _clear_frame_caches_selective(chain, idx - max_back, idx + max_fwd)

                completed.add(idx)
                pbar.update(1)

                if checkpoint and len(completed) % chunk_size == 0:
                    out.flush()
                    _save_progress(progress_path, "frames", n_frames, reduce_axis, completed)

            # Drain any remaining prefetch futures.
            for f in prefetch_futures:
                f.result()

        pbar.close()

    else:
        # ------------------------------------------------------------------
        # Base path: parallel reads into a contiguous chunk buffer, then one
        # sequential write per chunk.
        # ------------------------------------------------------------------
        pbar = _make_pbar(
            f"Exporting → {output_path.name}",
            total=n_frames,
            n_workers=n_workers,
            chunk_mb=chunk_mb,
            show=show_progress,
        )
        pbar.update(n_done_already)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for chunk_start in range(0, len(pending), chunk_size):
                chunk_indices = pending[chunk_start : chunk_start + chunk_size]
                n_chunk = len(chunk_indices)

                # Pre-allocate contiguous buffer so the memmap write is one
                # sequential block rather than n_chunk scattered seeks.
                chunk_buf = np.empty((n_chunk, *frame_shape), dtype=raw_dtype)
                pos_of = {i: pos for pos, i in enumerate(chunk_indices)}

                future_map = {
                    executor.submit(_read_frame, (diagnostic, i, reduce_axis)): i
                    for i in chunk_indices
                }
                for future in as_completed(future_map):
                    idx, arr = future.result()
                    chunk_buf[pos_of[idx]] = arr
                    completed.add(idx)
                    pbar.update(1)

                out_start = chunk_indices[0]
                out_end   = chunk_indices[-1] + 1
                out[out_start:out_end] = chunk_buf
                del chunk_buf

                if checkpoint:
                    out.flush()
                    _save_progress(progress_path, "frames", n_frames, reduce_axis, completed)

        pbar.close()

    if checkpoint:
        progress_path.unlink(missing_ok=True)
    return output_path


# ------------------------------------------------------------------
# Time-average export (streaming accumulator)
# ------------------------------------------------------------------

def _export_time_average(
    diagnostic,
    output_path,
    n_frames,
    frame_shape,
    raw_dtype,
    reduce_axis,
    n_workers,
    chunk_size,
    show_progress,
    overwrite,
    checkpoint,
    chain,
) -> Path:
    """Stream a running mean over all timesteps — O(frame_size) RAM.

    Uses a float64 accumulator for numerical stability.  The accumulator
    state is checkpointed to a ``.acc.npy`` sidecar after every chunk so
    that an interrupted run can resume without reprocessing all frames.
    """
    progress_path = _progress_path(output_path)
    acc_path = output_path.with_suffix(".acc.npy")
    frame_bytes = int(np.prod(frame_shape)) * np.dtype(np.float64).itemsize
    chunk_mb = chunk_size * frame_bytes / 1024**2

    # ---- resume detection ----
    completed: set[int] = set()
    if not overwrite:
        prev = _load_progress(progress_path, "time_average", n_frames, reduce_axis)
        if prev is not None and acc_path.exists():
            completed = prev
            acc = np.load(acc_path).astype(np.float64)
        else:
            acc = np.zeros(frame_shape, dtype=np.float64)
    else:
        acc = np.zeros(frame_shape, dtype=np.float64)

    pending = [i for i in range(n_frames) if i not in completed]
    if not pending:
        # All frames were already accumulated in a previous run; just finalise.
        acc /= n_frames
        np.save(output_path, acc)
        progress_path.unlink(missing_ok=True)
        acc_path.unlink(missing_ok=True)
        return output_path

    n_done_already = len(completed)
    pbar = _make_pbar(
        f"Time-averaging → {output_path.name}",
        total=n_frames,
        n_workers=n_workers,
        chunk_mb=chunk_mb,
        show=show_progress,
    )
    pbar.update(n_done_already)

    derived = len(chain) > 1 or n_workers == 1

    if derived:
        # Derived path: same prefetch pipeline as _export_frames.
        base_diags = _find_base_diagnostics(chain)
        max_back, max_fwd = _get_stencil_range(chain)
        n_io = max(1, min(_MAX_EXPORT_WORKERS, len(base_diags) * (max_fwd + 1)))

        with ThreadPoolExecutor(max_workers=n_io) as io_pool:
            warmup = [
                io_pool.submit(_prefetch_into_cache, d, j)
                for d in base_diags
                for j in range(min(max_fwd + 1, n_frames))
            ]
            for f in warmup:
                f.result()

            prefetch_futures: list = []
            for idx in pending:
                for f in prefetch_futures:
                    f.result()

                arr = diagnostic._frame(idx)
                if reduce_axis is not None:
                    arr = arr.mean(axis=reduce_axis)

                next_pf = idx + max_fwd + 1
                if next_pf < n_frames:
                    prefetch_futures = [
                        io_pool.submit(_prefetch_into_cache, d, next_pf)
                        for d in base_diags
                    ]
                else:
                    prefetch_futures = []

                acc += arr
                del arr
                _clear_frame_caches_selective(chain, idx - max_back, idx + max_fwd)
                completed.add(idx)
                pbar.update(1)

                if checkpoint and len(completed) % chunk_size == 0:
                    np.save(acc_path, acc)
                    _save_progress(progress_path, "time_average", n_frames, reduce_axis, completed)

            for f in prefetch_futures:
                f.result()
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for chunk_start in range(0, len(pending), chunk_size):
                chunk_indices = pending[chunk_start : chunk_start + chunk_size]

                future_map = {
                    executor.submit(_read_frame, (diagnostic, i, reduce_axis)): i
                    for i in chunk_indices
                }
                for future in as_completed(future_map):
                    idx, arr = future.result()
                    acc += arr
                    completed.add(idx)
                    pbar.update(1)

                if checkpoint:
                    np.save(acc_path, acc)
                    _save_progress(progress_path, "time_average", n_frames, reduce_axis, completed)

    pbar.close()

    acc /= n_frames
    np.save(output_path, acc)

    if checkpoint:
        progress_path.unlink(missing_ok=True)
        acc_path.unlink(missing_ok=True)
    return output_path


def export_simulation_to_npy(
    simulation: Simulation,
    quantities: list[str],
    output_folder: str | Path,
    n_workers: int | None = None,
    chunk_size: int = 64,
    show_progress: bool = True,
    overwrite: bool = False,
    reduce_axis: int | tuple[int, ...] | None = None,
    time_average: bool = False,
    checkpoint: bool = True,
) -> dict[str, Path]:
    """Export multiple quantities from a Simulation to ``.npy`` files.

    Each quantity is exported sequentially so parallel threads are fully
    utilised per quantity without competing for filesystem bandwidth.

    Parameters
    ----------
    simulation : Simulation
        A fully initialised ``Simulation`` object.
    quantities : list[str]
        Quantity names to export (e.g. ``["e1", "e2", "b3"]``).
        Species quantities use ``"species/quantity"`` syntax
        (e.g. ``"electrons/n"``), saved as ``electrons_n.npy``.
    output_folder : str or Path
        Directory for output files (created if missing).
    n_workers : int, optional
        Parallel reader threads per quantity.
    chunk_size : int, optional
        Frames per chunk (default 64).
    show_progress : bool, optional
        ``tqdm`` progress bars (default ``True``).
    overwrite : bool, optional
        Overwrite existing files (default ``False``).
    reduce_axis : int or tuple of int, optional
        Spatial axes to average within each frame (see :func:`export_to_npy`).
    time_average : bool, optional
        Save the time-average instead of the full time series.

    Returns
    -------
    dict[str, Path]
        ``{quantity_name: absolute_output_path}`` for every exported quantity.

    Examples
    --------
    >>> sim = ou.Simulation("path/to/os-stdin")

    Full export:

    >>> ou.export_simulation_to_npy(sim, ["e1", "e2", "b3"], "output/")

    x1-averages for all fields:

    >>> ou.export_simulation_to_npy(sim, ["e1", "e2"], "output/x1avg/",
    ...                             reduce_axis=0)

    Time-averages:

    >>> ou.export_simulation_to_npy(sim, ["e1", "e2"], "output/tavg/",
    ...                             time_average=True)
    """
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    for qty in quantities:
        if "/" in qty:
            species_name, field = qty.split("/", 1)
            diagnostic = simulation[species_name][field]
            filename = f"{species_name}_{field}.npy"
        else:
            diagnostic = simulation[qty]
            filename = f"{qty}.npy"

        results[qty] = export_to_npy(
            diagnostic,
            output_folder / filename,
            n_workers=n_workers,
            chunk_size=chunk_size,
            show_progress=show_progress,
            overwrite=overwrite,
            reduce_axis=reduce_axis,
            time_average=time_average,
            checkpoint=checkpoint,
        )

    return results
