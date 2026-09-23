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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..data.diagnostic import Diagnostic
    from ..data.simulation import Simulation

# Beyond ~32 threads, Lustre MDS contention dominates on HPC clusters.
_MAX_EXPORT_WORKERS = 32


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
    n_frames = len(diagnostic)
    if n_frames == 0:
        raise RuntimeError("Diagnostic has no timesteps.")

    # An interrupted run of this same export left a progress record next to
    # its partial output (the memmap, or the running sum): resume it.  Any
    # other existing output is refused unless overwrite=True.
    mode = "time_average" if time_average else "frames"
    progress_path = output_path.with_suffix(".progress")
    acc_path = output_path.with_suffix(".acc.npy")
    completed: set[int] = set()
    if not overwrite and (acc_path if time_average else output_path).exists():
        completed = _load_progress(progress_path, mode, n_frames, reduce_axis) or set()
    if output_path.exists() and not overwrite and not completed:
        raise FileExistsError(f"{output_path} already exists.  Pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe shape / dtype from the first frame (not kept in memory).
    try:
        first = diagnostic._frame(0)
    except Exception as exc:
        raise RuntimeError("Failed to read the first timestep.") from exc
    dtype = first.dtype
    frame_shape = (first if reduce_axis is None else first.mean(axis=reduce_axis)).shape
    del first

    # Derived diagnostics (arithmetic results, Derivative_Diagnostic, FFT, …)
    # keep a small per-instance LRU of frames (Diagnostic.frame_cache_size),
    # sized for sequential sliding-window access.  With N workers,
    # N x stencil_width frames are needed at once, the LRU evicts constantly
    # and files are re-read; sequentially, each frame after the stencil
    # warm-up costs one new read.  So derived diagnostics run on one worker,
    # with the chain's file-backed leaves prefetched ahead of it.
    if diagnostic.is_derived:
        if n_workers is not None and n_workers > 1:
            warnings.warn(
                f"Derived diagnostic detected — forcing n_workers=1 for memory and cache efficiency. "
                f"(You requested n_workers={n_workers}.) "
                f"See export_to_npy docstring for details.",
                stacklevel=2,
            )
        n_workers = 1
    else:
        n_workers = _resolve_workers(n_workers, n_frames)

    # Where each frame goes, and what a checkpoint writes: the memory-mapped
    # output (peak RAM ~ one chunk, whatever n_frames), or a float64 running
    # sum for the time average (O(frame) RAM).
    if time_average:
        acc = np.load(acc_path).astype(np.float64) if completed else np.zeros(frame_shape)

        def store(idx: int, arr: np.ndarray) -> None:
            np.add(acc, arr, out=acc)

        def save() -> None:
            np.save(acc_path, acc)

    else:
        out = np.lib.format.open_memmap(output_path, mode="r+" if completed else "w+", dtype=dtype, shape=(n_frames, *frame_shape))

        def store(idx: int, arr: np.ndarray) -> None:
            out[idx] = arr

        save = out.flush

    pending = [i for i in range(n_frames) if i not in completed]
    desc = f"{'Time-averaging' if time_average else 'Exporting'} → {output_path.name}"
    with tqdm.tqdm(total=n_frames, initial=len(completed), desc=desc, unit="frame", disable=not show_progress) as pbar:
        for idx, arr in _iter_frames(diagnostic, pending, reduce_axis, n_workers, chunk_size):
            store(idx, arr)
            completed.add(idx)
            pbar.update(1)
            if checkpoint and len(completed) % chunk_size == 0:
                save()
                _save_progress(progress_path, mode, n_frames, reduce_axis, completed)

    if time_average:
        np.save(output_path, acc / n_frames)
        acc_path.unlink(missing_ok=True)
    else:
        out.flush()
    progress_path.unlink(missing_ok=True)
    return output_path


# ======================================================================
# Internal helpers
# ======================================================================


def _resolve_workers(n_workers: int | None, n_frames: int) -> int:
    cpu = os.cpu_count() or 4
    if n_workers is None:
        return min(_MAX_EXPORT_WORKERS, n_frames, cpu)
    return max(1, min(n_workers, _MAX_EXPORT_WORKERS, n_frames))


def _read_frame(diagnostic: Diagnostic, index: int, reduce_axis) -> tuple[int, np.ndarray]:
    """One frame, spatially reduced if asked — in the worker, so only the reduced array crosses threads."""
    arr = diagnostic._frame(index)
    return index, arr if reduce_axis is None else arr.mean(axis=reduce_axis)


def _iter_frames(
    diagnostic: Diagnostic, pending: list[int], reduce_axis, n_workers: int, chunk_size: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, frame)`` for every *pending* index, in bounded memory.

    With several workers (a file-backed diagnostic), threads read up to
    *chunk_size* frames at a time — h5py releases the GIL, so they overlap
    I/O latency — and frames are yielded as they complete.  With one (a
    derived diagnostic), frames are computed in order on this thread while an
    I/O pool reads the next frame of every file-backed leaf of the chain into
    that leaf's frame cache, so the evaluation finds it without touching disk.
    """
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for start in range(0, len(pending), chunk_size):
                futures = [pool.submit(_read_frame, diagnostic, i, reduce_axis) for i in pending[start : start + chunk_size]]
                for future in as_completed(futures):
                    yield future.result()
        return

    chain = _collect_chain(diagnostic)
    leaves = [d for d in chain if getattr(d, "_file_list", None) is not None]  # the ones that read files
    reach = _prefetch_reach(chain)
    n_frames = len(diagnostic)
    with ThreadPoolExecutor(max_workers=max(1, min(_MAX_EXPORT_WORKERS, len(leaves) * (reach + 1)))) as io:
        # warm the stencil window of the first frame
        first = pending[0] if pending else 0
        for future in [io.submit(leaf._read_index, j) for leaf in leaves for j in range(first, min(first + reach + 1, n_frames))]:
            future.result()
        ahead: list = []
        for idx in pending:
            for future in ahead:
                future.result()
            frame = _read_frame(diagnostic, idx, reduce_axis)
            nxt = idx + reach + 1
            ahead = [io.submit(leaf._read_index, nxt) for leaf in leaves] if nxt < n_frames else []
            yield frame
        for future in ahead:
            future.result()


def _collect_chain(diagnostic: Diagnostic) -> list[Diagnostic]:
    """*diagnostic* and every diagnostic it is computed from (its ``_parents``, recursively), each once."""
    chain: list = []
    seen: set[int] = set()
    stack = [diagnostic]
    while stack:
        d = stack.pop()
        if id(d) in seen:
            continue
        seen.add(id(d))
        chain.append(d)
        stack.extend(getattr(d, "_parents", ()))
    return chain


def _prefetch_reach(chain: list[Diagnostic]) -> int:
    """How many frames ahead the time derivatives in *chain* read (0 without one)."""
    return max((max(d._fd_stencil) for d in chain if getattr(d, "_has_time", False)), default=0)


# ------------------------------------------------------------------
# Progress / checkpoint helpers
# ------------------------------------------------------------------


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

    if data.get("mode") != mode or data.get("n_frames") != n_frames or saved_reduce != current_reduce:
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
