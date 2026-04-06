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


def _read_frame(args: tuple) -> tuple[int, np.ndarray]:
    """Worker: read one frame and optionally reduce spatial axes.

    Spatial reduction is done inside the thread so that only the
    (much smaller) reduced array is returned to the main thread.
    """
    diagnostic, index, reduce_axis = args
    arr = diagnostic._read_index(index)
    if reduce_axis is not None:
        arr = arr.mean(axis=reduce_axis)
    return index, arr


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
        first_raw = diagnostic._read_index(0)
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
# Frame export (full time series)
# ------------------------------------------------------------------

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
) -> Path:
    """Write every frame to a memory-mapped npy file (bounded RAM).

    If a ``.progress`` sidecar exists from a previous interrupted run
    (and ``overwrite`` is False), only the missing frames are written.
    """
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
    pbar = _make_pbar(
        f"Exporting → {output_path.name}",
        total=n_frames,
        n_workers=n_workers,
        chunk_mb=chunk_mb,
        show=show_progress,
    )
    pbar.update(n_done_already)  # reflect already-done frames in the bar

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(pending), chunk_size):
            chunk_indices = pending[chunk_start : chunk_start + chunk_size]

            future_to_idx = {
                executor.submit(_read_frame, (diagnostic, i, reduce_axis)): i
                for i in chunk_indices
            }
            for future in as_completed(future_to_idx):
                idx, arr = future.result()
                out[idx] = arr
                completed.add(idx)
                pbar.update(1)

            out.flush()
            # Checkpoint after every chunk — worst-case re-work on resume
            # is one chunk, not the whole run.
            _save_progress(progress_path, "frames", n_frames, reduce_axis, completed)

    pbar.close()
    progress_path.unlink(missing_ok=True)  # clean up on success
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

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for chunk_start in range(0, len(pending), chunk_size):
            chunk_indices = pending[chunk_start : chunk_start + chunk_size]

            future_to_idx = {
                executor.submit(_read_frame, (diagnostic, i, reduce_axis)): i
                for i in chunk_indices
            }
            chunk_results: dict[int, np.ndarray] = {}
            for future in as_completed(future_to_idx):
                idx, arr = future.result()
                chunk_results[idx] = arr
                completed.add(idx)
                pbar.update(1)

            for arr in chunk_results.values():
                acc += arr

            # Checkpoint accumulator and progress after every chunk.
            np.save(acc_path, acc)
            _save_progress(progress_path, "time_average", n_frames, reduce_axis, completed)

    pbar.close()

    acc /= n_frames
    np.save(output_path, acc)

    # Clean up sidecars on success.
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
        )

    return results
