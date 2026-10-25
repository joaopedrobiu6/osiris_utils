"""Synthetic field series for the benchmarks — the repository ships no OSIRIS data.

Written with the same HDF5 writer the example scripts use, so the files have
the real OSIRIS layout and schema; cached in the OS temp directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "scripts"))
from synthetic_run import _bytes_attr, write_grid_file  # noqa: E402


def field_series(name: str = "e3", nt: int = 64, shape: tuple[int, ...] = (1024, 256)) -> Path:
    """Write *nt* random float32 frames of *name* (shape *shape*) and return the run folder.

    The folder holds ``MS/FLD/<name>/<name>-NNNNNN.h5``, which is all
    ``Diagnostic(simulation_folder=...).get_quantity(name)`` needs.
    """
    root = Path(tempfile.gettempdir()) / "osiris_utils_bench" / f"{name}_{nt}_{'x'.join(map(str, shape))}"
    folder = root / "MS" / "FLD" / name
    if len(list(folder.glob("*.h5"))) == nt:
        return root
    rng = np.random.default_rng(0)
    grid = np.array([[0.0, 0.25 * n] for n in shape])  # dx = 0.25 c/omega_p, as in the shock runs
    for i in range(nt):
        data = rng.standard_normal(shape).astype(np.float32)
        write_grid_file(folder / f"{name}-{i:06d}.h5", name=name, data=data, iteration=i, grid=grid)
    return root


def _particle_header(f: h5py.File, typ: str, quants: list[str]) -> None:
    sim = f.create_group("SIMULATION")
    sim.attrs.create("DT", [0.1])
    sim.attrs.create("NDIMS", [2])
    sim.attrs.create("XMIN", [0.0, 0.0])
    sim.attrs.create("XMAX", [100.0, 50.0])
    _bytes_attr(f, "NAME", "electrons")
    _bytes_attr(f, "TYPE", typ)
    for key in ("QUANTS", "UNITS", "LABELS"):
        f.attrs.create(key, [np.bytes_(q.encode()) for q in quants])


def raw_dump(n_particles: int = 20_000_000) -> Path:
    """One 2-D RAW dump: seven float32 quantities and the (n, 2) int32 tag, one dataset each."""
    path = Path(tempfile.gettempdir()) / "osiris_utils_bench" / f"RAW-electrons-{n_particles}.h5"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    quants = ["x1", "x2", "p1", "p2", "p3", "q", "ene", "tag"]
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        _particle_header(f, "particles", quants)
        f.attrs.create("TIME", [10.0])
        _bytes_attr(f, "TIME UNITS", "1 / \\omega_{pe}")
        f.attrs.create("ITER", [100])
        for q in quants[:-1]:
            f.create_dataset(q, data=rng.standard_normal(n_particles, dtype=np.float32))
        f.create_dataset("tag", data=rng.integers(1, 2**30, (n_particles, 2), dtype=np.int32))
    return path


def track_file(n_particles: int = 4000, n_iters: int = 1000, chunk: int = 50) -> Path:
    """A tracks-2 file: every particle written in chunks of *chunk* points, chunks interleaved between particles.

    Written as ``<run>/MS/TRACKS/electrons-tracks.h5``, so ``path.parents[2]`` is a run folder for ``Track_Diagnostic``.
    """
    root = Path(tempfile.gettempdir()) / "osiris_utils_bench" / f"tracks_{n_particles}x{n_iters}c{chunk}"
    path = root / "MS" / "TRACKS" / "electrons-tracks.h5"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    quants = ["n", "t", "q", "ene", "x1", "x2", "p1", "p2", "p3"]
    particle = np.tile(np.arange(1, n_particles + 1), n_iters // chunk)  # (M,) chunk-major, particles interleaved
    nstart = np.repeat(np.arange(0, n_iters, chunk), n_particles)  # (M,)
    itermap = np.stack([particle, np.full_like(particle, chunk), nstart], axis=1).astype(np.int32)  # (M, 3)
    data = np.random.default_rng(0).standard_normal((particle.size * chunk, len(quants) - 1))  # (N, nq)
    with h5py.File(path, "w") as f:
        _particle_header(f, "tracks-2", quants)
        f.attrs.create("NTRACKS", [n_particles])
        f.create_dataset("data", data=data)
        f.create_dataset("itermap", data=itermap)
    return path
