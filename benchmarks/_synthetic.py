"""Synthetic field series for the benchmarks — the repository ships no OSIRIS data.

Written with the same HDF5 writer the example scripts use, so the files have
the real OSIRIS layout and schema; cached in the OS temp directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "scripts"))
from synthetic_run import write_grid_file  # noqa: E402


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
