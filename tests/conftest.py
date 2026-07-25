"""Shared test fixtures.

The suite used to depend on ~350 MB of real OSIRIS HDF5 dumps committed under
``examples/example_data``. Those files were removed from the repo, which left
the suite unrunnable. Instead of re-committing binary data, this module *writes*
a small synthetic simulation tree that follows the exact OSIRIS output layout
and HDF5 schema, so tests are hermetic, fast and reproducible.

The synthetic run mirrors ``examples/example_data/thermal.1d``:
1D, ``nx = 500``, ``x in [0, 5]``, ``dt = 0.0099``, ``ndump = 20``, one
species ``electrons`` with ``rqm = -1``.

Every value is analytic (see :func:`grid_values`, :func:`raw_values` and
:func:`track_value`) so tests can assert against a closed form rather than
against magic constants copied out of a binary file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

# Canonical input deck for the test suite. Lives under tests/ so the suite owns
# its own inputs — it previously read examples/example_data/thermal.1d, which a
# git history rewrite removed along with the large binaries in that directory.
DECK_PATH = Path(__file__).resolve().parent / "data" / "thermal.1d"

# --- Parameters of the synthetic run (kept in sync with thermal.1d) --------
NX = 64  # smaller than the deck's 500: keeps files tiny, schema identical
XMIN = 0.0
XMAX = 5.0
DT = 0.0099
NDUMP = 20
NDIMS = 1
N_TIMESTEPS = 6  # number of dumps written per grid diagnostic
SPECIES = "electrons"
RQM = -1.0
TIME_UNITS = "1 / \\omega_p"

# RAW diagnostic
RAW_ITER = 50
RAW_N_PARTICLES = 64
RAW_QUANTS = ["x1", "p1", "p2", "p3", "q", "ene", "tag"]
RAW_UNITS = {
    "x1": "c/\\omega_p",
    "p1": "m_e c",
    "p2": "m_e c",
    "p3": "m_e c",
    "q": "e",
    "ene": "m_e c^2",
    "tag": "",
}
RAW_LABELS = {
    "x1": "x_1",
    "p1": "p_1",
    "p2": "p_2",
    "p3": "p_3",
    "q": "q",
    "ene": "Ene",
    "tag": "Tag",
}

# TRACKS diagnostic
TRACK_N_PARTICLES = 8
TRACK_N_ITERS = 10
TRACK_CHUNK = 5  # points per itermap chunk; two chunks per particle
TRACK_NITER = 1
# QUANTS[0] is the iteration counter 'n'; the 'data' dataset holds QUANTS[1:]
TRACK_QUANTS = ["n", "t", "q", "ene", "x1", "p1", "p2", "p3"]
TRACK_UNITS = {
    "t": "1/\\omega_p",
    "q": "e",
    "ene": "m_e c^2",
    "x1": "c/\\omega_p",
    "p1": "m_e c",
    "p2": "m_e c",
    "p3": "m_e c",
}
TRACK_LABELS = {
    "t": "t",
    "q": "q",
    "ene": "Ene",
    "x1": "x_1",
    "p1": "p_1",
    "p2": "p_2",
    "p3": "p_3",
}


# --- Analytic data definitions --------------------------------------------


def grid_values(iteration: int, nx: int = NX) -> np.ndarray:
    """Field values for a given dump: a smooth, differentiable profile.

    ``f_i(x) = sin(2*pi*x/L) + 0.01*i`` — smooth enough that finite-difference
    derivative tests have an exact reference to compare against.
    """
    x = np.linspace(XMIN, XMAX, nx, endpoint=False)
    return (np.sin(2 * np.pi * x / (XMAX - XMIN)) + 0.01 * iteration).astype(np.float32)


def raw_values(quant: str) -> np.ndarray:
    """Deterministic per-particle RAW values, indexed by particle number."""
    p = np.arange(RAW_N_PARTICLES)
    if quant == "x1":
        return (XMIN + (XMAX - XMIN) * p / RAW_N_PARTICLES).astype(np.float32)
    if quant == "p1":
        # spans 0 .. 0.063 so a `p1 > 0.025` style mask selects a known subset
        return (0.001 * p).astype(np.float32)
    if quant == "p2":
        return (-0.001 * p).astype(np.float32)
    if quant == "p3":
        return np.zeros(RAW_N_PARTICLES, dtype=np.float32)
    if quant == "q":
        return np.full(RAW_N_PARTICLES, -1.0, dtype=np.float32)
    if quant == "ene":
        return (0.5e-6 * p**2).astype(np.float32)
    if quant == "tag":
        # node id in column 0, particle id in column 1; some tags are negative
        # to exercise the abs() in create_file_tags
        node = (p % 4) + 1
        node = np.where(p % 8 == 0, -node, node)
        return np.stack([node, p + 1], axis=1).astype(np.int32)
    raise KeyError(quant)


def track_value(quant: str, particle: int, k: int) -> float:
    """Value of ``quant`` for 1-based ``particle`` at time index ``k``."""
    if quant == "t":
        return k * DT
    if quant == "q":
        return -1.0
    if quant == "ene":
        return 0.1 * particle + 0.001 * k
    if quant == "x1":
        return particle + 0.01 * k
    if quant == "p1":
        return 0.001 * particle + 0.0001 * k
    if quant == "p2":
        return -0.001 * particle
    if quant == "p3":
        return 0.0
    raise KeyError(quant)


# --- Low-level writers ----------------------------------------------------


def _bytes_attr(obj, key: str, value: str) -> None:
    obj.attrs.create(key, [np.bytes_(value.encode())])


def write_grid_file(
    path: Path,
    *,
    name: str,
    data: np.ndarray,
    iteration: int,
    units: str = "e \\omega_p^2 / c",
    label: str = "\\rho",
    dt: float = DT,
    ndump: int = NDUMP,
    grid: np.ndarray | None = None,
) -> Path:
    """Write one OSIRIS grid dump in the layout ``OsirisGridFile`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    ndims = data.ndim
    if grid is None:
        grid = np.array([[XMIN, XMAX]] * ndims, dtype=float)
    grid = np.atleast_2d(grid)

    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [dt])
        sim.attrs.create("NDIMS", [ndims])

        f.attrs.create("TIME", [iteration * dt * ndump])
        _bytes_attr(f, "TIME UNITS", TIME_UNITS)
        f.attrs.create("ITER", [iteration * ndump])
        _bytes_attr(f, "NAME", name)
        _bytes_attr(f, "TYPE", "grid")
        _bytes_attr(f, "UNITS", units)
        _bytes_attr(f, "LABEL", label)

        # OSIRIS stores the array transposed relative to (x1, x2, ...) ordering
        f.create_dataset(name, data=data.T)

        axis_group = f.create_group("AXIS")
        for i in range(ndims):
            ax = axis_group.create_dataset(f"AXIS{i + 1}", data=np.asarray(grid[i], dtype=float))
            _bytes_attr(ax, "NAME", f"x{i + 1}")
            _bytes_attr(ax, "UNITS", "c / \\omega_p")
            _bytes_attr(ax, "LONG_NAME", f"x_{i + 1}")
            _bytes_attr(ax, "TYPE", "linear")
    return path


def write_grid_series(
    directory: Path,
    *,
    name: str,
    prefix: str,
    n_timesteps: int = N_TIMESTEPS,
    nx: int = NX,
    **kwargs,
) -> list[Path]:
    """Write ``prefix-000000.h5 ... prefix-00000N.h5`` into ``directory``.

    The trailing ``NNNNNN.h5`` is required: ``Diagnostic._scan_files`` derives
    its file template by stripping the last 9 characters.
    """
    paths = []
    for i in range(n_timesteps):
        paths.append(
            write_grid_file(
                directory / f"{prefix}-{i:06d}.h5",
                name=name,
                data=grid_values(i, nx),
                iteration=i,
                **kwargs,
            )
        )
    return paths


def write_raw_file(path: Path, *, species: str = SPECIES, iteration: int = RAW_ITER) -> Path:
    """Write an OSIRIS RAW particle dump in the layout ``OsirisRawFile`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [DT])
        sim.attrs.create("NDIMS", [NDIMS])
        sim.attrs.create("XMIN", [XMIN])
        sim.attrs.create("XMAX", [XMAX])

        f.attrs.create("TIME", [iteration * DT])
        _bytes_attr(f, "TIME UNITS", TIME_UNITS)
        f.attrs.create("ITER", [iteration])
        _bytes_attr(f, "NAME", species)
        _bytes_attr(f, "TYPE", "particles")
        f.attrs.create("QUANTS", [np.bytes_(q.encode()) for q in RAW_QUANTS])
        f.attrs.create("UNITS", [np.bytes_(RAW_UNITS[q].encode()) for q in RAW_QUANTS])
        f.attrs.create("LABELS", [np.bytes_(RAW_LABELS[q].encode()) for q in RAW_QUANTS])

        for quant in RAW_QUANTS:
            f.create_dataset(quant, data=raw_values(quant))
    return path


def write_track_file(path: Path, *, species: str = SPECIES) -> Path:
    """Write an OSIRIS 'tracks-2' file in the layout ``OsirisTrackFile`` expects.

    Particles are split across two itermap chunks each, and the chunks are
    interleaved between particles — the same shape as real OSIRIS output, so
    the reordering logic is genuinely exercised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data_quants = TRACK_QUANTS[1:]

    itermap_rows: list[list[int]] = []
    data_rows: list[list[float]] = []
    n_chunks = TRACK_N_ITERS // TRACK_CHUNK
    for chunk in range(n_chunks):
        nstart = chunk * TRACK_CHUNK
        for particle in range(1, TRACK_N_PARTICLES + 1):
            itermap_rows.append([particle, TRACK_CHUNK, nstart])
            for offset in range(TRACK_CHUNK):
                k = nstart + offset
                data_rows.append([track_value(q, particle, k) for q in data_quants])

    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [DT])
        sim.attrs.create("NDIMS", [NDIMS])
        sim.attrs.create("XMIN", [XMIN])
        sim.attrs.create("XMAX", [XMAX])

        _bytes_attr(f, "NAME", species)
        _bytes_attr(f, "TYPE", "tracks-2")
        f.attrs.create("NTRACKS", [TRACK_N_PARTICLES])
        f.attrs.create("NITER", [TRACK_NITER])
        f.attrs.create("QUANTS", [np.bytes_(q.encode()) for q in TRACK_QUANTS])
        f.attrs.create("UNITS", [np.bytes_(b"")] + [np.bytes_(TRACK_UNITS[q].encode()) for q in data_quants])
        f.attrs.create("LABELS", [np.bytes_(b"n")] + [np.bytes_(TRACK_LABELS[q].encode()) for q in data_quants])

        f.create_dataset("data", data=np.array(data_rows, dtype=np.float64))
        f.create_dataset("itermap", data=np.array(itermap_rows, dtype=np.int32))
    return path


def build_simulation_tree(root: Path) -> Path:
    """Create a complete synthetic OSIRIS run under ``root``.

    Layout::

        root/thermal.1d
        root/MS/FLD/e3/e3-NNNNNN.h5
        root/MS/DENSITY/electrons/charge/charge-electrons-NNNNNN.h5
        root/MS/UDIST/electrons/{vfl1,T11}/...-electrons-NNNNNN.h5
        root/MS/RAW/electrons/RAW-electrons-000050.h5
        root/MS/TRACKS/electrons-tracks.h5
    """
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DECK_PATH, root / "thermal.1d")

    ms = root / "MS"
    write_grid_series(ms / "FLD" / "e3", name="e3", prefix="e3", units="m_e c \\omega_p / e", label="E_3")
    write_grid_series(
        ms / "DENSITY" / SPECIES / "charge",
        name="charge",
        prefix=f"charge-{SPECIES}",
        units="e \\omega_p^2 / c",
        label="\\rho",
    )
    write_grid_series(
        ms / "UDIST" / SPECIES / "vfl1",
        name="vfl1",
        prefix=f"vfl1-{SPECIES}",
        units="c",
        label="v_1",
    )
    write_grid_series(
        ms / "UDIST" / SPECIES / "T11",
        name="T11",
        prefix=f"T11-{SPECIES}",
        units="m_e c^2",
        label="T_{11}",
    )
    write_raw_file(ms / "RAW" / SPECIES / f"RAW-{SPECIES}-{RAW_ITER:06d}.h5")
    write_track_file(ms / "TRACKS" / f"{SPECIES}-tracks.h5")
    return root


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(scope="session")
def _sim_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic tree once per session."""
    return build_simulation_tree(tmp_path_factory.mktemp("osiris_sim_template"))


@pytest.fixture
def sim_dir(_sim_template: Path, tmp_path: Path) -> Path:
    """A writable per-test copy of the synthetic simulation folder."""
    target = tmp_path / "sim"
    shutil.copytree(_sim_template, target)
    return target


@pytest.fixture
def deck_path(sim_dir: Path) -> Path:
    """Path to the input deck inside the per-test simulation folder."""
    return sim_dir / "thermal.1d"


@pytest.fixture
def raw_path(sim_dir: Path) -> Path:
    """Path to the synthetic RAW dump."""
    return sim_dir / "MS" / "RAW" / SPECIES / f"RAW-{SPECIES}-{RAW_ITER:06d}.h5"


@pytest.fixture
def track_path(sim_dir: Path) -> Path:
    """Path to the synthetic TRACKS file."""
    return sim_dir / "MS" / "TRACKS" / f"{SPECIES}-tracks.h5"
