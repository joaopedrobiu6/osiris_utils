"""Write a small synthetic OSIRIS run so every example script is runnable.

The repository ships no binary OSIRIS output (a 2-D production run is hundreds
of MB), so the examples build their own tree instead.  The layout and the HDF5
schema are the ones OSIRIS writes, therefore ``osiris_utils`` reads it exactly
as it reads a real run::

    <root>/os-stdin                                   input deck
    <root>/MS/FLD/<fld>/<fld>-NNNNNN.h5               e1..e3, b1..b3
    <root>/MS/DENSITY/<sp>/charge/charge-<sp>-NNNNNN.h5
    <root>/MS/UDIST/<sp>/<mom>/<mom>-<sp>-NNNNNN.h5   vfl, ufl, T, P, Q
    <root>/MS/RAW/<sp>/RAW-<sp>-NNNNNN.h5             particle dump
    <root>/MS/TRACKS/<sp>-tracks.h5                   tracked particles
    <root>/MS/HIST/par01_ene                          energy history
    <root>/MS/TIMINGS/timings-000010                  timing report

Physics of the synthetic data
-----------------------------
A 1-D shock-like profile in x1, modulated in the transverse direction x2, and
advected in time::

    f(x1, x2, t) = A_f [ 1 + a_f tanh((x1 - x_s(t)) / w) ]
                       [ 1 + eps cos(2 pi m x2 / L2 + phi_f) ]

with the shock front at ``x_s(t) = x_s0 + v_s t``.  Every field is smooth and
analytic, so finite differences, spectral filters and the mean/fluctuation
split all have something meaningful to act on; the density stays positive
(it divides the pressure term of the momentum equation) and the transverse
direction is exactly periodic (``m`` integer), matching the boundary
convention the databases and filters assume: x1 open, x2 periodic.

Nothing here is a physical solution of anything — it is a well-behaved field
with the right shape, units and metadata for the API examples.

Usage
-----
As a script::

    python examples/scripts/synthetic_run.py --root /tmp/osiris_demo

From another example (this is what every script does)::

    from synthetic_run import default_run
    deck = default_run()          # built once, cached under the OS temp dir
    sim = ou.Simulation(deck)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

__all__ = ["build_run", "default_run", "DEFAULTS"]


# --- run parameters --------------------------------------------------------
# Small enough that the whole tree is ~2 MB and every example runs in seconds,
# large enough that a 9-point Savitzky-Golay window and a 5-point derivative
# stencil still fit along x1.


class DEFAULTS:
    """Parameters of the synthetic 2-D run."""

    NX1, NX2 = 64, 16  # grid points (longitudinal, transverse)
    XMAX1, XMAX2 = 12.0, 4.0  # c / omega_p
    DT = 0.02  # 1 / omega_p
    NDUMP = 10  # iterations between dumps
    N_DUMPS = 8  # number of dumps written
    SPECIES = ("electrons", "ions")
    RQM = {"electrons": -1.0, "ions": 100.0}
    SHOCK_X0 = 4.0  # front position at t = 0
    SHOCK_V = 0.6  # front velocity, c
    SHOCK_W = 1.2  # front width, c / omega_p
    TRANSVERSE_MODE = 2  # integer -> exactly periodic in x2
    TRANSVERSE_EPS = 0.15  # fluctuation amplitude


# Moments written for every species.  OSIRIS only dumps the upper triangle of
# the symmetric pressure tensor, so P21/P31/P32 are absent here as well.
MOMENTS = (
    "vfl1",
    "vfl2",
    "vfl3",
    "ufl1",
    "ufl2",
    "ufl3",
    "T11",
    "T12",
    "T13",
    "T22",
    "T23",
    "T33",
    "P00",
    "P11",
    "P12",
    "P13",
    "P22",
    "P23",
    "P33",
    "Q111",
    "Q112",
    "Q113",
    "Q222",
    "Q223",
    "Q333",
)
FIELDS = ("e1", "e2", "e3", "b1", "b2", "b3")

# LaTeX units/labels, so the examples can show ``diag.units`` doing its job.
UNITS = {
    "e": "m_e c \\omega_p e^{-1}",
    "b": "m_e \\omega_p e^{-1}",
    "charge": "e \\omega_p^2 c^{-1}",
    "vfl": "c",
    "ufl": "c",
    "T": "m_e c^2",
    "P": "m_e c^2 \\omega_p^2 c^{-3}",
    "Q": "m_e c^3 \\omega_p^2 c^{-3}",
}
TIME_UNITS = "1 / \\omega_p"


def _units_for(name: str) -> str:
    for prefix, units in UNITS.items():
        if name.startswith(prefix):
            return units
    return ""


def _label_for(name: str) -> str:
    if name == "charge":
        return "\\rho"
    if len(name) == 2 and name[0] in "eb":  # e1 -> E_1
        return f"{name[0].upper()}_{name[1]}"
    if name.startswith(("vfl", "ufl")):  # vfl1 -> v_1
        return f"{name[0]}_{name[3]}"
    return f"{name[0]}_{{{name[1:]}}}"  # T11 -> T_{11}, Q112 -> Q_{112}


# --- analytic profile ------------------------------------------------------


def _noise(name: str, iteration: int, shape: tuple[int, ...]) -> np.ndarray:
    """Multiplicative per-cell noise, reproducible for a given (name, iteration)."""
    rng = np.random.default_rng((sum(name.encode()) * 1000 + iteration) % 2**32)
    return 1.0 + MOMENT_NOISE * rng.standard_normal(shape)


def _mesh(ndims: int) -> list[np.ndarray]:
    """Cell-centred coordinates of each axis (OSIRIS x1, x2, ... ordering)."""
    d = DEFAULTS
    lengths = [d.XMAX1, d.XMAX2, d.XMAX2][:ndims]
    points = [d.NX1, d.NX2, d.NX2 // 2][:ndims]
    return [np.linspace(0.0, L, n, endpoint=False) for L, n in zip(lengths, points, strict=True)]


def _amplitudes(name: str) -> tuple[float, float, float, float, int]:
    """Per-quantity shape parameters, deterministic in the name.

    ``(level, jump fraction, transverse phase, front width, ripple mode)``.
    The width and the longitudinal ripple differ from quantity to quantity so
    the fields are not all multiples of one profile — otherwise every feature
    of a database tensor would be perfectly correlated with every other, and
    examples that look for structure would find something meaningless.
    """
    h = sum(name.encode())
    return (
        1.0 + 0.1 * (h % 5),  # mean level
        0.2 + 0.05 * (h % 4),  # jump fraction across the front
        2 * np.pi * (h % 7) / 7,  # transverse phase
        0.6 + 0.25 * (h % 5),  # front width, in units of SHOCK_W
        1 + (h % 4),  # longitudinal ripple mode
    )


#: Relative amplitude of the particle noise added to species moments.  Fields
#: are left analytic, so the derivative and FFT examples still have an exact
#: reference to check themselves against.
MOMENT_NOISE = 0.02


def field(name: str, iteration: int, ndims: int = 2) -> np.ndarray:
    """Value of *name* at OSIRIS iteration *iteration* on the synthetic grid.

    Species moments carry a few percent of per-cell noise, as a real run does:
    they are estimated from a finite number of macro-particles, which is the
    whole reason the spatial filters exist.  The fields do not — that keeps an
    exact reference available for the derivative and FFT examples.
    """
    d = DEFAULTS
    axes = _mesh(ndims)
    grids = np.meshgrid(*axes, indexing="ij")
    x1 = grids[0]

    amp, jump, phase, width, ripple = _amplitudes(name)
    front = d.SHOCK_X0 + d.SHOCK_V * iteration * d.DT
    profile = amp * (1.0 + jump * np.tanh((x1 - front) / (width * d.SHOCK_W)))
    # A small longitudinal ripple, at a different wavenumber per quantity.
    profile = profile * (1.0 + 0.05 * np.sin(2 * np.pi * ripple * x1 / d.XMAX1 + phase))

    if ndims >= 2:
        # Transverse modulation at a fixed integer mode, so it is exactly
        # periodic in x2 and separable from the x1 structure above; only the
        # phase changes between quantities.
        x2 = grids[1]
        k2 = 2 * np.pi * d.TRANSVERSE_MODE / d.XMAX2
        profile = profile * (1.0 + d.TRANSVERSE_EPS * np.cos(k2 * x2 + phase))

    if name == "charge":
        # OSIRIS writes charge density; Diagnostic("n") flips it by sign(rqm).
        # Keep |charge| well away from zero: it divides the pressure term.
        return (-(2.0 + 0.3 * profile) * _noise(name, iteration, profile.shape)).astype(np.float32)
    if name.startswith(("vfl", "ufl")):  # subluminal
        return (0.3 * np.tanh(profile) * _noise(name, iteration, profile.shape)).astype(np.float32)
    if name[0] in "PTQ":  # the other particle moments
        return (profile * _noise(name, iteration, profile.shape)).astype(np.float32)
    return profile.astype(np.float32)  # fields: exactly analytic


# --- HDF5 writers ----------------------------------------------------------


def _bytes_attr(obj, key: str, value: str) -> None:
    obj.attrs.create(key, [np.bytes_(value.encode())])


def write_grid_file(path: Path, *, name: str, data: np.ndarray, iteration: int, grid: np.ndarray) -> None:
    """Write one grid dump in the layout ``OsirisGridFile`` expects."""
    d = DEFAULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [d.DT])
        sim.attrs.create("NDIMS", [data.ndim])

        f.attrs.create("TIME", [iteration * d.DT])
        _bytes_attr(f, "TIME UNITS", TIME_UNITS)
        f.attrs.create("ITER", [iteration])
        _bytes_attr(f, "NAME", name)
        _bytes_attr(f, "TYPE", "grid")
        _bytes_attr(f, "UNITS", _units_for(name))
        _bytes_attr(f, "LABEL", _label_for(name))

        # OSIRIS stores the array transposed relative to (x1, x2, ...) ordering
        f.create_dataset(name, data=data.T)

        axis_group = f.create_group("AXIS")
        for i in range(data.ndim):
            ax = axis_group.create_dataset(f"AXIS{i + 1}", data=np.asarray(grid[i], dtype=float))
            _bytes_attr(ax, "NAME", f"x{i + 1}")
            _bytes_attr(ax, "UNITS", "c / \\omega_p")
            _bytes_attr(ax, "LONG_NAME", f"x_{i + 1}")
            _bytes_attr(ax, "TYPE", "linear")


def _write_series(directory: Path, *, name: str, prefix: str, iterations: list[int], burst: bool, ndims: int) -> None:
    """Write one diagnostic series.

    ``burst=True`` reproduces OSIRIS burst-dump naming, where the trailing
    index of the filename is the absolute iteration ``n`` rather than the dump
    counter ``n / ndump``.
    """
    d = DEFAULTS
    grid = np.array([[0.0, d.XMAX1], [0.0, d.XMAX2], [0.0, d.XMAX2]][:ndims])
    for n in iterations:
        index = n if burst else n // d.NDUMP
        write_grid_file(
            directory / f"{prefix}-{index:06d}.h5",
            name=name,
            data=field(name, n, ndims),
            iteration=n,
            grid=grid,
        )


def _write_raw(path: Path, species: str, iteration: int, n_particles: int = 512) -> None:
    """Write a RAW particle dump: positions loaded on the synthetic density."""
    d = DEFAULTS
    rng = np.random.default_rng(abs(hash(species)) % 2**32)
    quants = ["x1", "x2", "p1", "p2", "p3", "q", "ene", "tag"]
    units = {
        "x1": "c/\\omega_p",
        "x2": "c/\\omega_p",
        "p1": "m_e c",
        "p2": "m_e c",
        "p3": "m_e c",
        "q": "e",
        "ene": "m_e c^2",
        "tag": "",
    }
    labels = {q: f"{q[0]}_{q[1]}" for q in ("x1", "x2", "p1", "p2", "p3")}
    labels.update({"q": "q", "ene": "Ene", "tag": "Tag"})

    x1 = rng.uniform(0.0, d.XMAX1, n_particles)
    x2 = rng.uniform(0.0, d.XMAX2, n_particles)
    # Downstream (x1 < front) is hotter: a p1 > threshold mask then selects a
    # spatially meaningful subset, which is what tagging is used for.
    front = d.SHOCK_X0 + d.SHOCK_V * iteration * d.DT
    uth = np.where(x1 < front, 0.25, 0.05)
    p1, p2, p3 = (rng.normal(0.0, uth) for _ in range(3))
    gamma = np.sqrt(1.0 + p1**2 + p2**2 + p3**2)

    node = (np.arange(n_particles) % 4) + 1
    node[::8] *= -1  # already-tracked particles carry a negative node id
    tag = np.stack([node, np.arange(1, n_particles + 1)], axis=1)

    values = {"x1": x1, "x2": x2, "p1": p1, "p2": p2, "p3": p3, "q": np.full(n_particles, -1.0), "ene": gamma - 1.0, "tag": tag}

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [d.DT])
        sim.attrs.create("NDIMS", [2])
        sim.attrs.create("XMIN", [0.0, 0.0])
        sim.attrs.create("XMAX", [d.XMAX1, d.XMAX2])

        f.attrs.create("TIME", [iteration * d.DT])
        _bytes_attr(f, "TIME UNITS", TIME_UNITS)
        f.attrs.create("ITER", [iteration])
        _bytes_attr(f, "NAME", species)
        _bytes_attr(f, "TYPE", "particles")
        f.attrs.create("QUANTS", [np.bytes_(q.encode()) for q in quants])
        f.attrs.create("UNITS", [np.bytes_(units[q].encode()) for q in quants])
        f.attrs.create("LABELS", [np.bytes_(labels[q].encode()) for q in quants])

        for q in quants:
            f.create_dataset(q, data=np.asarray(values[q], dtype=np.int32 if q == "tag" else np.float32))


def _write_tracks(path: Path, species: str, n_particles: int = 12, n_iters: int = 20, chunk: int = 5) -> None:
    """Write a 'tracks-2' file: particles crossing the shock front.

    Chunks are interleaved between particles, exactly as OSIRIS writes them,
    so the reordering ``OsirisTrackFile`` performs is genuinely exercised.
    """
    d = DEFAULTS
    quants = ["n", "t", "q", "ene", "x1", "x2", "p1", "p2", "p3"]
    data_quants = quants[1:]
    units = {
        "t": "1/\\omega_p",
        "q": "e",
        "ene": "m_e c^2",
        "x1": "c/\\omega_p",
        "x2": "c/\\omega_p",
        "p1": "m_e c",
        "p2": "m_e c",
        "p3": "m_e c",
    }
    labels = {"t": "t", "q": "q", "ene": "Ene", "x1": "x_1", "x2": "x_2", "p1": "p_1", "p2": "p_2", "p3": "p_3"}

    def value(quant: str, particle: int, k: int) -> float:
        t = k * d.DT
        if quant == "t":
            return t
        if quant == "q":
            return -1.0
        if quant == "x1":  # drifts towards the front, then slows down
            return 0.5 * particle + 0.4 * t
        if quant == "x2":
            return (0.3 * particle + 0.1 * t) % d.XMAX2
        # momentum rotates and grows: the particles gain energy as they go
        amp = 0.2 * (1.0 + 3.0 * t)
        if quant == "p1":
            return amp * np.cos(0.5 * particle + t)
        if quant == "p2":
            return amp * np.sin(0.5 * particle + t)
        if quant == "p3":
            return 0.01 * particle
        if quant == "ene":
            p1, p2, p3 = (value(q, particle, k) for q in ("p1", "p2", "p3"))
            return float(np.sqrt(1 + p1**2 + p2**2 + p3**2) - 1)
        raise KeyError(quant)

    itermap, rows = [], []
    for c in range(n_iters // chunk):
        nstart = c * chunk
        for particle in range(1, n_particles + 1):
            itermap.append([particle, chunk, nstart])
            for offset in range(chunk):
                rows.append([value(q, particle, nstart + offset) for q in data_quants])

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        sim = f.create_group("SIMULATION")
        sim.attrs.create("DT", [d.DT])
        sim.attrs.create("NDIMS", [2])
        sim.attrs.create("XMIN", [0.0, 0.0])
        sim.attrs.create("XMAX", [d.XMAX1, d.XMAX2])

        _bytes_attr(f, "NAME", species)
        _bytes_attr(f, "TYPE", "tracks-2")
        f.attrs.create("NTRACKS", [n_particles])
        f.attrs.create("NITER", [1])
        f.attrs.create("QUANTS", [np.bytes_(q.encode()) for q in quants])
        f.attrs.create("UNITS", [np.bytes_(b"")] + [np.bytes_(units[q].encode()) for q in data_quants])
        f.attrs.create("LABELS", [np.bytes_(b"n")] + [np.bytes_(labels[q].encode()) for q in data_quants])

        f.create_dataset("data", data=np.array(rows, dtype=np.float64))
        f.create_dataset("itermap", data=np.array(itermap, dtype=np.int32))


def _write_hist(path: Path, n_dumps: int) -> None:
    """Write a HIST energy file (``*_ene``): whitespace table, '!' comments."""
    d = DEFAULTS
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["! particle energy diagnostic", "     Iter          Time       KinEnergy"]
    for i in range(n_dumps):
        n = i * d.NDUMP
        # kinetic energy decays as the flow thermalises across the front
        ene = 1.0e-2 * np.exp(-0.05 * n * d.DT)
        lines.append(f"{n:9d}  {n * d.DT:12.6f}  {ene:14.6e}")
    path.write_text("\n".join(lines) + "\n")


def _write_timings(path: Path, iterations: int) -> None:
    """Write a timings report (``timings-NNNNNN``).

    ``OsirisTIMINGS`` reads it with ``sep=r"\\s{2,}"``, takes the iteration
    count from the first line and the column names from the second, and drops
    the rule under the header (file line 3) — so the blank line and the rule
    below are load-bearing, not decoration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("advance deposit", 0.512), ("field solver", 0.128), ("diagnostics", 0.064)]
    lines = [
        f" Timing information for n = {iterations}",
        "",
        "event                     total(s)      avg(s)      min(s)      max(s)",
        "-" * 69,
    ]
    for name, total in rows:
        avg = total / iterations
        lines.append(f"{name:<24}  {total:10.4f}  {avg:10.6f}  {avg * 0.9:10.6f}  {avg * 1.1:10.6f}")
    path.write_text("\n".join(lines) + "\n")


# --- input deck ------------------------------------------------------------


def _deck_text(ndims: int, species: tuple[str, ...], burst: bool) -> str:
    d = DEFAULTS
    nx = [d.NX1, d.NX2, d.NX2 // 2][:ndims]
    xmax = [d.XMAX1, d.XMAX2, d.XMAX2][:ndims]
    # x1 open, x2 (and x3) periodic: the convention the filters and databases assume
    periodic = ", ".join([".false."] + [".true."] * (ndims - 1))
    burst_lines = "\n\tif_use_burst_dump = .true.,\n\tburst_dump_range(1:2) = -1, 1," if burst else ""

    species_blocks = "\n".join(
        f"""species
{{
\tname = "{name}",
\trqm = {d.RQM[name]},
\tnum_par_x(1:{ndims}) = {", ".join(["8"] * ndims)},
\tadd_tag = .true.,
}}

udist
{{
\tuth_type = "thermal",
\tuth(1:3) = 0.1, 0.1, 0.1,
}}

profile
{{
\tprofile_type = "uniform",
\tdensity = 1,
}}

spe_bound
{{
}}

diag_species
{{
\tndump_fac = 1,
\tndump_fac_raw = 1,
\tndump_fac_tracks = 1,
\tniter_tracks = 1,
\treports = "charge",
\trep_udist = {", ".join(f'"{m}"' for m in MOMENTS)},{burst_lines}
}}
"""
        for name in species
    )

    return f"""! Synthetic run written by examples/scripts/synthetic_run.py
simulation
{{
\trandom_seed = 0,
}}

node_conf
{{
\tnode_number(1:{ndims}) = {", ".join(["1"] * ndims)},
\tif_periodic(1:{ndims}) = {periodic},
}}

grid
{{
\tnx_p(1:{ndims}) = {", ".join(map(str, nx))},
\tcoordinates = "cartesian",
}}

time_step
{{
\tdt = {d.DT},
\tndump = {d.NDUMP},
}}

space
{{
\txmin(1:{ndims}) = {", ".join(["0."] * ndims)},
\txmax(1:{ndims}) = {", ".join(str(x) for x in xmax)},
\tif_move(1:{ndims}) = {", ".join([".false."] * ndims)},
}}

time
{{
\ttmin = 0.0d0,
\ttmax = {(d.N_DUMPS - 1) * d.NDUMP * d.DT},
}}

el_mag_fld
{{
}}

emf_bound
{{
\ttype(1:2,1) = "open", "open",
}}

diag_emf
{{
\tndump_fac = 1,
\treports = {", ".join(f'"{f}"' for f in FIELDS)},{burst_lines}
}}

particles
{{
\tinterpolation = "quadratic",
\tnum_species = {len(species)},
}}

{species_blocks}"""


# --- public builders -------------------------------------------------------


def build_run(
    root: str | Path,
    *,
    ndims: int = 2,
    species: tuple[str, ...] = DEFAULTS.SPECIES,
    n_dumps: int = DEFAULTS.N_DUMPS,
    burst: bool = False,
    moments: tuple[str, ...] = MOMENTS,
    fields: tuple[str, ...] = FIELDS,
) -> Path:
    """Write a complete synthetic OSIRIS run and return the path to its deck.

    Parameters
    ----------
    root :
        Directory to write into (created; an existing one is replaced).
    ndims :
        1, 2 or 3.  RAW/TRACKS/HIST are only written for the 2-D run.
    species :
        Species names; each gets ``charge`` plus every moment in *moments*.
    n_dumps :
        Number of ordinary dumps (iterations ``0, ndump, 2*ndump, ...``).
    burst :
        Write burst dumps (also at ``n +/- 1``) and set ``if_use_burst_dump``
        in the deck, so :class:`osiris_utils.BurstConfig` has something to read.
    moments, fields :
        Which quantities to write.  Trim them for a faster, smaller tree.

    Returns
    -------
    Path
        Path of the input deck — the argument :class:`osiris_utils.Simulation`
        takes.
    """
    d = DEFAULTS
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Named "os-stdin" — the usual OSIRIS name, and one of the few the `utils`
    # CLI recognises as a deck rather than a data file (example 13).
    deck = root / "os-stdin"
    deck.write_text(_deck_text(ndims, species, burst))

    dumps = [m * d.NDUMP for m in range(n_dumps)]
    # Burst dumps add the two neighbours of every ordinary dump; iteration -1
    # does not exist, so the first group has no left neighbour.
    iters = sorted({n + k for n in dumps for k in (-1, 0, 1) if n + k >= 0}) if burst else dumps

    ms = root / "MS"
    for fld in fields:
        _write_series(ms / "FLD" / fld, name=fld, prefix=fld, iterations=iters, burst=burst, ndims=ndims)

    for sp in species:
        _write_series(
            ms / "DENSITY" / sp / "charge",
            name="charge",
            prefix=f"charge-{sp}",
            iterations=iters,
            burst=burst,
            ndims=ndims,
        )
        for mom in moments:
            _write_series(
                ms / "UDIST" / sp / mom,
                name=mom,
                prefix=f"{mom}-{sp}",
                iterations=iters,
                burst=burst,
                ndims=ndims,
            )

    if ndims == 2:
        for sp in species:
            _write_raw(ms / "RAW" / sp / f"RAW-{sp}-{dumps[-1]:06d}.h5", sp, dumps[-1])
            _write_tracks(ms / "TRACKS" / f"{sp}-tracks.h5", sp)
        _write_hist(ms / "HIST" / "par01_ene", n_dumps)
        _write_timings(ms / "TIMINGS" / f"timings-{dumps[-1]:06d}", max(dumps[-1], 1))

    return deck


def default_run(*, ndims: int = 2, burst: bool = False, rebuild: bool = False) -> Path:
    """Path to the shared synthetic run, building it on first use.

    Cached under the OS temp directory so repeated example runs are instant.
    Pass ``rebuild=True`` to force a fresh tree.
    """
    tag = f"osiris_utils_examples/{ndims}d{'_burst' if burst else ''}"
    root = Path(tempfile.gettempdir()) / tag
    deck = root / "os-stdin"
    if rebuild or not deck.exists():
        build_run(root, ndims=ndims, burst=burst)
    return deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="where to write the run (default: OS temp cache)")
    parser.add_argument("--ndims", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--burst", action="store_true", help="write burst dumps")
    parser.add_argument("--n-dumps", type=int, default=DEFAULTS.N_DUMPS)
    args = parser.parse_args()

    deck = (
        build_run(args.root, ndims=args.ndims, burst=args.burst, n_dumps=args.n_dumps)
        if args.root
        else default_run(ndims=args.ndims, burst=args.burst, rebuild=True)
    )
    n_files = sum(1 for _ in deck.parent.rglob("*") if _.is_file())
    size_mb = sum(p.stat().st_size for p in deck.parent.rglob("*") if p.is_file()) / 1024**2
    print(f"Wrote {n_files} files ({size_mb:.1f} MB) under {deck.parent}")
    print(f"Input deck: {deck}")


if __name__ == "__main__":
    main()
