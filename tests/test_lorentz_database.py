"""Lorentz-boosted database tensors.

``LorentzDatabaseCreator`` inherits the frame-streaming machinery of
``DatabaseCreator``, so it also inherits the state that machinery reads
(``save_folder`` as a ``Path``, the burst frame-key list).  These tests build a
real 2-D tree and run the creator end to end, which is the only thing that
catches the two halves drifting apart.

With ``beta = 0`` the boost is the identity (gamma = 1, D = 1), so every
transformed quantity collapses to the raw one and the tensor rows are exactly
the transverse averages of the fields on disk — a closed form to assert on.
"""

from __future__ import annotations

import numpy as np
import pytest

from osiris_utils.data.simulation import Simulation
from osiris_utils.database import LorentzDatabaseBuildConfig, LorentzDatabaseCreator
from osiris_utils.database.lorentz_database import LORENTZ_FEATURE_LABELS

from .conftest import write_grid_series

NX1, NX2 = 24, 8
XMAX1, XMAX2 = 6.0, 2.0
DT = 0.05
NDUMP = 10
N_DUMPS = 3
SPECIES = "electrons"

# Every quantity the boost reads (lorentz_database._load_raw_diagnostics),
# plus e1: set_limits() sizes the tensor from simulation["e1"].
FIELDS = ("e1", "e2", "e3", "b2", "b3")
MOMENTS = ("vfl1", "vfl2", "vfl3", "P00", "P11", "P12", "ufl1", "ufl2")

DECK = f"""
node_conf
{{
	node_number(1:2) = 1, 1,
	if_periodic(1:2) = .false., .true.,
}}
grid
{{
	nx_p(1:2) = {NX1}, {NX2},
}}
time_step
{{
	dt = {DT},
	ndump = {NDUMP},
}}
space
{{
	xmin(1:2) = 0., 0.,
	xmax(1:2) = {XMAX1}, {XMAX2},
}}
particles
{{
	num_species = 1,
}}
species
{{
	name = "{SPECIES}",
	rqm = -1.0,
}}
"""


def quantity(name: str, iteration: int) -> np.ndarray:
    """Smooth 2-D profile, distinct per quantity, positive where it must be."""
    x1 = np.linspace(0.0, XMAX1, NX1, endpoint=False)[:, None]
    x2 = np.linspace(0.0, XMAX2, NX2, endpoint=False)[None, :]
    seed = float(sum(name.encode()) % 5)
    value = np.sin(2 * np.pi * x1 / XMAX1 + seed) * np.cos(2 * np.pi * x2 / XMAX2) + 0.1 * iteration
    if name == "charge":  # n divides P11 -> T11; keep it away from zero
        return (-(2.0 + 0.1 * value)).astype(np.float32)
    if name.startswith(("vfl", "ufl")):  # subluminal, and |1 - beta*v| > 0
        return (0.2 * np.tanh(value)).astype(np.float32)
    if name.startswith("P"):  # pressures are positive
        return (1.5 + 0.2 * value).astype(np.float32)
    return value.astype(np.float32)


@pytest.fixture
def sim(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "shock.2d").write_text(DECK)
    grid = np.array([[0.0, XMAX1], [0.0, XMAX2]])
    kwargs = {"n_timesteps": N_DUMPS, "grid": grid, "dt": DT, "ndump": NDUMP}

    ms = root / "MS"
    for fld in FIELDS:
        write_grid_series(
            ms / "FLD" / fld,
            name=fld,
            prefix=fld,
            data_fn=lambda i, _nx, f=fld: quantity(f, i),
            units="",
            label=fld,
            **kwargs,
        )
    write_grid_series(
        ms / "DENSITY" / SPECIES / "charge",
        name="charge",
        prefix=f"charge-{SPECIES}",
        data_fn=lambda i, _nx: quantity("charge", i),
        units="",
        label="charge",
        **kwargs,
    )
    for mom in MOMENTS:
        write_grid_series(
            ms / "UDIST" / SPECIES / mom,
            name=mom,
            prefix=f"{mom}-{SPECIES}",
            data_fn=lambda i, _nx, m=mom: quantity(m, i),
            units="",
            label=mom,
            **kwargs,
        )
    return Simulation(str(root / "shock.2d"))


def _build(sim, save_folder, **cfg_kwargs):
    creator = LorentzDatabaseCreator(sim, SPECIES, save_folder, LorentzDatabaseBuildConfig(**cfg_kwargs))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="both")
    return creator


def test_tensor_shapes_and_labels(sim, tmp_path):
    """A str save_folder must work: the parent stores it as a Path for _build_tensors."""
    out = tmp_path / "db"
    creator = _build(sim, str(out), seed=0)

    inp = np.load(out / "lorentz_tensor.npy")
    out_t = np.load(out / "lorentz_output.npy")
    betas = np.load(out / "boost_velocities.npy")

    assert inp.shape == (N_DUMPS, len(LORENTZ_FEATURE_LABELS), NX1)
    assert out_t.shape == (N_DUMPS, 1, NX1)
    assert betas.shape == (N_DUMPS,)
    assert creator.feature_labels == LORENTZ_FEATURE_LABELS
    assert np.isfinite(inp).all()


def test_zero_boost_rows_are_the_raw_transverse_averages(sim, tmp_path):
    """beta = 0 is the identity boost, so the rows have a closed form."""
    out = tmp_path / "db0"
    _build(sim, out, boost_min=0.0, boost_max=0.0)

    inp = np.load(out / "lorentz_tensor.npy")
    row = {label: i for i, label in enumerate(LORENTZ_FEATURE_LABELS)}

    for t in range(N_DUMPS):
        # n' = gamma n (1 - beta v) = n; the Diagnostic flips charge by sign(rqm)
        assert inp[t, row["n_avg"]] == pytest.approx(-quantity("charge", t).mean(axis=1), rel=1e-5)
        assert inp[t, row["b3_avg"]] == pytest.approx(quantity("b3", t).mean(axis=1), rel=1e-5)
        assert inp[t, row["vfl1_avg"]] == pytest.approx(quantity("vfl1", t).mean(axis=1), rel=1e-5)


def test_seed_makes_the_boost_sequence_reproducible(sim, tmp_path):
    _build(sim, tmp_path / "a", seed=1234)
    _build(sim, tmp_path / "b", seed=1234)
    assert np.array_equal(
        np.load(tmp_path / "a" / "boost_velocities.npy"),
        np.load(tmp_path / "b" / "boost_velocities.npy"),
    )
