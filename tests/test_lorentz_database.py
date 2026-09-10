"""Lorentz-boosted database tensors.

``LorentzDatabaseCreator`` inherits the frame-streaming machinery of
``DatabaseCreator``, so it also inherits the state that machinery reads
(``save_folder`` as a ``Path``, the burst frame-key list).  These tests build a
real 2-D tree and run the creator end to end, which is the only thing that
catches the two halves drifting apart.

The boosted longitudinal derivative mixes in time (note Eq. 36), so the creator
always needs a centered stencil and the tensors are built on burst *midpoints*.
The tree here is written with the OSIRIS burst layout (filename index = absolute
iteration), exactly as ``test_burst_database`` does.

Two closed forms carry most of the assertions:

* ``beta = 0`` is the identity boost (gamma = 1, D = 1), so every transformed
  quantity collapses to the raw one and the rows are the transverse averages of
  the fields on disk.
* ``vfl2`` is untransformed (note Eq. 17), so its boosted derivatives are pure
  Eq. 36 / Eq. 36-squared applied to a field whose lab derivatives are known
  analytically.  Making it *linear in x* keeps the finite differences exact at
  the boundaries too, so the assertions are equalities rather than tolerances.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from osiris_utils.data.simulation import Simulation
from osiris_utils.database import BurstConfig, LorentzDatabaseBuildConfig, LorentzDatabaseCreator
from osiris_utils.database.lorentz_database import LORENTZ_FEATURE_LABELS

from .conftest import write_grid_file

NX1, NX2 = 24, 8
XMAX1, XMAX2 = 6.0, 2.0
DT = 0.05
NDUMP = 10
N_DUMPS = 4  # ordinary dumps at iterations 0, 10, 20, 30
SPECIES = "electrons"

# Every quantity the boost reads (lorentz_database._load_raw_diagnostics),
# plus e1: set_limits() sizes the tensor from simulation["e1"].
FIELDS = ("e1", "e2", "e3", "b2", "b3")
MOMENTS = ("vfl1", "vfl2", "vfl3", "P00", "P11", "P12", "ufl1", "ufl2")

# vfl2 = VFL2_SLOPE * x1 * (1 + t): linear in x1, bilinear in (x1, t).
#   d/dx1 = VFL2_SLOPE * (1 + t),  d/dt = VFL2_SLOPE * x1
#   d2/dx1^2 = 0,  d2/dt2 = 0,  d2/dt dx1 = VFL2_SLOPE
# so d/dx1' and d2/dx1'^2 both have exact closed forms under Eq. 36.
VFL2_SLOPE = 0.02

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


def _x1() -> np.ndarray:
    return np.linspace(0.0, XMAX1, NX1, endpoint=False)


def quantity(name: str, iteration: int, zero_vfl1: bool = False) -> np.ndarray:
    """Smooth 2-D profile, distinct per quantity, positive where it must be.

    ``zero_vfl1`` makes the longitudinal flow vanish, so D = 1 - beta*vfl1 is
    identically 1 and the transverse transform collapses to a constant 1/gamma
    rescale -- which is what keeps the Eq. 36 closed form usable.
    """
    x1 = _x1()[:, None]
    x2 = np.linspace(0.0, XMAX2, NX2, endpoint=False)[None, :]
    t = iteration * DT
    if name == "vfl1" and zero_vfl1:
        return np.zeros((NX1, NX2), dtype=np.float32)
    if name == "vfl2":  # closed-form derivative target, see module docstring
        return np.broadcast_to(VFL2_SLOPE * x1 * (1.0 + t), (NX1, NX2)).astype(np.float32)
    seed = float(sum(name.encode()) % 5)
    value = np.sin(2 * np.pi * x1 / XMAX1 + seed) * np.cos(2 * np.pi * x2 / XMAX2) + 0.1 * t
    if name == "charge":  # n divides P11 -> T11; keep it away from zero
        return (-(2.0 + 0.1 * value)).astype(np.float32)
    if name.startswith(("vfl", "ufl")):  # subluminal, and |1 - beta*v| > 0
        return (0.2 * np.tanh(value)).astype(np.float32)
    if name.startswith("P"):  # pressures are positive
        return (1.5 + 0.2 * value).astype(np.float32)
    return value.astype(np.float32)


def _write_series(directory, *, name: str, prefix: str, iterations: list[int], burst: bool, zero_vfl1: bool = False) -> None:
    """Write one diagnostic series, OSIRIS burst naming when *burst*."""
    for n in iterations:
        file_index = n if burst else n // NDUMP
        path = directory / f"{prefix}-{file_index:06d}.h5"
        write_grid_file(
            path,
            name=name,
            data=quantity(name, n, zero_vfl1),
            iteration=1,  # overwritten below
            dt=DT,
            ndump=NDUMP,
            grid=np.array([[0.0, XMAX1], [0.0, XMAX2]]),
            units="",
            label=name,
        )
        # write_grid_file stores ITER = iteration * ndump; rewrite it so the file
        # carries the true iteration, exactly as OSIRIS does.
        with h5py.File(path, "r+") as f:
            f.attrs["ITER"] = [n]
            f.attrs["TIME"] = [n * DT]


DUMPS = [m * NDUMP for m in range(N_DUMPS)]
BURSTED = sorted({n + k for n in DUMPS for k in (-1, 0, 1) if n + k >= 0})
# Midpoints with both neighbours: 10, 20, 30 -- iteration 0 has no left flank.
MIDPOINT_DUMPS = [1, 2, 3]


def _build_tree(root, *, burst_quantities: set[str] | None = None, zero_vfl1: bool = False) -> None:
    burst_quantities = {*FIELDS, "charge", *MOMENTS} if burst_quantities is None else burst_quantities
    root.mkdir(parents=True, exist_ok=True)
    (root / "shock.2d").write_text(DECK)
    ms = root / "MS"

    def iters(name):
        return BURSTED if name in burst_quantities else DUMPS

    for fld in FIELDS:
        _write_series(ms / "FLD" / fld, name=fld, prefix=fld, iterations=iters(fld), burst=fld in burst_quantities)
    _write_series(
        ms / "DENSITY" / SPECIES / "charge",
        name="charge",
        prefix=f"charge-{SPECIES}",
        iterations=iters("charge"),
        burst="charge" in burst_quantities,
    )
    for mom in MOMENTS:
        _write_series(
            ms / "UDIST" / SPECIES / mom,
            name=mom,
            prefix=f"{mom}-{SPECIES}",
            iterations=iters(mom),
            burst=mom in burst_quantities,
            zero_vfl1=zero_vfl1,
        )


@pytest.fixture
def sim(tmp_path):
    """Every diagnostic burst-dumped."""
    _build_tree(tmp_path / "run")
    return Simulation(str(tmp_path / "run" / "shock.2d"))


@pytest.fixture
def rest_sim(tmp_path):
    """vfl1 = 0 everywhere, so D = 1 and the transverse transform is 1/gamma."""
    _build_tree(tmp_path / "run", zero_vfl1=True)
    return Simulation(str(tmp_path / "run" / "shock.2d"))


@pytest.fixture
def mixed_sim(tmp_path):
    """Only the species moments burst-dumped; the fields keep the coarse cadence."""
    _build_tree(tmp_path / "run", burst_quantities={"charge", *MOMENTS})
    return Simulation(str(tmp_path / "run" / "shock.2d"))


def _build(sim, save_folder, **cfg_kwargs):
    creator = LorentzDatabaseCreator(sim, SPECIES, save_folder, LorentzDatabaseBuildConfig(**cfg_kwargs))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="both")
    return creator


ROW = {label: i for i, label in enumerate(LORENTZ_FEATURE_LABELS)}


def test_tensor_shapes_and_labels(sim, tmp_path):
    """A str save_folder must work: the parent stores it as a Path for _build_tensors."""
    out = tmp_path / "db"
    creator = _build(sim, str(out), seed=0)

    inp = np.load(out / "lorentz_tensor.npy")
    out_t = np.load(out / "lorentz_output.npy")
    betas = np.load(out / "boost_velocities.npy")

    # Iteration 0 has no left flank, so it is dropped: T is 3, not N_DUMPS.
    assert creator.T == len(MIDPOINT_DUMPS)
    assert inp.shape == (len(MIDPOINT_DUMPS), len(LORENTZ_FEATURE_LABELS), NX1)
    assert out_t.shape == (len(MIDPOINT_DUMPS), 1, NX1)
    assert betas.shape == (len(MIDPOINT_DUMPS),)
    assert creator.feature_labels == LORENTZ_FEATURE_LABELS
    assert np.isfinite(inp).all()


def test_rows_sit_on_the_burst_midpoints(sim, tmp_path):
    """Output row i is dump MIDPOINT_DUMPS[i], not raw frame i."""
    out = tmp_path / "db_mid"
    _build(sim, out, boost_min=0.0, boost_max=0.0)
    inp = np.load(out / "lorentz_tensor.npy")

    for i, dump in enumerate(MIDPOINT_DUMPS):
        n_expected = -quantity("charge", dump * NDUMP).mean(axis=1)  # Diagnostic flips charge by sign(rqm)
        assert inp[i, ROW["n_avg"]] == pytest.approx(n_expected, rel=1e-5)
        assert inp[i, ROW["b3_avg"]] == pytest.approx(quantity("b3", dump * NDUMP).mean(axis=1), rel=1e-5)
        assert inp[i, ROW["vfl1_avg"]] == pytest.approx(quantity("vfl1", dump * NDUMP).mean(axis=1), rel=1e-5)


def test_zero_boost_time_derivative_is_the_lab_one(sim, tmp_path):
    """beta = 0: d/dt' = d/dt, the centered in-burst difference."""
    out = tmp_path / "db_dt"
    _build(sim, out, boost_min=0.0, boost_max=0.0)
    inp = np.load(out / "lorentz_tensor.npy")

    h = 2 * DT  # burst_dump_range = -1, 1
    for i, dump in enumerate(MIDPOINT_DUMPS):
        n = dump * NDUMP
        expected = ((quantity("vfl1", n + 1) - quantity("vfl1", n - 1)) / h).mean(axis=1)
        assert inp[i, ROW["dvfl1_dt_avg"]] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.8])
def test_boosted_derivatives_of_vfl2_follow_eq36(rest_sim, tmp_path, beta):
    r"""Closed-form check of Eq. 36, including the ``beta * d/dt`` term.

    With ``vfl1 = 0`` the transverse transform is a constant rescale,
    ``v'_y = v_y / gamma`` (note Eq. 13; see ``_boost_fields``), and
    ``vfl2 = a x (1 + t)``, so

    .. math::
        \partial_{x'} v'_y &= \gamma(\beta\partial_t + \partial_x)(v_y/\gamma)
                             = \beta a x + a(1 + t) \\
        \partial_{x'}^2 v'_y &= \gamma^2(\beta^2\cdot 0 + 2\beta a + 0)/\gamma
                               = 2\gamma\beta a

    The old ``gamma * d/dx`` form misses the ``beta * d/dt`` piece entirely, so
    this fails on it for every beta > 0.
    """
    out = tmp_path / f"db_eq36_{beta}"
    _build(rest_sim, out, boost_min=beta, boost_max=beta)
    inp = np.load(out / "lorentz_tensor.npy")

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    x1 = _x1()
    h = 2 * DT  # burst_dump_range = -1, 1

    for i, dump in enumerate(MIDPOINT_DUMPS):
        t = dump * NDUMP * DT
        d_x = VFL2_SLOPE * (1.0 + t)  # d(vfl2)/dx1
        d_t = VFL2_SLOPE * x1  # d(vfl2)/dt
        assert inp[i, ROW["dvfl2_dx1_avg"]] == pytest.approx(beta * d_t + d_x, rel=1e-4, abs=1e-9)

        # d2/dx1^2 = 0 and d2/dt2 = 0; only the mixed term survives.
        expected_2 = np.full(NX1, 2.0 * gamma * beta * VFL2_SLOPE)
        # Each of the three second-derivative pieces is a cancelling difference
        # divided by a small step, so the float32 dumps put a noise floor under
        # the two whose exact value is 0.  Bound it from the round-off of one
        # sample rather than pretend it is not there.
        dx = XMAX1 / NX1
        sample_eps = np.finfo(np.float32).eps * VFL2_SLOPE * XMAX1 * (1.0 + t)
        noise = gamma * (beta**2 * 4.0 / h**2 + 2.0 * beta / (h * dx) + 1.0 / dx**2) * sample_eps
        assert inp[i, ROW["d2_vfl2_dx1_avg"]] == pytest.approx(expected_2, rel=1e-3, abs=8.0 * noise)


def test_zero_boost_first_derivative_matches_the_filter(sim, tmp_path):
    """beta = 0 collapses Eq. 36 to the plain lab x-derivative."""
    from osiris_utils.filters import NoFilter

    out = tmp_path / "db_dx0"
    _build(sim, out, boost_min=0.0, boost_max=0.0)
    inp = np.load(out / "lorentz_tensor.npy")

    dx = XMAX1 / NX1
    for i, dump in enumerate(MIDPOINT_DUMPS):
        b3 = quantity("b3", dump * NDUMP).astype(np.float64)
        expected = NoFilter().derivative(b3, dx, axis=0, order=1, periodic=False).mean(axis=1)
        assert inp[i, ROW["db2_dx1_avg"]] is not None  # row exists
        assert inp[i, ROW["db3_dx1_avg"]] == pytest.approx(expected, rel=1e-4, abs=1e-9)


def test_eta_is_unchanged_by_the_time_derivative_term(sim, tmp_path):
    r"""``d/dt'`` is linear, so it cancels between <e_vlasov> and the mean-field
    correction: it adds a feature row without moving eta."""
    out = tmp_path / "db_eta"
    _build(sim, out, boost_min=0.3, boost_max=0.3)

    eta = np.load(out / "lorentz_output.npy")
    inp = np.load(out / "lorentz_tensor.npy")
    assert np.isfinite(eta).all()
    # The row is real data, not a zero column that would make the claim vacuous.
    assert np.abs(inp[:, ROW["dvfl1_dt_avg"]]).max() > 1e-6


def test_mixed_cadence_is_refused(mixed_sim, tmp_path):
    """Fields dumped at a different cadence than the moments cannot be boosted:
    n'(t+k) would mix fields sampled at different instants."""
    with pytest.raises(ValueError, match="same flanking iterations"):
        _build(mixed_sim, tmp_path / "db_mixed", seed=0)


def test_burst_config_ndump_fac_is_honoured(sim, tmp_path):
    """deriv_quantities/require_centered are overridden, ndump_fac is not."""
    creator = LorentzDatabaseCreator(
        sim,
        SPECIES,
        tmp_path / "db_cfg",
        LorentzDatabaseBuildConfig(burst=BurstConfig(ndump_fac=2), seed=0),
    )
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="input")
    # stride = ndump * ndump_fac = 20, so only iteration 20 is a midpoint with
    # both flanks (0 has no left, 40 does not exist).
    assert creator.T == 1


def test_seed_makes_the_boost_sequence_reproducible(sim, tmp_path):
    _build(sim, tmp_path / "a", seed=1234)
    _build(sim, tmp_path / "b", seed=1234)
    assert np.array_equal(
        np.load(tmp_path / "a" / "boost_velocities.npy"),
        np.load(tmp_path / "b" / "boost_velocities.npy"),
    )


# --- parity with DatabaseCreator -------------------------------------------


def test_all_builds_input_output_and_e_vlasov(sim, tmp_path):
    """Same database-type vocabulary as DatabaseCreator, including e_vlasov."""
    out = tmp_path / "db_all"
    creator = LorentzDatabaseCreator(sim, SPECIES, out, LorentzDatabaseBuildConfig(seed=0))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="all")

    T = len(MIDPOINT_DUMPS)
    assert np.load(out / "lorentz_tensor.npy").shape == (T, len(LORENTZ_FEATURE_LABELS), NX1)
    assert np.load(out / "lorentz_output.npy").shape == (T, 1, NX1)
    assert np.load(out / "lorentz_e_vlasov.npy").shape == (T, 1, NX1)
    assert creator.e_vlasov_labels == ["e_vlasov_avg"]


def test_both_is_an_alias_for_InOut(sim, tmp_path):
    """The historical spelling keeps working."""
    _build(sim, tmp_path / "a", seed=7)  # database="both"
    creator = LorentzDatabaseCreator(sim, SPECIES, tmp_path / "b", LorentzDatabaseBuildConfig(seed=7))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="InOut")
    assert np.array_equal(
        np.load(tmp_path / "a" / "lorentz_tensor.npy"),
        np.load(tmp_path / "b" / "lorentz_tensor.npy"),
    )


def test_labels_are_derived_from_the_quantity_maps(sim, tmp_path):
    """Rows come from a named dict, so a label with no quantity is a hard error,
    not a silently misaligned row (the failure mode of a parallel np.stack)."""
    from osiris_utils.database.lorentz_database import _BASE_LABELS, _COMPOSITE_LABELS, _D1_LABELS, _D2_LABELS

    assert LORENTZ_FEATURE_LABELS == [*_BASE_LABELS, *_D1_LABELS, *_D2_LABELS, *_COMPOSITE_LABELS]
    assert len(set(LORENTZ_FEATURE_LABELS)) == len(LORENTZ_FEATURE_LABELS)


def test_ar_config_flags_gate_the_terms(sim, tmp_path):
    """Turning the magnetic force off must change eta, exactly as for DatabaseCreator."""
    from osiris_utils.ar import AnomalousResistivityConfig

    kw = {"boost_min": 0.3, "boost_max": 0.3}
    _build(sim, tmp_path / "on", **kw)
    _build(sim, tmp_path / "off", ar_config=AnomalousResistivityConfig(species=SPECIES, include_magnetic_force=False), **kw)

    on = np.load(tmp_path / "on" / "lorentz_output.npy")
    off = np.load(tmp_path / "off" / "lorentz_output.npy")
    assert not np.allclose(on, off)


def test_time_derivative_is_forced_on(sim, tmp_path):
    """include_time_derivative=False is overridden: d/dx' contains d/dt."""
    from osiris_utils.ar import AnomalousResistivityConfig

    creator = LorentzDatabaseCreator(
        sim,
        SPECIES,
        tmp_path / "db_dtflag",
        LorentzDatabaseBuildConfig(ar_config=AnomalousResistivityConfig(species=SPECIES, include_time_derivative=False), seed=0),
    )
    assert creator._resolve_flags().include_time_derivative is True
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="input")
    inp = np.load(tmp_path / "db_dtflag" / "lorentz_tensor.npy")
    assert np.abs(inp[:, ROW["dvfl1_dt_avg"]]).max() > 1e-6


def test_eta_lhs_is_e_vlasov_minus_the_mean_field_equation(sim, tmp_path):
    """The 'lhs' formula is an identity on the rows of the other two tensors,
    so it can be reassembled from them (rqm = -1 for these electrons)."""
    out = tmp_path / "db_lhs"
    creator = LorentzDatabaseCreator(sim, SPECIES, out, LorentzDatabaseBuildConfig(boost_min=0.4, boost_max=0.4))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="all")

    inp = np.load(out / "lorentz_tensor.npy")
    eta = np.load(out / "lorentz_output.npy")[:, 0]
    e_vlasov = np.load(out / "lorentz_e_vlasov.npy")[:, 0]

    rebuilt = (
        e_vlasov
        + inp[:, ROW["dufl1_dt_avg"]]
        + inp[:, ROW["vfl1_avg"]] * inp[:, ROW["dufl1_dx1_avg"]]
        + inp[:, ROW["dnT11_dx1_avg"]] / inp[:, ROW["n_avg"]]
        + inp[:, ROW["vfl2_avg"]] * inp[:, ROW["b3_avg"]]
        - inp[:, ROW["vfl3_avg"]] * inp[:, ROW["b2_avg"]]
    )
    assert eta == pytest.approx(rebuilt, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("beta", [0.0, 0.6])
def test_eta_thesis_equals_eta_lhs(sim, tmp_path, beta):
    """The two formulas are the same number, as they are in DatabaseCreator.

    ``"thesis"`` is the fluctuation expansion of the ``"lhs"`` residual, term
    for term — including the density-fluctuation correction, which is exactly
    ``d/dx'(<n'><T'11>) * (1/<n'> - <1/n'>)``.  They agree to round-off, so a
    transcription error in either decomposition shows up here as a gross
    mismatch rather than a subtle one.
    """
    kw = {"boost_min": beta, "boost_max": beta}
    _build(sim, tmp_path / f"lhs{beta}", eta_formula="lhs", **kw)
    _build(sim, tmp_path / f"thesis{beta}", eta_formula="thesis", **kw)

    lhs = np.load(tmp_path / f"lhs{beta}" / "lorentz_output.npy")
    thesis = np.load(tmp_path / f"thesis{beta}" / "lorentz_output.npy")
    assert np.isfinite(thesis).all()
    assert np.abs(lhs).max() > 1e-3  # not a vacuous comparison of two zero arrays
    assert thesis == pytest.approx(lhs, rel=1e-4, abs=1e-6)


def test_bad_eta_formula_is_refused(sim, tmp_path):
    with pytest.raises(ValueError, match="Invalid eta_formula"):
        _build(sim, tmp_path / "db_bad", eta_formula="nope")


def test_bad_database_type_is_refused(sim, tmp_path):
    creator = LorentzDatabaseCreator(sim, SPECIES, tmp_path / "db_bad2", LorentzDatabaseBuildConfig())
    creator.set_limits(0, N_DUMPS)
    with pytest.raises(ValueError, match="Invalid database"):
        creator.create_database(database="nonsense")


def test_conflicting_filters_are_refused(sim, tmp_path):
    """Same rule as DatabaseCreator: set the filter in one place, or agree."""
    from osiris_utils.ar import AnomalousResistivityConfig
    from osiris_utils.filters import GaussianFilter

    with pytest.raises(ValueError, match="disagree"):
        _build(
            sim,
            tmp_path / "db_filt",
            filters=GaussianFilter(sigma=1.0),
            ar_config=AnomalousResistivityConfig(species=SPECIES, filters=GaussianFilter(sigma=2.0)),
        )


def test_resume_reuses_the_boosts_and_completes(sim, tmp_path):
    """Parity with DatabaseCreator's per-frame resume."""
    out = tmp_path / "db_resume"
    _build(sim, out, seed=99)
    first = np.load(out / "lorentz_tensor.npy").copy()
    betas = np.load(out / "boost_velocities.npy").copy()

    creator = LorentzDatabaseCreator(sim, SPECIES, out, LorentzDatabaseBuildConfig(seed=99, resume=True))
    creator.set_limits(0, N_DUMPS)
    creator.create_database(database="both")

    assert np.array_equal(np.load(out / "boost_velocities.npy"), betas)
    assert np.array_equal(np.load(out / "lorentz_tensor.npy"), first)


def test_vnT_says_why_it_is_not_available(sim, tmp_path):
    """vnT needs 3rd/4th boosted derivatives, hence 4- and 5-point time stencils."""
    creator = LorentzDatabaseCreator(sim, SPECIES, tmp_path / "db_vnt", LorentzDatabaseBuildConfig())
    creator.set_limits(0, N_DUMPS)
    with pytest.raises(NotImplementedError, match="burst_dump_range"):
        creator.create_database(database="vnT")


# ----------------------------------------------------------------------
# Working set
# ----------------------------------------------------------------------
# A frame builder holds (3, nx, ny) float64 triples: 13 lab fields, the boost's
# intermediates, and the 9 boosted fields.  ``_build_tensors`` runs
# ``max_workers`` of them at once, so peak RSS is (workers x this x nx x ny) and
# a production grid (nx1 ~ 4e4) turns a few extra triples into tens of GB per
# worker.  This pins the cost per grid cell so it cannot silently grow again.

#: Bytes of peak working set per grid cell, measured on the reference
#: implementation (432 B = 54 float64 2-D frames).  The bound leaves ~30% head
#: room; a rewrite that keeps the 13 lab triples alive to the end costs 744 B
#: and trips it.
_MAX_BYTES_PER_CELL = 560

_BOOST_INPUTS = ("n", "e2", "e3", "b2", "b3", "vfl1", "vfl2", "vfl3", "P11", "P12", "P00", "ufl1", "ufl2")


def _synthetic_triples(nx: int, ny: int) -> dict[str, np.ndarray]:
    """``(3, nx, ny)`` lab fields with |v| < 1 and n > 0, as float32 on disk is."""
    rng = np.random.default_rng(0)
    f = {name: (rng.random((3, nx, ny)).astype(np.float32) + 1.0) for name in _BOOST_INPUTS}
    for name in ("vfl1", "vfl2", "vfl3"):
        f[name] = (0.3 * (rng.random((3, nx, ny)) - 0.5)).astype(np.float32)
    return f


def test_boost_fields_consumes_its_input():
    """``_boost_fields`` pops as it goes, so the lab triples are freed early."""
    from osiris_utils.database.lorentz_database import _boost_fields

    f = _synthetic_triples(16, 8)
    boosted = _boost_fields(f, 0.6, 1.25)
    assert f == {}, f"lab fields still alive after the boost: {sorted(f)}"
    assert set(boosted) == {"n", "b2", "b3", "vfl1", "ufl1", "vfl2", "vfl3", "T11", "T12"}


@pytest.mark.parametrize(("nx", "ny"), [(128, 64), (256, 64)])
def test_frame_working_set_is_bounded(nx, ny):
    """Peak allocation of one frame build, per grid cell (see _MAX_BYTES_PER_CELL)."""
    import tracemalloc

    from osiris_utils.ar import AnomalousResistivityConfig
    from osiris_utils.database.lorentz_database import _boost_frame_quantities
    from osiris_utils.filters import NoFilter

    raw = _synthetic_triples(nx, ny)
    idx = {name: (0, 1, 2) for name in raw}

    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        _boost_frame_quantities(
            raw,
            idx,
            1.0,
            0.6,
            NoFilter(),
            0.1,
            0.1,
            avg_axis=1,
            flags=AnomalousResistivityConfig(),
            eta_formula="lhs",
            compute_e_vlasov=True,
            compute_eta=True,
            rqm=-1.0,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    per_cell = (peak - start) / (nx * ny)
    assert per_cell < _MAX_BYTES_PER_CELL, f"{per_cell:.0f} B/cell ({per_cell / 8:.0f} float64 frames) at {nx}x{ny}"
