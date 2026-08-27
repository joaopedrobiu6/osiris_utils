"""Burst-dump support: iteration axis, BurstAxis, and midpoint databases.

A synthetic 2-D run is written with the exact file layout OSIRIS produces when
a report uses ``if_use_burst_dump`` with ``burst_dump_range = -1, 1``: the
filename holds the *absolute* iteration instead of the dump counter, so the
series is ``0, 1, 9, 10, 11, 19, 20, 21, ...`` for ``ndump = 10``.

Every field is linear in time, so the centered in-burst difference reproduces
∂/∂t exactly and the assertions can be equalities rather than tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from osiris_utils.data.simulation import Simulation
from osiris_utils.database import BurstAxis, BurstConfig, DatabaseBuildConfig, DatabaseCreator, describe_axes
from osiris_utils.database.database import TIME_DERIVATIVE_LABEL, input_feature_labels

from .conftest import write_grid_file

NX1, NX2 = 24, 8
XMAX1, XMAX2 = 6.0, 2.0
DT = 0.05
NDUMP = 10
N_DUMPS = 4  # ordinary dumps: iterations 0, 10, 20, 30
SPECIES = "electrons"

DECK = f"""
simulation
{{
	random_seed = 0,
}}
node_conf
{{
	node_number(1:2) = 1, 1,
	if_periodic(1:2) = .false., .true.,
}}
grid
{{
	nx_p(1:2) = {NX1}, {NX2},
	coordinates = "cartesian",
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
time
{{
	tmin = 0.0d0,
	tmax = 2.0,
}}
diag_emf
{{
	ndump_fac = 1,
	reports = "e1", "b2", "b3",
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
diag_species
{{
	ndump_fac = 1,
	reports = "charge",
	rep_udist = "vfl1", "vfl2", "vfl3", "T11", "T12",
}}
"""


# --- analytic fields -------------------------------------------------------
# f(x, n) = base(x) + rate(x) * (n * dt).  d/dt is `rate`, exactly, for any
# centered difference, so the tests assert on closed forms.


def _mesh() -> tuple[np.ndarray, np.ndarray]:
    x1 = np.linspace(0.0, XMAX1, NX1, endpoint=False)
    x2 = np.linspace(0.0, XMAX2, NX2, endpoint=False)
    return np.meshgrid(x1, x2, indexing="ij")


def _base(seed: float) -> np.ndarray:
    X1, X2 = _mesh()
    return np.sin(2 * np.pi * X1 / XMAX1) * np.cos(2 * np.pi * X2 / XMAX2) + seed


def _rate(seed: float) -> np.ndarray:
    X1, X2 = _mesh()
    return 0.3 * np.cos(2 * np.pi * X1 / XMAX1) + 0.05 * seed


def field(name: str, iteration: int) -> np.ndarray:
    seed = float(sum(name.encode()) % 7)
    value = _base(seed) + _rate(seed) * (iteration * DT)
    if name == "charge":  # density must stay positive: it divides the pressure term
        value = 2.0 + 0.1 * value
    return value.astype(np.float32)


def _write_series(directory, *, name: str, prefix: str, iterations: list[int], burst: bool) -> None:
    """Write one diagnostic series.

    ``burst=True`` reproduces OSIRIS burst naming (filename index = absolute
    iteration); otherwise the filename index is the dump counter ``n / ndump``.
    """
    for n in iterations:
        file_index = n if burst else n // NDUMP
        write_grid_file(
            directory / f"{prefix}-{file_index:06d}.h5",
            name=name,
            data=field(name, n),
            iteration=1,  # overwritten below
            dt=DT,
            ndump=NDUMP,
            grid=np.array([[0.0, XMAX1], [0.0, XMAX2]]),
            units="",
            label=name,
        )
        # write_grid_file stores ITER = iteration * ndump; rewrite it so the file
        # carries the true iteration, exactly as OSIRIS does.
        import h5py

        with h5py.File(directory / f"{prefix}-{file_index:06d}.h5", "r+") as f:
            f.attrs["ITER"] = [n]
            f.attrs["TIME"] = [n * DT]


def build_burst_tree(root, *, burst_quantities: set[str]) -> tuple[list[int], list[int]]:
    """Write a synthetic run; quantities in *burst_quantities* get burst dumps."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "shock.2d").write_text(DECK)

    dumps = [m * NDUMP for m in range(N_DUMPS)]
    bursted = sorted({n + k for n in dumps for k in (-1, 0, 1) if n + k >= 0})

    ms = root / "MS"
    for fld in ("e1", "b2", "b3"):
        iters = bursted if fld in burst_quantities else dumps
        _write_series(ms / "FLD" / fld, name=fld, prefix=fld, iterations=iters, burst=fld in burst_quantities)

    iters = bursted if "charge" in burst_quantities else dumps
    _write_series(
        ms / "DENSITY" / SPECIES / "charge",
        name="charge",
        prefix=f"charge-{SPECIES}",
        iterations=iters,
        burst="charge" in burst_quantities,
    )

    for moment in ("vfl1", "vfl2", "vfl3", "T11", "T12"):
        iters = bursted if moment in burst_quantities else dumps
        _write_series(
            ms / "UDIST" / SPECIES / moment,
            name=moment,
            prefix=f"{moment}-{SPECIES}",
            iterations=iters,
            burst=moment in burst_quantities,
        )
    return dumps, bursted


ALL_QUANTITIES = {"e1", "b2", "b3", "charge", "vfl1", "vfl2", "vfl3", "T11", "T12"}


@pytest.fixture
def burst_sim(tmp_path):
    """Every diagnostic burst-dumped (deck with burst on diag_emf and diag_species)."""
    build_burst_tree(tmp_path / "sim", burst_quantities=ALL_QUANTITIES)
    return Simulation(str(tmp_path / "sim" / "shock.2d"))


@pytest.fixture
def mixed_sim(tmp_path):
    """Only the species moments burst-dumped; the fields keep the coarse cadence."""
    build_burst_tree(tmp_path / "sim", burst_quantities={"charge", "vfl1", "vfl2", "vfl3", "T11", "T12"})
    return Simulation(str(tmp_path / "sim" / "shock.2d"))


# --- iteration axis --------------------------------------------------------


def test_iteration_axis_detects_burst_naming(burst_sim):
    vfl1 = burst_sim[SPECIES]["vfl1"]
    assert vfl1.iterations.tolist() == [0, 1, 9, 10, 11, 19, 20, 21, 29, 30, 31]
    assert vfl1._iter_stride == 1
    assert vfl1.index_of_iteration(20) == 6
    # time() follows the true iteration, not index * dt * ndump
    assert vfl1.time(6)[0] == pytest.approx(20 * DT)


def test_iteration_axis_unchanged_for_regular_dumps(mixed_sim):
    b2 = mixed_sim["b2"]
    assert b2.iterations.tolist() == [0, 10, 20, 30]
    assert b2._iter_stride == NDUMP
    assert b2.time(2)[0] == pytest.approx(20 * DT)


def test_time_derivative_refuses_non_uniform_frames(burst_sim):
    """A uniform-grid d/dt over a bursted series must fail loudly, not silently."""
    from osiris_utils.postprocessing.derivative import Derivative_Diagnostic, _uniform_frame_dt

    vfl1 = burst_sim[SPECIES]["vfl1"]
    with pytest.raises(ValueError, match="non-uniform frame spacing"):
        _uniform_frame_dt(vfl1)

    d = Derivative_Diagnostic(vfl1, "t", order=2)
    with pytest.raises(RuntimeError) as exc:  # Diagnostic.__getitem__ wraps loader errors
        _ = d[1]
    assert isinstance(exc.value.__cause__, ValueError)
    assert "non-uniform frame spacing" in str(exc.value.__cause__)


def test_uniform_frame_dt_unchanged_for_regular_dumps(mixed_sim):
    from osiris_utils.postprocessing.derivative import _uniform_frame_dt

    assert _uniform_frame_dt(mixed_sim["b2"]) == pytest.approx(DT * NDUMP)


# --- BurstAxis -------------------------------------------------------------


def _raw(sim):
    sp = sim[SPECIES]
    return {
        "n": sp["n"],
        "T11": sp["T11"],
        "T12": sp["T12"],
        "vfl1": sp["vfl1"],
        "vfl2": sp["vfl2"],
        "vfl3": sp["vfl3"],
        "b2": sim["b2"],
        "b3": sim["b3"],
    }


def test_burst_axis_groups_on_midpoints(burst_sim):
    axis = BurstAxis(_raw(burst_sim), dt=DT, ndump=NDUMP, config=BurstConfig())
    # iteration 0 has no left neighbour, so require_centered drops it
    assert axis.midpoints.tolist() == [10, 20, 30]
    assert axis.dump_indices.tolist() == [1, 2, 3]
    assert axis.n_groups == 3

    st = axis.stencil("vfl1", 0)  # midpoint n = 10 → frames n = 9 and n = 11
    assert st.centered
    assert st.h == pytest.approx(2 * DT)
    vfl1 = burst_sim[SPECIES]["vfl1"]
    assert vfl1.iterations[st.i_lo] == 9
    assert vfl1.iterations[st.i_hi] == 11


def test_burst_axis_keeps_first_group_when_one_sided_allowed(burst_sim):
    axis = BurstAxis(_raw(burst_sim), dt=DT, ndump=NDUMP, config=BurstConfig(require_centered=False))
    assert axis.midpoints.tolist() == [0, 10, 20, 30]
    st = axis.stencil("vfl1", 0)
    assert not st.centered
    assert st.h == pytest.approx(DT)  # forward difference 0 → 1


def test_burst_axis_aligns_mixed_cadences(mixed_sim):
    """A bursted moment and a non-bursted field land on the same physical time."""
    axis = BurstAxis(_raw(mixed_sim), dt=DT, ndump=NDUMP, config=BurstConfig())
    idx = axis.indices(1)  # midpoint iteration 20
    assert mixed_sim[SPECIES]["vfl1"].iterations[idx["vfl1"]] == 20
    assert mixed_sim["b2"].iterations[idx["b2"]] == 20
    assert idx["vfl1"] != idx["b2"]  # different file counts — the whole point


def test_describe_axes_reports_both_kinds(mixed_sim):
    text = describe_axes(_raw(mixed_sim), ndump=NDUMP)
    assert "burst" in text
    assert "regular" in text


# --- end-to-end database ---------------------------------------------------


def _build(sim, tmp_path, *, time_derivative: bool, burst: BurstConfig | None = None):
    import osiris_utils as ou

    burst = burst or BurstConfig()
    ar = ou.AnomalousResistivityConfig(
        species=SPECIES,
        mft_axis=2,
        include_time_derivative=time_derivative,
        include_convection=True,
        include_transverse_advection=False,
        include_pressure=True,
        include_magnetic_force=True,
    )
    db = DatabaseCreator(
        simulation=sim,
        species=SPECIES,
        save_folder=str(tmp_path / "out"),
        build_config=DatabaseBuildConfig(ar_config=ar, burst=burst, max_workers=1, dtype=np.float64),
    )
    db.set_limits(initial_iter=0, final_iter=N_DUMPS)
    db.create_database(database="input", name_input="input_tensor")
    return db, np.load(tmp_path / "out" / "input_tensor.npy")


def test_database_time_derivative_is_exact_at_the_midpoint(burst_sim, tmp_path):
    db, tensor = _build(burst_sim, tmp_path, time_derivative=True)

    labels = input_feature_labels(db.build_config.ar_config)
    assert labels[-1] == TIME_DERIVATIVE_LABEL
    assert tensor.shape == (3, len(labels), NX1)  # 3 groups: iterations 10, 20, 30

    # dvfl1/dt is `_rate` transversely averaged, at every midpoint
    expected = _rate(float(sum(b"vfl1") % 7)).mean(axis=1)
    for t in range(tensor.shape[0]):
        # float32 dumps: differencing two nearby frames costs ~4 digits
        np.testing.assert_allclose(tensor[t, -1], expected, rtol=2e-4, atol=1e-5)


def test_database_frames_sit_on_the_midpoints(burst_sim, tmp_path):
    """The non-derivative rows come from the midpoint frame, not a burst edge."""
    _, tensor = _build(burst_sim, tmp_path, time_derivative=True)
    labels = input_feature_labels(None)
    row = labels.index("vfl1_avg")
    for t, n_mid in enumerate((10, 20, 30)):
        np.testing.assert_allclose(tensor[t, row], field("vfl1", n_mid).mean(axis=1), rtol=1e-5, atol=1e-6)


def test_mixed_cadence_database_matches_fully_bursted(mixed_sim, burst_sim, tmp_path):
    """Bursting only the species moments gives the same tensor as bursting everything."""
    _, mixed = _build(mixed_sim, tmp_path / "a", time_derivative=True)
    _, full = _build(burst_sim, tmp_path / "b", time_derivative=True)
    np.testing.assert_allclose(mixed, full, rtol=1e-6, atol=1e-8)


def test_time_derivative_changes_e_vlasov_by_exactly_minus_dv_dt(burst_sim, tmp_path):
    import osiris_utils as ou

    def evlasov(time_derivative: bool, out: str):
        ar = ou.AnomalousResistivityConfig(
            species=SPECIES,
            mft_axis=2,
            include_time_derivative=time_derivative,
            include_transverse_advection=False,
        )
        db = DatabaseCreator(
            simulation=burst_sim,
            species=SPECIES,
            save_folder=str(tmp_path / out),
            build_config=DatabaseBuildConfig(ar_config=ar, burst=BurstConfig(), max_workers=1, dtype=np.float64),
        )
        db.set_limits(initial_iter=0, final_iter=N_DUMPS)
        db.create_database(database="e_vlasov", name_vlasov="e_vlasov_tensor")
        return np.load(tmp_path / out / "e_vlasov_tensor.npy")

    with_dt = evlasov(True, "with")
    without_dt = evlasov(False, "without")
    expected = _rate(float(sum(b"vfl1") % 7)).mean(axis=1)
    np.testing.assert_allclose(without_dt[:, 0] - with_dt[:, 0], np.broadcast_to(expected, without_dt[:, 0].shape), rtol=2e-4, atol=1e-5)


def test_time_derivative_without_burst_config_is_refused(burst_sim, tmp_path):
    import osiris_utils as ou

    ar = ou.AnomalousResistivityConfig(species=SPECIES, include_time_derivative=True)
    db = DatabaseCreator(
        simulation=burst_sim,
        species=SPECIES,
        save_folder=str(tmp_path / "out"),
        build_config=DatabaseBuildConfig(ar_config=ar, burst=None),
    )
    db.set_limits(initial_iter=0, final_iter=N_DUMPS)
    with pytest.raises(NotImplementedError, match="burst dumps"):
        db.create_database(database="input")


# --- in-burst time derivatives ---------------------------------------------


def test_frame_dt_axis_is_scalar_on_a_uniform_axis(mixed_sim):
    """A regular series keeps the historical single-step fast path."""
    from osiris_utils.postprocessing.derivative import _frame_dt_axis

    h, valid = _frame_dt_axis(mixed_sim["b2"], [-1, 0, 1], 1)
    assert valid is None
    assert h == pytest.approx(DT * NDUMP)


def test_burst_time_derivative_is_exact_at_midpoints(burst_sim):
    """(v[n+1] - v[n-1]) / (2 dt) on the midpoints, NaN everywhere else.

    The synthetic fields are linear in time, so the centered in-burst difference
    reproduces d/dt exactly and the midpoints can be checked against the closed
    form rather than a tolerance.
    """
    from osiris_utils.postprocessing.derivative import Derivative_Diagnostic

    vfl1 = burst_sim[SPECIES]["vfl1"]
    d = Derivative_Diagnostic(vfl1, "t", stencil=[-1, 0, 1], deriv_order=1)
    d.load_all()
    got = np.asarray(d.data)

    iterations = np.asarray(vfl1.iterations)
    # midpoints with both burst neighbours present; n = 0 has no n = -1
    expected_valid = {n for n in iterations if n % NDUMP == 0 and n - 1 in set(iterations) and n + 1 in set(iterations)}
    assert expected_valid, "fixture should provide at least one fully-flanked midpoint"

    rate = _rate(float(sum(b"vfl1") % 7))
    for i, n in enumerate(iterations):
        if int(n) in expected_valid:
            # atol as well as rtol: the analytic rate crosses zero, and the
            # inputs are stored float32, so a pure relative test fails there.
            assert np.allclose(got[i], rate, rtol=1e-4, atol=1e-5), f"iteration {n} should reproduce dv/dt"
        else:
            assert np.isnan(got[i]).all(), f"iteration {n} has no valid stencil and must be NaN"


def test_burst_time_derivative_matches_between_eager_and_lazy_paths(burst_sim):
    from osiris_utils.postprocessing.derivative import Derivative_Diagnostic

    vfl1 = burst_sim[SPECIES]["vfl1"]
    eager = Derivative_Diagnostic(vfl1, "t", stencil=[-1, 0, 1], deriv_order=1)
    eager.load_all()
    lazy = Derivative_Diagnostic(vfl1, "t", stencil=[-1, 0, 1], deriv_order=1)
    for i in range(len(vfl1.iterations)):
        np.testing.assert_allclose(np.asarray(lazy[i]), np.asarray(eager.data)[i])


def test_burst_stencil_must_fit_inside_the_burst_window(burst_sim):
    """burst_dump_range = -1, 1 gives one neighbour per side: no 5-point stencil."""
    from osiris_utils.postprocessing.derivative import _frame_dt_axis

    vfl1 = burst_sim[SPECIES]["vfl1"]
    _, valid_3 = _frame_dt_axis(vfl1, [-1, 0, 1], 1)
    _, valid_5 = _frame_dt_axis(vfl1, [-2, -1, 0, 1, 2], 1)
    assert valid_3.sum() > 0
    assert valid_5.sum() == 0


# --- species-aware momentum equation ---------------------------------------


def test_rqm_is_read_from_the_deck(burst_sim):
    from osiris_utils.utils import resolve_rqm

    assert resolve_rqm(burst_sim, SPECIES) == pytest.approx(-1.0)
    assert resolve_rqm(burst_sim, SPECIES, override=32.0) == pytest.approx(32.0)


def test_e_vlasov_is_linear_in_rqm(mixed_sim):
    """e_vlasov = rqm * (inertial + pressure) - (v x B); only the first part scales."""
    import osiris_utils as ou

    def ev(rqm):
        cfg = ou.AnomalousResistivityConfig(species=SPECIES)
        ar = ou.AnomalousResistivity(mixed_sim, SPECIES, cfg, rqm=rqm)
        return np.asarray(ar["e_vlasov_avg"][1], dtype=np.float64).ravel()

    e0, e1 = ev(0.0), ev(1.0)
    inertial_and_pressure, magnetic = e1 - e0, -e0
    for rqm in (-1.0, 32.0):
        np.testing.assert_allclose(ev(rqm), rqm * inertial_and_pressure - magnetic, rtol=1e-9, atol=1e-12)


def test_eta_follows_the_species_normalisation(mixed_sim):
    """eta = -|rqm| * non-magnetic fluctuations + sign(rqm) * magnetic ones."""
    import osiris_utils as ou

    def eta(rqm, key):
        cfg = ou.AnomalousResistivityConfig(species=SPECIES)
        ar = ou.AnomalousResistivity(mixed_sim, SPECIES, cfg, rqm=rqm)
        return np.asarray(ar[key][1], dtype=np.float64).ravel()

    for key in ("eta", "eta_new"):
        plus, minus = eta(1.0, key), eta(-1.0, key)
        magnetic, non_magnetic = (plus - minus) / 2, -(plus + minus) / 2
        for rqm in (32.0, -32.0):
            expected = -abs(rqm) * non_magnetic + np.sign(rqm) * magnetic
            np.testing.assert_allclose(eta(rqm, key), expected, rtol=1e-9, atol=1e-12)


def test_e_vlasov_is_namespaced_per_species(mixed_sim):
    """Two species on one Simulation must not share the e_vlasov diagnostic."""
    import osiris_utils as ou

    cfg = ou.AnomalousResistivityConfig(species=SPECIES)
    ar = ou.AnomalousResistivity(mixed_sim, SPECIES, cfg)
    assert ar.e_vlasov_key == f"e_vlasov_{SPECIES}"
    assert mixed_sim[ar.e_vlasov_key] is not None


@pytest.fixture
def regular_sim(tmp_path):
    """Nothing burst-dumped: an ordinary run, so d/dt spans whole dumps."""
    build_burst_tree(tmp_path / "sim", burst_quantities=set())
    return Simulation(str(tmp_path / "sim" / "shock.2d"))


def _ar(sim, *, time_derivative: bool):
    import osiris_utils as ou

    cfg = ou.AnomalousResistivityConfig(
        species=SPECIES,
        mft_axis=2,
        include_time_derivative=time_derivative,
        include_convection=True,
        include_transverse_advection=True,
        include_pressure=True,
        include_magnetic_force=True,
    )
    return ou.AnomalousResistivity(sim, SPECIES, cfg, rqm=-1.0)


# LHS = <e_vlasov> minus the mean-field equation, eta = the same thing written as
# fluctuation cross-terms: the two must agree.  float32 dumps differenced by a
# 5-point stencil put the floor a few ulps above machine precision.
_LHS_ETA_RTOL = 1e-5


def _lhs_minus_eta(ar, t: int = 2) -> float:
    edge = 3  # x1 boundary cells the 5-point stencil cannot fill
    lhs = np.asarray(ar["LHS"][t], dtype=np.float64)[edge:-edge]
    eta = np.asarray(ar["eta"][t], dtype=np.float64)[edge:-edge]
    return float(np.abs(lhs - eta).max() / np.abs(eta).max())


@pytest.mark.parametrize("time_derivative", [True, False])
def test_lhs_equals_eta(regular_sim, time_derivative):
    """The two routes to the turbulent term agree, with or without ∂t v1."""
    assert _lhs_minus_eta(_ar(regular_sim, time_derivative=time_derivative)) < _LHS_ETA_RTOL


def test_two_configs_on_one_simulation_do_not_share_e_vlasov(regular_sim):
    """The second config must not inherit the first one's e_vlasov.

    It used to: ``_ensure_diagnostic`` is idempotent by name, so the object built
    second kept a ⟨∂t v1⟩ it never subtracted and its LHS drifted away from its
    eta while the first object's stayed right.
    """
    with_dt = _ar(regular_sim, time_derivative=True)
    without_dt = _ar(regular_sim, time_derivative=False)

    assert with_dt.e_vlasov_key != without_dt.e_vlasov_key
    assert _lhs_minus_eta(with_dt) < _LHS_ETA_RTOL
    assert _lhs_minus_eta(without_dt) < _LHS_ETA_RTOL

    # ...and the shared LHS identity itself: ⟨∂t v1⟩ enters e_vlasov linearly and
    # is removed with the same weight, so enabling the term cannot move LHS.
    np.testing.assert_allclose(
        np.asarray(with_dt["LHS"][2], dtype=np.float64),
        np.asarray(without_dt["LHS"][2], dtype=np.float64),
        rtol=1e-5,
        atol=1e-8,
    )
