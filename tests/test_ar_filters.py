"""Spatial filters applied lazily, per frame, instead of at database-build time.

The database creators smooth every raw frame and then take derivatives with
the filter's own kernel.  ``Filtered_Simulation`` +
``Derivative_Diagnostic(filter=...)`` do the same two things inside the lazy
pipeline, so ``AnomalousResistivity(config=...filters=...)`` reproduces the
tensor the database would have written — one timestep at a time, with nothing
precomputed.  The parity tests below are the real specification.
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.simulation import Simulation
from osiris_utils.database import BurstConfig, DatabaseBuildConfig, DatabaseCreator
from osiris_utils.postprocessing.derivative import Derivative_Diagnostic
from osiris_utils.postprocessing.filtering import Filtered_Diagnostic, Filtered_Simulation

from .test_burst_database import ALL_QUANTITIES, N_DUMPS, NX1, NX2, SPECIES, build_burst_tree

# sigma small enough that the truncate=4 kernel radius fits inside the 8-cell
# transverse axis, which the filter wraps around.
FILTER = ou.GaussianFilter(sigma=1.0)

# Filtered: both sides read float32 dumps, promote to float64 before any
# arithmetic, and then differ only in the order they multiply the same factors.
_PARITY_FILTERED = {"rtol": 1e-9, "atol": 1e-12}

# Unfiltered: the database promotes every frame to float64 as it loads it
# (_load_filtered_fields), while an unfiltered AnomalousResistivity computes on
# the float32 frames as they came off disk.  The two pipelines are the same
# arithmetic at different precision, so parity holds only to float32 eps *of the
# frame's scale* — a pre-existing difference, unrelated to filtering.  Filtering
# removes it: the filtered view promotes on read, exactly like the database.
_FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _raw_tolerance(expected_row: np.ndarray) -> dict[str, float]:
    scale = float(np.abs(expected_row).max())
    return {"rtol": 1e-5, "atol": 20 * _FLOAT32_EPS * scale}


@pytest.fixture
def sim(tmp_path):
    """An ordinary (non-burst) 2-D run: 4 dumps, x1 open, x2 periodic."""
    build_burst_tree(tmp_path / "sim", burst_quantities=set())
    return Simulation(str(tmp_path / "sim" / "shock.2d"))


@pytest.fixture
def burst_sim(tmp_path):
    """The same run with every diagnostic burst-dumped."""
    build_burst_tree(tmp_path / "burst", burst_quantities=ALL_QUANTITIES)
    return Simulation(str(tmp_path / "burst" / "shock.2d"))


def _ar_config(**kwargs):
    defaults = {
        "species": SPECIES,
        "mft_axis": 2,
        "include_time_derivative": False,
        "include_convection": True,
        "include_transverse_advection": True,
        "include_pressure": True,
        "include_magnetic_force": True,
    }
    return ou.AnomalousResistivityConfig(**{**defaults, **kwargs})


# --- the filtered view -----------------------------------------------------


def test_filtered_frame_is_the_smoothed_raw_frame(sim):
    raw = sim[SPECIES]["vfl1"]
    filtered = Filtered_Diagnostic(raw, FILTER)

    # x1 open, x2 periodic — the database's boundary convention
    expected = FILTER.smooth(np.asarray(raw[2], dtype=np.float64), periodic=(False, True))
    np.testing.assert_allclose(filtered[2], expected, rtol=0, atol=0)
    assert filtered[2].shape == (NX1, NX2)


def test_filtering_stays_lazy(sim):
    """Asking for one frame must not pull the whole series into memory."""
    view = Filtered_Simulation(sim, FILTER)
    vfl1 = view[SPECIES]["vfl1"]
    _ = vfl1[1]
    assert not vfl1.all_loaded
    assert not sim[SPECIES]["vfl1"].all_loaded


def test_load_all_smooths_each_frame_separately(sim):
    """The loaded array carries time on axis 0; the kernel must not run along it."""
    view = Filtered_Simulation(sim, FILTER)
    loaded = view[SPECIES]["vfl1"].load_all()
    assert loaded.shape == (N_DUMPS, NX1, NX2)
    for t in range(N_DUMPS):
        np.testing.assert_allclose(loaded[t], view[SPECIES]["vfl1"][t], rtol=0, atol=0)


def test_periodic_axis_follows_mft_axis(sim):
    """mft_axis=1 makes x1 the wrapped axis instead of x2."""
    raw = sim["b3"]
    x2_periodic = Filtered_Simulation(sim, FILTER, mft_axis=2)["b3"][1]
    x1_periodic = Filtered_Simulation(sim, FILTER, mft_axis=1)["b3"][1]

    np.testing.assert_allclose(x2_periodic, FILTER.smooth(np.asarray(raw[1], dtype=np.float64), periodic=(False, True)))
    np.testing.assert_allclose(x1_periodic, FILTER.smooth(np.asarray(raw[1], dtype=np.float64), periodic=(True, False)))
    assert not np.allclose(x1_periodic, x2_periodic)


def test_derived_diagnostics_are_never_smoothed_twice(sim):
    """A filtered view owns its composites; it refuses to re-smooth one."""
    view = Filtered_Simulation(sim, FILTER)
    sp = view[SPECIES]

    nT11 = sp["n"] * sp["T11"]
    sp.add_diagnostic(nT11, "nT11")
    assert sp["nT11"] is nT11  # handed back untouched

    # The same name on the *wrapped* simulation is not silently adopted: it
    # carries whatever filter built it, and smoothing it would double the kernel.
    sim[SPECIES].add_diagnostic(sim[SPECIES]["n"] * sim[SPECIES]["T11"], "raw_nT11")
    with pytest.raises(KeyError, match="derived diagnostic"):
        _ = Filtered_Simulation(sim, FILTER)[SPECIES]["raw_nT11"]


# --- filtered derivatives --------------------------------------------------


def test_spatial_derivative_uses_the_filter_kernel(sim):
    vfl1 = Filtered_Simulation(sim, FILTER)[SPECIES]["vfl1"]
    d = Derivative_Diagnostic(vfl1, "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1, filter=FILTER)

    dx = float(sim[SPECIES]["vfl1"].dx[0])
    np.testing.assert_allclose(d[1], FILTER.derivative(vfl1[1], dx, axis=0, order=1, periodic=False), rtol=0, atol=0)

    # ... and it is a different scheme from the finite differences it replaces
    fd = Derivative_Diagnostic(vfl1, "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
    assert not np.allclose(d[1], fd[1])


def test_eager_and_lazy_filtered_derivatives_agree(sim):
    vfl1 = Filtered_Simulation(sim, FILTER)[SPECIES]["vfl1"]
    lazy = Derivative_Diagnostic(vfl1, "x1", deriv_order=1, filter=FILTER)
    eager = Derivative_Diagnostic(Filtered_Simulation(sim, FILTER)[SPECIES]["vfl1"], "x1", deriv_order=1, filter=FILTER)
    eager.load_all()
    for t in range(N_DUMPS):
        np.testing.assert_allclose(lazy[t], eager.data[t], rtol=1e-12, atol=0)


def test_time_derivative_refuses_a_spatial_filter(sim):
    with pytest.raises(ValueError, match="no derivative scheme along the time axis"):
        Derivative_Diagnostic(sim[SPECIES]["vfl1"], "t", stencil=[-1, 0, 1], filter=FILTER)


# --- AnomalousResistivity --------------------------------------------------


def test_no_filter_is_a_true_no_op(sim):
    """The default path must not change: no wrapper, no new derivative scheme."""
    default = ou.AnomalousResistivity(sim, SPECIES, _ar_config())
    assert isinstance(default.filter, ou.NoFilter)
    assert default.filtered_simulation is default.simulation is sim

    explicit = ou.AnomalousResistivity(sim, SPECIES, _ar_config(filters=ou.NoFilter()))
    assert explicit.filtered_simulation is sim
    np.testing.assert_array_equal(np.asarray(explicit["eta"][1]), np.asarray(default["eta"][1]))


def test_filter_changes_the_terms(sim):
    plain = ou.AnomalousResistivity(sim, SPECIES, _ar_config())
    smooth = ou.AnomalousResistivity(sim, SPECIES, _ar_config(filters=FILTER))

    assert smooth.filter is FILTER
    assert isinstance(smooth.filtered_simulation, Filtered_Simulation)
    assert smooth.simulation is sim
    for key in ("vfl1_avg", "e_vlasov_avg", "eta"):
        assert not np.allclose(np.asarray(smooth[key][1]), np.asarray(plain[key][1]))


def test_filter_namespaces_e_vlasov(sim):
    """Two filters on one Simulation must not share the e_vlasov diagnostic."""
    first = ou.AnomalousResistivity(sim, SPECIES, _ar_config(filters=FILTER))
    second = ou.AnomalousResistivity(sim, SPECIES, _ar_config(filters=ou.GaussianFilter(sigma=0.5)))

    assert first.e_vlasov_key == f"e_vlasov_{SPECIES}"
    assert second.e_vlasov_key != first.e_vlasov_key
    # both remain reachable from the underlying Simulation
    assert sim[first.e_vlasov_key] is not None
    assert sim[second.e_vlasov_key] is not None
    assert not np.allclose(np.asarray(sim[first.e_vlasov_key][1]), np.asarray(sim[second.e_vlasov_key][1]))


# --- parity with the database ----------------------------------------------


def _database(sim, out_dir, filters, ar_config=None, burst=None):
    """eta and e_vlasov tensors, shaped (T, X), from the batch pipeline."""
    creator = DatabaseCreator(
        simulation=sim,
        species=SPECIES,
        save_folder=str(out_dir),
        build_config=DatabaseBuildConfig(
            ar_config=ar_config or _ar_config(),
            filters=filters,
            eta_formula="thesis",
            max_workers=1,
            dtype=np.float64,
            burst=burst,
        ),
    )
    creator.set_limits(initial_iter=0, final_iter=N_DUMPS)
    creator.create_database(database="all", name_output="eta", name_vlasov="e_vlasov")
    return {
        "eta": np.load(out_dir / "eta.npy")[:, 0],
        "e_vlasov_avg": np.load(out_dir / "e_vlasov.npy")[:, 0],
    }


@pytest.mark.parametrize("filters", [(), FILTER], ids=["nofilter", "gaussian"])
def test_filtered_ar_reproduces_the_database_tensors(sim, tmp_path, filters):
    """``ar[key][t]`` equals the database row for frame *t*, filter and all.

    With ``filters=()`` this pins the two pipelines together as they already
    were; with a filter it is the whole point of applying it lazily.
    """
    tensors = _database(sim, tmp_path / "out", filters)
    ar = ou.AnomalousResistivity(sim, SPECIES, _ar_config(filters=filters))

    for key, expected in tensors.items():
        for t in range(N_DUMPS):
            got = np.asarray(ar[key][t], dtype=np.float64).ravel()
            tol = _PARITY_FILTERED if filters else _raw_tolerance(expected[t])
            np.testing.assert_allclose(got, expected[t], err_msg=f"{key} at t={t}", **tol)


def test_database_takes_the_filter_from_ar_config(sim, tmp_path):
    """One config drives both paths, so ar_config.filters must not be ignored."""
    from_ar = _database(sim, tmp_path / "ar", (), ar_config=_ar_config(filters=FILTER))
    from_build = _database(sim, tmp_path / "build", FILTER)
    np.testing.assert_allclose(from_ar["eta"], from_build["eta"], rtol=0, atol=0)


def test_database_refuses_two_different_filters(sim, tmp_path):
    with pytest.raises(ValueError, match="disagree"):
        _database(sim, tmp_path / "clash", ou.GaussianFilter(sigma=0.5), ar_config=_ar_config(filters=FILTER))


def test_filtered_ar_reproduces_the_burst_database(burst_sim, tmp_path):
    """Same parity on burst dumps, with the in-burst ∂v1/∂t term enabled.

    The database differences two *smoothed* frames for ∂/∂t; the lazy side
    reaches the same number by differencing two filtered diagnostics, so the
    time term carries the filter without the filter ever touching the time axis.
    """
    config = _ar_config(include_time_derivative=True, filters=FILTER)
    tensors = _database(burst_sim, tmp_path / "out", FILTER, ar_config=config, burst=BurstConfig())

    ar = ou.AnomalousResistivity(burst_sim, SPECIES, config)
    # Database rows sit on the burst midpoints; map each back to its frame index.
    iterations = burst_sim[SPECIES]["vfl1"].iterations
    frames = [int(np.flatnonzero(iterations == n)[0]) for n in (10, 20, 30)]

    for key, expected in tensors.items():
        for row, t in enumerate(frames):
            got = np.asarray(ar[key][t], dtype=np.float64).ravel()
            np.testing.assert_allclose(got, expected[row], err_msg=f"{key} at midpoint {row}", **_PARITY_FILTERED)

    # the time-derivative term is defined on the midpoints (NaN elsewhere)
    assert np.isfinite(np.asarray(ar["dvfl1_dt_avg"][frames[1]])).all()
