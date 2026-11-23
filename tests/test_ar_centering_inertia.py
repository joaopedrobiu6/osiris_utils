"""AnomalousResistivityConfig.center_fields and .inertia.

``center_fields`` runs the whole analysis on fields averaged from the Yee mesh
onto the cell nodes, where the moments live.  ``inertia="ufl"`` differentiates
``ufl1 = <gamma v1>`` instead of ``vfl1`` in the inertial terms.  Whatever the
options, ``LHS == eta`` must keep holding: that identity is the analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.simulation import Simulation
from osiris_utils.database import DatabaseBuildConfig, DatabaseCreator
from osiris_utils.postprocessing.derivative import Derivative_Diagnostic
from osiris_utils.postprocessing.field_centering import FieldCentering_Diagnostic, FieldCentering_Simulation

from .conftest import write_run_tree
from .test_burst_database import DECK, DT, N_DUMPS, NDUMP, NX1, NX2, SPECIES, XMAX1, XMAX2, build_burst_tree, field

MOMENTS = ("vfl1", "vfl2", "vfl3", "ufl1", "T11", "T12")


@pytest.fixture
def sim(tmp_path):
    """A regular 2-D run, x1 open and x2 periodic, that also dumps ufl1."""
    write_run_tree(
        tmp_path / "sim",
        deck=DECK.replace('rep_udist = "vfl1",', 'rep_udist = "ufl1", "vfl1",'),
        fields=("e1", "b2", "b3"),
        moments=MOMENTS,
        species=SPECIES,
        data=field,
        dumps=[m * NDUMP for m in range(N_DUMPS)],
        burst_quantities=set(),
        dt=DT,
        ndump=NDUMP,
        grid=np.array([[0.0, XMAX1], [0.0, XMAX2]]),
    )
    return Simulation(str(tmp_path / "sim" / "shock.2d"))


def _config(**kwargs):
    defaults = {
        "species": SPECIES,
        "mft_axis": 2,
        "method": "P",
        "include_time_derivative": True,
        "include_convection": True,
        "include_transverse_advection": True,
        "include_pressure": True,
        "include_magnetic_force": True,
    }
    return ou.AnomalousResistivityConfig(**{**defaults, **kwargs})


def _row(diag, t):
    return np.asarray(diag[t], dtype=np.float64).ravel()


def _center(a, axes, wrap):
    """Average with the left neighbour along each numpy axis in *axes*; edge replicated unless it wraps."""
    for ax in axes:
        if ax in wrap:
            left = np.roll(a, 1, axis=ax)
        else:
            left = np.concatenate([np.take(a, [0], axis=ax), np.take(a, range(a.shape[ax] - 1), axis=ax)], axis=ax)
        a = 0.5 * (a + left)
    return a


# --- the config ------------------------------------------------------------


def test_defaults_keep_the_historical_equation():
    cfg = ou.AnomalousResistivityConfig()
    assert cfg.center_fields is False
    assert cfg.inertia == "vfl"


def test_unknown_inertia_is_refused():
    with pytest.raises(ValueError, match="inertia"):
        ou.AnomalousResistivityConfig(inertia="gamma")


def test_ufl_inertia_needs_ufl1_in_the_dumps(tmp_path):
    build_burst_tree(tmp_path / "no_ufl", burst_quantities=set())
    no_ufl = Simulation(str(tmp_path / "no_ufl" / "shock.2d"))
    with pytest.raises(ValueError, match="ufl1"):
        ou.AnomalousResistivity(no_ufl, SPECIES, _config(inertia="ufl"))


# --- the identity ----------------------------------------------------------


@pytest.mark.parametrize("method", ["P", "nT"])
@pytest.mark.parametrize("center_fields", [False, True], ids=["yee", "centred"])
@pytest.mark.parametrize("inertia", ["vfl", "ufl"])
def test_lhs_equals_eta(sim, method, center_fields, inertia):
    ar = ou.AnomalousResistivity(sim, SPECIES, _config(method=method, center_fields=center_fields, inertia=inertia))
    for t in (0, 2):  # a one-sided d/dt frame and a centred one
        lhs, eta = _row(ar["LHS"], t), _row(ar["eta"], t)
        np.testing.assert_allclose(lhs, eta, rtol=0, atol=1e-5 * np.abs(eta).max())


def test_options_get_their_own_e_vlasov(sim):
    """One Simulation, four configs: none may reuse another's e_vlasov."""
    keys = {
        ou.AnomalousResistivity(sim, SPECIES, _config(center_fields=c, inertia=i)).e_vlasov_key
        for c in (False, True)
        for i in ("vfl", "ufl")
    }
    assert len(keys) == 4


# --- center_fields ---------------------------------------------------------


def test_centred_view_averages_the_fields_onto_the_nodes(sim):
    view = ou.AnomalousResistivity(sim, SPECIES, _config(center_fields=True)).filtered_simulation
    assert isinstance(view, FieldCentering_Simulation)

    # 2-D Yee mesh: e1, b2 half a cell off along x1; b3 along x1 and x2.  x2 wraps.
    for name, axes in (("e1", (0,)), ("b2", (0,)), ("b3", (0, 1))):
        raw = np.asarray(sim[name][2], dtype=np.float64)
        np.testing.assert_allclose(view[name][2], _center(raw, axes, wrap={1}), rtol=1e-6, err_msg=name)  # float32 frames

    # the moments pass through untouched
    np.testing.assert_array_equal(view[SPECIES]["vfl2"][2], sim[SPECIES]["vfl2"][2])


def test_centred_fields_enter_the_lorentz_term(sim):
    ar = ou.AnomalousResistivity(sim, SPECIES, _config(center_fields=True))
    plain = ou.AnomalousResistivity(sim, SPECIES, _config())

    v2 = np.asarray(sim[SPECIES]["vfl2"][1], dtype=np.float64)
    b3 = _center(np.asarray(sim["b3"][1], dtype=np.float64), (0, 1), wrap={1})
    expected = -((v2 * b3).mean(axis=1) - v2.mean(axis=1) * b3.mean(axis=1))  # electrons: -<dv2 dB3>

    np.testing.assert_allclose(_row(ar["mag_v2_b3"], 1), expected, rtol=1e-5, atol=1e-7)
    assert not np.allclose(_row(plain["mag_v2_b3"], 1), expected, rtol=1e-5, atol=1e-7)


def test_field_centering_wraps_only_the_named_axes(sim):
    raw = np.asarray(sim["b3"][1], dtype=np.float64)
    only_x2 = FieldCentering_Diagnostic(sim["b3"], periodic=(2,))[1]
    np.testing.assert_allclose(only_x2, _center(raw, (0, 1), wrap={1}), rtol=1e-6)
    assert not np.allclose(only_x2, FieldCentering_Diagnostic(sim["b3"], periodic=True)[1])


def test_fields_only_view_still_refuses_other_keys(sim):
    with pytest.raises(ValueError, match="center"):
        _ = FieldCentering_Simulation(sim)["vfl1"]


# --- inertia ---------------------------------------------------------------


def test_ufl_inertia_differentiates_ufl1(sim):
    ar = ou.AnomalousResistivity(sim, SPECIES, _config(inertia="ufl"))
    sp = sim[SPECIES]

    du1_dx1 = Derivative_Diagnostic(sp["ufl1"], "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
    du1_dt = Derivative_Diagnostic(sp["ufl1"], "t", stencil=[-1, 0, 1], deriv_order=1)
    v1 = np.asarray(sp["vfl1"][1], dtype=np.float64)
    d1u = np.asarray(du1_dx1[1], dtype=np.float64)

    # electrons: conv enters eta as -|rqm| <dv1 d(d1 u1)>, the lhs as |rqm| <v1> d1 <u1>
    conv = -((v1 * d1u).mean(axis=1) - v1.mean(axis=1) * d1u.mean(axis=1))
    np.testing.assert_allclose(_row(ar["conv_v1_dv1dx1"], 1), conv, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_row(ar["lhs_conv_v1_dv1dx1"], 1), v1.mean(axis=1) * d1u.mean(axis=1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_row(ar["lhs_dvfl1_dt"], 1), np.asarray(du1_dt[1], dtype=np.float64).mean(axis=1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_row(ar["ufl1_avg"], 1), np.asarray(sp["ufl1"][1], dtype=np.float64).mean(axis=1), rtol=1e-6)
    assert "dufl1_dt_avg" in ar.available_terms


# --- the databases ---------------------------------------------------------


@pytest.mark.parametrize("option", [{"center_fields": True}, {"inertia": "ufl"}], ids=["center_fields", "ufl"])
def test_database_refuses_options_it_does_not_implement(sim, tmp_path, option):
    creator = DatabaseCreator(
        simulation=sim,
        species=SPECIES,
        save_folder=str(tmp_path / "out"),
        build_config=DatabaseBuildConfig(ar_config=_config(include_time_derivative=False, **option), eta_formula="thesis", max_workers=1, dtype=np.float64),
    )
    creator.set_limits(initial_iter=0, final_iter=N_DUMPS)
    with pytest.raises(NotImplementedError, match="database builders"):
        creator.create_database(database="all", name_output="eta", name_vlasov="e_vlasov")
