"""Parallel .npy export.

``export_to_npy`` has two code paths: a plain one for file-backed diagnostics
and a stencil-aware one for *derived* ones (a derivative, a product), which
prefetches frames into the diagnostic's frame cache before evaluating each
output frame.  The prefetch reaches into ``Diagnostic``'s caching internals, so
these tests exercise it end to end — the derived path is the one that breaks
when the two sides drift apart.

Every assertion compares the exported array against the lazy path, which the
rest of the suite already pins to closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.simulation import Simulation

from .conftest import N_TIMESTEPS, NX


@pytest.fixture
def sim(sim_dir):
    return Simulation(str(sim_dir / "thermal.1d"))


def test_full_export_matches_the_lazy_path(sim, tmp_path):
    e3 = sim["e3"]
    path = ou.export_to_npy(e3, tmp_path / "e3.npy", show_progress=False)

    data = np.load(path)
    assert data.shape == (N_TIMESTEPS, NX)
    for i in range(N_TIMESTEPS):
        assert data[i] == pytest.approx(e3[i])


def test_reductions(sim, tmp_path):
    e3 = sim["e3"]
    full = e3[:]

    per_frame = np.load(ou.export_to_npy(e3, tmp_path / "a.npy", reduce_axis=0, show_progress=False))
    assert per_frame.shape == (N_TIMESTEPS,)
    assert per_frame == pytest.approx(full.mean(axis=1), rel=1e-6)

    time_avg = np.load(ou.export_to_npy(e3, tmp_path / "b.npy", time_average=True, show_progress=False))
    assert time_avg.shape == (NX,)
    assert time_avg == pytest.approx(full.mean(axis=0), rel=1e-5)

    both = np.load(ou.export_to_npy(e3, tmp_path / "c.npy", reduce_axis=0, time_average=True, show_progress=False))
    assert both.shape == ()
    assert float(both) == pytest.approx(full.mean(), rel=1e-5)


def test_derived_diagnostic_export(sim, tmp_path):
    """The stencil-aware path: a derivative reads neighbouring frames per output frame."""
    d = ou.Derivative_Diagnostic(sim["e3"], "x1", order=4)
    data = np.load(ou.export_to_npy(d, tmp_path / "d.npy", show_progress=False))

    assert data.shape == (N_TIMESTEPS, NX)
    reference = ou.Derivative_Diagnostic(sim["e3"], "x1", order=4)
    for i in range(N_TIMESTEPS):
        assert data[i] == pytest.approx(reference[i])


def test_expression_export(sim, tmp_path):
    """An arithmetic result carries _frame as an instance attribute — also 'derived'."""
    expr = sim["electrons"]["n"] * sim["electrons"]["vfl1"]
    data = np.load(ou.export_to_npy(expr, tmp_path / "e.npy", show_progress=False))
    for i in range(N_TIMESTEPS):
        assert data[i] == pytest.approx(expr[i])


def test_overwrite_is_refused_by_default(sim, tmp_path):
    out = tmp_path / "f.npy"
    ou.export_to_npy(sim["e3"], out, show_progress=False)
    with pytest.raises(FileExistsError):
        ou.export_to_npy(sim["e3"], out, show_progress=False)
    ou.export_to_npy(sim["e3"], out, overwrite=True, show_progress=False)


def test_export_simulation_writes_one_file_per_quantity(sim, tmp_path):
    written = ou.export_simulation_to_npy(sim, ["e3", "electrons/vfl1"], tmp_path / "bulk", show_progress=False)
    assert written["e3"].name == "e3.npy"
    assert written["electrons/vfl1"].name == "electrons_vfl1.npy"
    assert np.load(written["electrons/vfl1"]).shape == (N_TIMESTEPS, NX)
