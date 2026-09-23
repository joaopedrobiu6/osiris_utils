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


def test_loaded_operands_give_a_result_with_its_own_frames(sim_dir, tmp_path):
    """a * b of two loaded diagnostics is computed eagerly.  Its frames used to be
    read from a's files (it inherited a's file list), so exporting or
    differentiating it silently worked on a."""
    loaded = Simulation(str(sim_dir / "thermal.1d"))
    a, b = loaded["e3"], loaded["electrons"]["vfl1"]
    a.load_all()
    b.load_all()
    c = a * b

    fresh = Simulation(str(sim_dir / "thermal.1d"))
    lazy = fresh["e3"] * fresh["electrons"]["vfl1"]
    np.testing.assert_allclose(c._frame(1), lazy[1])
    np.testing.assert_allclose(np.load(ou.export_to_npy(c, tmp_path / "c.npy", show_progress=False)), lazy[:])
    np.testing.assert_allclose(ou.Derivative_Diagnostic(c, "x1")[2], ou.Derivative_Diagnostic(lazy, "x1")[2])


@pytest.mark.parametrize("time_average", [False, True])
def test_an_interrupted_export_resumes(sim, tmp_path, time_average):
    """Checkpoints every chunk_size frames; a rerun computes only what is missing."""
    e3 = sim["e3"] * 1.0  # derived: frames computed in order
    out = tmp_path / "r.npy"
    calls: list[int] = []
    frame = e3._frame

    def flaky(i, data_slice=None):
        if i == 4 and not calls.count(-1):
            calls.append(-1)
            raise OSError("node died")
        calls.append(i)
        return frame(i, data_slice)

    e3._frame = flaky
    with pytest.raises(OSError, match="node died"):
        ou.export_to_npy(e3, out, chunk_size=2, time_average=time_average, show_progress=False)
    calls.clear()
    calls.append(-1)  # the failure happened once; let the rerun through
    result = np.load(ou.export_to_npy(e3, out, chunk_size=2, time_average=time_average, show_progress=False))

    full = sim["e3"][:]
    np.testing.assert_allclose(result, full.mean(axis=0) if time_average else full, rtol=1e-6)
    assert calls[1:] == [0, 4, 5]  # probe frame 0, then only the frames after the last checkpoint
    assert not out.with_suffix(".progress").exists()


def test_chain_follows_arithmetic_and_post_processing(sim):
    from osiris_utils.io.export import _collect_chain, _prefetch_reach

    n, v = sim["electrons"]["n"], sim["electrons"]["vfl1"]
    d = ou.Derivative_Diagnostic(n * v, "t", order=4)
    chain = _collect_chain(d)
    assert any(x is n for x in chain) and any(x is v for x in chain)
    assert _prefetch_reach(chain) == 2  # the 5-point time stencil reads two frames ahead
    assert _prefetch_reach(_collect_chain(ou.Derivative_Diagnostic(n, "x1"))) == 0
