"""Regression tests for the heat-flux correction.

This module previously had no test coverage at all, which is why a constructor
arity mismatch and a non-symmetric pressure-tensor lookup both survived.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou

from .conftest import N_TIMESTEPS, NX, SPECIES, write_species_moment

# Every quantity gets a distinct, analytically known profile so a mis-wired
# argument produces a visibly wrong number rather than a plausible one.
_QUANTITIES = {
    "vfl1": 0.11,
    "vfl2": 0.13,
    "vfl3": 0.17,
    "P11": 1.1,
    "P12": 1.2,
    "P13": 1.3,
    "P22": 2.2,
    "P23": 2.3,
    "P33": 3.3,
    "Q111": 5.1,
    "Q112": 5.2,
    "Q113": 5.3,
    "Q222": 5.4,
    "Q223": 5.5,
    "Q333": 5.6,
}


def _profile(seed: float):
    """Distinct smooth profile per quantity, varying in space and time."""

    def fn(iteration: int, nx: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, nx, endpoint=False)
        return (seed * (1.0 + x) + 0.01 * iteration).astype(np.float32)

    return fn


@pytest.fixture
def heatflux_sim(sim_dir: Path) -> ou.Simulation:
    """Simulation tree carrying every moment the heat-flux correction needs."""
    for quant, seed in _QUANTITIES.items():
        write_species_moment(sim_dir, quant, _profile(seed))
    return ou.Simulation(str(sim_dir / "thermal.1d"))


def _reference(sim: ou.Simulation, comp: str, index: int) -> np.ndarray:
    """Q_ijk - (v_i P_jk + v_j P_ki + v_k P_ij) + 2 v_i v_j v_k n, built by hand."""
    i, j, k = (int(c) for c in comp[1:])
    sp = sim[SPECIES]

    def P(a: int, b: int) -> np.ndarray:
        return sp[f"P{min(a, b)}{max(a, b)}"][index]

    v = {a: sp[f"vfl{a}"][index] for a in (i, j, k)}
    n = sp["n"][index]
    Q = sp[comp][index]
    return Q - (v[i] * P(j, k) + v[j] * P(k, i) + v[k] * P(i, j)) + 2 * v[i] * v[j] * v[k] * n


@pytest.mark.parametrize("comp", ["Q111", "Q222", "Q333", "Q112", "Q113", "Q223"])
def test_heatflux_correction_can_be_constructed(heatflux_sim: ou.Simulation, comp: str) -> None:
    """Construction went through a call site that passed 5 args to a 4-arg __init__."""
    corrected = ou.HeatfluxCorrection_Simulation(heatflux_sim)[SPECIES][comp]

    assert corrected.name == f"{comp}_corrected"


@pytest.mark.parametrize("comp", ["Q111", "Q112", "Q223"])
def test_heatflux_correction_matches_reference(heatflux_sim: ou.Simulation, comp: str) -> None:
    """The mixed components exercise the symmetric P_ki lookup (P21, P32, ...)."""
    corrected = ou.HeatfluxCorrection_Simulation(heatflux_sim)[SPECIES][comp]

    np.testing.assert_allclose(corrected[2], _reference(heatflux_sim, comp, 2), rtol=1e-5)


def test_heatflux_correction_eager_matches_lazy(heatflux_sim: ou.Simulation) -> None:
    hf = ou.HeatfluxCorrection_Simulation(heatflux_sim)
    lazy = np.stack([hf[SPECIES]["Q112"][i] for i in range(N_TIMESTEPS)])

    eager = ou.HeatfluxCorrection_Simulation(heatflux_sim)[SPECIES]["Q112"].load_all()

    assert eager.shape == (N_TIMESTEPS, NX)
    np.testing.assert_allclose(eager, lazy, rtol=1e-5)


def test_heatflux_correction_rejects_non_heatflux_quantity(heatflux_sim: ou.Simulation) -> None:
    with pytest.raises(ValueError, match="Invalid heatflux component"):
        _ = ou.HeatfluxCorrection_Simulation(heatflux_sim)["P12"]
