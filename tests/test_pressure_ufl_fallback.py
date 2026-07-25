"""The ufl -> vfl fallback in PressureCorrection must be visible.

`ufl` is proper velocity (gamma*v), `vfl` is fluid velocity. Substituting one
for the other changes the physics of `P_jk - n*u_j*v_k`, so it must not happen
silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou

from .conftest import SPECIES, write_species_moment


def _profile(seed: float):
    def fn(iteration: int, nx: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, nx, endpoint=False)
        return (seed * (1.0 + x) + 0.01 * iteration).astype(np.float32)

    return fn


@pytest.fixture
def sim_without_ufl(sim_dir: Path) -> ou.Simulation:
    """A run that dumped vfl but not ufl — the case that triggers the fallback."""
    for quant, seed in {"vfl1": 0.11, "vfl2": 0.13, "P12": 1.2}.items():
        write_species_moment(sim_dir, quant, _profile(seed))
    return ou.Simulation(str(sim_dir / "thermal.1d"))


@pytest.fixture
def sim_with_ufl(sim_dir: Path) -> ou.Simulation:
    for quant, seed in {"vfl1": 0.11, "vfl2": 0.13, "ufl1": 0.19, "P12": 1.2}.items():
        write_species_moment(sim_dir, quant, _profile(seed))
    return ou.Simulation(str(sim_dir / "thermal.1d"))


def test_missing_ufl_warns_about_the_substitution(sim_without_ufl: ou.Simulation) -> None:
    pc = ou.PressureCorrection_Simulation(sim_without_ufl)

    with pytest.warns(UserWarning, match="ufl1"):
        corrected = pc[SPECIES]["P12"]

    # still usable — the fallback is kept, just no longer silent
    assert corrected[0].shape == sim_without_ufl[SPECIES]["P12"][0].shape


def test_present_ufl_does_not_warn(sim_with_ufl: ou.Simulation) -> None:
    pc = ou.PressureCorrection_Simulation(sim_with_ufl)

    with warnings_as_errors():
        corrected = pc[SPECIES]["P12"]

    sp = sim_with_ufl[SPECIES]
    expected = sp["P12"][1] - sp["n"][1] * sp["ufl1"][1] * sp["vfl2"][1]
    np.testing.assert_allclose(corrected[1], expected, rtol=1e-5)


def test_fallback_uses_vfl_when_ufl_absent(sim_without_ufl: ou.Simulation) -> None:
    pc = ou.PressureCorrection_Simulation(sim_without_ufl)

    with pytest.warns(UserWarning):
        corrected = pc[SPECIES]["P12"]

    sp = sim_without_ufl[SPECIES]
    expected = sp["P12"][1] - sp["n"][1] * sp["vfl1"][1] * sp["vfl2"][1]
    np.testing.assert_allclose(corrected[1], expected, rtol=1e-5)


def test_component_indices_are_not_shared_between_keys(sim_with_ufl: ou.Simulation) -> None:
    """Per-key index parsing must not leak into shared handler state."""
    write_species_moment(Path(sim_with_ufl._simulation_folder), "vfl3", _profile(0.17))
    write_species_moment(Path(sim_with_ufl._simulation_folder), "ufl2", _profile(0.23))
    write_species_moment(Path(sim_with_ufl._simulation_folder), "P23", _profile(2.3))

    handler = ou.PressureCorrection_Simulation(sim_with_ufl)[SPECIES]
    with warnings_as_errors():
        p12 = handler["P12"]
        p23 = handler["P23"]

    sp = sim_with_ufl[SPECIES]
    np.testing.assert_allclose(p12[1], sp["P12"][1] - sp["n"][1] * sp["ufl1"][1] * sp["vfl2"][1], rtol=1e-5)
    np.testing.assert_allclose(p23[1], sp["P23"][1] - sp["n"][1] * sp["ufl2"][1] * sp["vfl3"][1], rtol=1e-5)


class warnings_as_errors:  # noqa: N801
    """Context manager turning warnings into errors, for 'must not warn' tests."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
