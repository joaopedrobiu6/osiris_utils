"""Regression tests for Diagnostic attribute and persistence behaviour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou
from osiris_utils.data.data import OsirisGridFile
from osiris_utils.data.diagnostic import Diagnostic

from .conftest import N_TIMESTEPS, NDUMP, NX, SPECIES, grid_values


def _e3(sim_dir: Path) -> Diagnostic:
    d = Diagnostic(simulation_folder=str(sim_dir))
    d.get_quantity("e3")
    return d


# --- iter property --------------------------------------------------------


def test_iter_returns_iteration_not_ndump(sim_dir: Path) -> None:
    """`iter` is documented as the iteration number; it must not alias ndump."""
    d = _e3(sim_dir)

    # Without an input deck ndump falls back to 1, while metadata is read from
    # file 000001 whose ITER attribute is 1 * NDUMP — so the two must differ.
    assert d.ndump == 1
    assert d.iter == NDUMP


def test_iter_reads_iteration_from_file_with_deck(sim_dir: Path) -> None:
    sim = ou.Simulation(str(sim_dir / "thermal.1d"))
    d = sim["e3"]

    assert d.ndump == NDUMP
    assert d.iter == NDUMP


def test_iter_is_settable_and_readable_back() -> None:
    d = Diagnostic()
    d.ndump = 7

    d.iter = 42

    assert d.iter == 42
    assert d.ndump == 7


def test_iter_defaults_to_none_on_bare_diagnostic() -> None:
    assert Diagnostic().iter is None


def test_iter_survives_arithmetic(sim_dir: Path) -> None:
    d = _e3(sim_dir)

    doubled = d * 2.0

    assert doubled.iter == d.iter


# --- data property --------------------------------------------------------


def test_data_property_raises_valueerror_when_not_loaded() -> None:
    d = Diagnostic()

    with pytest.raises(ValueError, match="not loaded"):
        _ = d.data


def test_data_property_returns_array_after_load(sim_dir: Path) -> None:
    d = _e3(sim_dir)
    d.load_all(use_parallel=False)

    assert d.data.shape == (N_TIMESTEPS, NX)


# --- to_h5 ----------------------------------------------------------------


def test_to_h5_with_explicit_path_roundtrips(sim_dir: Path, tmp_path: Path) -> None:
    d = _e3(sim_dir)
    out = tmp_path / "out"

    d.to_h5(savename="e3copy", index=1, path=str(out))

    written = out / "e3copy-000001.h5"
    assert written.exists()

    reread = OsirisGridFile(str(written))
    np.testing.assert_allclose(reread.data, d[1], rtol=1e-6)
    assert reread.dim == d.dim
    assert reread.dt == pytest.approx(d.dt)
    np.testing.assert_allclose(reread.grid, d.grid)
    assert reread.nx == NX


def test_to_h5_default_path_writes_into_simulation_folder(sim_dir: Path) -> None:
    """path=None must fall back to the simulation folder rather than crash."""
    d = _e3(sim_dir)

    d.to_h5(savename="e3copy", index=0)

    matches = list((sim_dir / "MS" / "MISC").rglob("e3copy-000000.h5"))
    assert matches, "expected a file under the simulation folder's MS/MISC tree"
    np.testing.assert_allclose(OsirisGridFile(str(matches[0])).data, d[0], rtol=1e-6)


def test_to_h5_all_writes_every_timestep(sim_dir: Path, tmp_path: Path) -> None:
    d = _e3(sim_dir)
    out = tmp_path / "series"

    d.to_h5(savename="e3all", all=True, path=str(out))

    for i in range(N_TIMESTEPS):
        written = out / f"e3all-{i:06d}.h5"
        assert written.exists()
        np.testing.assert_allclose(OsirisGridFile(str(written)).data, grid_values(i), rtol=1e-6)


def test_to_h5_list_of_indices(sim_dir: Path, tmp_path: Path) -> None:
    d = _e3(sim_dir)
    out = tmp_path / "some"

    d.to_h5(savename="e3some", index=[0, 2], path=str(out))

    assert (out / "e3some-000000.h5").exists()
    assert (out / "e3some-000002.h5").exists()
    assert not (out / "e3some-000001.h5").exists()


def test_to_h5_without_index_or_all_raises(sim_dir: Path, tmp_path: Path) -> None:
    """Writing nothing at all is a silent failure — say so instead."""
    d = _e3(sim_dir)
    out = tmp_path / "nothing"

    with pytest.raises(ValueError, match="index"):
        d.to_h5(savename="e3none", path=str(out))


def test_to_h5_defaults_savename_to_diagnostic_name(sim_dir: Path, tmp_path: Path) -> None:
    d = _e3(sim_dir)
    out = tmp_path / "named"

    d.to_h5(index=0, path=str(out))

    assert (out / "e3-000000.h5").exists()


def test_to_h5_roundtrips_a_postprocessed_diagnostic(sim_dir: Path, tmp_path: Path) -> None:
    """Saving the result of an operation is the main use case for to_h5."""
    sim = ou.Simulation(str(sim_dir / "thermal.1d"))
    combined = sim[SPECIES]["charge"] * 2.0
    out = tmp_path / "misc"

    combined.to_h5(savename="charge2", index=1, path=str(out))

    reread = OsirisGridFile(str(out / "charge2-000001.h5"))
    np.testing.assert_allclose(reread.data, combined[1], rtol=1e-6)
