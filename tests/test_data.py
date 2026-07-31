from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from osiris_utils.data.data import OsirisGridFile
from osiris_utils.data.diagnostic import Diagnostic
from osiris_utils.decks.species import Species

from .conftest import DT, N_TIMESTEPS, NDUMP, NX, SPECIES, XMAX, XMIN, grid_values, write_grid_file


def test_osiris_grid_file(sim_dir: Path) -> None:
    file_path = sim_dir / "MS" / "DENSITY" / SPECIES / "charge" / f"charge-{SPECIES}-000000.h5"
    assert file_path.exists()

    grid_file = OsirisGridFile(str(file_path))

    assert grid_file.name == "charge"
    assert grid_file.type == "grid"
    assert grid_file.dim == 1
    assert grid_file.iter == 0
    assert np.isclose(grid_file.time[0], 0.0)

    assert grid_file.data.ndim == 1
    assert grid_file.data.shape == (NX,)
    np.testing.assert_allclose(grid_file.data, grid_values(0), rtol=1e-6)

    assert grid_file.nx == NX
    assert np.isclose(grid_file.dx, (XMAX - XMIN) / NX)
    np.testing.assert_allclose(grid_file.grid, [XMIN, XMAX])


def test_osiris_grid_file_metadata_and_axis(sim_dir: Path) -> None:
    grid_file = OsirisGridFile(str(sim_dir / "MS" / "FLD" / "e3" / "e3-000002.h5"))

    assert grid_file.label == "E_3"
    assert grid_file.units == "m_e c \\omega_p / e"
    assert grid_file.dt == pytest.approx(DT)
    assert grid_file.iter == 2 * NDUMP
    assert grid_file.time[0] == pytest.approx(2 * DT * NDUMP)
    assert grid_file.time[1] == "1 / \\omega_p"

    assert len(grid_file.axis) == 1
    assert grid_file.axis[0]["name"] == "x1"
    assert grid_file.axis[0]["long_name"] == "x_1"
    assert grid_file.axis[0]["units"] == "c / \\omega_p"
    assert grid_file.axis[0]["type"] == "linear"


def test_osiris_grid_file_metadata_only_skips_data(sim_dir: Path) -> None:
    grid_file = OsirisGridFile(str(sim_dir / "MS" / "FLD" / "e3" / "e3-000000.h5"), load_data=False)

    assert grid_file.data is None
    assert grid_file.nx == NX
    assert grid_file.dim == 1


def test_diagnostic_integration(sim_dir: Path) -> None:
    elec = Species(name=SPECIES, rqm=-1.0)
    diag = Diagnostic(simulation_folder=str(sim_dir), species=elec)
    diag.get_quantity("charge")

    assert diag.maxiter == N_TIMESTEPS

    data = diag.load_all()
    assert diag.all_loaded
    assert data is not None
    assert data.shape == (N_TIMESTEPS, NX)
    for i in range(N_TIMESTEPS):
        np.testing.assert_allclose(data[i], grid_values(i), rtol=1e-6)

    d0 = diag[0]
    assert d0.shape == data[0].shape
    np.testing.assert_allclose(d0, data[0])

    diag_sum = diag + diag
    assert isinstance(diag_sum, Diagnostic)
    np.testing.assert_allclose(diag_sum.load_all(), data * 2)


def test_diagnostic_density_flips_sign_with_rqm(sim_dir: Path) -> None:
    """The 'n' quantity is charge scaled by sign(rqm) — negative for electrons."""
    elec = Species(name=SPECIES, rqm=-1.0)

    charge = Diagnostic(simulation_folder=str(sim_dir), species=elec)
    charge.get_quantity("charge")

    density = Diagnostic(simulation_folder=str(sim_dir), species=elec)
    density.get_quantity("n")

    np.testing.assert_allclose(density[0], -charge[0], rtol=1e-6)


def test_diagnostic_rejects_unknown_quantity(sim_dir: Path) -> None:
    diag = Diagnostic(simulation_folder=str(sim_dir), species=Species(name=SPECIES, rqm=-1.0))
    with pytest.raises(ValueError, match="Invalid quantity"):
        diag.get_quantity("not_a_quantity")


# The rest of the suite is 1D, which takes a different branch in OsirisGridFile
# than multi-dimensional dumps do. These shapes have distinct, >1 extents on
# every axis: a square or a length-1 axis would hide both a transposition bug
# and a buffer-contiguity one.
@pytest.mark.parametrize("shape", [(8, 12), (4, 6, 10)])
def test_osiris_grid_file_multidim_roundtrip(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """A 2D/3D dump reads back in (x1, x2, ...) order, exactly as written."""
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    path = write_grid_file(tmp_path / "b3-000000.h5", name="b3", data=data, iteration=0)

    grid_file = OsirisGridFile(str(path))

    assert grid_file.dim == len(shape)
    assert grid_file.nx == shape
    assert grid_file.data.shape == shape
    np.testing.assert_array_equal(grid_file.data, data)
