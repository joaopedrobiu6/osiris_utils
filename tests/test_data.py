import os

import numpy as np
import pytest

from osiris_utils.data.data import OsirisGridFile
from osiris_utils.data.diagnostic import Diagnostic
from osiris_utils.decks.species import Species


@pytest.fixture
def example_data_dir():
    # Helper to get the absolute path to example_data
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/example_data"))


def test_osiris_grid_file(example_data_dir):
    # Path to a specific file
    file_path = os.path.join(example_data_dir, "MS/DENSITY/electrons/charge/charge-electrons-000000.h5")
    assert os.path.exists(file_path), f"Test file not found at {file_path}"

    grid_file = OsirisGridFile(file_path)

    # Check basic attributes
    assert grid_file.name == "charge"
    assert grid_file.type == "grid"
    assert grid_file.dim == 1
    assert grid_file.iter == 0
    # Check time: roughly 0.0
    assert np.isclose(grid_file.time[0], 0.0)

    # Check data shape
    # 1D data
    assert grid_file.data.ndim == 1
    # Check if data is loaded
    assert grid_file.data.size > 0

    # Check grid props
    assert grid_file.dx is not None
    assert grid_file.nx is not None


def test_diagnostic_integration(example_data_dir):
    # Test loading diagnostic from folder

    # Create a species object for context
    elec = Species(name="electrons", rqm=-1.0)

    # Initialize Diagnostic
    diag = Diagnostic(simulation_folder=example_data_dir, species=elec)

    # Request "charge" quantity
    diag.get_quantity("charge")

    # Check if files were scanned
    # Don't access private _file_list directly if possible, but for testing internals it might be needed
    # Or check public properties after scan
    # _maxiter should be populated
    assert diag._maxiter > 0

    # Load all data
    data = diag.load_all()
    assert diag._all_loaded
    assert data is not None

    # Check shape: (time, local_grid_size)
    assert data.shape[0] == diag._maxiter

    # Test indexing
    # Get first timestep
    d0 = diag[0]
    assert d0.shape == data[0].shape
    assert np.allclose(d0, data[0])

    # Basic math operations (Lazy evaluation)
    # diag + diag
    diag_sum = diag + diag
    # Should result in a new Diagnostic
    assert isinstance(diag_sum, Diagnostic)

    # Load sum data
    sum_data = diag_sum.load_all()
    assert np.allclose(sum_data, data * 2)
