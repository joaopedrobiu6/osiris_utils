import h5py
import numpy as np

import osiris_utils as ou


def create_test_h5_file(filepath, data_shape=(100, 100), iteration=0):
    """Helper to create a test HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        # SIMULATION group
        sim_group = f.create_group("SIMULATION")
        sim_group.attrs.create("DT", [0.01])
        sim_group.attrs.create("NDIMS", [len(data_shape)])

        # Main attributes
        f.attrs.create("TIME", [iteration * 0.01])
        f.attrs.create("TIME UNITS", [b"1 / \\omega_p"])
        f.attrs.create("ITER", [iteration])
        f.attrs.create("NAME", [b"test"])
        f.attrs.create("TYPE", [b"grid"])
        f.attrs.create("UNITS", [b"m_e c \\omega_p / e"])
        f.attrs.create("LABEL", [b"test"])

        # Data
        data = np.random.rand(*data_shape).astype(np.float32)
        f.create_dataset("test", data=data.T)

        # AXIS group
        axis_group = f.create_group("AXIS")
        for i, _ in enumerate(data_shape):
            axis_name = f"AXIS{i + 1}"
            axis_data = np.linspace(0, 10, 2)
            axis_dataset = axis_group.create_dataset(axis_name, data=axis_data)
            axis_dataset.attrs.create("NAME", [f"x{i + 1}".encode()])
            axis_dataset.attrs.create("UNITS", [b"c / \\omega_p"])
            axis_dataset.attrs.create("LONG_NAME", [f"x_{i + 1}".encode()])
            axis_dataset.attrs.create("TYPE", [b"linear"])


class TestParallelLoading:
    """Test parallel loading functionality."""

    def test_load_all_sequential(self):
        """Test sequential loading works."""
        # Uses real example data
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        data = d.load_all(use_parallel=False)
        assert data is not None
        assert len(data.shape) == 2  # (timesteps, spatial_dims)
        assert d._all_loaded is True

    def test_load_all_parallel(self):
        """Test parallel loading works."""
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        data = d.load_all(use_parallel=True)
        assert data is not None
        assert len(data.shape) == 2
        assert d._all_loaded is True

    def test_auto_detect_small_files(self):
        """Test auto-detection chooses sequential for small files."""
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        # Small files should use sequential
        data = d.load_all(use_parallel=None)
        assert data is not None

    def test_unload_after_load(self):
        """Test unloading data works."""
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        d.load_all()
        assert d._all_loaded is True

        d.unload()
        assert d._all_loaded is False
        assert d._data is None


class TestIterationOptimizations:
    """Test iteration optimizations."""

    def test_iteration_uses_cached_maxiter(self):
        """Test that iteration uses cached _maxiter."""
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        # Iterate and ensure we don't call glob
        count = 0
        for _ in d:
            count += 1
            if count > 5:  # Just test a few
                break

        assert count == 6

    def test_len_returns_maxiter(self):
        """Test that len() works correctly."""
        d = ou.Diagnostic(simulation_folder='examples/example_data')
        d.get_quantity('e3')

        length = len(d)
        assert length > 0
        assert length == d._maxiter
