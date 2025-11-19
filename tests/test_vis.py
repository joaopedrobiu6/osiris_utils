import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from osiris_utils.vis.plot3d import plot_3d


class TestPlot3D(unittest.TestCase):
    def setUp(self):
        # Create a mock diagnostic object
        self.mock_diag = MagicMock()
        self.mock_diag.dim = 3
        self.mock_diag.grid = np.array([[0, 10], [0, 10], [0, 10]])
        self.mock_diag.x = [np.linspace(0, 10, 5), np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
        # Mock data access
        self.mock_diag.__getitem__.return_value = np.random.rand(5, 5, 5)
        self.mock_diag._all_loaded = False

        # Mock metadata
        self.mock_diag.name = "test_field"
        self.mock_diag.units = "a.u."
        self.mock_diag.time.return_value = (10.0, "1/wp")
        self.mock_diag.axis = [
            {"long_name": "x1", "units": "c/wp"},
            {"long_name": "x2", "units": "c/wp"},
            {"long_name": "x3", "units": "c/wp"},
        ]

    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.figure")
    def test_plot_3d_basic(self, mock_figure, mock_colorbar):
        """Test basic 3D plotting functionality."""
        fig, ax = plot_3d(self.mock_diag, idx=0)

        # Verify figure creation
        mock_figure.assert_called_once()

        # Verify data access
        self.mock_diag.__getitem__.assert_called_with(0)

    def test_plot_3d_invalid_dim(self):
        """Test that plotting fails for non-3D diagnostics."""
        self.mock_diag.dim = 2
        with self.assertRaises(ValueError):
            plot_3d(self.mock_diag, idx=0)

    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.figure")
    def test_plot_3d_custom_boundaries(self, mock_figure, mock_colorbar):
        """Test plotting with custom boundaries."""
        boundaries = np.array([[2, 8], [2, 8], [2, 8]])
        plot_3d(self.mock_diag, idx=0, boundaries=boundaries)
        # Should run without error

    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.figure")
    def test_plot_3d_scale_types(self, mock_figure, mock_colorbar):
        """Test different scale types."""
        for scale in ["zero_centered", "pos", "neg", "default"]:
            plot_3d(self.mock_diag, idx=0, scale_type=scale)
            # Should run without error


if __name__ == "__main__":
    unittest.main()
