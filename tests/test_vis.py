from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osiris_utils.vis.plot3d import plot_3d


class MockDiagnostic:
    def __init__(self, dim=3):
        self.dim = dim
        self.grid = np.array([[0, 1], [0, 1], [0, 1]])
        self.x = [np.linspace(0, 1, 10) for _ in range(3)]
        # Shape: (timesteps, x, y, z) -> (2, 10, 10, 10)
        self._data = np.random.rand(2, 10, 10, 10)
        self.name = "test_data"
        self.units = "a.u."
        self.axis = [
            {"long_name": "x1", "units": "c/wp"},
            {"long_name": "x2", "units": "c/wp"},
            {"long_name": "x3", "units": "c/wp"},
        ]
        self._all_loaded = True

    def time(self, idx):
        return 10.0, "1/wp"

    def __getitem__(self, idx):
        # Allow slicing or integer indexing similar to Diagnostic
        return self._data[idx]


@patch("osiris_utils.vis.plot3d.plt")
def test_plot_3d_basic(mock_plt):
    """Test basic functionality of plot_3d."""
    diagnostic = MockDiagnostic()

    # Mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax

    fig, ax = plot_3d(diagnostic, idx=0)

    assert fig == mock_fig
    assert ax == mock_ax

    # Check if scatter was called
    assert mock_ax.scatter.called

    # Check labels
    assert mock_ax.set_xlabel.called
    assert mock_ax.set_ylabel.called
    assert mock_ax.set_zlabel.called
    assert mock_ax.set_title.called


def test_plot_3d_invalid_dim():
    """Test plot_3d raises error for non-3D diagnostics."""
    diagnostic = MockDiagnostic(dim=2)

    with pytest.raises(ValueError, match="only available for 3D diagnostics"):
        plot_3d(diagnostic, idx=0)


@patch("osiris_utils.vis.plot3d.plt")
def test_plot_3d_scale_types(mock_plt):
    """Test different scale types."""
    diagnostic = MockDiagnostic()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax

    # Test 'zero_centered'
    plot_3d(diagnostic, idx=0, scale_type="zero_centered")
    args, kwargs = mock_ax.scatter.call_args
    assert kwargs['cmap'] == 'seismic'

    # Test 'pos'
    plot_3d(diagnostic, idx=0, scale_type="pos")
    args, kwargs = mock_ax.scatter.call_args
    assert kwargs['cmap'] == 'plasma'

    # Test 'neg'
    plot_3d(diagnostic, idx=0, scale_type="neg")
    args, kwargs = mock_ax.scatter.call_args
    assert kwargs['cmap'] == 'plasma'


@patch("osiris_utils.vis.plot3d.plt")
def test_plot_3d_boundaries(mock_plt):
    """Test plot_3d with custom boundaries."""
    diagnostic = MockDiagnostic()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax

    boundaries = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
    plot_3d(diagnostic, idx=0, boundaries=boundaries)

    assert mock_ax.scatter.called

    # Test invalid boundaries shape warning/fallback
    with pytest.warns(UserWarning, match="boundaries should have shape"):
        plot_3d(diagnostic, idx=0, boundaries=np.array([0, 1]))
