import argparse
from unittest.mock import MagicMock, patch

import numpy as np

# Import CLI modules
from osiris_utils.cli import export, info, plot, validate

# -----------------------------------------------------------------------------
# Test Info Command
# -----------------------------------------------------------------------------


@patch("osiris_utils.cli.info.ou.OsirisGridFile")
def test_info_file(mock_grid_file, capsys):
    """Test 'info' command for a single file."""
    # Setup mock
    mock_obj = MagicMock()
    mock_obj.type = "grid"
    mock_obj.name = "e1"
    mock_obj.dim = 2
    mock_obj.nx = [100, 100]
    mock_obj.dx = [0.1, 0.1]
    mock_obj.time = 10.5
    mock_obj.dt = 0.5
    mock_obj.iter = 21
    mock_obj.units = "a.u."
    mock_obj.label = "Electric Field"
    mock_obj.data = np.zeros((100, 100))
    mock_obj.axis = [{"axis": [0, 10]}, {"axis": [0, 10]}]
    mock_grid_file.return_value = mock_obj

    # Create dummy file path (needs to 'exist' for Path check)
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", brief=False)
        ret = info.run(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "File: dummy.h5" in captured.out
        assert "Type: grid" in captured.out
        assert "Grid Information:" in captured.out


@patch("osiris_utils.cli.info.ou.Simulation")
def test_info_simulation(mock_simulation, capsys):
    """Test 'info' command for a simulation directory."""
    # Setup mock
    mock_sim = MagicMock()
    mock_sim.species = ["electrons", "protons"]
    mock_simulation.return_value = mock_sim

    # Mock file system structure
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_dir") as mock_is_dir,
        patch("pathlib.Path.iterdir"),
    ):
        # Scenario: Directory exists, input.deck exists
        def exists_side_effect(self):
            # Check if checking for existing path or input deck
            if str(self) == "sim_dir" or str(self).endswith("input.deck"):
                return True
            return False

        # mock_exists.side_effect = lambda: True # Simplification
        # Proper side effect is tricky with Path objects, let's just make everything exist
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        args = argparse.Namespace(path="sim_dir", brief=False, verbose=False)
        ret = info.run(args)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Simulation: sim_dir" in captured.out
        assert "Species:" in captured.out


# -----------------------------------------------------------------------------
# Test Export Command
# -----------------------------------------------------------------------------


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
@patch("numpy.save")
@patch("pandas.DataFrame.to_csv")
def test_export_file(mock_to_csv, mock_np_save, mock_grid_file):
    """Test 'export' command for single file."""
    mock_obj = MagicMock()
    mock_obj.data = np.zeros((10, 10))
    mock_obj.dim = 2
    mock_obj.x = [np.linspace(0, 1, 10), np.linspace(0, 1, 10)]
    mock_obj.name = "data"
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        # Test CSV export
        args = argparse.Namespace(path="dummy.h5", format="csv", output="out.csv", timestep=None, no_coords=False)
        export.run(args)
        assert mock_to_csv.called

        # Test NPY export
        args.format = "npy"
        args.output = "out.npy"
        export.run(args)
        assert mock_np_save.called


# -----------------------------------------------------------------------------
# Test Plot Command
# -----------------------------------------------------------------------------


@patch("osiris_utils.cli.plot.ou.OsirisGridFile")
@patch("osiris_utils.cli.plot.plt")
def test_plot_file(mock_plt, mock_grid_file):
    """Test 'plot' command."""
    # Setup plt mock
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    mock_obj = MagicMock()
    mock_obj.data = np.zeros((10, 10))
    mock_obj.dim = 2
    mock_obj.x = [np.linspace(0, 1, 10), np.linspace(0, 1, 10)]
    mock_obj.label = "test"
    mock_obj.units = "u"
    mock_obj.time = 0
    mock_obj.axis = [{"units": "x"}, {"units": "y"}]
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", save="plot.png", display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
        assert ret == 0
        assert mock_plt.savefig.called


# -----------------------------------------------------------------------------
# Test Validate Command
# -----------------------------------------------------------------------------


@patch("h5py.File")
@patch("osiris_utils.cli.validate.ou.OsirisGridFile")
def test_validate_file(mock_grid_file, mock_h5):
    """Test 'validate' command for a file."""

    # Successful validation scenario
    mock_h5.return_value.__enter__.return_value = {"AXIS": {}}

    mock_obj = MagicMock()
    mock_obj.data = np.zeros((10, 10))  # Non-empty
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", check_missing=False, strict=False)
        ret = validate.run(args)
        assert ret == 0
