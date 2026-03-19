import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import CLI modules
from osiris_utils.cli import export, info, plot, validate
from osiris_utils.cli.__main__ import main

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


# -----------------------------------------------------------------------------
# Test __main__.main entry point
# -----------------------------------------------------------------------------


def test_main_version_exits():
    """--version should print version and exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_main_no_args_exits():
    """Calling main with no subcommand should exit with a non-zero code."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_main_routes_to_subcommand():
    """main() should dispatch to the correct subcommand and return its exit code."""
    with (
        patch("osiris_utils.cli.info.run", return_value=0) as mock_run,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        ret = main(["info", "some_path"])
    assert ret == 0
    mock_run.assert_called_once()


def test_main_returns_1_on_exception_non_verbose():
    """Exceptions in subcommands are swallowed and return 1 when not verbose."""
    with (
        patch("osiris_utils.cli.info.run", side_effect=RuntimeError("boom")),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        ret = main(["info", "some_path"])
    assert ret == 1


def test_main_reraises_on_exception_verbose():
    """Exceptions in subcommands are re-raised when --verbose is set."""
    with (
        patch("osiris_utils.cli.info.run", side_effect=RuntimeError("boom")),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main(["-v", "info", "some_path"])


# -----------------------------------------------------------------------------
# Additional info command coverage
# -----------------------------------------------------------------------------


def test_info_path_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        args = argparse.Namespace(path="missing.h5", brief=False, verbose=False)
        ret = info.run(args)
    assert ret == 1


@patch("osiris_utils.cli.info.ou.Simulation")
def test_info_deck_file(mock_simulation, capsys):
    """Deck file paths are routed to show_simulation_info."""
    mock_sim = MagicMock()
    mock_sim.species = ["electrons"]
    mock_simulation.return_value = mock_sim

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.iterdir", return_value=iter([])),
    ):
        args = argparse.Namespace(path="input.deck", brief=False, verbose=False)
        ret = info.run(args)
    assert ret == 0
    assert "Species:" in capsys.readouterr().out


def test_info_directory_no_deck(capsys):
    """Directory without an input deck returns 1."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        # No deck candidate exists
        with patch("pathlib.Path.__truediv__", return_value=MagicMock(exists=lambda: False)):
            args = argparse.Namespace(path="sim_dir", brief=False, verbose=False)
            ret = info.run(args)
    assert ret == 1


@patch("osiris_utils.cli.info.ou.OsirisGridFile")
def test_info_file_brief(mock_grid_file, capsys):
    """Brief mode skips detailed grid/time/data sections."""
    mock_obj = MagicMock()
    mock_obj.type = "grid"
    mock_obj.name = "e1"
    mock_obj.dim = 1
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", brief=True)
        ret = info.run(args)
    assert ret == 0
    out = capsys.readouterr().out
    assert "Grid Information:" not in out


@patch("osiris_utils.cli.info.ou.OsirisGridFile", side_effect=OSError("bad file"))
def test_info_file_error(mock_grid_file):
    """show_file_info returns 1 when the file cannot be loaded."""
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="bad.h5", brief=False)
        ret = info.run(args)
    assert ret == 1


# -----------------------------------------------------------------------------
# Additional export command coverage
# -----------------------------------------------------------------------------


def test_export_path_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        args = argparse.Namespace(path="missing.h5", format="csv", output="out.csv", timestep=None, no_coords=False)
        ret = export.run(args)
    assert ret == 1


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_exception_returns_1(mock_grid_file):
    mock_grid_file.side_effect = RuntimeError("load error")
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", format="csv", output="out.csv", timestep=None, no_coords=False)
        ret = export.run(args)
    assert ret == 1


@patch("builtins.open", new_callable=MagicMock)
@patch("json.dump")
@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_json(mock_grid_file, mock_json_dump, mock_open):
    """Export to JSON format calls json.dump."""
    mock_obj = MagicMock()
    mock_obj.data = np.zeros((5, 5))
    mock_obj.x = [np.linspace(0, 1, 5), np.linspace(0, 1, 5)]
    mock_obj.name = "e1"
    mock_obj.type = "grid"
    mock_obj.units = "a.u."
    mock_obj.time = 1.0
    mock_obj.iter = 10
    mock_obj.axis = [{"units": "x"}, {"units": "y"}]
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", format="json", output="out.json", timestep=None, no_coords=False)
        ret = export.run(args)
    assert ret == 0
    assert mock_json_dump.called


def test_export_to_csv_1d(tmp_path):
    """export_to_csv handles 1D data."""
    mock_obj = MagicMock()
    mock_obj.name = "e1"
    mock_obj.x = [np.linspace(0, 1, 5)]
    data = np.ones(5)
    out = tmp_path / "out.csv"

    export.export_to_csv(data, mock_obj, out, no_coords=False)
    assert out.exists()
    import pandas as pd

    df = pd.read_csv(out)
    assert "e1" in df.columns
    assert "x" in df.columns


def test_export_to_csv_1d_no_coords(tmp_path):
    mock_obj = MagicMock()
    mock_obj.name = "e1"
    data = np.ones(5)
    out = tmp_path / "out.csv"

    export.export_to_csv(data, mock_obj, out, no_coords=True)
    import pandas as pd

    df = pd.read_csv(out)
    assert "e1" in df.columns
    assert "x" not in df.columns


def test_export_to_csv_2d_no_coords(tmp_path):
    mock_obj = MagicMock()
    mock_obj.name = "b3"
    data = np.ones((3, 4))
    out = tmp_path / "out.csv"

    export.export_to_csv(data, mock_obj, out, no_coords=True)
    assert out.exists()


def test_export_to_csv_3d(tmp_path):
    mock_obj = MagicMock()
    mock_obj.name = "f"
    data = np.ones((2, 3, 4))
    out = tmp_path / "out.csv"

    export.export_to_csv(data, mock_obj, out, no_coords=False)
    assert out.exists()


def test_export_to_json_no_coords(tmp_path):
    mock_obj = MagicMock()
    mock_obj.name = "e1"
    mock_obj.type = "grid"
    mock_obj.units = "a.u."
    mock_obj.time = 0.0
    mock_obj.iter = 0
    data = np.ones(3)
    out = tmp_path / "out.json"

    export.export_to_json(data, mock_obj, out, no_coords=True)
    import json

    with open(out) as f:
        obj = json.load(f)
    assert obj["name"] == "e1"
    assert "grid" not in obj


def test_export_to_json_with_coords(tmp_path):
    mock_obj = MagicMock()
    mock_obj.name = "e1"
    mock_obj.type = "grid"
    mock_obj.units = "a.u."
    mock_obj.time = 0.0
    mock_obj.iter = 0
    mock_obj.x = [np.linspace(0, 1, 3)]
    mock_obj.axis = [{"units": "x"}]
    data = np.ones(3)
    out = tmp_path / "out.json"

    export.export_to_json(data, mock_obj, out, no_coords=False)
    import json

    with open(out) as f:
        obj = json.load(f)
    assert "grid" in obj


# -----------------------------------------------------------------------------
# Additional validate command coverage
# -----------------------------------------------------------------------------


def test_validate_path_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        args = argparse.Namespace(path="missing.h5", check_missing=False, strict=False)
        ret = validate.run(args)
    assert ret == 1


@patch("h5py.File")
@patch("osiris_utils.cli.validate.ou.OsirisGridFile")
def test_validate_file_missing_axis(mock_grid_file, mock_h5, capsys):
    """validate_file warns when AXIS is absent."""
    mock_h5.return_value.__enter__.return_value = {}  # no AXIS key
    mock_obj = MagicMock()
    mock_obj.data = np.ones((5, 5))
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", check_missing=False, strict=False)
        ret = validate.run(args)
    assert ret == 0  # warnings don't fail unless strict
    assert "Missing AXIS" in capsys.readouterr().out


@patch("h5py.File")
@patch("osiris_utils.cli.validate.ou.OsirisGridFile")
def test_validate_strict_warnings_fail(mock_grid_file, mock_h5):
    """Strict mode returns 1 when there are warnings."""
    mock_h5.return_value.__enter__.return_value = {}  # triggers missing AXIS warning
    mock_obj = MagicMock()
    mock_obj.data = np.ones((5, 5))
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", check_missing=False, strict=True)
        ret = validate.run(args)
    assert ret == 1


@patch("h5py.File")
@patch("osiris_utils.cli.validate.ou.OsirisGridFile")
def test_validate_file_nan_data(mock_grid_file, mock_h5, capsys):
    """validate_file warns about NaN/Inf in data."""
    mock_h5.return_value.__enter__.return_value = {"AXIS": {}}
    mock_obj = MagicMock()
    mock_obj.data = np.array([1.0, np.nan, np.inf])
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", check_missing=False, strict=False)
        ret = validate.run(args)
    assert ret == 0
    assert "NaN or Inf" in capsys.readouterr().out


@patch("h5py.File", side_effect=OSError("bad file"))
def test_validate_file_oserror(mock_h5):
    """validate_file returns error count 1 on OSError."""
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="bad.h5", check_missing=False, strict=False)
        ret = validate.run(args)
    assert ret == 1


# -----------------------------------------------------------------------------
# Additional plot command coverage
# -----------------------------------------------------------------------------


def test_plot_path_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        args = argparse.Namespace(path="missing.h5", save=None, display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
    assert ret == 1


def test_plot_not_a_file():
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=False):
        args = argparse.Namespace(path="dir/", save="out.png", display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
    assert ret == 1


def test_plot_no_output_target():
    """Must specify --save or --display."""
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", save=None, display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
    assert ret == 1


@patch("osiris_utils.cli.plot.ou.OsirisGridFile")
@patch("osiris_utils.cli.plot.plt")
def test_plot_1d(mock_plt, mock_grid_file):
    """1D data is routed to create_1d_plot."""
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    mock_obj = MagicMock()
    mock_obj.dim = 1
    mock_obj.data = np.ones(10)
    mock_obj.x = np.linspace(0, 1, 10)
    mock_obj.label = "e1"
    mock_obj.units = "a.u."
    mock_obj.time = 0.0
    mock_obj.axis = [{"units": "x"}]
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", save="out.png", display=False, title="My Plot", cmap="viridis", dpi=100, log_scale=True)
        ret = plot.run(args)
    assert ret == 0


@patch("osiris_utils.cli.plot.ou.OsirisGridFile")
@patch("osiris_utils.cli.plot.plt")
def test_plot_unsupported_dim(mock_plt, mock_grid_file):
    """3D+ data is not supported and returns 1."""
    mock_obj = MagicMock()
    mock_obj.dim = 3
    mock_grid_file.return_value = mock_obj

    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", save="out.png", display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
    assert ret == 1


@patch("osiris_utils.cli.plot.ou.OsirisGridFile")
@patch("osiris_utils.cli.plot.plt")
def test_plot_2d_with_title_and_log_scale(mock_plt, mock_grid_file):
    """2D plot with a custom title and log_scale uses SymLogNorm."""
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.colorbar.return_value = MagicMock()

    mock_obj = MagicMock()
    mock_obj.dim = 2
    mock_obj.data = np.ones((10, 10))
    mock_obj.x = [np.linspace(0, 1, 10), np.linspace(0, 1, 10)]
    mock_obj.label = "b3"
    mock_obj.units = "a.u."
    mock_obj.time = 1.0
    mock_obj.axis = [{"units": "x"}, {"units": "y"}]
    mock_grid_file.return_value = mock_obj

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("matplotlib.colors.SymLogNorm", return_value=MagicMock()) as mock_norm,
    ):
        args = argparse.Namespace(
            path="dummy.h5", save="out.png", display=False, title="Custom Title", cmap="viridis", dpi=100, log_scale=True
        )
        ret = plot.run(args)
    assert ret == 0
    mock_norm.assert_called_once()


@patch("osiris_utils.cli.plot.ou.OsirisGridFile", side_effect=RuntimeError("load fail"))
@patch("osiris_utils.cli.plot.plt")
def test_plot_exception_returns_1(mock_plt, mock_grid_file):
    """Exceptions during plot creation return 1."""
    with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.is_file", return_value=True):
        args = argparse.Namespace(path="dummy.h5", save="out.png", display=False, title=None, cmap="viridis", dpi=100, log_scale=False)
        ret = plot.run(args)
    assert ret == 1


# -----------------------------------------------------------------------------
# Additional validate coverage — deck file, directory, validate_simulation
# -----------------------------------------------------------------------------


@patch("osiris_utils.cli.validate.ou.Simulation")
def test_validate_directory_with_deck(mock_simulation, tmp_path):
    """A directory containing an input deck validates the simulation."""
    mock_sim = MagicMock()
    mock_sim.species = ["electrons"]
    mock_simulation.return_value = mock_sim

    deck = tmp_path / "input.deck"
    deck.write_text("")

    ms_path = tmp_path / "MS"
    ms_path.mkdir()

    ret = validate.run(argparse.Namespace(path=str(tmp_path), check_missing=False, strict=False))
    assert ret == 0


@patch("osiris_utils.cli.validate.ou.Simulation", side_effect=RuntimeError("no sim"))
def test_validate_simulation_load_error(mock_simulation, tmp_path):
    """validate_simulation returns errors when simulation cannot be loaded."""
    deck = tmp_path / "input.deck"
    deck.write_text("")

    ret = validate.run(argparse.Namespace(path=str(tmp_path), check_missing=False, strict=False))
    assert ret == 1


@patch("osiris_utils.cli.validate.ou.Simulation")
def test_validate_simulation_no_ms_dir(mock_simulation, tmp_path, capsys):
    """validate_simulation warns when MS directory is missing."""
    mock_sim = MagicMock()
    mock_sim.species = ["electrons"]
    mock_simulation.return_value = mock_sim

    deck = tmp_path / "input.deck"
    deck.write_text("")
    # No MS directory created

    ret = validate.run(argparse.Namespace(path=str(tmp_path), check_missing=False, strict=False))
    assert ret == 0  # warnings only
    assert "MS directory not found" in capsys.readouterr().out


def test_validate_diagnostic_dir_check_missing(tmp_path):
    """validate_diagnostic_dir detects missing sequential iterations."""
    diag_path = tmp_path / "FLD"
    sub = diag_path / "e1"
    sub.mkdir(parents=True)
    # Create files 0, 1, 3 — iteration 2 is missing
    for idx in [0, 1, 3]:
        (sub / f"e1-{idx:06d}.h5").write_text("")

    errors, warnings = validate.validate_diagnostic_dir(diag_path, check_missing=True)
    assert warnings >= 1


def test_validate_diagnostic_dir_no_check_missing(tmp_path):
    """validate_diagnostic_dir without check_missing produces no warnings."""
    diag_path = tmp_path / "FLD"
    sub = diag_path / "e1"
    sub.mkdir(parents=True)
    for idx in [0, 1, 3]:
        (sub / f"e1-{idx:06d}.h5").write_text("")

    errors, warnings = validate.validate_diagnostic_dir(diag_path, check_missing=False)
    assert warnings == 0


def test_validate_deck_file_path(tmp_path):
    """A deck file passed directly triggers validate_simulation."""
    with patch("osiris_utils.cli.validate.ou.Simulation", side_effect=RuntimeError("fail")):
        deck = tmp_path / "input.deck"
        deck.write_text("")
        ret = validate.run(argparse.Namespace(path=str(deck), check_missing=False, strict=False))
    assert ret == 1


# -----------------------------------------------------------------------------
# Additional export coverage — directory input, diagnostic dir, multi-timestep
# -----------------------------------------------------------------------------


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_diagnostic_dir_npy(mock_grid_file, tmp_path):
    """export_diagnostic_dir stacks all h5 files into a npy when format=npy."""
    mock_obj = MagicMock()
    mock_obj.data = np.ones((3, 3))
    mock_grid_file.return_value = mock_obj

    diag_dir = tmp_path / "FLD" / "e1"
    diag_dir.mkdir(parents=True)
    for i in range(3):
        (diag_dir / f"e1-{i:06d}.h5").write_text("")

    out = tmp_path / "out.npy"
    with patch("numpy.save") as mock_save:
        export.export_diagnostic_dir(
            diag_dir,
            argparse.Namespace(format="npy", output=str(out), timestep=None, no_coords=False),
        )
    assert mock_save.called


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_diagnostic_dir_specific_timestep(mock_grid_file, tmp_path):
    """export_diagnostic_dir exports a single timestep when --timestep is set."""
    mock_obj = MagicMock()
    mock_obj.data = np.ones(5)
    mock_obj.name = "e1"
    mock_obj.x = [np.linspace(0, 1, 5)]
    mock_grid_file.return_value = mock_obj

    diag_dir = tmp_path / "e1"
    diag_dir.mkdir()
    for i in range(3):
        (diag_dir / f"e1-{i:06d}.h5").write_text("")

    out = tmp_path / "out.csv"
    export.export_diagnostic_dir(
        diag_dir,
        argparse.Namespace(format="csv", output=str(out), timestep=1, no_coords=False),
    )
    assert mock_grid_file.called


def test_export_diagnostic_dir_timestep_out_of_range(tmp_path):
    """Out-of-range --timestep raises ValueError."""
    diag_dir = tmp_path / "e1"
    diag_dir.mkdir()
    (diag_dir / "e1-000000.h5").write_text("")

    with pytest.raises(ValueError, match="out of range"):
        export.export_diagnostic_dir(
            diag_dir,
            argparse.Namespace(format="csv", output="out.csv", timestep=99, no_coords=False),
        )


def test_export_diagnostic_dir_no_h5_files(tmp_path):
    """Empty diagnostic directory raises ValueError."""
    diag_dir = tmp_path / "empty"
    diag_dir.mkdir()

    with pytest.raises(ValueError, match="No HDF5 files"):
        export.export_diagnostic_dir(
            diag_dir,
            argparse.Namespace(format="csv", output="out.csv", timestep=None, no_coords=False),
        )


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_multi_timestep_csv(mock_grid_file, tmp_path):
    """export_multi_timestep writes a combined CSV."""
    mock_obj = MagicMock()
    mock_obj.time = 0.0
    mock_obj.data = np.ones(3)
    mock_grid_file.return_value = mock_obj

    h5_files = []
    for i in range(2):
        p = tmp_path / f"e1-{i:06d}.h5"
        p.write_text("")
        h5_files.append(p)

    out = tmp_path / "out.csv"
    export.export_multi_timestep(h5_files, argparse.Namespace(format="csv", output=str(out)))
    assert out.exists()


@patch("osiris_utils.cli.export.ou.OsirisGridFile")
def test_export_multi_timestep_json(mock_grid_file, tmp_path):
    """export_multi_timestep writes a JSON array."""
    mock_obj = MagicMock()
    mock_obj.time = 0.0
    mock_obj.iter = 0
    mock_obj.data = np.ones(3)
    mock_grid_file.return_value = mock_obj

    h5_files = []
    for i in range(2):
        p = tmp_path / f"e1-{i:06d}.h5"
        p.write_text("")
        h5_files.append(p)

    out = tmp_path / "out.json"
    export.export_multi_timestep(h5_files, argparse.Namespace(format="json", output=str(out)))
    import json

    with open(out) as f:
        result = json.load(f)
    assert len(result) == 2


# -----------------------------------------------------------------------------
# Additional info coverage — not-a-file-or-dir, show_simulation_info verbose
# -----------------------------------------------------------------------------


def test_info_not_file_or_dir(capsys):
    """run() returns 1 for paths that are neither file nor directory."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=False),
    ):
        args = argparse.Namespace(path="weird_path", brief=False, verbose=False)
        ret = info.run(args)
    assert ret == 1


@patch("osiris_utils.cli.info.ou.Simulation", side_effect=RuntimeError("oops"))
def test_info_simulation_error_verbose_reraises(mock_sim):
    """show_simulation_info re-raises when verbose=True."""
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.is_dir", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="oops"):
            args = argparse.Namespace(path="input.deck", brief=False, verbose=True)
            info.run(args)


# -----------------------------------------------------------------------------
# Additional utils coverage
# -----------------------------------------------------------------------------
