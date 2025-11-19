import pytest

# Import from the correct location - cli.py not cli/__init__.py
from osiris_utils import cli as cli_module


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_version(self, capsys):
        """Test --version flag."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["--version"])
        assert exc_info.value.code == 0

    def test_help(self, capsys):
        """Test --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "osiris" in captured.out
        assert "info" in captured.out
        assert "export" in captured.out

    def test_no_command(self, capsys):
        """Test running without a command."""
        with pytest.raises(SystemExit):
            cli_module.main([])

    def test_invalid_command(self, capsys):
        """Test running with an invalid command."""
        with pytest.raises(SystemExit):
            cli_module.main(["invalid_command"])


class TestInfoCommand:
    """Test the info command."""

    def test_info_help(self, capsys):
        """Test info --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["info", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "info" in captured.out

    def test_info_missing_path(self):
        """Test info without a path."""
        with pytest.raises(SystemExit):
            cli_module.main(["info"])

    def test_info_nonexistent_path(self, capsys):
        """Test info with nonexistent path."""
        result = cli_module.main(["info", "/nonexistent/path"])
        assert result == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err


class TestExportCommand:
    """Test the export command."""

    def test_export_help(self, capsys):
        """Test export --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["export", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "export" in captured.out

    def test_export_missing_args(self):
        """Test export without required arguments."""
        with pytest.raises(SystemExit):
            cli_module.main(["export"])

    def test_export_nonexistent_path(self, capsys):
        """Test export with nonexistent path."""
        result = cli_module.main(["export", "/nonexistent/path", "--output", "/tmp/test.csv"])
        assert result == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err


class TestPlotCommand:
    """Test the plot command."""

    def test_plot_help(self, capsys):
        """Test plot --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["plot", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "plot" in captured.out

    def test_plot_missing_save_or_display(self, capsys, tmp_path):
        """Test plot without --save or --display."""
        # Create a fake file
        fake_file = tmp_path / "fake.h5"
        fake_file.touch()

        result = cli_module.main(["plot", str(fake_file)])
        assert result == 1
        captured = capsys.readouterr()
        assert "Must specify --save or --display" in captured.err


class TestValidateCommand:
    """Test the validate command."""

    def test_validate_help(self, capsys):
        """Test validate --help."""
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(["validate", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "validate" in captured.out

    def test_validate_missing_path(self):
        """Test validate without a path."""
        with pytest.raises(SystemExit):
            cli_module.main(["validate"])

    def test_validate_nonexistent_path(self, capsys):
        """Test validate with nonexistent path."""
        result = cli_module.main(["validate", "/nonexistent/path"])
        assert result == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err


class TestVerboseFlag:
    """Test the --verbose flag across commands."""

    def test_verbose_flag_with_error(self, capsys):
        """Test that --verbose shows detailed error info."""
        # This should fail with an error code, not raise an exception
        result = cli_module.main(["--verbose", "info", "/nonexistent/path"])
        assert result == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err
