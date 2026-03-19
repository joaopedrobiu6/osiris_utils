"""Tests for osiris_utils.postprocessing.postprocess.PostProcess."""
import pytest
from unittest.mock import MagicMock

from osiris_utils.postprocessing.postprocess import PostProcess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MinimalSim:
    """Minimal simulation-like object with _species and __getitem__."""

    def __init__(self, species=None):
        self._species = species or ["electrons"]
        self.species = self._species  # required: getattr evaluates default eagerly

    def __getitem__(self, key):
        return MagicMock()


class NoGetItemSim:
    """Has species attribute but no __getitem__."""
    _species = ["electrons"]


class NoSpeciesSim:
    """Has __getitem__ but neither _species nor species."""

    def __getitem__(self, key):
        return MagicMock()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestPostProcessInit:
    def test_init_with_underscore_species(self):
        sim = MinimalSim(["electrons", "ions"])
        pp = PostProcess("test", sim)
        assert pp.species == ["electrons", "ions"]

    def test_init_prefers_underscore_species_over_dot_species(self):
        sim = MinimalSim(["electrons"])
        sim.species = ["ions"]          # dot-species differs on purpose
        pp = PostProcess("test", sim)
        # _species takes priority
        assert pp.species == ["electrons"]

    def test_init_falls_back_to_dot_species(self):
        sim = MagicMock()
        del sim._species                # remove _species
        sim.species = ["protons"]
        pp = PostProcess("test", sim)
        assert pp.species == ["protons"]

    def test_init_no_getitem_raises_type_error(self):
        with pytest.raises(TypeError, match="__getitem__"):
            PostProcess("test", NoGetItemSim())

    def test_init_no_species_raises_type_error(self):
        with pytest.raises(TypeError, match="_species or species"):
            PostProcess("test", NoSpeciesSim())


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestPostProcessProperties:
    def test_loaded_diagnostics_delegates(self):
        sim = MinimalSim()
        sentinel = {"diag1": object()}
        sim.loaded_diagnostics = sentinel
        pp = PostProcess("test", sim)
        assert pp.loaded_diagnostics is sentinel

    def test_loaded_diagnostics_returns_empty_when_absent(self):
        sim = MinimalSim()          # no loaded_diagnostics attribute
        pp = PostProcess("test", sim)
        assert pp.loaded_diagnostics == {}


# ---------------------------------------------------------------------------
# add_diagnostic
# ---------------------------------------------------------------------------

class TestPostProcessAddDiagnostic:
    def test_add_diagnostic_delegates_to_simulation(self):
        sim = MinimalSim()
        sim.add_diagnostic = MagicMock(return_value="added_name")
        pp = PostProcess("test", sim)

        mock_diag = MagicMock()
        result = pp.add_diagnostic(mock_diag, "mydiag")

        sim.add_diagnostic.assert_called_once_with(mock_diag, name="mydiag")
        assert result == "added_name"

    def test_add_diagnostic_raises_when_not_supported(self):
        sim = MinimalSim()          # no add_diagnostic method
        pp = PostProcess("test", sim)
        with pytest.raises(AttributeError, match="add_diagnostic"):
            pp.add_diagnostic(MagicMock(), "x")


# ---------------------------------------------------------------------------
# delete_all_diagnostics / delete_diagnostic
# ---------------------------------------------------------------------------

class TestPostProcessDelete:
    def test_delete_all_diagnostics_delegates(self):
        sim = MinimalSim()
        sim.delete_all_diagnostics = MagicMock()
        pp = PostProcess("test", sim)
        pp.delete_all_diagnostics()
        sim.delete_all_diagnostics.assert_called_once()

    def test_delete_all_diagnostics_is_noop_when_absent(self):
        sim = MinimalSim()
        pp = PostProcess("test", sim)
        pp.delete_all_diagnostics()    # must not raise

    def test_delete_diagnostic_delegates(self):
        sim = MinimalSim()
        sim.delete_diagnostic = MagicMock()
        pp = PostProcess("test", sim)
        pp.delete_diagnostic("key1")
        sim.delete_diagnostic.assert_called_once_with("key1")

    def test_delete_diagnostic_is_noop_when_absent(self):
        sim = MinimalSim()
        pp = PostProcess("test", sim)
        pp.delete_diagnostic("key1")   # must not raise
