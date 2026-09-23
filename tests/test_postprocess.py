"""Tests for osiris_utils.postprocessing.postprocess.PostProcess.

A view implements one hook, ``_process(container, key)``; the base class owns
species dispatch, the cache and user-registered diagnostics.
"""

from unittest.mock import MagicMock

import pytest

from osiris_utils.data.diagnostic import Diagnostic
from osiris_utils.postprocessing.postprocess import PostProcess

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Handler:
    """Species-handler stand-in: quantity -> "<species>/<quantity>"."""

    def __init__(self, name):
        self.species = name

    def __getitem__(self, key):
        return f"{self.species}/{key}"


class MinimalSim:
    """Simulation stand-in: quantity -> its name, species -> a Handler."""

    def __init__(self, species=("electrons",)):
        self.species = list(species)

    def __getitem__(self, key):
        return Handler(key) if key in self.species else key


class Tagging(PostProcess):
    """A view whose processing is visible: it tags the input and counts calls."""

    def __init__(self, simulation, **kwargs):
        super().__init__("tag", simulation, **kwargs)
        self.calls = []

    def _process(self, container, key):
        self.calls.append((container, key))
        return ("tagged", container[key])


class NoGetItemSim:
    species = ["electrons"]


class NoSpeciesSim:
    def __getitem__(self, key):
        return key


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_species_come_from_the_wrapped_simulation():
    assert Tagging(MinimalSim(["electrons", "ions"])).species == ["electrons", "ions"]


def test_explicit_species_override_the_simulation():
    """species=() (e.g. field centering): a species name is processed like any key."""
    view = Tagging(MinimalSim(["electrons"]), species=())
    assert view.species == []
    tag, handler = view["electrons"]
    assert tag == "tagged" and isinstance(handler, Handler)


def test_wrapped_object_needs_getitem():
    with pytest.raises(TypeError, match="__getitem__"):
        PostProcess("test", NoGetItemSim())


def test_wrapped_object_needs_species():
    with pytest.raises(TypeError, match="species attribute"):
        PostProcess("test", NoSpeciesSim())


def test_base_class_has_no_processing():
    with pytest.raises(NotImplementedError):
        PostProcess("test", MinimalSim())["e1"]


# ---------------------------------------------------------------------------
# Processing and species views
# ---------------------------------------------------------------------------


def test_quantity_is_processed_once_and_cached():
    sim = MinimalSim()
    view = Tagging(sim)

    first = view["e1"]
    assert first == ("tagged", "e1")
    assert view["e1"] is first
    assert view.calls == [(sim, "e1")]


def test_species_view_processes_that_species_quantities():
    sim = MinimalSim()
    view = Tagging(sim)

    electrons = view["electrons"]
    assert electrons is view["electrons"]  # one view per species
    assert electrons["vfl1"] == ("tagged", "electrons/vfl1")
    assert electrons["vfl1"] is electrons["vfl1"]
    # the parent's hook ran once, with the species handler as the container
    (container, key), = view.calls
    assert isinstance(container, Handler) and container.species == "electrons" and key == "vfl1"
    assert electrons.species == "electrons"


def test_views_chain():
    inner = Tagging(MinimalSim())
    outer = Tagging(inner)
    assert outer["e1"] == ("tagged", ("tagged", "e1"))
    assert outer["electrons"]["n"] == ("tagged", ("tagged", "electrons/n"))


# ---------------------------------------------------------------------------
# add_diagnostic / delete / loaded_diagnostics
# ---------------------------------------------------------------------------


def test_added_diagnostic_is_handed_back_unprocessed_and_stays_on_the_view():
    sim = MinimalSim()
    sim.add_diagnostic = MagicMock()
    view = Tagging(sim)
    diag = MagicMock(spec=Diagnostic)

    assert view.add_diagnostic(diag, "mine") == "mine"
    assert view["mine"] is diag
    assert view.calls == []
    sim.add_diagnostic.assert_not_called()


def test_added_diagnostic_gets_a_free_name():
    view = Tagging(MinimalSim())
    assert view.add_diagnostic(MagicMock(spec=Diagnostic)) == "custom_diag_1"
    assert view.add_diagnostic(MagicMock(spec=Diagnostic)) == "custom_diag_2"


def test_only_diagnostics_can_be_added():
    with pytest.raises(ValueError, match="Only Diagnostic"):
        Tagging(MinimalSim()).add_diagnostic("not a diagnostic", "x")


def test_loaded_diagnostics_lists_built_and_added():
    view = Tagging(MinimalSim())
    built = view["e1"]
    diag = MagicMock(spec=Diagnostic)
    view.add_diagnostic(diag, "mine")
    assert view.loaded_diagnostics == {"e1": built, "mine": diag}


def test_delete_diagnostic_rebuilds_on_next_access(caplog):
    view = Tagging(MinimalSim())
    view["e1"]
    view.delete_diagnostic("e1")
    view["e1"]
    assert len(view.calls) == 2

    view.delete_diagnostic("never-built")
    assert "not found" in caplog.text


def test_delete_all_diagnostics_clears_the_view():
    view = Tagging(MinimalSim())
    view["e1"]
    view.add_diagnostic(MagicMock(spec=Diagnostic), "mine")
    view.delete_all_diagnostics()
    assert view.loaded_diagnostics == {}
