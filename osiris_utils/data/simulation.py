from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..data.diagnostic import Diagnostic
from ..data.track_diagnostic import Track_Diagnostic
from ..decks.decks import InputDeckIO

__all__ = ["Simulation", "Species_Handler"]

logger = logging.getLogger(__name__)


def _register_on_load(diag: Diagnostic, registry: dict, key: str) -> Diagnostic:
    """Put *diag* in *registry* once its data is fully loaded.

    ``loaded_diagnostics`` means exactly that -- the diagnostics holding data in
    memory -- so a diagnostic enters the registry when ``load_all()`` succeeds,
    not when it is first accessed.  ``load_all()`` then returns the diagnostic
    itself rather than the array, which is what the Simulation-level API has
    always handed back.
    """
    original = diag.load_all

    def load_all(*args, **kwargs):
        original(*args, **kwargs)
        registry[key] = diag
        return diag

    diag.load_all = load_all
    return diag


def _register(store: dict, diagnostic: Diagnostic, name: str | None) -> str:
    """Put *diagnostic* in *store* under *name* (``custom_diag_<i>`` if None) and return the name."""
    if not isinstance(diagnostic, Diagnostic):
        raise ValueError("Only Diagnostic objects are supported")
    if name is None:
        i = 1
        while f"custom_diag_{i}" in store:
            i += 1
        name = f"custom_diag_{i}"
    store[name] = diagnostic
    return name


class _DiagnosticRegistry:
    """Diagnostics by key: each opened once, on first access, plus the loaded and added ones.

    Shared by :class:`Simulation` and :class:`Species_Handler`, which differ
    only in how they open a quantity (:meth:`_open`).
    """

    def __init__(self) -> None:
        self._diagnostics: dict[str, Any] = {}  # loaded (see _register_on_load) or added with add_diagnostic
        # Every quantity ever asked for, loaded or not.  Opening a Diagnostic
        # globs its dump directory and reads a file for the grid metadata, so
        # handing out a new one per access made `sim["b2"]` in a loop scan the
        # filesystem again every time.
        self._opened: dict[str, Any] = {}

    def _open(self, key: str) -> Any:
        raise NotImplementedError

    def _get(self, key: str) -> Any:
        if key in self._diagnostics:
            return self._diagnostics[key]
        if key not in self._opened:
            self._opened[key] = _register_on_load(self._open(key), self._diagnostics, key)
        return self._opened[key]

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Register a custom diagnostic, reachable afterwards as ``self[name]``.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic to add, e.g. the result of arithmetic on others.
        name : str, optional
            Key to store it under.  Default: the first free ``custom_diag_<i>``.

        Returns
        -------
        str
            The key used.

        Example
        -------
        >>> sim = Simulation('path/to/simulation/input_deck.txt')
        >>> nT = sim['electrons']['n'] * sim['electrons']['T11']
        >>> sim.add_diagnostic(nT, 'nT')
        >>> sim['nT']  # Access the custom diagnostic
        """
        return _register(self._diagnostics, diagnostic, name)

    def delete_diagnostic(self, key: str) -> None:
        """Forget *key*: added, loaded or merely opened."""
        found = self._diagnostics.pop(key, None) is not None
        found = self._opened.pop(key, None) is not None or found
        if not found:
            logger.warning("Diagnostic '%s' not found.", key)

    def delete_all_diagnostics(self) -> None:
        """Forget every diagnostic opened, loaded or added so far."""
        self._diagnostics = {}
        self._opened = {}

    @property
    def loaded_diagnostics(self) -> dict[str, Diagnostic]:
        """Diagnostics holding their data in memory, and the ones added with :meth:`add_diagnostic`."""
        return self._diagnostics


class Simulation(_DiagnosticRegistry):
    """
    Class to handle the simulation data. It is a wrapper for the Diagnostic class.

    Parameters
    ----------
    input_deck_path : str
        Path to the input deck (It must be in the folder where the simulation was run)

    Attributes
    ----------
    simulation_folder : str
        The simulation folder.
    species : list[str]
        Names of the species in the input deck; ``sim[name]`` is their :class:`Species_Handler`.
    loaded_diagnostics : dict
        Diagnostics loaded with ``load_all`` or added with ``add_diagnostic``.
    """

    def __init__(self, input_deck_path: str) -> None:
        super().__init__()
        self._input_deck = InputDeckIO(input_deck_path, verbose=False)
        self._species = list(self._input_deck.species.keys())
        self._simulation_folder = str(Path(input_deck_path).parent)
        self._species_handler: dict[str, Species_Handler] = {}

    def __getitem__(self, key: str) -> Diagnostic | Species_Handler:
        if key in self._species:
            if key not in self._species_handler:
                self._species_handler[key] = Species_Handler(self._simulation_folder, self._input_deck.species[key], self._input_deck)
            return self._species_handler[key]
        return self._get(key)

    def _open(self, key: str) -> Diagnostic:
        if key == "tracks":
            raise ValueError("Tracks diagnostics require a specie.")
        # Quantities that are not species related
        diag = Diagnostic(simulation_folder=self._simulation_folder, species=None, input_deck=self._input_deck)
        diag.get_quantity(key)
        return diag

    @property
    def species(self) -> list[str]:
        return self._species


class Species_Handler(_DiagnosticRegistry):
    """The diagnostics of one species: ``sim["electrons"]["vfl1"]``, ``sim["electrons"]["tracks"]``."""

    def __init__(self, simulation_folder: str, species_name: Any, input_deck: Any) -> None:
        super().__init__()
        self._simulation_folder = simulation_folder
        self._species_name = species_name
        self._input_deck = input_deck

    def __getitem__(self, key: str) -> Diagnostic:
        return self._get(key)

    def _open(self, key: str) -> Diagnostic | Track_Diagnostic:
        if key == "tracks":
            return Track_Diagnostic(simulation_folder=self._simulation_folder, species=self._species_name, input_deck=self._input_deck)
        diag = Diagnostic(simulation_folder=self._simulation_folder, species=self._species_name, input_deck=self._input_deck)
        diag.get_quantity(key)
        return diag

    @property
    def species(self) -> Any:
        return self._species_name
