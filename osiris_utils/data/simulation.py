from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.diagnostic import Diagnostic
from ..data.track_diagnostic import Track_Diagnostic
from ..decks.decks import InputDeckIO

__all__ = ["Simulation", "Species_Handler"]


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


class Simulation:
    """
    Class to handle the simulation data. It is a wrapper for the Diagnostic class.'

    Parameters
    ----------
    input_deck_path : str
        Path to the input deck (It must be in the folder where the simulation was run)

    Attributes
    ----------
    simulation_folder : str
        The simulation folder.
    species : Species object
        The species to analyze.
    diagnostics : dict
        Dictionary to store diagnostics for each quantity when `load_all` method is used.

    Methods
    -------
    delete_all_diagnostics()
        Delete all diagnostics.
    delete_diagnostic(key)
        Delete a diagnostic.
    __getitem__(key)
        Get a diagnostic.
    """

    def __init__(self, input_deck_path: str) -> None:
        folder_path = str(Path(input_deck_path).parent)
        self._input_deck_path = input_deck_path
        self._input_deck = InputDeckIO(self._input_deck_path, verbose=False)

        self._species = list(self._input_deck.species.keys())

        self._simulation_folder = folder_path
        self._diagnostics = {}  # Dictionary to store diagnostics for each quantity
        # Every quantity ever asked for, loaded or not.  Building a Diagnostic
        # globs its dump directory and opens a file for the grid metadata, so
        # handing out a new one per access made `sim["b2"]` in a loop scan the
        # filesystem again every time.
        self._opened = {}
        self._species_handler = {}

    def delete_all_diagnostics(self):
        """
        Delete all diagnostics.
        """
        self._diagnostics = {}
        self._opened = {}

    def delete_diagnostic(self, key):
        """
        Delete a diagnostic."
        """
        if key in self._diagnostics or key in self._opened:
            self._diagnostics.pop(key, None)
            self._opened.pop(key, None)
        else:
            print(f"Diagnostic {key} not found in simulation")

    def __getitem__(self, key: str) -> Diagnostic | Species_Handler:
        # check if key is a species
        if key in self._species:
            # check if species handler already exists
            if key not in self._species_handler:
                self._species_handler[key] = Species_Handler(
                    self._simulation_folder,
                    self._input_deck.species[key],
                    self._input_deck,
                )
            return self._species_handler[key]

        if key in self._diagnostics:
            return self._diagnostics[key]
        if key in self._opened:
            return self._opened[key]

        if key == "tracks":
            raise ValueError("Tracks diagnostics require a specie.")

        # Quantities that are not species related
        diag = Diagnostic(simulation_folder=self._simulation_folder, species=None, input_deck=self._input_deck)
        diag.get_quantity(key)
        self._opened[key] = _register_on_load(diag, self._diagnostics, key)
        return diag

    def add_diagnostic(self, diagnostic, name=None):
        """
        Add a custom diagnostic to the simulation.

        Parameters
        ----------
        diagnostic : Diagnostic or array-like
            The diagnostic to add. If not a Diagnostic object, it will be wrapped
            in a Diagnostic object.
        name : str, optional
            The name to use as the key for accessing the diagnostic.
            If None, an auto-generated name will be used.

        Returns
        -------
        str
            The name (key) used to store the diagnostic

        Example
        -------
        >>> sim = Simulation('path/to/simulation/input_deck.txt')
        >>> nT = sim['electrons']['n'] * sim['electrons']['T11']
        >>> sim.add_diagnostic(nT, 'nT')
        >>> sim['nT']  # Access the custom diagnostic
        """
        # Generate a name if none provided
        if name is None:
            # Find an unused name
            i = 1
            while f"custom_diag_{i}" in self._diagnostics:
                i += 1
            name = f"custom_diag_{i}"

        # If already a Diagnostic, store directly
        if isinstance(diagnostic, Diagnostic):
            self._diagnostics[name] = diagnostic
            return name
        else:
            raise ValueError("Only Diagnostic objects are supported for now")

    @property
    def species(self) -> list[str]:
        return self._species

    @property
    def loaded_diagnostics(self) -> dict[str, Diagnostic]:
        return self._diagnostics


# This is to handle species related diagnostics
class Species_Handler:
    def __init__(self, simulation_folder: str, species_name: Any, input_deck: Any) -> None:
        self._simulation_folder = simulation_folder
        self._species_name = species_name
        self._input_deck = input_deck
        self._diagnostics = {}
        self._opened = {}  # every quantity ever asked for; see Simulation.__init__

    def __getitem__(self, key: str) -> Diagnostic:
        if key in self._diagnostics:
            return self._diagnostics[key]
        if key in self._opened:
            return self._opened[key]

        if key == "tracks":
            diag = Track_Diagnostic(simulation_folder=self._simulation_folder, species=self._species_name, input_deck=self._input_deck)
        else:
            diag = Diagnostic(simulation_folder=self._simulation_folder, species=self._species_name, input_deck=self._input_deck)
            diag.get_quantity(key)

        self._opened[key] = _register_on_load(diag, self._diagnostics, key)
        return diag

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """
        Add a custom diagnostic to the simulation.

        Parameters
        ----------
        diagnostic : Diagnostic or array-like
            The diagnostic to add. If not a Diagnostic object, it will be wrapped
            in a Diagnostic object.
        name : str, optional
            The name to use as the key for accessing the diagnostic.
            If None, an auto-generated name will be used.

        Returns
        -------
        str
            The name (key) used to store the diagnostic

        """
        # Generate a name if none provided
        if name is None:
            # Find an unused name
            i = 1
            while f"custom_diag_{i}" in self._diagnostics:
                i += 1
            name = f"custom_diag_{i}"

        # If already a Diagnostic, store directly
        if isinstance(diagnostic, Diagnostic):
            self._diagnostics[name] = diagnostic
            return name
        else:
            raise ValueError("Only Diagnostic objects are supported for now")

    def delete_diagnostic(self, key: str) -> None:
        """
        Delete a diagnostic.
        """
        if key in self._diagnostics or key in self._opened:
            self._diagnostics.pop(key, None)
            self._opened.pop(key, None)
        else:
            print(f"Diagnostic {key} not found in species {self._species_name}")
            return None

    def delete_all_diagnostics(self) -> None:
        """
        Delete all diagnostics.
        """
        self._diagnostics = {}
        self._opened = {}

    @property
    def species(self) -> Any:
        return self._species_name

    @property
    def loaded_diagnostics(self) -> dict[str, Diagnostic]:
        return self._diagnostics
