import os

from ..data.diagnostic import Diagnostic
from ..decks.decks import InputDeckIO


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
    species : Specie object
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

    def __init__(self, input_deck_path):
        folder_path = os.path.dirname(input_deck_path)
        self._input_deck_path = input_deck_path
        self._input_deck = InputDeckIO(self._input_deck_path, verbose=False)

        self._species = list(self._input_deck.species.keys())

        self._simulation_folder = folder_path
        self._diagnostics = {}  # Dictionary to store diagnostics for each quantity
        self._species_handler = {}

    def delete_all_diagnostics(self):
        """
        Delete all diagnostics.
        """
        self._diagnostics = {}

    def delete_diagnostic(self, key):
        """
        Delete a diagnostic."
        """
        if key in self._diagnostics:
            del self._diagnostics[key]
        else:
            print(f"Diagnostic {key} not found in simulation")

    def __getitem__(self, key):
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

        # Create a temporary diagnostic for this quantity - this is for quantities that are not species related
        diag = Diagnostic(
            simulation_folder=self._simulation_folder,
            species=None,
            input_deck=self._input_deck,
        )
        diag.get_quantity(key)

        original_load_all = diag.load_all

        def patched_load_all(*args, **kwargs):
            result = original_load_all(*args, **kwargs)  # noqa: F841
            self._diagnostics[key] = diag
            return diag

        diag.load_all = patched_load_all

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
        >>> sim = Simulation('path/to/simulation', 'input_deck.txt')
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
        else:
            raise ValueError("Only Diagnostic objects are supported for now")

    @property
    def species(self):
        return self._species

    @property
    def loaded_diagnostics(self):
        return self._diagnostics


# This is to handle species related diagnostics
class Species_Handler:
    def __init__(self, simulation_folder, species_name, input_deck):
        self._simulation_folder = simulation_folder
        self._species_name = species_name
        self._input_deck = input_deck
        self._diagnostics = {}

    def __getitem__(self, key):
        if key in self._diagnostics:
            return self._diagnostics[key]

        # Create a temporary diagnostic for this quantity
        diag = Diagnostic(
            simulation_folder=self._simulation_folder,
            species=self._species_name,
            input_deck=self._input_deck,
        )
        diag.get_quantity(key)

        original_load_all = diag.load_all

        def patched_load_all(*args, **kwargs):
            result = original_load_all(*args, **kwargs)  # noqa: F841
            self._diagnostics[key] = diag
            return diag

        diag.load_all = patched_load_all

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
        else:
            raise ValueError("Only Diagnostic objects are supported for now")

    def delete_diagnostic(self, key):
        """
        Delete a diagnostic.
        """
        if key in self._diagnostics:
            del self._diagnostics[key]
        else:
            print(f"Diagnostic {key} not found in species {self._species_name}")
            return None

    def delete_all_diagnostics(self):
        """
        Delete all diagnostics.
        """
        self._diagnostics = {}

    @property
    def species(self):
        return self._species_name

    @property
    def loaded_diagnostics(self):
        return self._diagnostics
