from ..data.diagnostic import *
from ..utils import *


class Simulation:
    '''
    Class to handle the simulation data. It is a wrapper for the Diagnostic class.'
    
    Parameters
    ----------
    species : str
        The species to analyze.
    simulation_folder : str
        The simulation folder.
    
    Attributes
    ----------
    species : str
        The species to analyze.
    simulation_folder : str
        The simulation folder.
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

    Example
    -------
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> diag = sim['e1']
    >>> diag.load_all()
    
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> diag = sim['e1']
    >>> diag[<index>]
    '''
    def __init__(self, simulation_folder, species = None):
        self._species = species
        self._simulation_folder = simulation_folder
        self._diagnostics = {}  # Dictionary to store diagnostics for each quantity
    

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
        if key in self._diagnostics:
            return self._diagnostics[key]
        
        # Create a temporary diagnostic for this quantity
        diag = Diagnostic(simulation_folder=self._simulation_folder, species=self._species)
        diag.get_quantity(key)
        
        original_load_all = diag.load_all
        
        def patched_load_all(*args, **kwargs):
            result = original_load_all(*args, **kwargs)
            self._diagnostics[key] = diag
            return diag
        
        diag.load_all = patched_load_all
        
        return diag

