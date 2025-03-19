from ..data.diagnostic import *
from ..utils import *
import functools


class Simulation:
    def __init__(self, species, simulation_folder = None):
        self._species = species
        self._simulation_folder = simulation_folder
        self._diagnostics = {}  # Dictionary to store diagnostics for each quantity
    

    def _delete_all_diagnostics(self):
        self._diagnostics = {}

    def _delete_diagnostic(self, key):
        if key in self._diagnostics:
            del self._diagnostics[key]
        else:
            print(f"Diagnostic {key} not found in simulation")

    def __getitem__(self, key):
        if key in self._diagnostics:
            return self._diagnostics[key]
        
        # Create a temporary diagnostic for this quantity
        diag = Diagnostic(self._species, self._simulation_folder)
        diag.get_quantity(key)
        
        original_load_all = diag.load_all
        original_load_all_parallel = diag.load_all_parallel
        
        def patched_load_all(*args, **kwargs):
            result = original_load_all(*args, **kwargs)
            self._diagnostics[key] = diag
            return diag
        
        def patched_load_all_parallel(*args, **kwargs):
            result = original_load_all_parallel(*args, **kwargs)
            self._diagnostics[key] = diag
            return diag
            
        diag.load_all = patched_load_all
        diag.load_all_parallel = patched_load_all_parallel
        
        return diag

