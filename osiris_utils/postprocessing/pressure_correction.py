from ..utils import *
from ..data.simulation import Simulation
from .postprocess import PostProcess
from ..data.diagnostic import Diagnostic

OSIRIS_P = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]

class PressureCorrection_Simulation(PostProcess):
    def __init__(self, simulation):
        super().__init__(f"PressureCorrection Simulation")
        """
        Class to correct pressure tensor components by subtracting Reynolds stress.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        pressure : str
            The pressure component to center.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._pressure_corrected = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = PressureCorrection_Species_Handler(self._simulation[key], self._simulation)
            return self._species_handler[key]
        if key not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {key}. Supported: {OSIRIS_P}.")
        if key not in self._pressure_corrected:
            self._pressure_corrected[key] = PressureCorrection_Diagnostic(self._simulation[key], self._simulation)
        return self._pressure_corrected[key]

    
    def delete_all(self):
        self._pressure_corrected = {}

    def delete(self, key):
        if key in self._pressure_corrected:
            del self._pressure_corrected[key]
        else:
            print(f"Pressure {key} not found in simulation")
    
    def process(self, diagnostic):
        """Apply pressure correction to a diagnostic"""
        return PressureCorrection_Diagnostic(diagnostic, self._simulation)
    
class PressureCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic, simulation):

        """
        Class to correct the pressure in the simulation.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic object.
        """
        if hasattr(diagnostic, '_species'):
            super().__init__(simulation_folder=diagnostic._simulation_folder if hasattr(diagnostic, '_simulation_folder') else None, 
                             species=diagnostic._species)
        else:
            super().__init__(None)
        
        if diagnostic._name not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {diagnostic._name}. Supported: {OSIRIS_P}")
        
        self._diag = diagnostic
        self._simulation = simulation

        for attr in ['_dt', '_dx', '_ndump', '_axis', '_nx', '_x', '_grid', '_dim', '_maxiter']:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

        self._original_name = diagnostic._name
        self._name = diagnostic._name + "_corrected"

        
        self._j, self._k = self._original_name[-2], self._original_name[-1]


        self._data = None
        self._all_loaded = False

    def load_all(self):
        if self._data is not None:
            return self._data
        
        if not hasattr(self._diag, '_data') or self._diag._data is None:
            self._diag.load_all()

        print(f"Loading {self._species._name} {self._original_name} diagnostic")
        self._simulation[self._species._name]["n"].load_all()
        self._simulation[self._species._name][f"ufl{self._j}"].load_all()
        self._simulation[self._species._name][f"vfl{self._k}"].load_all()

        # Then access the data
        n = self._simulation[self._species._name]["n"].data
        u = self._simulation[self._species._name][f"ufl{self._j}"].data
        v = self._simulation[self._species._name][f"vfl{self._k}"].data
        
        self._data = self._diag.data - n * v * u
        self._all_loaded = True
        return self._data
    
    def __getitem__(self, index):
        """Get data at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]
        
        if isinstance(index, int):
            return next(self._data_generator(index))
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            stop = self._diag._maxiter if index.stop is None else index.stop
            return np.array([next(self._data_generator(i)) for i in range(start, stop, step)])
        else:
            raise ValueError("Invalid index type. Use int or slice.")

    def _data_generator(self, index):
        n = self._simulation[self._species]["n"][index]
        v = self._simulation[self._species][f"vfl{self._k}"][index]
        u = self._simulation[self._species][f"ufl{self._j}"][index]
        yield self._diag[index] - n * v * u
        
class PressureCorrection_Species_Handler:
    """
    Class to handle derivatives for a species.
    Acts as a wrapper for the Derivative_Diagnostic class.

    Not intended to be used directly, but through the Derivative_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.
    """
    def __init__(self, species_handler, simulation):
        self._species_handler = species_handler
        self._simulation = simulation
        self._derivatives_computed = {}

    def __getitem__(self, key):
        if key not in self._derivatives_computed:
            diag = self._species_handler[key]
            self._derivatives_computed[key] = PressureCorrection_Diagnostic(diag, self._simulation)
        return self._derivatives_computed[key]