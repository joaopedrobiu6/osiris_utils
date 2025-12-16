import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

OSIRIS_P = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]


class PressureCorrection_Simulation(PostProcess):
    def __init__(self, simulation):
        super().__init__("PressureCorrection Simulation")
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
                self._species_handler[key] = PressureCorrection_Species_Handler(self._simulation[key])
            return self._species_handler[key]
        if key not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {key}. Supported: {OSIRIS_P}.")
        if key not in self._pressure_corrected:
            print("Weird that it got here - pressure is always species dependent on OSIRIS")
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
    def __init__(self, diagnostic, n, ufl_j, vfl_k):
        """
        Class to correct the pressure in the simulation.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic object.
        """
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        self.postprocess_name = "P_CORR"

        if diagnostic._name not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {diagnostic._name}. Supported: {OSIRIS_P}")

        self._diag = diagnostic

        # The density and velocities are now passed as arguments (so it can doesn't depend on the simulation)
        self._n = n
        self._ufl_j = ufl_j
        self._vfl_k = vfl_k

        for attr in [
            "_dt",
            "_dx",
            "_ndump",
            "_axis",
            "_nx",
            "_x",
            "_grid",
            "_dim",
            "_maxiter",
            "_type",
        ]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

        self._original_name = diagnostic._name
        self._name = diagnostic._name + "_corrected"

        self._data = None
        self._all_loaded = False

    def load_all(self):
        if self._data is not None:
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()

        print(f"Loading {self._species._name} {self._original_name} diagnostic")
        self._n.load_all()
        self._ufl_j.load_all()
        self._vfl_k.load_all()

        # Then access the data
        n = self._n.data
        u = self._ufl_j.data
        v = self._vfl_k.data

        self._data = self._diag.data - n * v * u
        self._all_loaded = True

        # Unload the data to save memory
        # self._n.unload()
        # self._ufl_j.unload()
        # self._vfl_k.unload()

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
        yield (self._diag[index] - self._n[index] * self._vfl_k[index] * self._ufl_j[index])


class PressureCorrection_Species_Handler:
    """
    Class to handle pressure correction for a species.
    Acts as a wrapper for the PressureCorrection_Diagnostic class.

    Not intended to be used directly, but through the PressureCorrection_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.
    """

    def __init__(self, species_handler):
        self._species_handler = species_handler
        self._pressure_corrected = {}

    def __getitem__(self, key):
        if key not in self._pressure_corrected:
            diag = self._species_handler[key]

            # Density and velocities alwayes depend on the species so this can be done here

            n = self._species_handler["n"]
            self._j, self._k = key[-2], key[-1]
            try:
                ufl = self._species_handler[f"ufl{self._j}"]
            except Exception:
                ufl = self._species_handler[f"vfl{self._j}"]
            vfl = self._species_handler[f"vfl{self._k}"]
            self._pressure_corrected[key] = PressureCorrection_Diagnostic(diag, n, ufl, vfl)
        return self._pressure_corrected[key]
