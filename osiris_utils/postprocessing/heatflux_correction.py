import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess
from .pressure_correction import PressureCorrection_Simulation

OSIRIS_H = ["q1", "q2", "q3"]


class HeatfluxCorrection_Simulation(PostProcess):
    def __init__(self, simulation):
        super().__init__("HeatfluxCorrection Simulation")
        """
        Class to correct pressure tensor components by subtracting Reynolds stress.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        heatflux : str
            The heatflux component to center.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._heatflux_corrected = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = HeatfluxCorrection_Species_Handler(self._simulation[key], self._simulation)
            return self._species_handler[key]
        if key not in OSIRIS_H:
            raise ValueError(f"Invalid heatflux component {key}. Supported: {OSIRIS_H}.")
        if key not in self._heatflux_corrected:
            print("Weird that it got here - heatflux is always species dependent on OSIRIS")
            self._heatflux_corrected[key] = HeatfluxCorrection_Diagnostic(self._simulation[key], self._simulation)
        return self._heatflux_corrected[key]

    def delete_all(self):
        self._heatflux_corrected = {}

    def delete(self, key):
        if key in self._heatflux_corrected:
            del self._heatflux_corrected[key]
        else:
            print(f"Heatflux {key} not found in simulation")

    def process(self, diagnostic):
        """Apply heatflux correction to a diagnostic"""
        return HeatfluxCorrection_Diagnostic(diagnostic, self._simulation)


class HeatfluxCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic, vfl_i, Pjj_list, vfl_j_list, Pji_list):
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

        self.postprocess_name = "HFL_CORR"

        if diagnostic._name not in OSIRIS_H:
            raise ValueError(f"Invalid heatflux component {diagnostic._name}. Supported: {OSIRIS_H}")

        self._diag = diagnostic

        # The density and velocities are now passed as arguments (so it can doesn't depend on the simulation)
        self._vfl_i = vfl_i
        self._Pjj_list = Pjj_list
        self._vfl_j_list = vfl_j_list
        self._Pji_list = Pji_list

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

        self._vfl_i.load_all()

        for vfl_j in self._vfl_j_list:
            vfl_j.load_all()
        for Pji in self._Pji_list:
            Pji.load_all()
        for Pjj in self._Pjj_list:
            Pjj.load_all()

        q = self._diag.data
        vfl_i = self._vfl_i.data

        trace_P = sum(Pjj.data for Pjj in self._Pjj_list)

        # Sum over j: vfl_j * Pji
        vfl_dot_Pji = sum(vfl_j.data * Pji.data for vfl_j, Pji in zip(self._vfl_j_list, self._Pji_list, strict=False))

        self._data = 2 * q - 0.5 * vfl_i * trace_P - vfl_dot_Pji
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
        q = self._diag[index]
        vfl_i = self._vfl_i[index]
        trace_P = sum(Pjj[index] for Pjj in self._Pjj_list)
        vfl_dot_Pji = sum(vfl_j[index] * Pji[index] for vfl_j, Pji in zip(self._vfl_j_list, self._Pji_list, strict=False))
        yield 2 * q - 0.5 * vfl_i * trace_P - vfl_dot_Pji


class HeatfluxCorrection_Species_Handler:
    """
    Class to handle heatflux correction for a species.
    Acts as a wrapper for the HeatfluxCorrection_Diagnostic class.

    Not intended to be used directly, but through the HeatfluxCorrection_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    simulation : Simulation
        The simulation object.
    """

    def __init__(self, species_handler, simulation):
        self._species_handler = species_handler
        self._simulation = simulation
        self._heatflux_corrected = {}

    def __getitem__(self, key):
        if key not in self._heatflux_corrected:
            diag = self._species_handler[key]

            # Velocities alwayes depend on the species so this can be done here

            i = int(key[-1])  # Get i from 'q1', 'q2', etc.

            vfl_i = self._species_handler[f"vfl{i}"]

            # Load trace(P): sum over Pjj
            Pjj_list = [self._species_handler[f"P{j}{j}"] for j in range(1, diag._dim + 1)]

            # Compute quantities for vfl_j * P_{ji}
            vfl_j_list = [self._species_handler[f"vfl{j}"] for j in range(1, diag._dim + 1)]
            Pji_list = [PressureCorrection_Simulation(self._simulation)[diag._species._name][f"P{j}{i}"] for j in range(1, diag._dim + 1)]

            self._heatflux_corrected[key] = HeatfluxCorrection_Diagnostic(diag, vfl_i, Pjj_list, vfl_j_list, Pji_list)
        return self._heatflux_corrected[key]
