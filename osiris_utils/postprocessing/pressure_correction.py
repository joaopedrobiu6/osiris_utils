from __future__ import annotations

from typing import Any

import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

OSIRIS_P = ["P11", "P12", "P13", "P22", "P23", "P33"]

__all__ = [
    "PressureCorrection_Simulation",
    "PressureCorrection_Diagnostic",
    "PressureCorrection_Species_Handler",
]


class PressureCorrection_Simulation(PostProcess):
    def __init__(self, simulation: Simulation):
        super().__init__("PressureCorrection Simulation", simulation)
        """
        Class to correct pressure tensor components by subtracting Reynolds stress.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        pressure : str
            The pressure component to center.
        """
        self._pressure_corrected: dict[str, PressureCorrection_Diagnostic] = {}
        self._species_handler: dict[str, PressureCorrection_Species_Handler] = {}

    def __getitem__(self, key: str) -> PressureCorrection_Species_Handler | PressureCorrection_Diagnostic:
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = PressureCorrection_Species_Handler(self._simulation[key])
            return self._species_handler[key]
        if key not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {key}. Supported: {OSIRIS_P}.")
        if key not in self._pressure_corrected:
            raise KeyError("Pressure diagnostics are species-dependent. Use sim['species']['P12'] not sim['P12'].")
        return self._pressure_corrected[key]

    def delete_all(self) -> None:
        self._pressure_corrected = {}

    def delete(self, key: str) -> None:
        if key in self._pressure_corrected:
            del self._pressure_corrected[key]
        else:
            print(f"Pressure {key} not found in simulation")


class PressureCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic, n: Diagnostic, ufl_j: Diagnostic, vfl_k: Diagnostic):
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=getattr(diagnostic, "_simulation_folder", None),
                species=getattr(diagnostic, "_species", None),
            )
        else:
            super().__init__(None)

        self.postprocess_name = "P_CORR"

        if diagnostic._name not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {diagnostic._name}. Supported: {OSIRIS_P}")

        self._diag = diagnostic
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

        self._data: np.ndarray | None = None
        self._all_loaded = False

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        # Ensure all operands are loaded
        if not getattr(self._diag, "_all_loaded", False) or self._diag._data is None:
            self._diag.load_all()
        if not getattr(self._n, "_all_loaded", False) or self._n._data is None:
            self._n.load_all()
        if not getattr(self._ufl_j, "_all_loaded", False) or self._ufl_j._data is None:
            self._ufl_j.load_all()
        if not getattr(self._vfl_k, "_all_loaded", False) or self._vfl_k._data is None:
            self._vfl_k.load_all()

        # Core formula: Pjk - n * u_j * v_k
        self._data = self._diag._data - self._n._data * self._ufl_j._data * self._vfl_k._data
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        """
        Lazy per-timestep evaluation, supports spatial slicing.
        data_slice applies to spatial axes only (same convention as Diagnostic).
        """
        P = self._diag._frame(index, data_slice=data_slice)
        n = self._n._frame(index, data_slice=data_slice)
        u = self._ufl_j._frame(index, data_slice=data_slice)
        v = self._vfl_k._frame(index, data_slice=data_slice)
        return P - n * u * v


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

    def __init__(self, species_handler: Any):
        self._species_handler = species_handler
        self._pressure_corrected: dict[str, PressureCorrection_Diagnostic] = {}

    def __getitem__(self, key: str) -> PressureCorrection_Diagnostic:
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
