from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..data.diagnostic import Diagnostic
from .postprocess import PostProcess

if TYPE_CHECKING:
    import numpy as np

    from ..data.simulation import Simulation

OSIRIS_H = ["Q111", "Q222", "Q333", "Q112", "Q113", "Q221", "Q223", "Q331", "Q332"]

__all__ = [
    "HeatfluxCorrection_Simulation",
    "HeatfluxCorrection_Diagnostic",
    "HeatfluxCorrection_Species_Handler",
]


class HeatfluxCorrection_Simulation(PostProcess):
    def __init__(self, simulation: Simulation):
        super().__init__("HeatfluxCorrection Simulation", simulation)
        """
        Class to correct pressure tensor components by subtracting Reynolds stress.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        heatflux : str
            The heatflux component to center.
        """
        self._heatflux_corrected: dict[str, HeatfluxCorrection_Diagnostic] = {}
        self._species_handler: dict[str, HeatfluxCorrection_Species_Handler] = {}

    def __getitem__(self, key: str) -> HeatfluxCorrection_Species_Handler | HeatfluxCorrection_Diagnostic:
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = HeatfluxCorrection_Species_Handler(self._simulation[key], self._simulation)
            return self._species_handler[key]
        if key not in OSIRIS_H:
            raise ValueError(f"Invalid heatflux component {key}. Supported: {OSIRIS_H}.")
        if key not in self._heatflux_corrected:
            print("Weird that it got here - heatflux is always species dependent on OSIRIS")
            # This part seems to lack some arguments for HeatfluxCorrection_Diagnostic,
            # but keeping as is for structural consistency if reached
            # self._heatflux_corrected[key] = HeatfluxCorrection_Diagnostic(self._simulation[key], self._simulation)
        return self._heatflux_corrected[key]

    def delete_all(self) -> None:
        self._heatflux_corrected = {}

    def delete(self, key: str) -> None:
        if key in self._heatflux_corrected:
            del self._heatflux_corrected[key]
        else:
            print(f"Heatflux {key} not found in simulation")


class HeatfluxCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic, P_list: list[Diagnostic], vfl_list: list[Diagnostic], n: Diagnostic):
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
        self._P_list = P_list
        self._vfl_list = vfl_list
        self._n = n

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

    def _compute(
        self,
        Q: np.ndarray,
        vfl_i: np.ndarray,
        vfl_j: np.ndarray,
        vfl_k: np.ndarray,
        P_ij: np.ndarray,
        P_jk: np.ndarray,
        P_ki: np.ndarray,
        n: np.ndarray,
    ) -> np.ndarray:
        """
        Centralize the correction formula so eager/lazy always match.
        """
        return Q - (vfl_i * P_jk + vfl_j * P_ki + vfl_k * P_ij) + 2 * vfl_i * vfl_j * vfl_k * n

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()

        print(f"Loading {self._species._name} {self._original_name} diagnostic")

        for index in range(len(self._vfl_list)):
            self._vfl_list[index].load_all()

        for index in range(len(self._P_list)):
            self._P_list[index].load_all()

        self._n.load_all()

        self._data = self._compute(
            self._diag.data,
            self._vfl_list[0].data,
            self._vfl_list[1].data,
            self._vfl_list[2].data,
            self._P_list[0].data,
            self._P_list[1].data,
            self._P_list[2].data,
            self._n.data,
        )
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        """
        Lazy per-timestep correction. Reads only requested slices from disk,
        provided the underlying diagnostics support data_slice.
        """
        # Read one timestep (and only the requested spatial slice)
        Q = self._diag._frame(index, data_slice=data_slice)
        vfl_i = self._vfl_list[0]._frame(index, data_slice=data_slice)
        vfl_j = self._vfl_list[1]._frame(index, data_slice=data_slice)
        vfl_k = self._vfl_list[2]._frame(index, data_slice=data_slice)

        Pij = self._P_list[0]._frame(index, data_slice=data_slice)
        Pjk = self._P_list[1]._frame(index, data_slice=data_slice)
        Pki = self._P_list[2]._frame(index, data_slice=data_slice)

        return self._compute(Q, vfl_i, vfl_j, vfl_k, Pij, Pjk, Pki, self._n._frame(index, data_slice=data_slice))


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

    def __init__(self, species_handler: Any, simulation: Simulation):
        self._species_handler = species_handler
        self._simulation = simulation
        self._heatflux_corrected: dict[str, HeatfluxCorrection_Diagnostic] = {}

    def __getitem__(self, key: str) -> HeatfluxCorrection_Diagnostic:
        if key not in self._heatflux_corrected:
            diag = self._species_handler[key]

            # Velocities alwayes depend on the species so this can be done here

            # Q_ijk

            i = int(key[-3])  # Get i from 'Q111', 'Q222', etc.
            j = int(key[-2])  # Get j from 'Q111', 'Q222', etc.
            k = int(key[-1])  # Get k from 'Q111', 'Q222', etc.

            vfl_i = self._species_handler[f"vfl{i}"]
            vfl_j = self._species_handler[f"vfl{j}"]
            vfl_k = self._species_handler[f"vfl{k}"]

            # Load Pij, Pjk, Pki
            Pij = self._species_handler[f"P{i}{j}"]
            Pjk = self._species_handler[f"P{j}{k}"]
            Pki = self._species_handler[f"P{k}{i}"]

            n = self._species_handler["n"]

            self._heatflux_corrected[key] = HeatfluxCorrection_Diagnostic(diag, vfl_i, [Pij, Pjk, Pki], [vfl_i, vfl_j, vfl_k], n)
        return self._heatflux_corrected[key]
