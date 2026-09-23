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
]


class HeatfluxCorrection_Simulation(PostProcess):
    """
    Heat flux in the fluid frame:
    ``Q_ijk - (v_i P_jk + v_j P_ki + v_k P_ij) + 2 v_i v_j v_k n``.

    Heat flux is species-dependent, so only ``sim["species"]["Q112"]`` exists.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    """

    def __init__(self, simulation: Simulation):
        super().__init__("HeatfluxCorrection Simulation", simulation)

    def _process(self, container: Any, key: str) -> HeatfluxCorrection_Diagnostic:
        if key not in OSIRIS_H:
            raise ValueError(f"Invalid heatflux component {key}. Supported: {OSIRIS_H}.")
        if container is self._simulation:
            raise KeyError("Heat-flux diagnostics are species-dependent. Use sim['species']['Q112'] not sim['Q112'].")

        i, j, k = (int(c) for c in key[-3:])  # Q_ijk

        # The pressure tensor is symmetric and OSIRIS only dumps the upper
        # triangle (P11, P12, P13, P22, P23, P33). Order the index pair, or
        # mixed components ask for P21/P31/P32 and fail.
        def P(a: int, b: int) -> Diagnostic:
            return container[f"P{min(a, b)}{max(a, b)}"]

        vfl = [container[f"vfl{a}"] for a in (i, j, k)]
        return HeatfluxCorrection_Diagnostic(container[key], [P(i, j), P(j, k), P(k, i)], vfl, container["n"])


class HeatfluxCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic, P_list: list[Diagnostic], vfl_list: list[Diagnostic], n: Diagnostic):
        """
        Class to correct the pressure in the simulation.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic object.
        """
        super().__init__()
        self._derive_from(diagnostic)

        self.postprocess_name = "HFL_CORR"

        if diagnostic._name not in OSIRIS_H:
            raise ValueError(f"Invalid heatflux component {diagnostic._name}. Supported: {OSIRIS_H}")

        # The density and velocities are now passed as arguments (so it can doesn't depend on the simulation)
        self._P_list = P_list
        self._vfl_list = vfl_list
        self._n = n

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
