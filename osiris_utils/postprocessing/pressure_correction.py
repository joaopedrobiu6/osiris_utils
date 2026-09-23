from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ..data.diagnostic import Diagnostic
from .postprocess import PostProcess

if TYPE_CHECKING:
    import numpy as np

    from ..data.simulation import Simulation

OSIRIS_P = ["P11", "P12", "P13", "P22", "P23", "P33"]

__all__ = [
    "PressureCorrection_Simulation",
    "PressureCorrection_Diagnostic",
]


class PressureCorrection_Simulation(PostProcess):
    """
    Pressure tensor with the Reynolds stress subtracted: ``P_jk - n u_j v_k``.

    Pressure is species-dependent, so only ``sim["species"]["P12"]`` exists.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    """

    def __init__(self, simulation: Simulation):
        super().__init__("PressureCorrection Simulation", simulation)

    def _process(self, container: Any, key: str) -> PressureCorrection_Diagnostic:
        if key not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {key}. Supported: {OSIRIS_P}.")
        if container is self._simulation:
            raise KeyError("Pressure diagnostics are species-dependent. Use sim['species']['P12'] not sim['P12'].")

        n = container["n"]
        j, k = key[-2], key[-1]
        try:
            ufl = container[f"ufl{j}"]
        except (FileNotFoundError, ValueError, KeyError):
            # ufl is the proper velocity (gamma*v), vfl the fluid velocity —
            # different quantities, so substituting changes the physics of
            # P_jk - n*u_j*v_k. Keep the fallback for runs that only dumped
            # vfl, but never let it happen quietly.
            warnings.warn(
                f"ufl{j} is not available for species '{getattr(container, '_species_name', '?')}'; "
                f"falling back to vfl{j} when correcting {key}. ufl is the proper velocity (gamma*v) and vfl the "
                f"fluid velocity, so the correction P_{j}{k} - n*u_{j}*v_{k} will differ from the intended one. "
                f"Dump ufl{j} to avoid this.",
                stacklevel=2,
            )
            ufl = container[f"vfl{j}"]
        return PressureCorrection_Diagnostic(container[key], n, ufl, container[f"vfl{k}"])


class PressureCorrection_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic, n: Diagnostic, ufl_j: Diagnostic, vfl_k: Diagnostic):
        super().__init__()
        self._derive_from(diagnostic)

        self.postprocess_name = "P_CORR"

        if diagnostic._name not in OSIRIS_P:
            raise ValueError(f"Invalid pressure component {diagnostic._name}. Supported: {OSIRIS_P}")

        self._n = n
        self._ufl_j = ufl_j
        self._vfl_k = vfl_k

        self._original_name = diagnostic._name
        self._name = diagnostic._name + "_corrected"

        self._data: np.ndarray | None = None
        self._all_loaded = False

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
