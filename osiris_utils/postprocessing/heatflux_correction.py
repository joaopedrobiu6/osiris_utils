from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..data.diagnostic import Diagnostic
from .postprocess import PostProcess

if TYPE_CHECKING:
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

    def _process(self, container: Any, key: str) -> Diagnostic:
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


def HeatfluxCorrection_Diagnostic(
    diagnostic: Diagnostic, P_list: list[Diagnostic], vfl_list: list[Diagnostic], n: Diagnostic
) -> Diagnostic:  # noqa: N802
    """``Q_ijk - (v_i P_jk + v_j P_ki + v_k P_ij) + 2 v_i v_j v_k n``: the heat flux in the fluid frame.

    Parameters
    ----------
    diagnostic : Diagnostic
        ``Q_ijk``.
    P_list : list of Diagnostic
        ``[P_ij, P_jk, P_ki]``.
    vfl_list : list of Diagnostic
        ``[v_i, v_j, v_k]``.
    n : Diagnostic
        Number density.

    Plain Diagnostic arithmetic, so it stays lazy like any other expression.
    """
    if diagnostic._name not in OSIRIS_H:
        raise ValueError(f"Invalid heatflux component {diagnostic._name}. Supported: {OSIRIS_H}")
    P_ij, P_jk, P_ki = P_list
    v_i, v_j, v_k = vfl_list
    corrected = diagnostic - (v_i * P_jk + v_j * P_ki + v_k * P_ij) + 2 * v_i * v_j * v_k * n
    corrected._name = diagnostic._name + "_corrected"
    corrected.created_diagnostic_name = "HFL_CORR"  # to_h5's default folder
    return corrected
