from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data.diagnostic import OSIRIS_FLD, Diagnostic
from .postprocess import PostProcess

if TYPE_CHECKING:
    from ..data.simulation import Simulation

__all__ = ["FieldCentering_Simulation", "FieldCentering_Diagnostic"]

#: OSIRIS axes (1-indexed) along which each field component sits half a cell
#: off the cell centre on the Yee mesh, per dimensionality; the part_ and ext_
#: fields share their component's staggering.  E is staggered along itself, B
#: along the other two axes.
_YEE_STAGGER: dict[int, dict[str, tuple[int, ...]]] = {
    1: {"e1": (1,), "b2": (1,), "b3": (1,)},
    2: {"e1": (1,), "e2": (2,), "b1": (2,), "b2": (1,), "b3": (1, 2)},
    3: {"e1": (1,), "e2": (2,), "e3": (3,), "b1": (2, 3), "b2": (1, 3), "b3": (1, 2)},
}


class FieldCentering_Simulation(PostProcess):
    """
    Class to handle the field centering on data.It converts fields from the Osiris yee mesh to the center of the cells.
    Works as a wrapper for the FieldCentering_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    periodic : bool, optional
        Boundary treatment along the centered axes; see FieldCentering_Diagnostic.
        Defaults to True (wrap), matching OSIRIS runs with periodic boundaries.
    """

    def __init__(self, simulation: Simulation, periodic: bool = True):
        """
        Class to center the field in the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        periodic : bool, optional
            Boundary treatment along the centered axes. Defaults to True.
        """
        super().__init__("FieldCentering Simulation", simulation, species=())  # fields only
        self._periodic = periodic

    def _process(self, container, key: str) -> FieldCentering_Diagnostic:
        if key not in OSIRIS_FLD:
            raise ValueError(f"Does it make sense to center {key} field? Only {OSIRIS_FLD} are supported.")
        return FieldCentering_Diagnostic(container[key], periodic=self._periodic)


class FieldCentering_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic, periodic: bool = True):
        """
        Class to center the field in the simulation. It converts fields from the Osiris yee mesh to the center of the cells.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic object.
        periodic : bool, optional
            Boundary treatment along the axes being centered. ``True`` (default)
            wraps, pairing the first cell with the last — correct for periodic
            runs. Set ``False`` for open/conducting boundaries, where the first
            cell along a centered axis keeps its own value instead of borrowing
            from the opposite edge.
        """
        super().__init__()
        self._derive_from(diagnostic)

        self.postprocess_name = "FLD_CTR"

        if diagnostic._name not in OSIRIS_FLD:
            raise ValueError(f"Does it make sense to center {diagnostic._name} field? Only {OSIRIS_FLD} are supported.")

        self._periodic = periodic

        self._original_name = diagnostic._name
        self._name = diagnostic._name + "_centered"

        if self._dim not in _YEE_STAGGER:
            raise ValueError(f"Unknown dimension {self._dim}.")
        component = self._original_name.lower().removeprefix("part_").removeprefix("ext_")
        self._center_axes_cached = _YEE_STAGGER[self._dim].get(component, ())

    @staticmethod
    def _center_along_axis(data: np.ndarray, axis: int, periodic: bool = True) -> np.ndarray:
        """Average each cell with its left neighbour along the given numpy axis.

        ``periodic=True`` uses ``np.roll``, so the first cell pairs with the last
        one — correct only when the domain really is periodic. ``periodic=False``
        replicates the edge value instead, leaving the first cell equal to its
        own (uncentred) value rather than borrowing from the far boundary.
        """
        if periodic:
            return 0.5 * (data + np.roll(data, shift=1, axis=axis))

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (1, 0)
        shifted = np.pad(data, pad_width, mode="edge")
        take = [slice(None)] * data.ndim
        take[axis] = slice(0, data.shape[axis])
        return 0.5 * (data + shifted[tuple(take)])

    def _check_slice_safe(self, data_slice: tuple[slice, ...] | None, axes_to_center_osiris: tuple[int, ...]) -> None:
        """
        For correctness without halos: if user slices along a centering axis,
        we require full slice(None) along that axis.
        """
        if data_slice is None:
            return
        # data_slice is spatial-only, ordered (x1,x2,x3) depending on dim
        for ax_osiris in axes_to_center_osiris:
            spatial_index = ax_osiris - 1  # x1->0, x2->1, x3->2 within data_slice
            if spatial_index < len(data_slice):
                s = data_slice[spatial_index]
                if isinstance(s, slice) and (s.start is None and s.stop is None and s.step is None):
                    continue
                raise ValueError(
                    "FieldCentering with spatial slicing requires full slices (:) along axes that need centering. "
                    f"Field {self._original_name} needs centering along x{ax_osiris}, but got slice {s}. "
                    "Either use full domain on that axis, or implement halo reads."
                )

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        axes_to_center = self._center_axes_cached

        # Ensure slicing won't produce wrong values without halos
        self._check_slice_safe(data_slice, axes_to_center)

        # read one timestep lazily; returned array shape: (x1[,x2[,x3]])
        f = self._diag._frame(index, data_slice=data_slice)

        # Map OSIRIS spatial axes to numpy axes in a single-timestep array:
        # per-frame has no time dim, so x1->0, x2->1, x3->2
        result = f
        for ax_osiris in axes_to_center:
            np_axis = ax_osiris - 1
            result = self._center_along_axis(result, axis=np_axis, periodic=self._periodic)

        return result
