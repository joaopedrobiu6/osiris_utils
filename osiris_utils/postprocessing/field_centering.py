from __future__ import annotations

import numpy as np

from ..data.diagnostic import OSIRIS_FLD, Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

__all__ = ["FieldCentering_Simulation", "FieldCentering_Diagnostic"]


class FieldCentering_Simulation(PostProcess):
    """
    Class to handle the field centering on data.It converts fields from the Osiris yee mesh to the center of the cells.
    Works as a wrapper for the FieldCentering_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    It only works for periodic boundaries.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    field : str
        The field to center.
    """

    def __init__(self, simulation: Simulation):
        """
        Class to center the field in the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object.
        """
        super().__init__("FieldCentering Simulation", simulation)
        self._field_centered = {}

    def __getitem__(self, key: str) -> FieldCentering_Diagnostic:
        if key not in OSIRIS_FLD:
            raise ValueError(f"Does it make sense to center {key} field? Only {OSIRIS_FLD} are supported.")
        if key not in self._field_centered:
            self._field_centered[key] = FieldCentering_Diagnostic(self._simulation[key])
        return self._field_centered[key]

    def delete_all(self) -> None:
        self._field_centered = {}

    def delete(self, key: str) -> None:
        if key in self._field_centered:
            del self._field_centered[key]
        else:
            print(f"Field {key} not found in simulation")


class FieldCentering_Diagnostic(Diagnostic):
    def __init__(self, diagnostic: Diagnostic):
        """
        Class to center the field in the simulation. It converts fields from the Osiris yee mesh to the center of the cells.
        It only works for periodic boundaries.

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

        self.postprocess_name = "FLD_CTR"

        if diagnostic._name not in OSIRIS_FLD:
            raise ValueError(f"Does it make sense to center {diagnostic._name} field? Only {OSIRIS_FLD} are supported.")

        self._diag = diagnostic

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
        ]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

        self._original_name = diagnostic._name
        self._name = diagnostic._name + "_centered"
        self._all_loaded = False
        self._data = None

    def _needs_center_axes_osiris(self) -> tuple[int, ...]:
        """
        Return OSIRIS spatial axes (1..3) along which this field must be averaged
        to move it to cell centers.
        """
        name = self._original_name.lower()

        if self._dim == 1:
            # data shape: (t, x1) or per-frame (x1)
            # E1 is staggered in x1; B2/B3 staggered in x1 in 1D Yee
            if name in {"e1", "part_e1", "ext_e1", "b2", "part_b2", "ext_b2", "b3", "part_b3", "ext_b3"}:
                return (1,)
            return ()

        if self._dim == 2:
            # E1,B2 staggered in x1; E2,B1 staggered in x2; B3 staggered in both x1 and x2; E3 centered
            if name in {"e1", "part_e1", "ext_e1", "b2", "part_b2", "ext_b2"}:
                return (1,)
            if name in {"e2", "part_e2", "ext_e2", "b1", "part_b1", "ext_b1"}:
                return (2,)
            if name in {"b3", "part_b3", "ext_b3"}:
                return (1, 2)
            return ()

        if self._dim == 3:
            # Typical Yee: E components are staggered along their own axis; B components staggered in the other two axes.
            # Your original code implements that. We'll preserve it:
            if name in {"e1", "part_e1", "ext_e1"}:
                return (1,)
            if name in {"e2", "part_e2", "ext_e2"}:
                return (2,)
            if name in {"e3", "part_e3", "ext_e3"}:
                return (3,)

            if name in {"b1", "part_b1", "ext_b1"}:
                return (2, 3)
            if name in {"b2", "part_b2", "ext_b2"}:
                return (1, 3)
            if name in {"b3", "part_b3", "ext_b3"}:
                return (1, 2)

            return ()

        raise ValueError(f"Unknown dimension {self._dim}.")

    @staticmethod
    def _center_along_axis(data: np.ndarray, axis: int) -> np.ndarray:
        """0.5 * (f + roll(f, +1)) along the given numpy axis."""
        return 0.5 * (data + np.roll(data, shift=1, axis=axis))

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

        # ---- eager path ----

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        self._diag.load_all()
        data = self._diag.data  # shape: (t, x1[,x2[,x3]])

        axes_to_center = self._needs_center_axes_osiris()

        # Map OSIRIS spatial axes (1..3) to numpy axes in the loaded array:
        # loaded includes time at axis 0, so x1->1, x2->2, x3->3
        result = data
        for ax_osiris in axes_to_center:
            np_axis = ax_osiris  # because time is axis 0
            result = self._center_along_axis(result, axis=np_axis)

        self._data = result
        self._all_loaded = True
        return self._data

    # ---- lazy path ----

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        axes_to_center = self._needs_center_axes_osiris()

        # Ensure slicing won't produce wrong values without halos
        self._check_slice_safe(data_slice, axes_to_center)

        # read one timestep lazily; returned array shape: (x1[,x2[,x3]])
        f = self._diag._frame(index, data_slice=data_slice)

        # Map OSIRIS spatial axes to numpy axes in a single-timestep array:
        # per-frame has no time dim, so x1->0, x2->1, x3->2
        result = f
        for ax_osiris in axes_to_center:
            np_axis = ax_osiris - 1
            result = self._center_along_axis(result, axis=np_axis)

        return result
