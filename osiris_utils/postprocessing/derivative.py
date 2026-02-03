from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

__all__ = ["Derivative_Diagnostic", "Derivative_Simulation", "Derivative_Species_Handler"]


class Derivative_Simulation(PostProcess):
    """Class to compute the derivative of a diagnostic. Works as a wrapper for the Derivative_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    This class can be initialized with either a Simulation object or another Derivative_Simulation object,
    allowing for chaining derivatives (e.g., second derivative = Derivative_Simulation(Derivative_Simulation(...))).

    Parameters
    ----------
    simulation : Simulation or Derivative_Simulation
        The simulation object or another derivative simulation object.
    deriv_type : str
        The type of derivative to compute. Options are:
        - 't' for time derivative.
        - 'x1' for first spatial derivative.
        - 'x2' for second spatial derivative.
        - 'x3' for third spatial derivative.
        - 'xx' for second spatial derivative in two axis.
        - 'xt' for mixed derivative in time and one spatial axis.
        - 'tx' for mixed derivative in one spatial axis and time.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.
    order : int
        The order of the derivative. Currently only 2 and 4 are supported.
        Order 2 uses central differences with edge_order=2 in numpy.gradient.
        Order 4 uses a higher order finite difference scheme. For the edge points,
        a lower order scheme is used to avoid going out of bounds.

    """

    def __init__(
        self,
        simulation: Simulation | Derivative_Simulation,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 2,
    ):
        super().__init__(f"Derivative({deriv_type})")
        # Accept both Simulation and Derivative_Simulation objects
        if not isinstance(simulation, (Simulation, Derivative_Simulation)):
            raise ValueError("simulation must be a Simulation or Derivative_Simulation object.")
        self._simulation = simulation
        self._deriv_type = deriv_type
        self._axis = axis
        self._derivatives_computed = {}
        self._species_handler = {}
        self._order = order

        # Copy species list to make this class behave like a Simulation
        if hasattr(simulation, "_species"):
            self._species = simulation._species
        else:
            self._species = []

    def __getitem__(self, key: Any) -> Derivative_Species_Handler | Derivative_Diagnostic:
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = Derivative_Species_Handler(self._simulation[key], self._deriv_type, self._axis, self._order)
            return self._species_handler[key]

        if key not in self._derivatives_computed:
            self._derivatives_computed[key] = Derivative_Diagnostic(
                diagnostic=self._simulation[key],
                deriv_type=self._deriv_type,
                axis=self._axis,
                order=self._order,
            )
        return self._derivatives_computed[key]

    def delete_all(self) -> None:
        self._derivatives_computed = {}

    def delete(self, key: Any) -> None:
        if key in self._derivatives_computed:
            del self._derivatives_computed[key]
        else:
            print(f"Derivative {key} not found in simulation")

    def process(self, diagnostic: Diagnostic) -> Derivative_Diagnostic:
        """Apply derivative to a diagnostic"""
        return Derivative_Diagnostic(diagnostic, self._deriv_type, self._axis)

    @property
    def species(self) -> list:
        """Return list of species, making this compatible with Simulation interface"""
        return self._species

    @property
    def loaded_diagnostics(self) -> dict:
        """Return loaded diagnostics, making this compatible with Simulation interface"""
        return self._derivatives_computed

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Add a custom diagnostic to the derivative simulation.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic to add.
        name : str, optional
            The name to use as the key for accessing the diagnostic.
            If None, an auto-generated name will be used.

        Returns
        -------
        str
            The name (key) used to store the diagnostic

        """
        if name is None:
            i = 1
            while f"custom_diag_{i}" in self._derivatives_computed:
                i += 1
            name = f"custom_diag_{i}"

        if isinstance(diagnostic, Diagnostic):
            self._derivatives_computed[name] = diagnostic
            return name
        raise ValueError("Only Diagnostic objects are supported")


class Derivative_Diagnostic(Diagnostic):
    """Auxiliar class to compute the derivative of a diagnostic, for it to be similar in behavior to a Diagnostic object.
    Inherits directly from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    deriv_type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types

    Methods
    -------
    load_all()
        Load all the data and compute the derivative.
    __getitem__(index)
        Get data at a specific index.

    """

    def __init__(self, diagnostic: Diagnostic, deriv_type: str, axis: int | tuple[int, int] | None = None, order: int = 2) -> None:
        # Initialize using parent's __init__ with the same species
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        self.postprocess_name = "DERIV"

        # self._name = f"D[{diagnostic._name}, {type}]"
        self._diag = diagnostic
        self._deriv_type = deriv_type
        self._axis = axis if axis is not None else diagnostic._axis
        self._data = None
        self._all_loaded = False
        self._order = order

        # Copy all relevant attributes from diagnostic
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

    @staticmethod
    def _compute_fourth_order_spatial(data: np.ndarray, dx: float, axis: int) -> np.ndarray:
        """Compute 4th-order spatial derivative along specified axis using vectorized operations.

        Uses the 4th-order central difference stencil:
        Interior: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
        Boundaries: 2nd-order forward/backward differences

        Parameters
        ----------
        data : np.ndarray
            Input data array
        dx : float
            Grid spacing
        axis : int
            Axis along which to compute derivative

        Returns
        -------
        np.ndarray
            Derivative of the input data

        """
        result = np.zeros_like(data)

        # Build slice objects for vectorized operations
        # This is much faster than looping through indices

        # Central differences (vectorized)
        # For each point i in [2, n-2), compute:
        # (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)

        slices_center = [slice(None)] * data.ndim
        slices_p2 = [slice(None)] * data.ndim
        slices_p1 = [slice(None)] * data.ndim
        slices_m1 = [slice(None)] * data.ndim
        slices_m2 = [slice(None)] * data.ndim

        # Target region: indices 2 to -2
        slices_center[axis] = slice(2, -2)
        # For the stencil, we need aligned slices
        slices_p2[axis] = slice(4, None)  # i+2: starts at 4, goes to end
        slices_p1[axis] = slice(3, -1)  # i+1: starts at 3, goes to -1
        slices_m1[axis] = slice(1, -3)  # i-1: starts at 1, goes to -3
        slices_m2[axis] = slice(0, -4)  # i-2: starts at 0, goes to -4

        result[tuple(slices_center)] = (-data[tuple(slices_p2)] + 8 * data[tuple(slices_p1)] - 8 * data[tuple(slices_m1)] + data[tuple(slices_m2)]) / (12 * dx)

        # Boundary points using 2nd-order differences
        # First point: forward difference
        slices_0 = [slice(None)] * data.ndim
        slices_1 = [slice(None)] * data.ndim
        slices_2 = [slice(None)] * data.ndim
        slices_0[axis] = 0
        slices_1[axis] = 1
        slices_2[axis] = 2
        result[tuple(slices_0)] = (-3 * data[tuple(slices_0)] + 4 * data[tuple(slices_1)] - data[tuple(slices_2)]) / (2 * dx)

        # Second point: central difference
        result[tuple(slices_1)] = (data[tuple(slices_2)] - data[tuple(slices_0)]) / (2 * dx)

        # Second-to-last point: central difference
        slices_m2 = [slice(None)] * data.ndim
        slices_m1 = [slice(None)] * data.ndim
        slices_m3 = [slice(None)] * data.ndim
        slices_m2[axis] = -2
        slices_m1[axis] = -1
        slices_m3[axis] = -3
        result[tuple(slices_m2)] = (data[tuple(slices_m1)] - data[tuple(slices_m3)]) / (2 * dx)

        # Last point: backward difference
        result[tuple(slices_m1)] = (3 * data[tuple(slices_m1)] - 4 * data[tuple(slices_m2)] + data[tuple(slices_m3)]) / (2 * dx)

        return result

    def load_all(self) -> np.ndarray:
        """Load all data and compute the derivative"""
        if self._data is not None:
            print("Using cached derivative")
            return self._data

        # Load diagnostic data if needed
        if not self._diag._all_loaded:
            self._diag.load_all()

        # Use diagnostic data
        print("Using cached data from diagnostic")
        self._data = self._diag._data

        if self._order == 2:
            if self._deriv_type == "t":
                result = np.gradient(self._data, self._diag._dt * self._diag._ndump, axis=0, edge_order=2)

            elif self._deriv_type == "x1":
                # Handle dx - extract scalar for 1D, use first element for multi-D
                dx = self._diag._dx
                if self._dim == 1 and isinstance(dx, (list, tuple, np.ndarray)):
                    dx = dx[0] if len(dx) >= 1 else dx
                elif self._dim > 1:
                    dx = self._diag._dx[0]
                result = np.gradient(self._data, dx, axis=1, edge_order=2)

            elif self._deriv_type == "x2":
                result = np.gradient(self._data, self._diag._dx[1], axis=2, edge_order=2)

            elif self._deriv_type == "x3":
                result = np.gradient(self._data, self._diag._dx[2], axis=3, edge_order=2)

            elif self._deriv_type == "xx":
                if len(self._axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._axis[0] - 1],
                        axis=self._axis[0],
                        edge_order=2,
                    ),
                    self._diag._dx[self._axis[1] - 1],
                    axis=self._axis[1],
                    edge_order=2,
                )

            elif self._deriv_type == "xt":
                if not isinstance(self._axis, int):
                    raise ValueError("Axis must be an integer.")
                result = np.gradient(
                    np.gradient(self._data, self._diag._dt, axis=0, edge_order=2),
                    self._diag._dx[self._axis - 1],
                    axis=self._axis,
                    edge_order=2,
                )

            elif self._deriv_type == "tx":
                if not isinstance(self._axis, int):
                    raise ValueError("Axis must be an integer.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._axis - 1],
                        axis=self._axis,
                        edge_order=2,
                    ),
                    self._diag._dt,
                    axis=0,
                    edge_order=2,
                )
            else:
                raise ValueError("Invalid derivative type.")

        elif self._order == 4:
            if self._deriv_type in ["x1", "x2", "x3"]:
                axis = {"x1": 1, "x2": 2, "x3": 3}[self._deriv_type]
                # Extract dx as a scalar
                if self._dim > 1:
                    dx = self._diag._dx[axis - 1]
                else:
                    # For 1D, _dx might be a list with one element or a scalar
                    dx = self._diag._dx[0] if isinstance(self._diag._dx, (list, tuple, np.ndarray)) else self._diag._dx
                # Ensure dx is a scalar float
                if isinstance(dx, (list, tuple, np.ndarray)):
                    dx = float(dx) if np.isscalar(dx) else float(dx[0])
                # Use vectorized helper function for massive speedup
                result = self._compute_fourth_order_spatial(self._data, dx, axis)
            else:
                raise ValueError("Order 4 is only implemented for spatial derivatives 'x1', 'x2' and 'x3'.")

        # Store the result in the cache
        self._all_loaded = True
        self._data = result
        return self._data

    def _get_padded_slice(self, req_slice: int | slice, max_len: int, halo: int) -> tuple[slice, slice, bool]:
        """
        Calculate the read slice (with halo) and the crop slice to extract the original request.

        This method ensures that we read enough extra data (the halo) to compute derivatives
        at the boundaries of the requested slice, while respecting the physical limits of the data array.

        Parameters
        ----------
        req_slice : int | slice
            The requested index or slice from the user.
        max_len : int
            The maximum length of this dimension (grid size or time steps).
        halo : int
            The number of extra points needed on each side (1 for 2nd order, 2 for 4th order).

        Returns
        -------
        read_slice : slice
            The slice to read from the source data (includes halo).
        crop_slice : slice
            The slice to apply to the read chunk to get the exact requested data.
        squeeze : bool
            Whether the dimension was originally an integer and should be squeezed from the final result.
        """
        squeeze = False

        if isinstance(req_slice, int):
            squeeze = True
            # Handle negative indices (e.g., -1 becomes max_len - 1)
            if req_slice < 0:
                req_slice += max_len
            # Convert integer index to a single-element slice [i : i+1]
            start, stop = req_slice, req_slice + 1
        else:
            # Handle slice object: normalize None to 0 or max_len
            start = 0 if req_slice.start is None else req_slice.start
            stop = max_len if req_slice.stop is None else req_slice.stop
            # Handle negative start/stop indices
            if start < 0:
                start += max_len
            if stop < 0:
                stop += max_len

        # We try to expand bounds by 'halo' on both sides
        read_start = start - halo
        read_stop = stop + halo

        # Ensure we don't try to read before index 0.
        # NOTE: We assume non-periodic boundaries.
        # If we are at the edge, we clamp the read to the physical limit.
        # The derivative computation handles the reduced stencil size / lower order methods at these boundaries.
        pad_left = 0
        if read_start < 0:
            read_start = 0
            # Track how much we couldn't expand to the left.
            # This shifts where our desired data sits relative to the read chunk start.
            pad_left = start - 0
        else:
            # We successfully expanded by full halo, so the data of interest starts 'halo' index into the chunk
            pad_left = halo

        # Ensure we don't read past the end of the array
        if read_stop > max_len:
            read_stop = max_len

        read_slice = slice(read_start, read_stop)

        # The 'read_slice' gives us a chunk of data. We now need to slice *that chunk*
        # to get back exactly what the user asked for.

        # The requested data starts at 'start'. The chunk starts at 'read_start'.
        # So relative to the chunk, our data starts at:
        relative_start = start - read_start

        # We need to extract the length of the original request
        req_len = stop - start

        relative_stop = relative_start + req_len

        # Handle stride/step from original request
        step = 1
        if isinstance(req_slice, slice) and req_slice.step is not None:
            step = req_slice.step

        # This slice, when applied to the chunk, returns the original requested region
        crop_slice = slice(relative_start, relative_stop, step)

        return read_slice, crop_slice, squeeze

    def _compute_derivative(self, data: np.ndarray, deriv_type: str, axis_map: dict) -> np.ndarray:
        """
        Compute derivative on a loaded data chunk.

        This method handles the actual numerical differentiation. It is designed to work
        on an arbitrary chunk of data, provided an 'axis_map' that links logical names
        (like 't', 'x1') to the actual axes of the 'data' array.

        NOTE: This method does NOT assume periodic boundaries.
        - For order 2: Uses np.gradient which applies 2nd-order accurate one-sided differences at boundaries.
        - For order 4: Uses explicit lower-order (2nd order) stencils at the physical boundaries (indices 0, 1, -2, -1).
        """

        # axis_map maps 't'->0, 'x1'->1, etc.

        if self._order == 2:
            # --- Time Derivative (t) ---
            if deriv_type == "t":
                return np.gradient(data, self._diag._dt * self._diag._ndump, axis=axis_map['t'], edge_order=2)

            # --- Spatial Derivatives (x1, x2, x3) ---
            elif deriv_type in ["x1", "x2", "x3"]:
                ax_idx = axis_map[deriv_type]
                dx = self._diag._dx
                # Handle dx being scalar (1D) or list (multi-D)
                if self._dim > 1:
                    dx = dx[int(deriv_type[1:]) - 1]  # x1 is index 0
                else:
                    dx = dx[0] if isinstance(dx, (list, tuple, np.ndarray)) else dx

                return np.gradient(data, dx, axis=ax_idx, edge_order=2)

            # --- Mixed Derivatives (xx) ---
            elif deriv_type == "xx":
                if len(self._axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")

                # Identify axes indices and spacing for both derivatives
                ax1 = axis_map[f"x{self._axis[0]}"]
                ax2 = axis_map[f"x{self._axis[1]}"]
                dx1 = self._diag._dx[self._axis[0] - 1]
                dx2 = self._diag._dx[self._axis[1] - 1]

                # Apply gradients sequentially
                res = np.gradient(data, dx1, axis=ax1, edge_order=2)
                return np.gradient(res, dx2, axis=ax2, edge_order=2)

            # --- Mixed Derivatives (xt) ---
            elif deriv_type == "xt":
                ax_mp = axis_map[f"x{self._axis}"]
                dx = self._diag._dx[self._axis - 1]

                # d/dt then d/dx
                res = np.gradient(data, self._diag._dt, axis=axis_map['t'], edge_order=2)
                return np.gradient(res, dx, axis=ax_mp, edge_order=2)

            # --- Mixed Derivatives (tx) ---
            elif deriv_type == "tx":
                ax_mp = axis_map[f"x{self._axis}"]
                dx = self._diag._dx[self._axis - 1]

                # d/dx then d/dt
                res = np.gradient(data, dx, axis=ax_mp, edge_order=2)
                return np.gradient(res, self._diag._dt, axis=axis_map['t'], edge_order=2)

        elif self._order == 4:
            # --- 4th Order Spatial ---
            if deriv_type in ["x1", "x2", "x3"]:
                ax_idx = axis_map[deriv_type]
                dx = self._diag._dx
                if self._dim > 1:
                    dx = dx[int(deriv_type[1:]) - 1]
                else:
                    dx = dx[0] if isinstance(dx, (list, tuple, np.ndarray)) else dx

                # Ensure dx is scalar float for the helper function
                if isinstance(dx, (list, tuple, np.ndarray)):
                    dx = float(dx) if np.isscalar(dx) else float(dx[0])

                return self._compute_fourth_order_spatial(data, dx, ax_idx)

            # --- 4th Order Time ---
            elif deriv_type == "t":
                # Vectorized 4th order time using the spatial helper
                # because 't' is just another axis here
                axis = axis_map['t']
                dt = self._diag._dt * self._diag._ndump
                return self._compute_fourth_order_spatial(data, dt, axis)

        raise ValueError(f"Derivative type {deriv_type} with order {self._order} not fully supported via slice optimization.")

    def _read_and_compute(self, time_idx: int | slice, spatial_indices: tuple) -> np.ndarray:
        """
        Helper method to read partial data and compute derivative.
        Used by both __getitem__ and _data_generator.
        """
        # Pad spatial indices if the user didn't specify all dimensions
        if len(spatial_indices) < self._dim:
            # Append slice(None) aka ':' for missing dimensions
            spatial_indices = spatial_indices + (slice(None),) * (self._dim - len(spatial_indices))

        halo = 2 if self._order == 4 else 1

        # Identify which axes actually NEED a halo
        axes_needing_halo = set()

        if self._deriv_type == 't':
            axes_needing_halo.add(0)  # Time axis
        elif self._deriv_type.startswith('x') and len(self._deriv_type) == 2 and self._deriv_type[1].isdigit():
            dim_idx = int(self._deriv_type[1])
            axes_needing_halo.add(dim_idx)
        elif self._deriv_type == 'xx':
            axes_needing_halo.add(self._axis[0])
            axes_needing_halo.add(self._axis[1])
        elif self._deriv_type in ['xt', 'tx']:
            axes_needing_halo.add(0)  # time
            axes_needing_halo.add(self._axis)  # spatial

        read_slices = []  # Slices to read from disk
        crop_slices = []  # Slices to cut result
        squeeze_map = []  # Dimensions to squeeze

        # -> Process Time Axis (0)
        max_t = self._diag._maxiter
        h = halo if 0 in axes_needing_halo else 0
        r_sl, c_sl, sq = self._get_padded_slice(time_idx, max_t, h)
        read_slices.append(r_sl)
        crop_slices.append(c_sl)
        squeeze_map.append(sq)

        # -> Process Spatial Axes (1..N)
        nx_arr = self._diag._nx
        if self._dim == 1 and np.isscalar(nx_arr):
            nx_arr = [nx_arr]

        for i, idx in enumerate(spatial_indices):
            dim_axis = i + 1
            max_len = nx_arr[i]
            h = halo if dim_axis in axes_needing_halo else 0

            r_sl, c_sl, sq = self._get_padded_slice(idx, max_len, h)
            read_slices.append(r_sl)
            crop_slices.append(c_sl)
            squeeze_map.append(sq)

        data_chunk = self._diag[tuple(read_slices)]

        axis_map = {'t': 0}
        for d in range(self._dim):
            axis_map[f'x{d + 1}'] = d + 1

        derivative_chunk = self._compute_derivative(data_chunk, self._deriv_type, axis_map)

        result = derivative_chunk[tuple(crop_slices)]

        for axis_idx in range(len(squeeze_map) - 1, -1, -1):
            if squeeze_map[axis_idx]:
                result = result.squeeze(axis=axis_idx)

        return result

    def _data_generator(self, index: int, data_slice: tuple | None = None) -> Generator[np.ndarray, None, None]:
        """
        Generate data for a specific index on-demand.
        Standard method used by Diagnostic.__getitem__ and others.
        """
        if data_slice is None:
            data_slice = ()
        yield self._read_and_compute(index, data_slice)

    def __getitem__(self, index: int | slice | tuple) -> np.ndarray:
        """
        Get data at a specific index, loading ONLY the necessary partial data (with halo)
        to compute the derivative locally.
        """
        if self._all_loaded and self._data is not None:
            return self._data[index]

        if not isinstance(index, tuple):
            index = (index,)

        time_idx = index[0]
        spatial_indices = index[1:]

        return self._read_and_compute(time_idx, spatial_indices)


class Derivative_Species_Handler:
    """Class to handle derivatives for a species.
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

    def __init__(self, species_handler: Any, deriv_type: str, axis: int | tuple[int, int] | None = None, order: int = 2) -> None:
        self._species_handler = species_handler
        self._deriv_type = deriv_type
        self._axis = axis
        self._order = order
        self._derivatives_computed: dict[Any, Derivative_Diagnostic] = {}

    def __getitem__(self, key: Any) -> Derivative_Diagnostic:
        if key not in self._derivatives_computed:
            diag = self._species_handler[key]
            self._derivatives_computed[key] = Derivative_Diagnostic(diag, self._deriv_type, self._axis, self._order)
        return self._derivatives_computed[key]
