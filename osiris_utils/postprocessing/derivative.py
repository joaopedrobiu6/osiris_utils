from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from ..data.diagnostic import Diagnostic
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
        simulation: Any,  # Simulation-like (Simulation or PostProcess wrapper)
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 2,
    ):
        super().__init__(f"Derivative({deriv_type})", simulation)

        # Capability checks (supports chaining)
        if not hasattr(simulation, "__getitem__"):
            raise TypeError("simulation must be Simulation-like: supports __getitem__.")
        if not (hasattr(simulation, "_species") or hasattr(simulation, "species")):
            raise TypeError("simulation must be Simulation-like: has _species or species.")

        self._deriv_type = deriv_type
        self._op_axis = axis
        self._order = order

        self._derivatives_computed: dict[Any, Derivative_Diagnostic] = {}
        self._species_handler: dict[Any, Derivative_Species_Handler] = {}

        # species already set by PostProcess, but keep it explicit if you want:
        self._species = getattr(simulation, "_species", getattr(simulation, "species", []))

    def __getitem__(self, key: Any):
        if key in self._species:
            if key not in self._species_handler:
                self._species_handler[key] = Derivative_Species_Handler(
                    self._simulation[key],  # species handler from wrapped sim
                    self._deriv_type,
                    self._op_axis,
                    self._order,
                )
            return self._species_handler[key]

        if key not in self._derivatives_computed:
            self._derivatives_computed[key] = Derivative_Diagnostic(
                diagnostic=self._simulation[key],
                deriv_type=self._deriv_type,
                axis=self._op_axis,
                order=self._order,
            )
        return self._derivatives_computed[key]

    def delete_all(self) -> None:
        self._derivatives_computed = {}
        self._species_handler = {}

    def delete(self, key: Any) -> None:
        if key in self._derivatives_computed:
            del self._derivatives_computed[key]
        elif key in self._species_handler:
            del self._species_handler[key]
        else:
            print(f"Derivative {key} not found")

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
                simulation_folder=(getattr(diagnostic, "_simulation_folder", None)),
                species=getattr(diagnostic, "_species", None),
            )
        else:
            super().__init__(None)

        self.postprocess_name = "DERIV"

        # self._name = f"D[{diagnostic._name}, {type}]"
        self._diag = diagnostic
        self._deriv_type = deriv_type
        self._op_axis = axis
        self._data = None
        self._all_loaded = False
        self._order = order

        self._cache = OrderedDict()
        self._cache_max = 6  # Maximum number of items to keep in cache (enough for 4th-order time stencil)

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

        result[tuple(slices_center)] = (
            -data[tuple(slices_p2)] + 8 * data[tuple(slices_p1)] - 8 * data[tuple(slices_m1)] + data[tuple(slices_m2)]
        ) / (12 * dx)

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
                if len(self._op_axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._op_axis[0] - 1],
                        axis=self._op_axis[0],
                        edge_order=2,
                    ),
                    self._diag._dx[self._op_axis[1] - 1],
                    axis=self._op_axis[1],
                    edge_order=2,
                )

            elif self._deriv_type == "xx":
                if not isinstance(self._op_axis, (tuple, list)) or len(self._op_axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._op_axis[0] - 1],
                        axis=self._op_axis[0],
                        edge_order=2,
                    ),
                    self._diag._dx[self._op_axis[1] - 1],
                    axis=self._op_axis[1],
                    edge_order=2,
                )

            elif self._deriv_type == "tx":
                if not isinstance(self._op_axis, int):
                    raise ValueError("Axis must be an integer.")
                result = np.gradient(
                    np.gradient(
                        self._data,
                        self._diag._dx[self._op_axis - 1],
                        axis=self._op_axis,
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

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        # dimension guards
        if self._deriv_type in ("x2", "xt", "tx") and self._diag._dim < 2:
            raise ValueError(f"{self._deriv_type} requested but diagnostic dim={self._diag._dim}")
        if self._deriv_type in ("x3",) and self._diag._dim < 3:
            raise ValueError(f"{self._deriv_type} requested but diagnostic dim={self._diag._dim}")
        if self._deriv_type == "xx":
            if not isinstance(self._op_axis, (tuple, list)) or len(self._op_axis) != 2:
                raise ValueError("For 'xx', axis must be a tuple/list of two spatial axes (1..3).")
            if max(self._op_axis) > self._diag._dim:
                raise ValueError(f"xx requested for axes {self._op_axis} but dim={self._diag._dim}")

        n = self._diag._maxiter
        dt = float(self._diag._dt * self._diag._ndump)

        # ---------- helpers ----------
        def spatial_axis_np_from_osiris(ax_osiris: int) -> int:
            # OSIRIS spatial axes are 1..3; per-timestep array axes are 0..2
            if not isinstance(ax_osiris, int):
                raise ValueError("Axis must be int for spatial derivatives.")
            if ax_osiris < 1 or ax_osiris > 3:
                raise ValueError(f"Spatial axis must be 1..3, got {ax_osiris}")
            return ax_osiris - 1

        def dx_for_np_axis(ax_np: int) -> float:
            dx = self._diag._dx
            if self._diag._dim == 1:
                return float(dx[0] if isinstance(dx, (list, tuple, np.ndarray)) else dx)
            return float(dx[ax_np])

        def d_dx(f: np.ndarray, ax_np: int, order: int) -> np.ndarray:
            dx = dx_for_np_axis(ax_np)
            if order == 2:
                return np.gradient(f, dx, axis=ax_np, edge_order=2)
            if order == 4:
                return self._compute_fourth_order_spatial(f, dx, ax_np)
            raise ValueError("Only order 2 and 4 supported.")

        def d_dt_at(i: int, order: int) -> np.ndarray:
            # Uses upstream frames, all with identical data_slice
            if order == 2:
                if i == 0:
                    f0 = self._base(0, data_slice)
                    f1 = self._base(1, data_slice)
                    f2 = self._base(2, data_slice)
                    return (-3 * f0 + 4 * f1 - f2) / (2 * dt)
                if i == n - 1:
                    f0 = self._base(n - 1, data_slice)
                    f1 = self._base(n - 2, data_slice)
                    f2 = self._base(n - 3, data_slice)
                    return (3 * f0 - 4 * f1 + f2) / (2 * dt)
                fp = self._base(i + 1, data_slice)
                fm = self._base(i - 1, data_slice)
                return (fp - fm) / (2 * dt)

            if order == 4:
                # fall back near edges
                if i < 2 or i > n - 3:
                    return d_dt_at(i, order=2)
                f_p2 = self._base(i + 2, data_slice)
                f_p1 = self._base(i + 1, data_slice)
                f_m1 = self._base(i - 1, data_slice)
                f_m2 = self._base(i - 2, data_slice)
                return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dt)

            raise ValueError("Only order 2 and 4 supported.")

        # ---------- dispatch ----------
        if self._deriv_type in ("x1", "x2", "x3"):
            f = self._base(index, data_slice)
            ax_np = {"x1": 0, "x2": 1, "x3": 2}[self._deriv_type]
            return d_dx(f, ax_np, self._order)

        if self._deriv_type == "t":
            return d_dt_at(index, self._order)

        if self._deriv_type == "xx":
            # second derivative along two spatial axes (can be same or different)
            if not isinstance(self._op_axis, (tuple, list)) or len(self._op_axis) != 2:
                raise ValueError("For 'xx', axis must be a tuple/list of two OSIRIS spatial axes, e.g. (1,2).")
            ax1_np = spatial_axis_np_from_osiris(self._op_axis[0])
            ax2_np = spatial_axis_np_from_osiris(self._op_axis[1])

            f = self._base(index, data_slice)
            # order=4 here is not implemented as a true 4th-order second derivative; keep order=2 behavior
            g = d_dx(f, ax1_np, order=2)
            return d_dx(g, ax2_np, order=2)

        if self._deriv_type == "xt":
            # d/dx ( d/dt f )
            if not isinstance(self._op_axis, int):
                raise ValueError("For 'xt', axis must be an OSIRIS spatial axis int (1..3).")
            ax_np = spatial_axis_np_from_osiris(self._op_axis)

            ft = d_dt_at(index, order=2 if self._order == 4 else self._order)  # keep stable
            return d_dx(ft, ax_np, order=2)

        if self._deriv_type == "tx":
            # d/dt ( d/dx f )
            if not isinstance(self._op_axis, int):
                raise ValueError("For 'tx', axis must be an OSIRIS spatial axis int (1..3).")
            ax_np = spatial_axis_np_from_osiris(self._op_axis)

            # do spatial derivative on each needed time frame, then time-stencil those
            def spatial_frame(i: int) -> np.ndarray:
                fi = self._base(i, data_slice)
                return d_dx(fi, ax_np, order=2)

            if self._order == 2:
                if index == 0:
                    return (-3 * spatial_frame(0) + 4 * spatial_frame(1) - spatial_frame(2)) / (2 * dt)
                if index == n - 1:
                    return (3 * spatial_frame(n - 1) - 4 * spatial_frame(n - 2) + spatial_frame(n - 3)) / (2 * dt)
                return (spatial_frame(index + 1) - spatial_frame(index - 1)) / (2 * dt)

            if self._order == 4:
                if index < 2 or index > n - 3:
                    # edge fallback
                    return (spatial_frame(min(index + 1, n - 1)) - spatial_frame(max(index - 1, 0))) / (2 * dt)
                return (
                    -spatial_frame(index + 2) + 8 * spatial_frame(index + 1) - 8 * spatial_frame(index - 1) + spatial_frame(index - 2)
                ) / (12 * dt)

            raise ValueError("Only order 2 and 4 supported.")

        raise ValueError("Invalid derivative type.")

    def _cache_get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, val):
        self._cache[key] = val
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _base(self, idx: int, data_slice: tuple | None):
        # base diagnostic frame WITH slice (this is the key!)
        key = (idx, data_slice)
        v = self._cache_get(key)
        if v is None:
            v = self._diag._frame(idx, data_slice=data_slice)
            self._cache_put(key, v)
        return v


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
