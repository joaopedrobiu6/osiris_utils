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

    def _data_generator(self, index: int) -> Generator[np.ndarray, None, None]:
        """Generate data for a specific index on-demand"""
        if self._order == 2:
            if self._deriv_type == "x1":
                if self._dim == 1:
                    yield np.gradient(self._diag[index], self._diag._dx, axis=0, edge_order=2)
                else:
                    yield np.gradient(self._diag[index], self._diag._dx[0], axis=0, edge_order=2)

            elif self._deriv_type == "x2":
                yield np.gradient(self._diag[index], self._diag._dx[1], axis=1, edge_order=2)

            elif self._deriv_type == "x3":
                yield np.gradient(self._diag[index], self._diag._dx[2], axis=2, edge_order=2)

            elif self._deriv_type == "t":
                if index == 0:
                    yield (-3 * self._diag[index] + 4 * self._diag[index + 1] - self._diag[index + 2]) / (
                        2 * self._diag._dt * self._diag._ndump
                    )
                elif index == self._diag._maxiter - 1:
                    yield (3 * self._diag[index] - 4 * self._diag[index - 1] + self._diag[index - 2]) / (
                        2 * self._diag._dt * self._diag._ndump
                    )
                else:
                    yield (self._diag[index + 1] - self._diag[index - 1]) / (2 * self._diag._dt * self._diag._ndump)
            else:
                raise ValueError("Invalid derivative type. Use 'x1', 'x2', 'x3' or 't'.")

        elif self._order == 4:
            if self._deriv_type in ["x1", "x2", "x3"]:
                # Use vectorized helper function
                data = self._diag[index]
                axis_map = {"x1": 0, "x2": 1, "x3": 2}
                axis = axis_map[self._deriv_type]

                if self._deriv_type == "x1":
                    dx = self._diag._dx if self._dim == 1 else self._diag._dx[0]
                elif self._deriv_type == "x2":
                    dx = self._diag._dx[1]
                else:  # x3
                    dx = self._diag._dx[2]

                yield self._compute_fourth_order_spatial(data, dx, axis)

            elif self._deriv_type == "t":
                idx = index
                # Fourth-order time derivative
                if idx < 2:
                    # Forward difference for first two points
                    yield (-3 * self._diag[idx] + 4 * self._diag[idx + 1] - self._diag[idx + 2]) / (2 * self._diag._dt * self._diag._ndump)
                elif idx >= self._diag._maxiter - 2:
                    # Backward difference for last two points
                    yield (3 * self._diag[idx] - 4 * self._diag[idx - 1] + self._diag[idx - 2]) / (2 * self._diag._dt * self._diag._ndump)
                else:
                    # Fourth-order central: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
                    yield (-self._diag[idx + 2] + 8 * self._diag[idx + 1] - 8 * self._diag[idx - 1] + self._diag[idx - 2]) / (
                        12 * self._diag._dt * self._diag._ndump
                    )
            else:
                raise ValueError("Invalid derivative type. Use 'x1', 'x2', 'x3' or 't'.")

    def __getitem__(self, index: int | slice) -> np.ndarray:
        """Get data at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]

        if isinstance(index, int):
            return next(self._data_generator(index))
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            stop = self._diag._maxiter if index.stop is None else index.stop

            # Pre-allocate array for better performance
            indices = range(start, stop, step)
            if len(indices) > 0:
                first_result = next(self._data_generator(indices[0]))
                result = np.empty((len(indices),) + first_result.shape, dtype=first_result.dtype)
                result[0] = first_result
                for i, idx in enumerate(indices[1:], start=1):
                    result[i] = next(self._data_generator(idx))
                return result
            return np.array([])
        raise ValueError("Invalid index type. Use int or slice.")


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
