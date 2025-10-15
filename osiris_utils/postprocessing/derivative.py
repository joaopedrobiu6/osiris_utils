import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess


class Derivative_Simulation(PostProcess):
    """
    Class to compute the derivative of a diagnostic. Works as a wrapper for the Derivative_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
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
        The order of the derivative. Currently only 2 is supported.
        Order 2 uses central differences with edge_order=2 in numpy.gradient.
        Order 4 uses a higher order finite difference scheme. For the edge points,
        a lower order scheme is used to avoid going out of bounds.

    """

    def __init__(self, simulation, deriv_type, axis=None, order=2):
        super().__init__(f"Derivative({deriv_type})")
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._deriv_type = deriv_type
        self._axis = axis
        self._derivatives_computed = {}
        self._species_handler = {}
        self._order = order

    def __getitem__(self, key):
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

    def delete_all(self):
        self._derivatives_computed = {}

    def delete(self, key):
        if key in self._derivatives_computed:
            del self._derivatives_computed[key]
        else:
            print(f"Derivative {key} not found in simulation")

    def process(self, diagnostic):
        """Apply derivative to a diagnostic"""
        return Derivative_Diagnostic(diagnostic, self._deriv_type, self._axis)


class Derivative_Diagnostic(Diagnostic):
    """
    Auxiliar class to compute the derivative of a diagnostic, for it to be similar in behavior to a Diagnostic object.
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

    def __init__(self, diagnostic, deriv_type, axis=None, order=2):
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

    def load_all(self):
        """Load all data and compute the derivative"""
        if self._data is not None:
            print("Using cached derivative")
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()
            self._data = self._diag._data

        if self._diag._all_loaded is True:
            print("Using cached data from diagnostic")
            self._data = self._diag._data

        if self._order == 2:
            if self._deriv_type == "t":
                result = np.gradient(self._data, self._diag._dt * self._diag._ndump, axis=0, edge_order=2)

            elif self._deriv_type == "x1":
                if self._dim == 1:
                    result = np.gradient(self._data, self._diag._dx, axis=1, edge_order=2)
                else:
                    result = np.gradient(self._data, self._diag._dx[0], axis=1, edge_order=2)

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
                    axis=self._axis[0],
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
                dx = self._diag._dx[axis - 1] if self._dim > 1 else self._diag._dx

            # Central differences for interior points
            result = np.empty_like(self._data)
            result_slice = [slice(None)] * self._data.ndim
            for i in range(2, self._data.shape[axis] - 2):
                result_slice[axis] = i
                result[tuple(result_slice)] = (
                    -self._data[tuple(result_slice[:axis] + [i + 2] + result_slice[axis + 1 :])]
                    + 8 * self._data[tuple(result_slice[:axis] + [i + 1] + result_slice[axis + 1 :])]
                    - 8 * self._data[tuple(result_slice[:axis] + [i - 1] + result_slice[axis + 1 :])]
                    + self._data[tuple(result_slice[:axis] + [i - 2] + result_slice[axis + 1 :])]
                ) / (12 * dx)

                # Forward difference for the first two points
                for i in range(2):
                    result_slice[axis] = i
                    result[tuple(result_slice)] = (
                        -25 * self._data[tuple(result_slice)]
                        + 48 * self._data[tuple(result_slice[:axis] + [i + 1] + result_slice[axis + 1 :])]
                        - 36 * self._data[tuple(result_slice[:axis] + [i + 2] + result_slice[axis + 1 :])]
                        + 16 * self._data[tuple(result_slice[:axis] + [i + 3] + result_slice[axis + 1 :])]
                        - 3 * self._data[tuple(result_slice[:axis] + [i + 4] + result_slice[axis + 1 :])]
                    ) / (12 * dx)

                # Backward difference for the last two points
                for i in range(self._data.shape[axis] - 2, self._data.shape[axis]):
                    result_slice[axis] = i
                    result[tuple(result_slice)] = (
                        25 * self._data[tuple(result_slice)]
                        - 48 * self._data[tuple(result_slice[:axis] + [i - 1] + result_slice[axis + 1 :])]
                        + 36 * self._data[tuple(result_slice[:axis] + [i - 2] + result_slice[axis + 1 :])]
                        - 16 * self._data[tuple(result_slice[:axis] + [i - 3] + result_slice[axis + 1 :])]
                        + 3 * self._data[tuple(result_slice[:axis] + [i - 4] + result_slice[axis + 1 :])]
                    ) / (12 * dx)
            else:
                raise ValueError("Order 4 is only implemented for spatial derivatives 'x1', 'x2' and 'x3'.")

        # Store the result in the cache
        self._all_loaded = True
        self._data = result
        return self._data

    def _data_generator(self, index):
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

        if self._order == 4:
            if self._deriv_type == "x1":
                if self._dim == 1:
                    # Fourth-order central difference for 1D case
                    data = self._diag[index]
                    h = self._diag._dx
                    result = np.zeros_like(data)

                    # Interior points: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
                    result[2:-2] = (-data[4:] + 8 * data[3:-1] - 8 * data[1:-3] + data[:-4]) / (12 * h)

                    # Boundary points (fallback to second-order)
                    result[0] = (-3 * data[0] + 4 * data[1] - data[2]) / (2 * h)
                    result[1] = (data[2] - data[0]) / (2 * h)
                    result[-2] = (data[-1] - data[-3]) / (2 * h)
                    result[-1] = (3 * data[-1] - 4 * data[-2] + data[-3]) / (2 * h)

                    yield result
                else:
                    # Multi-dimensional case (same as before)
                    data = self._diag[index]
                    h = self._diag._dx[0]
                    result = np.zeros_like(data)

                    result[2:-2] = (-data[4:] + 8 * data[3:-1] - 8 * data[1:-3] + data[:-4]) / (12 * h)

                    result[0] = (-3 * data[0] + 4 * data[1] - data[2]) / (2 * h)
                    result[1] = (data[2] - data[0]) / (2 * h)
                    result[-2] = (data[-1] - data[-3]) / (2 * h)
                    result[-1] = (3 * data[-1] - 4 * data[-2] + data[-3]) / (2 * h)

                    yield result

            elif self._deriv_type == "x2":
                data = self._diag[index]
                h = self._diag._dx[1]
                result = np.zeros_like(data)

                result[:, 2:-2] = (-data[:, 4:] + 8 * data[:, 3:-1] - 8 * data[:, 1:-3] + data[:, :-4]) / (12 * h)

                # Boundaries
                result[:, 0] = (-3 * data[:, 0] + 4 * data[:, 1] - data[:, 2]) / (2 * h)
                result[:, 1] = (data[:, 2] - data[:, 0]) / (2 * h)
                result[:, -2] = (data[:, -1] - data[:, -3]) / (2 * h)
                result[:, -1] = (3 * data[:, -1] - 4 * data[:, -2] + data[:, -3]) / (2 * h)

                yield result

            elif self._deriv_type == "x3":
                data = self._diag[index]
                h = self._diag._dx[2]
                result = np.zeros_like(data)

                result[:, :, 2:-2] = (-data[:, :, 4:] + 8 * data[:, :, 3:-1] - 8 * data[:, :, 1:-3] + data[:, :, :-4]) / (12 * h)

                # Boundaries
                result[:, :, 0] = (-3 * data[:, :, 0] + 4 * data[:, :, 1] - data[:, :, 2]) / (2 * h)
                result[:, :, 1] = (data[:, :, 2] - data[:, :, 0]) / (2 * h)
                result[:, :, -2] = (data[:, :, -1] - data[:, :, -3]) / (2 * h)
                result[:, :, -1] = (3 * data[:, :, -1] - 4 * data[:, :, -2] + data[:, :, -3]) / (2 * h)

                yield result

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

    def __getitem__(self, index):
        """Get data at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]

        if isinstance(index, int):
            return next(self._data_generator(index))
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            stop = self._diag._maxiter if index.stop is None else index.stop
            return np.array([next(self._data_generator(i)) for i in range(start, stop, step)])
        else:
            raise ValueError("Invalid index type. Use int or slice.")


class Derivative_Species_Handler:
    """
    Class to handle derivatives for a species.
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

    def __init__(self, species_handler, deriv_type, axis=None, order=2):
        self._species_handler = species_handler
        self._deriv_type = deriv_type
        self._axis = axis
        self._order = order
        self._derivatives_computed = {}

    def __getitem__(self, key):
        if key not in self._derivatives_computed:
            diag = self._species_handler[key]
            self._derivatives_computed[key] = Derivative_Diagnostic(diag, self._deriv_type, self._axis, self._order)
        return self._derivatives_computed[key]
