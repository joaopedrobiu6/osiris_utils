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

    """

    def __init__(self, simulation, deriv_type, axis=None):
        super().__init__(f"Derivative({deriv_type})")
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._deriv_type = deriv_type
        self._axis = axis
        self._derivatives_computed = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = Derivative_Species_Handler(self._simulation[key], self._deriv_type, self._axis)
            return self._species_handler[key]

        if key not in self._derivatives_computed:
            self._derivatives_computed[key] = Derivative_Diagnostic(
                diagnostic=self._simulation[key],
                deriv_type=self._deriv_type,
                axis=self._axis,
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

    def __init__(self, diagnostic, deriv_type, axis=None):
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

        # Store the result in the cache
        self._all_loaded = True
        self._data = result
        return self._data

    def _data_generator(self, index):
        """Generate data for a specific index on-demand"""
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
                yield (3 * self._diag[index] - 4 * self._diag[index - 1] + self._diag[index - 2]) / (2 * self._diag._dt * self._diag._ndump)
            else:
                yield (self._diag[index + 1] - self._diag[index - 1]) / (2 * self._diag._dt * self._diag._ndump)
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

    def __init__(self, species_handler, deriv_type, axis=None):
        self._species_handler = species_handler
        self._deriv_type = deriv_type
        self._axis = axis
        self._derivatives_computed = {}

    def __getitem__(self, key):
        if key not in self._derivatives_computed:
            diag = self._species_handler[key]
            self._derivatives_computed[key] = Derivative_Diagnostic(diag, self._deriv_type, self._axis)
        return self._derivatives_computed[key]
