import numpy as np

from ..data.diagnostic import OSIRIS_FLD, Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess


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
        super().__init__("FieldCentering Simulation")
        """
        Class to center the field in the simulation.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        field : str
            The field to center.
        """
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation

        self._field_centered = {}
        # no need to create a species handler for field centering since fields are not species related

    def __getitem__(self, key):
        if key not in OSIRIS_FLD:
            raise ValueError(f"Does it make sense to center {key} field? Only {OSIRIS_FLD} are supported.")
        if key not in self._field_centered:
            self._field_centered[key] = FieldCentering_Diagnostic(self._simulation[key])
        return self._field_centered[key]

    def delete_all(self):
        self._field_centered = {}

    def delete(self, key):
        if key in self._field_centered:
            del self._field_centered[key]
        else:
            print(f"Field {key} not found in simulation")

    def process(self, diagnostic):
        """Apply field centering to a diagnostic"""
        return FieldCentering_Diagnostic(diagnostic)


class FieldCentering_Diagnostic(Diagnostic):
    def __init__(self, diagnostic):
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

        self._data = None
        self._all_loaded = False

    def load_all(self):
        if self._data is not None:
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()

        if self._dim == 1:
            if self._original_name.lower() in [
                "b2",
                "part_b2",
                "ext_b2",
                "b3",
                "part_b3",
                "ext_b3",
                "e1",
                "part_e1",
                "ext_e1",
            ]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)
            elif self._original_name.lower() in [
                "b1",
                "part_b1",
                "ext_b1",
                "e2",
                "part_e2",
                "ext_e2",
                "e3",
                "part_e3",
                "ext_e3",
            ]:
                result = self._diag.data

        elif self._dim == 2:
            if self._original_name.lower() in [
                "e1",
                "part_e1",
                "ext_e1",
                "b2",
                "part_b2",
                "ext_b2",
            ]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)
            elif self._original_name.lower() in [
                "e2",
                "part_e2",
                "ext_e2",
                "b1",
                "part_b1",
                "ext_b1",
            ]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=2) + self._diag.data)
            elif self._original_name.lower() in ["b3", "part_b3", "ext_b3"]:
                result = 0.5 * (
                    np.roll(
                        (0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)),
                        shift=1,
                        axis=2,
                    )
                    + (0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data))
                )
            elif self._original_name.lower() in ["e3", "part_e3", "ext_e3"]:
                result = self._diag.data

        elif self._dim == 3:
            # TODO test this
            if self._original_name in ["b1", "part_b1", "ext_b1"]:
                result = 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag.data, shift=1, axis=2) + self._diag.data),
                        shift=1,
                        axis=3,
                    )
                    + 0.5 * (np.roll(self._diag.data, shift=1, axis=2) + self._diag.data)
                )
            elif self._original_name in ["b2", "part_b2", "ext_b2"]:
                result = 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data),
                        shift=1,
                        axis=3,
                    )
                    + 0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)
                )
            elif self._original_name in ["b3", "part_b3", "ext_b3"]:
                result = 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data),
                        shift=1,
                        axis=2,
                    )
                    + 0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)
                )
            elif self._original_name in ["e1", "part_e1", "ext_e1"]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=1) + self._diag.data)
            elif self._original_name in ["e2", "part_e2", "ext_e2"]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=2) + self._diag.data)
            elif self._original_name in ["e3", "part_e3", "ext_e3"]:
                result = 0.5 * (np.roll(self._diag.data, shift=1, axis=3) + self._diag.data)

        else:
            raise ValueError(f"Unknown dimension {self._dim}.")

        self._data = result
        self._all_loaded = True
        return self._data

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

    def _data_generator(self, index):
        if self._dim == 1:
            if self._original_name.lower() in [
                "b2",
                "part_b2",
                "ext_b2",
                "b3",
                "part_b3",
                "ext_b3",
                "e1",
                "part_e1",
                "ext_e1",
            ]:
                yield 0.5 * (np.roll(self._diag[index], shift=1) + self._diag[index])
            elif self._original_name.lower() in [
                "b1",
                "part_b1",
                "ext_b1",
                "e2",
                "part_e2",
                "ext_e2",
                "e3",
                "part_e3",
                "ext_e3",
            ]:  # it's already centered but self._data does not exist
                yield self._diag[index]
            else:
                raise ValueError(f"Unknown field {self._original_name}.")

        elif self._dim == 2:
            if self._original_name in [
                "e1",
                "part_e1",
                "ext_e1",
                "b2",
                "part_b2",
                "ext_b2",
            ]:
                yield 0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index])
            elif self._original_name in [
                "e2",
                "part_e2",
                "ext_e2",
                "b1",
                "part_b1",
                "ext_b1",
            ]:
                yield 0.5 * (np.roll(self._diag[index], shift=1, axis=1) + self._diag[index])
            elif self._original_name in ["b3", "part_b3", "ext_b3"]:
                yield 0.5 * (
                    np.roll(
                        (0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index])),
                        shift=1,
                        axis=1,
                    )
                    + (0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index]))
                )
            elif self._original_name in ["e3", "part_e3", "ext_e3"]:
                yield self._diag[index]
            else:
                raise ValueError(f"Unknown field {self._original_name}.")

        elif self._dim == 3:
            if self._original_name in ["b1", "part_b1", "ext_b1"]:
                yield 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag[index], shift=1, axis=1) + self._diag[index]),
                        shift=1,
                        axis=2,
                    )
                    + 0.5 * (np.roll(self._diag[index], shift=1, axis=1) + self._diag[index])
                )
            elif self._original_name in ["b2", "part_b2", "ext_b2"]:
                yield 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index]),
                        shift=1,
                        axis=2,
                    )
                    + 0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index])
                )
            elif self._original_name in ["b3", "part_b3", "ext_b3"]:
                yield 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index]),
                        shift=1,
                        axis=1,
                    )
                    + 0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index])
                )
            elif self._original_name in ["e1", "part_e1", "ext_e1"]:
                yield 0.5 * (np.roll(self._diag[index], shift=1, axis=0) + self._diag[index])
            elif self._original_name in ["e2", "part_e2", "ext_e2"]:
                yield 0.5 * (np.roll(self._diag[index], shift=1, axis=1) + self._diag[index])
            elif self._original_name in ["e3", "part_e3", "ext_e3"]:
                yield 0.5 * (np.roll(self._diag[index], shift=1, axis=2) + self._diag[index])

        else:
            raise ValueError(f"Unknown dimension {self._dim}.")
