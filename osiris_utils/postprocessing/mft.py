from __future__ import annotations

import numpy as np

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

__all__ = [
    "MFT_Simulation",
    "MFT_Diagnostic",
    "MFT_Species_Handler",
]


class MFT_Simulation(PostProcess):
    """
    Class to compute the Mean Field Theory approximation of a diagnostic. Works as a wrapper for the MFT_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    mft_axis : int
        The axis to compute the mean field theory.

    """

    def __init__(self, simulation: Simulation, mft_axis: int | None = None):
        super().__init__(f"MeanFieldTheory({mft_axis})", simulation)
        self._mft_axis = mft_axis
        self._mft_computed = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = MFT_Species_Handler(self._simulation[key], self._mft_axis)
            return self._species_handler[key]
        if key not in self._mft_computed:
            self._mft_computed[key] = MFT_Diagnostic(self._simulation[key], self._mft_axis)
        return self._mft_computed[key]

    def delete_all(self):
        self._mft_computed = {}

    def delete(self, key):
        if key in self._mft_computed:
            del self._mft_computed[key]
        else:
            print(f"MeanFieldTheory {key} not found in simulation")


class MFT_Diagnostic(Diagnostic):
    """
    Container: gives access to "avg" and "delta" diagnostics.
    Not itself a time-indexed diagnostic.
    """

    def __init__(self, diagnostic: Diagnostic, mft_axis: int):
        if hasattr(diagnostic, "_species"):
            super().__init__(simulation_folder=getattr(diagnostic, "_simulation_folder", None), species=diagnostic._species)
        else:
            super().__init__(None)

        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified (1..dim).")
        if not (1 <= mft_axis <= diagnostic._dim):
            raise ValueError(f"mft_axis must be in 1..{diagnostic._dim}, got {mft_axis}")

        self._name = f"MFT[{diagnostic._name}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._components: dict[str, Diagnostic] = {}

        # copy metadata if needed
        for attr in ["_dt", "_dx", "_ndump", "_axis", "_nx", "_x", "_grid", "_dim", "_maxiter", "_tunits", "_type"]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def __getitem__(self, key: str) -> Diagnostic:
        if key == "avg":
            if "avg" not in self._components:
                self._components["avg"] = MFT_Diagnostic_Average(self._diag, self._mft_axis)
            return self._components["avg"]
        if key == "delta":
            if "delta" not in self._components:
                self._components["delta"] = MFT_Diagnostic_Fluctuations(self._diag, self._mft_axis)
            return self._components["delta"]
        raise ValueError("Invalid MFT component. Use 'avg' or 'delta'.")


class MFT_Diagnostic_Average(Diagnostic):
    """
    Class to compute the average component of mean field theory.
    Inherits from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute the mean field theory.

    """

    def __init__(self, diagnostic: Diagnostic, mft_axis: int):
        if hasattr(diagnostic, "_species"):
            super().__init__(simulation_folder=getattr(diagnostic, "_simulation_folder", None), species=diagnostic._species)
        else:
            super().__init__(None)

        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified (1..dim).")
        if not (1 <= mft_axis <= diagnostic._dim):
            raise ValueError(f"mft_axis must be in 1..{diagnostic._dim}, got {mft_axis}")

        self.postprocess_name = "MFT_AVG"
        self._name = f"MFT_avg[{diagnostic._name}, x{mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

        for attr in ["_dt", "_dx", "_ndump", "_axis", "_nx", "_x", "_grid", "_dim", "_maxiter", "_type"]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        self._diag.load_all()
        # loaded data includes time at axis 0; spatial x1..xd are axes 1..d
        ax = self._mft_axis  # OSIRIS axis -> numpy axis in loaded array
        self._data = self._diag.data.mean(axis=ax, keepdims=True)
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        # per-frame array has only spatial dims; x1..xd -> numpy 0..d-1
        ax = self._mft_axis - 1

        f = self._diag._frame(index, data_slice=data_slice)
        return f.mean(axis=ax, keepdims=True)


class MFT_Diagnostic_Fluctuations(Diagnostic):
    """
    delta = f - avg, where avg uses keepdims to broadcast.
    """

    def __init__(self, diagnostic: Diagnostic, mft_axis: int):
        if hasattr(diagnostic, "_species"):
            super().__init__(simulation_folder=getattr(diagnostic, "_simulation_folder", None), species=diagnostic._species)
        else:
            super().__init__(None)

        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified (1..dim).")
        if not (1 <= mft_axis <= diagnostic._dim):
            raise ValueError(f"mft_axis must be in 1..{diagnostic._dim}, got {mft_axis}")

        self.postprocess_name = "MFT_FLT"
        self._name = f"MFT_delta[{diagnostic._name}, x{mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

        for attr in ["_dt", "_dx", "_ndump", "_axis", "_nx", "_x", "_grid", "_dim", "_maxiter", "_type"]:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            return self._data

        self._diag.load_all()
        ax = self._mft_axis  # spatial axis in loaded array (time is 0)
        avg = self._diag.data.mean(axis=ax, keepdims=True)
        self._data = self._diag.data - avg
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        ax = self._mft_axis - 1
        f = self._diag._frame(index, data_slice=data_slice)
        avg = f.mean(axis=ax, keepdims=True)
        return f - avg


class MFT_Species_Handler:
    """
    Class to handle mean field theory for a species.
    Acts as a wrapper for the MFT_Diagnostic class.

    Not intended to be used directly, but through the MFT_Simulation class.

    Parameters
    ----------
    species_handler : Species_Handler
        The species handler object.
    mft_axis : int
        The axis to compute the mean field theory.
    """

    def __init__(self, species_handler, mft_axis):
        self._species_handler = species_handler
        self._mft_axis = mft_axis
        self._mft_computed = {}

    def __getitem__(self, key):
        if key not in self._mft_computed:
            diag = self._species_handler[key]
            self._mft_computed[key] = MFT_Diagnostic(diag, self._mft_axis)
        return self._mft_computed[key]
