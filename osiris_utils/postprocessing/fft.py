from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import tqdm as tqdm

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess

__all__ = ["FFT_Simulation", "FFT_Diagnostic", "FFT_Species_Handler"]


class FFT_Simulation(PostProcess):
    """
    Class to handle the Fast Fourier Transform on data. Works as a wrapper for the FFT_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------

    simulation : Simulation
        The simulation object.
    axis : int
        The axis to compute the FFT.

    """

    def __init__(self, simulation: Simulation, fft_axis: int | list[int]) -> None:
        super().__init__("FFT")
        if not isinstance(simulation, Simulation):
            raise ValueError("simulation must be a Simulation-compatible object.")
        self._simulation = simulation
        self._fft_axis = fft_axis
        self._fft_computed: dict[Any, FFT_Diagnostic] = {}
        self._species_handler: dict[Any, FFT_Species_Handler] = {}

    def __getitem__(self, key: Any) -> FFT_Species_Handler | FFT_Diagnostic:
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = FFT_Species_Handler(self._simulation[key], self._fft_axis)
            return self._species_handler[key]

        if key not in self._fft_computed:
            self._fft_computed[key] = FFT_Diagnostic(self._simulation[key], self._fft_axis)
        return self._fft_computed[key]

    def delete_all(self) -> None:
        self._fft_computed = {}

    def delete(self, key: Any) -> None:
        if key in self._fft_computed:
            del self._fft_computed[key]
        else:
            print(f"FFT {key} not found in simulation")

    def process(self, diagnostic: Diagnostic) -> FFT_Diagnostic:
        """Apply FFT to a diagnostic"""
        return FFT_Diagnostic(diagnostic, self._fft_axis)


class FFT_Diagnostic(Diagnostic):
    """
    Auxiliar class to compute the FFT of a diagnostic, for it to be similar in behavior to a Diagnostic object.
    Inherits directly from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic to compute the FFT.
    axis : int
        The axis to compute the FFT.

    Methods
    -------
    load_all()
        Load all the data and compute the FFT.
    omega()
        Get the angular frequency array for the FFT.
    __getitem__(index)
        Get data at a specific index.

    """

    def __init__(self, diagnostic: Diagnostic, fft_axis: int | list[int]) -> None:
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=(diagnostic._simulation_folder if hasattr(diagnostic, "_simulation_folder") else None),
                species=diagnostic._species,
            )
        else:
            super().__init__(None)

        self.postprocess_name = "FFT"

        self._name = f"FFT[{diagnostic._name}, {fft_axis}]"
        self._diag = diagnostic
        self._fft_axis = fft_axis
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

        if isinstance(self._dx, (int, float)):
            self._kmax = np.pi / (self._dx)
        else:
            # Handle if fft_axis is int
            axes = [self._fft_axis] if isinstance(self._fft_axis, int) else self._fft_axis
            self._kmax = np.pi / np.array([self._dx[ax - 1] for ax in axes if ax != 0])

    def load_all(self) -> np.ndarray:
        if self._data is not None:
            print("Using cached data.")
            return self._data

        if not hasattr(self._diag, "_data") or self._diag._data is None:
            self._diag.load_all()
            self._diag._data = np.nan_to_num(self._diag._data)

        # Apply appropriate windows based on which axes we're transforming
        if isinstance(self._fft_axis, (list, tuple)):
            if self._diag._data is None:
                raise ValueError(f"Unable to load data for diagnostic {self._diag._name}. The data is None even after loading.")

            result = self._diag._data.copy()

            for axis in self._fft_axis:
                if axis == 0:  # Time axis
                    window = np.hanning(result.shape[0]).reshape(-1, *([1] * (result.ndim - 1)))
                    result = result * window
                else:  # Spatial axis
                    window = self._get_window(result.shape[axis], axis)
                    result = self._apply_window(result, window, axis)

            with tqdm.tqdm(total=1, desc="FFT calculation") as pbar:
                data_fft = np.fft.fftn(result, axes=self._fft_axis)
                pbar.update(0.5)
                result = np.fft.fftshift(data_fft, axes=self._fft_axis)
                pbar.update(0.5)

        else:
            if self._fft_axis == 0:
                hanning_window = np.hanning(self._diag._data.shape[0]).reshape(-1, *([1] * (self._diag._data.ndim - 1)))
                data_windowed = hanning_window * self._diag._data
            else:
                window = self._get_window(self._diag._data.shape[self._fft_axis], self._fft_axis)
                data_windowed = self._apply_window(self._diag._data, window, self._fft_axis)

            with tqdm.tqdm(total=1, desc="FFT calculation") as pbar:
                data_fft = np.fft.fft(data_windowed, axis=self._fft_axis)
                pbar.update(0.5)
                result = np.fft.fftshift(data_fft, axes=self._fft_axis)
                pbar.update(0.5)

        self.omega_max = np.pi / self._dt / self._ndump

        self._all_loaded = True
        self._data = np.abs(result) ** 2
        return self._data

    def _data_generator(self, index: int) -> Generator[np.ndarray, None, None]:
        # Get the data for this index
        original_data = self._diag[index]

        if self._fft_axis == 0:
            raise ValueError("Cannot generate FFT along time axis for a single timestep. Use load_all() instead.")

        # For spatial FFT, we can apply a spatial window if desired
        if isinstance(self._fft_axis, (list, tuple)):
            result = original_data
            for axis in self._fft_axis:
                if axis != 0:  # Skip time axis
                    # Apply window along this spatial dimension
                    window = self._get_window(original_data.shape[axis - 1], axis - 1)
                    result = self._apply_window(result, window, axis - 1)

            # Compute FFT
            result_fft = np.fft.fftn(result, axes=[ax - 1 for ax in self._fft_axis if ax != 0])
            result_fft = np.fft.fftshift(result_fft, axes=[ax - 1 for ax in self._fft_axis if ax != 0])

        else:
            if self._fft_axis > 0:  # Spatial axis
                window = self._get_window(original_data.shape[self._fft_axis - 1], self._fft_axis - 1)
                windowed_data = self._apply_window(original_data, window, self._fft_axis - 1)

                result_fft = np.fft.fft(windowed_data, axis=self._fft_axis - 1)
                result_fft = np.fft.fftshift(result_fft, axes=self._fft_axis - 1)

        yield np.abs(result_fft) ** 2

    def _get_window(self, length: int, axis: int) -> np.ndarray:
        return np.hanning(length)

    def _apply_window(self, data: np.ndarray, window: np.ndarray, axis: int) -> np.ndarray:
        ndim = data.ndim
        window_shape = [1] * ndim
        window_shape[axis] = len(window)

        reshaped_window = window.reshape(window_shape)

        return data * reshaped_window

    def __getitem__(self, index: int | slice) -> np.ndarray:
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

    def omega(self) -> np.ndarray:
        """
        Get the angular frequency array for the FFT along the time dimension (axis 0).

        Returns
        -------
        np.ndarray
            Angular frequency array for the time axis.
        """
        if not self._all_loaded:
            raise ValueError("Load the data first using load_all() method.")

        # If the FFT was computed along the time axis (0) return temporal frequencies
        if isinstance(self._fft_axis, (list, tuple)):
            if 0 in self._fft_axis:
                dt = self._dt * self._ndump
                omega = np.fft.fftfreq(self._data.shape[0], d=dt) * 2 * np.pi
                return np.fft.fftshift(omega)
            # If FFT was computed along spatial axes only and a single spatial axis
            spatial_axes = [ax for ax in self._fft_axis if ax != 0]
            if len(spatial_axes) == 1:
                return self.k(spatial_axes[0])
            # Multi-dimensional spatial FFT: return concatenated or dict of k arrays
            return self.k()
        else:
            if self._fft_axis == 0:
                dt = self._dt * self._ndump
                omega = np.fft.fftfreq(self._data.shape[0], d=dt) * 2 * np.pi
                return np.fft.fftshift(omega)
            # Single spatial axis: return wavenumber array for that axis
            return self.k(self._fft_axis)

    def k(self, axis: int | None = None) -> np.ndarray | dict[int, np.ndarray]:
        """
        Get the wavenumber array for the FFT along spatial dimension(s).

        Parameters
        ----------
        axis : int or None, optional
            The spatial axis to compute wavenumber for (1, 2, or 3).
            If None, returns wavenumbers for all spatial axes in fft_axis.

        Returns
        -------
        np.ndarray or dict
            If axis is specified: wavenumber array for that axis.
            If axis is None: dictionary mapping axis -> wavenumber array.

        Notes
        -----
        When load_all() is used, time axis is 0 and spatial axes are 1,2,3.
        When accessing single timesteps, spatial axes are 0,1,2.
        """
        if self._data is None:
            raise ValueError("Load the data first using load_all() or access via indexing.")

        # Determine if we have the time dimension in the data
        # If all_loaded is True, then axis 0 is time, spatial axes are 1,2,3
        # If all_loaded is False, we're looking at a single timestep, spatial axes are 0,1,2
        has_time_axis = self._all_loaded

        # Determine which axes to compute k for
        if axis is not None:
            # Single axis specified
            if axis == 0:
                raise ValueError("axis must be a spatial dimension (1, 2, or 3), not 0 (time).")
            if axis < 1 or axis > 3:
                raise ValueError(f"axis must be 1, 2, or 3, got {axis}")

            # Get dx for this axis
            if isinstance(self._dx, (int, float)):
                dx = self._dx
            else:
                dx = self._dx[axis - 1]

            # Compute the actual data axis index
            # If we have time axis, spatial axis N is at index N
            # If no time axis (single timestep), spatial axis N is at index N-1
            data_axis = axis if has_time_axis else axis - 1

            # Compute wavenumber
            k_array = np.fft.fftfreq(self._data.shape[data_axis], d=dx) * 2 * np.pi
            k_array = np.fft.fftshift(k_array)
            return k_array
        else:
            # axis is None: return k for all spatial axes in fft_axis
            result = {}

            if isinstance(self._fft_axis, (list, tuple)):
                # Multi-axis FFT: return k for all spatial axes
                spatial_axes = [ax for ax in self._fft_axis if ax != 0]
            elif self._fft_axis == 0:
                # Only time FFT, no spatial axes
                raise ValueError("No spatial FFT axes to compute wavenumber for. fft_axis is 0 (time only).")
            else:
                # Single spatial axis
                spatial_axes = [self._fft_axis]

            for ax in spatial_axes:
                if isinstance(self._dx, (int, float)):
                    dx = self._dx
                else:
                    dx = self._dx[ax - 1]

                # Compute the actual data axis index
                data_axis = ax if has_time_axis else ax - 1

                k_array = np.fft.fftfreq(self._data.shape[data_axis], d=dx) * 2 * np.pi
                k_array = np.fft.fftshift(k_array)
                result[ax] = k_array

            return result

    @property
    def kmax(self) -> float | np.ndarray:
        return self._kmax


class FFT_Species_Handler:
    def __init__(self, species_handler: Any, fft_axis: int | list[int]) -> None:
        self._species_handler = species_handler
        self._fft_axis = fft_axis
        self._fft_computed: dict[Any, FFT_Diagnostic] = {}

    def __getitem__(self, key: Any) -> FFT_Diagnostic:
        if key not in self._fft_computed:
            diag = self._species_handler[key]
            self._fft_computed[key] = FFT_Diagnostic(diag, self._fft_axis)
        return self._fft_computed[key]
