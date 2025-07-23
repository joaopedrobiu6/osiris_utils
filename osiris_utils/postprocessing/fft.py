import numpy as np
import tqdm as tqdm

from ..data.diagnostic import Diagnostic
from ..data.simulation import Simulation
from .postprocess import PostProcess


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

    def __init__(self, simulation, fft_axis):
        super().__init__("FFT")
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._fft_axis = fft_axis
        self._fft_computed = {}
        self._species_handler = {}

    def __getitem__(self, key):
        if key in self._simulation._species:
            if key not in self._species_handler:
                self._species_handler[key] = FFT_Species_Handler(self._simulation[key], self._fft_axis)
            return self._species_handler[key]

        if key not in self._fft_computed:
            self._fft_computed[key] = FFT_Diagnostic(self._simulation[key], self._fft_axis)
        return self._fft_computed[key]

    def delete_all(self):
        self._fft_computed = {}

    def delete(self, key):
        if key in self._fft_computed:
            del self._fft_computed[key]
        else:
            print(f"FFT {key} not found in simulation")

    def process(self, diagnostic):
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

    def __init__(self, diagnostic, fft_axis):
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
            self._kmax = np.pi / np.array([self._dx[ax - 1] for ax in self._fft_axis if ax != 0])

    def load_all(self):
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

    def _data_generator(self, index):
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

    def _get_window(self, length, axis):
        return np.hanning(length)

    def _apply_window(self, data, window, axis):
        ndim = data.ndim
        window_shape = [1] * ndim
        window_shape[axis] = len(window)

        reshaped_window = window.reshape(window_shape)

        return data * reshaped_window

    def __getitem__(self, index):
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

    def omega(self):
        """
        Get the angular frequency array for the FFT.
        """
        if not self._all_loaded:
            raise ValueError("Load the data first using load_all() method.")

        omega = np.fft.fftfreq(self._data.shape[self._fft_axis], d=self._dx[self._fft_axis - 1])
        omega = np.fft.fftshift(omega)
        return omega

    @property
    def kmax(self):
        return self._kmax


class FFT_Species_Handler:
    def __init__(self, species_handler, fft_axis):
        self._species_handler = species_handler
        self._fft_axis = fft_axis
        self._fft_computed = {}

    def __getitem__(self, key):
        if key not in self._fft_computed:
            diag = self._species_handler[key]
            self._fft_computed[key] = FFT_Diagnostic(diag, self._fft_axis)
        return self._fft_computed[key]
