from __future__ import annotations

from functools import lru_cache
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
    fft_axis : int or list of int
        The axis or axes to compute the FFT.
    """

    def __init__(self, simulation: Simulation, fft_axis: int | list[int]) -> None:
        super().__init__("FFT", simulation)
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
    window : str or None, optional
        The window to apply before computing the FFT. Can be "hann" or None (default "hann").
    detrend : str or None, optional
        The detrending method to apply before computing the FFT. Can be "mean" or None (default "mean").
    normalize : str, optional
        The normalization method for the FFT. Can be "none", "ortho", or "density" (default "ortho").
    assume_periodic : bool, optional
        Whether to assume the data is periodic when choosing default windowing for spatial FFTs (default True).
        If True, no spatial window is applied by default; if False, the same window is

    Methods
    -------
    load_all() -> np.ndarray
        Load all data and compute FFT (possibly including time axis). Returns power spectrum `|FFT|^2` stored in self._data.
    _frame(index: int, data_slice: tuple | None = None) -> np.ndarray
        Per-timestep (lazy) FFT. Only allowed for spatial FFT. If fft_axis includes time (0), user must call load_all().
        Returns power spectrum `|FFT|^2`.
    omega() -> np.ndarray
        Get the angular frequency array for the FFT along the time dimension.
    k(axis: int | None = None) -> np.ndarray | dict[int, np.ndarray]
        Get the wavenumber array for the FFT along spatial dimension(s).
    """

    def __init__(
        self,
        diagnostic: Diagnostic,
        fft_axis: int | list[int],
        *,
        window: str | None = "hann",  # "hann" or None
        window_for_spatial: str | None = None,  # default None if periodic; override if desired
        detrend: str | None = "mean",  # "mean" or None
        normalize: str = "ortho",  # "none" or "ortho" or "density"
        assume_periodic: bool = False,  # affects default windowing for spatial FFT)
    ) -> None:
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

        self._assume_periodic = assume_periodic
        self._detrend = detrend
        self._normalize = normalize
        self._window_time = window

        if window_for_spatial is not None:
            self._window_spatial = window_for_spatial
        else:
            # default: no spatial window for periodic domains
            self._window_spatial = None if assume_periodic else window

        # numpy supports norm=None or "ortho"
        self._np_norm = "ortho" if self._normalize == "ortho" else None

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
        r"""
        Load all data and compute FFT (possibly including time axis).
        Returns power spectrum `|FFT|^2` stored in self._data.
        """
        if self._data is not None:
            print("Using cached data.")
            return self._data

        # Ensure base diagnostic is loaded
        if self._diag._data is None or not getattr(self._diag, "_all_loaded", False):
            self._diag.load_all()

        data = np.nan_to_num(self._diag._data, copy=True)

        axes_osiris = [self._fft_axis] if isinstance(self._fft_axis, int) else list(self._fft_axis)

        # Detrend across transform axes in the FULL array
        # (here axes_osiris match numpy axes because loaded data includes time at axis 0)
        data = self._detrend_data(data, axes_osiris)

        # Apply windows along the transform axes
        result = data
        for ax in axes_osiris:
            if ax == 0:
                w = self._get_window(result.shape[0], self._window_time)
                result = self._apply_window(result, w, axis=0)
            else:
                w = self._get_window(result.shape[ax], self._window_spatial)
                result = self._apply_window(result, w, axis=ax)

        # FFT + shift
        with tqdm.tqdm(total=1, desc="FFT calculation") as pbar:
            if len(axes_osiris) == 1:
                ax = axes_osiris[0]
                fft = np.fft.fft(result, axis=ax, norm=self._np_norm)
                pbar.update(0.5)
                fft = np.fft.fftshift(fft, axes=ax)
                pbar.update(0.5)
            else:
                fft = np.fft.fftn(result, axes=axes_osiris, norm=self._np_norm)
                pbar.update(0.5)
                fft = np.fft.fftshift(fft, axes=axes_osiris)
                pbar.update(0.5)

        # Useful for time FFT
        # self.omega_max is a property, no need to set it

        # Optional "density" scaling (includes dt/dx factors)
        if self._normalize == "density":
            fft = fft * self._spacing_product(axes_osiris)

        self._data = np.abs(fft) ** 2
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        r"""
        Per-timestep (lazy) FFT. Only allowed for spatial FFT.
        If fft_axis includes time (0), user must call load_all().
        Returns power spectrum `|FFT|^2`.
        """
        axes_osiris = [self._fft_axis] if isinstance(self._fft_axis, int) else list(self._fft_axis)

        # Time FFT cannot be computed from a single timestep
        if 0 in axes_osiris:
            raise ValueError("Cannot compute FFT along time axis for a single timestep. Use load_all() instead.")

        # Read ONLY requested spatial slice from the underlying diagnostic
        # data_slice refers to spatial dimensions only (no time dimension here)
        f = self._diag._frame(index, data_slice=data_slice)

        # Map OSIRIS axes (1,2,3) -> per-timestep numpy axes (0,1,2)
        spatial_axes_osiris = [ax for ax in axes_osiris if ax != 0]
        data_axes = [ax - 1 for ax in spatial_axes_osiris]  # 1->0, 2->1, 3->2

        # Detrend over transform axes (e.g., remove mean)
        f = self._detrend_data(f, data_axes)

        # Choose spatial window; if user passed a slice and no window, you may want to force one
        # (uncomment if you want forced windowing on cropped FFTs)
        # spatial_window_kind = self._window_spatial
        # if data_slice is not None and spatial_window_kind is None:
        #     spatial_window_kind = "hann"
        spatial_window_kind = self._window_spatial
        if data_slice is not None and spatial_window_kind is None:
            spatial_window_kind = "hann"

        # Apply window(s) along transformed axes
        result = f
        for dax in data_axes:
            w = self._get_window(result.shape[dax], spatial_window_kind)
            result = self._apply_window(result, w, dax)

        # FFT + shift
        if len(data_axes) == 1:
            fft = np.fft.fft(result, axis=data_axes[0], norm=self._np_norm)
            fft = np.fft.fftshift(fft, axes=data_axes[0])
        else:
            fft = np.fft.fftn(result, axes=data_axes, norm=self._np_norm)
            fft = np.fft.fftshift(fft, axes=data_axes)

        # Optional "density" scaling (includes dt/dx factors)
        if self._normalize == "density":
            fft = fft * self._spacing_product(spatial_axes_osiris)

        return np.abs(fft) ** 2

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_window(length: int, kind: str | None) -> np.ndarray | None:
        """
        Get the window array of given kind and length.

        Parameters
        ----------
        length : int
            Length of the window.
        kind : str or None
            Type of window ("hann", "hanning", or None).

        Returns
        -------
        np.ndarray or None
            The window array, or None if no window is applied.
        """
        if kind is None or kind == "none":
            return None
        if kind in ("hann", "hanning"):
            return np.hanning(length)
        raise ValueError(f"Unknown window: {kind}")

    def _apply_window(self, data: np.ndarray, window: np.ndarray | None, axis: int) -> np.ndarray:
        """
        Apply the given window along the specified axis of the data.

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        window : np.ndarray or None
            The window array to apply, or None for no window.
        axis : int
            The axis along which to apply the window.

        Returns
        -------
        np.ndarray
            The windowed data array.
        """
        if window is None:
            return data
        window_shape = [1] * data.ndim
        window_shape[axis] = len(window)
        return data * window.reshape(window_shape)

    def _detrend_data(self, data: np.ndarray, axes: list[int]) -> np.ndarray:
        """
        Detrend the data along the specified axes.

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        axes : list of int
            The axes along which to detrend.

        Returns
        -------
        np.ndarray
            The detrended data array.
        """
        if self._detrend is None or self._detrend == "none":
            return data
        if self._detrend == "mean":
            # remove mean over the transform axes (keeps other dims intact)
            mean = data.mean(axis=tuple(axes), keepdims=True)
            return data - mean
        raise ValueError(f"Unknown detrend: {self._detrend}")

    def _spacing_product(self, axes_osiris: list[int]) -> float:
        """
        Compute the product of spacing factors (dt, dx, dy, dz) for the given OSIRIS axes.

        Parameters
        ----------
        axes_osiris : list of int
            The OSIRIS axes along which the FFT was computed.
            (0=time, 1=x, 2=y, 3=z)

        Returns
        -------
        float
        The product of spacing factors.
        """
        # axes_osiris uses: time=0, space=1,2,3
        prod = 1.0
        for ax in axes_osiris:
            if ax == 0:
                prod *= float(self._dt * self._ndump)
            else:
                dx = self._dx if isinstance(self._dx, (int, float)) else self._dx[ax - 1]
                prod *= float(dx)
        return prod

    def omega(self) -> np.ndarray:
        """
        Get the angular frequency array for the FFT along the time dimension.
        """
        if not self._all_loaded:
            raise ValueError("Load the data first using load_all() method.")

        axes_osiris = [self._fft_axis] if isinstance(self._fft_axis, int) else list(self._fft_axis)
        if 0 not in axes_osiris:
            raise ValueError(f"FFT was not computed along time axis (0). fft_axis={self._fft_axis}. Use k() instead.")

        dt = self._dt * self._ndump
        omega = np.fft.fftfreq(self._data.shape[0], d=dt) * 2 * np.pi
        return np.fft.fftshift(omega)

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

        # Which OSIRIS axes did we transform?
        axes_osiris = [self._fft_axis] if isinstance(self._fft_axis, int) else list(self._fft_axis)
        spatial_fft_axes = {ax for ax in axes_osiris if ax != 0}  # {1,2,3}

        # If user requests a specific k-axis, it must be one we transformed
        if axis is not None and axis not in spatial_fft_axes:
            raise ValueError(f"Requested k for axis {axis}, but FFT was not computed along that axis. fft_axis={self._fft_axis}")

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
        """Nyquist wavenumber (rad/m). Upper limit of representable k."""
        return self._kmax

    @property
    def omega_max(self):
        """Nyquist angular frequency (rad/s). Upper limit of representable ω."""
        return np.pi / (self._dt * self._ndump)


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
