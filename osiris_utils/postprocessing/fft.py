from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..data.diagnostic import Diagnostic
from .postprocess import PostProcess

__all__ = ["FFT_Simulation", "FFT_Diagnostic"]


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

    def __init__(self, simulation: Any, fft_axis: int | list[int]) -> None:
        super().__init__("FFT", simulation)
        self._fft_axis = fft_axis

    def process(self, diagnostic: Diagnostic) -> FFT_Diagnostic:
        """Apply FFT to a diagnostic"""
        return FFT_Diagnostic(diagnostic, self._fft_axis)

    def _process(self, container: Any, key: str) -> FFT_Diagnostic:
        return self.process(container[key])


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
        super().__init__()
        self._derive_from(diagnostic)

        self.postprocess_name = "FFT"

        self._name = f"FFT[{diagnostic._name}, {fft_axis}]"
        self._fft_axis = fft_axis
        self._axes = [fft_axis] if isinstance(fft_axis, int) else list(fft_axis)  # OSIRIS axes: 0 = time, 1..3 = space

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

    def load_all(self, n_workers: int | None = None, use_parallel: bool | None = None) -> np.ndarray:
        r"""Power spectrum ``|FFT|^2`` of every frame, stored in ``self._data``.

        A purely spatial transform is taken frame by frame, like any other
        diagnostic; one along time (axis 0) needs the whole series at once.
        """
        if 0 not in self._axes:
            return super().load_all(n_workers=n_workers, use_parallel=use_parallel)
        if self._data is not None:
            return self._data
        # The whole series; here the OSIRIS axes are the numpy axes (time is axis 0).
        windows = [self._window_time if ax == 0 else self._window_spatial for ax in self._axes]
        self._diag.load_all()  # (a Simulation's diagnostics return themselves from load_all, not the array)
        self._data = self._power_spectrum(self._diag.data, self._axes, self._axes, windows)
        self._all_loaded = True
        return self._data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        r"""
        Per-timestep (lazy) FFT. Only allowed for spatial FFT.
        If fft_axis includes time (0), user must call load_all().
        Returns power spectrum `|FFT|^2`.
        """
        # Time FFT cannot be computed from a single timestep
        if 0 in self._axes:
            raise ValueError("Cannot compute FFT along time axis for a single timestep. Use load_all() instead.")
        # A cropped (sliced) domain is not periodic, so it always gets a window.
        window = "hann" if data_slice is not None and self._window_spatial is None else self._window_spatial
        frame = self._diag._frame(index, data_slice=data_slice)  # only the requested slice is read
        np_axes = [ax - 1 for ax in self._axes]  # a frame has no time axis: x1 -> 0, x2 -> 1, x3 -> 2
        return self._power_spectrum(frame, np_axes, self._axes, [window] * len(np_axes))

    def _power_spectrum(self, data: np.ndarray, np_axes: list[int], osiris_axes: list[int], windows: list[str | None]) -> np.ndarray:
        """``|FFT|^2`` of *data* over *np_axes*: NaN -> 0, detrend, window, transform, shift, optional density scaling."""
        data = self._detrend_data(np.nan_to_num(data), np_axes)
        for ax, kind in zip(np_axes, windows, strict=True):
            data = self._apply_window(data, self._get_window(data.shape[ax], kind), axis=ax)
        fft = np.fft.fftshift(np.fft.fftn(data, axes=np_axes, norm=self._np_norm), axes=np_axes)
        if self._normalize == "density":  # includes the dt/dx factors
            fft = fft * self._spacing_product(osiris_axes)
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
            prod *= float(self._dt * self._ndump) if ax == 0 else float(np.atleast_1d(self._dx)[ax - 1])
        return prod

    def omega(self) -> np.ndarray:
        """Angular frequencies of a transform along time, in fftshift order."""
        if 0 not in self._axes:
            raise ValueError(f"FFT was not computed along time axis (0). fft_axis={self._fft_axis}. Use k() instead.")
        return np.fft.fftshift(np.fft.fftfreq(self._maxiter, d=self._dt * self._ndump) * 2 * np.pi)

    def k(self, axis: int | None = None) -> np.ndarray | dict[int, np.ndarray]:
        """
        Get the wavenumber array for the FFT along spatial dimension(s), in fftshift order.

        Computed from the grid (``nx``, ``dx``), so it needs no loaded data.

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
        """
        spatial = [ax for ax in self._axes if ax != 0]
        if axis is None:
            if not spatial:
                raise ValueError("No spatial FFT axes to compute wavenumber for. fft_axis is 0 (time only).")
            return {ax: self.k(ax) for ax in spatial}
        if axis not in spatial:
            raise ValueError(f"Requested k for axis {axis}, but FFT was not computed along that axis. fft_axis={self._fft_axis}")
        n = int(np.atleast_1d(self._nx)[axis - 1])
        dx = float(np.atleast_1d(self._dx)[axis - 1])
        return np.fft.fftshift(np.fft.fftfreq(n, d=dx) * 2 * np.pi)

    @property
    def kmax(self) -> float | np.ndarray:
        """Nyquist wavenumber (rad/m). Upper limit of representable k."""
        if isinstance(self._dx, (int, float)):
            return np.pi / self._dx
        return np.pi / np.array([self._dx[ax - 1] for ax in self._axes if ax != 0])

    @property
    def omega_max(self):
        """Nyquist angular frequency (rad/s). Upper limit of representable ω."""
        return np.pi / (self._dt * self._ndump)
