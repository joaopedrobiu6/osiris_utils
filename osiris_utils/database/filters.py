"""Spatial filters applied to 2-D frames before the transverse average.

Every filter implements the :class:`SpatialFilter` interface:

``smooth(f, periodic)``
    Smooth *f* along the filter's configured axes.  Called once per raw
    2-D frame by the database creators, **before** any physics
    (e_vlasov, mean fields, derivatives) is computed.

``derivative(f, dx, axis, order, periodic)``
    Differentiate *f* along *axis* using the filter's native scheme.
    Savitzky-Golay and Gaussian filters compute an *order*-th derivative
    in a **single pass** (``savgol_filter(deriv=order)`` /
    ``gaussian_filter1d(order=order)``), so every derivative order
    carries exactly one smoothing pass along the derivative axis — the
    effective filter scale is the same for all orders.  Savitzky-Golay
    therefore requires ``polyorder >= order`` (the derivative of the
    local fit must exist at that order).  :class:`NoFilter` falls back
    to the 4th-order finite differences in :func:`fd_derivative`,
    applied *order* times (repeated first derivatives — the historical
    scheme; no smoothing kernel is involved so chaining is exact).

    ``derivative`` does not assume *f* is raw: in the database pipeline
    fields are smoothed once and derivatives are then taken of the
    smoothed fields, each adding the single derivative-kernel pass.

Boundary conventions follow the shock-simulation setup: the
longitudinal axis (x1) is non-periodic, the transverse axis (x2) is
periodic.  The caller passes this per axis via the ``periodic``
arguments.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "FilterChain",
    "GaussianFilter",
    "NoFilter",
    "SavitzkyGolayFilter",
    "SpatialFilter",
    "as_filter",
    "fd_derivative",
]


# ----------------------------------------------------------------------
# Finite-difference fallback (moved from lorentz_database._grad4)
# ----------------------------------------------------------------------


def fd_derivative(f: np.ndarray, dx: float, axis: int = 0, order: int = 1, periodic: bool = False) -> np.ndarray:
    r"""4th-order finite-difference derivative of *f* along *axis*.

    Higher *order* derivatives are computed by repeated application of the
    first-derivative scheme — this matches how the database creators
    historically chained first-derivative diagnostics (e.g. the 4th-order
    x1-derivatives of the vnT tensor).

    Interior points use the standard 5-point centered stencil:

    .. math::
        f'_i = \frac{f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}}{12\,\Delta x}

    Boundary treatment
    ------------------
    periodic=True (transverse / x2 direction)
        Wrap-around indexing via ``np.roll``; all points use the centered stencil.
    periodic=False (longitudinal / x1 direction)
        4th-order one-sided stencils at the four edge points:

        .. math::
            f'_0     &= \frac{-25f_0 + 48f_1 - 36f_2 + 16f_3 - 3f_4}
                              {12\,\Delta x}  \\[4pt]
            f'_1     &= \frac{-3f_0 - 10f_1 + 18f_2 - 6f_3 + f_4}
                              {12\,\Delta x}  \\[4pt]
            f'_{N-2} &= \frac{-f_{N-5} + 6f_{N-4} - 18f_{N-3} + 10f_{N-2} + 3f_{N-1}}
                              {12\,\Delta x}  \\[4pt]
            f'_{N-1} &= \frac{3f_{N-5} - 16f_{N-4} + 36f_{N-3} - 48f_{N-2} + 25f_{N-1}}
                              {12\,\Delta x}

    Requires at least 5 points along *axis*.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}.")
    out = np.asarray(f, dtype=np.float64)
    for _ in range(order):
        out = _grad4(out, dx, axis=axis, periodic=periodic)
    return out


def _grad4(f: np.ndarray, dx: float, axis: int, periodic: bool = False, fourier: bool = False) -> np.ndarray:
    """Single application of the 4th-order first-derivative stencil."""
    if periodic:
        if fourier:
            # Fourier derivative: FFT, multiply by ik, IFFT
            f_hat = np.fft.fft(f, axis=axis)
            k = np.fft.fftfreq(f.shape[axis], d=dx) * 2.0 * np.pi
            k = np.moveaxis(k, 0, axis)
            return np.fft.ifft(1j * k * f_hat, axis=axis).real
        else:
            return (
                np.roll(f, 2, axis=axis) - 8.0 * np.roll(f, 1, axis=axis) + 8.0 * np.roll(f, -1, axis=axis) - np.roll(f, -2, axis=axis)
            ) / (12.0 * dx)

    # Move the target axis to front for clean 1-D slicing
    fm = np.moveaxis(f, axis, 0)
    out = np.empty_like(fm)

    # Interior: centered stencil
    out[2:-2] = (fm[:-4] - 8.0 * fm[1:-3] + 8.0 * fm[3:-1] - fm[4:]) / (12.0 * dx)

    # Left boundary: forward-biased
    out[0] = (-25.0 * fm[0] + 48.0 * fm[1] - 36.0 * fm[2] + 16.0 * fm[3] - 3.0 * fm[4]) / (12.0 * dx)
    out[1] = (-3.0 * fm[0] - 10.0 * fm[1] + 18.0 * fm[2] - 6.0 * fm[3] + fm[4]) / (12.0 * dx)

    # Right boundary: backward-biased
    out[-2] = (-fm[-5] + 6.0 * fm[-4] - 18.0 * fm[-3] + 10.0 * fm[-2] + 3.0 * fm[-1]) / (12.0 * dx)
    out[-1] = (3.0 * fm[-5] - 16.0 * fm[-4] + 36.0 * fm[-3] - 48.0 * fm[-2] + 25.0 * fm[-1]) / (12.0 * dx)

    return np.moveaxis(out, 0, axis)


# ----------------------------------------------------------------------
# Filter interface
# ----------------------------------------------------------------------


def _normalize_periodic(ndim: int, periodic: bool | Sequence[bool] | None) -> tuple[bool, ...]:
    """Expand *periodic* into a per-axis tuple of length *ndim*."""
    if periodic is None:
        return (False,) * ndim
    if isinstance(periodic, bool):
        return (periodic,) * ndim
    p = tuple(bool(x) for x in periodic)
    if len(p) != ndim:
        raise ValueError(f"periodic has {len(p)} entries but the array has {ndim} axes.")
    return p


class SpatialFilter(ABC):
    """Base class for spatial filters. See module docstring for semantics."""

    @abstractmethod
    def smooth(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:
        """Return *f* smoothed along the filter's configured axes.

        Parameters
        ----------
        f :
            Input array (any dimensionality).
        periodic :
            Per-axis boundary flags (bool, sequence of bools matching
            ``f.ndim``, or None for all non-periodic).
        """

    @abstractmethod
    def _first_derivative(self, f: np.ndarray, dx: float, axis: int, periodic: bool) -> np.ndarray:
        """Return the first derivative of *f* along *axis* using the filter's native scheme."""

    def _nth_derivative(self, f: np.ndarray, dx: float, axis: int, order: int, periodic: bool) -> np.ndarray:
        """Return the *order*-th derivative of *f* along *axis*.

        Default: apply the first-derivative operator *order* times.  Exact
        for :class:`NoFilter` (finite differences have no kernel), but
        smoothing filters MUST override this with a single-pass scheme —
        chaining would apply the kernel *order* times, making the effective
        filter scale depend on the derivative order.
        """
        out = f
        for _ in range(order):
            out = self._first_derivative(out, dx, axis=axis, periodic=periodic)
        return out

    def derivative(self, f: np.ndarray, dx: float, axis: int = 0, order: int = 1, periodic: bool = False) -> np.ndarray:
        """Return the *order*-th derivative of *f* along *axis* (spacing *dx*).

        Computed in a single pass by the smoothing filters (one kernel
        application regardless of *order*); :class:`NoFilter` applies its
        finite-difference stencil *order* times.
        """
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}.")
        return self._nth_derivative(np.asarray(f, dtype=np.float64), dx, axis=axis, order=order, periodic=periodic)

    def __call__(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:
        return self.smooth(f, periodic=periodic)

    def _axes_for(self, f: np.ndarray) -> tuple[int, ...]:
        axes = getattr(self, "_axes", None)
        if axes is None:
            return tuple(range(f.ndim))
        for ax in axes:
            if not (-f.ndim <= ax < f.ndim):
                raise ValueError(f"Filter axis {ax} out of range for array with {f.ndim} axes.")
        return tuple(ax % f.ndim for ax in axes)


class NoFilter(SpatialFilter):
    """Identity filter: no smoothing, 4th-order finite-difference derivatives.

    This is the default for all database creators and reproduces the
    historical (unfiltered) tensors exactly.
    """

    def smooth(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(f)

    def _first_derivative(self, f: np.ndarray, dx: float, axis: int, periodic: bool) -> np.ndarray:
        return _grad4(f, dx, axis=axis, periodic=periodic)

    def __repr__(self) -> str:
        return "NoFilter()"


class SavitzkyGolayFilter(SpatialFilter):
    """Savitzky-Golay smoothing with polynomial-fit derivatives.

    Parameters
    ----------
    window_length :
        Length of the fit window in grid points (odd, > polyorder).
    polyorder :
        Order of the local polynomial fit (>= 1 to compute derivatives).
    axes :
        Axes smoothed by :meth:`smooth`.  None (default) smooths every
        axis of the input array.

    Notes
    -----
    Derivatives use ``scipy.signal.savgol_filter(deriv=order, delta=dx)``
    in a single pass: the *order*-th derivative of the local polynomial
    fit, not finite differences.  Every derivative order therefore
    carries exactly one smoothing pass along the derivative axis, and
    ``polyorder >= order`` is required (the local fit must have a
    non-trivial derivative at that order).
    Non-periodic boundaries use ``mode="interp"`` (polynomial fit on the
    edge window); periodic ones use ``mode="wrap"``.
    """

    def __init__(self, window_length: int, polyorder: int, axes: Sequence[int] | None = None):
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError(f"window_length must be odd and >= 3, got {window_length}.")
        if polyorder >= window_length:
            raise ValueError(f"polyorder ({polyorder}) must be < window_length ({window_length}).")
        self._window_length = int(window_length)
        self._polyorder = int(polyorder)
        self._axes = None if axes is None else tuple(int(a) for a in axes)

    def smooth(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:
        out = np.asarray(f, dtype=np.float64)
        per = _normalize_periodic(out.ndim, periodic)
        for ax in self._axes_for(out):
            mode = "wrap" if per[ax] else "interp"
            out = savgol_filter(out, self._window_length, self._polyorder, axis=ax, mode=mode)
        return out

    def _first_derivative(self, f: np.ndarray, dx: float, axis: int, periodic: bool) -> np.ndarray:
        return self._nth_derivative(f, dx, axis=axis, order=1, periodic=periodic)

    def _nth_derivative(self, f: np.ndarray, dx: float, axis: int, order: int, periodic: bool) -> np.ndarray:
        if self._polyorder < order:
            raise ValueError(
                f"Savitzky-Golay derivative of order {order} needs polyorder >= {order} "
                f"(got polyorder={self._polyorder}): the derivative is taken in a single "
                "pass from the local polynomial fit, so the fit must support that order."
            )
        mode = "wrap" if periodic else "interp"
        return savgol_filter(
            f,
            self._window_length,
            self._polyorder,
            deriv=order,
            delta=dx,
            axis=axis,
            mode=mode,
        )

    def __repr__(self) -> str:
        return f"SavitzkyGolayFilter(window_length={self._window_length}, polyorder={self._polyorder}, axes={self._axes})"


class GaussianFilter(SpatialFilter):
    """Gaussian smoothing with derivative-of-Gaussian derivatives.

    Parameters
    ----------
    sigma :
        Kernel standard deviation **in grid points** (scipy convention).
    axes :
        Axes smoothed by :meth:`smooth`.  None (default) smooths every
        axis of the input array.
    truncate :
        Kernel radius in units of sigma.

    Notes
    -----
    Derivatives use ``scipy.ndimage.gaussian_filter1d(order=order)`` in a
    single pass (convolution with the analytic *order*-th
    derivative-of-Gaussian kernel) scaled by ``1/dx**order`` to convert
    from per-pixel to physical units.  Every derivative order therefore
    carries exactly one smoothing pass of width ``sigma`` along the
    derivative axis.  For ``order >= 2`` the kernel truncation radius is
    internally widened to at least 8 sigma: high-order derivative kernels
    have a low-k response ~``k^order``, which truncation tails at the
    default ``truncate=4`` would otherwise swamp.  Non-periodic
    boundaries use ``mode="nearest"``; periodic ones use ``mode="wrap"``.
    """

    def __init__(self, sigma: float, axes: Sequence[int] | None = None, truncate: float = 4.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        self._sigma = float(sigma)
        self._axes = None if axes is None else tuple(int(a) for a in axes)
        self._truncate = float(truncate)

    def smooth(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:
        out = np.asarray(f, dtype=np.float64)
        per = _normalize_periodic(out.ndim, periodic)
        for ax in self._axes_for(out):
            mode = "wrap" if per[ax] else "nearest"
            out = gaussian_filter1d(out, self._sigma, axis=ax, mode=mode, truncate=self._truncate)
        return out

    def _first_derivative(self, f: np.ndarray, dx: float, axis: int, periodic: bool) -> np.ndarray:
        return self._nth_derivative(f, dx, axis=axis, order=1, periodic=periodic)

    def _nth_derivative(self, f: np.ndarray, dx: float, axis: int, order: int, periodic: bool) -> np.ndarray:
        mode = "wrap" if periodic else "nearest"
        # Kernels of order >= 2 need wider support than the smoothing kernel:
        # the low-k response of an order-m derivative kernel scales as k^m, so
        # the exp(-truncate^2/2) truncation tails that are negligible for
        # smoothing and d1 dominate high-order derivatives of smooth fields at
        # the default truncate=4 (d4 errors of orders of magnitude).  truncate=8
        # puts the tails at ~1e-14, giving ~1e-9 relative accuracy for d3/d4.
        # Order 1 keeps the configured truncate — identical to the historical
        # first-derivative operator.
        truncate = self._truncate if order == 1 else max(self._truncate, 8.0)
        d = gaussian_filter1d(
            f,
            self._sigma,
            axis=axis,
            order=order,
            mode=mode,
            truncate=truncate,
        )
        return d / dx**order

    def __repr__(self) -> str:
        return f"GaussianFilter(sigma={self._sigma}, axes={self._axes}, truncate={self._truncate})"


class FilterChain(SpatialFilter):
    """Apply several filters in sequence.

    ``smooth`` runs every filter in order.  ``derivative`` uses the **last**
    filter's single-pass derivative scheme: in the database pipeline
    ``derivative`` is always called on data already smoothed by the full
    chain, so re-applying the earlier filters inside it would smooth twice.
    Order the chain so the filter whose derivative kernel you want comes
    last.
    """

    def __init__(self, *filters: SpatialFilter):
        flat: list[SpatialFilter] = []
        for f in filters:
            if isinstance(f, FilterChain):
                flat.extend(f.filters)
            elif isinstance(f, SpatialFilter):
                flat.append(f)
            else:
                raise TypeError(f"FilterChain entries must be SpatialFilter instances, got {type(f).__name__}.")
        if not flat:
            raise ValueError("FilterChain needs at least one filter (use NoFilter() for no filtering).")
        self._filters = tuple(flat)

    @property
    def filters(self) -> tuple[SpatialFilter, ...]:
        return self._filters

    def smooth(self, f: np.ndarray, periodic: bool | Sequence[bool] | None = None) -> np.ndarray:
        out = np.asarray(f)
        for filt in self._filters:
            out = filt.smooth(out, periodic=periodic)
        return out

    def _first_derivative(self, f: np.ndarray, dx: float, axis: int, periodic: bool) -> np.ndarray:
        return self._filters[-1]._first_derivative(f, dx, axis=axis, periodic=periodic)

    def _nth_derivative(self, f: np.ndarray, dx: float, axis: int, order: int, periodic: bool) -> np.ndarray:
        return self._filters[-1]._nth_derivative(f, dx, axis=axis, order=order, periodic=periodic)

    def __repr__(self) -> str:
        return f"FilterChain({', '.join(repr(f) for f in self._filters)})"


def as_filter(filters: SpatialFilter | Sequence[SpatialFilter] | None) -> SpatialFilter:
    """Normalise a config value into a single :class:`SpatialFilter`.

    None / empty sequence → :class:`NoFilter`; a single filter is returned
    as-is; several filters are wrapped in a :class:`FilterChain`.
    """
    if filters is None:
        return NoFilter()
    if isinstance(filters, SpatialFilter):
        return filters
    filters = tuple(filters)
    if len(filters) == 0:
        return NoFilter()
    if len(filters) == 1:
        return as_filter(filters[0])
    return FilterChain(*filters)
