from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.diagnostic import Diagnostic
from ..filters import SpatialFilter, as_filter
from .postprocess import PostProcess

logger = logging.getLogger(__name__)


def _uniform_frame_dt(diag) -> float:
    """Spacing in time between consecutive frames of *diag*.

    Every time-derivative scheme in this module assumes a uniformly spaced time
    axis.  Burst dumps (OSIRIS ``if_use_burst_dump``) break that assumption —
    frames come in tight clusters around each ordinary dump — so a d/dt computed
    with ``dt * ndump`` would be wrong by orders of magnitude.  Detect it and
    refuse rather than return a silently wrong derivative.
    """
    h, valid = _frame_dt_axis(diag)
    if valid is not None:  # pragma: no cover - _frame_dt_axis raises without a stencil
        raise ValueError("Non-uniform frame axis has no single time step.")
    return float(h)


def _frame_dt_axis(diag, stencil=None, deriv_order: int = 1):
    """Time step of every frame, and where *stencil* is meaningful.

    Returns ``(h, valid)``.

    * **Uniform axis** (the ordinary case, and any non-burst run): ``h`` is a
      float and ``valid`` is ``None`` — callers take the historical fast path.
    * **Burst axis**: the frames are unevenly spaced, so no single step exists.
      ``h`` is a per-frame array of local steps and ``valid`` marks the frames
      whose stencil spans *uniformly spaced* iterations — the burst midpoints
      that carry a full symmetric set of neighbours. Everywhere else the scheme
      has no meaning at all and the caller must emit NaN rather than a number.

    Concretely, for ``stencil = [-1, 0, 1]`` on ``0, 1, 199, 200, 201, 399, …``
    only the frames at 200, 400, … are valid, and there ``h = dt`` — the in-burst
    spacing — instead of ``dt * ndump``.
    """
    key = (tuple(int(x) for x in stencil) if stencil is not None else None, int(deriv_order))
    cache = getattr(diag, "_frame_dt_cache", None)
    if cache is None:
        cache = {}
        diag._frame_dt_cache = cache
    if key in cache:
        return cache[key]

    iterations = getattr(diag, "_iterations", None)
    dt = float(diag._dt) if diag._dt is not None else 1.0
    if iterations is None or len(iterations) < 2:
        cache[key] = (float(diag._dt * diag._ndump), None)
        return cache[key]

    iters = np.asarray(iterations, dtype=np.int64)
    steps = np.diff(iters)
    if steps.min() == steps.max():
        cache[key] = (dt * float(steps[0]), None)
        return cache[key]

    if stencil is None:
        raise ValueError(
            "Time derivatives require equally spaced frames, but this diagnostic has "
            f"non-uniform frame spacing (iteration steps {steps.min()}..{steps.max()}). "
            "This is what burst dumps look like — pass an explicit stencil= to "
            "differentiate inside each burst (valid only on the midpoints), or use "
            "osiris_utils.database.BurstAxis / DatabaseCreator with a BurstConfig."
        )

    s = np.asarray(stencil, dtype=int)
    span = int(s.max() - s.min())
    n = int(iters.size)
    h = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = i + int(s.min()), i + int(s.max())
        if lo < 0 or hi >= n or span == 0:
            continue
        total = int(iters[hi]) - int(iters[lo])
        if total <= 0 or total % span:
            continue
        step = total // span
        # every stencil point must sit exactly on the arithmetic progression
        if np.array_equal(iters[i + s], iters[i] + s * step):
            h[i] = dt * step
            valid[i] = True

    logger.warning(
        "Non-uniform frame axis (burst dumps): the %s stencil is only meaningful on %d of %d "
        "frames (iterations %s); every other frame of this time derivative is NaN.",
        [int(x) for x in s],
        int(valid.sum()),
        n,
        [int(v) for v in iters[valid][:6]],
    )
    cache[key] = (h, valid)
    return cache[key]


# ---------------------------------------------------------------------------
# Optional Numba JIT acceleration for the FD stencil hot paths.
#
# HPC notes
# ---------
# * cache=True  — compiled kernels are stored on disk (per dtype/shape
#   signature) so the JIT cost is paid only on the very first run.
#   Point NUMBA_CACHE_DIR at a fast local scratch directory on HPC nodes:
#       export NUMBA_CACHE_DIR=/tmp/$SLURM_JOB_ID/numba_cache
# * parallel=True — uses OpenMP/TBB threading controlled by NUMBA_NUM_THREADS
#   (defaults to OMP_NUM_THREADS).  Set it in your job script:
#       export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# * No MPI dependency — each rank compiles and caches independently.
# * If Numba is not installed, the code falls back to the pure-NumPy path
#   transparently; _NUMBA_AVAILABLE guards every call site.
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_AVAILABLE = True

    @_numba.njit(parallel=True, cache=True)
    def _nb_stencil_interior(data_2d, out_2d, c, s0, start, stop):
        """Fill out_2d[start:stop, :] via the FD stencil sum.

        data_2d / out_2d are C-contiguous float64 views shaped (n_ax, n_rest),
        where axis-0 is the derivative axis and n_rest collapses all others.
        """
        n_rest = data_2d.shape[1]
        for i in _numba.prange(start, stop):
            for j in range(n_rest):
                acc = 0.0
                for k in range(len(c)):
                    acc += c[k] * data_2d[i + s0[k], j]
                out_2d[i, j] = acc

    @_numba.njit(parallel=True, cache=True)
    def _nb_stencil_periodic(p_2d, out_2d, c, s0, n_ax, smax):
        """Fill out_2d via the periodic FD stencil sum.

        p_2d is the wrap-padded array shaped (n_ax + 2*smax, n_rest).
        out_2d is shaped (n_ax, n_rest).
        """
        n_rest = out_2d.shape[1]
        for i in _numba.prange(n_ax):
            for j in range(n_rest):
                acc = 0.0
                for k in range(len(c)):
                    acc += c[k] * p_2d[smax + i + s0[k], j]
                out_2d[i, j] = acc

except ImportError:
    _NUMBA_AVAILABLE = False


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["Derivative_Diagnostic", "Derivative_Simulation"]


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
        Scheme selector (2 or 4). Ignored if stencil is provided.
    stencil : Iterable[int] or None
        Integer offsets (in grid steps), e.g. [-2,-1,0,1,2] or [0,1,2,3].
    deriv_order : int
        Derivative order (1 for first derivative, 2 for second derivative, ...).
    periodic : bool
        If True, derivatives are computed with periodic boundary conditions.
    filter : SpatialFilter or None
        If given, spatial derivatives use the filter's own single-pass scheme
        instead of finite differences (see :class:`Derivative_Diagnostic`).
    """

    def __init__(
        self,
        simulation: Any,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 4,
        stencil: Iterable[int] | None = None,
        deriv_order: int = 1,
        periodic: bool = False,
        filter: SpatialFilter | Sequence[SpatialFilter] | None = None,  # noqa: A002
    ):
        super().__init__(f"Derivative({deriv_type})", simulation)
        # Derivative_Diagnostic's arguments, the same for every quantity
        self._derivative_args = dict(
            deriv_type=deriv_type,
            axis=axis,
            order=order,
            stencil=stencil,
            deriv_order=deriv_order,
            periodic=periodic,
            filter=filter,
        )

    def _process(self, container: Any, key: str) -> Derivative_Diagnostic:
        return Derivative_Diagnostic(container[key], **self._derivative_args)


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
    order : int
        Scheme selector (2 or 4). Ignored if stencil is provided.
    stencil : Iterable[int] or None
        Integer offsets (in grid steps), e.g. [-2,-1,0,1,2] or [0,1,2,3].
    deriv_order : int
        Derivative order (1 for first derivative, 2 for second derivative, ...).
        Only used if stencil is provided.
    periodic : bool
        If True, derivatives are computed with periodic boundary conditions.
    filter : SpatialFilter or None
        A :class:`~osiris_utils.filters.SpatialFilter` whose own derivative
        scheme replaces finite differences on the **spatial** axes: one pass of
        the filter's analytic order-``deriv_order`` kernel
        (``savgol_filter(deriv=...)`` / ``gaussian_filter1d(order=...)``)
        instead of the stencil, exactly as the database creators compute their
        derivatives.  ``stencil`` and ``order`` are then unused.  This is the
        derivative half of filtering; the smoothing half belongs to the input
        diagnostic (:class:`~osiris_utils.postprocessing.filtering.Filtered_Diagnostic`),
        and the database pipeline applies both — one smoothing pass on the
        field, one kernel pass per derivative.

        Time derivatives are never filtered: a spatial kernel has no meaning
        along the time axis, so ``deriv_type="t"`` with a filter is an error,
        and the mixed types (``"xt"``, ``"tx"``) filter their spatial factor
        only.  This matches
        :func:`osiris_utils.database.database._time_derivative_frame`, which
        differences two *already smoothed* frames.

    Methods
    -------
    load_all()
        Load all the data and compute the derivative.
    __getitem__(index)
        Get data at a specific index.

    """

    #: Frames of the *input* diagnostic kept per instance: enough for a
    #: 4th-order time stencil, which asks for five neighbours around a point.
    frame_cache_size: int = 6

    def __init__(
        self,
        diagnostic: Diagnostic,
        deriv_type: str,
        axis: int | tuple[int, int] | None = None,
        order: int = 4,
        stencil: Iterable[int] | None = None,
        deriv_order: int = 1,
        periodic: bool = False,
        filter: SpatialFilter | Sequence[SpatialFilter] | None = None,  # noqa: A002
    ) -> None:
        super().__init__()
        self._derive_from(diagnostic)

        self.postprocess_name = "DERIV"

        self._deriv_type = deriv_type
        self._op_axis = axis
        self._order = int(order)

        self._stencil = None if stencil is None else tuple(int(s) for s in stencil)
        self._deriv_order = int(deriv_order)
        self._periodic = periodic
        self._filter = None if filter is None else as_filter(filter)

        # Every deriv_type is an optional time derivative followed by spatial
        # ones: 't' -> d/dt, 'x1' -> d/dx1, 'xx' -> d/dxa d/dxb, 'xt'/'tx' ->
        # d/dt then d/dxa.  The whole configuration is checked here, once.
        if deriv_type in ("x1", "x2", "x3"):
            spatial = (int(deriv_type[1]),)
        elif deriv_type == "xx":
            if not isinstance(axis, (tuple, list)) or len(axis) != 2:
                raise ValueError("For 'xx', axis must be a tuple of two OSIRIS spatial axes (1..3).")
            spatial = tuple(int(a) for a in axis)
        elif deriv_type in ("xt", "tx"):
            if not isinstance(axis, (int, np.integer)):
                raise ValueError(f"For '{deriv_type}', axis must be an OSIRIS spatial axis int (1..3).")
            spatial = (int(axis),)
        elif deriv_type == "t":
            spatial = ()
        else:
            raise ValueError(f"Invalid derivative type {deriv_type!r}: use 't', 'x1', 'x2', 'x3', 'xx', 'xt' or 'tx'.")
        for ax in spatial:
            if not 1 <= ax <= (self._dim or 3):
                raise ValueError(f"'{deriv_type}' along x{ax}, but the diagnostic is {self._dim}-D.")
        self._spatial_axes = tuple(ax - 1 for ax in spatial)  # numpy axes of one frame, applied in order
        self._has_time = deriv_type in ("t", "xt", "tx")

        if self._filter is not None and deriv_type == "t":
            raise ValueError(
                "A spatial filter has no derivative scheme along the time axis. "
                "Drop filter= for deriv_type='t' (the frames it differences are "
                "already smoothed if the input diagnostic is filtered)."
            )
        if self._stencil is not None:
            self._validate_stencil(self._stencil, self._deriv_order)
        elif self._order not in (2, 4) and (self._filter is None or self._has_time):
            raise ValueError("Only order 2 and 4 supported.")

        # Step of every frame, and where the time stencil means anything (a
        # burst axis); refuses a non-uniform axis without an explicit stencil.
        self._frame_h, self._frame_h_valid = (
            _frame_dt_axis(diagnostic, self._stencil, self._deriv_order) if self._has_time else (None, None)
        )

    @staticmethod
    def _validate_stencil(stencil: tuple[int, ...], deriv_order: int) -> None:
        if len(stencil) == 0:
            raise ValueError("Stencil cannot be empty")
        if len(set(stencil)) != len(stencil):
            raise ValueError(f"Stencil offsets must be distinct. Got {stencil}.")
        if deriv_order < 0:
            raise ValueError("deriv_order must be >= 0")
        if deriv_order >= len(stencil):
            raise ValueError(f"Need len(stencil) > deriv_order. Got len={len(stencil)} deriv_order={deriv_order}.")

    # Compute FD coefficients for given stencil and derivative order
    @staticmethod
    @lru_cache(maxsize=2048)
    def _fd_unit_coeffs(stencil: tuple[int, ...], deriv_order: int) -> np.ndarray:
        """Unit-spacing coefficients (h=1). Scale by 1/h^d when applying.

        Parameters
        ----------
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        np.ndarray
            The finite difference coefficients.

        Maths
        -----
        Computes sum_j c_j s_j^n = n! * delta_{n,d},   n = 0..N-1
        """
        Derivative_Diagnostic._validate_stencil(stencil, deriv_order)

        s = np.asarray(stencil, dtype=float)
        N = s.size

        # V[n,j] = s_j^n, n=0..N-1
        V = np.vander(s, N=N, increasing=True).T  # (N,N)
        rhs = np.zeros(N, dtype=float)
        rhs[deriv_order] = math.factorial(deriv_order)

        return np.linalg.solve(V, rhs)

    @staticmethod
    def _shift_stencil_to_fit(stencil: np.ndarray, i: int, n: int) -> np.ndarray:
        """Shift stencil by an integer k so i + (s_j + k) in [0,n-1] for all j.
        Prefer k=0 if feasible.

        Parameters
        ----------
        stencil : np.ndarray
            The stencil offsets (in grid steps).
        i : int
            The current index where the stencil will be applied.
        n : int
            The total number of points along the axis.

        Returns
        -------
        np.ndarray
            The shifted stencil.

        Examples
        --------
        >>> s = np.array([-2, -1, 0, 1, 2])
        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=1, n=10)
        array([-1,  0,  1,  2,  3])

        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=9, n=10)
        array([5, 6, 7, 8, 9])

        >>> Derivative_Diagnostic._shift_stencil_to_fit(s, i=5, n=10)
        array([-2, -1,  0,  1,  2])
        """
        s = stencil.astype(int)

        k_min = -(10**18)
        k_max = 10**18
        for sj in s:
            k_min = max(k_min, -i - sj)
            k_max = min(k_max, (n - 1) - i - sj)

        if k_min > k_max:
            raise ValueError(f"Stencil {stencil.tolist()} cannot be shifted to fit at i={i}, n={n}.")

        if k_min <= 0 <= k_max:
            k = 0
        else:
            # pick nearest feasible integer
            k = k_min if abs(k_min) < abs(k_max) else k_max

        return s + int(k)

    @staticmethod
    @lru_cache(maxsize=128)
    def _edge_plan(n: int, axis: int, stencil: tuple[int, ...], deriv_order: int):  # noqa: ARG004
        """Precompute which indices need shifted stencils and what those shifted stencils are.

        Parameters
        ----------
        n : int
            Number of points along the axis.
        axis : int
            Axis along which the derivative is computed.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        idxs : list[int]
            Indices where shifted stencils are needed.
        stencils : list[tuple[int,...]]
            Shifted stencils (same length as idxs)

        Examples
        --------
        >>> Derivative_Diagnostic._edge_plan(n=10, axis=1, stencil=(-2, -1, 0, 1, 2), deriv_order=1)
        ([0, 1, 8, 9], [(-2, -1, 0, 1, 2), (-1, 0, 1, 2, 3), (5, 6, 7, 8, 9), (6, 7, 8, 9, 10)])
        """

        # Pass stencil as np array for easier min/max
        s0 = np.asarray(stencil, dtype=int)
        smin = int(s0.min())  # minimum shift
        smax = int(s0.max())  # maximum shift

        start = -smin
        stop = n - smax  # interior indices are [start, stop)
        idxs: list[int] = []
        stencils: list[tuple[int, ...]] = []

        # left edge
        for i in range(0, max(0, start)):
            s_i = Derivative_Diagnostic._shift_stencil_to_fit(s0, i=i, n=n)
            idxs.append(i)
            stencils.append(tuple(int(x) for x in s_i.tolist()))

        # right edge
        for i in range(max(0, stop), n):
            s_i = Derivative_Diagnostic._shift_stencil_to_fit(s0, i=i, n=n)
            idxs.append(i)
            stencils.append(tuple(int(x) for x in s_i.tolist()))

        return idxs, stencils

    def _apply_stencil_unshifted_interior(
        self,
        data: np.ndarray,
        h: float,
        axis: int,
        stencil: tuple[int, ...],
        deriv_order: int,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Apply unshifted stencil on the interior region where it fits. Returns (out, (start,stop)).

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        h : float
            The grid spacing.
        axis : int
            The axis along which to apply the stencil.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).
        deriv_order : int
            The derivative order.

        Returns
        -------
        out : np.ndarray
            The output array with the derivative applied in the interior region.
        (start, stop) : tuple[int, int]
            The start and stop indices of the interior region where the stencil was applied.
        """
        s0 = np.asarray(stencil, dtype=int)
        n = data.shape[axis]
        smin = int(s0.min())
        smax = int(s0.max())

        start = -smin
        stop = n - smax
        out = np.zeros_like(data, dtype=float)

        if stop <= start:
            return out, (start, stop)

        c_unit = self._fd_unit_coeffs(stencil, deriv_order)
        c = c_unit / (float(h) ** deriv_order)

        if _NUMBA_AVAILABLE:
            # Move derivative axis to front, flatten remaining dims → 2D, JIT.
            data_m = np.ascontiguousarray(np.moveaxis(data, axis, 0))
            n_ax_d = data_m.shape[0]
            data_2d = data_m.reshape(n_ax_d, -1).astype(np.float64)
            out_2d = np.zeros_like(data_2d)
            _nb_stencil_interior(
                data_2d,
                out_2d,
                c.astype(np.float64),
                s0.astype(np.int64),
                start,
                stop,
            )
            out = np.ascontiguousarray(np.moveaxis(out_2d.reshape(data_m.shape), 0, axis))
        else:
            tgt = [slice(None)] * data.ndim
            tgt[axis] = slice(start, stop)
            interior_view = data[tuple(tgt)]
            acc = np.zeros_like(interior_view, dtype=float)
            for cj, sj in zip(c, s0, strict=False):
                src = [slice(None)] * data.ndim
                src[axis] = slice(start + sj, stop + sj)
                acc += cj * data[tuple(src)]
            out[tuple(tgt)] = acc

        return out, (start, stop)

    def _fd_apply_along_axis(
        self,
        data: np.ndarray,
        h: float,
        axis: int,
        deriv_order: int,
        stencil: tuple[int, ...],
    ) -> np.ndarray:
        """Apply arbitrary stencil along an axis:
        - vectorized interior (unshifted)
        - precomputed shifted stencils at edges

        Parameters
        ----------
        data : np.ndarray
            The input data array.
        h : float
            The grid spacing.
        axis : int
            The axis along which to apply the stencil.
        deriv_order : int
            The derivative order.
        stencil : tuple[int,...]
            The stencil offsets (in grid steps).

        Returns
        -------
        np.ndarray
            The output array with the derivative applied.
        """
        self._validate_stencil(stencil, deriv_order)

        c_unit = self._fd_unit_coeffs(stencil, deriv_order)
        c = c_unit / (float(h) ** deriv_order)

        if self._periodic:
            smax = max(abs(int(s)) for s in stencil)
            pad_w = [(0, 0)] * data.ndim
            pad_w[axis] = (smax, smax)
            p = np.pad(data, pad_w, mode="wrap")
            n_ax = data.shape[axis]

            if _NUMBA_AVAILABLE:
                # Move derivative axis to front for both padded and unpadded shapes.
                data_m_shape = np.moveaxis(data, axis, 0).shape  # (n_ax, *other_dims)
                p_m = np.ascontiguousarray(np.moveaxis(p, axis, 0))
                n_rest = p_m.size // p_m.shape[0]
                p_2d = p_m.reshape(p_m.shape[0], n_rest).astype(np.float64)
                out_2d = np.zeros((n_ax, n_rest), dtype=np.float64)
                _nb_stencil_periodic(
                    p_2d,
                    out_2d,
                    c.astype(np.float64),
                    np.asarray(stencil, dtype=np.int64),
                    n_ax,
                    smax,
                )
                out = np.ascontiguousarray(np.moveaxis(out_2d.reshape(data_m_shape), 0, axis))
            else:
                # pad once, read shifted views as zero-copy slices, accumulate in-place.
                out = np.zeros(data.shape, dtype=float)
                tmp = np.empty(data.shape, dtype=float)
                for cj, sj in zip(c, stencil, strict=False):
                    sl = [slice(None)] * p.ndim
                    sl[axis] = slice(smax + int(sj), smax + int(sj) + n_ax)
                    np.multiply(p[tuple(sl)], cj, out=tmp)
                    out += tmp
            return out

        n = data.shape[axis]
        # Central interior (vectorized)
        out, (start, stop) = self._apply_stencil_unshifted_interior(data, h, axis, stencil, deriv_order)

        # edges plan + apply (small loops only over edge indices)
        idxs, stencils = self._edge_plan(n, axis, stencil, deriv_order)

        # Apply edge stencils
        for i, s_i in zip(idxs, stencils, strict=False):
            # Compute coefficients for this shifted stencil
            c_unit = self._fd_unit_coeffs(s_i, deriv_order)
            # Scale by h^d
            c_i = c_unit / (float(h) ** deriv_order)
            # Convert stencil to array of ints
            s_arr = np.asarray(s_i, dtype=int)

            # Target index
            tgt = [slice(None)] * data.ndim
            tgt[axis] = i

            # Apply stencil at index i
            acc = 0.0
            for cj, sj in zip(c_i, s_arr, strict=False):
                src = [slice(None)] * data.ndim
                src[axis] = i + int(sj)
                acc = acc + cj * data[tuple(src)]
            out[tuple(tgt)] = acc

        return out

    @staticmethod
    def _periodic_first_derivative(data: np.ndarray, h: float, axis: int, order: int) -> np.ndarray:
        """Periodic first derivative using centered finite differences.

        Uses ``np.pad(mode='wrap')`` + contiguous slice views instead of multiple
        ``np.roll`` calls. Each ``np.roll`` allocates a full copy of the array;
        the pad approach allocates one slightly-larger array and reads shifts as
        zero-copy views, cutting peak intermediate allocations from O(order * N)
        to O(N).
        """
        n = data.shape[axis]
        pad_w = [(0, 0)] * data.ndim

        def sl(p: np.ndarray, center: int, off: int) -> tuple:
            s = [slice(None)] * p.ndim
            s[axis] = slice(center + off, center + off + n)
            return tuple(s)

        if order == 2:
            pad_w[axis] = (1, 1)
            p = np.pad(data, pad_w, mode="wrap")
            # (f[i+1] - f[i-1]) / (2h) — 1 temp allocation
            return (p[sl(p, 1, 1)] - p[sl(p, 1, -1)]) / (2 * h)

        if order == 4:
            pad_w[axis] = (2, 2)
            p = np.pad(data, pad_w, mode="wrap")
            # (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12h) — 2 temp allocations
            out = p[sl(p, 2, -2)] - p[sl(p, 2, 2)]  # f[i-2] - f[i+2]
            out += 8.0 * (p[sl(p, 2, 1)] - p[sl(p, 2, -1)])  # + 8*(f[i+1] - f[i-1])
            out /= 12.0 * h
            return out

        raise ValueError("Only order 2 and 4 supported.")

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

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        f = self._d_dt(index, data_slice) if self._has_time else self._base(index, data_slice)
        for ax in self._spatial_axes:
            f = self._d_dx(f, ax)
        return f

    def _d_dx(self, f: np.ndarray, ax_np: int) -> np.ndarray:
        """Configured spatial derivative of one frame *f* along its numpy axis *ax_np*."""
        dx = float(np.atleast_1d(self._dx)[ax_np])

        # a filter brings its own single-pass scheme and overrides both
        if self._filter is not None:
            return self._filter.derivative(f, dx, axis=ax_np, order=self._deriv_order, periodic=self._periodic)

        # stencil overrides legacy order
        if self._stencil is not None:
            return self._fd_apply_along_axis(f, h=dx, axis=ax_np, deriv_order=self._deriv_order, stencil=self._stencil)

        if self._order == 2:
            if self._periodic:
                return self._periodic_first_derivative(f, h=dx, axis=ax_np, order=2)
            return np.gradient(f, dx, axis=ax_np, edge_order=2)
        if self._periodic:
            return self._periodic_first_derivative(f, h=dx, axis=ax_np, order=4)
        return self._compute_fourth_order_spatial(f, dx, ax_np)

    def _d_dt(self, i: int, data_slice: tuple | None) -> np.ndarray:
        """Configured time derivative at frame *i*, from the input's neighbouring frames."""
        n = int(self._maxiter)
        # On a burst axis the step is per-frame and the stencil is only
        # meaningful on the midpoints; elsewhere there is no right answer,
        # so return NaN rather than a plausible wrong number.
        if self._frame_h_valid is not None:
            if not self._frame_h_valid[i]:
                return self._base(i, data_slice) * np.nan
            dt_i = float(self._frame_h[i])
        else:
            dt_i = float(self._frame_h)

        # stencil overrides legacy order
        if self._stencil is not None:
            if self._periodic:
                s_i = np.asarray(self._stencil, dtype=int)
                idx = (i + s_i) % n
            else:
                s_i = self._shift_stencil_to_fit(np.asarray(self._stencil, dtype=int), i=i, n=n)
                idx = i + s_i
            c = self._fd_unit_coeffs(tuple(int(x) for x in s_i), self._deriv_order) / dt_i**self._deriv_order
            acc = 0.0
            for cj, j in zip(c, idx, strict=True):
                acc = acc + cj * self._base(int(j), data_slice)
            return acc

        if self._order == 2:
            if self._periodic:
                fp = self._base((i + 1) % n, data_slice)
                fm = self._base((i - 1) % n, data_slice)
                return (fp - fm) / (2 * dt_i)
            if i == 0:
                f0 = self._base(0, data_slice)
                f1 = self._base(1, data_slice)
                f2 = self._base(2, data_slice)
                return (-3 * f0 + 4 * f1 - f2) / (2 * dt_i)
            if i == n - 1:
                f0 = self._base(n - 1, data_slice)
                f1 = self._base(n - 2, data_slice)
                f2 = self._base(n - 3, data_slice)
                return (3 * f0 - 4 * f1 + f2) / (2 * dt_i)
            fp = self._base(i + 1, data_slice)
            fm = self._base(i - 1, data_slice)
            return (fp - fm) / (2 * dt_i)

        if self._periodic:
            f_p2 = self._base((i + 2) % n, data_slice)
            f_p1 = self._base((i + 1) % n, data_slice)
            f_m1 = self._base((i - 1) % n, data_slice)
            f_m2 = self._base((i - 2) % n, data_slice)
            return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dt_i)
        if i < 2 or i > n - 3:
            # edge fallback to 2nd-order
            if i == 0:
                f0 = self._base(0, data_slice)
                f1 = self._base(1, data_slice)
                f2 = self._base(2, data_slice)
                return (-3 * f0 + 4 * f1 - f2) / (2 * dt_i)
            if i == n - 1:
                f0 = self._base(n - 1, data_slice)
                f1 = self._base(n - 2, data_slice)
                f2 = self._base(n - 3, data_slice)
                return (3 * f0 - 4 * f1 + f2) / (2 * dt_i)
            fp = self._base(min(i + 1, n - 1), data_slice)
            fm = self._base(max(i - 1, 0), data_slice)
            return (fp - fm) / (2 * dt_i)

        f_p2 = self._base(i + 2, data_slice)
        f_p1 = self._base(i + 1, data_slice)
        f_m1 = self._base(i - 1, data_slice)
        f_m2 = self._base(i - 2, data_slice)
        return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * dt_i)

    def _base(self, idx: int, data_slice: tuple | None):
        """One frame of the input, from Diagnostic's LRU (frame_cache_size deep)."""
        key = ("base", idx, repr(data_slice))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        return self._cache_put(key, self._diag._frame(idx, data_slice=data_slice))
