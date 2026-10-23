from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.diagnostic import Diagnostic
from ..filters import SpatialFilter, as_filter
from ..finite_differences import fd_apply, fd_coefficients, stencil_at
from .postprocess import PostProcess

logger = logging.getLogger(__name__)


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
        instead of the stencil.  ``stencil`` and ``order`` are then unused.
        That is d/dx of the filtered field only for a **raw** input: on a
        :class:`~osiris_utils.postprocessing.filtering.Filtered_Diagnostic`
        (or anything built from one) it smooths a second time.  Derivatives
        of already-smoothed fields, as in the database pipeline, take no
        ``filter``.

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
        # The finite-difference scheme, used along time and space alike.
        # order=2/4 are first derivatives with the centred 3- and 5-point
        # stencils; order 4 keeps its historical 2nd-order stencil at the two
        # points per open edge where the 5-point one does not fit.
        if self._stencil is not None:
            fd_coefficients(self._stencil, self._deriv_order)  # validates the stencil
            self._fd_stencil, self._fd_edge, self._fd_order = self._stencil, None, self._deriv_order
        elif self._order == 2:
            self._fd_stencil, self._fd_edge, self._fd_order = (-1, 0, 1), None, 1
        elif self._order == 4:
            self._fd_stencil, self._fd_edge, self._fd_order = (-2, -1, 0, 1, 2), (-1, 0, 1), 1
        elif self._filter is None or self._has_time:
            raise ValueError("Only order 2 and 4 supported.")

        # Step of every frame, and where the time stencil means anything (a
        # burst axis); refuses a non-uniform axis without an explicit stencil.
        self._frame_h, self._frame_h_valid = (
            _frame_dt_axis(diagnostic, self._stencil, self._deriv_order) if self._has_time else (None, None)
        )

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        f = self._d_dt(index, data_slice) if self._has_time else self._base(index, data_slice)
        for ax in self._spatial_axes:
            f = self._d_dx(f, ax)
        return f

    def _d_dx(self, f: np.ndarray, ax_np: int) -> np.ndarray:
        """Configured spatial derivative of one frame *f* along its numpy axis *ax_np*."""
        dx = float(np.atleast_1d(self._dx)[ax_np])
        # a filter brings its own single-pass scheme
        if self._filter is not None:
            return self._filter.derivative(f, dx, axis=ax_np, order=self._deriv_order, periodic=self._periodic)
        return fd_apply(f, dx, ax_np, self._fd_stencil, self._fd_order, self._periodic, self._fd_edge)

    def _d_dt(self, i: int, data_slice: tuple | None) -> np.ndarray:
        """Configured time derivative at frame *i*: the stencil sum over the input's neighbouring frames."""
        n = int(self._maxiter)
        # On a burst axis the step is per-frame and the stencil is only
        # meaningful on the midpoints; elsewhere there is no right answer,
        # so return NaN rather than a plausible wrong number.
        if self._frame_h_valid is not None:
            if not self._frame_h_valid[i]:
                return self._base(i, data_slice) * np.nan
            h = float(self._frame_h[i])
        else:
            h = float(self._frame_h)

        if self._periodic:
            offsets = self._fd_stencil
            frames = (i + np.asarray(offsets)) % n
        else:
            offsets = stencil_at(i, n, self._fd_stencil, self._fd_edge)
            frames = i + np.asarray(offsets)
        c = fd_coefficients(offsets, self._fd_order) / h**self._fd_order
        acc = 0.0
        for cj, j in zip(c, frames, strict=True):
            acc = acc + cj * self._base(int(j), data_slice)
        return acc

    def _base(self, idx: int, data_slice: tuple | None):
        """One frame of the input, from Diagnostic's LRU (frame_cache_size deep)."""
        key = ("base", idx, repr(data_slice))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        return self._cache_put(key, self._diag._frame(idx, data_slice=data_slice))
