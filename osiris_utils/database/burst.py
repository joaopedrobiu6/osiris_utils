"""Burst-dump time axis: midpoint frames + in-burst time derivatives.

OSIRIS burst dumps (``if_use_burst_dump`` in the ``time_step`` and diagnostic
namelists) write a report not only at the ordinary dump iterations
``m * ndump * ndump_fac`` but also at a small window of offsets around each of
them, e.g. ``burst_dump_range = -1, 1`` gives::

    n = 0, 1, 199, 200, 201, 399, 400, 401, ...     (ndump = 200)

Two consequences for post-processing:

1. Frames are no longer equally spaced in time, so ``dt * ndump`` is not the
   spacing between consecutive files and every uniform-grid time derivative in
   :mod:`osiris_utils.postprocessing.derivative` is wrong on such a series.
2. A quantity written with burst dumps has ~3x as many files as one written
   without, so a plain frame index no longer refers to the same physical time
   across diagnostics.

:class:`BurstAxis` fixes both.  It aligns every diagnostic on the *midpoint*
iterations (the ordinary dump times, where all quantities exist whether or not
they were bursted) and, for the quantities that were bursted, exposes the pair
of neighbouring frames needed for a centered time derivative evaluated exactly
at that midpoint::

    dv/dt|_{n}  =  ( v[n + k] - v[n - k] ) / (2 k dt)

so the time-derivative term sits at the same time and the same grid points as
every other term in the equation, instead of being offset by half a coarse dump
interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = ["BurstAxis", "BurstConfig", "BurstStencil", "describe_axes"]


@dataclass(frozen=True)
class BurstConfig:
    """How to read a burst-dumped run.

    Parameters
    ----------
    ndump_fac :
        ``ndump_fac`` of the diagnostic being read.  Midpoints are the
        iterations that are multiples of ``ndump * ndump_fac`` — i.e. the dumps
        that would exist without bursting.
    deriv_quantities :
        Quantities whose time derivative the pipeline needs.  These *must* be
        bursted in the input deck; the others may be, but need not be.
    require_centered :
        If True (default) a group is only kept when the derivative quantity has
        both neighbours (``n - k`` and ``n + k``) available, so every frame of
        the database uses the same second-order centered scheme.  This drops the
        ``n = 0`` group, which has no left neighbour.  Set to False to keep it
        with a first-order one-sided difference instead.
    """

    ndump_fac: int = 1
    deriv_quantities: tuple[str, ...] = ("vfl1",)
    require_centered: bool = True


@dataclass(frozen=True)
class BurstStencil:
    """Frame indices and denominator of one in-burst time derivative.

    ``(f[i_hi] - f[i_lo]) / h`` approximates ``df/dt`` at the group midpoint:
    second-order accurate when :attr:`centered`, first-order otherwise.
    """

    i_lo: int
    i_hi: int
    h: float
    centered: bool


class BurstAxis:
    """Midpoint-aligned time axis over a set of (possibly bursted) diagnostics.

    Parameters
    ----------
    diagnostics :
        ``{name: Diagnostic}`` — every raw quantity the frame pipeline reads.
        Each must expose ``iterations`` (see
        :attr:`osiris_utils.data.diagnostic.Diagnostic.iterations`).
    dt :
        Simulation timestep.
    ndump :
        Global ``ndump`` from the ``time_step`` namelist.
    config :
        See :class:`BurstConfig`.

    Attributes
    ----------
    midpoints :
        Iteration number of each retained group, ascending.
    dump_indices :
        ``midpoints // (ndump * ndump_fac)`` — the frame index the same data
        would have had in a non-bursted run.  Use these to select a time range,
        so ``--t0/--t1`` keep meaning "dump number" regardless of bursting.
    """

    def __init__(
        self,
        diagnostics: Mapping[str, Any],
        dt: float,
        ndump: int,
        config: BurstConfig | None = None,
    ) -> None:
        self.config = config or BurstConfig()
        self.dt = float(dt)
        self.stride = int(ndump) * int(self.config.ndump_fac)
        if self.stride < 1:
            raise ValueError(f"ndump * ndump_fac must be >= 1 (got {self.stride}).")

        missing = [q for q in self.config.deriv_quantities if q not in diagnostics]
        if missing:
            raise KeyError(f"deriv_quantities {missing} are not among the loaded diagnostics {sorted(diagnostics)}.")

        self._axes: dict[str, np.ndarray] = {name: np.asarray(diag.iterations, dtype=np.int64) for name, diag in diagnostics.items()}
        for name, iters in self._axes.items():
            if iters.size == 0:
                raise ValueError(f"Diagnostic '{name}' has no frames.")

        self._report_burst_status()

        midpoints = self._common_midpoints()
        self._index: dict[str, np.ndarray] = {}
        self._stencils: dict[str, list[BurstStencil | None]] = {}

        keep = np.ones(midpoints.size, dtype=bool)
        stencils: dict[str, list[BurstStencil | None]] = {q: [] for q in self.config.deriv_quantities}
        for q in self.config.deriv_quantities:
            for gi, n_mid in enumerate(midpoints):
                st = self._stencil_at(self._axes[q], int(n_mid))
                stencils[q].append(st)
                if st is None or (self.config.require_centered and not st.centered):
                    keep[gi] = False

        dropped = int((~keep).sum())
        if dropped:
            logger.info(
                "Burst axis: dropping %d of %d groups without a full centered stencil (iterations %s).",
                dropped,
                midpoints.size,
                np.asarray(midpoints)[~keep][:5].tolist(),
            )
        if not keep.any():
            raise ValueError(
                "No burst group has usable time-derivative neighbours. Check that "
                f"{list(self.config.deriv_quantities)} really were dumped with if_use_burst_dump."
            )

        self.midpoints = midpoints[keep]
        self.dump_indices = self.midpoints // self.stride
        for name, iters in self._axes.items():
            lookup = {int(n): i for i, n in enumerate(iters)}
            self._index[name] = np.array([lookup[int(n)] for n in self.midpoints], dtype=np.int64)
        self._stencils = {q: [st for st, k in zip(stencils[q], keep, strict=True) if k] for q in self.config.deriv_quantities}

        logger.info(
            "Burst axis: %d groups, iterations %d..%d (stride %d), dt=%g.",
            self.n_groups,
            int(self.midpoints[0]),
            int(self.midpoints[-1]),
            self.stride,
            self.dt,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _report_burst_status(self) -> None:
        for name, iters in self._axes.items():
            steps = np.unique(np.diff(iters)) if iters.size > 1 else np.array([])
            kind = "burst" if steps.size > 1 else "regular"
            logger.debug("Diagnostic '%s': %d frames, %s (iteration steps %s).", name, iters.size, kind, steps[:4].tolist())

    def _common_midpoints(self) -> np.ndarray:
        """Iterations that are ordinary dump times and exist in every diagnostic."""
        common: np.ndarray | None = None
        for name, iters in self._axes.items():
            on_grid = iters[iters % self.stride == 0]
            if on_grid.size == 0:
                raise ValueError(
                    f"Diagnostic '{name}' has no frame on the dump grid (multiples of {self.stride}). Is ndump_fac set correctly?"
                )
            common = on_grid if common is None else np.intersect1d(common, on_grid, assume_unique=True)
        assert common is not None
        if common.size == 0:
            raise ValueError(
                "The diagnostics share no common dump iteration. They were probably written with "
                "different ndump_fac values; build separate databases or align the deck."
            )
        return np.sort(common)

    def _stencil_at(self, iters: np.ndarray, n_mid: int) -> BurstStencil | None:
        """Best available derivative stencil for iteration *n_mid* on axis *iters*."""
        pos = int(np.searchsorted(iters, n_mid))
        if pos >= iters.size or iters[pos] != n_mid:
            return None

        left = int(iters[pos - 1]) if pos > 0 else None
        right = int(iters[pos + 1]) if pos + 1 < iters.size else None

        # Centered: both neighbours present and symmetric about the midpoint.
        # Asymmetric neighbours would still be a valid (non-uniform) 2-point
        # formula, but it evaluates df/dt at (left+right)/2, not at n_mid —
        # exactly the offset this class exists to avoid.
        if left is not None and right is not None and (n_mid - left) == (right - n_mid):
            return BurstStencil(i_lo=pos - 1, i_hi=pos + 1, h=(right - left) * self.dt, centered=True)

        if right is not None:
            return BurstStencil(i_lo=pos, i_hi=pos + 1, h=(right - n_mid) * self.dt, centered=False)
        if left is not None:
            return BurstStencil(i_lo=pos - 1, i_hi=pos, h=(n_mid - left) * self.dt, centered=False)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_groups(self) -> int:
        return int(self.midpoints.size)

    def indices(self, group: int) -> dict[str, int]:
        """Frame index of the midpoint of *group*, per diagnostic."""
        return {name: int(idx[group]) for name, idx in self._index.items()}

    def stencil(self, quantity: str, group: int) -> BurstStencil:
        """Time-derivative stencil for *quantity* at the midpoint of *group*."""
        try:
            st = self._stencils[quantity][group]
        except KeyError as e:
            raise KeyError(f"'{quantity}' is not in deriv_quantities {list(self.config.deriv_quantities)}.") from e
        if st is None:  # pragma: no cover - groups without a stencil are dropped in __init__
            raise ValueError(f"No time-derivative stencil for '{quantity}' at iteration {int(self.midpoints[group])}.")
        return st

    def time(self, group: int) -> float:
        """Physical time of the midpoint of *group*."""
        return float(self.midpoints[group]) * self.dt

    def groups_in_dump_range(self, first: int, last: int) -> list[int]:
        """Group ids whose dump index lies in ``[first, last)``."""
        sel = np.flatnonzero((self.dump_indices >= int(first)) & (self.dump_indices < int(last)))
        return [int(g) for g in sel]

    def summary(self) -> str:
        centered = sum(1 for q in self.config.deriv_quantities for st in self._stencils[q] if st and st.centered)
        total = sum(len(self._stencils[q]) for q in self.config.deriv_quantities)
        return (
            f"BurstAxis({self.n_groups} groups, dump index "
            f"{int(self.dump_indices[0])}..{int(self.dump_indices[-1])}, "
            f"{centered}/{total} centered stencils)"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def describe_axes(diagnostics: Mapping[str, Any], ndump: int, ndump_fac: int = 1) -> str:
    """Human-readable report of which diagnostics were burst-dumped.

    Useful as a sanity check before building a database::

        print(describe_axes(db._load_raw_diagnostics(), ndump=200))
    """
    stride = int(ndump) * int(ndump_fac)
    lines = [f"dump grid: multiples of {stride}"]
    for name, diag in diagnostics.items():
        iters: Sequence[int] = np.asarray(diag.iterations, dtype=np.int64)
        steps = np.unique(np.diff(iters)) if len(iters) > 1 else np.array([])
        kind = "burst" if steps.size > 1 else "regular"
        head = ", ".join(str(int(i)) for i in iters[:6])
        lines.append(f"  {name:>6}: {len(iters):6d} frames  {kind:<7}  steps={steps[:4].tolist()}  n = {head}, ...")
    return "\n".join(lines)
