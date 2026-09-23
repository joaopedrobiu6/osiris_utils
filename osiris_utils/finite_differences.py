"""Finite-difference stencils: the one kernel behind every derivative in the package.

``Derivative_Diagnostic`` (lazy, time and space) and ``NoFilter`` (the
database creators' unfiltered derivatives) both go through :func:`fd_apply`;
the per-frame time derivative, which works on neighbouring *frames* rather
than an array axis, picks its stencil with :func:`stencil_at` and its
weights with :func:`fd_coefficients`.

A scheme is a stencil — integer offsets in grid steps, e.g. ``(-1, 0, 1)`` —
plus a derivative order.  On a periodic axis the stencil wraps; on an open
one it is used wherever it fits and shifted inwards near the edges (a
one-sided stencil of the same width, so the same order of accuracy), unless
an explicit ``edge_stencil`` is given.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

__all__ = ["fd_apply", "fd_coefficients", "stencil_at"]

# ---------------------------------------------------------------------------
# Optional Numba JIT acceleration for the stencil sums.
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
# * If Numba is not installed, the pure-NumPy path gives the same result;
#   _NUMBA_AVAILABLE guards every call site.
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_AVAILABLE = True

    @_numba.njit(parallel=True, cache=True)
    def _nb_stencil_sum(src_2d, out_2d, c, offsets, start, stop):
        """out_2d[i, :] = sum_k c[k] * src_2d[i + offsets[k], :] for i in [start, stop).

        Axis 0 is the derivative axis; axis 1 collapses all the others.
        """
        n_rest = src_2d.shape[1]
        for i in _numba.prange(start, stop):
            for j in range(n_rest):
                acc = 0.0
                for k in range(len(c)):
                    acc += c[k] * src_2d[i + offsets[k], j]
                out_2d[i, j] = acc

except ImportError:
    _NUMBA_AVAILABLE = False


@lru_cache(maxsize=2048)
def fd_coefficients(stencil: tuple[int, ...], deriv_order: int) -> np.ndarray:
    r"""Unit-spacing weights :math:`c_j` of a finite-difference stencil.

    :math:`\sum_j c_j f(x + s_j h) = h^d f^{(d)}(x) + O(h^{N})`, from the Taylor
    conditions :math:`\sum_j c_j s_j^m = m!\,\delta_{m d}`, :math:`m = 0..N-1`
    (a Vandermonde system).  Divide by :math:`h^d` when applying.
    """
    if len(stencil) == 0:
        raise ValueError("Stencil cannot be empty")
    if len(set(stencil)) != len(stencil):
        raise ValueError(f"Stencil offsets must be distinct. Got {stencil}.")
    if deriv_order < 0:
        raise ValueError("deriv_order must be >= 0")
    if deriv_order >= len(stencil):
        raise ValueError(f"Need len(stencil) > deriv_order. Got len={len(stencil)} deriv_order={deriv_order}.")
    s = np.asarray(stencil, dtype=float)
    vander = np.vander(s, N=s.size, increasing=True).T  # vander[m, j] = s_j**m
    rhs = np.zeros(s.size)
    rhs[deriv_order] = math.factorial(deriv_order)
    c = np.linalg.solve(vander, rhs)
    # The solve leaves ~1e-16 where the exact weight is 0 (the centre of a
    # symmetric first-derivative stencil); make it 0 so the sums can skip it.
    c[np.abs(c) < 1e-12 * np.abs(c).max()] = 0.0
    return c


def stencil_at(i: int, n: int, stencil: tuple[int, ...], edge_stencil: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """The stencil used at point *i* of an open axis of *n* points.

    *stencil* wherever it fits.  Near an edge, *edge_stencil* (default:
    *stencil* itself) shifted by the integer closest to zero that keeps
    every point inside ``[0, n)``.
    """
    if i + min(stencil) >= 0 and i + max(stencil) <= n - 1:
        return tuple(stencil)
    s = stencil if edge_stencil is None else edge_stencil
    k_min, k_max = -i - min(s), n - 1 - i - max(s)  # admissible shifts
    if k_min > k_max:
        raise ValueError(f"Stencil {list(s)} cannot be shifted to fit at i={i}, n={n}.")
    k = min(max(0, k_min), k_max)
    return tuple(int(sj) + k for sj in s)


def fd_apply(
    f: np.ndarray,
    h: float,
    axis: int,
    stencil: tuple[int, ...],
    deriv_order: int = 1,
    periodic: bool = False,
    edge_stencil: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Derivative of order *deriv_order* of *f* along *axis* (spacing *h*), as float64.

    Parameters
    ----------
    f : np.ndarray
        Input array, any dimensionality.
    h : float
        Grid spacing along *axis*.
    axis : int
        Axis to differentiate along.
    stencil : tuple of int
        Offsets in grid steps, e.g. ``(-2, -1, 0, 1, 2)``.
    deriv_order : int
        Order of the derivative.
    periodic : bool
        Wrap around the axis.  Otherwise the edges use :func:`stencil_at`.
    edge_stencil : tuple of int, optional
        Stencil for the points where *stencil* does not fit (open axis only).
    """
    stencil = tuple(int(s) for s in stencil)
    c = fd_coefficients(stencil, deriv_order) / float(h) ** deriv_order
    terms = [(cj, sj) for cj, sj in zip(c, stencil, strict=True) if cj != 0.0]
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[axis]

    if periodic:
        # Interior from shifted views of f itself (no padded copy, no
        # transpose); only the |s|max points at each end wrap around.
        smax = max(abs(s) for s in stencil)
        out = np.empty(f.shape)
        if n > 2 * smax:
            _views_sum(f, terms, axis, smax, n - smax, out)
        for i in [*range(0, min(smax, n)), *range(max(n - smax, smax), n)]:
            out[(slice(None),) * (axis % f.ndim) + (i,)] = sum(cj * _along(f, axis, (i + sj) % n) for cj, sj in terms)
        return out

    # Open axis.  Interior: the stencil fits, one vectorised sum.  Edges: one
    # small sum per point with that point's own (shifted) stencil.
    fm = np.moveaxis(f, axis, 0)  # derivative axis first, (n, ...)
    start, stop = -min(stencil), n - max(stencil)
    out = np.zeros(fm.shape) if stop <= start else _stencil_sum(fm, terms, start, stop)
    for i in [*range(0, max(0, start)), *range(max(0, stop), n)]:
        s_i = stencil_at(i, n, stencil, edge_stencil)
        c_i = fd_coefficients(s_i, deriv_order) / float(h) ** deriv_order
        out[i] = sum(cj * fm[i + sj] for cj, sj in zip(c_i, s_i, strict=True))
    return np.moveaxis(out, 0, axis)


def _along(a: np.ndarray, axis: int, index: int | slice) -> np.ndarray:
    """``a[..., index, ...]`` with *index* on *axis* (a view)."""
    return a[(slice(None),) * (axis % a.ndim) + (index,)]


def _views_sum(f: np.ndarray, terms: list[tuple[float, int]], axis: int, start: int, stop: int, out: np.ndarray) -> None:
    """``out[i] = sum c * f[i + s]`` over the ``(c, s)`` *terms*, for ``i`` in ``[start, stop)`` along *axis*."""
    dst = _along(out, axis, slice(start, stop))
    (c0, s0), *rest = terms
    np.multiply(_along(f, axis, slice(start + s0, stop + s0)), c0, out=dst)
    for cj, sj in rest:
        dst += cj * _along(f, axis, slice(start + sj, stop + sj))


def _stencil_sum(src: np.ndarray, terms: list[tuple[float, int]], start: int, stop: int) -> np.ndarray:
    """:func:`_views_sum` along axis 0 into a zeroed array, JIT-compiled when numba is available."""
    if _NUMBA_AVAILABLE:
        src_2d = np.ascontiguousarray(src).reshape(src.shape[0], -1)
        out_2d = np.zeros(src_2d.shape)
        c = np.array([cj for cj, _ in terms])
        offsets = np.array([sj for _, sj in terms], dtype=np.int64)
        _nb_stencil_sum(src_2d, out_2d, c, offsets, start, stop)
        return out_2d.reshape(src.shape)
    out = np.zeros(src.shape)
    _views_sum(src, terms, 0, start, stop, out)
    return out
