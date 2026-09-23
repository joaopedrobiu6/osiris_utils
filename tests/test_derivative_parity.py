"""Lazy/eager parity for Derivative_Diagnostic.

`diag[i]` (lazy, via _frame) and `diag.load_all()` (eager) must accept the same
set of configurations and produce the same numbers. They used to diverge: the
default order=4 time derivative worked lazily but raised in load_all, 'xt' was
missing from load_all entirely, and 'tx' on 1D data was rejected by _frame while
load_all accepted it.
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils as ou


class _Mock:
    """Diagnostic stand-in with data already loaded."""

    def __init__(self, data: np.ndarray, dt: float = 0.5, dx: float = 0.25, ndump: int = 1):
        self._data = data
        self._dt = dt
        self._ndump = ndump
        self._all_loaded = True
        self._name = "mock"
        self._type = "grid"
        self._dim = data.ndim - 1
        self._nx = data.shape[1:]
        self._maxiter = data.shape[0]
        self._dx = [dx] * max(1, self._dim)
        self._axis = 1

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __getitem__(self, i):
        return self._data[i]

    def _frame(self, i, data_slice=None):
        return self._data[i] if data_slice is None else self._data[i][data_slice]

    def load_all(self):
        return self._data


def _smooth_2d(nt: int = 8, n1: int = 12, n2: int = 10) -> np.ndarray:
    """Smooth analytic field over (t, x1, x2) — differentiable in every axis."""
    t = np.arange(nt)[:, None, None]
    a = np.linspace(0, 2 * np.pi, n1, endpoint=False)[None, :, None]
    b = np.linspace(0, 2 * np.pi, n2, endpoint=False)[None, None, :]
    return (np.sin(a) * np.cos(b) + 0.1 * t).astype(np.float64)


def _smooth_1d(nt: int = 8, n1: int = 16) -> np.ndarray:
    t = np.arange(nt)[:, None]
    a = np.linspace(0, 2 * np.pi, n1, endpoint=False)[None, :]
    return (np.sin(a) + 0.1 * t).astype(np.float64)


# (deriv_type, kwargs) combinations that _frame already supports.
_CONFIGS = [
    ("t", {"order": 2}),
    ("t", {"order": 4}),
    ("t", {"order": 2, "periodic": True}),
    ("t", {"order": 4, "periodic": True}),
    ("t", {"stencil": [-2, -1, 0, 1, 2], "deriv_order": 1}),
    ("x1", {"order": 2}),
    ("x1", {"order": 4}),
    ("x1", {"order": 2, "periodic": True}),
    ("x1", {"order": 4, "periodic": True}),
    ("x1", {"stencil": [-2, -1, 0, 1, 2], "deriv_order": 1}),
    ("x2", {"order": 2}),
    ("x2", {"order": 4}),
    ("xx", {"axis": (1, 2), "order": 2}),
    ("xx", {"axis": (1, 2), "order": 4}),
    ("tx", {"axis": 1, "order": 2}),
    ("tx", {"axis": 1, "order": 4}),
    ("xt", {"axis": 1, "order": 2}),
    ("xt", {"axis": 1, "order": 4}),
]


def _ids(configs):
    return [f"{t}-{'-'.join(f'{k}={v}' for k, v in kw.items())}" for t, kw in configs]


@pytest.mark.parametrize(("deriv_type", "kwargs"), _CONFIGS, ids=_ids(_CONFIGS))
def test_lazy_and_eager_agree(deriv_type: str, kwargs: dict) -> None:
    data = _smooth_2d()
    n = data.shape[0]

    lazy_diag = ou.Derivative_Diagnostic(_Mock(data), deriv_type=deriv_type, **kwargs)
    lazy = np.stack([lazy_diag[i] for i in range(n)])

    eager = ou.Derivative_Diagnostic(_Mock(data), deriv_type=deriv_type, **kwargs).load_all()

    assert eager.shape == lazy.shape
    np.testing.assert_allclose(eager, lazy, rtol=1e-10, atol=1e-12)


def test_order4_time_derivative_is_available_eagerly() -> None:
    """order=4 is the default, so load_all() must support time derivatives."""
    data = _smooth_1d()

    result = ou.Derivative_Diagnostic(_Mock(data), deriv_type="t", order=4).load_all()

    assert result.shape == data.shape


def test_mixed_derivative_on_1d_data_works_both_ways() -> None:
    """d/dt d/dx1 is meaningful on 1D data; _frame used to reject it outright."""
    data = _smooth_1d()
    n = data.shape[0]

    for deriv_type in ("tx", "xt"):
        lazy_diag = ou.Derivative_Diagnostic(_Mock(data), deriv_type=deriv_type, axis=1, order=2)
        lazy = np.stack([lazy_diag[i] for i in range(n)])
        eager = ou.Derivative_Diagnostic(_Mock(data), deriv_type=deriv_type, axis=1, order=2).load_all()

        np.testing.assert_allclose(eager, lazy, rtol=1e-10, atol=1e-12)


def test_spatial_axis_beyond_dimension_still_rejected() -> None:
    """Relaxing the 1D guard must not let x2 through on 1D data."""
    with pytest.raises(ValueError, match="along x2, but the diagnostic is 1-D"):
        ou.Derivative_Diagnostic(_Mock(_smooth_1d()), deriv_type="x2", order=2)


def test_order4_spatial_matches_analytic_derivative() -> None:
    """Guard against the restructure changing the numerics.

    Truncation error for the 4th-order stencil at dx ~ 0.098 is ~3e-6, while the
    2nd-order scheme errs by ~1.6e-3 — so this tolerance distinguishes them.
    """
    n1 = 64
    x = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    dx = x[1] - x[0]
    data = np.sin(x).reshape(1, -1)

    result = ou.Derivative_Diagnostic(_Mock(data, dx=dx), deriv_type="x1", order=4, periodic=True).load_all()

    np.testing.assert_allclose(result[0], np.cos(x), atol=1e-5)


def test_order2_spatial_is_less_accurate_than_order4() -> None:
    """Confirms the tolerance above actually discriminates between the schemes."""
    n1 = 64
    x = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    dx = x[1] - x[0]
    data = np.sin(x).reshape(1, -1)

    order2 = ou.Derivative_Diagnostic(_Mock(data, dx=dx), deriv_type="x1", order=2, periodic=True).load_all()

    assert np.max(np.abs(order2[0] - np.cos(x))) > 1e-4


# --- mixed derivatives --------------------------------------------------------
# The field in _CONFIGS, sin(x1)cos(x2) + 0.1 t, has d/dt d/dx1 = 0, so any time
# stencil passes on it.  These use f = sin(x1 + w t) cos(x2), which does not.

_OMEGA = 1.0


def _coupled_2d(nt: int = 10, n1: int = 32, n2: int = 8, dt: float = 0.1) -> tuple[np.ndarray, float, float, np.ndarray]:
    """``(f, dt, dx, exact d/dt d/dx1 f)`` for ``f = sin(x1 + w t) cos(x2)`` on a (t, x1, x2) grid."""
    dx = 2 * np.pi / n1
    t = (np.arange(nt) * dt)[:, None, None]
    x1 = (np.arange(n1) * dx)[None, :, None]
    x2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)[None, None, :]
    f = np.sin(x1 + _OMEGA * t) * np.cos(x2)
    exact = -_OMEGA * np.sin(x1 + _OMEGA * t) * np.cos(x2)
    return f, dt, dx, exact


_MIXED_CONFIGS = [
    {"axis": 1, "order": 2},
    {"axis": 1, "order": 4},
    {"axis": 1, "order": 2, "periodic": True},
    {"axis": 1, "order": 4, "periodic": True},
    {"axis": 1, "stencil": [-1, 0, 1]},
    {"axis": 1, "stencil": [-2, -1, 0, 1, 2]},
]


@pytest.mark.parametrize("kwargs", _MIXED_CONFIGS, ids=_ids([("tx", kw) for kw in _MIXED_CONFIGS]))
def test_tx_and_xt_are_the_same_operator(kwargs: dict) -> None:
    """d/dt d/dx1 = d/dx1 d/dt exactly: the two stencils act on different axes.

    The lazy 'tx' used to carry a time stencil of its own, which ignored
    periodic= and stencil= and divided the one-sided end differences by 2 dt.
    """
    f, dt, dx, _ = _coupled_2d()
    n = f.shape[0]

    def lazy(deriv_type: str) -> np.ndarray:
        diag = ou.Derivative_Diagnostic(_Mock(f, dt=dt, dx=dx), deriv_type=deriv_type, **kwargs)
        return np.stack([diag[i] for i in range(n)])

    eager = ou.Derivative_Diagnostic(_Mock(f, dt=dt, dx=dx), deriv_type="tx", **kwargs).load_all()

    np.testing.assert_allclose(lazy("tx"), lazy("xt"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lazy("tx"), eager, rtol=1e-10, atol=1e-12)


def test_mixed_derivative_matches_its_closed_form() -> None:
    """Default order=4 'tx' against the exact -w sin(x1 + w t) cos(x2), end frames included."""
    f, dt, dx, exact = _coupled_2d()
    diag = ou.Derivative_Diagnostic(_Mock(f, dt=dt, dx=dx), deriv_type="tx", axis=1)
    got = np.stack([diag[i] for i in range(f.shape[0])])

    inner = (slice(None), slice(2, -2))  # x1 edges fall back to 2nd order
    # Interior frames: 4th order in t and x1, error ~ (w dt)^4, (k dx)^4 ~ 1e-4.
    np.testing.assert_allclose(got[2:-2][inner], exact[2:-2][inner], atol=1e-3 * _OMEGA)
    # End frames: 2nd-order one-sided in t, error ~ (w dt)^2 / 3 ~ 3e-3.  The old
    # code returned (f1 - f0) / (2 dt) here, about half the true derivative.
    np.testing.assert_allclose(got[[0, -1]][inner], exact[[0, -1]][inner], atol=1e-2 * _OMEGA)
