"""Regression tests for Derivative result caching.

load_all() used to park the *source* array in self._data before computing, so
any failure mid-computation left the undifferentiated input in the result slot
and the next call returned it as a "cached derivative".
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils as ou


class _Mock:
    """Minimal Diagnostic stand-in with data already loaded."""

    def __init__(self, data: np.ndarray, dt: float = 1.0, dx: float = 1.0, ndump: int = 1):
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


@pytest.fixture
def quadratic() -> np.ndarray:
    """f(t) = t^2 sampled at t = 0..9, one spatial point."""
    return (np.arange(10.0) ** 2).reshape(-1, 1)


def test_failed_load_all_does_not_cache_source_data(quadratic: np.ndarray) -> None:
    # 'xx' with axis=None fails validation partway through load_all()
    deriv = ou.Derivative_Diagnostic(_Mock(quadratic), deriv_type="xx", axis=None, order=2)

    with pytest.raises(ValueError, match="Axis must be a tuple"):
        deriv.load_all()

    # The retry must fail the same way, not hand back the untouched input.
    with pytest.raises(ValueError, match="Axis must be a tuple"):
        deriv.load_all()


def test_failed_load_all_leaves_data_unset(quadratic: np.ndarray) -> None:
    deriv = ou.Derivative_Diagnostic(_Mock(quadratic), deriv_type="xx", axis=None, order=2)

    with pytest.raises(ValueError):
        deriv.load_all()

    assert deriv._data is None
    assert deriv._all_loaded is False


def test_successful_load_all_caches_the_derivative(quadratic: np.ndarray) -> None:
    deriv = ou.Derivative_Diagnostic(_Mock(quadratic), deriv_type="t", order=2)

    first = deriv.load_all()
    second = deriv.load_all()

    # d/dt of t^2 is 2t; np.gradient is exact for a quadratic including edges
    np.testing.assert_allclose(first[:, 0], 2 * np.arange(10.0))
    assert second is first
    assert deriv._all_loaded is True


def test_load_all_does_not_alias_the_source_array(quadratic: np.ndarray) -> None:
    """The result must never be the same buffer as the input."""
    mock = _Mock(quadratic)
    deriv = ou.Derivative_Diagnostic(mock, deriv_type="t", order=2)

    result = deriv.load_all()

    assert result is not mock._data
    np.testing.assert_allclose(mock._data[:, 0], np.arange(10.0) ** 2)
