"""The shared finite-difference kernel, against closed forms."""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils.finite_differences as fd
from osiris_utils.finite_differences import fd_apply, fd_coefficients, stencil_at


def test_coefficients_of_the_standard_stencils():
    np.testing.assert_array_equal(fd_coefficients((-1, 0, 1), 1), [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(fd_coefficients((-2, -1, 0, 1, 2), 1), np.array([1, -8, 0, 8, -1]) / 12, atol=1e-15)
    assert fd_coefficients((-2, -1, 0, 1, 2), 1)[2] == 0.0  # exactly, so the sums skip it
    np.testing.assert_allclose(fd_coefficients((-1, 0, 1), 2), [1.0, -2.0, 1.0], atol=1e-14)


@pytest.mark.parametrize("bad", [(), (0, 0, 1), (0,)])
def test_coefficients_refuse_a_bad_stencil(bad):
    with pytest.raises(ValueError):
        fd_coefficients(bad, 1)


def test_stencil_is_shifted_inwards_at_the_edges():
    s5, s3 = (-2, -1, 0, 1, 2), (-1, 0, 1)
    assert stencil_at(5, 10, s5) == s5
    assert stencil_at(0, 10, s5) == (0, 1, 2, 3, 4)
    assert stencil_at(1, 10, s5) == (-1, 0, 1, 2, 3)
    assert stencil_at(9, 10, s5) == (-4, -3, -2, -1, 0)
    # an explicit edge stencil replaces the main one where that does not fit
    assert stencil_at(0, 10, s5, s3) == (0, 1, 2)
    assert stencil_at(1, 10, s5, s3) == (-1, 0, 1)
    with pytest.raises(ValueError, match="cannot be shifted"):
        stencil_at(0, 3, s5)


def test_open_axis_is_exact_for_polynomials_edges_included():
    """A 5-point stencil (shifted at the edges) differentiates any quartic exactly."""
    x = np.linspace(0.0, 2.0, 11)
    h = x[1] - x[0]
    f = x**4 - 3 * x**3 + x
    np.testing.assert_allclose(fd_apply(f, h, 0, (-2, -1, 0, 1, 2)), 4 * x**3 - 9 * x**2 + 1, atol=1e-11)
    np.testing.assert_allclose(fd_apply(f, h, 0, (-2, -1, 0, 1, 2), deriv_order=2), 12 * x**2 - 18 * x, atol=1e-9)
    # the legacy order-4 scheme's 3-point edges are exact for a quadratic
    np.testing.assert_allclose(fd_apply(x**2, h, 0, (-2, -1, 0, 1, 2), edge_stencil=(-1, 0, 1)), 2 * x, atol=1e-12)


def test_periodic_axis_on_a_fourier_mode():
    """The centred stencil turns k into sin(k h) / h, exactly; any axis of an N-D array."""
    n, k = 16, 3
    h = 2 * np.pi / n
    x = np.arange(n) * h
    f2d = np.sin(k * x)[None, :] * np.ones((4, 1))
    got = fd_apply(f2d, h, 1, (-1, 0, 1), periodic=True)
    np.testing.assert_allclose(got, np.sin(k * h) / h * np.cos(k * x)[None, :] * np.ones((4, 1)), atol=1e-13)


@pytest.mark.skipif(not fd._NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_and_numpy_paths_agree(monkeypatch):
    f = np.random.default_rng(0).standard_normal((40, 7, 3))
    jit = fd_apply(f, 0.3, 0, (-2, -1, 0, 1, 2))
    monkeypatch.setattr(fd, "_NUMBA_AVAILABLE", False)
    np.testing.assert_allclose(fd_apply(f, 0.3, 0, (-2, -1, 0, 1, 2)), jit, rtol=1e-13, atol=1e-13)
