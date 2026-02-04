import numpy as np
import pytest

from osiris_utils.postprocessing.derivative import Derivative_Diagnostic
from osiris_utils.postprocessing.fft import FFT_Diagnostic


class MockDiagnostic:
    def __init__(self, data, dt=1.0, dx=1.0, axis=1, ndump=1):
        self._data = data
        self._dt = dt
        self._dx = dx
        self._axis = axis  # integer usually
        self._ndump = ndump
        self._name = "MockDiag"
        self._all_loaded = True  # Mocking that data is loaded

        # Dimensions derived from data shape
        self._dim = data.ndim - 1  # assuming first dim is time
        self._nx = data.shape[1:]
        self._maxiter = data.shape[0]
        self._type = "grid"  # or particles

        # Handle _dx as list/array always if expected by consumers
        if isinstance(dx, (list, tuple, np.ndarray)):
            self._dx = dx
        else:
            # If scalar, make it a list with repeated values or single
            # For 1D, it needs to be subscriptable for fft.py:231
            self._dx = [dx] * max(1, self._dim)

    @property
    def data(self):
        return self._data

    def __getitem__(self, index):
        return self._data[index]

    def load_all(self):
        return self._data


def test_derivative_t():
    # t = 0..9. f(t) = t^2. df/dt = 2t.
    # ndump = 1. dt = 1.
    t = np.arange(10)
    data = t**2
    # Diagnostic expects time as axis 0. Let's say it's 1D data changing in time.
    # shape (10, 1).
    data = data.reshape(-1, 1)

    mock = MockDiagnostic(data, dt=1.0, ndump=1, dx=1.0, axis=1)

    # Compute derivative in 't'
    deriv = Derivative_Diagnostic(mock, deriv_type='t', order=2)
    res = deriv.load_all()

    # Gradient uses central difference interior, one-sided edges.
    # Interior: (f(x+h) - f(x-h)) / 2h.
    # t=1: (4 - 0) / 2 = 2. Correct (2*1).
    # t=2: (9 - 1) / 2 = 4. Correct (2*2).
    # Check interior points
    expected = 2 * t
    # Gradient accuracy might vary at edges or due to discretization but for t^2 central difference is exact?
    # f(t+1) - f(t-1) = (t+1)^2 - (t-1)^2 = (t^2+2t+1) - (t^2-2t+1) = 4t. / 2 = 2t. Yes.

    assert np.allclose(res[1:-1, 0], expected[1:-1])


def test_derivative_x():
    # f(x) = x^2. x along axis 1.
    x = np.arange(10)
    data = x**2
    # shape (1, 10). 1 time step.
    data = data.reshape(1, -1)

    mock = MockDiagnostic(data, dt=1.0, ndump=1, dx=1.0, axis=1)  # dim 1

    deriv = Derivative_Diagnostic(mock, deriv_type='x1')
    res = deriv.load_all()

    expected = 2 * x
    assert np.allclose(res[0, 1:-1], expected[1:-1])


def test_derivative_errors():
    mock = MockDiagnostic(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Derivative_Diagnostic(mock, deriv_type='invalid').load_all()


def test_fft_simple():
    # Simple sine wave in spatial direction x1.
    # f(x) = sin(k0 * x). FFT should have peak at k0.
    L = 2 * np.pi
    N = 64
    dx = L / N
    x = np.arange(N) * dx  # 0 .. 2pi

    # k0 = 3. 3 wavelengths in box.
    k0 = 3.0
    data = np.sin(k0 * x)
    data = data.reshape(1, -1)  # (1, 64)

    mock = MockDiagnostic(data, dt=1.0, dx=dx, axis=1, ndump=1)

    # FFT along axis 1 (spatial)
    # FFT_Diagnostic constructor takes fft_axis.
    # If 1D spatial -> axis 1 is x1. (Axis 0 is time).
    fft_diag = FFT_Diagnostic(mock, fft_axis=1)
    res = fft_diag.load_all()  # Returns |FFT|^2 shifted.

    # Check frequency array
    # k() returns wavenumbers.
    freqs = fft_diag.k(1)

    # Peak should be at k0 = +/- 3.
    # Find index of max
    max_idx = np.argmax(res[0])
    peak_freq = freqs[max_idx]

    # Depending on resolution, should be close to 3 or -3.
    # Since real input, symmetric.
    assert np.isclose(abs(peak_freq), 3.0)
