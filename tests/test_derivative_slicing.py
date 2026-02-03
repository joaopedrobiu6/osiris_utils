import numpy as np

import osiris_utils as ou


class MockDiagnostic:
    def __init__(self, data, dt=1.0, dx=1.0, axis=1, ndump=1):
        self._data = data
        self._dt = dt
        self._dx = dx
        self._axis = axis
        self._ndump = ndump
        self._name = "MockDiag"
        self._all_loaded = False  # Simulate not loaded to force slicing logic
        self._simulation_folder = "/tmp"

        self._dim = data.ndim - 1
        self._nx = data.shape[1:]
        self._maxiter = data.shape[0]
        self._type = "grid"

        if isinstance(dx, (list, tuple, np.ndarray)):
            self._dx = dx
        else:
            self._dx = [dx] * max(1, self._dim)

    def __getitem__(self, index):
        # Simulate loading from disk by slicing the in-memory array
        return self._data[index]

    def _data_generator(self, index, data_slice=None):
        # Not used by Derivative_Diagnostic directly in _read_and_compute
        # But required by Diagnostic interface
        pass


def test_derivative_slicing_boundary():
    print("Testing Derivative Slicing at Boundaries")

    # 10 time steps, 1D space (1 value per time step for simplicity to test time derivative)
    t = np.arange(10)
    data = (t**2).reshape(-1, 1)  # f(t) = t^2

    mock = MockDiagnostic(data, dt=1.0)

    # Time derivative
    deriv = ou.Derivative_Diagnostic(mock, deriv_type='t', order=2)

    # Test index 0 (Boundary) - previously crashed
    print("Accessing index 0...")
    res0 = deriv[0]
    print(f"Result at 0: {res0}")

    # Expected: 2*t at t=0 -> 0 using forward difference of 2nd order
    # With 3 points: 0, 1, 4. 2nd order forward diff at 0: (-3*0 + 4*1 - 4)/2 = 0. Correct.
    assert res0.shape == (1,)
    assert np.isclose(res0[0], 0.0)

    # Test index 1 (Internal but close to boundary)
    print("Accessing index 1...")
    res1 = deriv[1]
    # Expected: 2.0
    assert np.isclose(res1[0], 2.0)

    # Test last index (Boundary)
    print("Accessing last index...")
    res_last = deriv[9]  # t=9. 2*9=18.
    # Backward difference 2nd order.
    assert np.isclose(res_last[0], 18.0)

    print("Passed boundary tests!")


def test_derivative_slicing_spatial():
    print("Testing Derivative Slicing Spatial")
    # 1 time step, 10 spatial points. f(x) = x^3
    x = np.linspace(0, 10, 11)  # 0, 1, ..., 10
    dx = 1.0
    data = (x**2).reshape(1, -1)

    mock = MockDiagnostic(data, dx=dx, dt=1.0)

    # Spatial derivative x1
    deriv = ou.Derivative_Diagnostic(mock, deriv_type='x1', order=2)

    # Access time 0, spatial slice [0:1] (Boundary)
    # This involves derivative along axis 1.
    # _read_and_compute will be called with time_idx=0.

    # If we ask for deriv[0], we get full spatial array.
    res = deriv[0]
    expected = 2 * x
    assert np.allclose(res, expected)

    # If we ask for deriv[0, 0:1] (Just first pixel)
    # The system should load enough spatial neighbors.
    res_pixel = deriv[0, 0:1]
    assert np.isclose(res_pixel[0], expected[0])

    # Last pixel
    res_last = deriv[0, 10:11]
    assert np.isclose(res_last[0], expected[10])

    print("Passed spatial slicing tests!")
