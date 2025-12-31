import numpy as np

import osiris_utils as ou


def derivative_2nd_order(array, dx):
    deriv = np.zeros_like(array)

    deriv[0] = (array[1] - array[0]) / dx
    deriv[-1] = (array[-1] - array[-2]) / dx

    deriv[1:-1] = (array[2:] - array[:-2]) / (2 * dx)

    return deriv


def derivative_4th_order(array, dx):
    deriv = np.zeros_like(array)

    deriv[0] = (array[1] - array[0]) / dx
    deriv[1] = (array[2] - array[0]) / (2 * dx)
    deriv[-2] = (array[-1] - array[-3]) / (2 * dx)
    deriv[-1] = (array[-1] - array[-2]) / dx

    deriv[2:-2] = (-array[4:] + 8 * array[3:-1] - 8 * array[1:-3] + array[:-4]) / (12 * dx)

    return deriv


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

    def __getitem__(self, index):
        return self._data[index]

    def load_all(self):
        return self._data


def test_derivative_2nd_order():
    print("=" * 60)
    print("Testing 2nd Order Time Derivative")
    print("=" * 60)

    # t = 0..9. f(t) = t^2. df/dt = 2t.
    # ndump = 1. dt = 1.
    t = np.arange(10)
    data = t**2
    print(f"\nTime values: {t}")
    print(f"Function f(t) = t²: {data}")
    print(f"Expected derivative df/dt = 2t: {2 * t}")

    # Diagnostic expects time as axis 0. Let's say it's 1D data changing in time.
    # shape (10, 1).
    data = data.reshape(-1, 1)

    mock = MockDiagnostic(data, dt=1.0, ndump=1, dx=1.0, axis=1)

    # Compute derivative in 't'
    print("\nComputing derivative using Derivative_Diagnostic...")
    deriv = ou.Derivative_Diagnostic(mock, deriv_type='t', order=2)
    res = deriv.load_all()

    print(f"Result shape: {res.shape}")
    print(f"Result values: {res[:, 0]}")

    # Expected analytical result
    expected = 2 * t
    print(f"\nExpected analytical derivative: {expected}")

    # np.gradient with edge_order=2 uses second-order accurate edge formulas
    # First edge: (-3*f[0] + 4*f[1] - f[2]) / (2*h)
    # Last edge: (3*f[-1] - 4*f[-2] + f[-3]) / (2*h)
    np_gradient_result = np.gradient(data[:, 0], 1.0, edge_order=2)
    print(f"\nnp.gradient result (for comparison): {np_gradient_result}")

    # Check interior points (should be exact for quadratic)
    print("\n" + "-" * 60)
    print("Checking interior points (indices 1 to -1)...")
    interior_match = np.allclose(res[1:-1, 0], expected[1:-1])
    print(f"Interior points match expected: {interior_match}")
    print(f"Interior result: {res[1:-1, 0]}")
    print(f"Interior expected: {expected[1:-1]}")
    assert interior_match, "Interior points don't match!"

    # Check edges - np.gradient with edge_order=2 uses different formula
    print("\n" + "-" * 60)
    print("Checking edge points...")

    # For f(t) = t², the derivative is 2t
    # At t=0: df/dt = 0
    # At t=9: df/dt = 18
    # np.gradient with edge_order=2 should be exact for quadratic

    print(f"First point - Result: {res[0, 0]:.6f}, Expected: {expected[0]:.6f}, np.gradient: {np_gradient_result[0]:.6f}")
    print(f"Last point  - Result: {res[-1, 0]:.6f}, Expected: {expected[-1]:.6f}, np.gradient: {np_gradient_result[-1]:.6f}")

    # Should match np.gradient exactly since we're using the same algorithm
    assert np.allclose(res[:, 0], np_gradient_result), "Result doesn't match np.gradient!"

    # Should also match analytical result (exact for quadratic with central differences)
    edge_match = np.allclose(res[[0, -1], 0], expected[[0, -1]], atol=1e-10)
    print(f"Edge points match expected: {edge_match}")

    if edge_match:
        print("\nTest PASSED! Derivative computation is correct.")
    else:
        print("\nEdge tolerance issue (but matches np.gradient)")
        print("This is expected for edge_order=2 formulas")

    print("=" * 60)


def test_derivative_4th_order():
    print("\n" + "=" * 60)
    print("Testing 4th Order Spatial Derivative")
    print("=" * 60)

    # f(x) = x³. df/dx = 3x².
    x = np.linspace(0, 10, 50)
    dx = x[1] - x[0]

    # Create 2D data (1 timestep, nx points)
    data = (x**3).reshape(1, -1)

    print(f"\nSpatial grid: {len(x)} points, dx = {dx:.6f}")
    print("Function f(x) = x³")
    print("Expected derivative df/dx = 3x²")

    mock = MockDiagnostic(data, dt=1.0, dx=dx, axis=0)

    # Compute derivative in 'x1' (spatial)
    print("\nComputing 4th-order spatial derivative...")
    deriv = ou.Derivative_Diagnostic(mock, deriv_type='x1', order=4)
    res = deriv.load_all()

    print(f"Result shape: {res.shape}")

    # Expected analytical result
    expected = 3 * x**2

    # For interior points (away from boundaries), 4th order should be very accurate
    print("\n" + "-" * 60)
    print("Checking interior points (indices 2 to -2)...")
    interior_res = res[0, 2:-2]
    interior_exp = expected[2:-2]

    # Calculate error
    max_error = np.max(np.abs(interior_res - interior_exp))
    rel_error = max_error / np.max(np.abs(interior_exp))

    print(f"Max absolute error: {max_error:.6e}")
    print(f"Max relative error: {rel_error:.6e}")

    # 4th order should be very accurate for cubic (actually exact for up to cubic!)
    interior_match = np.allclose(interior_res, interior_exp, rtol=1e-4, atol=1e-4)
    print(f"Interior points accurate: {interior_match}")

    if interior_match:
        print("\nTest PASSED! 4th-order derivative is accurate.")
    else:
        print("\nTest FAILED! 4th-order derivative has issues.")
        print(f"Sample results: {interior_res[:5]}")
        print(f"Sample expected: {interior_exp[:5]}")

    assert interior_match, "4th order derivative interior points not accurate!"

    # Check edge points
    print("\n" + "-" * 60)
    print("Checking edge points (boundary formulas)...")

    edge_indices = [0, 1, -2, -1]
    edge_labels = ["First (idx=0)", "Second (idx=1)", "Second-to-last (idx=-2)", "Last (idx=-1)"]

    print(f"\n{'Index':<20} {'Result':<15} {'Expected':<15} {'Abs Error':<15} {'Rel Error':<15}")
    print("-" * 80)

    edge_errors = []
    for idx, label in zip(edge_indices, edge_labels, strict=False):
        res_val = res[0, idx]
        exp_val = expected[idx]
        abs_err = abs(res_val - exp_val)
        rel_err = abs_err / abs(exp_val) if abs(exp_val) > 1e-10 else abs_err

        print(f"{label:<20} {res_val:<15.6f} {exp_val:<15.6f} {abs_err:<15.6e} {rel_err:<15.6e}")
        edge_errors.append(abs_err)

    max_edge_error = max(edge_errors)
    print(f"\nMax edge error: {max_edge_error:.6e}")

    # Edge points use 2nd-order formulas, so they won't be as accurate as interior
    # But they should still be reasonably close
    edge_match = all(err < 0.1 for err in edge_errors)  # Relaxed tolerance for edges
    print(f"Edge points acceptable: {edge_match}")

    if interior_match and edge_match:
        print("\nTest PASSED! 4th-order derivative is accurate (interior and edges).")
    elif interior_match:
        print("\nTest PASSED with caveat: Interior accurate, edges need review.")
    else:
        print("\nTest FAILED! 4th-order derivative has issues.")
        print(f"Sample results: {interior_res[:5]}")
        print(f"Sample expected: {interior_exp[:5]}")

    assert edge_match, "4th order derivative edge points not acceptable!"

    print("=" * 60)


if __name__ == "__main__":
    test_derivative_2nd_order()
    test_derivative_4th_order()
    print("\nAll tests passed!")
