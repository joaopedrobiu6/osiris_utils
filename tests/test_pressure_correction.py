import numpy as np
import pytest

import osiris_utils.postprocessing.pressure_correction as pc
from osiris_utils.data.diagnostic import Diagnostic


def _array_diag(data: np.ndarray, name: str) -> Diagnostic:
    """A Diagnostic whose frames are the rows of *data* (time first)."""
    d = Diagnostic()
    d._name = name
    d._dim = data.ndim - 1
    d._maxiter = data.shape[0]
    d._frame = lambda i, data_slice=None: data[i]
    return d


def test_pressure_correction_math():
    # P_corr = P - n * u * v

    # 1 timestep, 1 data point for simplicity

    P_data = np.array([[10.0]])
    n_data = np.array([[2.0]])
    u_data = np.array([[3.0]])
    v_data = np.array([[4.0]])

    mock_P = _array_diag(P_data, "P12")  # Needs to be in OSIRIS_P
    mock_n = _array_diag(n_data, "rho")
    mock_u = _array_diag(u_data, "ufl1")
    mock_v = _array_diag(v_data, "vfl2")

    corrector = pc.PressureCorrection_Diagnostic(mock_P, mock_n, mock_u, mock_v)
    res = corrector.load_all()

    # Expected: 10 - 2 * 3 * 4 = 10 - 24 = -14
    assert np.isclose(res[0, 0], -14.0)


def test_pressure_correction_validation():
    mock_P = _array_diag(np.zeros((1, 1)), "bad_name")
    mock_n = _array_diag(np.zeros((1, 1)), "n")
    mock_u = _array_diag(np.zeros((1, 1)), "u")
    mock_v = _array_diag(np.zeros((1, 1)), "v")

    with pytest.raises(ValueError, match="Invalid pressure component"):
        pc.PressureCorrection_Diagnostic(mock_P, mock_n, mock_u, mock_v)
