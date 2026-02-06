import numpy as np
import pytest

import osiris_utils.postprocessing.pressure_correction as pc


class MockDiagnostic:
    def __init__(self, data, name="mock_diag", dim=1):
        self._data = data
        self._name = name
        self._dim = dim
        self._all_loaded = True
        self._simulation_folder = "/tmp"
        self._species = None

    @property
    def data(self):
        return self._data

    def load_all(self):
        return self._data

    def _frame(self, index, data_slice=None):
        return self._data[index]


def test_pressure_correction_math():
    # P_corr = P - n * u * v

    # 1 timestep, 1 data point for simplicity

    P_data = np.array([[10.0]])
    n_data = np.array([[2.0]])
    u_data = np.array([[3.0]])
    v_data = np.array([[4.0]])

    mock_P = MockDiagnostic(P_data, "P12")  # Needs to be in OSIRIS_P
    mock_n = MockDiagnostic(n_data, "rho")
    mock_u = MockDiagnostic(u_data, "ufl1")
    mock_v = MockDiagnostic(v_data, "vfl2")

    corrector = pc.PressureCorrection_Diagnostic(mock_P, mock_n, mock_u, mock_v)
    res = corrector.load_all()

    # Expected: 10 - 2 * 3 * 4 = 10 - 24 = -14
    assert np.isclose(res[0, 0], -14.0)


def test_pressure_correction_validation():
    mock_P = MockDiagnostic(np.zeros((1, 1)), "bad_name")
    mock_n = MockDiagnostic(np.zeros((1, 1)), "n")
    mock_u = MockDiagnostic(np.zeros((1, 1)), "u")
    mock_v = MockDiagnostic(np.zeros((1, 1)), "v")

    with pytest.raises(ValueError, match="Invalid pressure component"):
        pc.PressureCorrection_Diagnostic(mock_P, mock_n, mock_u, mock_v)
