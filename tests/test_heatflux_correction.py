import numpy as np
import pytest

import osiris_utils.postprocessing.heatflux_correction as hfc


class MockDiagnostic:
    def __init__(self, data, name="mock_diag", dim=1, species_name="electron"):
        self._data = data
        self._name = name
        self._dim = dim
        self._all_loaded = True

        # For heatflux, it checks self._species
        class Species:
            def __init__(self, name):
                self._name = name

        self._species = Species(species_name) if species_name else None
        self._simulation_folder = "/tmp"

    @property
    def data(self):
        return self._data

    def load_all(self):
        return self._data

    def _frame(self, index, data_slice=None):
        return self._data[index]


def test_heatflux_correction_math():
    # Formula: 2*q - vfl_i * trace_P - 2 * (vfl dot Pji)

    # 1D case (dim=1)
    # i=1 (q1)
    # trace_P = P11
    # vfl_dot_Pji = vfl1 * P11
    # res = 2*q1 - vfl1 * P11 - 2 * (vfl1 * P11)
    #     = 2*q1 - 3 * vfl1 * P11

    q_data = np.array([[100.0]])
    vfl1_data = np.array([[2.0]])
    P11_data = np.array([[5.0]])

    mock_q = MockDiagnostic(q_data, "q1", dim=1)
    mock_vfl1 = MockDiagnostic(vfl1_data, "vfl1", dim=1)
    mock_P11 = MockDiagnostic(P11_data, "P11", dim=1)

    # In 1D:
    # Pjj_list = [P11]
    # vfl_j_list = [vfl1]
    # Pji_list = [P11] (Since i=1, j=1 -> P11)

    corrector = hfc.HeatfluxCorrection_Diagnostic(
        mock_q,
        mock_vfl1,
        [mock_P11],  # Pjj_list
        [mock_vfl1],  # vfl_j_list
        [mock_P11],  # Pji_list
    )

    res = corrector.load_all()

    # Expected:
    # q = 100
    # vfl_i = 2
    # trace_P = 5
    # vfl_dot_Pji = 2 * 5 = 10
    # term1 = 2 * 100 = 200
    # term2 = 2 * 5 = 10
    # term3 = 2 * 10 = 20
    # result = 200 - 10 - 20 = 170

    assert np.isclose(res[0, 0], 170.0)


def test_heatflux_correction_validation():
    mock_q = MockDiagnostic(np.zeros((1, 1)), "bad_name")
    # Need to catch ValueError for name check or other checks
    # The code checks self._name in OSIRIS_H ["q1", "q2", "q3"]

    # Also need to provide valid species or it might fail elsewhere if Diagnostic checks species

    with pytest.raises(ValueError, match="Invalid heatflux component"):
        hfc.HeatfluxCorrection_Diagnostic(mock_q, mock_q, [], [], [])
