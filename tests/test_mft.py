import numpy as np
import pytest

import osiris_utils.postprocessing.mft as mft


class MockDiagnostic:
    def __init__(self, data, dim=2, name="mock_diag", dx=1.0):
        self._data = data
        self._dim = dim  # Need dim >= axis
        self._name = name
        self._all_loaded = True
        self._dx = [dx] * dim
        self._dt = 1.0
        self._ndump = 1
        self._axis = 1
        self._nx = data.shape[1:]
        self._maxiter = data.shape[0]
        self._type = "grid"
        self._simulation_folder = "/tmp"
        self._species = None

    @property
    def data(self):
        return self._data

    def load_all(self):
        return self._data

    def _frame(self, index, data_slice=None):
        return self._data[index]


def test_mft_average():
    # 2D data (t, x1, x2). MFT along x2 (axis 2).
    # Shape: (1, 2, 4).
    # x1 has 2 points, x2 has 4 points.
    data = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])  # Shape (1, 2, 4)

    mock = MockDiagnostic(data, dim=2)

    # MFT Diagnostic
    mft_diag = mft.MFT_Diagnostic(mock, mft_axis=2)

    # Get 'avg' component
    avg_diag = mft_diag['avg']
    res = avg_diag.load_all()

    # Expected: mean along axis 2 (the last axis here, since loaded data has axis 0=t, 1=x1, 2=x2)
    # Row 1: mean(1,2,3,4) = 2.5
    # Row 2: mean(5,6,7,8) = 6.5
    # Result shape should be (1, 2, 1) due to keepdims=True

    assert res.shape == (1, 2, 1)
    assert np.isclose(res[0, 0, 0], 2.5)
    assert np.isclose(res[0, 1, 0], 6.5)


def test_mft_fluctuations():
    # Same data
    data = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    mock = MockDiagnostic(data, dim=2)
    mft_diag = mft.MFT_Diagnostic(mock, mft_axis=2)

    delta_diag = mft_diag['delta']
    res = delta_diag.load_all()

    # Expected: val - mean
    # Row 1: [1-2.5, 2-2.5, 3-2.5, 4-2.5] = [-1.5, -0.5, 0.5, 1.5]

    assert res.shape == (1, 2, 4)
    expected_row1 = np.array([-1.5, -0.5, 0.5, 1.5])
    assert np.allclose(res[0, 0, :], expected_row1)

    # Verify sum of delta is 0
    assert np.allclose(np.sum(res, axis=2), 0.0)


def test_mft_validation():
    mock = MockDiagnostic(np.zeros((1, 2)), dim=1)

    # Invalid axis
    with pytest.raises(ValueError, match="mft_axis must be in"):
        mft.MFT_Diagnostic(mock, mft_axis=2)

    # Invalid component key
    mft_diag = mft.MFT_Diagnostic(mock, mft_axis=1)
    with pytest.raises(ValueError, match="Invalid MFT component"):
        _ = mft_diag['invalid_key']
