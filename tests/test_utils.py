# tests/test_utils.py
import math

import numpy as np
import pytest

from osiris_utils.utils import courant2D, transverse_average


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (1.0, 1.0, 1.0 / math.sqrt(1.0 / 1.0**2 + 1.0 / 1.0**2)),
        (2.0, 2.0, 1.0 / math.sqrt(1.0 / 2.0**2 + 1.0 / 2.0**2)),
        (0.5, 1.0, 1.0 / math.sqrt(1.0 / 0.5**2 + 1.0 / 1.0**2)),
    ],
)
def test_courant2D(dx, dy, expected):
    dt = courant2D(dx, dy)
    assert pytest.approx(dt, rel=1e-9) == expected


def test_courant2D_bad_input():
    with pytest.raises(TypeError):
        # passing a non-float should error
        _ = courant2D("a", 1.0)


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([2.0, 5.0])),
        (np.ones((5, 4)), np.ones(5)),
    ],
)
def test_transverse_average(data, expected):
    out = transverse_average(data)
    # shape and values
    assert isinstance(out, np.ndarray)
    assert out.shape == (data.shape[0],)
    assert np.allclose(out, expected)


def test_transverse_average_bad_dim():
    with pytest.raises(ValueError):
        transverse_average(np.zeros((3, 3, 3)))
