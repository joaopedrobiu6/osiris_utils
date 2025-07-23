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


@pytest.mark.parametrize(
    "n_cells, ppc, t_steps, n_cpu, push_time, hours, expected",
    [
        (1000, 10, 100, 4, 1e-7, False, 0.0025),
        (1000, 10, 100, 4, 1e-7, True, 0.0000006944444444444445),
        (2000, 20, 200, 8, 2e-7, False, 0.005),
        (2000, 20, 200, 8, 2e-7, True, 0.0013888888888888889),
    ],
)
def test_time_estimation(n_cells, ppc, t_steps, n_cpu, push_time, hours, expected):
    from osiris_utils.utils import time_estimation

    estimated_time = time_estimation(n_cells, ppc, t_steps, n_cpu, push_time, hours)
    assert pytest.approx(estimated_time, rel=1e-9) == expected


@pytest.mark.parametrize(
    "n_gridpoints, expected",
    [
        (1000, 0.003814697265625),
        (2000, 0.00762939453125),
        (5000, 0.019073486328125),
    ],
)
def test_filesize_estimation(n_gridpoints, expected):
    from osiris_utils.utils import filesize_estimation

    estimated_size = filesize_estimation(n_gridpoints)
    assert pytest.approx(estimated_size, rel=1e-9) == expected
