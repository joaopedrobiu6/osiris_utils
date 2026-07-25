"""Non-periodic boundary handling for FieldCentering.

The Yee-mesh centering averages each cell with its left neighbour. The default
uses np.roll, which wraps — correct only for periodic runs, and documented as
such. With periodic=False the first cell along a centred axis has no left
neighbour, so it must not borrow the value from the far boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import osiris_utils.postprocessing.field_centering as fc


class _Mock:
    def __init__(self, data: np.ndarray, name: str, dim: int = 1, dx: float = 1.0, dt: float = 1.0):
        self._data = data
        self._name = name
        self._dim = dim
        self._dx = [dx] * dim if dim > 1 else dx
        self._dt = dt
        self._all_loaded = True
        self._nx = data.shape[1:]
        self._maxiter = data.shape[0]
        self._simulation_folder = "/tmp"
        self._species = None

    @property
    def data(self):
        return self._data

    def __getitem__(self, index):
        return self._data[index]

    def load_all(self):
        return self._data

    def _frame(self, index, data_slice=None):
        return self._data[index]


def test_non_periodic_does_not_wrap_the_first_cell() -> None:
    data = np.array([0.0, 2.0, 4.0, 6.0]).reshape(1, 4)

    centered = fc.FieldCentering_Diagnostic(_Mock(data, "e1"), periodic=False).load_all()

    # periodic would give 0.5*(0 + 6) = 3.0 here
    assert centered[0, 0] == pytest.approx(0.0)
    np.testing.assert_allclose(centered[0], [0.0, 1.0, 3.0, 5.0])


def test_non_periodic_matches_periodic_in_the_interior() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((2, 12))

    periodic = fc.FieldCentering_Diagnostic(_Mock(data, "e1"), periodic=True).load_all()
    non_periodic = fc.FieldCentering_Diagnostic(_Mock(data, "e1"), periodic=False).load_all()

    np.testing.assert_allclose(periodic[:, 1:], non_periodic[:, 1:])
    assert not np.allclose(periodic[:, 0], non_periodic[:, 0])


def test_default_is_periodic_and_unchanged() -> None:
    """The existing contract must not shift under anyone's feet."""
    data = np.array([0.0, 2.0, 4.0, 6.0]).reshape(1, 4)

    default = fc.FieldCentering_Diagnostic(_Mock(data, "e1")).load_all()

    np.testing.assert_allclose(default[0], 0.5 * (data[0] + np.roll(data[0], 1)))


def test_non_periodic_lazy_matches_eager() -> None:
    rng = np.random.default_rng(1)
    data = rng.random((4, 10))

    diag = fc.FieldCentering_Diagnostic(_Mock(data, "e1"), periodic=False)
    lazy = np.stack([diag[i] for i in range(4)])
    eager = fc.FieldCentering_Diagnostic(_Mock(data, "e1"), periodic=False).load_all()

    np.testing.assert_allclose(lazy, eager)


def test_non_periodic_2d_centers_both_axes() -> None:
    rng = np.random.default_rng(2)
    data = rng.random((1, 5, 6))

    centered = fc.FieldCentering_Diagnostic(_Mock(data, "b3", dim=2), periodic=False).load_all()

    def edge_avg(a, axis):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (1, 0)
        shifted = np.pad(a, pad, mode="edge")
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(0, a.shape[axis])
        return 0.5 * (a + shifted[tuple(sl)])

    np.testing.assert_allclose(centered, edge_avg(edge_avg(data, 1), 2))


def test_simulation_wrapper_threads_the_flag(sim_dir) -> None:
    import osiris_utils as ou

    sim = ou.Simulation(str(sim_dir / "thermal.1d"))

    centered = ou.FieldCentering_Simulation(sim, periodic=False)["e3"]

    assert centered._periodic is False
