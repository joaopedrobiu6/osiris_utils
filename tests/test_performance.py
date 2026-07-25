from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import osiris_utils as ou

from .conftest import N_TIMESTEPS, NX, grid_values


def _e3(sim_dir: Path) -> ou.Diagnostic:
    d = ou.Diagnostic(simulation_folder=str(sim_dir))
    d.get_quantity("e3")
    return d


def _expected() -> np.ndarray:
    return np.stack([grid_values(i) for i in range(N_TIMESTEPS)])


class TestParallelLoading:
    """Test parallel loading functionality."""

    def test_load_all_sequential(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        data = d.load_all(use_parallel=False)

        assert data.shape == (N_TIMESTEPS, NX)
        assert d.all_loaded is True
        np.testing.assert_allclose(data, _expected(), rtol=1e-6)

    def test_load_all_parallel_threads(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        data = d.load_all(use_parallel=True, executor_type="thread")

        assert data.shape == (N_TIMESTEPS, NX)
        assert d.all_loaded is True
        np.testing.assert_allclose(data, _expected(), rtol=1e-6)

    def test_parallel_and_sequential_agree(self, sim_dir: Path) -> None:
        """Out-of-order future completion must not scramble the time axis."""
        sequential = _e3(sim_dir).load_all(use_parallel=False)
        parallel = _e3(sim_dir).load_all(use_parallel=True, executor_type="thread")

        np.testing.assert_allclose(sequential, parallel)

    def test_auto_detect_uses_sequential_for_small_files(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        data = d.load_all(use_parallel=None)

        np.testing.assert_allclose(data, _expected(), rtol=1e-6)

    def test_load_all_is_idempotent(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        first = d.load_all()
        second = d.load_all()

        assert second is first

    def test_unload_after_load(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        d.load_all()
        assert d.all_loaded is True

        d.unload()
        assert d.all_loaded is False
        assert d._data is None

    def test_process_executor_rejected_for_arithmetic_result(self, sim_dir: Path) -> None:
        """_binary_op attaches _frame to the instance; workers would bypass it."""
        derived = _e3(sim_dir) * 2.0

        with pytest.raises(RuntimeError, match="executor_type='process'"):
            derived.load_all(use_parallel=True, executor_type="process")

    def test_process_executor_rejected_for_subclass_overriding_frame(self, sim_dir: Path) -> None:
        """A subclass transforming frames must not be dispatched to raw-file workers."""

        class Scaled(ou.Diagnostic):
            def _frame(self, index, data_slice=None):
                return 3.0 * super()._frame(index, data_slice=data_slice)

        d = Scaled(simulation_folder=str(sim_dir))
        d.get_quantity("e3")

        with pytest.raises(RuntimeError, match="executor_type='process'"):
            d.load_all(use_parallel=True, executor_type="process")

    def test_postprocessing_load_all_does_not_accept_process_executor(self, sim_dir: Path) -> None:
        """Post-processing classes narrow load_all(), so the kwarg is rejected outright."""
        sim = ou.Simulation(str(sim_dir / "thermal.1d"))
        deriv = ou.Derivative_Simulation(sim, "x1")["e3"]

        with pytest.raises(TypeError, match="use_parallel"):
            deriv.load_all(use_parallel=True, executor_type="process")


class TestIterationOptimizations:
    """Test iteration optimizations."""

    def test_iteration_yields_every_timestep(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        frames = list(d)

        assert len(frames) == N_TIMESTEPS
        np.testing.assert_allclose(np.stack(frames), _expected(), rtol=1e-6)

    def test_len_returns_maxiter(self, sim_dir: Path) -> None:
        d = _e3(sim_dir)

        assert len(d) == N_TIMESTEPS
        assert len(d) == d.maxiter
