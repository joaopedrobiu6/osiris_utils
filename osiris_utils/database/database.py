from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import tqdm as tqdm

from ..ar import AnomalousResistivity, AnomalousResistivityConfig
from ..postprocessing import Derivative_Diagnostic, Derivative_Simulation, MFT_Simulation
from ..profiling import _start_timer, _stop_timer

logger = logging.getLogger(__name__)

__all__ = ["DatabaseBuildConfig", "DatabaseCreator"]

_VALID_DATABASE_TYPES = {"both", "input", "output", "eta_new", "e_vlasov", "all", "vnT"}

# Feature spec type: (label, frame_getter(t_idx) -> 1-D array)
FeatureSpec = tuple[str, Callable[[int], np.ndarray]]


@dataclass(frozen=True)
class DatabaseBuildConfig:
    """Configuration for database tensor creation.

    Parameters
    ----------
    dtype :
        NumPy dtype for saved tensors. Default: float32.
    max_workers :
        Worker threads for parallel frame building. None = ThreadPoolExecutor default.
    mft_axis :
        Axis along which mean-field theory averages are computed. Default: 2 (y-axis).
    ar_config :
        AnomalousResistivityConfig used for output/e_vlasov databases.
        If None, a default is constructed at runtime (no time derivative, with convection,
        pressure, and magnetic force).
    validate_output :
        If True, NaN/inf values are replaced with 0 per frame and logged.
    resume :
        If True, skip frames already written in a previous (possibly crashed) run.
        Requires the ``.npy`` and ``.npy.progress`` files from that run to exist.
        Progress is tracked per-frame, so a job can be safely killed and restarted.
    flush_every :
        Flush the memory-mapped output to disk and save the progress file every
        *N* completed frames. Lower values protect more data at the cost of extra
        I/O. Default: 128. Set to 1 for maximum safety on unstable HPC jobs.
    """

    dtype: type = np.float32
    max_workers: int | None = None
    mft_axis: int = 2
    ar_config: AnomalousResistivityConfig | None = None
    validate_output: bool = True
    resume: bool = False
    flush_every: int = 128


class DatabaseCreator:
    """
    Build labeled NumPy tensors (T, F, X) from an OSIRIS simulation.

    Workflow
    --------
    1. Instantiate with a simulation, species, and save folder.
    2. Call ``set_limits(t0, t1)`` to define the time range.
    3. Call ``create_database(database=...)`` to build and save the tensors.

    Supported database types
    ------------------------
    ``"input"``     — 22-feature mean-field input tensor.
    ``"output"``    — eta (thesis pressure decomposition) tensor.
    ``"eta_new"``   — eta_new (simplified 4-term pressure) tensor.
    ``"e_vlasov"``  — mean-field Vlasov electric field tensor.
    ``"both"``      — input + output (eta).
    ``"all"``       — all four tensors above.
    ``"vnT"``       — 25-feature tensor: vfl1, n, T11, n*vfl1, n*T11 and their x1-derivatives up to 4th order, all MFT-averaged.
    """

    def __init__(
        self,
        simulation,
        species: str,
        save_folder: str,
        build_config: DatabaseBuildConfig | None = None,
    ):
        self.simulation = simulation
        self.species = species
        self.save_folder = save_folder
        self.build_config = build_config or DatabaseBuildConfig()

        # Iteration / spatial limits (set by set_limits)
        self.initial_iter: int = 0
        self.final_iter: int | None = None
        self.T: int = 0
        self.X: int = 0

        # Lazily initialised
        self._ar: AnomalousResistivity | None = None
        self._sim_mft: MFT_Simulation | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_limits(self, initial_iter: int = 0, final_iter: int | None = None) -> None:
        """Set the iteration range ``[initial_iter, final_iter)`` used for tensors.

        If *final_iter* is ``None`` it is inferred from ``simulation["e1"]``.
        """
        self.initial_iter = int(initial_iter)

        if final_iter is None:
            try:
                final_iter = len(self.simulation["e1"])
            except Exception as e:
                raise ValueError("final_iter=None but could not infer simulation time length from simulation['e1'].") from e

        self.final_iter = int(final_iter)

        if self.final_iter <= self.initial_iter:
            raise ValueError(f"final_iter must be > initial_iter (got {self.final_iter} <= {self.initial_iter}).")

        self.T = self.final_iter - self.initial_iter

        try:
            self.X = int(self.simulation["e1"].nx[0])
        except Exception as e:
            raise ValueError("Could not determine spatial size X from simulation['e1'].nx[0].") from e

        logger.info("Limits set: t=[%d, %d), T=%d, X=%d", self.initial_iter, self.final_iter, self.T, self.X)

    def create_database(
        self,
        database: str = "both",
        name_input: str = "input_tensor",
        name_output: str = "eta_tensor",
        name_eta_new: str = "eta_new_tensor",
        name_vlasov: str = "e_vlasov_tensor",
        name_vnT: str = "vnT_tensor",
    ) -> None:
        """Build and save database tensors as ``.npy`` files in ``save_folder``.

        Tensors are written frame-by-frame to memory-mapped files — the full
        tensor is **never held in RAM**.  Progress is checkpointed every
        ``flush_every`` frames so interrupted jobs can be resumed with
        ``DatabaseBuildConfig(resume=True)``.

        Parameters
        ----------
        database :
            Which tensors to build. One of: ``"input"``, ``"output"``, ``"eta_new"``,
            ``"e_vlasov"``, ``"vnT"``, ``"both"`` (input + output), ``"all"`` (all).
        name_input, name_output, name_eta_new, name_vlasov, name_vnT :
            File-stem names (without ``.npy``) for each tensor.
        """
        if database not in _VALID_DATABASE_TYPES:
            raise ValueError(f"Invalid database type '{database}'. Choose from: {sorted(_VALID_DATABASE_TYPES)}.")

        os.makedirs(self.save_folder, exist_ok=True)

        if self.final_iter is None:
            logger.warning("set_limits() not called; inferring from simulation['e1'].")
            self.set_limits(0, None)

        build_input = database in {"input", "both", "all"}
        build_output = database in {"output", "both", "all"}
        build_eta_new = database in {"eta_new", "all"}
        build_vlasov = database in {"e_vlasov", "all"}
        build_vnT = database in {"vnT", "all"}

        _timer = _start_timer(
            f"create_database({database}, T={self.T}, X={self.X}, "
            f"dtype={self.build_config.dtype.__name__ if hasattr(self.build_config.dtype, '__name__') else self.build_config.dtype})"
        )
        try:
            if build_input:
                logger.info("Building input database '%s'...", name_input)
                self._input_database(name=name_input)

            if build_output:
                logger.info("Building output (eta) database '%s'...", name_output)
                self._ar_database(key="eta", name=name_output, desc="Creating eta database")

            if build_eta_new:
                logger.info("Building eta_new database '%s'...", name_eta_new)
                self._ar_database(key="eta_new", name=name_eta_new, desc="Creating eta_new database")

            if build_vlasov:
                logger.info("Building e_vlasov database '%s'...", name_vlasov)
                self._ar_database(key="e_vlasov_avg", name=name_vlasov, desc="Creating e_vlasov database")

            if build_vnT:
                logger.info("Building vnT database '%s'...", name_vnT)
                self._vnT_database(name=name_vnT)

            logger.info("All requested databases saved to '%s'.", self.save_folder)
        finally:
            _stop_timer(_timer)

    # ------------------------------------------------------------------
    # Input database
    # ------------------------------------------------------------------

    def _setup_input_diagnostics(self) -> MFT_Simulation:
        """Compute all diagnostics needed for the input tensor and return the MFT wrapper."""
        d_dx1 = Derivative_Simulation(self.simulation, deriv_type="x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
        # d_dt = Derivative_Simulation(self.simulation, "t", stencil=[-2, -1, 0, 1, 2], deriv_order=1)

        sp = self.simulation[self.species]

        self._ensure_diagnostic(sp, sp["n"] * sp["T11"], "nT11")
        self._ensure_diagnostic(sp, sp["n"] * sp["T12"], "nT12")

        # First derivatives (x1)
        self._ensure_diagnostic(
            sp, Derivative_Diagnostic(sp["nT11"], deriv_type="x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1), "dnT11_dx1"
        )
        self._ensure_diagnostic(
            sp, Derivative_Diagnostic(sp["nT12"], deriv_type="x2", stencil=[-2, -1, 0, 1, 2], deriv_order=1, periodic=True), "dnT12_dx2"
        )
        # self._ensure_diagnostic(sp, d_dt[self.species]["vfl1"], "dvfl1_dt")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl1"], "dvfl1_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl2"], "dvfl2_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["vfl3"], "dvfl3_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["b2"], "db2_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["b3"], "db3_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["n"], "dn_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["T11"], "dT11_dx1")

        # Second derivatives
        self._ensure_diagnostic(sp, d_dx1[self.species]["dnT11_dx1"], "d2_nT11_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl1_dx1"], "d2_vfl1_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl2_dx1"], "d2_vfl2_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dvfl3_dx1"], "d2_vfl3_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["db2_dx1"], "d2_b2_dx1")
        self._ensure_diagnostic(self.simulation, d_dx1["db3_dx1"], "d2_b3_dx1")
        self._ensure_diagnostic(sp, d_dx1[self.species]["dn_dx1"], "d2_n_dx1")

        return MFT_Simulation(self.simulation, mft_axis=self.build_config.mft_axis)

    def _input_feature_specs(self, sim_mft: MFT_Simulation) -> list[FeatureSpec]:
        """Return an ordered list of (label, frame_getter) pairs for the input tensor.

        Each getter takes a time index and returns a 1-D array of shape (X,).
        Override this method in a subclass to customise the feature set.
        """
        sp = self.species
        sm = sim_mft

        dnT11_dx_avg = Derivative_Diagnostic(
            sm[sp]["n"]["avg"] * sm[sp]["T11"]["avg"], deriv_type="x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1
        )

        def _sp(name: str) -> Callable[[int], np.ndarray]:
            return lambda t: sm[sp][name]["avg"][t].flatten()

        def _field(name: str) -> Callable[[int], np.ndarray]:
            return lambda t: sm[name]["avg"][t].flatten()

        return [
            # Fields
            ("b2_avg", _field("b2")),
            ("b3_avg", _field("b3")),
            # Species fluid velocities
            ("vfl1_avg", _sp("vfl1")),
            ("vfl2_avg", _sp("vfl2")),
            ("vfl3_avg", _sp("vfl3")),
            # Density and temperature
            ("n_avg", _sp("n")),
            ("T11_avg", _sp("T11")),
            ("T12_avg", _sp("T12")),
            # First derivatives (x1)
            ("dvfl1_dx1_avg", _sp("dvfl1_dx1")),
            ("dvfl2_dx1_avg", _sp("dvfl2_dx1")),
            ("dvfl3_dx1_avg", _sp("dvfl3_dx1")),
            ("dn_dx1_avg", _sp("dn_dx1")),
            ("dT11_dx1_avg", _sp("dT11_dx1")),
            ("db2_dx1_avg", _field("db2_dx1")),
            ("db3_dx1_avg", _field("db3_dx1")),
            # Second derivatives (x1)
            ("d2_vfl1_dx1_avg", _sp("d2_vfl1_dx1")),
            ("d2_vfl2_dx1_avg", _sp("d2_vfl2_dx1")),
            ("d2_vfl3_dx1_avg", _sp("d2_vfl3_dx1")),
            ("d2_b2_dx1_avg", _field("d2_b2_dx1")),
            ("d2_b3_dx1_avg", _field("d2_b3_dx1")),
            ("d2_n_dx1_avg", _sp("d2_n_dx1")),
            # Composite: d/dx1 (n_avg * T11_avg)
            ("dnT11_dx1_avg", lambda t: dnT11_dx_avg[t].flatten()),
        ]

    def _input_database(self, name: str) -> None:
        sim_mft = self._setup_input_diagnostics()
        specs = self._input_feature_specs(sim_mft)
        F_in = len(specs)
        labels, getters = zip(*specs, strict=True)

        logger.info("Input tensor: %d features — %s", F_in, list(labels))

        def get_frame(t_idx: int) -> np.ndarray:
            return np.stack([g(t_idx) for g in getters])

        self._build_tensor(
            name=name,
            shape=(self.T, F_in, self.X),
            frame_fn=get_frame,
            desc="Building input tensor",
            validate=False,
        )

    # ------------------------------------------------------------------
    # Output databases (eta, eta_new, e_vlasov)
    # ------------------------------------------------------------------

    @property
    def _ar_instance(self) -> AnomalousResistivity:
        """Lazily create (and cache) the AnomalousResistivity object."""
        if self._ar is None:
            ar_config = self.build_config.ar_config or AnomalousResistivityConfig(
                species=self.species,
                include_time_derivative=False,
                include_convection=True,
                include_pressure=True,
                include_magnetic_force=True,
            )
            self._ar = AnomalousResistivity(self.simulation, self.species, config=ar_config)
        return self._ar

    def _ar_database(self, *, key: str, name: str, desc: str) -> None:
        """Generic builder for any scalar AR term (eta, eta_new, e_vlasov_avg, …)."""
        ar = self._ar_instance

        def get_frame(t_idx: int) -> np.ndarray:
            return np.stack([ar[key][t_idx].flatten()])

        self._build_tensor(
            name=name,
            shape=(self.T, 1, self.X),
            frame_fn=get_frame,
            desc=desc,
            validate=self.build_config.validate_output,
        )

    # ------------------------------------------------------------------
    # vnT database (vfl1, n, T11, n*vfl1, n*T11 + x1-derivatives 1–4)
    # ------------------------------------------------------------------

    def _setup_vnT_diagnostics(self) -> MFT_Simulation:
        """Register all diagnostics needed for the vnT tensor."""
        d_dx1 = Derivative_Simulation(self.simulation, deriv_type="x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
        sp = self.simulation[self.species]

        # Composite base quantities
        self._ensure_diagnostic(sp, sp["n"] * sp["vfl1"], "nvfl1")
        self._ensure_diagnostic(sp, sp["n"] * sp["T11"], "nT11")

        # First derivatives
        for name in ("vfl1", "n", "T11", "nvfl1", "nT11"):
            self._ensure_diagnostic(sp, d_dx1[self.species][name], f"d_{name}_dx1")

        # Second derivatives
        for name in ("vfl1", "n", "T11", "nvfl1", "nT11"):
            self._ensure_diagnostic(sp, d_dx1[self.species][f"d_{name}_dx1"], f"d2_{name}_dx1")

        # Third derivatives
        for name in ("vfl1", "n", "T11", "nvfl1", "nT11"):
            self._ensure_diagnostic(sp, d_dx1[self.species][f"d2_{name}_dx1"], f"d3_{name}_dx1")

        # Fourth derivatives
        for name in ("vfl1", "n", "T11", "nvfl1", "nT11"):
            self._ensure_diagnostic(sp, d_dx1[self.species][f"d3_{name}_dx1"], f"d4_{name}_dx1")

        return MFT_Simulation(self.simulation, mft_axis=self.build_config.mft_axis)

    def _vnT_feature_specs(self, sim_mft: MFT_Simulation) -> list[FeatureSpec]:
        """Return (label, frame_getter) pairs for the vnT tensor (25 features)."""
        sp = self.species
        sm = sim_mft

        def _sp(name: str) -> Callable[[int], np.ndarray]:
            return lambda t: sm[sp][name]["avg"][t].flatten()

        base = ["vfl1", "n", "T11", "nvfl1", "nT11"]
        specs: list[FeatureSpec] = [(f"{q}_avg", _sp(q)) for q in base]
        for order, prefix in enumerate(("d_", "d2_", "d3_", "d4_"), start=1):
            for q in base:
                diag_name = f"{prefix}{q}_dx1"
                specs.append((f"{diag_name}_avg", _sp(diag_name)))
        return specs

    def _vnT_database(self, name: str) -> None:
        sim_mft = self._setup_vnT_diagnostics()
        specs = self._vnT_feature_specs(sim_mft)
        F = len(specs)
        labels, getters = zip(*specs, strict=True)

        logger.info("vnT tensor: %d features — %s", F, list(labels))

        def get_frame(t_idx: int) -> np.ndarray:
            return np.stack([g(t_idx) for g in getters])

        self._build_tensor(
            name=name,
            shape=(self.T, F, self.X),
            frame_fn=get_frame,
            desc="Building vnT tensor",
            validate=False,
        )

    # ------------------------------------------------------------------
    # Core tensor builder
    # ------------------------------------------------------------------

    def _build_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, int, int],
        frame_fn: Callable[[int], np.ndarray],
        desc: str,
        validate: bool = False,
    ) -> None:
        """Build tensor of *shape* by streaming frames to a memory-mapped file.

        The full tensor is **never held in RAM** — each frame is written directly
        to the ``.npy`` file on disk via ``numpy.lib.format.open_memmap``.
        This makes 60 GB tensors feasible on any node regardless of RAM.

        Crash safety / resume
        ---------------------
        A companion ``<name>.npy.progress`` file (a boolean array of length T)
        tracks which frames have been flushed to disk.  On the next run with
        ``DatabaseBuildConfig(resume=True)``, already-completed frames are
        skipped.  Progress is saved every ``flush_every`` frames.

        The safety guarantee is:
        * A frame is only marked *done* in the progress file **after** the
          memmap has been flushed.  So on crash, at worst the last
          ``flush_every`` frames are recomputed — no corrupt data.
        """
        if self.T <= 0:
            raise ValueError("Nothing to build: T <= 0. Did you call set_limits()?")

        save_path = os.path.join(self.save_folder, f"{name}.npy")
        progress_path = f"{save_path}.progress"
        dtype = self.build_config.dtype

        # ── Open / create the memory-mapped file ──────────────────────────
        if self.build_config.resume and os.path.exists(save_path) and os.path.exists(progress_path):
            done_mask = np.load(progress_path)
            if done_mask.shape != (self.T,):
                logger.warning(
                    "Progress file shape %s != (%d,); restarting '%s' from scratch.",
                    done_mask.shape,
                    self.T,
                    name,
                )
                done_mask = np.zeros(self.T, dtype=bool)
                arr = np.lib.format.open_memmap(save_path, mode="w+", dtype=dtype, shape=shape)
            else:
                arr = np.lib.format.open_memmap(save_path, mode="r+", dtype=dtype, shape=shape)
                logger.info(
                    "Resuming '%s': %d / %d frames already done.",
                    name,
                    int(done_mask.sum()),
                    self.T,
                )
        else:
            done_mask = np.zeros(self.T, dtype=bool)
            arr = np.lib.format.open_memmap(save_path, mode="w+", dtype=dtype, shape=shape)

        work_items = [(out_i, t_idx) for out_i, t_idx in enumerate(range(self.initial_iter, self.final_iter)) if not done_mask[out_i]]

        if not work_items:
            logger.info("'%s' already complete — skipping.", name)
            del arr
            return

        # ── Thread-safe flush / progress helpers ──────────────────────────
        _lock = threading.Lock()
        _n_written = [0]

        def _write_frame(out_i: int, t_idx: int) -> None:
            frame = frame_fn(t_idx)
            if validate:
                frame = _clean_frame(frame, name, out_i)
            # Write to mmap (different out_i → different disk pages → thread-safe)
            arr[out_i] = frame.astype(dtype, copy=False)
            with _lock:
                done_mask[out_i] = True
                _n_written[0] += 1
                if _n_written[0] % self.build_config.flush_every == 0:
                    # Flush first, then save progress — so progress only reflects
                    # data that is confirmed on disk.
                    arr.flush()
                    np.save(progress_path, done_mask.copy())

        # ── Warm-up: first frame on main thread ───────────────────────────
        # Catches lazy diagnostic initialisation / shape mismatches before
        # spawning worker threads.
        _write_frame(*work_items[0])
        remaining = work_items[1:]

        # ── Parallel frame building ────────────────────────────────────────
        if remaining:
            with ThreadPoolExecutor(max_workers=self.build_config.max_workers) as ex:
                futures = {ex.submit(_write_frame, out_i, t_idx): (out_i, t_idx) for out_i, t_idx in remaining}
                with tqdm.tqdm(total=len(work_items), initial=1, desc=desc) as pbar:
                    for fut in as_completed(futures):
                        out_i, t_idx = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            # Emergency flush before propagating — preserve progress
                            with _lock:
                                arr.flush()
                                np.save(progress_path, done_mask.copy())
                            raise RuntimeError(f"Failed building '{name}' at output index {out_i} (simulation t_idx={t_idx}).") from e
                        pbar.update(1)

        # ── Final flush and cleanup ────────────────────────────────────────
        arr.flush()
        del arr  # close memmap handle (does not delete the file)

        if os.path.exists(progress_path):
            os.remove(progress_path)

        size_gb = (shape[0] * shape[1] * shape[2] * np.dtype(dtype).itemsize) / 1e9
        logger.info(
            "Saved '%s' (%.2f GB, %s) → %s",
            name,
            size_gb,
            "x".join(str(s) for s in shape),
            save_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def feature_labels(self) -> list[str]:
        """Return the ordered list of input feature labels (requires diagnostics to be set up)."""
        sm = MFT_Simulation(self.simulation, mft_axis=self.build_config.mft_axis)
        specs = self._input_feature_specs(sm)
        return [label for label, _ in specs]

    @property
    def x(self) -> Any:
        return self.simulation["e1"].x[0]

    @property
    def dx(self) -> Any:
        return self.simulation["e1"].dx[0]

    @staticmethod
    def _ensure_diagnostic(container, diagnostic, name: str) -> None:
        """Add a diagnostic only if not already present."""
        try:
            _ = container[name]
        except Exception:
            container.add_diagnostic(diagnostic, name)

    # _validate_and_clean_data kept for backwards compatibility.
    # New code uses the module-level _clean_frame() which works per-frame.
    @staticmethod
    def _validate_and_clean_data(data: np.ndarray) -> np.ndarray:
        """Replace NaN/inf with 0 in-place and log a warning if any are found."""
        mask = ~np.isfinite(data)
        if np.any(mask):
            nan_count = int(np.count_nonzero(np.isnan(data)))
            inf_count = int(np.count_nonzero(np.isinf(data)))
            logger.warning("Found %d NaN and %d inf values — replacing with zeros.", nan_count, inf_count)
            data[mask] = 0.0
        return data


def _clean_frame(frame: np.ndarray, name: str, out_i: int) -> np.ndarray:
    """Replace NaN/inf with 0 in a single frame and log if any are found.

    Returns the original array if no bad values are found (no copy).
    Returns a cleaned copy otherwise.
    """
    mask = ~np.isfinite(frame)
    if np.any(mask):
        nan_count = int(np.count_nonzero(np.isnan(frame)))
        inf_count = int(np.count_nonzero(np.isinf(frame)))
        logger.warning(
            "'%s' frame %d: replacing %d NaN + %d inf with zeros.",
            name,
            out_i,
            nan_count,
            inf_count,
        )
        frame = frame.copy()
        frame[mask] = 0.0
    return frame
