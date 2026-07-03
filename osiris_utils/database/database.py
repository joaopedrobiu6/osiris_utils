from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import tqdm as tqdm

from ..ar import AnomalousResistivityConfig
from ..profiling import _start_timer, _stop_timer
from .filters import SpatialFilter, as_filter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = ["DatabaseBuildConfig", "DatabaseCreator"]

_VALID_DATABASE_TYPES = {"both", "input", "output", "e_vlasov", "all", "vnT"}
_VALID_ETA_FORMULAS = {"thesis", "lhs"}

# =====================================================================
# Database parameters — edit here to change what the tensors contain
# =====================================================================
# Boundary convention (2-D shock simulations): x1 is longitudinal and
# non-periodic, x2 is transverse and periodic.  The transverse average
# is taken along ``DatabaseBuildConfig.mft_axis`` (OSIRIS 1-indexed,
# default 2 = x2).
#
# Each label below must be a key of the quantity dict produced by
# ``_mean_field_frame_quantities`` (input/output/e_vlasov) or
# ``_vnT_frame_quantities`` (vnT).  Tensor rows follow list order, so
# adding/removing/reordering a feature is an edit here (plus, for a new
# quantity, one line in the frame function that computes it).

#: Rows of the "input" tensor (mean-field features).
INPUT_FEATURE_LABELS: list[str] = [
    # Fields
    "b2_avg",
    "b3_avg",
    # Species fluid velocities
    "vfl1_avg",
    "vfl2_avg",
    "vfl3_avg",
    # Density and temperature
    "n_avg",
    "T11_avg",
    "T12_avg",
    "nvfl1_avg",
    # First derivatives (x1)
    "d1_b2_dx1_avg",
    "d1_b3_dx1_avg",
    "d1_vfl1_dx1_avg",
    "d1_vfl2_dx1_avg",
    "d1_vfl3_dx1_avg",
    "d1_n_dx1_avg",
    "d1_T11_dx1_avg",
    "d1_T12_dx1_avg",
    # Second derivatives (x1)
    "d2_b2_dx1_avg",
    "d2_b3_dx1_avg",
    "d2_vfl1_dx1_avg",
    "d2_vfl2_dx1_avg",
    "d2_vfl3_dx1_avg",
    "d2_n_dx1_avg",
    "d2_T11_dx1_avg",
    "d2_T12_dx1_avg",
    # Three derivatives (x1)
    "d3_b2_dx1_avg",
    "d3_b3_dx1_avg",
    "d3_vfl1_dx1_avg",
    "d3_vfl2_dx1_avg",
    "d3_vfl3_dx1_avg",
    "d3_n_dx1_avg",
    "d3_T11_dx1_avg",
    "d3_T12_dx1_avg",
    # Four derivatives (x1)
    "d4_b2_dx1_avg",
    "d4_b3_dx1_avg",
    "d4_vfl1_dx1_avg",
    "d4_vfl2_dx1_avg",
    "d4_vfl3_dx1_avg",
    "d4_n_dx1_avg",
    "d4_T11_dx1_avg",
    "d4_T12_dx1_avg",
    # Composite: d/dx1 (n_avg * T11_avg)
    "d1_nT11_dx1_avg",
    "d1_nvfl1_dx1_avg",
    # Composite: d2/dx1 (n_avg * T11_avg)
    "d2_nT11_dx1_avg",
    "d2_nvfl1_dx1_avg",
    # Composite: d/dx1 (n_avg * T11_avg) / n_avg
    "dnT11_dx1_over_n_avg",
]

#: Rows of the "output" tensor (anomalous resistivity).
OUTPUT_LABELS: list[str] = ["eta_avg"]

#: Rows of the "e_vlasov" tensor.
E_VLASOV_LABELS: list[str] = ["e_vlasov_avg"]

#: Base (transverse-averaged) quantities of the vnT tensor; the tensor
#: holds these plus their x1-derivatives of orders VNT_DERIV_ORDERS.
VNT_BASE_QUANTITIES: tuple[str, ...] = ("vfl1", "n", "T11", "nvfl1", "nT11")
VNT_DERIV_ORDERS: tuple[int, ...] = (1, 2, 3, 4)


def vnT_feature_labels() -> list[str]:
    """Ordered rows of the vnT tensor, derived from the parameters above."""
    labels = [f"{q}_avg" for q in VNT_BASE_QUANTITIES]
    for order in VNT_DERIV_ORDERS:
        prefix = "d_" if order == 1 else f"d{order}_"
        labels += [f"{prefix}{q}_dx1_avg" for q in VNT_BASE_QUANTITIES]
    return labels


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
        Axis along which mean-field theory averages are computed
        (OSIRIS 1-indexed). Default: 2 (transverse / x2).
    ar_config :
        AnomalousResistivityConfig whose ``include_*`` flags gate which terms
        enter e_vlasov and eta. If None, the defaults are used (no time
        derivative; convection, pressure and magnetic force included).
        ``include_time_derivative=True`` is not supported by the per-frame
        pipeline and raises NotImplementedError.
    filters :
        Spatial filters applied, in order, to every raw 2-D frame *before*
        any physics (e_vlasov, mean fields, derivatives) is computed.
        Empty (default) = no filtering, 4th-order finite-difference
        derivatives — identical to the historical tensors.  Filters like
        :class:`~osiris_utils.database.filters.SavitzkyGolayFilter` also
        supply their own analytic derivative scheme, which is then used for
        every derivative in the pipeline.
    eta_formula :
        Which formula the output (eta) tensor uses:
        ``"thesis"`` (default) — 7-term pressure decomposition built from
        transverse fluctuation cross-terms (matches the historical output);
        ``"lhs"`` — ⟨e_vlasov⟩ plus mean-field corrections (the form used by
        the Lorentz database).
    validate_output :
        If True, NaN/inf values are replaced with 0 per frame and logged
        (output and e_vlasov tensors only).
    resume :
        If True, skip frames already written in a previous (possibly crashed) run.
        Requires the ``.npy`` and ``.npy.progress.npy`` files from that run to exist.
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
    filters: Sequence[SpatialFilter] | SpatialFilter | None = ()
    eta_formula: str = "thesis"
    validate_output: bool = True
    resume: bool = False
    flush_every: int = 128


# ----------------------------------------------------------------------
# Per-frame computation (module-level for clean stack traces)
# ----------------------------------------------------------------------


def _stack_rows(quantities: dict[str, np.ndarray], labels: Sequence[str]) -> np.ndarray:
    """Stack the 1-D quantities selected by *labels* into a (F, X) array."""
    try:
        return np.stack([np.asarray(quantities[label], dtype=np.float64).ravel() for label in labels])
    except KeyError as e:
        raise KeyError(f"Feature '{e.args[0]}' is not computed by the frame pipeline. Available: {sorted(quantities)}") from e


def _load_filtered_fields(
    raw: dict[str, Any],
    names: Sequence[str],
    t_idx: int,
    filt: SpatialFilter,
    avg_axis: int,
) -> dict[str, np.ndarray]:
    """Load the raw 2-D fields *names* at *t_idx* and smooth each one."""
    periodic = tuple(ax == avg_axis for ax in range(2))
    return {name: filt.smooth(np.asarray(raw[name][t_idx], dtype=np.float64), periodic=periodic) for name in names}


def _mean_field_frame_quantities(
    raw: dict[str, Any],
    t_idx: int,
    filt: SpatialFilter,
    dx: float,
    dx2: float,
    avg_axis: int,
    flags: AnomalousResistivityConfig,
    eta_formula: str = "thesis",
    compute_e_vlasov: bool = True,
    compute_eta: bool = True,
) -> dict[str, np.ndarray]:
    r"""Compute all mean-field quantities for one timestep.

    Pipeline (per timestep):

        raw (nx, ny) frames
          → spatial filter (2-D)
          → x1-derivatives, e_vlasov and composites in 2-D   [same (nx, ny)]
          → transverse mean ⟨·⟩ along *avg_axis*             [(nx,)]
          → eta ("thesis" or "lhs" formula)

    Returns a dict of named 1-D arrays; tensors select rows from it via
    :data:`INPUT_FEATURE_LABELS` / :data:`OUTPUT_LABELS` / :data:`E_VLASOV_LABELS`.

    Notes
    -----
    - All per-field x1-derivatives are taken on the filtered 2-D data
      **before** the transverse average (along the longitudinal axis,
      i.e. the first array dimension for the default ``mft_axis=2``).
      The exceptions are the mean-field composites ``d{1,2}_nT11_dx1_avg``
      = ∂x1(⟨n⟩⟨T11⟩) and ``d{1,2}_nvfl1_dx1_avg`` = ∂x1(⟨n⟩⟨vfl1⟩), which by
      definition differentiate the product of the averaged profiles.
    - ``eta_formula="lhs"``:

      .. math::
          \eta = \langle e_{vlasov}\rangle
               + \langle v_1\rangle\,\partial_{x}\langle v_1\rangle
               + \frac{1}{\langle n\rangle}\partial_{x}(\langle n\rangle\langle T_{11}\rangle)
               + \langle v_2\rangle\langle B_3\rangle - \langle v_3\rangle\langle B_2\rangle

    - ``eta_formula="thesis"``: fluctuation cross-terms
      (:math:`f' = f - \langle f\rangle`), replicating
      ``AnomalousResistivity._compute_mft_terms`` / ``_compute_eta_values``:

      .. math::
          \eta = -\langle v_1'\,(\partial_x v_1)'\rangle
               - \langle v_2' B_3'\rangle + \langle v_3' B_2'\rangle
               + \Bigl\langle\frac{\partial_x(\langle n\rangle\langle T_{11}\rangle)}{n}
                              \frac{n'}{\langle n\rangle}\Bigr\rangle
               - \sum \Bigl\langle\frac{\partial(\text{pressure cross-terms})}{n}\Bigr\rangle
    """
    if compute_eta and eta_formula == "lhs":
        compute_e_vlasov = True

    x_axis = 1 - avg_axis  # longitudinal axis in 2-D arrays (avg_axis=1 → x_axis=0)

    f = _load_filtered_fields(raw, ("n", "T11", "T12", "vfl1", "vfl2", "vfl3", "b2", "b3"), t_idx, filt, avg_axis)
    n, T11, T12 = f["n"], f["T11"], f["T12"]
    vfl1, vfl2, vfl3 = f["vfl1"], f["vfl2"], f["vfl3"]
    b2, b3 = f["b2"], f["b3"]

    # Recursive derivative helpers (2-D, before the transverse average)
    def d_x1(g: np.ndarray, order: int = 1) -> np.ndarray:
        return filt.derivative(g, dx, axis=x_axis, order=order, periodic=False)

    def d_x2(g: np.ndarray, order: int = 1) -> np.ndarray:
        return filt.derivative(g, dx2, axis=avg_axis, order=order, periodic=True)

    def avg(g: np.ndarray) -> np.ndarray:
        return g.mean(axis=avg_axis)

    # ── Transverse averages ⟨·⟩ ───────────────────────────────────────
    q: dict[str, np.ndarray] = {f"{name}_avg": avg(arr) for name, arr in f.items()}

    # ── x1-derivatives in 2-D (before the transverse average), orders 1-4 ──
    deriv_fields = ("vfl1", "vfl2", "vfl3", "n", "T11", "T12", "b2", "b3")
    prev_2d = f
    d_2d_by_order: dict[int, dict[str, np.ndarray]] = {}
    for order in (1, 2, 3, 4):
        cur_2d = {name: d_x1(prev_2d[name]) for name in deriv_fields}
        d_2d_by_order[order] = cur_2d
        for name in deriv_fields:
            q[f"d{order}_{name}_dx1_avg"] = avg(cur_2d[name])
        prev_2d = cur_2d
    d1_2d = d_2d_by_order[1]

    # Mean-field composites ⟨n⟩⟨T11⟩, ⟨n⟩⟨vfl1⟩: products of the *averaged*
    # profiles, so their derivatives can only be taken after the average
    # (unlike the per-field derivatives above, which are taken in 2-D first).
    nT11_mf = q["n_avg"] * q["T11_avg"]
    nvfl1_mf = q["n_avg"] * q["vfl1_avg"]
    q["nvfl1_avg"] = nvfl1_mf
    q["d1_nT11_dx1_avg"] = filt.derivative(nT11_mf, dx, axis=0, order=1, periodic=False)
    q["d2_nT11_dx1_avg"] = filt.derivative(nT11_mf, dx, axis=0, order=2, periodic=False)
    q["d1_nvfl1_dx1_avg"] = filt.derivative(nvfl1_mf, dx, axis=0, order=1, periodic=False)
    q["d2_nvfl1_dx1_avg"] = filt.derivative(nvfl1_mf, dx, axis=0, order=2, periodic=False)
    q["dnT11_dx1_over_n_avg"] = q["d1_nT11_dx1_avg"] / q["n_avg"]

    if not (compute_e_vlasov or compute_eta):
        return q

    # ── Remaining 2-D derivatives for e_vlasov / eta ───────────────────
    dvfl1_dx1_2d = d1_2d["vfl1"]
    dvfl1_dx2_2d = d_x2(vfl1) if flags.include_transverse_advection else None

    # ── e_vlasov in 2-D (no E_x), then transverse average ─────────────
    if compute_e_vlasov:
        e_vlasov = np.zeros_like(n)
        if flags.include_convection:
            e_vlasov -= vfl1 * dvfl1_dx1_2d
        if flags.include_transverse_advection:
            e_vlasov -= vfl2 * dvfl1_dx2_2d
        if flags.include_pressure:
            e_vlasov -= (d_x1(n * T11) + d_x2(n * T12)) / n
        if flags.include_magnetic_force:
            e_vlasov += -vfl2 * b3 + vfl3 * b2
        q["e_vlasov_avg"] = avg(e_vlasov)

    if not compute_eta:
        return q

    # ── eta ────────────────────────────────────────────────────────────
    if eta_formula == "lhs":
        eta = q["e_vlasov_avg"].copy()
        if flags.include_convection:
            eta += q["vfl1_avg"] * q["d1_vfl1_dx1_avg"]
        if flags.include_transverse_advection:
            eta += q["vfl2_avg"] * avg(dvfl1_dx2_2d)
        if flags.include_pressure:
            eta += q["dnT11_dx1_over_n_avg"]
        if flags.include_magnetic_force:
            eta += q["vfl2_avg"] * q["b3_avg"] - q["vfl3_avg"] * q["b2_avg"]

    elif eta_formula == "thesis":

        def delta(g: np.ndarray) -> np.ndarray:
            return g - g.mean(axis=avg_axis, keepdims=True)

        eta = np.zeros_like(q["n_avg"])
        if flags.include_convection:
            eta -= avg(delta(vfl1) * delta(dvfl1_dx1_2d))
        if flags.include_transverse_advection:
            eta -= avg(delta(vfl2) * delta(dvfl1_dx2_2d))
        if flags.include_magnetic_force:
            eta -= avg(delta(vfl2) * delta(b3))
            eta += avg(delta(vfl3) * delta(b2))
        if flags.include_pressure:
            # Thesis decomposition: density-fluctuation correction + 6 pressure
            # cross-terms (avg×delta, delta×avg, delta×delta for xx and xy).
            n_avg_kd = n.mean(axis=avg_axis, keepdims=True)
            n_d = n - n_avg_kd
            T11_avg_kd = T11.mean(axis=avg_axis, keepdims=True)
            T11_d = T11 - T11_avg_kd
            T12_avg_kd = T12.mean(axis=avg_axis, keepdims=True)
            T12_d = T12 - T12_avg_kd

            eta += avg((d_x1(n_avg_kd * T11_avg_kd) / n) * (n_d / n_avg_kd))
            eta -= avg(d_x1(n_avg_kd * T11_d) / n)
            eta -= avg(d_x1(n_d * T11_avg_kd) / n)
            eta -= avg(d_x1(n_d * T11_d) / n)
            eta -= avg(d_x2(n_avg_kd * T12_d) / n)
            eta -= avg(d_x2(n_d * T12_avg_kd) / n)
            eta -= avg(d_x2(n_d * T12_d) / n)

    else:
        raise ValueError(f"Invalid eta_formula '{eta_formula}'. Choose from: {sorted(_VALID_ETA_FORMULAS)}.")

    q["eta_avg"] = eta
    return q


def _vnT_frame_quantities(
    raw: dict[str, Any],
    t_idx: int,
    filt: SpatialFilter,
    dx: float,
    avg_axis: int,
) -> dict[str, np.ndarray]:
    """Compute the vnT quantities for one timestep.

    Transverse-averaged base quantities (:data:`VNT_BASE_QUANTITIES`) plus
    their x1-derivatives of orders :data:`VNT_DERIV_ORDERS`.  Derivatives are
    taken on the filtered 2-D data **before** the transverse average, by
    recursively chaining the filter's first-derivative operator along the
    longitudinal axis (for :class:`NoFilter` this is repeated 4th-order FD —
    identical to the historical chained first derivatives).
    """
    x_axis = 1 - avg_axis  # longitudinal axis in 2-D arrays (avg_axis=1 → x_axis=0)
    f = _load_filtered_fields(raw, ("n", "T11", "vfl1"), t_idx, filt, avg_axis)

    base_2d = {
        "vfl1": f["vfl1"],
        "n": f["n"],
        "T11": f["T11"],
        "nvfl1": f["n"] * f["vfl1"],
        "nT11": f["n"] * f["T11"],
    }
    if set(base_2d) != set(VNT_BASE_QUANTITIES):
        raise KeyError(f"VNT_BASE_QUANTITIES {VNT_BASE_QUANTITIES} do not match the computed quantities {sorted(base_2d)}.")

    q: dict[str, np.ndarray] = {}
    for name in VNT_BASE_QUANTITIES:
        q[f"{name}_avg"] = base_2d[name].mean(axis=avg_axis)
        d_2d = base_2d[name]
        prev_order = 0
        for order in VNT_DERIV_ORDERS:
            d_2d = filt.derivative(d_2d, dx, axis=x_axis, order=order - prev_order, periodic=False)
            prev_order = order
            prefix = "d_" if order == 1 else f"d{order}_"
            q[f"{prefix}{name}_dx1_avg"] = d_2d.mean(axis=avg_axis)
    return q


class DatabaseCreator:
    """
    Build labeled NumPy tensors (T, F, X) from an OSIRIS simulation.

    Workflow
    --------
    1. Instantiate with a simulation, species, and save folder.
    2. Call ``set_limits(t0, t1)`` to define the time range.
    3. Call ``create_database(database=...)`` to build and save the tensors.

    Per timestep, the raw 2-D frames are loaded, passed through the spatial
    filters from ``DatabaseBuildConfig.filters`` (if any), then all
    x1-derivatives and e_vlasov are computed in 2-D, and only then is the
    transverse average taken.  What each tensor contains is controlled by
    the parameter block at the top of this module (``INPUT_FEATURE_LABELS``
    etc.).

    Supported database types
    ------------------------
    ``"input"``     — mean-field input tensor (rows = ``INPUT_FEATURE_LABELS``).
    ``"output"``    — eta tensor (formula selected by ``DatabaseBuildConfig.eta_formula``).
    ``"e_vlasov"``  — mean-field Vlasov electric field tensor.
    ``"both"``      — input + output, computed in a single pass.
    ``"all"``       — all three tensors above plus vnT, in a single pass.
    ``"vnT"``       — vnT tensor: ``VNT_BASE_QUANTITIES`` and their x1-derivatives
    of orders ``VNT_DERIV_ORDERS``, all transverse-averaged.
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
        name_vlasov: str = "e_vlasov_tensor",
        name_vnT: str = "vnT_tensor",
    ) -> None:
        """Build and save database tensors as ``.npy`` files in ``save_folder``.

        All requested tensors are computed in a **single pass** over the
        simulation frames — each raw field is loaded and filtered exactly
        once per timestep.  Tensors are written frame-by-frame to
        memory-mapped files, so the full tensor is never held in RAM, and
        progress is checkpointed every ``flush_every`` frames so interrupted
        jobs can be resumed with ``DatabaseBuildConfig(resume=True)``.

        Parameters
        ----------
        database :
            Which tensors to build. One of: ``"input"``, ``"output"``,
            ``"e_vlasov"``, ``"vnT"``, ``"both"`` (input + output), ``"all"`` (all).
        name_input, name_output, name_vlasov, name_vnT :
            File-stem names (without ``.npy``) for each tensor.
        """
        if database not in _VALID_DATABASE_TYPES:
            raise ValueError(f"Invalid database type '{database}'. Choose from: {sorted(_VALID_DATABASE_TYPES)}.")

        cfg = self.build_config
        if cfg.eta_formula not in _VALID_ETA_FORMULAS:
            raise ValueError(f"Invalid eta_formula '{cfg.eta_formula}'. Choose from: {sorted(_VALID_ETA_FORMULAS)}.")

        flags = cfg.ar_config or AnomalousResistivityConfig(species=self.species)
        if flags.include_time_derivative:
            raise NotImplementedError("include_time_derivative=True is not supported by the per-frame database pipeline.")

        os.makedirs(self.save_folder, exist_ok=True)

        if self.final_iter is None:
            logger.warning("set_limits() not called; inferring from simulation['e1'].")
            self.set_limits(0, None)

        build_input = database in {"input", "both", "all"}
        build_output = database in {"output", "both", "all"}
        build_vlasov = database in {"e_vlasov", "all"}
        build_vnT = database in {"vnT", "all"}
        need_mean_field = build_input or build_output or build_vlasov

        filt = as_filter(cfg.filters)
        dx = float(self.simulation["e1"].dx[0])  # longitudinal grid spacing
        dx2 = float(self.simulation["e1"].dx[1])  # transverse grid spacing
        avg_axis = cfg.mft_axis - 1  # 0-indexed numpy axis

        vnT_labels = vnT_feature_labels()

        # (name, row labels, validate) for each requested tensor, in output order.
        specs: list[tuple[str, list[str], bool]] = []
        if build_input:
            specs.append((name_input, INPUT_FEATURE_LABELS, False))
        if build_output:
            specs.append((name_output, OUTPUT_LABELS, cfg.validate_output))
        if build_vlasov:
            specs.append((name_vlasov, E_VLASOV_LABELS, cfg.validate_output))
        if build_vnT:
            specs.append((name_vnT, vnT_labels, False))

        raw = self._load_raw_diagnostics()

        def frame_fn(t_idx: int) -> list[np.ndarray]:
            frames: list[np.ndarray] = []
            if need_mean_field:
                q = _mean_field_frame_quantities(
                    raw,
                    t_idx,
                    filt,
                    dx,
                    dx2,
                    avg_axis,
                    flags,
                    eta_formula=cfg.eta_formula,
                    compute_e_vlasov=build_vlasov,
                    compute_eta=build_output,
                )
                if build_input:
                    frames.append(_stack_rows(q, INPUT_FEATURE_LABELS))
                if build_output:
                    frames.append(_stack_rows(q, OUTPUT_LABELS))
                if build_vlasov:
                    frames.append(_stack_rows(q, E_VLASOV_LABELS))
            if build_vnT:
                qv = _vnT_frame_quantities(raw, t_idx, filt, dx, avg_axis)
                frames.append(_stack_rows(qv, vnT_labels))
            return frames

        _timer = _start_timer(
            f"create_database({database}, T={self.T}, X={self.X}, filter={filt!r}, "
            f"dtype={cfg.dtype.__name__ if hasattr(cfg.dtype, '__name__') else cfg.dtype})"
        )
        try:
            for name, labels, _ in specs:
                logger.info("Tensor '%s': %d features — %s", name, len(labels), labels)
            self._build_tensors(specs, frame_fn, desc=f"Building '{database}' database")
            logger.info("All requested databases saved to '%s'.", self.save_folder)
        finally:
            _stop_timer(_timer)

    # ------------------------------------------------------------------
    # Raw field access
    # ------------------------------------------------------------------

    def _load_raw_diagnostics(self) -> dict[str, Any]:
        """Return lazy Diagnostic handles for every raw field the pipeline may need."""
        sp = self.species
        sim = self.simulation
        return {
            "n": sim[sp]["n"],
            "T11": sim[sp]["T11"],
            "T12": sim[sp]["T12"],
            "vfl1": sim[sp]["vfl1"],
            "vfl2": sim[sp]["vfl2"],
            "vfl3": sim[sp]["vfl3"],
            "b2": sim["b2"],
            "b3": sim["b3"],
        }

    # ------------------------------------------------------------------
    # Core tensor builder
    # ------------------------------------------------------------------

    def _build_tensors(
        self,
        specs: Sequence[tuple[str, Sequence[str], bool]],
        frame_fn: Callable[[int], Sequence[np.ndarray]],
        desc: str,
    ) -> None:
        """Stream frames into one memory-mapped ``.npy`` file per spec, single pass.

        Parameters
        ----------
        specs :
            ``(name, row_labels, validate)`` per tensor.  Tensor *i* gets shape
            ``(T, len(row_labels), X)``; ``validate=True`` replaces NaN/inf
            with 0 per frame.
        frame_fn :
            Callable returning one ``(F_i, X)`` array per spec for a given t_idx.
        desc :
            tqdm progress-bar description.

        The full tensors are **never held in RAM** — each frame is written
        directly to disk via ``numpy.lib.format.open_memmap``.

        Crash safety / resume
        ---------------------
        A single companion ``<first name>.npy.progress.npy`` file (a boolean
        array of length T) tracks which frames have been flushed to disk.  On
        the next run with ``DatabaseBuildConfig(resume=True)``, already-completed
        frames are skipped.  A frame is only marked *done* **after** the memmaps
        have been flushed, so on crash at worst the last ``flush_every`` frames
        are recomputed — no corrupt data.
        """
        if self.T <= 0:
            raise ValueError("Nothing to build: T <= 0. Did you call set_limits()?")
        if not specs:
            raise ValueError("No tensors requested.")

        dtype = self.build_config.dtype
        names = [name for name, _, _ in specs]
        save_paths = [os.path.join(self.save_folder, f"{name}.npy") for name in names]
        shapes = [(self.T, len(labels), self.X) for _, labels, _ in specs]
        validates = [validate for _, _, validate in specs]
        # NOTE: end the progress path in `.npy` so np.save does NOT silently
        # append another `.npy` (it does when the name lacks the extension).
        # Otherwise the checkpoint is written as `<save>.npy.progress.npy` while
        # np.load / os.remove look for `<save>.npy.progress` — so resume never
        # finds it (restarts from scratch) and stray files are never cleaned up.
        progress_path = f"{save_paths[0]}.progress.npy"

        # ── Open / create the memory-mapped files ─────────────────────────
        can_resume = self.build_config.resume and all(os.path.exists(p) for p in save_paths) and os.path.exists(progress_path)
        if can_resume:
            done_mask = np.load(progress_path)
            if done_mask.shape != (self.T,):
                logger.warning(
                    "Progress file shape %s != (%d,); restarting %s from scratch.",
                    done_mask.shape,
                    self.T,
                    names,
                )
                done_mask = np.zeros(self.T, dtype=bool)
                arrs = [np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=s) for p, s in zip(save_paths, shapes, strict=True)]
            else:
                arrs = [np.lib.format.open_memmap(p, mode="r+", dtype=dtype, shape=s) for p, s in zip(save_paths, shapes, strict=True)]
                logger.info("Resuming %s: %d / %d frames already done.", names, int(done_mask.sum()), self.T)
        else:
            done_mask = np.zeros(self.T, dtype=bool)
            arrs = [np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=s) for p, s in zip(save_paths, shapes, strict=True)]

        work_items = [(out_i, t_idx) for out_i, t_idx in enumerate(range(self.initial_iter, self.final_iter)) if not done_mask[out_i]]

        if not work_items:
            logger.info("%s already complete — skipping.", names)
            arrs.clear()  # close memmap handles
            return

        # ── Thread-safe flush / progress helpers ──────────────────────────
        _lock = threading.Lock()
        _n_written = [0]

        def _flush_all() -> None:
            for arr in arrs:
                arr.flush()
            np.save(progress_path, done_mask.copy())

        def _write_frame(out_i: int, t_idx: int) -> None:
            frames = frame_fn(t_idx)
            if len(frames) != len(specs):
                raise ValueError(f"frame_fn returned {len(frames)} arrays but {len(specs)} tensors were requested.")
            for arr, frame, name, validate in zip(arrs, frames, names, validates, strict=True):
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
                    _flush_all()

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
                                _flush_all()
                            raise RuntimeError(f"Failed building {names} at output index {out_i} (simulation t_idx={t_idx}).") from e
                        pbar.update(1)

        # ── Final flush and cleanup ────────────────────────────────────────
        for arr in arrs:
            arr.flush()
        arrs.clear()  # close memmap handles (does not delete the files)

        if os.path.exists(progress_path):
            os.remove(progress_path)

        for name, shape, save_path in zip(names, shapes, save_paths, strict=True):
            size_gb = (shape[0] * shape[1] * shape[2] * np.dtype(dtype).itemsize) / 1e9
            logger.info("Saved '%s' (%.2f GB, %s) → %s", name, size_gb, "x".join(str(s) for s in shape), save_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def feature_labels(self) -> list[str]:
        """Ordered list of input feature labels."""
        return list(INPUT_FEATURE_LABELS)

    @property
    def output_labels(self) -> list[str]:
        """Ordered list of output (eta) labels."""
        return list(OUTPUT_LABELS)

    @property
    def vnT_labels(self) -> list[str]:
        """Ordered list of vnT feature labels."""
        return vnT_feature_labels()

    @property
    def x(self) -> Any:
        return self.simulation["e1"].x[0]

    @property
    def dx(self) -> Any:
        return self.simulation["e1"].dx[0]


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
