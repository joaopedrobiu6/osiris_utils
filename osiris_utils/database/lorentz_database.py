from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import tqdm

from .database import DatabaseCreator, _clean_frame

logger = logging.getLogger(__name__)

__all__ = ["LorentzDatabaseBuildConfig", "LorentzDatabaseCreator"]

# 22 input features — mirrors the input tensor from DatabaseCreator._input_feature_specs
LORENTZ_FEATURE_LABELS: list[str] = [
    "n_avg",
    "b2_avg",
    "b3_avg",
    "vfl1_avg",
    "vfl2_avg",
    "vfl3_avg",
    "T11_avg",
    "T12_avg",
    "dvfl1_dx1_avg",
    "dvfl2_dx1_avg",
    "dvfl3_dx1_avg",
    "dn_dx1_avg",
    "dT11_dx1_avg",
    "db2_dx1_avg",
    "db3_dx1_avg",
    "d2_vfl1_dx1_avg",
    "d2_vfl2_dx1_avg",
    "d2_vfl3_dx1_avg",
    "d2_b2_dx1_avg",
    "d2_b3_dx1_avg",
    "d2_n_dx1_avg",
    "dnT11_dx1_avg",
]
_N_FEATURES = len(LORENTZ_FEATURE_LABELS)

# 1 output feature — boosted anomalous resistivity (mean-field Ohm's law residual)
# $\eta' = \langle e'_{vlasov}\rangle
#          + \langle v'_x\rangle\,\partial_{x'}\langle v'_x\rangle
#          + \frac{1}{\langle n'\rangle}\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)
#          + \langle v_y\rangle\langle B'_z\rangle - \langle v_z\rangle\langle B'_y\rangle$
LORENTZ_OUTPUT_LABELS: list[str] = ["eta_avg"]
_N_OUTPUT = len(LORENTZ_OUTPUT_LABELS)

_VALID_DATABASE_TYPES = {"input", "output", "both"}


@dataclass(frozen=True)
class LorentzDatabaseBuildConfig:
    r"""Configuration for :class:`LorentzDatabaseCreator`.

    Parameters
    ----------
    dtype :
        NumPy dtype for saved tensors.
    max_workers :
        Worker threads for parallel frame building.
    mft_axis :
        Axis along which the transverse average is taken (1-indexed, OSIRIS convention).
        Default 2 means average over x2 (the y-direction).
    boost_min, boost_max :
        Range for the uniform distribution from which :math:`\beta = v/c` is sampled per
        timestep.  ``boost_max`` should be < 1; values >= 1 produce NaN
        via :math:`\gamma = 1/\sqrt{1 - \beta^2}`.
    seed :
        Seed for the NumPy default_rng used to draw β values.
        Pass an integer for reproducible augmentation.
    validate_output :
        Replace NaN/inf with 0 per frame in the output tensor when True.
    resume :
        If True, reuse the existing boost_velocities file and skip already-written frames.
    flush_every :
        Flush memory-mapped output(s) every N completed frames.
    """

    dtype: type = np.float32
    max_workers: int | None = None
    mft_axis: int = 2
    boost_min: float = 0.0
    boost_max: float = 0.9
    seed: int | None = None
    validate_output: bool = True
    resume: bool = False
    flush_every: int = 128


class LorentzDatabaseCreator(DatabaseCreator):
    r"""
    Build ``(T, F, X)`` input and ``(T, 1, X)`` output tensors by applying a
    random x-direction Lorentz boost independently at each timestep, then
    transverse-averaging.

    Pipeline per timestep *t*:

        raw (nx, ny) frames
          → Lorentz boost with :math:`\beta_t`                    [same (nx, ny)]
          → transverse mean :math:`\langle\cdot\rangle_y`          [(nx,)]
          → :math:`\gamma\,\partial/\partial x` derivatives        [(nx,)]
          → stack 22 input features + 1 output (η)                 [(23, nx)]

    The input and output are always computed in a single pass per timestep —
    fields are loaded from disk exactly once regardless of which tensors are
    requested.  :math:`\beta_t \sim \mathcal{U}(\beta_{\min}, \beta_{\max})` is
    drawn once, saved to disk, and reused on resume.

    Notes
    -----
    - Transverse velocities ``vfl2``, ``vfl3`` are **not** transformed (follows
      the convention in the ``lorentz_transform`` notebook class).
    - Spatial derivatives use the covariant approximation

      .. math::
          \frac{\partial f}{\partial x'} \approx \gamma \frac{\partial f}{\partial x}

      (the :math:`\partial f/\partial t` term is neglected).
    - ``T11`` and ``T12`` are derived from the full pressure-tensor transforms
      :math:`P'_{11}` and :math:`P'_{12}` divided by :math:`n'`.  In OSIRIS,

      .. math::
          P_{11} = \int u_1^2 \, f \, d^3p, \qquad P_{00} = \int \gamma^2 \, f \, d^3p.
    """

    def __init__(
        self,
        simulation,
        species: str,
        save_folder: str,
        build_config: LorentzDatabaseBuildConfig | None = None,
    ) -> None:
        self.simulation = simulation
        self.species = species
        self.save_folder = save_folder
        self.build_config = build_config or LorentzDatabaseBuildConfig()

        self.initial_iter: int = 0
        self.final_iter: int | None = None
        self.T: int = 0
        self.X: int = 0

        # Unused inherited attributes — kept to avoid AttributeError in inherited methods
        self._ar = None
        self._sim_mft = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_database(
        self,
        database: str = "both",
        name_input: str = "lorentz_tensor",
        name_output: str = "lorentz_output",
        name_boosts: str = "boost_velocities",
    ) -> None:
        """Build Lorentz-augmented tensors and save them to disk.

        The input ``(T, 22, X)`` and output ``(T, 1, X)`` tensors are always
        produced from a **single pass** over the simulation frames — fields are
        loaded from disk only once per timestep.

        Parameters
        ----------
        database :
            Which tensors to save: ``"input"``, ``"output"``, or ``"both"``.
        name_input :
            File stem (without ``.npy``) for the 22-feature input tensor.
        name_output :
            File stem for the 1-feature output tensor (boosted η).
        name_boosts :
            File stem for the ``(T,)`` array of per-timestep :math:`\\beta` values.
            Saved before any frames are written so a crashed job resumes with
            the same boost sequence.
        """
        if database not in _VALID_DATABASE_TYPES:
            raise ValueError(
                f"Invalid database '{database}'. Choose from: {sorted(_VALID_DATABASE_TYPES)}."
            )

        os.makedirs(self.save_folder, exist_ok=True)

        if self.final_iter is None:
            logger.warning("set_limits() not called; inferring from simulation['e1'].")
            self.set_limits(0, None)

        # β values generated once, shared across input and output builds.
        betas = self._load_or_generate_betas(name_boosts)

        dx       = float(self.simulation["e1"].dx[0])   # longitudinal grid spacing
        dx2      = float(self.simulation["e1"].dx[1])   # transverse grid spacing
        avg_axis = self.build_config.mft_axis - 1       # 0-indexed numpy axis

        raw = self._load_raw_diagnostics()

        def get_combined_frame(t_idx: int) -> np.ndarray:
            beta = float(betas[t_idx - self.initial_iter])
            return _boost_combined_frame(raw, t_idx, beta, dx, dx2, avg_axis)

        if database == "both":
            # Single pass: write input and output simultaneously.
            logger.info(
                "Lorentz input + output (single pass): T=%d, X=%d", self.T, self.X
            )
            self._build_two_tensors(
                name_a=name_input,
                name_b=name_output,
                shape_a=(self.T, _N_FEATURES, self.X),
                shape_b=(self.T, _N_OUTPUT, self.X),
                frame_fn=get_combined_frame,
                n_a=_N_FEATURES,
                desc="Building Lorentz-boosted input + output tensors",
                validate_b=self.build_config.validate_output,
            )

        elif database == "input":
            logger.info("Lorentz input tensor: T=%d, X=%d", self.T, self.X)
            self._build_tensor(
                name=name_input,
                shape=(self.T, _N_FEATURES, self.X),
                frame_fn=lambda t: get_combined_frame(t)[:_N_FEATURES],
                desc="Building Lorentz-boosted input tensor",
                validate=False,
            )

        else:  # "output"
            logger.info("Lorentz output tensor: T=%d, X=%d", self.T, self.X)
            self._build_tensor(
                name=name_output,
                shape=(self.T, _N_OUTPUT, self.X),
                frame_fn=lambda t: get_combined_frame(t)[_N_FEATURES:],
                desc="Building Lorentz-boosted output tensor",
                validate=self.build_config.validate_output,
            )

    @property
    def feature_labels(self) -> list[str]:
        return list(LORENTZ_FEATURE_LABELS)

    @property
    def output_labels(self) -> list[str]:
        return list(LORENTZ_OUTPUT_LABELS)

    # ------------------------------------------------------------------
    # Single-pass two-tensor builder
    # ------------------------------------------------------------------

    def _build_two_tensors(
        self,
        name_a: str,
        name_b: str,
        shape_a: tuple,
        shape_b: tuple,
        frame_fn: Callable[[int], np.ndarray],
        n_a: int,
        desc: str,
        validate_b: bool = False,
    ) -> None:
        """Stream frames to two memory-mapped files in a single pass.

        ``frame_fn(t_idx)`` must return an array of shape ``(n_a + n_b, X)``.
        The first ``n_a`` rows are written to tensor A (input); the rest to B (output).
        A single progress file (keyed to ``name_a``) tracks completed frames so
        interrupted SLURM jobs can resume with ``resume=True``.

        Parameters
        ----------
        name_a, name_b :
            File stems (without ``.npy``) for the two output tensors.
        shape_a, shape_b :
            ``(T, F_a, X)`` and ``(T, F_b, X)`` shapes.
        frame_fn :
            Callable returning a ``(F_a + F_b, X)`` array for a given t_idx.
        n_a :
            Number of feature rows belonging to tensor A.
        validate_b :
            Replace NaN/inf with 0 in tensor B frames when True.
        """
        save_path_a  = os.path.join(self.save_folder, f"{name_a}.npy")
        save_path_b  = os.path.join(self.save_folder, f"{name_b}.npy")
        progress_path = f"{save_path_a}.progress"
        dtype = self.build_config.dtype

        # ── Open / create the memory-mapped files ─────────────────────
        if (
            self.build_config.resume
            and os.path.exists(save_path_a)
            and os.path.exists(save_path_b)
            and os.path.exists(progress_path)
        ):
            done_mask = np.load(progress_path)
            if done_mask.shape != (self.T,):
                logger.warning(
                    "Progress file shape %s != (%d,); restarting from scratch.",
                    done_mask.shape, self.T,
                )
                done_mask = np.zeros(self.T, dtype=bool)
                arr_a = np.lib.format.open_memmap(save_path_a, mode="w+", dtype=dtype, shape=shape_a)
                arr_b = np.lib.format.open_memmap(save_path_b, mode="w+", dtype=dtype, shape=shape_b)
            else:
                arr_a = np.lib.format.open_memmap(save_path_a, mode="r+", dtype=dtype, shape=shape_a)
                arr_b = np.lib.format.open_memmap(save_path_b, mode="r+", dtype=dtype, shape=shape_b)
                logger.info(
                    "Resuming '%s'+'%s': %d / %d frames done.",
                    name_a, name_b, int(done_mask.sum()), self.T,
                )
        else:
            done_mask = np.zeros(self.T, dtype=bool)
            arr_a = np.lib.format.open_memmap(save_path_a, mode="w+", dtype=dtype, shape=shape_a)
            arr_b = np.lib.format.open_memmap(save_path_b, mode="w+", dtype=dtype, shape=shape_b)

        work_items = [
            (out_i, t_idx)
            for out_i, t_idx in enumerate(range(self.initial_iter, self.final_iter))
            if not done_mask[out_i]
        ]

        if not work_items:
            logger.info("'%s' + '%s' already complete — skipping.", name_a, name_b)
            del arr_a, arr_b
            return

        # ── Thread-safe write / flush helpers ─────────────────────────
        _lock = threading.Lock()
        _n_written = [0]

        def _write_frame(out_i: int, t_idx: int) -> None:
            combined = frame_fn(t_idx)           # (n_a + n_b, X)
            frame_a  = combined[:n_a]
            frame_b  = combined[n_a:]
            if validate_b:
                frame_b = _clean_frame(frame_b, name_b, out_i)
            arr_a[out_i] = frame_a.astype(dtype, copy=False)
            arr_b[out_i] = frame_b.astype(dtype, copy=False)
            with _lock:
                done_mask[out_i] = True
                _n_written[0] += 1
                if _n_written[0] % self.build_config.flush_every == 0:
                    arr_a.flush()
                    arr_b.flush()
                    np.save(progress_path, done_mask.copy())

        # ── Warm-up: first frame on main thread ───────────────────────
        _write_frame(*work_items[0])
        remaining = work_items[1:]

        # ── Parallel frame building ────────────────────────────────────
        if remaining:
            with ThreadPoolExecutor(max_workers=self.build_config.max_workers) as ex:
                futures = {
                    ex.submit(_write_frame, out_i, t_idx): (out_i, t_idx)
                    for out_i, t_idx in remaining
                }
                with tqdm.tqdm(total=len(work_items), initial=1, desc=desc) as pbar:
                    for fut in as_completed(futures):
                        out_i, t_idx = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            with _lock:
                                arr_a.flush()
                                arr_b.flush()
                                np.save(progress_path, done_mask.copy())
                            raise RuntimeError(
                                f"Failed building '{name_a}'/'{name_b}' at output "
                                f"index {out_i} (t_idx={t_idx})."
                            ) from e
                        pbar.update(1)

        # ── Final flush and cleanup ────────────────────────────────────
        arr_a.flush()
        arr_b.flush()
        del arr_a, arr_b

        if os.path.exists(progress_path):
            os.remove(progress_path)

        size_a = shape_a[0] * shape_a[1] * shape_a[2] * np.dtype(dtype).itemsize / 1e9
        size_b = shape_b[0] * shape_b[1] * shape_b[2] * np.dtype(dtype).itemsize / 1e9
        logger.info(
            "Saved '%s' (%.2f GB) + '%s' (%.2f GB) → %s",
            name_a, size_a, name_b, size_b, self.save_folder,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_or_generate_betas(self, name_boosts: str) -> np.ndarray:
        r"""Load existing :math:`\beta` array when resuming, otherwise sample and save a new one."""
        betas_path = os.path.join(self.save_folder, f"{name_boosts}.npy")

        if self.build_config.resume and os.path.exists(betas_path):
            betas = np.load(betas_path)
            if betas.shape != (self.T,):
                raise ValueError(
                    f"Existing boost file has shape {betas.shape} but T={self.T}. "
                    f"Delete the file or set resume=False to restart."
                )
            logger.info("Resuming: loaded boost velocities from %s", betas_path)
            return betas

        rng = np.random.default_rng(self.build_config.seed)
        betas = rng.uniform(
            self.build_config.boost_min,
            self.build_config.boost_max,
            size=self.T,
        ).astype(np.float32)
        np.save(betas_path, betas)
        logger.info("Saved %d boost velocities → %s", self.T, betas_path)
        return betas

    def _load_raw_diagnostics(self) -> dict:
        """Return lazy Diagnostic handles for every field needed by the boost."""
        sp  = self.species
        sim = self.simulation
        return {
            "n":    sim[sp]["n"],
            "e2":   sim["e2"],
            "e3":   sim["e3"],
            "b2":   sim["b2"],
            "b3":   sim["b3"],
            "vfl1": sim[sp]["vfl1"],
            "vfl2": sim[sp]["vfl2"],
            "vfl3": sim[sp]["vfl3"],
            "P11":  sim[sp]["P11"],
            "P12":  sim[sp]["P12"],
            "P00":  sim[sp]["P00"],
            "ufl1": sim[sp]["ufl1"],
            "ufl2": sim[sp]["ufl2"],
        }


# ----------------------------------------------------------------------
# Finite-difference helper
# ----------------------------------------------------------------------


def _grad4(f: np.ndarray, dx: float, axis: int, periodic: bool = False) -> np.ndarray:
    r"""4th-order finite difference of *f* along *axis*.

    Interior points use the standard 5-point centered stencil:

    .. math::
        f'_i = \frac{f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}}{12\,\Delta x}

    Boundary treatment
    ------------------
    periodic=True (transverse / x2 direction)
        Wrap-around indexing via ``np.roll``; all points use the centered stencil.
    periodic=False (longitudinal / x1 direction)
        4th-order one-sided stencils at the four edge points:

        .. math::
            f'_0     &= \frac{-25f_0 + 48f_1 - 36f_2 + 16f_3 - 3f_4}
                              {12\,\Delta x}  \\[4pt]
            f'_1     &= \frac{-3f_0 - 10f_1 + 18f_2 - 6f_3 + f_4}
                              {12\,\Delta x}  \\[4pt]
            f'_{N-2} &= \frac{-f_{N-5} + 6f_{N-4} - 18f_{N-3} + 10f_{N-2} + 3f_{N-1}}
                              {12\,\Delta x}  \\[4pt]
            f'_{N-1} &= \frac{3f_{N-5} - 16f_{N-4} + 36f_{N-3} - 48f_{N-2} + 25f_{N-1}}
                              {12\,\Delta x}

    Requires at least 5 points along *axis*.
    """
    if periodic:
        return (
            np.roll(f,  2, axis=axis)
            - 8.0 * np.roll(f,  1, axis=axis)
            + 8.0 * np.roll(f, -1, axis=axis)
            -       np.roll(f, -2, axis=axis)
        ) / (12.0 * dx)

    # Move the target axis to front for clean 1-D slicing
    fm  = np.moveaxis(f, axis, 0)
    out = np.empty_like(fm)

    # Interior: centered stencil
    out[2:-2] = (fm[:-4] - 8.0 * fm[1:-3] + 8.0 * fm[3:-1] - fm[4:]) / (12.0 * dx)

    # Left boundary: forward-biased
    out[0] = (-25.0 * fm[0] + 48.0 * fm[1] - 36.0 * fm[2] + 16.0 * fm[3] -  3.0 * fm[4]) / (12.0 * dx)
    out[1] = ( -3.0 * fm[0] - 10.0 * fm[1] + 18.0 * fm[2] -  6.0 * fm[3] +        fm[4]) / (12.0 * dx)

    # Right boundary: backward-biased
    out[-2] = (      -fm[-5] + 6.0 * fm[-4] - 18.0 * fm[-3] + 10.0 * fm[-2] +  3.0 * fm[-1]) / (12.0 * dx)
    out[-1] = (3.0 * fm[-5] - 16.0 * fm[-4] + 36.0 * fm[-3] - 48.0 * fm[-2] + 25.0 * fm[-1]) / (12.0 * dx)

    return np.moveaxis(out, 0, axis)


# ----------------------------------------------------------------------
# Combined frame computation (module-level for clean stack traces)
# ----------------------------------------------------------------------


def _boost_combined_frame(
    raw: dict,
    t_idx: int,
    beta: float,
    dx: float,
    dx2: float,
    avg_axis: int,
) -> np.ndarray:
    r"""Compute all 22 input features **and** the output η in a single pass.

    Fields are loaded from disk once and the Lorentz transforms are applied
    once — no work is duplicated regardless of which tensors are saved.

    Returns a ``(23, X)`` float64 array:

    * rows ``0–21`` — input features (order matches :data:`LORENTZ_FEATURE_LABELS`)
    * row ``22``    — boosted anomalous resistivity η

    Lorentz transforms applied
    --------------------------
    :math:`D \equiv 1 - \beta v_x`

    .. math::
        n'     &= \gamma n D \\
        B'_y   &= \gamma(B_y + \beta E_z) \\
        B'_z   &= \gamma(B_z - \beta E_y) \\
        v'_x   &= (v_x - \beta)/D \\
        P'_{11}&= \gamma^2(P_{11} - 2\beta n u_1 + \beta^2 n P_{00})
                  - \gamma n(v_x-\beta)\,\gamma[(1+\beta^2)u_1
                  - \beta P_{11}/n - \beta P_{00}]\,/\,D \\
        P'_{12}&= \gamma^2(P_{12} - 2\beta n u_2)
                  - \gamma n(v_x-\beta)(u_2 - \beta P_{12}/n)\,/\,D \\
        T'_{11}&= P'_{11}/n', \quad T'_{12} = P'_{12}/n'

    :math:`v_y,\,v_z` are left untransformed (notebook convention).

    Output η (Steps 1–3)
    --------------------
    **Step 1** — boosted 2-D :math:`e'_{vlasov}` (no :math:`E_x`):

    .. math::
        e'_{vlasov} = -v'_x\,\partial_{x'} v'_x
                      - \frac{1}{n'}\bigl(
                          \partial_{x'}(n' T'_{11})
                        + \partial_y(n' T'_{12})\bigr)
                      - v_y B'_z + v_z B'_y

    **Step 2** — transverse average :math:`\langle e'_{vlasov}\rangle_y`.

    **Step 3** — mean-field corrections (reusing quantities already computed
    for the input features):

    .. math::
        \eta' = \langle e'_{vlasov}\rangle
              + \langle v'_x\rangle \underbrace{\partial_{x'}\langle v'_x\rangle}_{
                  \text{input feature 9}}
              + \frac{1}{\langle n'\rangle}
                \underbrace{\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)}_{
                  \text{input feature 22}}
              + \langle v_y\rangle\langle B'_z\rangle
              - \langle v_z\rangle\langle B'_y\rangle
    """
    gamma  = 1.0 / np.sqrt(1.0 - beta * beta)
    x_axis = 1 - avg_axis  # longitudinal axis in 2-D arrays (avg_axis=1 → x_axis=0)

    # ── Load raw 2-D fields ───────────────────────────────────────────
    n    = np.asarray(raw["n"][t_idx],    dtype=np.float64)
    e2   = np.asarray(raw["e2"][t_idx],   dtype=np.float64)
    e3   = np.asarray(raw["e3"][t_idx],   dtype=np.float64)
    b2   = np.asarray(raw["b2"][t_idx],   dtype=np.float64)
    b3   = np.asarray(raw["b3"][t_idx],   dtype=np.float64)
    vfl1 = np.asarray(raw["vfl1"][t_idx], dtype=np.float64)
    vfl2 = np.asarray(raw["vfl2"][t_idx], dtype=np.float64)
    vfl3 = np.asarray(raw["vfl3"][t_idx], dtype=np.float64)
    P11  = np.asarray(raw["P11"][t_idx],  dtype=np.float64)
    P12  = np.asarray(raw["P12"][t_idx],  dtype=np.float64)
    P00  = np.asarray(raw["P00"][t_idx],  dtype=np.float64)
    ufl1 = np.asarray(raw["ufl1"][t_idx], dtype=np.float64)
    ufl2 = np.asarray(raw["ufl2"][t_idx], dtype=np.float64)

    # ── Lorentz transforms ────────────────────────────────────────────
    # $D \equiv 1 - \beta v_x$
    denom = 1.0 - beta * vfl1

    # $n' = \gamma\,n\,D$
    n_t = gamma * n * denom

    # $B'_y = \gamma(B_y + \beta E_z)$,  $B'_z = \gamma(B_z - \beta E_y)$
    b2_t = gamma * (b2 + beta * e3)
    b3_t = gamma * (b3 - beta * e2)

    # $v'_x = (v_x - \beta)/D$ ;  $v_y,\,v_z$ untransformed
    vfl1_t = (vfl1 - beta) / denom

    # $P'_{11} = \gamma^2(P_{11} - 2\beta n u_1 + \beta^2 n P_{00})$
    # $\quad - \gamma n(v_x-\beta)\,\gamma[(1+\beta^2)u_1
    #          - \beta P_{11}/n - \beta P_{00}]\,/\,D$
    p11_t = (
        gamma**2 * (P11 - 2.0 * beta * n * ufl1 + beta**2 * n * P00)
        - gamma * n * (vfl1 - beta)
        * (gamma * ((1.0 + beta**2) * ufl1 - beta * (P11 / n) - beta * P00))
        / denom
    )

    # $P'_{12} = \gamma^2(P_{12} - 2\beta n u_2)
    #            - \gamma n(v_x-\beta)(u_2 - \beta P_{12}/n)\,/\,D$
    p12_t = (
        gamma**2 * (P12 - 2.0 * beta * n * ufl2)
        - gamma * n * (vfl1 - beta) * (ufl2 - beta * (P12 / n)) / denom
    )

    # $T'_{11} = P'_{11}/n'$,  $T'_{12} = P'_{12}/n'$
    T11_t = p11_t / n_t
    T12_t = p12_t / n_t

    # ── Transverse average $\langle\cdot\rangle_y$ ───────────────────
    n_avg    = n_t.mean(axis=avg_axis)
    b2_avg   = b2_t.mean(axis=avg_axis)
    b3_avg   = b3_t.mean(axis=avg_axis)
    vfl1_avg = vfl1_t.mean(axis=avg_axis)
    vfl2_avg = vfl2.mean(axis=avg_axis)    # untransformed
    vfl3_avg = vfl3.mean(axis=avg_axis)    # untransformed
    T11_avg  = T11_t.mean(axis=avg_axis)
    T12_avg  = T12_t.mean(axis=avg_axis)

    # ── 1-D derivatives on averaged fields (longitudinal, non-periodic) ──
    # Covariant approximation: $\partial/\partial x' \approx \gamma\,\partial/\partial x$
    def _d(f: np.ndarray) -> np.ndarray:
        return gamma * _grad4(f, dx, axis=0)

    dvfl1 = _d(vfl1_avg)
    dvfl2 = _d(vfl2_avg)
    dvfl3 = _d(vfl3_avg)
    dn    = _d(n_avg)
    dT11  = _d(T11_avg)
    db2   = _d(b2_avg)
    db3   = _d(b3_avg)
    dnT11 = _d(n_avg * T11_avg)   # reused in Step 3 below

    # ── Input features (22, X) ────────────────────────────────────────
    input_features = np.stack([
        n_avg,        # 0
        b2_avg,       # 1
        b3_avg,       # 2
        vfl1_avg,     # 3
        vfl2_avg,     # 4
        vfl3_avg,     # 5
        T11_avg,      # 6
        T12_avg,      # 7
        dvfl1,        # 8   ∂v'x/∂x'
        dvfl2,        # 9
        dvfl3,        # 10
        dn,           # 11
        dT11,         # 12
        db2,          # 13
        db3,          # 14
        _d(dvfl1),    # 15  ∂²v'x/∂x'²
        _d(dvfl2),    # 16
        _d(dvfl3),    # 17
        _d(db2),      # 18
        _d(db3),      # 19
        _d(dn),       # 20
        dnT11,        # 21  ∂(n'T'11)/∂x'  — also used in η Step 3
    ])  # (22, X)

    # ── Output: boosted η (1, X) ──────────────────────────────────────
    # Longitudinal 2-D derivative: $\partial/\partial x' \approx \gamma\,\partial/\partial x$
    # (non-periodic boundaries)
    def _d2d(f: np.ndarray) -> np.ndarray:
        return gamma * _grad4(f, dx, axis=x_axis, periodic=False)

    # Transverse 2-D derivative: $\partial/\partial y' = \partial/\partial y$
    # (no $\gamma$; periodic wrap-around for standard shock-sim BCs)
    def _d2d_y(f: np.ndarray) -> np.ndarray:
        return _grad4(f, dx2, axis=avg_axis, periodic=True)

    # Step 1: 2-D e_vlasov on full boosted field (no E_x)
    # $e'_{vlasov} = -v'_x\,\partial_{x'} v'_x
    #                - \frac{1}{n'}\bigl(\partial_{x'}(n' T'_{11})
    #                                   + \partial_y(n' T'_{12})\bigr)
    #                - v_y B'_z + v_z B'_y$
    e_vlasov_2d = (
        -vfl1_t * _d2d(vfl1_t)
        - (1.0 / n_t) * (_d2d(n_t * T11_t) + _d2d_y(n_t * T12_t))
        - vfl2 * b3_t
        + vfl3 * b2_t
    )

    # Step 2: transverse average
    e_vlasov_avg = e_vlasov_2d.mean(axis=avg_axis)

    # Step 3: mean-field corrections — reuse dvfl1 and dnT11 from input features
    # $\eta' = \langle e'_{vlasov}\rangle
    #         + \langle v'_x\rangle\,\partial_{x'}\langle v'_x\rangle
    #         + \frac{1}{\langle n'\rangle}\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)
    #         + \langle v_y\rangle\langle B'_z\rangle - \langle v_z\rangle\langle B'_y\rangle$
    eta = (
        e_vlasov_avg
        + vfl1_avg * dvfl1              # dvfl1 = input feature 8
        + (1.0 / n_avg) * dnT11         # dnT11 = input feature 21
        + vfl2_avg * b3_avg
        - vfl3_avg * b2_avg
    )

    return np.concatenate([input_features, np.stack([eta])], axis=0)  # (23, X)
