from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .database import DatabaseCreator
from .filters import SpatialFilter, as_filter

logger = logging.getLogger(__name__)

__all__ = ["LorentzDatabaseBuildConfig", "LorentzDatabaseCreator"]

# =====================================================================
# Database parameters — edit here to change what the tensors contain
# =====================================================================

# 22 input features — mirrors the input tensor from DatabaseCreator (INPUT_FEATURE_LABELS)
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
    filters :
        Spatial filters applied, in order, to every raw 2-D field right after
        loading — *before* the Lorentz boost and the transverse average.
        Empty (default) = no filtering, 4th-order finite-difference
        derivatives.  Filters like
        :class:`~osiris_utils.database.filters.SavitzkyGolayFilter` also
        supply their own analytic derivative scheme, used for every
        derivative in the pipeline.
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
    filters: Sequence[SpatialFilter] | SpatialFilter | None = ()
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
          → spatial filter (2-D, from ``build_config.filters``)   [same (nx, ny)]
          → Lorentz boost with :math:`\beta_t`                    [same (nx, ny)]
          → :math:`\gamma\,\partial/\partial x` derivatives (2-D) [same (nx, ny)]
          → transverse mean :math:`\langle\cdot\rangle_y`          [(nx,)]
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
            raise ValueError(f"Invalid database '{database}'. Choose from: {sorted(_VALID_DATABASE_TYPES)}.")

        os.makedirs(self.save_folder, exist_ok=True)

        if self.final_iter is None:
            logger.warning("set_limits() not called; inferring from simulation['e1'].")
            self.set_limits(0, None)

        # β values generated once, shared across input and output builds.
        betas = self._load_or_generate_betas(name_boosts)

        dx = float(self.simulation["e1"].dx[0])  # longitudinal grid spacing
        dx2 = float(self.simulation["e1"].dx[1])  # transverse grid spacing
        avg_axis = self.build_config.mft_axis - 1  # 0-indexed numpy axis
        filt = as_filter(self.build_config.filters)

        raw = self._load_raw_diagnostics()

        def get_combined_frame(t_idx: int) -> np.ndarray:
            beta = float(betas[t_idx - self.initial_iter])
            return _boost_combined_frame(raw, t_idx, beta, dx, dx2, avg_axis, filt)

        specs: list[tuple[str, list[str], bool]] = []
        splits: list[slice] = []
        if database in {"input", "both"}:
            specs.append((name_input, LORENTZ_FEATURE_LABELS, False))
            splits.append(slice(0, _N_FEATURES))
        if database in {"output", "both"}:
            specs.append((name_output, LORENTZ_OUTPUT_LABELS, self.build_config.validate_output))
            splits.append(slice(_N_FEATURES, _N_FEATURES + _N_OUTPUT))

        def frame_fn(t_idx: int) -> list[np.ndarray]:
            combined = get_combined_frame(t_idx)  # (23, X)
            return [combined[sl] for sl in splits]

        logger.info("Lorentz database '%s': T=%d, X=%d, filter=%r", database, self.T, self.X, filt)
        self._build_tensors(specs, frame_fn, desc=f"Building Lorentz-boosted '{database}' tensors")

    @property
    def feature_labels(self) -> list[str]:
        return list(LORENTZ_FEATURE_LABELS)

    @property
    def output_labels(self) -> list[str]:
        return list(LORENTZ_OUTPUT_LABELS)

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
                    f"Existing boost file has shape {betas.shape} but T={self.T}. Delete the file or set resume=False to restart."
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
        sp = self.species
        sim = self.simulation
        return {
            "n": sim[sp]["n"],
            "e2": sim["e2"],
            "e3": sim["e3"],
            "b2": sim["b2"],
            "b3": sim["b3"],
            "vfl1": sim[sp]["vfl1"],
            "vfl2": sim[sp]["vfl2"],
            "vfl3": sim[sp]["vfl3"],
            "P11": sim[sp]["P11"],
            "P12": sim[sp]["P12"],
            "P00": sim[sp]["P00"],
            "ufl1": sim[sp]["ufl1"],
            "ufl2": sim[sp]["ufl2"],
        }


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
    filt: SpatialFilter,
) -> np.ndarray:
    r"""Compute all 22 input features **and** the output η in a single pass.

    Fields are loaded from disk once, smoothed by *filt* (before the boost —
    the boost transforms are pointwise, so they stay valid on filtered data),
    and the Lorentz transforms are applied once — no work is duplicated
    regardless of which tensors are saved.  All derivatives use
    ``filt.derivative`` (4th-order finite differences for
    :class:`~osiris_utils.database.filters.NoFilter`).

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
              + \langle v'_x\rangle \underbrace{\langle\partial_{x'} v'_x\rangle}_{
                  \text{input feature 9}}
              + \frac{1}{\langle n'\rangle}
                \underbrace{\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)}_{
                  \text{input feature 22}}
              + \langle v_y\rangle\langle B'_z\rangle
              - \langle v_z\rangle\langle B'_y\rangle

    All per-field derivatives are taken on the boosted 2-D data **before**
    the transverse average; only the mean-field composite
    :math:`\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)` is, by
    definition, computed after it.
    """
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    x_axis = 1 - avg_axis  # longitudinal axis in 2-D arrays (avg_axis=1 → x_axis=0)
    periodic = tuple(ax == avg_axis for ax in range(2))

    # ── Load raw 2-D fields and smooth them (before the boost) ────────
    def _load(name: str) -> np.ndarray:
        return filt.smooth(np.asarray(raw[name][t_idx], dtype=np.float64), periodic=periodic)

    n = _load("n")
    e2 = _load("e2")
    e3 = _load("e3")
    b2 = _load("b2")
    b3 = _load("b3")
    vfl1 = _load("vfl1")
    vfl2 = _load("vfl2")
    vfl3 = _load("vfl3")
    P11 = _load("P11")
    P12 = _load("P12")
    P00 = _load("P00")
    ufl1 = _load("ufl1")
    ufl2 = _load("ufl2")

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
        - gamma * n * (vfl1 - beta) * (gamma * ((1.0 + beta**2) * ufl1 - beta * (P11 / n) - beta * P00)) / denom
    )

    # $P'_{12} = \gamma^2(P_{12} - 2\beta n u_2)
    #            - \gamma n(v_x-\beta)(u_2 - \beta P_{12}/n)\,/\,D$
    p12_t = gamma**2 * (P12 - 2.0 * beta * n * ufl2) - gamma * n * (vfl1 - beta) * (ufl2 - beta * (P12 / n)) / denom

    # $T'_{11} = P'_{11}/n'$,  $T'_{12} = P'_{12}/n'$
    T11_t = p11_t / n_t
    T12_t = p12_t / n_t

    # ── Transverse average $\langle\cdot\rangle_y$ ───────────────────
    n_avg = n_t.mean(axis=avg_axis)
    b2_avg = b2_t.mean(axis=avg_axis)
    b3_avg = b3_t.mean(axis=avg_axis)
    vfl1_avg = vfl1_t.mean(axis=avg_axis)
    vfl2_avg = vfl2.mean(axis=avg_axis)  # untransformed
    vfl3_avg = vfl3.mean(axis=avg_axis)  # untransformed
    T11_avg = T11_t.mean(axis=avg_axis)
    T12_avg = T12_t.mean(axis=avg_axis)

    # ── Longitudinal derivatives in 2-D (before the transverse average) ──
    # Covariant approximation: $\partial/\partial x' \approx \gamma\,\partial/\partial x$
    # (order n picks up a $\gamma^n$ factor; non-periodic boundaries)
    def _d2d(f: np.ndarray, order: int = 1) -> np.ndarray:
        return gamma**order * filt.derivative(f, dx, axis=x_axis, order=order, periodic=False)

    def _d_avg(f: np.ndarray, order: int = 1) -> np.ndarray:
        return _d2d(f, order).mean(axis=avg_axis)

    dvfl1 = _d_avg(vfl1_t)
    dvfl2 = _d_avg(vfl2)  # untransformed
    dvfl3 = _d_avg(vfl3)  # untransformed
    dn = _d_avg(n_t)
    dT11 = _d_avg(T11_t)
    db2 = _d_avg(b2_t)
    db3 = _d_avg(b3_t)
    # Mean-field composite ∂x'(⟨n'⟩⟨T'11⟩): by definition the derivative of the
    # product of the *averaged* profiles — reused in Step 3 below.
    dnT11 = gamma * filt.derivative(n_avg * T11_avg, dx, axis=0, order=1, periodic=False)

    # ── Input features (22, X) ────────────────────────────────────────
    input_features = np.stack(
        [
            n_avg,  # 0
            b2_avg,  # 1
            b3_avg,  # 2
            vfl1_avg,  # 3
            vfl2_avg,  # 4
            vfl3_avg,  # 5
            T11_avg,  # 6
            T12_avg,  # 7
            dvfl1,  # 8   ⟨∂v'x/∂x'⟩
            dvfl2,  # 9
            dvfl3,  # 10
            dn,  # 11
            dT11,  # 12
            db2,  # 13
            db3,  # 14
            _d_avg(vfl1_t, 2),  # 15  ⟨∂²v'x/∂x'²⟩
            _d_avg(vfl2, 2),  # 16
            _d_avg(vfl3, 2),  # 17
            _d_avg(b2_t, 2),  # 18
            _d_avg(b3_t, 2),  # 19
            _d_avg(n_t, 2),  # 20
            dnT11,  # 21  ∂(⟨n'⟩⟨T'11⟩)/∂x'  — also used in η Step 3
        ]
    )  # (22, X)

    # ── Output: boosted η (1, X) ──────────────────────────────────────
    # Transverse 2-D derivative: $\partial/\partial y' = \partial/\partial y$
    # (no $\gamma$; periodic wrap-around for standard shock-sim BCs)
    def _d2d_y(f: np.ndarray) -> np.ndarray:
        return filt.derivative(f, dx2, axis=avg_axis, order=1, periodic=True)

    # Step 1: 2-D e_vlasov on full boosted field (no E_x)
    # $e'_{vlasov} = -v'_x\,\partial_{x'} v'_x
    #                - \frac{1}{n'}\bigl(\partial_{x'}(n' T'_{11})
    #                                   + \partial_y(n' T'_{12})\bigr)
    #                - v_y B'_z + v_z B'_y$
    e_vlasov_2d = -vfl1_t * _d2d(vfl1_t) - (1.0 / n_t) * (_d2d(n_t * T11_t) + _d2d_y(n_t * T12_t)) - vfl2 * b3_t + vfl3 * b2_t

    # Step 2: transverse average
    e_vlasov_avg = e_vlasov_2d.mean(axis=avg_axis)

    # Step 3: mean-field corrections — reuse dvfl1 and dnT11 from input features
    # $\eta' = \langle e'_{vlasov}\rangle
    #         + \langle v'_x\rangle\,\partial_{x'}\langle v'_x\rangle
    #         + \frac{1}{\langle n'\rangle}\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)
    #         + \langle v_y\rangle\langle B'_z\rangle - \langle v_z\rangle\langle B'_y\rangle$
    eta = (
        e_vlasov_avg
        + vfl1_avg * dvfl1  # dvfl1 = input feature 8
        + (1.0 / n_avg) * dnT11  # dnT11 = input feature 21
        + vfl2_avg * b3_avg
        - vfl3_avg * b2_avg
    )

    return np.concatenate([input_features, np.stack([eta])], axis=0)  # (23, X)
