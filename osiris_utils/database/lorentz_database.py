from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..ar import AnomalousResistivityConfig
from ..profiling import _start_timer, _stop_timer
from .burst import BurstAxis, BurstConfig
from .database import _VALID_ETA_FORMULAS, DatabaseCreator, _resolve_filter, _stack_rows

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..filters import SpatialFilter

logger = logging.getLogger(__name__)

__all__ = ["LorentzDatabaseBuildConfig", "LorentzDatabaseCreator"]

# =====================================================================
# Database parameters — edit here to change what the tensors contain
# =====================================================================
# Each label below is a key of the quantity dict produced by
# ``_boost_frame_quantities``, exactly as in :mod:`osiris_utils.database.database`.
# Tensor rows follow the declaration order of these maps, and the label list is
# *derived* from them, so a label and the quantity it names cannot drift apart.

#: ``label -> boosted field`` for the plain transverse averages.
_BASE_LABELS: dict[str, str] = {
    "n_avg": "n",
    "b2_avg": "b2",
    "b3_avg": "b3",
    "vfl1_avg": "vfl1",
    "ufl1_avg": "ufl1",
    "vfl2_avg": "vfl2",
    "vfl3_avg": "vfl3",
    "T11_avg": "T11",
    "T12_avg": "T12",
}

#: ``label -> boosted field`` for the first boosted x-derivatives.
_D1_LABELS: dict[str, str] = {
    "dvfl1_dx1_avg": "vfl1",
    "dufl1_dx1_avg": "ufl1",
    "dvfl2_dx1_avg": "vfl2",
    "dvfl3_dx1_avg": "vfl3",
    "dn_dx1_avg": "n",
    "dT11_dx1_avg": "T11",
    "db2_dx1_avg": "b2",
    "db3_dx1_avg": "b3",
}

#: ``label -> boosted field`` for the second boosted x-derivatives.
_D2_LABELS: dict[str, str] = {
    "d2_vfl1_dx1_avg": "vfl1",
    "d2_ufl1_dx1_avg": "ufl1",
    "d2_vfl2_dx1_avg": "vfl2",
    "d2_vfl3_dx1_avg": "vfl3",
    "d2_b2_dx1_avg": "b2",
    "d2_b3_dx1_avg": "b3",
    "d2_n_dx1_avg": "n",
}

#: Mean-field composite ∂x'(⟨n'⟩⟨T'11⟩), and the time derivative.  The latter is
#: never optional here: ∂/∂x' *contains* ∂/∂t (note Eq. 36), so a Lorentz build
#: without it is not a cheaper build, it is a wrong one.
_COMPOSITE_LABELS: list[str] = ["dnT11_dx1_avg", "dvfl1_dt_avg", "dufl1_dt_avg"]

#: Rows of the "input" tensor.
LORENTZ_FEATURE_LABELS: list[str] = [*_BASE_LABELS, *_D1_LABELS, *_D2_LABELS, *_COMPOSITE_LABELS]

#: Rows of the "output" tensor (boosted anomalous resistivity).
LORENTZ_OUTPUT_LABELS: list[str] = ["eta_avg"]

#: Rows of the "e_vlasov" tensor (boosted mean-field Vlasov electric field).
LORENTZ_E_VLASOV_LABELS: list[str] = ["e_vlasov_avg"]

#: ``"both"`` is the historical name for input + output; kept as an alias so
#: existing scripts keep working, but ``"InOut"`` is the one that matches
#: :class:`~osiris_utils.database.database.DatabaseCreator`.
_DATABASE_ALIASES = {"both": "InOut"}
_VALID_DATABASE_TYPES = {"input", "output", "e_vlasov", "InOut", "all"}


@dataclass(frozen=True)
class LorentzDatabaseBuildConfig:
    r"""Configuration for :class:`LorentzDatabaseCreator`.

    Mirrors :class:`~osiris_utils.database.database.DatabaseBuildConfig` field
    for field, plus the boost parameters; the differences are noted below.

    Parameters
    ----------
    dtype :
        NumPy dtype for saved tensors.
    max_workers :
        Worker threads for parallel frame building.  Each one holds the burst
        triple of all 13 lab fields plus the boost's intermediates as
        ``(3, nx1, nx2)`` float64 — a peak of ~432 B per grid cell, so the
        process needs roughly ``max_workers * 432 B * nx1 * nx2`` and a large
        grid OOMs long before the thread count becomes the bottleneck.  Size it
        against the node's memory, not its core count
        (``tests/test_lorentz_database.py::test_frame_working_set_is_bounded``
        pins the per-cell figure).
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
    ar_config :
        ``include_*`` flags gating which terms enter :math:`e'_{vlasov}` and
        :math:`\eta'`, exactly as for ``DatabaseCreator``.  None = the defaults.
        ``include_time_derivative`` is forced True — see :data:`_COMPOSITE_LABELS`.
    filters :
        Spatial filters applied, in order, to every raw 2-D field right after
        loading — *before* the Lorentz boost and the transverse average.
        Empty (default) = no filtering, 4th-order finite-difference
        derivatives.  Filters like
        :class:`~osiris_utils.filters.SavitzkyGolayFilter` also supply their own
        analytic derivative scheme, used for every derivative in the pipeline.
        As in ``DatabaseBuildConfig``, this and ``ar_config.filters`` may not
        disagree.
    eta_formula :
        ``"lhs"`` (default here) — :math:`\langle e'_{vlasov}\rangle` minus the
        mean-field momentum equation.  ``"thesis"`` — the fluctuation
        cross-term decomposition.  Both are the same formulas
        ``DatabaseCreator`` uses, evaluated on boosted fields with
        :math:`\partial_{x'}` in place of :math:`\partial_x`.  The default
        differs from ``DatabaseCreator``'s (``"thesis"``) only because it is
        what this class has always produced.
    validate_output :
        Replace NaN/inf with 0 per frame in the output and e_vlasov tensors when True.
    burst :
        How to read the run's time axis.  Never optional: :math:`\partial/\partial x'`
        contains a :math:`\partial/\partial t` term (note Eq. 36), so every frame
        needs its two flanking frames.  ``None`` (default) means ``BurstConfig()``.
        Only ``ndump_fac`` is taken from it — ``deriv_quantities`` is forced to
        *every* field the boost reads (the transforms are nonlinear, so
        :math:`\partial_t` of a boosted quantity needs all of its inputs at both
        flanking frames) and ``require_centered`` is forced to True (the second
        derivative needs the midpoint as well).  On a run without burst dumps this
        still works: the flanking frames are then the neighbouring ordinary dumps
        and the time derivative is correspondingly coarse.
    resume :
        If True, reuse the existing boost_velocities file and skip already-written frames.
    flush_every :
        Flush memory-mapped output(s) every N completed frames.
    rqm :
        ``m / q`` of the species in OSIRIS units (-1 electrons, +32 the shock-deck
        ions).  Sets the sign and mass scaling of the inertial and pressure terms
        of e_vlasov and eta.  None (default) reads it from the input deck.
    """

    dtype: type = np.float32
    max_workers: int | None = None
    mft_axis: int = 2
    boost_min: float = 0.0
    boost_max: float = 0.9
    seed: int | None = None
    ar_config: AnomalousResistivityConfig | None = None
    filters: Sequence[SpatialFilter] | SpatialFilter | None = ()
    eta_formula: str = "lhs"
    validate_output: bool = True
    burst: BurstConfig | None = None
    resume: bool = False
    flush_every: int = 128
    rqm: float | None = None


class LorentzDatabaseCreator(DatabaseCreator):
    r"""
    Build ``(T, F, X)`` tensors by applying a random x-direction Lorentz boost
    independently at each timestep, then transverse-averaging.

    Same workflow as :class:`~osiris_utils.database.database.DatabaseCreator` —
    construct, ``set_limits(t0, t1)``, ``create_database(database=...)`` — and
    the same guarantees: one pass over the frames, memory-mapped streaming
    output, per-frame resume, and rows selected from a named quantity dict by
    the label lists at the top of this module.

    Pipeline per timestep *t* (a burst *group*: a midpoint and its two flanking
    frames):

        raw (nx, ny) frames at t-k, t, t+k                        [(3, nx, ny)]
          → spatial filter (2-D, from ``build_config.filters``)   [same]
          → Lorentz boost with :math:`\beta_t`, all three levels  [same]
          → boosted derivatives (note Eqs. 35-38), 2-D            [(nx, ny)]
          → transverse mean :math:`\langle\cdot\rangle_y`          [(nx,)]
          → eta ("lhs" or "thesis" formula)

    :math:`\beta_t \sim \mathcal{U}(\beta_{\min}, \beta_{\max})` is drawn once,
    saved to disk, and reused on resume.

    Supported database types
    ------------------------
    ``"input"``     — boosted mean-field features (:data:`LORENTZ_FEATURE_LABELS`).
    ``"output"``    — boosted eta (formula from ``eta_formula``).
    ``"e_vlasov"``  — boosted mean-field Vlasov electric field.
    ``"InOut"``     — input + output in a single pass (alias: ``"both"``).
    ``"all"``       — all three.

    Why three time levels
    ---------------------
    The boosted longitudinal derivative is *not* a rescaled lab derivative: it
    mixes in time (note Eq. 36).  Every frame therefore needs its two flanking
    frames, and because the boost is nonlinear in the lab fields the flanking
    frames must be boosted too — :math:`\partial_t` of a boosted quantity is not
    the boost of :math:`\partial_t`.  That costs 3x the I/O per output frame and
    requires every field the boost reads to exist at all three iterations; the
    :class:`~osiris_utils.database.burst.BurstAxis` enforces it and drops groups
    that do not have a full centered stencil.

    """

    def __init__(
        self,
        simulation,
        species: str,
        save_folder: str,
        build_config: LorentzDatabaseBuildConfig | None = None,
    ) -> None:
        # Delegate: the inherited _build_tensors reads state the parent sets up
        # (save_folder as a Path, the burst frame-key list), so re-implementing
        # __init__ here silently drifts out of sync with it.
        super().__init__(simulation, species, save_folder, build_config=None)
        self.build_config = build_config or LorentzDatabaseBuildConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_database(
        self,
        database: str = "InOut",
        name_input: str = "lorentz_tensor",
        name_output: str = "lorentz_output",
        name_vlasov: str = "lorentz_e_vlasov",
        name_boosts: str = "boost_velocities",
    ) -> None:
        r"""Build Lorentz-augmented tensors and save them to disk.

        All requested tensors are computed in a **single pass** over the
        simulation frames — each raw field is read and boosted exactly once per
        timestep — and streamed to memory-mapped ``.npy`` files, so the full
        tensor never sits in RAM and an interrupted job resumes.

        Parameters
        ----------
        database :
            Which tensors to build: ``"input"``, ``"output"``, ``"e_vlasov"``,
            ``"InOut"`` (input + output, also spelled ``"both"``), or ``"all"``.
        name_input, name_output, name_vlasov :
            File stems (without ``.npy``) for each tensor.
        name_boosts :
            File stem for the ``(T,)`` array of per-timestep :math:`\beta` values.
            Saved before any frames are written so a crashed job resumes with
            the same boost sequence.
        """
        database = _DATABASE_ALIASES.get(database, database)
        if database == "vnT":
            raise NotImplementedError(
                "The vnT tensor holds x1-derivatives up to 4th order, and a boosted d^n/dx'^n needs "
                "d^n/dt^n (note Eq. 36) -- orders 3 and 4 need 4- and 5-point time stencils, i.e. at "
                "least burst_dump_range = -2, 2. Build vnT unboosted with DatabaseCreator, or widen the "
                "burst range and extend _boost_frame_quantities to those orders."
            )
        if database not in _VALID_DATABASE_TYPES:
            raise ValueError(f"Invalid database '{database}'. Choose from: {sorted(_VALID_DATABASE_TYPES)} (or 'both' for 'InOut').")

        cfg = self.build_config
        if cfg.eta_formula not in _VALID_ETA_FORMULAS:
            raise ValueError(f"Invalid eta_formula '{cfg.eta_formula}'. Choose from: {sorted(_VALID_ETA_FORMULAS)}.")

        flags = self._resolve_flags()
        self.save_folder.mkdir(parents=True, exist_ok=True)

        if self.final_iter is None:
            logger.warning("set_limits() not called; inferring from simulation['e1'].")
            self.set_limits(0, None)

        build_input = database in {"input", "InOut", "all"}
        build_output = database in {"output", "InOut", "all"}
        build_vlasov = database in {"e_vlasov", "all"}

        filt = _resolve_filter(cfg.filters, flags.filters)
        dx = float(self.simulation["e1"].dx[0])  # longitudinal grid spacing
        dx2 = float(self.simulation["e1"].dx[1])  # transverse grid spacing
        avg_axis = cfg.mft_axis - 1  # 0-indexed numpy axis

        # (name, row labels, validate) for each requested tensor, in output order.
        specs: list[tuple[str, list[str], bool]] = []
        if build_input:
            specs.append((name_input, LORENTZ_FEATURE_LABELS, False))
        if build_output:
            specs.append((name_output, LORENTZ_OUTPUT_LABELS, cfg.validate_output))
        if build_vlasov:
            specs.append((name_vlasov, LORENTZ_E_VLASOV_LABELS, cfg.validate_output))

        rqm = self._resolve_rqm()
        logger.info("Momentum equation for species '%s': rqm = m/q = %g.", self.species, rqm)

        raw = self._load_raw_diagnostics()
        # Resolve the time axis *before* sampling β: groups without a full
        # centered stencil are dropped, which lowers self.T, and the β array has
        # to be one value per surviving group.
        self._burst_axis = self._build_burst_axis(raw)
        self._frame_keys = self._resolve_frame_keys()
        betas = self._load_or_generate_betas(name_boosts)
        # Keyed by burst group, not by frame index: _build_tensors walks
        # _frame_keys in order, so group -> β is the same map as output row -> β.
        beta_of = dict(zip(self._frame_keys, betas, strict=True))

        def frame_fn(key: int) -> list[np.ndarray]:
            idx, h = self._group_frames(key, raw)
            q = _boost_frame_quantities(
                raw,
                idx,
                h,
                float(beta_of[key]),
                filt,
                dx,
                dx2,
                avg_axis,
                flags,
                eta_formula=cfg.eta_formula,
                compute_e_vlasov=build_vlasov,
                compute_eta=build_output,
                rqm=rqm,
            )
            frames: list[np.ndarray] = []
            if build_input:
                frames.append(_stack_rows(q, LORENTZ_FEATURE_LABELS))
            if build_output:
                frames.append(_stack_rows(q, LORENTZ_OUTPUT_LABELS))
            if build_vlasov:
                frames.append(_stack_rows(q, LORENTZ_E_VLASOV_LABELS))
            return frames

        _timer = _start_timer(
            f"create_lorentz_database({database}, T={self.T}, X={self.X}, "
            f"beta=[{cfg.boost_min:g}, {cfg.boost_max:g}], eta_formula={cfg.eta_formula!r}, filter={filt!r})"
        )
        try:
            for name, labels, _ in specs:
                logger.info("Tensor '%s': %d features — %s", name, len(labels), labels)
            self._build_tensors(specs, frame_fn, desc=f"Building Lorentz-boosted '{database}' tensors")
            logger.info("All requested databases saved to '%s'.", self.save_folder)
        finally:
            _stop_timer(_timer)

    @property
    def feature_labels(self) -> list[str]:
        return list(LORENTZ_FEATURE_LABELS)

    @property
    def output_labels(self) -> list[str]:
        return list(LORENTZ_OUTPUT_LABELS)

    @property
    def e_vlasov_labels(self) -> list[str]:
        return list(LORENTZ_E_VLASOV_LABELS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_flags(self) -> AnomalousResistivityConfig:
        """Term flags for this build, with the time derivative forced on.

        ``d/dx'`` *contains* ``d/dt`` (note Eq. 36), so the flanking frames are
        read whatever the flag says; leaving the term out of e_vlasov while
        using it in every spatial derivative would be an inconsistent momentum
        equation, not a cheaper one.
        """
        flags = self.build_config.ar_config or AnomalousResistivityConfig(species=self.species)
        if not flags.include_time_derivative:
            logger.info("Lorentz build: forcing include_time_derivative=True (d/dx' contains d/dt).")
            flags = replace(flags, include_time_derivative=True)
        return flags

    def _build_burst_axis(self, raw: dict[str, Any]) -> BurstAxis:
        """Midpoint-aligned time axis over *every* field the boost reads.

        The parent differentiates only ``vfl1``, so it asks for a stencil on
        ``vfl1`` alone.  Here the boost mixes all 13 raw fields nonlinearly, so
        :math:`\\partial_t` of any boosted quantity needs all of them at both
        flanking iterations; and :math:`\\partial^2/\\partial x'^2` needs
        :math:`\\partial_t^2`, which is a 3-point formula.  Hence every field is a
        derivative quantity and the stencil must be centered.
        """
        cfg = self.build_config.burst or BurstConfig()
        if set(cfg.deriv_quantities) != set(raw) or not cfg.require_centered:
            logger.info(
                "Lorentz build: overriding BurstConfig.deriv_quantities with all %d boosted fields and forcing require_centered.",
                len(raw),
            )
        cfg = BurstConfig(ndump_fac=cfg.ndump_fac, deriv_quantities=tuple(raw), require_centered=True)

        ref = self.simulation["e1"]
        axis = BurstAxis(raw, dt=float(ref.dt), ndump=int(ref.ndump or 1), config=cfg)
        logger.info("Burst dumps: %s", axis.summary())
        return axis

    def _group_frames(self, key: int, raw: dict[str, Any]) -> tuple[dict[str, tuple[int, int, int]], float]:
        """``({field: (i_lo, i_mid, i_hi)}, h)`` for burst group *key*.

        The three levels are differenced as one field triple per boosted
        quantity, so they must sit at the *same* physical times for every field —
        otherwise ``n'(t+k)`` would be built from fields sampled at different
        instants.  A common ``h`` is exactly that condition, since each stencil is
        centered on the same midpoint iteration.
        """
        assert self._burst_axis is not None
        mid = self._burst_axis.indices(key)
        stencils = {name: self._burst_axis.stencil(name, key) for name in raw}

        steps = {name: st.h for name, st in stencils.items()}
        if len(set(steps.values())) > 1:
            spread = ", ".join(f"{name}: h={h:g}" for name, h in sorted(steps.items(), key=lambda kv: kv[1]))
            raise ValueError(
                f"Burst group {key} has different time-derivative steps per field ({spread}). "
                "The boost is nonlinear, so every field must be dumped at the same flanking iterations. "
                "Dump them all with if_use_burst_dump and the same burst_dump_range, or none of them."
            )

        idx = {name: (st.i_lo, mid[name], st.i_hi) for name, st in stencils.items()}
        return idx, float(next(iter(steps.values())))

    def _load_or_generate_betas(self, name_boosts: str) -> np.ndarray:
        r"""Load existing :math:`\beta` array when resuming, otherwise sample and save a new one."""
        betas_path = self.save_folder / f"{name_boosts}.npy"

        if self.build_config.resume and betas_path.exists():
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

    def _load_raw_diagnostics(self) -> dict[str, Any]:
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
# Per-frame computation (module-level for clean stack traces)
# ----------------------------------------------------------------------


def _boost_fields(f: dict[str, np.ndarray], beta: float, gamma: float) -> dict[str, np.ndarray]:
    r"""Pointwise Lorentz transforms of the lab moments (note Eqs. 14-34).

    Every input is a ``(3, nx, ny)`` stack of the three time levels; the
    transforms are pointwise in space *and* time, so one expression boosts all
    three levels at once.

    :math:`D \equiv 1 - \beta\langle v_x\rangle`

    .. math::
        n'     &= \gamma_\beta n D                                    &&(14) \\
        \langle v'_x\rangle &= (\langle v_x\rangle - \beta)/D         &&(16) \\
        \langle v'_y\rangle &= \langle v_y\rangle,\quad
        \langle v'_z\rangle = \langle v_z\rangle                      &&(17,18) \\
        B'_y   &= \gamma_\beta(B_y + \beta E_z)                       &&(33) \\
        B'_z   &= \gamma_\beta(B_z - \beta E_y)                       &&(34)

    The pressures are :math:`P'_{ij} = n'\langle v'_iu'_j\rangle - n'\langle v'_i\rangle\langle u'_j\rangle`
    (22), with the first term from (23)/(24) and :math:`\langle u'_j\rangle` from
    (19)/(20).  Note the asymmetry between the two: :math:`u'_x = \gamma_\beta(u_x - \beta\gamma_p)`
    picks up a second factor of :math:`\gamma_\beta` and — because
    :math:`\langle v_x\gamma_p\rangle = \langle u_x\rangle` — a doubled
    :math:`\beta` term, while :math:`u'_y = u_y` picks up neither.

    *f* is consumed: its entries are popped as they are used and the dict is
    empty on return.  Pass a dict you own.
    """
    # Memory: every entry of *f* and every intermediate below is a (3, nx, ny)
    # float64 triple, and with 32 frame builders in flight it is the number of
    # them ALIVE AT ONCE that sets the job's peak RSS.  So lab fields are popped
    # from *f* as they are consumed (this empties the caller's dict) and each
    # derived quantity is folded into its final form instead of being kept as a
    # named intermediate.  See ``LorentzDatabaseBuildConfig.max_workers``.
    n, vfl1, ufl1 = f.pop("n"), f.pop("vfl1"), f.pop("ufl1")

    denom = 1.0 - beta * vfl1
    n_t = gamma * n * denom

    # $\langle v'_x\rangle$ (16).  Eq. 15's
    # $n'\langle v'_x\rangle = \gamma_\beta n(\langle v_x\rangle - \beta)$ is then
    # exactly ``n_t * vfl1_t``, so it needs no triple of its own below.
    vfl1_t = (vfl1 - beta) / denom
    del vfl1

    # The OSIRIS P moments are *unnormalised* (see the class docstring):
    #   P11 = n<v_x u_x>,  P12 = n<v_x u_y>,  P00 = n<gamma_p>,
    # while vfl and ufl are per-particle averages.  So every place Eqs. 19/20/23
    # want a bracket <.> the P has to be divided by n.
    #
    # $\langle u'_x\rangle$ (19).  $\langle v_xu_x\rangle = P_{11}/n$ and
    # $\langle\gamma_p\rangle = P_{00}/n$ enter only here and with the same
    # coefficient, so they are summed before the division rather than formed as
    # two triples.
    ufl1_t = gamma * ((1.0 + beta**2) * ufl1 - beta * (f["P11"] + f["P00"]) / n) / denom

    # $n/n' = 1/(\gamma_\beta D)$, the density ratio the transverse velocities carry.
    transverse = 1.0 / (gamma * denom)
    del denom

    # $n'\langle v'_xu'_x\rangle$ (23) and $n'\langle v'_xu'_y\rangle$ (24).
    # The last term of (23) is $\beta^2 n\langle\gamma_p\rangle = \beta^2 P_{00}$.
    nvu_11 = gamma**2 * (f.pop("P11") - 2.0 * beta * n * ufl1 + beta**2 * f.pop("P00"))
    nvu_12 = gamma * (f.pop("P12") - beta * n * f.pop("ufl2"))
    del n, ufl1

    # $\langle v'_y\rangle$, $\langle v'_z\rangle$.  The note's Eqs. 17-18 state these are
    # unchanged, but that contradicts its own Eq. 13: with
    # $v'_y = u_y/[\gamma_\beta\gamma_p(1-\beta v_x)]$ and $\det J = \gamma_\beta(1-\beta v_x)$,
    # the product $v'_y\det J = u_y/\gamma_p = v_y$ exactly, so
    #     $n'\langle v'_y\rangle = \int f v_y d^3u = n\langle v_y\rangle$
    # and the transverse velocity picks up the density ratio $n/n' = 1/(\gamma_\beta D)$.
    # Verified against direct quadrature over a distribution function to machine
    # precision (tests/test_lorentz_moments.py); leaving them unchanged is wrong by
    # 6% at beta = 0.3 and 20% at beta = 0.8.
    vfl2_t = f.pop("vfl2") * transverse
    vfl3_t = f.pop("vfl3") * transverse
    del transverse

    # $P'_{ij} = n'\langle v'_iu'_j\rangle - n'\langle v'_i\rangle\langle u'_j\rangle$ (22),
    # and $\Pi'_{12}$, the (1,2) momentum flux of the *momentum equation*:
    #     $\Pi'_{12} = n'\langle v'_2u'_1\rangle - n'\langle v'_2\rangle\langle u'_1\rangle$
    # Its raw part is Eq. 24 unchanged, because $v'_2u'_1 = v'_1u'_2$ pointwise;
    # only the mean subtraction differs from the note's $P'_{12}$ (Eq. 10), which
    # subtracts $\langle v'_1\rangle\langle u'_2\rangle$ instead.  The momentum
    # equation needs the flux of $u_1$ through the y-face, so this is the one.
    # Both are wanted only as $T'/n'$, so the $1/n'$ is distributed over the two
    # terms and $P'_{11}$, $\Pi'_{12}$ never exist as triples of their own.
    T11_t = nvu_11 / n_t - vfl1_t * ufl1_t
    T12_t = nvu_12 / n_t - vfl2_t * ufl1_t
    del nvu_11, nvu_12

    return {
        "n": n_t,
        "b2": gamma * (f.pop("b2") + beta * f.pop("e3")),
        "b3": gamma * (f.pop("b3") - beta * f.pop("e2")),
        "vfl1": vfl1_t,
        "ufl1": ufl1_t,
        "vfl2": vfl2_t,
        "vfl3": vfl3_t,
        "T11": T11_t,
        "T12": T12_t,
    }


def _boost_frame_quantities(
    raw: dict[str, Any],
    idx: dict[str, tuple[int, int, int]],
    h: float,
    beta: float,
    filt: SpatialFilter,
    dx: float,
    dx2: float,
    avg_axis: int,
    flags: AnomalousResistivityConfig,
    eta_formula: str = "lhs",
    compute_e_vlasov: bool = True,
    compute_eta: bool = True,
    rqm: float = -1.0,
) -> dict[str, np.ndarray]:
    r"""Compute all boosted mean-field quantities for one timestep.

    The boosted counterpart of
    :func:`~osiris_utils.database.database._mean_field_frame_quantities`: same
    return contract (a dict of named 1-D arrays, selected into tensor rows by
    the label lists at the top of this module), same eta formulas, with
    :math:`\partial_{x'}` in place of :math:`\partial_x`.

    Parameters
    ----------
    raw :
        ``{name: Diagnostic}`` for the 13 lab fields the boost reads.
    idx :
        ``{name: (i_lo, i_mid, i_hi)}`` — the burst triple per field.  All three
        levels are read, smoothed and boosted; outputs are evaluated at ``i_mid``.
    h :
        ``t(i_hi) - t(i_lo)``, the full width of the centered stencil.
    beta :
        Boost velocity :math:`\beta = v/c` for this frame.

    Derivatives
    -----------
    Note Eqs. 35-38, with the lab derivatives taken on the boosted fields:

    .. math::
        \partial_{t'}g   &= \gamma_\beta(\partial_t g + \beta\,\partial_x g) \\
        \partial_{x'}g   &= \gamma_\beta(\beta\,\partial_t g + \partial_x g) \\
        \partial_{x'}^2 g &= \gamma_\beta^2\bigl(\beta^2\partial_t^2 g
                             + 2\beta\,\partial_t\partial_x g + \partial_x^2 g\bigr) \\
        \partial_{y'}g   &= \partial_y g

    :math:`\partial_t` is the centered in-burst difference
    :math:`(g_{hi}-g_{lo})/h`, :math:`\partial_t^2` is
    :math:`4(g_{hi}-2g_{mid}+g_{lo})/h^2` (the half-width is :math:`h/2`), and
    :math:`\partial_t\partial_x` differences the spatial derivative across the
    same triple.  Spatial derivatives come from ``filt.derivative`` (4th-order
    finite differences for :class:`~osiris_utils.filters.NoFilter`).

    e_vlasov and eta
    ----------------
    :math:`e'_{vlasov}` is the species momentum equation solved for
    :math:`E'_x`, on the boosted 2-D midpoint field.  :math:`E'_x = E_x` (note
    Eq. 29) so it is invariant and drops out of eta either way:

    .. math::
        e'_{vlasov} = \mathrm{rqm}\Bigl(\partial_{t'}v'_x + v'_x\partial_{x'}v'_x
                      + v_y\partial_{y}v'_x
                      + \frac{\partial_{x'}(n'T'_{11}) + \partial_{y}(n'T'_{12})}{n'}\Bigr)
                      - v_yB'_z + v_zB'_y

    ``eta_formula="lhs"`` removes the mean-field momentum equation from
    :math:`\langle e'_{vlasov}\rangle`; ``"thesis"`` builds the same residual
    directly out of fluctuation cross-terms :math:`\langle f'g'\rangle`.  Both
    are the formulas of ``_mean_field_frame_quantities``, term for term.

    :math:`\partial_{t'}` and :math:`\partial_{x'}` are linear and commute with
    :math:`\langle\cdot\rangle_y`, so the time-derivative term cancels exactly
    between :math:`\langle e'_{vlasov}\rangle` and the ``"lhs"`` correction —
    eta is numerically unchanged by it, but ``dvfl1_dt_avg`` is a row of the
    input tensor in its own right.
    """
    if compute_eta and eta_formula == "lhs":
        compute_e_vlasov = True

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    x_axis = 1 - avg_axis  # longitudinal axis in 2-D arrays (avg_axis=1 -> x_axis=0)
    periodic = tuple(ax == avg_axis for ax in range(2))

    # ── Load the three time levels of every lab field and smooth them ──
    # Smoothing precedes the boost: the transforms are pointwise, so they stay
    # valid on filtered data.
    def _load3(name: str) -> np.ndarray:
        return np.stack([filt.smooth(np.asarray(raw[name][i], dtype=np.float64), periodic=periodic) for i in idx[name]])

    b = _boost_fields({name: _load3(name) for name in raw}, beta, gamma)  # each (3, nx, ny)

    # ── Derivative primitives on a (3, ...) time triple ────────────────
    def d_t(g: np.ndarray) -> np.ndarray:
        return (g[2] - g[0]) / h

    def d_tt(g: np.ndarray) -> np.ndarray:
        # Half-width of the centered stencil is h/2, hence the factor 4.
        return 4.0 * (g[2] - 2.0 * g[1] + g[0]) / (h * h)

    def d_x(g: np.ndarray, axis: int, order: int = 1) -> np.ndarray:
        return filt.derivative(g[1], dx, axis=axis, order=order, periodic=False)

    def d_tx(g: np.ndarray, axis: int) -> np.ndarray:
        lo = filt.derivative(g[0], dx, axis=axis, order=1, periodic=False)
        hi = filt.derivative(g[2], dx, axis=axis, order=1, periodic=False)
        return (hi - lo) / h

    # ── Boosted derivatives (note Eqs. 35-38) ─────────────────────────
    def dt_p(g: np.ndarray, axis: int = x_axis) -> np.ndarray:
        return gamma * (d_t(g) + beta * d_x(g, axis))

    def dx_p(g: np.ndarray, axis: int = x_axis) -> np.ndarray:
        return gamma * (beta * d_t(g) + d_x(g, axis))

    def dxx_p(g: np.ndarray, axis: int = x_axis) -> np.ndarray:
        return gamma**2 * (beta**2 * d_tt(g) + 2.0 * beta * d_tx(g, axis) + d_x(g, axis, 2))

    def dy_p(g: np.ndarray) -> np.ndarray:
        # $\partial_{y'} = \partial_y$ (37); periodic wrap for shock-sim BCs.
        return filt.derivative(g[1], dx2, axis=avg_axis, order=1, periodic=True)

    def avg(g2d: np.ndarray) -> np.ndarray:
        return g2d.mean(axis=avg_axis)

    # ── Rows of the input tensor ──────────────────────────────────────
    q: dict[str, np.ndarray] = {label: avg(b[name][1]) for label, name in _BASE_LABELS.items()}
    q |= {label: avg(dx_p(b[name])) for label, name in _D1_LABELS.items()}
    q |= {label: avg(dxx_p(b[name])) for label, name in _D2_LABELS.items()}

    # Mean-field composite $\partial_{x'}(\langle n'\rangle\langle T'_{11}\rangle)$:
    # by definition the derivative of the product of the *averaged* profiles, so
    # the average is taken first, at all three time levels, and the boosted
    # derivative applied to the resulting (3, X) triple.
    nT11_mf = b["n"].mean(axis=avg_axis + 1) * b["T11"].mean(axis=avg_axis + 1)  # (3, X)
    q["dnT11_dx1_avg"] = dx_p(nT11_mf, axis=0)
    q["dvfl1_dt_avg"] = avg(dt_p(b["vfl1"]))
    q["dufl1_dt_avg"] = avg(dt_p(b["ufl1"]))

    if not (compute_e_vlasov or compute_eta):
        return q

    # 2-D boosted midpoint fields and the derivatives eta / e_vlasov reuse.
    n_t, T11_t, T12_t = b["n"], b["T11"], b["T12"]
    vfl1_2d, vfl2_2d, vfl3_2d = b["vfl1"][1], b["vfl2"][1], b["vfl3"][1]
    b2_2d, b3_2d = b["b2"][1], b["b3"][1]
    dufl1_dx1_2d = dx_p(b["ufl1"])
    dufl1_dx2_2d = dy_p(b["ufl1"]) if flags.include_transverse_advection else None

    # ── e_vlasov in 2-D, then transverse average ──────────────────────
    # Relativistic momentum equation of the species, solved for E'_x.  Taking
    # the u_1 moment of the Vlasov equation and eliminating d_t n with
    # continuity gives, exactly,
    #
    #   rqm [ d_t <u_1> + <v_j> d_j <u_1> + (d_1 Pi_11 + d_2 Pi_12) / n ]
    #       = E_1 + (<v> x B)_1,        Pi_1j = n<v_j u_1> - n<v_j><u_1>
    #
    # The advected quantity is the *proper* velocity <u_1> = <gamma_p v_1>; only
    # the advecting velocity is <v_j>.  Using <v_1> in both places is the
    # non-relativistic limit and is wrong by O(gamma - 1): on an exact
    # free-streaming Vlasov solution with E = B = 0 it leaves a residual ~4e-3
    # where the form above leaves ~1e-6, and it is not boost-invariant even
    # though E'_x = E_x (note Eq. 29).  See tests/test_lorentz_moments.py.
    #
    # rqm = m/q of the species (-1 electrons, +32 the shock-deck ions): the
    # inertial and pressure terms flip sign and scale with the mass ratio
    # between species, the magnetic term does not.
    if compute_e_vlasov:
        e_vlasov = np.zeros_like(n_t[1])
        if flags.include_time_derivative:
            e_vlasov += rqm * dt_p(b["ufl1"])
        if flags.include_convection:
            e_vlasov += rqm * (vfl1_2d * dufl1_dx1_2d)
        if flags.include_transverse_advection:
            e_vlasov += rqm * (vfl2_2d * dufl1_dx2_2d)
        if flags.include_pressure:
            e_vlasov += rqm * ((dx_p(n_t * T11_t) + dy_p(n_t * T12_t)) / n_t[1])
        if flags.include_magnetic_force:
            e_vlasov += -vfl2_2d * b3_2d + vfl3_2d * b2_2d
        q["e_vlasov_avg"] = avg(e_vlasov)

    if not compute_eta:
        return q

    # ── eta ────────────────────────────────────────────────────────────
    if eta_formula == "lhs":
        # Remove the mean-field momentum equation from <e_vlasov> so eta keeps only
        # the fluctuation (turbulent) contributions.  Each term is removed with the
        # same rqm it entered e_vlasov with; the magnetic term again carries none.
        eta = -np.sign(rqm) * q["e_vlasov_avg"].copy()
        if flags.include_time_derivative:
            # <e_vlasov> carries rqm <dt' u'x>; remove it.  dt' is linear, so this
            # cancels exactly and eta_lhs is unchanged by the term.
            eta += np.abs(rqm) * q["dufl1_dt_avg"]
        if flags.include_convection:
            eta += np.abs(rqm) * q["vfl1_avg"] * q["dufl1_dx1_avg"]
        if flags.include_pressure:
            eta += np.abs(rqm) * (q["dnT11_dx1_avg"] / q["n_avg"])
        if flags.include_magnetic_force:
            eta -= np.sign(rqm) * (q["vfl2_avg"] * q["b3_avg"] - q["vfl3_avg"] * q["b2_avg"])

    elif eta_formula == "thesis":
        # No time-derivative term here by construction: eta is built from
        # fluctuation cross-terms <f' g'> and dt' v'x enters the equation
        # linearly, so <dt' v'x> = dt' <v'x> contributes nothing to eta.

        def delta(g2d: np.ndarray) -> np.ndarray:
            return g2d - g2d.mean(axis=avg_axis, keepdims=True)

        def split(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """(mean, fluctuation) time triples of a boosted field.

            Kept as triples, not 2-D midpoints: the pressure cross-terms below
            are differentiated with d/dx', which needs all three levels.
            """
            mean = g.mean(axis=avg_axis + 1, keepdims=True)
            return mean, g - mean

        # Every non-magnetic cross-term inherits the rqm of the species' momentum
        # equation (see e_vlasov above); with rqm = -1 these reduce to the
        # historical electron signs.
        eta = np.zeros_like(q["n_avg"])
        if flags.include_convection:
            eta -= np.abs(rqm) * avg(delta(vfl1_2d) * delta(dufl1_dx1_2d))
        if flags.include_transverse_advection:
            eta -= np.abs(rqm) * avg(delta(vfl2_2d) * delta(dufl1_dx2_2d))
        if flags.include_magnetic_force:
            eta += np.sign(rqm) * avg(delta(vfl2_2d) * delta(b3_2d))
            eta += -np.sign(rqm) * avg(delta(vfl3_2d) * delta(b2_2d))
        if flags.include_pressure:
            # Thesis decomposition: density-fluctuation correction + 6 pressure
            # cross-terms (avg x delta, delta x avg, delta x delta for xx and xy).
            n_a, n_d = split(n_t)
            T11_a, T11_d = split(T11_t)
            T12_a, T12_d = split(T12_t)
            n_mid = n_t[1]

            eta += np.abs(rqm) * avg((dx_p(n_a * T11_a) / n_mid) * (n_d[1] / n_a[1]))
            eta -= np.abs(rqm) * avg(dx_p(n_a * T11_d) / n_mid)
            eta -= np.abs(rqm) * avg(dx_p(n_d * T11_a) / n_mid)
            eta -= np.abs(rqm) * avg(dx_p(n_d * T11_d) / n_mid)
            eta -= np.abs(rqm) * avg(dy_p(n_a * T12_d) / n_mid)
            eta -= np.abs(rqm) * avg(dy_p(n_d * T12_a) / n_mid)
            eta -= np.abs(rqm) * avg(dy_p(n_d * T12_d) / n_mid)

    else:
        raise ValueError(f"Invalid eta_formula '{eta_formula}'. Choose from: {sorted(_VALID_ETA_FORMULAS)}.")

    q["eta_avg"] = eta
    return q
