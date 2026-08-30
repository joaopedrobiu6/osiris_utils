from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from ..postprocessing.derivative import Derivative_Diagnostic, Derivative_Simulation
from ..postprocessing.mft import MFT_Diagnostic, MFT_Simulation
from ..utils import resolve_rqm

logger = logging.getLogger(__name__)

#: ``Simulation -> {e_vlasov key: the terms that key was built from}``.
#: ``_ensure_diagnostic`` is idempotent by name, so two AnomalousResistivity
#: objects on one Simulation would otherwise share whichever e_vlasov was built
#: first: the second one's LHS would then keep terms it never subtracted (e.g.
#: ``<dt v1>`` when only the first config enabled the time derivative), while its
#: eta -- which never touches e_vlasov -- stayed right.
_E_VLASOV_REGISTRY: WeakKeyDictionary = WeakKeyDictionary()

__all__ = [
    "AnomalousResistivity",
    "AnomalousResistivityConfig",
    "vlasov_electric_field",
]

_X1_STENCIL = [-2, -1, 0, 1, 2]
_X2_STENCIL = [-1, 0, 1]
_T_STENCIL = [-1, 0, 1]


@dataclass(frozen=True)
class AnomalousResistivityConfig:
    """Configuration for anomalous resistivity computation."""

    species: str = "electrons"
    mft_axis: int = 2
    include_time_derivative: bool = False
    include_convection: bool = True
    include_transverse_advection: bool = False
    include_pressure: bool = True
    include_magnetic_force: bool = True


class AnomalousResistivityABC(ABC):
    """Abstract base class for computing anomalous resistivity."""

    def __init__(self, simulation, species: str = "electrons"):
        self.simulation = simulation
        self.species = species

    @abstractmethod
    def compute_vlasov_electric_field(self):
        pass

    @abstractmethod
    def compute_mean_field_terms(self):
        pass

    @abstractmethod
    def __getitem__(self, item: str):
        pass


class AnomalousResistivity(AnomalousResistivityABC):
    """
    Compute Vlasov electric field and mean-field terms required for anomalous resistivity.

    Notes:
    - Adds diagnostics into the provided simulation (idempotently where possible).
    - Respects config flags for which terms are included.
    - `eta` uses the thesis pressure decomposition (7 cross-terms).
    - `eta_new` uses the simplified formulation (4 terms) from the research code.
    - `include_transverse_advection` adds the `rqm * vfl2 * dvfl1_dx2` term to
      e_vlasov and the corresponding fluctuation term to eta / eta_new.

    Terms
    -----
    ``terms_dict`` (and ``ar[...]``) holds three groups:

    - plain transverse averages (``*_avg``), unweighted;
    - ``lhs_*``: the individual contributions to ``LHS``, so ``LHS`` is exactly
      their sum;
    - the fluctuation cross-terms (``conv_*``, ``mag_*``, ``press_*``), so
      ``eta`` / ``eta_new`` are exactly the sums of their respective terms.

    Every term of the last two groups is **already multiplied by its
    coefficient** — plotting them stacked reproduces ``LHS`` / ``eta`` with no
    further sign or ``rqm`` bookkeeping.  The coefficients are in
    :attr:`term_coefficients`, and the unweighted MFT diagnostics in
    :attr:`mft_terms`.

    Species
    -------
    Every inertial and pressure term is multiplied by ``rqm = m/q`` (-1 for
    electrons, +32 for the shock-deck ions), so it flips sign and scales with the
    mass ratio between species; the magnetic term carries no ``rqm`` because E and
    v x B are divided by q together.  ``eta`` is normalised as
    ``-sign(rqm) * (<e_vlasov> - mean-field)``, matching
    :mod:`osiris_utils.database.database`, so the non-magnetic fluctuation terms
    enter with the same negative coefficient for both species.  ``rqm`` is read
    from the input deck; pass ``rqm=`` only to override a deck that cannot be
    parsed.  With ``rqm = -1`` every expression reduces to the historical
    electron form.
    """

    def __init__(
        self,
        simulation,
        species: str = "electrons",
        config: AnomalousResistivityConfig | None = None,
        rqm: float | None = None,
    ):
        self._simulation = simulation
        self.species = species
        self._config = config or AnomalousResistivityConfig(species=species)

        self._validate_inputs()
        self._e_vlasov_key = f"e_vlasov_{species}"
        self._rqm = resolve_rqm(simulation, species, rqm)
        logger.info("Momentum equation for species '%s': rqm = m/q = %g.", species, self._rqm)

        try:
            self.compute_vlasov_electric_field()
            self._terms_dict = self.compute_mean_field_terms()
            logger.info("Initialized AnomalousResistivity for species=%s", species)
        except Exception as e:
            logger.exception("Failed to initialize AnomalousResistivity.")
            raise RuntimeError(f"Failed to initialize AnomalousResistivity: {e}") from e

    # ----------------------------
    # Validation + diagnostic helpers
    # ----------------------------

    def _validate_inputs(self):
        sim = self._simulation
        sp = self.species

        if not hasattr(sim, "species") or sp not in sim.species:
            raise ValueError(f"Species '{sp}' not found in simulation.")

        required_quantities = ["n", "T11", "T12", "vfl1", "vfl2", "vfl3"]
        for q in required_quantities:
            try:
                _ = sim[sp][q]
            except Exception as e:
                raise ValueError(f"Required quantity '{q}' not available for species '{sp}'.") from e

        required_fields = ["b2", "b3"]
        for f in required_fields:
            try:
                _ = sim[f]
            except Exception as e:
                raise ValueError(f"Required field '{f}' not available in simulation.") from e

    @staticmethod
    def _ensure_diagnostic(container, diagnostic, name: str):
        """Add a diagnostic only if it doesn't already exist."""
        try:
            _ = container[name]
            return
        except Exception:
            container.add_diagnostic(diagnostic, name)

    # ----------------------------
    # Vlasov electric field
    # ----------------------------

    def compute_vlasov_electric_field(self):
        logger.info("Computing Vlasov electric field...")

        d_dx1 = Derivative_Simulation(self._simulation, "x1", stencil=_X1_STENCIL, deriv_order=1)
        d_dt = Derivative_Simulation(self._simulation, "t", stencil=_T_STENCIL, deriv_order=1)

        sp = self._simulation[self.species]

        # Composite diagnostics
        self._ensure_diagnostic(sp, sp["n"] * sp["T11"], "nT11")
        self._ensure_diagnostic(sp, sp["n"] * sp["T12"], "nT12")

        if self._config.include_pressure:
            self._ensure_diagnostic(sp, Derivative_Diagnostic(sp["nT11"], "x1", stencil=_X1_STENCIL, deriv_order=1), "dnT11_dx1")
            self._ensure_diagnostic(
                sp, Derivative_Diagnostic(sp["nT12"], "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True), "dnT12_dx2"
            )

        if self._config.include_time_derivative:
            self._ensure_diagnostic(sp, d_dt[self.species]["vfl1"], "dvfl1_dt")

        if self._config.include_convection:
            self._ensure_diagnostic(sp, d_dx1[self.species]["vfl1"], "dvfl1_dx1")

        if self._config.include_transverse_advection:
            d_dx2 = Derivative_Simulation(self._simulation, "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True)
            self._ensure_diagnostic(sp, d_dx2[self.species]["vfl1"], "dvfl1_dx2")

        E_vlasov = self._compute_vlasov_field_terms()
        # Namespaced per species *and* per momentum equation: e_vlasov is stored on
        # the simulation-level container, and _ensure_diagnostic is idempotent by
        # name.  Under a single shared name a second species — or a second config
        # of the same species — would silently reuse the first one's field.
        self._e_vlasov_key = self._register_e_vlasov(E_vlasov)

        logger.info("Vlasov electric field computed.")

    def _e_vlasov_signature(self) -> tuple:
        """Everything e_vlasov is built from: the enabled terms and rqm."""
        c = self._config
        return (
            c.include_time_derivative,
            c.include_convection,
            c.include_transverse_advection,
            c.include_pressure,
            c.include_magnetic_force,
            float(self._rqm),
        )

    def _e_vlasov_suffix(self) -> str:
        """Readable tag of the enabled terms, used to disambiguate keys."""
        c = self._config
        enabled = [
            name
            for name, on in (
                ("dt", c.include_time_derivative),
                ("conv", c.include_convection),
                ("tadv", c.include_transverse_advection),
                ("press", c.include_pressure),
                ("mag", c.include_magnetic_force),
            )
            if on
        ]
        return f"{'-'.join(enabled) or 'empty'}_rqm{self._rqm:g}"

    def _register_e_vlasov(self, E_vlasov) -> str:
        """Store e_vlasov under a key unique to the terms it contains.

        The plain ``e_vlasov_<species>`` name is kept for the first configuration
        seen on a Simulation, so single-config use is unchanged.  A second
        configuration of the same species gets its own key instead of silently
        inheriting the first one's field.
        """
        base = f"e_vlasov_{self.species}"
        signature = self._e_vlasov_signature()
        registry = _E_VLASOV_REGISTRY.setdefault(self._simulation, {})

        key = base
        if registry.get(base, signature) != signature:
            key = f"{base}__{self._e_vlasov_suffix()}"
            logger.warning(
                "Simulation already carries '%s' built from a different momentum equation; "
                "storing this one as '%s'. Prefer one Simulation per AnomalousResistivity config.",
                base,
                key,
            )
        registry[key] = signature
        self._ensure_diagnostic(self._simulation, E_vlasov, key)
        return key

    def _compute_vlasov_field_terms(self):
        sim = self._simulation
        sp = self.species
        terms = []

        # Momentum equation of the species solved for E1:
        #   rqm [ dt v1 + v1 d1 v1 + v2 d2 v1 + (d1(n T11) + d2(n T12))/n ]
        #       = E1 + (v x B)_1
        # so the inertial and pressure terms carry rqm and the magnetic one does
        # not.  rqm = -1 reproduces the historical electron expression exactly.
        rqm = self._rqm

        if self._config.include_time_derivative:
            terms.append(rqm * sim[sp]["dvfl1_dt"])

        if self._config.include_convection:
            terms.append(rqm * sim[sp]["vfl1"] * sim[sp]["dvfl1_dx1"])

        if self._config.include_transverse_advection:
            terms.append(rqm * sim[sp]["vfl2"] * sim[sp]["dvfl1_dx2"])

        if self._config.include_pressure:
            pressure_term = (rqm / sim[sp]["n"]) * (sim[sp]["dnT11_dx1"] + sim[sp]["dnT12_dx2"])
            terms.append(pressure_term)

        if self._config.include_magnetic_force:
            terms.append(-1 * sim[sp]["vfl2"] * sim["b3"])
            terms.append(sim[sp]["vfl3"] * sim["b2"])

        return sum(terms) if terms else 0.0

    # ----------------------------
    # Mean field terms
    # ----------------------------

    def compute_mean_field_terms(self):
        logger.info("Computing mean field terms...")

        self.sim_mft = MFT_Simulation(self._simulation, mft_axis=self._config.mft_axis)

        # Avg pressure derivative used in multiple places (computed directly)
        dnT11_dx_avg = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T11"]["avg"],
            "x1",
            stencil=_X1_STENCIL,
            deriv_order=1,
        )
        self.dnT11_dx_avg = dnT11_dx_avg

        lhs_terms, lhs_coeffs = self._compute_lhs_terms(dnT11_dx_avg)
        mft_terms = self._compute_mft_terms(dnT11_dx_avg)
        thesis_coeffs, new_coeffs = self._eta_coefficients()
        weighted_terms, mft_coeffs = self._weighted_mft_terms(mft_terms, (thesis_coeffs, new_coeffs))
        eta_values = self._compute_eta_values(weighted_terms, thesis_coeffs, new_coeffs)

        terms_dict = {
            **self._get_average_quantities(dnT11_dx_avg),
            **lhs_terms,
            "LHS": sum(lhs_terms.values()),
            **eta_values,
            **weighted_terms,
        }

        self._mft_terms = mft_terms
        self._term_coefficients = {**lhs_coeffs, **mft_coeffs}
        self._terms_dict = terms_dict
        logger.info("Mean field terms computed.")
        return terms_dict

    def _compute_lhs_terms(self, dnT11_dx_avg) -> tuple[dict[str, Any], dict[str, float]]:
        """The individual, already-weighted contributions to ``LHS``.

        ``LHS`` itself is their sum,
        ``-sign(rqm) * (<e_vlasov> - mean-field momentum equation)``: removing
        the mean-field equation leaves only the fluctuation (turbulent)
        contributions.  Each mean-field term is removed with the same weight it
        entered e_vlasov with, then the whole thing is normalised by
        ``-sign(rqm)`` so the non-magnetic terms carry ``+|rqm|`` for either
        species.  For electrons (``rqm = -1``) every coefficient below is +1 and
        the sum is the historical expression.

        Returns the weighted terms and the coefficient each one was multiplied
        by, so the unweighted average can be recovered.
        """
        rqm = self._rqm
        sign = 1.0 if rqm > 0 else -1.0
        mag = abs(rqm)

        sm = self.sim_mft
        sp = self.species

        terms: dict[str, Any] = {"lhs_e_vlasov": -sign * sm[self._e_vlasov_key]["avg"]}
        coeffs: dict[str, float] = {"lhs_e_vlasov": -sign}

        if self._config.include_time_derivative:
            terms["lhs_dvfl1_dt"] = mag * sm[sp]["dvfl1_dt"]["avg"]
            coeffs["lhs_dvfl1_dt"] = mag

        if self._config.include_convection:
            terms["lhs_conv_v1_dv1dx1"] = mag * sm[sp]["vfl1"]["avg"] * sm[sp]["dvfl1_dx1"]["avg"]
            coeffs["lhs_conv_v1_dv1dx1"] = mag

        # This is zero! d/dx2 of a mean term is zero
        # if self._config.include_transverse_advection:
        #     terms["lhs_conv_v2_dv1dx2"] = mag * sm[sp]["vfl2"]["avg"] * sm[sp]["dvfl1_dx2"]["avg"]

        if self._config.include_pressure:
            terms["lhs_press_dnT11_dx1"] = mag * (1 / sm[sp]["n"]["avg"]) * dnT11_dx_avg
            coeffs["lhs_press_dnT11_dx1"] = mag

        if self._config.include_magnetic_force:
            terms["lhs_mag_v2_b3"] = -sign * (sm[sp]["vfl2"]["avg"] * sm["b3"]["avg"])
            terms["lhs_mag_v3_b2"] = sign * (sm[sp]["vfl3"]["avg"] * sm["b2"]["avg"])
            coeffs["lhs_mag_v2_b3"] = -sign
            coeffs["lhs_mag_v3_b2"] = sign

        return terms, coeffs

    def _compute_mft_terms(self, dnT11_dx_avg) -> dict[str, MFT_Diagnostic]:
        """
        Compute mean-field-theory RHS terms.

        These are gated by config flags so disabling a physics contribution
        also removes its MFT terms from eta / eta_new.
        """
        sm = self.sim_mft
        sp = self.species
        sim = self._simulation

        terms: dict[str, Any] = {}

        if self._config.include_convection:
            terms["conv_v1_dv1dx1"] = sm[sp]["vfl1"]["delta"] * sm[sp]["dvfl1_dx1"]["delta"]

        if self._config.include_transverse_advection:
            terms["conv_v2_dv1dx2"] = sm[sp]["vfl2"]["delta"] * sm[sp]["dvfl1_dx2"]["delta"]

        if self._config.include_magnetic_force:
            terms["mag_v2_b3"] = sm[sp]["vfl2"]["delta"] * sm["b3"]["delta"]
            terms["mag_v3_b2"] = sm[sp]["vfl3"]["delta"] * sm["b2"]["delta"]

        if self._config.include_pressure:
            # Thesis decomposition: 7 cross-terms (avg×delta, delta×avg, delta×delta for xx and xy)
            terms["press_density_fluct_corr"] = (dnT11_dx_avg / sim[sp]["n"]) * (sm[sp]["n"]["delta"] / sm[sp]["n"]["avg"])

            dnT11_dx_ad = Derivative_Diagnostic(sm[sp]["n"]["avg"] * sm[sp]["T11"]["delta"], "x1", stencil=_X1_STENCIL, deriv_order=1)
            dnT11_dx_da = Derivative_Diagnostic(sm[sp]["n"]["delta"] * sm[sp]["T11"]["avg"], "x1", stencil=_X1_STENCIL, deriv_order=1)
            dnT11_dx_dd = Derivative_Diagnostic(sm[sp]["n"]["delta"] * sm[sp]["T11"]["delta"], "x1", stencil=_X1_STENCIL, deriv_order=1)
            dnT12_dx_ad = Derivative_Diagnostic(
                sm[sp]["n"]["avg"] * sm[sp]["T12"]["delta"], "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True
            )
            dnT12_dx_da = Derivative_Diagnostic(
                sm[sp]["n"]["delta"] * sm[sp]["T12"]["avg"], "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True
            )
            dnT12_dx_dd = Derivative_Diagnostic(
                sm[sp]["n"]["delta"] * sm[sp]["T12"]["delta"], "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True
            )

            terms["press_dnT11_dx_ad_over_n"] = dnT11_dx_ad / sim[sp]["n"]
            terms["press_dnT11_dx_da_over_n"] = dnT11_dx_da / sim[sp]["n"]
            terms["press_dnT11_dx_dd_over_n"] = dnT11_dx_dd / sim[sp]["n"]
            terms["press_dnT12_dx_ad_over_n"] = dnT12_dx_ad / sim[sp]["n"]
            terms["press_dnT12_dx_da_over_n"] = dnT12_dx_da / sim[sp]["n"]
            terms["press_dnT12_dx_dd_over_n"] = dnT12_dx_dd / sim[sp]["n"]

            # New decomposition: 4 terms using full-sim pressure minus delta×delta correction
            # d/dx1 (n * T11) / n * (n_delta / n_avg)  -  d/dx1 (n_delta * T11_delta) / n_avg
            dnT11_full_dx1 = Derivative_Diagnostic(sim[sp]["n"] * sim[sp]["T11"], "x1", stencil=_X1_STENCIL, deriv_order=1)
            dnT12_full_dx2 = Derivative_Diagnostic(sim[sp]["n"] * sim[sp]["T12"], "x2", stencil=_X2_STENCIL, deriv_order=1, periodic=True)
            terms["press_new_xx_mixed"] = (dnT11_full_dx1 / sim[sp]["n"]) * (sm[sp]["n"]["delta"] / sm[sp]["n"]["avg"])
            terms["press_new_xx_dd"] = dnT11_dx_dd / sm[sp]["n"]["avg"]
            terms["press_new_xy_mixed"] = (dnT12_full_dx2 / sim[sp]["n"]) * (sm[sp]["n"]["delta"] / sm[sp]["n"]["avg"])
            terms["press_new_xy_dd"] = dnT12_dx_dd / sm[sp]["n"]["avg"]

        return {name: MFT_Diagnostic(expr, mft_axis=self._config.mft_axis) for name, expr in terms.items()}

    def _eta_coefficients(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Coefficients of eta (thesis 7-term pressure decomposition) and eta_new (simplified 4-term).

        The coefficient tables below are written for electrons (``rqm = -1``) and
        then scaled for the actual species: non-magnetic terms by ``|rqm|`` and
        magnetic terms by ``-sign(rqm)``.  This is the same normalisation used by
        :mod:`osiris_utils.database.database`, i.e.
        ``eta = -|rqm| * (inertial + pressure fluctuations) + sign(rqm) * (magnetic fluctuations)``.

        Electron sign conventions:
          -conv fluct, -v2'b3', +v3'b2', +density_fluct_corr, -(all press cross-terms)/n
        """
        thesis_coeffs: dict[str, float] = {}
        new_coeffs: dict[str, float] = {}

        rqm = self._rqm
        sign = 1.0 if rqm > 0 else -1.0
        mag = abs(rqm)

        if self._config.include_convection:
            thesis_coeffs["conv_v1_dv1dx1"] = -mag
            new_coeffs["conv_v1_dv1dx1"] = -mag

        if self._config.include_transverse_advection:
            thesis_coeffs["conv_v2_dv1dx2"] = -mag
            new_coeffs["conv_v2_dv1dx2"] = -mag

        if self._config.include_magnetic_force:
            thesis_coeffs["mag_v2_b3"] = sign
            thesis_coeffs["mag_v3_b2"] = -sign
            new_coeffs["mag_v2_b3"] = sign
            new_coeffs["mag_v3_b2"] = -sign

        if self._config.include_pressure:
            # Thesis: density-fluctuation correction + 6 cross-term derivatives
            thesis_coeffs["press_density_fluct_corr"] = mag
            thesis_coeffs["press_dnT11_dx_ad_over_n"] = -mag
            thesis_coeffs["press_dnT11_dx_da_over_n"] = -mag
            thesis_coeffs["press_dnT11_dx_dd_over_n"] = -mag
            thesis_coeffs["press_dnT12_dx_ad_over_n"] = -mag
            thesis_coeffs["press_dnT12_dx_da_over_n"] = -mag
            thesis_coeffs["press_dnT12_dx_dd_over_n"] = -mag
            # New: 4-term formulation
            new_coeffs["press_new_xx_mixed"] = +mag
            new_coeffs["press_new_xx_dd"] = -mag
            new_coeffs["press_new_xy_mixed"] = +mag
            new_coeffs["press_new_xy_dd"] = -mag

        return thesis_coeffs, new_coeffs

    @staticmethod
    def _weighted_mft_terms(
        mft_terms: dict[str, MFT_Diagnostic],
        coeff_tables: tuple[dict[str, float], ...],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Scale each MFT term's average by the coefficient it enters eta with.

        The thesis and the simplified formulation share their inertial and
        magnetic terms and use disjoint pressure terms, so every term has one
        unambiguous coefficient; a term appearing in both tables with different
        signs would make the stored value meaningless, hence the hard error.

        Returns the weighted terms and the coefficients used.
        """
        coeffs: dict[str, float] = {}
        for table in coeff_tables:
            for name, c in table.items():
                if coeffs.setdefault(name, c) != c:
                    raise ValueError(f"MFT term '{name}' has conflicting eta coefficients ({coeffs[name]} vs {c}).")

        weighted: dict[str, Any] = {}
        used: dict[str, float] = {}
        for name, diag in mft_terms.items():
            if name in coeffs:
                weighted[name] = coeffs[name] * diag["avg"]
                used[name] = coeffs[name]
            else:
                # Should not happen: the same config flags gate both tables.
                logger.warning("MFT term '%s' has no eta coefficient; storing its unweighted average.", name)
                weighted[name] = diag["avg"]
                used[name] = 1.0

        return weighted, used

    @staticmethod
    def _compute_eta_values(
        weighted_terms: dict[str, Any],
        thesis_coeffs: dict[str, float],
        new_coeffs: dict[str, float],
    ) -> dict[str, Any]:
        """Sum the already-weighted terms belonging to each formulation."""

        def _sum(coeffs):
            result = 0.0
            for name in coeffs:
                if name in weighted_terms:
                    result = result + weighted_terms[name]
            return result

        return {
            "eta": _sum(thesis_coeffs),
            "eta_new": _sum(new_coeffs),
        }

    def _get_average_quantities(self, dnT11_dx_avg) -> dict[str, Any]:
        out = {
            "e_vlasov_avg": self.sim_mft[self._e_vlasov_key]["avg"],
            "vfl1_avg": self.sim_mft[self.species]["vfl1"]["avg"],
            "vfl2_avg": self.sim_mft[self.species]["vfl2"]["avg"],
            "vfl3_avg": self.sim_mft[self.species]["vfl3"]["avg"],
            "b2_avg": self.sim_mft["b2"]["avg"],
            "b3_avg": self.sim_mft["b3"]["avg"],
            "n_avg": self.sim_mft[self.species]["n"]["avg"],
            "dnT11_dx1_avg": dnT11_dx_avg,
        }

        if self._config.include_time_derivative:
            out["dvfl1_dt_avg"] = self.sim_mft[self.species]["dvfl1_dt"]["avg"]

        if self._config.include_convection:
            out["dvfl1_dx1_avg"] = self.sim_mft[self.species]["dvfl1_dx1"]["avg"]

        if self._config.include_transverse_advection:
            out["dvfl1_dx2_avg"] = self.sim_mft[self.species]["dvfl1_dx2"]["avg"]

        return out

    def __getitem__(self, item: str):
        if item not in self._terms_dict:
            raise KeyError(f"Term '{item}' not found. Available: {list(self._terms_dict.keys())}")
        return self._terms_dict[item]

    @property
    def simulation(self):
        return self._simulation

    @property
    def mft(self):
        return self.sim_mft

    @property
    def x(self):
        return self._simulation["b2"].x[0]

    @property
    def dx(self):
        return self._simulation["b2"].dx[0]

    @property
    def terms_dict(self):
        return self._terms_dict.copy()

    @property
    def mft_terms(self) -> dict[str, MFT_Diagnostic]:
        """The raw (unweighted) MFT diagnostics behind the fluctuation terms."""
        return self._mft_terms.copy()

    @property
    def term_coefficients(self) -> dict[str, float]:
        """Coefficient already applied to each ``lhs_*`` / fluctuation term."""
        return self._term_coefficients.copy()

    @property
    def available_terms(self) -> list[str]:
        return list(self._terms_dict.keys())

    @property
    def e_vlasov_key(self) -> str:
        """Name under which this species' e_vlasov is stored on the Simulation."""
        return self._e_vlasov_key


def vlasov_electric_field(
    simulation,
    species: str = "electrons",
    config: AnomalousResistivityConfig | None = None,
    rqm: float | None = None,
):
    """Compute e_vlasov for *species* and return ``simulation['e_vlasov_<species>']``.

    The diagnostic is namespaced by species so several species can coexist on one
    Simulation; their momentum equations differ by ``rqm``.
    """
    ar = AnomalousResistivity(simulation, species, config=config, rqm=rqm)
    return simulation[ar.e_vlasov_key]
