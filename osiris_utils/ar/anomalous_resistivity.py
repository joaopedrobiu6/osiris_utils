from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..postprocessing.derivative import Derivative_Diagnostic, Derivative_Simulation
from ..postprocessing.mft import MFT_Diagnostic, MFT_Simulation

logger = logging.getLogger(__name__)

__all__ = [
    "AnomalousResistivity",
    "AnomalousResistivityConfig",
    "vlasov_electric_field",
]


@dataclass(frozen=True)
class AnomalousResistivityConfig:
    """Configuration for anomalous resistivity computation."""

    species: str = "electrons"
    include_time_derivative: bool = True
    include_convection: bool = True
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
    """

    def __init__(self, simulation, species: str = "electrons", config: AnomalousResistivityConfig | None = None):
        self._simulation = simulation
        self.species = species
        self._config = config or AnomalousResistivityConfig(species=species)

        self._validate_inputs()

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

        d_dx1 = Derivative_Simulation(self._simulation, "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
        d_dt = Derivative_Simulation(self._simulation, "t", stencil=[-2, -1, 0, 1, 2], deriv_order=1)

        sp = self._simulation[self.species]

        # Composite diagnostics
        self._ensure_diagnostic(sp, sp["n"] * sp["T11"], "nT11")
        self._ensure_diagnostic(sp, sp["n"] * sp["T12"], "nT12")

        # Pressure gradients always needed if include_pressure
        if self._config.include_pressure:
            self._ensure_diagnostic(sp, Derivative_Diagnostic(sp["nT11"], "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1), "dnT11_dx1")
            self._ensure_diagnostic(sp, Derivative_Diagnostic(sp["nT12"], "x2", stencil=[-2, -1, 0, 1, 2], deriv_order=1), "dnT12_dx2")

        # Time derivative needed only if enabled
        if self._config.include_time_derivative:
            self._ensure_diagnostic(sp, d_dt[self.species]["vfl1"], "dvfl1_dt")

        # Convection needs dvfl1_dx1
        if self._config.include_convection:
            self._ensure_diagnostic(sp, d_dx1[self.species]["vfl1"], "dvfl1_dx1")

        # Magnetic force terms need b2/b3 (already validated)
        # No additional diagnostics required.

        E_vlasov = self._compute_vlasov_field_terms()
        self._ensure_diagnostic(self._simulation, E_vlasov, "e_vlasov")

        logger.info("Vlasov electric field computed.")

    def _compute_vlasov_field_terms(self):
        sim = self._simulation
        sp = self.species
        terms = []

        if self._config.include_time_derivative:
            terms.append(-1 * sim[sp]["dvfl1_dt"])

        if self._config.include_convection:
            terms.append(-1 * sim[sp]["vfl1"] * sim[sp]["dvfl1_dx1"])

        if self._config.include_pressure:
            pressure_term = (-1 / sim[sp]["n"]) * (sim[sp]["dnT11_dx1"] + sim[sp]["dnT12_dx2"])
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

        self.sim_mft = MFT_Simulation(self._simulation, mft_axis=2)

        # Avg pressure derivative used in multiple places (computed directly)
        dnT11_dx_avg = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T11"]["avg"],
            "x1",
            stencil=[-2, -1, 0, 1, 2],
            deriv_order=1,
        )
        self.dnT11_dx_avg = dnT11_dx_avg

        lhs = self._compute_lhs(dnT11_dx_avg)
        mft_terms = self._compute_mft_terms(dnT11_dx_avg)
        eta_values = self._compute_eta_values(mft_terms)

        terms_dict = {
            **self._get_average_quantities(dnT11_dx_avg),
            "LHS": lhs,
            **eta_values,
            **{name: diag["avg"] for name, diag in mft_terms.items()},
        }

        self._terms_dict = terms_dict
        logger.info("Mean field terms computed.")
        return terms_dict

    def _compute_lhs(self, dnT11_dx_avg):
        terms = [self.sim_mft["e_vlasov"]["avg"]]

        if self._config.include_time_derivative:
            terms.append(self.sim_mft[self.species]["dvfl1_dt"]["avg"])

        if self._config.include_convection:
            terms.append(self.sim_mft[self.species]["vfl1"]["avg"] * self.sim_mft[self.species]["dvfl1_dx1"]["avg"])

        if self._config.include_pressure:
            terms.append((1 / self.sim_mft[self.species]["n"]["avg"]) * dnT11_dx_avg)

        if self._config.include_magnetic_force:
            terms.append(
                self.sim_mft[self.species]["vfl2"]["avg"] * self.sim_mft["b3"]["avg"]
                - self.sim_mft[self.species]["vfl3"]["avg"] * self.sim_mft["b2"]["avg"]
            )

        return sum(terms)

    def _compute_mft_terms(self, dnT11_dx_avg) -> dict[str, MFT_Diagnostic]:
        """
        Compute mean-field-theory RHS terms.

        These are gated by config flags so disabling a physics contribution
        also removes its MFT terms from eta.
        """
        sm = self.sim_mft
        sp = self.species
        sim = self._simulation

        terms: dict[str, Any] = {}

        # Convection-like fluctuation term: v1' * (dv1/dx1)'
        if self._config.include_convection:
            # dvfl1_dx1 delta is available via MFT diagnostics of dvfl1_dx1 (computed during vlasov if convection=True)
            terms["conv_v1_dv1dx1"] = sm[sp]["vfl1"]["delta"] * sm[sp]["dvfl1_dx1"]["delta"]

        # Magnetic-force fluctuation terms
        if self._config.include_magnetic_force:
            terms["mag_v2_b3"] = sm[sp]["vfl2"]["delta"] * sm["b3"]["delta"]
            terms["mag_v3_b2"] = sm[sp]["vfl3"]["delta"] * sm["b2"]["delta"]

        # Pressure-related fluctuation/gradient terms
        if self._config.include_pressure:
            # The original code included a "density fluctuation" correction term:
            terms["press_density_fluct_corr"] = (dnT11_dx_avg / sim[sp]["n"]) * (sm[sp]["n"]["delta"] / sm[sp]["n"]["avg"])

            # Build explicitly (more readable than the previous factory)
            dnT11_dx_ad = Derivative_Diagnostic(sm[sp]["n"]["avg"] * sm[sp]["T11"]["delta"], "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
            dnT11_dx_da = Derivative_Diagnostic(sm[sp]["n"]["delta"] * sm[sp]["T11"]["avg"], "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
            dnT11_dx_dd = Derivative_Diagnostic(
                sm[sp]["n"]["delta"] * sm[sp]["T11"]["delta"], "x1", stencil=[-2, -1, 0, 1, 2], deriv_order=1
            )
            dnT12_dx_ad = Derivative_Diagnostic(sm[sp]["n"]["avg"] * sm[sp]["T12"]["delta"], "x2", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
            dnT12_dx_da = Derivative_Diagnostic(sm[sp]["n"]["delta"] * sm[sp]["T12"]["avg"], "x2", stencil=[-2, -1, 0, 1, 2], deriv_order=1)
            dnT12_dx_dd = Derivative_Diagnostic(
                sm[sp]["n"]["delta"] * sm[sp]["T12"]["delta"], "x2", stencil=[-2, -1, 0, 1, 2], deriv_order=1
            )

            terms["press_dnT11_dx_ad_over_n"] = dnT11_dx_ad / sim[sp]["n"]
            terms["press_dnT11_dx_da_over_n"] = dnT11_dx_da / sim[sp]["n"]
            terms["press_dnT11_dx_dd_over_n"] = dnT11_dx_dd / sim[sp]["n"]

            terms["press_dnT12_dx_ad_over_n"] = dnT12_dx_ad / sim[sp]["n"]
            terms["press_dnT12_dx_da_over_n"] = dnT12_dx_da / sim[sp]["n"]
            terms["press_dnT12_dx_dd_over_n"] = dnT12_dx_dd / sim[sp]["n"]

        # Wrap all into MFT diagnostics
        return {name: MFT_Diagnostic(expr, mft_axis=2) for name, expr in terms.items()}

    def _compute_eta_values(self, mft_terms: dict[str, MFT_Diagnostic]) -> dict[str, Any]:
        """
        Compute eta and eta_dom using the same sign conventions as the original code,
        but now mapped onto descriptive term names.

        Original mapping (approx):
          term1 -> -conv fluct
          term2 -> -v2'b3'
          term3 -> +v3'b2'
          term4 -> +density fluct correction
          term5-10 -> -(pressure product derivative terms)/n
        """
        coeffs: dict[str, float] = {}

        # convection
        if self._config.include_convection:
            coeffs["conv_v1_dv1dx1"] = -1.0

        # magnetic
        if self._config.include_magnetic_force:
            coeffs["mag_v2_b3"] = -1.0
            coeffs["mag_v3_b2"] = +1.0

        # pressure
        if self._config.include_pressure:
            coeffs["press_density_fluct_corr"] = +1.0
            coeffs["press_dnT11_dx_ad_over_n"] = -1.0
            coeffs["press_dnT11_dx_da_over_n"] = -1.0
            coeffs["press_dnT11_dx_dd_over_n"] = -1.0
            coeffs["press_dnT12_dx_ad_over_n"] = -1.0
            coeffs["press_dnT12_dx_da_over_n"] = -1.0
            coeffs["press_dnT12_dx_dd_over_n"] = -1.0

        eta = 0.0
        for name, c in coeffs.items():
            if name in mft_terms:
                eta = eta + c * mft_terms[name]["avg"]

        # "eta_dom" in your original code used a subset; preserve a similar notion:
        dom_names = {
            "mag_v2_b3",
            "press_density_fluct_corr",
            "press_dnT11_dx_ad_over_n",
            "press_dnT11_dx_da_over_n",
            "press_dnT11_dx_dd_over_n",
            "press_dnT12_dx_ad_over_n",
            "press_dnT12_dx_da_over_n",
            "press_dnT12_dx_dd_over_n",
        }
        eta_dom = 0.0
        for name, c in coeffs.items():
            if name in dom_names and name in mft_terms:
                eta_dom = eta_dom + c * mft_terms[name]["avg"]

        return {"eta": eta, "eta_dom": eta_dom}

    def _get_average_quantities(self, dnT11_dx_avg) -> dict[str, Any]:
        out = {
            "e_vlasov_avg": self.sim_mft["e_vlasov"]["avg"],
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
    def available_terms(self) -> list[str]:
        return list(self._terms_dict.keys())


def vlasov_electric_field(simulation, species: str = "electrons", config: AnomalousResistivityConfig | None = None):
    """Convenience function: compute e_vlasov and return simulation['e_vlasov']."""
    _ = AnomalousResistivity(simulation, species, config=config)
    return simulation["e_vlasov"]
