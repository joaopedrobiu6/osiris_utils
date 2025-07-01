from ..postprocessing.derivative import Derivative_Diagnostic, Derivative_Simulation
from ..postprocessing.mft import MFT_Simulation, MFT_Diagnostic
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnomalousResistivityConfig:
    """Configuration for anomalous resistivity computation."""
    species: str = "electrons"
    include_time_derivative: bool = True
    include_convection: bool = True
    include_pressure: bool = True
    include_magnetic_force: bool = True

class TermType(Enum):
    """Types of terms in the anomalous resistivity calculation."""
    VELOCITY_DERIVATIVE = "velocity_derivative"
    VELOCITY_FIELD = "velocity_field" 
    PRESSURE_GRADIENT = "pressure_gradient"
    MIXED_PRESSURE = "mixed_pressure"

class TermFactory:
    """Factory for creating different types of terms in AR calculation."""
    
    def __init__(self, sim_mft, species: str, simulation):
        self.sim_mft = sim_mft
        self.species = species
        self.simulation = simulation
    
    def create_product_derivative_term(self, 
                               quantity1_type: str, 
                               quantity1_component: str, 
                               quantity2_type: str, 
                               quantity2_component: str, 
                               axis: str) -> Derivative_Diagnostic:
        """
        Create a derivative term from the product of two quantities.
        Usually used for computing gradients of products like n*T11 or n*T12.
        """
        q1 = self.sim_mft[self.species][quantity1_type][quantity1_component]
        q2 = self.sim_mft[self.species][quantity2_type][quantity2_component]
        return Derivative_Diagnostic(q1 * q2, axis)
    
    def create_velocity_field_term(self, velocity_type: str, velocity_component: str, field_type: str, field_component: str):
        """Create velocity * field term."""
        return (self.sim_mft[self.species][velocity_type][velocity_component] * 
                self.sim_mft[field_type][field_component])
    
    def create_velocity_derivative_term(self):
        """Create velocity * velocity_derivative term."""
        return (self.sim_mft[self.species]["vfl1"]["delta"] * 
                self.sim_mft[self.species]["dvfl1_dx1"]["delta"])

class AnomalousResistivityABC(ABC):
    """
    Abstract base class for computing anomalous resistivity.
    This class is intended to be subclassed for specific implementations.
    """
    def __init__(self, simulation, species="electrons"):
        self.simulation = simulation
        self.species = species

    def compute_vlasov_electric_field(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_mean_field_terms(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __getitem__(self, item):
        raise NotImplementedError("Subclasses should implement this method.")

class AnomalousResistivity(AnomalousResistivityABC):
    """
    Class to compute the anomalous resistivity from shock simulation data.
    
    This class computes the Vlasov electric field and mean field terms
    required for anomalous resistivity calculation using a clean, 
    configuration-driven approach.

    Parameters
    ----------
    simulation : Simulation
        The simulation object containing the data.
    species : str, optional
        The species for which to compute anomalous resistivity. Default is "electrons".
    config : VlasovFieldConfig, optional
        Configuration for Vlasov field computation.

    Attributes
    ----------
    simulation : Simulation
        The simulation object containing the data.
    species : str
        The species for which the anomalous resistivity is computed.
    terms_dict : dict
        Dictionary containing the computed terms for the anomalous resistivity.
    mft : MeanFieldTheory_Simulation
        The mean field theory simulation object.
    """
 
    def __init__(self, simulation, species: str = "electrons", 
                 config: AnomalousResistivityConfig = None):
        """Initialize the AnomalousResistivity class."""
        self._validate_species_inputs(simulation, species)
        self._validate_fields_inputs(simulation)
        
        self._simulation = simulation
        self.species = species
        self._config = config or AnomalousResistivityConfig(species=species)
        
        try:
            self.compute_vlasov_electric_field()
            self._terms_dict = self.compute_mean_field_terms()
            logger.info(f"Successfully initialized AnomalousResistivity for species: {species}")
        except Exception as e:
            logger.error(f"Failed to initialize AnomalousResistivity: {e}")
            raise RuntimeError(f"Failed to initialize AnomalousResistivity: {e}")

    def _validate_species_inputs(self, simulation, species: str):
        """Validate input parameters."""
        if not hasattr(simulation, 'species') or species not in simulation.species:
            raise ValueError(f"Species '{species}' not found in simulation")
        
        required_quantities = ['n', 'T11', 'T12', 'vfl1', 'vfl2', 'vfl3']
        for qty in required_quantities:
            try:
                _ = simulation[species][qty]
            except Exception:
                raise ValueError(f"Required quantity '{qty}' not available for species '{species}'")
    
    def _validate_fields_inputs(self, simulation):
        """Validate that the simulation has the required fields."""
        required_fields = ['b2', 'b3']
        for field in required_fields:
            try:
                _ = simulation[field]
            except Exception:
                raise ValueError(f"Required field '{field}' not available in simulation")

    def compute_vlasov_electric_field(self):
        """Compute the Vlasov electric field using configuration."""
        logger.info("Computing Vlasov electric field...")
        
        # Create derivative operators
        d_dx1 = Derivative_Simulation(self._simulation, "x1")
        d_dt = Derivative_Simulation(self._simulation, "t")

        # Add composite diagnostics
        self._add_composite_diagnostics()
        
        # Add derivative diagnostics
        self._add_derivative_diagnostics(d_dt, d_dx1)
        
        # Compute Vlasov electric field
        E_vlasov = self._compute_vlasov_field_terms()
        
        self._simulation.add_diagnostic(E_vlasov, "e_vlasov")
        logger.info("Vlasov electric field computed successfully")

    def _add_composite_diagnostics(self):
        """Add composite diagnostics like nT11, nT12."""
        diagnostics_to_add = [
            (self._simulation[self.species]["n"] * self._simulation[self.species]["T11"], "nT11"),
            (self._simulation[self.species]["n"] * self._simulation[self.species]["T12"], "nT12")
        ]
        
        for diagnostic, name in diagnostics_to_add:
            self._simulation[self.species].add_diagnostic(diagnostic, name)

    def _add_derivative_diagnostics(self, d_dt, d_dx1):
        """Add derivative diagnostics."""
        derivative_diagnostics = [
            (Derivative_Diagnostic(self._simulation[self.species]["nT11"], "x1"), "dnT11_dx1"),
            (Derivative_Diagnostic(self._simulation[self.species]["nT12"], "x2"), "dnT12_dx2"),
            (d_dt[self.species]["vfl1"], "dvfl1_dt"),
            (d_dx1[self.species]["vfl1"], "dvfl1_dx1")
        ]
        
        for diagnostic, name in derivative_diagnostics:
            self._simulation[self.species].add_diagnostic(diagnostic, name)

    def _compute_vlasov_field_terms(self):
        """Compute individual terms of the Vlasov electric field."""
        terms = []
        
        if self._config.include_time_derivative:
            terms.append(-1 * self._simulation[self.species]["dvfl1_dt"])
        
        if self._config.include_convection:
            terms.append(-1 * self._simulation[self.species]["vfl1"] * self._simulation[self.species]["dvfl1_dx1"])

        if self._config.include_pressure:
            pressure_term = (-1 / self._simulation[self.species]["n"]) * (
                self._simulation[self.species]["dnT11_dx1"] + self._simulation[self.species]["dnT12_dx2"]
            )
            terms.append(pressure_term)
        
        if self._config.include_magnetic_force:
            terms.extend([
                -1 * self._simulation[self.species]["vfl2"] * self._simulation["b3"],
                self._simulation[self.species]["vfl3"] * self._simulation["b2"]
            ])
        
        return sum(terms)

    def compute_mean_field_terms(self):
        """Main method orchestrating all MFT computations."""
        logger.info("Computing mean field terms...")
        
        # Initialize MFT
        self.sim_mft = MFT_Simulation(self._simulation, mft_axis=2)
        
        # Compute components
        self.dnT11_dx_avg = self._compute_average_derivative()
        lhs = self._compute_lhs()
        derivative_terms = self._compute_derivative_terms()
        mft_terms = self._compute_mft_terms(derivative_terms)
        eta_values = self._compute_eta_values(mft_terms)
        
        # Build final dictionary
        self._terms_dict = {
            **self._get_average_quantities(),
            "LHS": lhs,
            **eta_values,
            **{name: term["avg"] for name, term in mft_terms.items()}
        }
        
        logger.info("Mean field terms computed successfully")
        return self._terms_dict

    def _compute_average_derivative(self):
        """Compute the average derivative term."""
        return Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T11"]["avg"], 
            "x1"
        )

    def _compute_lhs(self):
        """Compute the left-hand side of the equation."""
        terms = []
        terms.append(self.sim_mft["e_vlasov"]["avg"])
        if self._config.include_time_derivative:
            terms.append(self.sim_mft[self.species]["dvfl1_dt"]["avg"])
        if self._config.include_convection:
            terms.append(self.sim_mft[self.species]["vfl1"]["avg"] * self.sim_mft[self.species]["dvfl1_dx1"]["avg"])
        if self._config.include_pressure:
            terms.append((1 / self.sim_mft[self.species]["n"]["avg"]) * self.dnT11_dx_avg)
        if self._config.include_magnetic_force:
            terms.append(
                self.sim_mft[self.species]["vfl2"]["avg"] * self.sim_mft["b3"]["avg"] -
                self.sim_mft[self.species]["vfl3"]["avg"] * self.sim_mft["b2"]["avg"]
            )
        return sum(terms)
      
    def _compute_derivative_terms(self) -> Dict[str, Derivative_Diagnostic]:
        """Compute all derivative terms using factory pattern."""
        factory = TermFactory(self.sim_mft, self.species, self._simulation)
        
        # Define derivative term configurations
        derivative_configs = [
            ("dnT11_dx_dd", "n", "delta", "T11", "delta", "x1"),
            ("dnT11_dx_ad", "n", "avg", "T11", "delta", "x1"),
            ("dnT11_dx_da", "n", "delta", "T11", "avg", "x1"),
            ("dnT12_dx_ad", "n", "avg", "T12", "delta", "x2"),
            ("dnT12_dx_da", "n", "delta", "T12", "avg", "x2"),
            ("dnT12_dx_dd", "n", "delta", "T12", "delta", "x2"),
        ]
        
        terms = {}
        for name, q1_type, q1_comp, q2_type, q2_comp, axis in derivative_configs:
            terms[name] = factory.create_product_derivative_term(q1_type, q1_comp, q2_type, q2_comp, axis)
        
        return terms

    def _compute_mft_terms(self, derivative_terms: Dict[str, Derivative_Diagnostic]) -> Dict[str, MFT_Diagnostic]:
        """Compute MFT terms using configuration."""
        factory = TermFactory(self.sim_mft, self.species, self._simulation)
        
        # Define basic term computation rules
        basic_terms = {
            "term1": factory.create_velocity_derivative_term(),
            "term2": factory.create_velocity_field_term("vfl2", "delta", "b3", "delta"),
            "term3": factory.create_velocity_field_term("vfl3", "delta", "b2", "delta"),
            "term4": (self.dnT11_dx_avg / self._simulation[self.species]["n"]) * 
                    (self.sim_mft[self.species]["n"]["delta"] / self.sim_mft[self.species]["n"]["avg"])
        }
        
        # Add derivative terms (term5-term10)
        derivative_term_names = ["dnT11_dx_ad", "dnT11_dx_da", "dnT11_dx_dd", 
                               "dnT12_dx_ad", "dnT12_dx_da", "dnT12_dx_dd"]
        
        for i, deriv_name in enumerate(derivative_term_names, start=5):
            if deriv_name in derivative_terms:
                basic_terms[f"term{i}"] = derivative_terms[deriv_name] / self._simulation[self.species]["n"]
        
        # Convert to MFT diagnostics
        mft_terms = {}
        for name, term_result in basic_terms.items():
            mft_terms[name] = MFT_Diagnostic(term_result, mft_axis=2)
        
        return mft_terms

    def _compute_eta_values(self, mft_terms: Dict[str, MFT_Diagnostic]) -> Dict[str, Any]:
        """Compute eta values with clear coefficient mapping."""
        # Define coefficients for each term in eta calculation
        eta_coefficients = {
            "term1": -1, "term2": -1, "term3": 1, "term4": 1, "term5": -1,
            "term6": -1, "term7": -1, "term8": -1, "term9": -1, "term10": -1
        }
        
        eta_dom_coefficients = {
            "term2": -1, "term4": 1, "term5": -1, "term6": -1,
            "term7": -1, "term8": -1, "term10": -1
        }
        
        # Compute eta using coefficients
        eta = sum(coeff * mft_terms[term]["avg"] 
                 for term, coeff in eta_coefficients.items() 
                 if term in mft_terms)
        
        eta_dom = sum(coeff * mft_terms[term]["avg"] 
                     for term, coeff in eta_dom_coefficients.items() 
                     if term in mft_terms)
        
        return {"eta": eta, "eta_dom": eta_dom}

    def _get_average_quantities(self) -> Dict[str, Any]:
        """Get all average quantities in one place."""
        return {
            "e_vlasov_avg": self.sim_mft["e_vlasov"]["avg"],
            "vfl1_avg": self.sim_mft[self.species]["vfl1"]["avg"],
            "vfl2_avg": self.sim_mft[self.species]["vfl2"]["avg"],
            "vfl3_avg": self.sim_mft[self.species]["vfl3"]["avg"],
            "b2_avg": self.sim_mft["b2"]["avg"],
            "b3_avg": self.sim_mft["b3"]["avg"],
            "n_avg": self.sim_mft[self.species]["n"]["avg"],
            "dvfl1_dt_avg": self.sim_mft[self.species]["dvfl1_dt"]["avg"],
            "dvfl1_dx1_avg": self.sim_mft[self.species]["dvfl1_dx1"]["avg"],
            "dnT11_dx1_avg": self.dnT11_dx_avg,
        }

    def __getitem__(self, item: str):
        """Get a computed term by name."""
        if item not in self._terms_dict:
            available_keys = list(self._terms_dict.keys())
            raise KeyError(f"Term '{item}' not found. Available terms: {available_keys}")
        return self._terms_dict[item]

    @property
    def simulation(self):
        """Get the simulation object."""
        return self._simulation
    
    @property
    def mft(self):
        """Get the mean field theory simulation object."""
        return self.sim_mft
    
    @property
    def x(self):
        """Get the x-coordinate array."""
        return self._simulation["b2"].x[0]
    
    @property
    def dx(self):   
        """Get the spatial step size in x-direction."""
        return self._simulation["b2"].dx[0]
    
    @property
    def terms_dict(self):
        """Get the dictionary of all computed terms."""
        return self._terms_dict.copy()  # Return copy to prevent external modification
    
    @property
    def available_terms(self) -> List[str]:
        """Get list of all available computed terms."""
        return list(self._terms_dict.keys())

def vlasov_electric_field(simulation, species: str = "electrons", 
                                config: AnomalousResistivityConfig = None):
    """
    Standalone function to compute Vlasov electric field.
    
    Parameters
    ----------
    simulation : Simulation
        The simulation object containing the data.
    species : str, optional
        The species for which to compute the field. Default is "electrons".
    config : VlasovFieldConfig, optional
        Configuration for the computation.
        
    Returns
    -------
    Diagnostic
        The computed Vlasov electric field.
    """
    if config is None:
        config = AnomalousResistivityConfig(species=species)
    
    # Create temporary AR object to compute field
    ar = AnomalousResistivity(simulation, species, config)  # noqa: F841
    return simulation["e_vlasov"]