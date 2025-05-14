from ..postprocessing.derivative import Derivative_Diagnostic, Derivative_Simulation
from ..postprocessing.mft import MeanFieldTheory_Simulation, MFT_Diagnostic

class AnomalousResistivity:
    """
    Class to compute the anomalous resistivity from the shock simulation data.
    This class computes the Vlasov electric field and the mean field terms
    required for the anomalous resistivity calculation.

    Parameters
    ----------
    simulation : Simulation
        The simulation object containing the data.
    species : str, optional

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
    x : float
        The x-coordinate of the simulation.
    dx : float
        The spatial step size in the x-direction.
    e_vlasov : Diagnostic
        The Vlasov electric field diagnostic.
    dnT11_dx1 : Diagnostic

    """
 
    def __init__(self, simulation, species="electrons"):
        """
        Initialize the AnomalousResistivity class.

        Parameters
        ----------
        simulation : Simulation
            The simulation object containing the data.
        """
        self._simulation = simulation
        self.species = species
        self.compute_vlasov_electric_field()
        self._terms_dict = self.compute_mean_field_terms()

    def compute_vlasov_electric_field(self):
        d_dx1 = Derivative_Simulation(self._simulation, "x1")
        d_dt = Derivative_Simulation(self._simulation, "t")

        self._simulation.add_diagnostic(self._simulation[self.species]["n"] * self._simulation[self.species]["T11"], "nT11")
        self._simulation.add_diagnostic(self._simulation[self.species]["n"] * self._simulation[self.species]["T12"], "nT12")
        self._simulation.add_diagnostic(Derivative_Diagnostic(self._simulation["nT11"], "x1"), "dnT11_dx1")
        self._simulation.add_diagnostic(Derivative_Diagnostic(self._simulation["nT12"], "x2"), "dnT12_dx2")
        self._simulation.add_diagnostic(d_dt[self.species]["vfl1"], "dvfl1_dt")
        self._simulation.add_diagnostic(d_dx1[self.species]["vfl1"], "dvfl1_dx1")
        
        E_vlasov = (
            -1* self._simulation["dvfl1_dt"]
            - 1 *  self._simulation[self.species]["vfl1"]* self._simulation["dvfl1_dx1"]
            - (1/ self._simulation[self.species]["n"]) * ( self._simulation["dnT11_dx1"] +  self._simulation["dnT12_dx2"])
            - 1 *  self._simulation[self.species]["vfl2"] * self._simulation["b3"]
            +  self._simulation[self.species]["vfl3"] * self._simulation["b2"]
        )

        self._simulation.add_diagnostic(E_vlasov, "e_vlasov")

    def compute_mean_field_terms(self):
        self.sim_mft = MeanFieldTheory_Simulation(self._simulation, mft_axis=2)
        self.dnT11_dx_avg = Derivative_Diagnostic(self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T11"]["avg"], "x1")

        LHS = (
            self.sim_mft["e_vlasov"]["avg"]
            + self.sim_mft["dvfl1_dt"]["avg"]
            + self.sim_mft[self.species]["vfl1"]["avg"]*self.sim_mft["dvfl1_dx1"]["avg"]
            + self.sim_mft[self.species]["vfl2"]["avg"]*self.sim_mft["b3"]["avg"]
            - self.sim_mft[self.species]["vfl3"]["avg"]*self.sim_mft["b2"]["avg"]
            + (1/(self.sim_mft[self.species]["n"]["avg"]))*(self.dnT11_dx_avg)
        )

        dvdx_delta = self.sim_mft["dvfl1_dx1"]["delta"]
        dnT11_dx_dd = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["delta"] * self.sim_mft[self.species]["T11"]["delta"],
            "x1")

        dnT11_dx_ad = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T11"]["delta"],
            "x1")

        dnT11_dx_da = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["delta"] * self.sim_mft[self.species]["T11"]["avg"],
            "x1")

        dnT12_dx_ad = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["avg"] * self.sim_mft[self.species]["T12"]["delta"],
            "x2")

        dnT12_dx_da = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["delta"] * self.sim_mft[self.species]["T12"]["avg"],
            "x2")

        dnT12_dx_dd = Derivative_Diagnostic(
            self.sim_mft[self.species]["n"]["delta"] * self.sim_mft[self.species]["T12"]["delta"],
            "x2")

        term1 = self.sim_mft[self.species]["vfl1"]["delta"] * dvdx_delta
        term2 = self.sim_mft[self.species]["vfl2"]["delta"] * self.sim_mft["b3"]["delta"]
        term3 = self.sim_mft[self.species]["vfl3"]["delta"] * self.sim_mft["b2"]["delta"]
        term4 = (self.dnT11_dx_avg/(self._simulation[self.species]["n"])) * \
            (self.sim_mft[self.species]["n"]["delta"]/self.sim_mft[self.species]["n"]["avg"])

        term5 = dnT11_dx_ad/self._simulation[self.species]["n"]
        term6 = dnT11_dx_da/self._simulation[self.species]["n"]
        term7 = dnT11_dx_dd/self._simulation[self.species]["n"]

        term8 = (dnT12_dx_ad/self._simulation[self.species]["n"])
        term9 = (dnT12_dx_da/self._simulation[self.species]["n"])
        term10 = (dnT12_dx_dd/self._simulation[self.species]["n"])

        term1_mft = MFT_Diagnostic(term1, mft_axis=2)
        term2_mft = MFT_Diagnostic(term2, mft_axis=2)
        term3_mft = MFT_Diagnostic(term3, mft_axis=2)
        term4_mft = MFT_Diagnostic(term4, mft_axis=2)
        term5_mft = MFT_Diagnostic(term5, mft_axis=2)
        term6_mft = MFT_Diagnostic(term6, mft_axis=2)
        term7_mft = MFT_Diagnostic(term7, mft_axis=2)
        term8_mft = MFT_Diagnostic(term8, mft_axis=2)
        term9_mft = MFT_Diagnostic(term9, mft_axis=2)
        term10_mft = MFT_Diagnostic(term10, mft_axis=2)

        eta = (-1 * term1_mft["avg"] - 1 * term2_mft["avg"] + term3_mft["avg"]
                + term4_mft["avg"] - 1 * term5_mft["avg"]
                - 1 * term6_mft["avg"] - 1 * term7_mft["avg"]
                - 1 * term8_mft["avg"] - 1 * term9_mft["avg"]
                - 1 * term10_mft["avg"]
                )
        
        #- self.term2 + self.term4 - self.term5 - self.term6 - self.term7 - self.term8 - self.term10

        eta_dom = (- 1 * term2_mft["avg"]
                + term4_mft["avg"] - 1 * term5_mft["avg"]
                - 1 * term6_mft["avg"] - 1 * term7_mft["avg"]
                - 1 * term8_mft["avg"] 
                - 1 * term10_mft["avg"]
                )

        #   LHS = (
        #     self.sim_mft["e_vlasov"]["avg"]
        #     + self.sim_mft["dvfl1_dt"]["avg"]
        #     + self.sim_mft[self.species]["vfl1"]["avg"]*self.sim_mft["dvfl1_dx1"]["avg"]
        #     + self.sim_mft[self.species]["vfl2"]["avg"]*self.sim_mft["b3"]["avg"]
        #     - self.sim_mft[self.species]["vfl3"]["avg"]*self.sim_mft["b2"]["avg"]
        #     + (1/(self.sim_mft[self.species]["n"]["avg"]))*(self.dnT11_dx_avg)
        # )

        self._terms_dict = {
            "e_vlasov_avg": self.sim_mft["e_vlasov"]["avg"],
            "vfl1_avg": self.sim_mft[self.species]["vfl1"]["avg"],
            "vfl2_avg": self.sim_mft[self.species]["vfl2"]["avg"],
            "vfl3_avg": self.sim_mft[self.species]["vfl3"]["avg"],
            "b2_avg": self.sim_mft["b2"]["avg"],
            "b3_avg": self.sim_mft["b3"]["avg"],
            "n_avg": self.sim_mft[self.species]["n"]["avg"],
            "dvfl1_dt_avg": self.sim_mft["dvfl1_dt"]["avg"],
            "dvfl1_dx1_avg": self.sim_mft["dvfl1_dx1"]["avg"],
            "dnT11_dx1_avg": self.dnT11_dx_avg,
            "LHS": LHS,
            "eta": eta,
            "eta_dom": eta_dom,
            "term1": term1_mft["avg"],
            "term2": term2_mft["avg"],
            "term3": term3_mft["avg"],
            "term4": term4_mft["avg"],
            "term5": term5_mft["avg"],
            "term6": term6_mft["avg"],
            "term7": term7_mft["avg"],
            "term8": term8_mft["avg"],
            "term9": term9_mft["avg"],
            "term10": term10_mft["avg"]
        }

        return self._terms_dict

    def __getitem__(self, item):
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
        return self.terms_dict
    
def compute_vlasov_electric_field(simulation, species="electrons"):
        d_dx1 = Derivative_Simulation(simulation, "x1")
        d_dt = Derivative_Simulation(simulation, "t")

        simulation.add_diagnostic(simulation[species]["n"] * simulation[species]["T11"], "nT11")
        simulation.add_diagnostic(simulation[species]["n"] * simulation[species]["T12"], "nT12")
        simulation.add_diagnostic(Derivative_Diagnostic(simulation["nT11"], "x1"), "dnT11_dx1")
        simulation.add_diagnostic(Derivative_Diagnostic(simulation["nT12"], "x2"), "dnT12_dx2")
        simulation.add_diagnostic(d_dt[species]["vfl1"], "dvfl1_dt")
        simulation.add_diagnostic(d_dx1[species]["vfl1"], "dvfl1_dx1")
        
        E_vlasov = (
            -1 * simulation["dvfl1_dt"]
            - 1 * simulation[species]["vfl1"]* simulation["dvfl1_dx1"]
            - (1 / simulation[species]["n"]) * (simulation["dnT11_dx1"] + simulation["dnT12_dx2"])
            - 1 * simulation[species]["vfl2"] * simulation["b3"]
            + simulation[species]["vfl3"] * simulation["b2"]
        )

        return E_vlasov