from osiris_utils.data.create_object import CustomDiagnostic
from osiris_utils.data.diagnostic import Diagnostic
from osiris_utils.postprocessing.mean_field_theory import MeanFieldTheory as mft
from osiris_utils.utils import transverse_average
import numpy as np

class AR():
    def __init__(self, folder1, folder2 = None, index = None, species = None):
        self._folder1 = folder1
        self._folder2 = folder2

        self._load_data()

    def _load_data(self):
        self._E1 = Diagnostic(self._folder1)
        self._E1.get_field("e1")
        self._E1.load_all()
        self._B2 = Diagnostic(self._folder1)
        self._B2.get_field("b2")
        self._B2.load_all()
        self._B3 = Diagnostic(self._folder1)
        self._B3.get_field("b3")
        self._B3.load_all()
        self._V1 = Diagnostic(self._folder1)
        self._V1.get_moment("electrons", "vfl1")
        self._V1.load_all()
        self._V2 = Diagnostic(self._folder1)
        self._V2.get_moment("electrons", "vfl2")
        self._V2.load_all()
        self._V3 = Diagnostic(self._folder1)
        self._V3.get_moment("electrons", "vfl3")
        self._V3.load_all()
        self._ne = Diagnostic(self._folder1)
        self._ne.get_density("electrons", "charge")
        self._ne.load_all()
        self._ne.data = -self._ne.data
        self._T11 = Diagnostic(self._folder1)
        self._T11.get_moment("electrons", "T11")
        self._T11.load_all()
        self._T12 = Diagnostic(self._folder1)
        self._T12.get_moment("electrons", "T12")
        self._T12.load_all()
        if self._folder2 is not None:
            self._V1_b = Diagnostic(self._folder2)
            self._V1_b.get_moment("electrons", "vfl1")
            self._V1_b.load_all()
            self._V1_a = Diagnostic(self._folder2)
            self._V1_a.get_moment("electrons", "vfl1")
            self._V1_a.load_all()

        self._nT11 = CustomDiagnostic()
        self._nT11.set_data(self._ne.data * self._T11.data, self._T11.nx, self._T11.dx, self._T11.dt, self._T11.grid, self._T11.dim, self._T11.axis, name = "n_e T_{11}")

        self._nT12 = CustomDiagnostic()
        self._nT12.set_data(self._ne.data * self._T12.data, self._T12.nx, self._T12.dx, self._T12.dt, self._T12.grid, self._T12.dim, self._T12.axis, name = "n_e T_{12}")

        self._compute_derivatives()
        self._ohm_law()
        self._MeanFieldExpansion(axis = "x2")

    def _compute_derivatives(self):
        self._V1.derivative("all", "t")
        self._V1.derivative("all", "x1")
        self._nT11.derivative("all", "x1")
        self._nT12.derivative("all", "x2")


    def _ohm_law(self):
        # This is the Electric field from the Vlasov equation
        ohm_law = - self._V1.deriv_t - self._V1.deriv_x1 * self._V1.data - (1/self._ne.data) * (self._nT11.deriv_x1 + self._nT12.deriv_x2) - (self._V2.data * self._B3.data - self._V3.data * self._B2.data)
        self._Evlasov = CustomDiagnostic()
        self._Evlasov.set_data(ohm_law, self._V1.nx, self._V1.dx, self._V1.dt, self._V1.grid, self._V1.dim, self._V1.axis, name = "E_{Vlasov}")

    def _MeanFieldExpansion(self, axis):
        self._Ev_MFT = mft(self._Evlasov, axis=axis)
        self._B2_MFT = mft(self._B2, axis=axis)
        self._B3_MFT = mft(self._B3, axis=axis)
        self._E1_MFT = mft(self._E1, axis=axis)
        self._V1_MFT = mft(self._V1, axis=axis)
        self._V2_MFT = mft(self._V2, axis=axis)
        self._V3_MFT = mft(self._V3, axis=axis)
        self._ne_MFT = mft(self._ne, axis=axis)
        self._T11_MFT = mft(self._T11, axis=axis)
        self._T12_MFT = mft(self._T12, axis=axis)
        self._nT11_MFT = mft(self._nT11, axis=axis)
        self._nT12_MFT = mft(self._nT12, axis=axis)
        if self._folder2 is not None:
            self._V1_b_MFT = mft(self._V1_b, axis=axis)
            self._V1_a_MFT = mft(self._V1_a, axis=axis)

    def MomEq_for_AvgQuant(self):
        self._MomEq = self._Ev_MFT.mean_field + np.gradient(self._V1_MFT.mean_field, self._V1_MFT.sim.dx[0], axis=0) + self._V1_MFT.mean_field * np.gradient(self._V1_MFT.mean_field, self._V1_MFT.sim.dx[0], axis=0) + (1/self._ne_MFT.mean_field) * (np.gradient(self._nT11_MFT.mean_field, self._T11_MFT.sim.dx[0], axis=0) + np.gradient(self._nT12_MFT.mean_field, self._T12_MFT.sim.dx[1], axis=1)) + (self._V2_MFT.mean_field * self._B3_MFT.mean_field - self._V3_MFT.mean_field * self._B2_MFT.mean_field)

    def AnomalousResistivity(self):
        self.term1 = transverse_average(self._V1_MFT.fluctuations * np.gradient(self._V1_MFT.fluctuations, self._V1_MFT.sim.dx[0], axis=0))                          # -
        self.term2 = transverse_average(self._V2_MFT.fluctuations * self._B3_MFT.fluctuations)                                                                       # -
        self.term3 = transverse_average(self._V3_MFT.fluctuations * self._B2_MFT.fluctuations)                                                                       # -

        self.commom_terms = - self.term1 - self.term2 + self.term3

        # Thermal pressure gradient term xx 
        # Model XX1
        self.xx1_term1 = transverse_average((np.gradient(self._ne_MFT.mean_field*self._T11_MFT.mean_field, self._T11.sim.dx[0], axis=0)/self._ne.data)*(self._ne_MFT.delta/self._ne_MFT.mean_field))               # +
        self.xx1_term2 = transverse_average(np.gradient(self._ne_MFT.mean_field*self._T11_MFT.delta, self._T11_MFT.sim.dx[0], axis=0)/self._ne.data)                                           # -
        self.xx1_term3 = transverse_average(np.gradient(self._ne_MFT.delta*self._T11_MFT.mean_field, self._T11_MFT.sim.dx[0], axis=0)/self._ne.data)                                           # -
        self.xx1_term4 = transverse_average(np.gradient(self._ne_MFT.delta*self._T11_MFT.delta, self._T11_MFT.sim.dx[0], axis=0)/self._ne.data)                                         # -

        self.xx1_full = self.xx1_term1 - self.xx1_term2 - self.xx1_term3 - self.xx1_term4

        # Thermal pressure gradient term xy
        # Model XY1
        self.xy1_term1 = transverse_average(np.gradient(self._ne_MFT.mean_field*self._T12_MFT.delta, self._T12_MFT.sim.dx[1], axis=1)/self._ne.data)                                           # -
        self.xy1_term2 = transverse_average(np.gradient(self._ne_MFT.delta*self._T12_MFT.mean_field, self._T12_MFT.sim.dx[1], axis=1)/self._ne.data)                                           # -
        self.xy1_term3 = transverse_average(np.gradient(self._ne_MFT.delta*self._T12_MFT.delta, self._T12_MFT.sim.dx[1], axis=1)/self._ne.data)                                         # -
        
        self.xy1_full = - self.xy1_term1 - self.xy1_term2 - self.xy1_term3

        self._eta = self.commom_terms + self.xx1_full + self.xy1_full