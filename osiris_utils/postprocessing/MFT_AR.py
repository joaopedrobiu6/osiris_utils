from ..data.simulation_data import OsirisSimulation
import numpy as np

class AR():
    def __init__(self, folder1, folder2 = None):
        self._folder1 = folder1
        self._folder2 = folder2

    def _load_data(self, t):
        self._E1 = OsirisSimulation(self._folder1)
        self._E1.get_field("e1")
        self._B2 = OsirisSimulation(self._folder1)
        self._B2.get_field("b2")
        self._B3 = OsirisSimulation(self._folder1)
        self._B3.get_field("b3")
        self._V1 = OsirisSimulation(self._folder1)
        self._V1.get_field("vfl1")
        self._V2 = OsirisSimulation(self._folder1)
        self._V2.get_field("vfl2")
        self._V3 = OsirisSimulation(self._folder1)
        self._V3.get_field("vfl3")
        self._ne = OsirisSimulation(self._folder1)
        self._ne.get_density("charge")
        self._T11 = OsirisSimulation(self._folder1)
        self._T11.get_field("T11")
        self._T12 = OsirisSimulation(self._folder1)
        self._T12.get_field("T12")
        if self._folder2 is not None:
            self._V1_b = OsirisSimulation(self._folder2)
            self._V1_b.get_field("vfl1")
            self._V1_a = OsirisSimulation(self._folder2)
            self._V1_a.get_field("vfl1")


"""
self.E1 = OsirisGridFile(quantity_folder + f'FLD/e1/e1-{iter:06}.h5')
self.B2 = OsirisGridFile(quantity_folder + f'FLD/b2/b2-{iter:06}.h5')
self.B3 = OsirisGridFile(quantity_folder + f'FLD/b3/b3-{iter:06}.h5')

self.V1 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl1/vfl1-electrons-{iter:06}.h5')
self.V2 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl2/vfl2-electrons-{iter:06}.h5')
self.V3 = OsirisGridFile(quantity_folder + f'UDIST/electrons/vfl3/vfl3-electrons-{iter:06}.h5')

self.V1_b = OsirisGridFile(velocity_folder + f"vfl1-electrons-{vt_minus:06}.h5")
self.V1_a = OsirisGridFile(velocity_folder + f"vfl1-electrons-{vt_plus:06}.h5")

self.ne = OsirisGridFile(quantity_folder + f'DENSITY/electrons/charge/charge-electrons-{iter:06}.h5')
self.ne.data = -self.ne.data

self.T11 = OsirisGridFile(quantity_folder + f'UDIST/electrons/T11/T11-electrons-{iter:06}.h5')
self.T12 = OsirisGridFile(quantity_folder + f'UDIST/electrons/T12/T12-electrons-{iter:06}.h5')
self.P11 = OsirisGridFile(quantity_folder + f'UDIST/electrons/P11/P11-electrons-{iter:06}.h5')
self.P11.data = self.P11.data - self.ne.data*self.V1.data*self.V1.data

# Compute components of the mometum equation
self.dV1dt = (self.V1_a.data - self.V1_b.data)/(2*self.V1_a.dt)
self.V1_dV1dx = self.V1.data * np.gradient(self.V1.data, self.V1.dx[0], axis=0)

self.dT11nedx = np.gradient(self.T11.data*self.ne.data, self.T11.dx[0], axis=0)
self.dT12nedy = np.gradient(self.T12.data*self.ne.data, self.T12.dx[1], axis=1)

self.V2B3 = self.V2.data * self.B3.data
self.V3B2 = self.V3.data * self.B2.data  

self.E_vlasov = - self.dV1dt - self.V1_dV1dx - (1/self.ne.data)*(self.dT11nedx + self.dT12nedy) - (self.V2B3 - self.V3B2)

"""