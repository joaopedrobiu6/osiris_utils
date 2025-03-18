import numpy as np
from ..data.simulation_data import OsirisSimulation

AXIS_WITH_TIME = {"t": 0, "x1": 1, "x2": 2, "x3": 3}
AXIS = {"x1": 0, "x2": 1, "x3": 2}

class MeanFieldTheory():
    def __init__(self, OsirisSimulation, axis, t = None):
        self.sim = OsirisSimulation
        self.axis = axis
        self._t = t
        self._compute_mean_field()

    def _compute_mean_field(self):
        if not hasattr(self.sim, 'data') and self._t is None:
            print("No data to compute the mean field. Loading all data.")
            self.sim.load_all()
        elif self._t is not None:
            self._mft_data = self.sim[self._t]
            self._mean_field = np.mean(self._mft_data, axis=AXIS[self.axis])
            self._fluctuations = self._mft_data - np.expand_dims(self._mean_field, axis=AXIS[self.axis])
        else:
            self._mean_field = np.mean(self.sim.data, axis=AXIS_WITH_TIME[self.axis])
            self._fluctuations = self.sim.data - np.expand_dims(self._mean_field, AXIS_WITH_TIME[self.axis])

    
    @property
    def mean_field(self):
        return self._mean_field
    
    @property
    def fluctuations(self):
        return self._fluctuations
    
    @property
    def avg(self):
        return self._mean_field
    
    @property
    def delta(self):
        return self._fluctuations

