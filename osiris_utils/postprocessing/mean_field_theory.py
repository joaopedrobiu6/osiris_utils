import numpy as np
from ..data.diagnostic import Diagnostic

AXIS_WITH_TIME = {"t": 0, "x1": 1, "x2": 2, "x3": 3}
AXIS = {"x1": 0, "x2": 1, "x3": 2}

class MeanFieldTheory():
    def __init__(self, Diagnostic, axis, data = None, index = None):
        self._sim = Diagnostic
        self._axis = axis
        self._t = index

        if data is not None:
            self.sim.data = data

        self._compute_mean_field()
        
    def _compute_mean_field(self):
        if not hasattr(self._sim, 'data') and self._t is None:
            print("No data to compute the mean field. Loading all data.")
            self._sim.load_all()
        elif self._t is not None:
            _mft_data = self._sim[self._t]
            self._mean_field = np.mean(_mft_data, axis=AXIS[self._axis])
            self._fluctuations = _mft_data - np.expand_dims(self._mean_field, axis=AXIS[self._axis])
        else:
            self._mean_field = np.mean(self._sim.data, axis=AXIS_WITH_TIME[self._axis])
            self._fluctuations = self._sim.data - np.expand_dims(self._mean_field, AXIS_WITH_TIME[self._axis])

    @property
    def sim(self):
        return self._sim
    
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

