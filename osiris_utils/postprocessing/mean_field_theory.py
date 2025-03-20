import numpy as np
from ..data.diagnostic import Diagnostic

AXIS_WITH_TIME = {"t": 0, "x1": 1, "x2": 2, "x3": 3}
AXIS = {"x1": 0, "x2": 1, "x3": 2}

class MeanFieldTheory_Diagnostic():
    def __init__(self, Diagnostic, axis, data = None, index = None):
        self._diag = Diagnostic
        self._axis = axis
        self._t = index

        if data is not None:
            self._diag.data = data

        self._compute_mean_field()
        
    def _compute_mean_field(self):
        if not hasattr(self._diag, 'data') and self._t is None:
            print("No data to compute the mean field. Loading all data.")
            self._diag.load_all()
        elif self._t is not None:
            _mft_data = self._diag[self._t]
            self._mean_field = np.mean(_mft_data, axis=AXIS[self._axis])
            self._fluctuations = _mft_data - np.expand_dims(self._mean_field, axis=AXIS[self._axis])
        else:
            self._mean_field = np.mean(self._diag.data, axis=AXIS_WITH_TIME[self._axis])
            self._fluctuations = self._diag.data - np.expand_dims(self._mean_field, AXIS_WITH_TIME[self._axis])

    @property
    def sim(self):
        return self._diag
    
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

