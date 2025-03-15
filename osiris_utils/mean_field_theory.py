import numpy as np
from .data import OsirisGridFile

        
class MeanFieldTheory(OsirisGridFile):
    '''
    Class to handle the mean field theory on data. Inherits from OsirisGridFile.
    
    Parameters
    ----------
    source : str or OsirisGridFile
        The filename or an OsirisGridFile object.
    axis : int
        The axis to average over.
    '''
    def __init__(self, source, axis=1):
        if isinstance(source, OsirisGridFile):
            self.__dict__.update(source.__dict__)
        else:
            super().__init__(source)
        self._compute_mean_field(axis=axis)

    def _compute_mean_field(self, axis=1):
        self._average = np.expand_dims(np.mean(self.data, axis=axis), axis=axis)
        self._fluctuations = self.data - self._average

    def __array__(self):
        return self.data

    @property
    def average(self):
        return self._average
    
    @property
    def delta(self):
        return self._fluctuations
    
    def __str__(self):
        return super().__str__() + f'\nAverage: {self.average.shape}\nDelta: {self.delta.shape}'
    
    def derivative(self, field, axis=0):
        '''
        Compute the derivative of the average or the fluctuations.
        
        Parameters
        ----------
        field : MeanFieldTheory.average or MeanFieldTheory.delta
            The field to compute the derivative.
        axis : int
            The axis to compute the derivative.
        '''
        return np.gradient(field, self.dx[axis], axis=0)