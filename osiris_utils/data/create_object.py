from .simulation_data import OsirisSimulation
from ..postprocessing.fft import FastFourierTransform
from .data import *
import numpy as np

class CustomOsirisSimulation(OsirisSimulation):
    """
    Class to create an OsirisSimulation object given the data. 
    Basically a wrapper around the OsirisSimulation class with setters to load info.
    """
    def __init__(self):
        super().__init__()

    def set_data(self, data, nx, dx, dt, grid, dim, axis, name, ):
        self.data = data
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.grid = grid
        self.dim = dim
        self.axis = axis
        self.name = name

    def set_fft(self, axis):
        self._fft = FastFourierTransform(self._sim, axis)

    @property
    def fft(self):
        return self._fft
