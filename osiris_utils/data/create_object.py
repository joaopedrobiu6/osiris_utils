from .simulation_data import OsirisSimulation
from .data import *
import numpy as np

class CustomOsirisSimulation():
    """
    Class to create an OsirisSimulation object given the data. 
    Basically a wrapper around the OsirisSimulation class with setters to load info.
    """
    def __init__(self, data, dx, dt, data_type='grid', grid=None, **kwargs):
        self.data = data
        self.dx = dx
        self.dt = dt
        self.data_type = data_type
        self.grid = grid
        self.kwargs = kwargs
        self._load_data()