from .diagnostic import Diagnostic
from .data import *
import numpy as np

class CustomDiagnostic(Diagnostic):
    """
    Class to create an Diagnostic object given the data. 
    Basically a wrapper around the Diagnostic class with setters to load info.
    """
    def __init__(self):
        super().__init__()

    def set_data(self, data, nx, dx, dt, grid, dim, axis, name):
        self.data = data
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.grid = grid
        self.dim = dim
        self.axis = axis
        self.name = name