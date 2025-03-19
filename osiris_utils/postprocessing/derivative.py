from torch import set_float32_matmul_precision
from ..data.diagnostic import Diagnostic
from ..utils import *
from ..data.simulation import Simulation

class Derivative:
    def __init__(self, simulation, type, axis=None):
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._type = type
        self._axis = axis
        self._derivatives_computed = {}

    def __getitem__(self, key):
        return Derivative_Auxiliar(self._simulation[key], self._type, self._axis)
    
class Derivative_Auxiliar:
    def __init__(self, Diagnostic, type, axis=None):
        self._diag = Diagnostic
        self._type = type
        self._axis = axis
        self._dim = getattr(Diagnostic, "_dim", 1)
        self._cached_all = None

    def compute_all(self, point):
        if self._cached_all is not None:
            return self._cached_all
        
        if not hasattr(self._diag, 'data') or self._diag.data is None:
            self._diag.load_all()

        if self._type == "t":
            result = np.gradient(self._diag.data, self._diag._dt * self._diag._ndump, axis=0, edge_order=2)

        elif self._type == "x1":
            if self._dim == 1:
                result = np.gradient(self._diag.data, self._diag._dx, axis=1, edge_order=2)
            else:
                result = np.gradient(self._diag.data, self._diag._dx[0], axis=1, edge_order=2)
                    
        elif self._type == "x2":
            result = np.gradient(self._diag.data, self._diag._dx[0], axis=2, edge_order=2)

        elif self._type == "x3":
            result = np.gradient(self._diag.data, self._diag._dx[0], axis=3, edge_order=2)

        elif self._type == "xx":
            if len(self._axis) != 2:
                raise ValueError("Axis must be a tuple with two elements.")
            result = np.gradient(np.gradient(self._diag.data, self._diag.dx[self._axis[0]-1], axis=self._axis[0], edge_order=2), self._diag.dx[self._axis[1]-1], axis=self._axis[1], edge_order=2)
            
        elif self._type == "xt":
            if not isinstance(self._axis, int):
                raise ValueError("Axis must be an integer.")
            result = np.gradient(np.gradient(self._diag.data, self._diag.dt, axis=0, edge_order=2), self._diag.dx[self._axis-1], axis=self._axis[0], edge_order=2)
            
        elif self._type == "tx":
            if not isinstance(self._axis, int):
                raise ValueError("Axis must be an integer.")
            result = np.gradient(np.gradient(self._diag.data, self._diag.dx[self._axis-1], axis=self._axis, edge_order=2), self._diag.dt, axis=0, edge_order=2)
        else:
            raise ValueError("Invalid self._type.")
        
        self._cached_all = result
        return result
    
    def compute_at_index(self, index):
        try:
            if self._type == "x1":
                if self._dim == 1:
                    return np.gradient(self._diag[index], self._diag._dx, axis=0)
                else:
                    return np.gradient(self._diag[index], self._diag._dx[0], axis=0)
            
            elif self._type == "x2":
                return np.gradient(self._diag[index], self._diag._dx[1], axis=1)
            
            elif self._type == "x3":
                return np.gradient(self._diag[index], self._diag._dx[2], axis=2)
                
            elif self._type == "t":
                if index == 0:
                    return (-3 * self._diag[index] + 4 * self._diag[index + 1] - self._diag[index + 2]) / (2 * self._diag._dt * self._diag._ndump)
                # derivate at last point not implemented yet
                # elif self[point + 1] is None:
                #     return (3 * self[point] - 4 * self[point - 1] + self[point - 2]) / (2 * self._dt)
                else:
                    return (self[index + 1] - self[index - 1]) / (2 * self._diag._dt * self._diag._ndump)
            else:
                raise ValueError("Invalid derivative type. Use 'x1', 'x2' or 't'.")
                
        except Exception as e:
            raise ValueError(f"Error computing derivative at point {index}: {str(e)}")

    def __getitem__(self, index):
        if not isinstance(index, int) and index != "all":
            raise ValueError("Index must be an integer or 'all'.")
        elif index == "all":
            return self.compute_all()
        else:
            return self.compute_at_index(index)