from ..utils import *
from ..data.simulation import Simulation

class Derivative:
    """
    Class to compute the derivative of a diagnostic.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    type : str
        The type of derivative to compute. Options are:
        - 't' for time derivative.
        - 'x1' for first spatial derivative.
        - 'x2' for second spatial derivative.
        - 'x3' for third spatial derivative.
        - 'xx' for second spatial derivative in two axis.
        - 'xt' for mixed derivative in time and one spatial axis.
        - 'tx' for mixed derivative in one spatial axis and time.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types.
    """
    def __init__(self, simulation, type, axis=None):
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._type = type
        self._axis = axis
        self._derivatives_computed = {}

    def __getitem__(self, key):
        if key not in self._derivatives_computed:
            self._derivatives_computed[key] = Derivative_Diagnostic(self._simulation[key], self._type, self._axis)
        return self._derivatives_computed[key]
    
    def delete_all(self):
        self._derivatives_computed = {}

    def delete(self, key):
        if key in self._derivatives_computed:
            del self._derivatives_computed[key]
        else:
            print(f"Derivative {key} not found in simulation")
    
class Derivative_Diagnostic:
    """
    Auxiliar class to compute the derivative of a diagnostic, for it to be similar in behavior to a Diagnostic object.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    type : str
        The type of derivative to compute. Options are: 't', 'x1', 'x2', 'x3', 'xx', 'xt' and 'tx'.
    axis : int or tuple
        The axis to compute the derivative. Only used for 'xx', 'xt' and 'tx' types
    """
    def __init__(self, diagnostic, type, axis=None):
        self._diag = diagnostic
        self._type = type
        self._axis = axis
        self._data = None    
        self.load_metadata()    

        self._all_loaded = False

    
    def load_metadata(self):
        self._name = "D[" + self._diag._name + ", " + self._type + "]"
        self._dt = self._diag._dt
        self._dx = self._diag._dx
        self._ndump = self._diag._ndump
        self._axis = self._diag._axis
        self._nx = self._diag._nx
        self._x = self._diag._x
        self._grid = self._diag._grid
        self._dim = self._diag._dim

    def load_all(self):
        if self._data is not None:
            print("Using cached derivative")
            return self._data
        
        if not hasattr(self._diag, 'data') or self._diag._data is None:
            self._diag.load_all()

        if self._type == "t":
            result = np.gradient(self._diag._data, self._diag._dt * self._diag._ndump, axis=0, edge_order=2)

        elif self._type == "x1":
            if self._dim == 1:
                result = np.gradient(self._diag._data, self._diag._dx, axis=1, edge_order=2)
            else:
                result = np.gradient(self._diag._data, self._diag._dx[0], axis=1, edge_order=2)
                    
        elif self._type == "x2":
            result = np.gradient(self._diag._data, self._diag._dx[0], axis=2, edge_order=2)

        elif self._type == "x3":
            result = np.gradient(self._diag._data, self._diag._dx[0], axis=3, edge_order=2)

        elif self._type == "xx":
            if len(self._axis) != 2:
                raise ValueError("Axis must be a tuple with two elements.")
            result = np.gradient(np.gradient(self._diag._data, self._diag._dx[self._axis[0]-1], axis=self._axis[0], edge_order=2), self._diag._dx[self._axis[1]-1], axis=self._axis[1], edge_order=2)
            
        elif self._type == "xt":
            if not isinstance(self._axis, int):
                raise ValueError("Axis must be an integer.")
            result = np.gradient(np.gradient(self._diag._data, self._diag._dt, axis=0, edge_order=2), self._diag._dx[self._axis-1], axis=self._axis[0], edge_order=2)
            
        elif self._type == "tx":
            if not isinstance(self._axis, int):
                raise ValueError("Axis must be an integer.")
            result = np.gradient(np.gradient(self._diag._data, self._diag._dx[self._axis-1], axis=self._axis, edge_order=2), self._diag._dt, axis=0, edge_order=2)
        else:
            raise ValueError("Invalid self._type.")
        
        # Store the result in the cache
        self._all_loaded = True
        self._data = result

    def _data_generator(self, index):
        if self._type == "x1":
            if self._dim == 1:
                yield np.gradient(self._diag[index], self._diag._dx, axis=0, edge_order=2)
            else:
                yield np.gradient(self._diag[index], self._diag._dx[0], axis=0, edge_order=2)
        
        elif self._type == "x2":
            yield np.gradient(self._diag[index], self._diag._dx[1], axis=1, edge_order=2)
        
        elif self._type == "x3":
            yield np.gradient(self._diag[index], self._diag._dx[2], axis=2, edge_order=2)
            
        elif self._type == "t":
            if index == 0:
                yield (-3 * self._diag[index] + 4 * self._diag[index + 1] - self._diag[index + 2]) / (2 * self._diag._dt * self._diag._ndump)
            # derivate at last point not implemented yet
            # elif self[point + 1] is None:
            #     return (3 * self[point] - 4 * self[point - 1] + self[point - 2]) / (2 * self._dt)
            else:
                yield (self._diag[index + 1] - self._diag[index - 1]) / (2 * self._diag._dt * self._diag._ndump)
        else:
            raise ValueError("Invalid derivative type. Use 'x1', 'x2' or 't'.")

    def __getitem__(self, index):
        return next(self._data_generator(index))
        
    def __add__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            # Create a new Derivative_Diagnostic instance
            result = Derivative_Diagnostic(self._diag, self._type, self._axis)
            
            # Copy all necessary attributes
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            # Update the name to reflect the operation
            result._name = self._name + " + " + str(other) if isinstance(other, (int, float)) else self._name + " + np.ndarray"
            
            if self._all_loaded:
                result._data = self._data + other
                result._all_loaded = True
            else:
                def gen_scalar_add(original_gen, scalar):
                    for i in original_gen:
                        yield i + scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_add(original_generator(index), other)

            return result
        
        elif isinstance(other, Derivative_Diagnostic):
            result = Derivative_Diagnostic(self._diag*other._diag, self._type, self._axis)
            result._name = self._name + " + " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data + other._data
                result._all_loaded = True
            else:
                def gen_derivative_add(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i + j

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_add(original_generator(index), other_generator(index))

            return result

        elif other.__class__.__name__ == "Diagnostic":
            result = Derivative_Diagnostic(self._diag*other, self._type, self._axis)
            result._name = self._name + " + " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data + other._data
                result._all_loaded = True
            else:
                def gen_derivative_add(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i + j

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_add(original_generator(index), other_generator(index))

            return result
        
        else:
            raise ValueError("Invalid type for addition operation.")
        
    def __sub__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            # Create a new Derivative_Diagnostic instance
            result = Derivative_Diagnostic(self._diag, self._type, self._axis)
            
            # Copy all necessary attributes
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            # Update the name to reflect the operation
            result._name = self._name + " - " + str(other) if isinstance(other, (int, float)) else self._name + " - np.ndarray"
            
            if self._all_loaded:
                result._data = self._data - other
                result._all_loaded = True
            else:
                def gen_scalar_sub(original_gen, scalar):
                    for i in original_gen:
                        yield i - scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_sub(original_generator(index), other)

            return result
        
        elif isinstance(other, Derivative_Diagnostic):
            result = Derivative_Diagnostic(self._diag*other._diag, self._type, self._axis)
            result._name = self._name + " - " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))  

            if self._all_loaded:
                other.load_all()
                result._data = self._data - other._data
                result._all_loaded = True
            else:
                def gen_derivative_sub(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i - j 

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_sub(original_generator(index), other_generator(index))

            return result
        
        elif other.__class__.__name__ == "Diagnostic":
            result = Derivative_Diagnostic(self._diag*other, self._type, self._axis)
            result._name = self._name + " - " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data - other._data
                result._all_loaded = True
            else:
                def gen_derivative_sub(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i - j


                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_sub(original_generator(index), other_generator(index))

            return result
        
        else:
            raise ValueError("Invalid type for subtraction operation.")
        
    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            # Create a new Derivative_Diagnostic instance
            result = Derivative_Diagnostic(self._diag, self._type, self._axis)
            
            # Copy all necessary attributes
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            # Update the name to reflect the operation
            result._name = self._name + " * " + str(other) if isinstance(other, (int, float)) else self._name + " * np.ndarray"
            
            if self._all_loaded:
                result._data = self._data * other
                result._all_loaded = True
            else:
                def gen_scalar_mul(original_gen, scalar):
                    for i in original_gen:
                        yield i * scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_mul(original_generator(index), other)

            return result
        
        elif isinstance(other, Derivative_Diagnostic):
            result = Derivative_Diagnostic(self._diag*other._diag, self._type, self._axis)
            result._name = self._name + " * " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data * other._data
                result._all_loaded = True
            else:
                def gen_derivative_mul(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i * j

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_mul(original_generator(index), other_generator(index))

            return result
        
        elif other.__class__.__name__ == "Diagnostic":
            result = Derivative_Diagnostic(self._diag*other, self._type, self._axis)
            result._name = self._name + " * " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data * other._data
                result._all_loaded = True
            else:
                def gen_derivative_mul(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i * j     

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_mul(original_generator(index), other_generator(index))

            return result
        
        else:
            raise ValueError("Invalid type for multiplication operation.")
        
    def __truediv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            # Create a new Derivative_Diagnostic instance
            result = Derivative_Diagnostic(self._diag, self._type, self._axis)
            
            # Copy all necessary attributes
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            # Update the name to reflect the operation
            result._name = self._name + " / " + str(other) if isinstance(other, (int, float)) else self._name + " / np.ndarray"
            
            if self._all_loaded:
                result._data = self._data / other
                result._all_loaded = True
            else:
                def gen_scalar_div(original_gen, scalar):
                    for i in original_gen:
                        yield i / scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_div(original_generator(index), other)

            return result
        
        elif isinstance(other, Derivative_Diagnostic):
            result = Derivative_Diagnostic(self._diag*other._diag, self._type, self._axis)
            result._name = self._name + " / " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data / other._data
                result._all_loaded = True
            else:
                def gen_derivative_div(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i / j

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_div(original_generator(index), other_generator(index))

            return result
        
        elif other.__class__.__name__ == "Diagnostic":
            result = Derivative_Diagnostic(self._diag*other, self._type, self._axis)
            result._name = self._name + " / " + other._name

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

            if self._all_loaded:
                other.load_all()
                result._data = self._data / other._data
                result._all_loaded = True
            else:
                def gen_derivative_div(original_gen, other_gen):
                    for i, j in zip(original_gen, other_gen):
                        yield i / j 

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_derivative_div(original_generator(index), other_generator(index))

            return result
        
        else:
            raise ValueError("Invalid type for division operation.")
        
    def __pow__(self, other):
        raise NotImplementedError("Power operation not implemented yet.")
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return self / other