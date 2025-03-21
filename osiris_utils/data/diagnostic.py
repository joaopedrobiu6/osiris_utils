"""
The utilities on data.py are cool but not useful when you want to work with whole data of a simulation instead
of just a single file. This is what this file is for - deal with ''folders'' of data.

Took some inspiration from Diogo and Madox's work.

This would be awsome to compute time derivatives. 
"""
import numpy as np
import os

from .data import OsirisGridFile
import tqdm

OSIRIS_DENSITY = "n"
OSIRIS_SPECIE_REPORTS = ["charge", "q1", "q2", "q3", "j1", "j2", "j3"]
OSIRIS_SPECIE_REP_UDIST = [
    "vfl1",
    "vfl2",
    "vfl3",
    "ufl1",
    "ufl2",
    "ufl3",
    "P11",
    "P12",
    "P13",
    "P22",
    "P23",
    "P33",
    "T11",
    "T12",
    "T13",
    "T22",
    "T23",
    "T33",
]
OSIRIS_FLD = ["e1", "e2", "e3", "b1", "b2", "b3"]
OSIRIS_PHA = ["p1x1", "p1x2", "p1x3", "p2x1", "p2x2", "p2x3", "p3x1", "p3x2", "p3x3", "gammax1", "gammax2", "gammax3"] # there may be more that I don't know


class Diagnostic:
    """
    Class to handle the diagnostics of a simulation. This is mainly used by Simulation class to handle the diagnostics.

    Parameters
    ----------
    species : str
        The species to handle the diagnostics.
    simulation_folder : str
        The path to the simulation folder. This is the path to the folder where the input deck is located.
    """
    def __init__(self, species, simulation_folder=None):
        self._species = species

        self._dx = None
        self._nx = None
        self._x = None
        self._dt = None
        self._grid = None
        self._axis = None
        self._units = None
        self._name = None
        self._dim = None
        self._ndump = None
        self._maxiter = None
        
        if simulation_folder:
            self._simulation_folder = simulation_folder
            if not os.path.isdir(simulation_folder):
                raise FileNotFoundError(f"Simulation folder {simulation_folder} not found.")
        else:
            self._simulation_folder = None

        self._all_loaded = False
    
    def get_quantity(self, quantity):
        if quantity in OSIRIS_SPECIE_REP_UDIST:
            self._get_moment(self._species, quantity)
        elif quantity in OSIRIS_SPECIE_REPORTS:
            self._get_density(self._species, quantity)
        elif quantity in OSIRIS_FLD:
            self._get_field(quantity)
        elif quantity in OSIRIS_PHA:
            self._get_phase_space(self._species, quantity)
        else:
            raise ValueError(f"Invalid quantity {quantity}. Or it's not implemented yet (this may happen for phase space quantities).")

    def _get_moment(self, species, moment):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._maxiter = len(os.listdir(self._path))
        self._load_attributes(self._file_template)
    
    def _get_field(self, field):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._maxiter = len(os.listdir(self._path))
        self._load_attributes(self._file_template)
        
    def _get_density(self, species, quantity):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._maxiter = len(os.listdir(self._path))
        self._load_attributes(self._file_template)

    def _get_phase_space(self, species, type):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/PHA/{type}/{species}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._maxiter = len(os.listdir(self._path))
        self._load_attributes(self._file_template)

    def _load_attributes(self, file_template): # this will be replaced by reading the input deck
        # This can go wrong! NDUMP
        path_file1 = os.path.join(self._path, file_template + "000001.h5")
        dump1 = OsirisGridFile(path_file1)
        self._dx = dump1.dx
        self._nx = dump1.nx
        self._x = dump1.x
        self._dt = dump1.dt
        self._grid = dump1.grid
        self._axis = dump1.axis
        self._units = dump1.units
        self._name = dump1.name
        self._dim = dump1.dim
        self._ndump = dump1.iter
    
    def _data_generator(self, index):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        file = os.path.join(self._path, self._file_template + f"{index:06d}.h5")
        data_object = OsirisGridFile(file)
        yield data_object.data
    
    # def load_all(self, centered=False):
    #     print("Loading all data. This may take a while.")
    #     if self._simulation_folder is None:
    #         raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
    #     self._current_centered = centered
    #     size = len(sorted(os.listdir(self._path)))
    #     self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])
    #     self._all_loaded = True

    def load_all(self):
        # If data is already loaded, don't do anything
        if self._all_loaded and self._data is not None:
            print("Data already loaded.")
            return self._data
        
        # If this is a derived diagnostic without files
        if self._simulation_folder is None:
            # If it has a data generator but no direct files
            try:
                print("This appears to be a derived diagnostic. Loading data from generators...")
                # Get the maximum size from the diagnostic attributes
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    size = self._maxiter
                else:
                    # Try to infer from a related diagnostic
                    if hasattr(self, '_diag') and hasattr(self._diag, '_maxiter'):
                        size = self._diag._maxiter
                    else:
                        # Default to a reasonable number if we can't determine
                        size = 100
                        print(f"Warning: Could not determine timestep count, using {size}.")
                
                # Load data for all timesteps using the generator
                self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])
                self._all_loaded = True
                return self._data
            except Exception as e:
                raise ValueError(f"Could not load derived diagnostic data: {str(e)}")
        
        # Original implementation for file-based diagnostics
        print("Loading all data from files. This may take a while.")
        size = len(sorted(os.listdir(self._path)))
        self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])
        self._all_loaded = True
        return self._data
    
    def load(self, index):
        self._data = next(self._data_generator(index))

    def __getitem__(self, index):
        # For standard diagnostics with files
        if self._simulation_folder is not None and hasattr(self, '_data_generator'):
            return next(self._data_generator(index))
        
        # For derived diagnostics with cached data
        if self._all_loaded and self._data is not None:
            return self._data[index]
        
        # For derived diagnostics with custom generators
        if hasattr(self, '_data_generator') and callable(self._data_generator):
            return next(self._data_generator(index))
        
        # If we get here, we don't know how to get data for this index
        raise ValueError("Cannot retrieve data for this diagnostic at index {index}. No data loaded and no generator available.")   
    
    def __iter__(self):
        # If this is a file-based diagnostic
        if self._simulation_folder is not None:
            for i in range(len(sorted(os.listdir(self._path)))):
                yield next(self._data_generator(i))
        
        # If this is a derived diagnostic and data is already loaded
        elif self._all_loaded and self._data is not None:
            for i in range(self._data.shape[0]):
                yield self._data[i]
        
        # If this is a derived diagnostic with custom generator but no loaded data
        elif hasattr(self, '_data_generator') and callable(self._data_generator):
            # Determine how many iterations to go through
            max_iter = self._maxiter
            if max_iter is None:
                if hasattr(self, '_diag') and hasattr(self._diag, '_maxiter'):
                    max_iter = self._diag._maxiter
                else:
                    max_iter = 100  # Default if we can't determine
                    print(f"Warning: Could not determine iteration count for iteration, using {max_iter}.")
            
            for i in range(max_iter):
                yield next(self._data_generator(i))
        
        # If we don't know how to handle this
        else:
            raise ValueError("Cannot iterate over this diagnostic. No data loaded and no generator available.")

    def __add__(self, other):
        # Scalar addition
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " + " + str(other) if isinstance(other, (int, float)) else self._name + " + np.ndarray"
            
            # If data already loaded, add directly
            if self._all_loaded:
                result._data = self._data + other
                result._all_loaded = True
            else:
                def gen_scalar_add(original_gen, scalar):
                    for val in original_gen:
                        yield val + scalar
                
                # Override the data generator
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_add(original_generator(index), other)
                
            return result

        # Handle diagnostic addition
        elif isinstance(other, Diagnostic):
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " + " + str(other)

            if self._all_loaded:
                other.load_all()
                result._data = self._data + other._data
                result._all_loaded = True
            else:
                def gen_diag_add(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield val1 + val2
                
                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_add(original_generator(index), other_generator(index))

            return result
    

    def __sub__(self, other):
        # Scalar subtraction
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " - " + str(other) if isinstance(other, (int, float)) else self._name + " - np.ndarray"
            
            # If data already loaded, add directly
            if self._all_loaded:
                result._data = self._data - other
                result._all_loaded = True
            else:
                def gen_scalar_sub(original_gen, scalar):
                    for val in original_gen:
                        yield val - scalar
                
                # Override the data generator
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_sub(original_generator(index), other)
                
            return result

        # Handle diagnostic subtraction
        elif isinstance(other, Diagnostic):
                
            
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " - " + str(other)

            if self._all_loaded:
                other.load_all()
                result._data = self._data - other._data
                result._all_loaded = True
            else:
                def gen_diag_sub(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield val1 - val2
                
                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_sub(original_generator(index), other_generator(index))

            return result
    
    def __mul__(self, other):
        # Scalar multiplication
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " * " + str(other) if isinstance(other, (int, float)) else self._name + " * np.ndarray"
            
            # If data already loaded, add directly
            if self._all_loaded:
                result._data = self._data * other
                result._all_loaded = True
            else:
                def gen_scalar_mul(original_gen, scalar):
                    for val in original_gen:
                        yield val * scalar
                
                # Override the data generator
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_mul(original_generator(index), other)
                
            return result

        # Handle diagnostic multiplication
        elif isinstance(other, Diagnostic):
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " * " + str(other) 

            if self._all_loaded:
                other.load_all()
                result._data = self._data * other._data
                result._all_loaded = True
            else:
                def gen_diag_mul(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield val1 * val2
                
                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_mul(original_generator(index), other_generator(index))

            return result
    
    def __truediv__(self, other):
        # Scalar division
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " / " + str(other) if isinstance(other, (int, float)) else self._name + " / np.ndarray"
            
            # If data already loaded, add directly
            if self._all_loaded:
                result._data = self._data / other
                result._all_loaded = True
            else:
                def gen_scalar_div(original_gen, scalar):
                    for val in original_gen:
                        yield val / scalar
                
                # Override the data generator
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_div(original_generator(index), other)
                
            return result

        # Handle diagnostic division
        elif isinstance(other, Diagnostic):
                
            
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " / " + str(other)

            if self._all_loaded:
                other.load_all()
                result._data = self._data / other._data
                result._all_loaded = True
            else:
                def gen_diag_div(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield val1 / val2
                
                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_div(original_generator(index), other_generator(index))

            return result
        
    def __pow__(self, other):
       raise NotImplementedError("Power operation not implemented for Diagnostic objects.")

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        # Scalar divided by diagnostic
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = str(other) + " / " + self._name if isinstance(other, (int, float)) else "np.ndarray / " + self._name
            
            # If data already loaded, divide directly
            if self._all_loaded:
                result._data = other / self._data
                result._all_loaded = True
            else:
                def gen_scalar_rdiv(scalar, original_gen):
                    for val in original_gen:
                        yield scalar / val
                
                # Override the data generator
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_rdiv(other, original_generator(index))
                
            return result
        
        elif isinstance(other, Diagnostic):
            
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name =  str(other) + " / " + self._name

            if self._all_loaded:
                other.load_all()
                result._data =  other._data / self._data
                result._all_loaded = True
            else:
                def gen_diag_div(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield  val2 / val1
                
                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_div(original_generator(index), other_generator(index))

            return result

    # Getters
    @property
    def data(self):
        if self._data is None:
            raise ValueError("Data not loaded into memory. Use get_* method with load_all=True or access via generator/index.")
        return self._data

    @property
    def dx(self):
        return self._dx
    
    @property
    def nx(self):
        return self._nx
    
    @property
    def x(self):
        return self._x
    
    @property
    def dt(self):
        return self._dt
    
    @property
    def grid(self):
        return self._grid
    
    @property
    def axis(self):
        return self._axis
    
    @property
    def units(self):
        return self._units
    
    @property
    def name(self):
        return self._name
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def path(self):
        return self
    
    @property
    def simulation_folder(self):
        return self._simulation_folder
    
    @property
    def ndump(self):
        return self._ndump
    
    @property
    def all_loaded(self):
        return self._all_loaded
    
    def time(self, index):
        return [index * self._dt * self._ndump, r"$1 / \omega_p$"]
    
    @dx.setter
    def dx(self, value):
        self._dx = value
    
    @nx.setter
    def nx(self, value):
        self._nx = value

    @x.setter
    def x(self, value):
        self._x = value

    @dt.setter
    def dt(self, value):
        self._dt = value

    @grid.setter
    def grid(self, value):
        self._grid = value

    @axis.setter
    def axis(self, value):
        self._axis = value

    @units.setter
    def units(self, value):
        self._units = value

    @name.setter
    def name(self, value):
        self._name = value

    @dim.setter
    def dim(self, value):
        self._dim = value

    @ndump.setter
    def ndump(self, value):
        self._ndump = value
 
    @data.setter
    def data(self, value):
        self._data = value       