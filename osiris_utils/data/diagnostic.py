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
import itertools
import multiprocessing as mp

OSIRIS_DENSITY = "n"
OSIRIS_SPECIE_REPORTS = [
    "charge",
    "q1",
    "q2",
    "q3",
    "j1",
    "j2",
    "j3",
]
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
        self._current_centered = False
        
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
        self._load_attributes(self._file_template)
    
    def _get_field(self, field, centered=False):
        self._current_centered = False
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        if centered:
            self._current_centered = True
            self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._load_attributes(self._file_template)
        
    def _get_density(self, species, quantity):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._load_attributes(self._file_template)

    def _get_phase_space(self, species, type):
        self._current_centered = False
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/PHA/{type}/{species}/"
        self._file_template = os.listdir(self._path)[0][:-9]
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
        if self._current_centered:
            data_object.yeeToCellCorner(boundary="periodic")
        yield data_object.data_centered if self._current_centered else data_object.data
    
    def load_all(self, centered=False):
        print("Loading all data. This may take a while.")
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._current_centered = centered
        size = len(sorted(os.listdir(self._path)))
        self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])
        self._all_loaded = True

    def load_all_parallel(self, centered=False, processes=None):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._current_centered = centered
        files = sorted(os.listdir(self._path))
        size = len(files)
        
        if processes is None:
            processes = mp.cpu_count()
            print(f"Using {processes} CPUs for parallel loading")
        
        with mp.Pool(processes=processes) as pool:
            data = list(tqdm.tqdm(pool.imap(self.__getitem__, range(size)), total=size, desc="Loading data"))
        
        self._data = np.stack(data)
        self._all_loaded = True
    
    def load(self, index, centered=False):
        self._current_centered = centered
        self._data = next(self._data_generator(index))

    def __getitem__(self, index):
        return next(self._data_generator(index))
    
    def __iter__(self):
        for i in itertools.count():
            yield next(self._data_generator(i))

    def __add__(self, other):
        # Scalar addition
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

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
        if not isinstance(other, Diagnostic):
            raise TypeError("Can only add Diagnostic objects or scalars.")
        
        result = Diagnostic(self._species)

        for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
            setattr(result, attr, getattr(self, attr))
        
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
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

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
        if not isinstance(other, Diagnostic):
            raise TypeError("Can only subtract Diagnostic objects or scalars.")
        
        result = Diagnostic(self._species)

        for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
            setattr(result, attr, getattr(self, attr))
        
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
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

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

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))
            
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
        
        elif other.__class__.__name__ == "Derivative_Auxiliar":
            return other * self
    
    def __truediv__(self, other):
        # Scalar division
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
                setattr(result, attr, getattr(self, attr))

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
        if not isinstance(other, Diagnostic):
            raise TypeError("Can only divide Diagnostic objects or scalars.")
        
        result = Diagnostic(self._species)

        for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump']:
            setattr(result, attr, getattr(self, attr))
        
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
        return self / other
    
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