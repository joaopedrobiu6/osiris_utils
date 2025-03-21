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
import matplotlib.pyplot as plt
import warnings
from typing import Literal

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
        self._label = None
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
        self._label = dump1.label
        self._dim = dump1.dim
        self._ndump = dump1.iter
    
    def _data_generator(self, index):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        file = os.path.join(self._path, self._file_template + f"{index:06d}.h5")
        data_object = OsirisGridFile(file)
        yield data_object.data

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
                
                # Load data for all timesteps using the generator - this may take a while
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
        raise ValueError(f"Cannot retrieve data for this diagnostic at index {index}. No data loaded and no generator available.")   
    
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
            
            if self._all_loaded:
                result._data = self._data + other
                result._all_loaded = True
            else:
                def gen_scalar_add(original_gen, scalar):
                    for val in original_gen:
                        yield val + scalar
                
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_add(original_generator(index), other)
                
            return result

        elif isinstance(other, Diagnostic):
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " + " + str(other._name)

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
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " - " + str(other) if isinstance(other, (int, float)) else self._name + " - np.ndarray"

            if self._all_loaded:
                result._data = self._data - other
                result._all_loaded = True
            else:
                def gen_scalar_sub(original_gen, scalar):
                    for val in original_gen:
                        yield val - scalar
                
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_sub(original_generator(index), other)
                
            return result

        elif isinstance(other, Diagnostic):
                
            
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " - " + str(other._name)

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
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " * " + str(other) if isinstance(other, (int, float)) else self._name + " * np.ndarray"
            
            if self._all_loaded:
                result._data = self._data * other
                result._all_loaded = True
            else:
                def gen_scalar_mul(original_gen, scalar):
                    for val in original_gen:
                        yield val * scalar
                
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_mul(original_generator(index), other)
                
            return result

        elif isinstance(other, Diagnostic):
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " * " + str(other._name)

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
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " / " + str(other) if isinstance(other, (int, float)) else self._name + " / np.ndarray"
            
            if self._all_loaded:
                result._data = self._data / other
                result._all_loaded = True
            else:
                def gen_scalar_div(original_gen, scalar):
                    for val in original_gen:
                        yield val / scalar
                
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_div(original_generator(index), other)
                
            return result

        elif isinstance(other, Diagnostic):
                
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name = self._name + " / " + str(other._name)

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
       # power by scalar
        if isinstance(other, (int, float)):
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = self._name + " ^(" + str(other) + ")"
            result._label = self._label + rf"$ ^{other}$"

            if self._all_loaded:
                result._data = self._data ** other
                result._all_loaded = True
            else:
                def gen_scalar_pow(original_gen, scalar):
                    for val in original_gen:
                        yield val ** scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_pow(original_generator(index), other)

            return result
        
        # power by another diagnostic
        elif isinstance(other, Diagnostic):
            raise ValueError("Power by another diagnostic is not supported. Why would you do that?")

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other): # I don't know if this is correct because I'm not sure if the order of the subtraction is correct
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other): # division is not commutative
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(self._species)
        
            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter

            result._name = str(other) + " / " + self._name if isinstance(other, (int, float)) else "np.ndarray / " + self._name
            
            if self._all_loaded:
                result._data = other / self._data
                result._all_loaded = True
            else:
                def gen_scalar_rdiv(scalar, original_gen):
                    for val in original_gen:
                        yield scalar / val
                
                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_rdiv(other, original_generator(index))
                
            return result
        
        elif isinstance(other, Diagnostic):
            
            result = Diagnostic(self._species)

            for attr in ['_dx', '_nx', '_x', '_dt', '_grid', '_axis', '_dim', '_ndump', '_maxiter']:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))
        
            if not hasattr(result, '_maxiter') or result._maxiter is None:
                if hasattr(self, '_maxiter') and self._maxiter is not None:
                    result._maxiter = self._maxiter
            
            result._name =  str(other._name) + " / " + self._name

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

    def plot_3d(self, idx, scale_type: Literal["zero_centered", "pos", "neg", "default"] = "default", boundaries: np.ndarray = None):
        """
        Plots a 3D scatter plot of the diagnostic data (grid data).

        Parameters
        ----------
        idx : int
            Index of the data to plot.
        scale_type : Literal["zero_centered", "pos", "neg", "default"], optional
            Type of scaling for the colormap:
            - "zero_centered": Center colormap around zero.
            - "pos": Colormap for positive values.
            - "neg": Colormap for negative values.
            - "default": Standard colormap.
        boundaries : np.ndarray, optional
            Boundaries to plot part of the data. (3,2) If None, uses the default grid boundaries.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axes object of the plot.

        Example
        -------
        sim = ou.Simulation("electrons", "path/to/simulation")
        fig, ax = sim["b3"].plot_3d(55, scale_type="zero_centered",  boundaries= [[0, 40], [0, 40], [0, 20]])
        plt.show()
        """


        if self._dim != 3:
            raise ValueError("This method is only available for 3D diagnostics.")
        
        if boundaries is None:
            boundaries = self._grid

        if not isinstance(boundaries, np.ndarray):
            try :
                boundaries = np.array(boundaries)
            except:
                boundaries = self._grid 
                warnings.warn("boundaries cannot be accessed as a numpy array with shape (3, 2), using default instead")

        if boundaries.shape != (3, 2):
            warnings.warn("boundaries should have shape (3, 2), using default instead")
            boundaries = self._grid 

        # Load data
        if self._all_loaded:
            data = self._data[idx]
        else:
            data = self[idx]

        # Create grid points
        x = self._x[0]
        y = self._x[1]
        z = self._x[2]

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Flatten arrays for scatter plot
        X_flat, Y_flat, Z_flat, = X.ravel(), Y.ravel(), Z.ravel()
        data_flat = data.ravel()

        # Apply filter: Keep only chosen points
        mask = (X_flat > boundaries[0][0]) & (X_flat < boundaries[0][1]) & (Y_flat > boundaries[1][0]) & (Y_flat < boundaries[1][1]) & (Z_flat > boundaries[2][0]) & (Z_flat < boundaries[2][1])
        X_cut, Y_cut, Z_cut, data_cut = X_flat[mask], Y_flat[mask], Z_flat[mask], data_flat[mask]

        if scale_type == "zero_centered":
            # Center colormap around zero
            cmap = "seismic"
            vmax = np.max(np.abs(data_flat))  # Find max absolute value
            vmin = -vmax
        elif scale_type == "pos":
            cmap = "plasma"
            vmax = np.max(data_flat)
            vmin = 0

        elif scale_type == "neg":
            cmap = "plasma"
            vmax = 0
            vmin = np.min(data_flat)
        else:
            cmap = "plasma"
            vmax = np.max(data_flat)
            vmin = np.min(data_flat)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot with seismic colormap
        sc = ax.scatter(X_cut, Y_cut, Z_cut, c=data_cut, cmap=cmap, norm=norm, alpha=1)

        # Set limits to maintain full background
        ax.set_xlim([self.grid[0][0], self.grid[0][1]])
        ax.set_ylim([self.grid[1][0], self.grid[1][1]])
        ax.set_zlim([self.grid[2][0], self.grid[2][1]])

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

        # Labels
        # TODO try to use a latex label instaead of _name
        cbar.set_label(r"${}$".format(self._name) + r"$\  [{}]$".format(self._units))
        ax.set_title(r"$t={:.2f}$".format(self.time(idx)[0]) + r"$\  [{}]$".format(self.time(idx)[1]))
        ax.set_xlabel(r"${}$".format(self.axis[0]["long_name"]) + r"$\  [{}]$".format(self.axis[0]["units"]))
        ax.set_ylabel(r"${}$".format(self.axis[1]["long_name"]) + r"$\  [{}]$".format(self.axis[1]["units"]))
        ax.set_zlabel(r"${}$".format(self.axis[2]["long_name"]) + r"$\  [{}]$".format(self.axis[2]["units"]))

        return fig, ax

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
    
    @property
    def maxiter(self):
        return self._maxiter
    
    @property
    def label(self):
        return self._label
    
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