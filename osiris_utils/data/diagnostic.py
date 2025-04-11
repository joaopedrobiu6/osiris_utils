"""
The utilities on data.py are cool but not useful when you want to work with whole data of a simulation instead
of just a single file. This is what this file is for - deal with ''folders'' of data.

Took some inspiration from Diogo and Madox's work.

This would be awsome to compute time derivatives.
"""

from torch import isin
import numpy as np
import os
import glob
import h5py

from .data import OsirisGridFile
import tqdm
import matplotlib.pyplot as plt
import warnings
from typing import Literal
from ..decks.decks import InputDeckIO, deval


def get_dimension_from_deck(deck: InputDeckIO) -> int:
    for dim in range(1, 4):
        try:
            deck.get_param(section="grid", param=f"nx_p(1:{dim})")
            return dim
        except:
            continue

    raise Exception("Error parsing grid dimension")


OSIRIS_DENSITY = ["n"]
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
OSIRIS_FLD = [
    "e1",
    "e2",
    "e3",
    "b1",
    "b2",
    "b3",
    "part_e1",
    "part_e2",
    "epart_3",
    "part_b1",
    "part_b2",
    "part_b3",
    "ext_e1",
    "ext_e2",
    "ext_e3",
    "ext_b1",
    "ext_b2",
    "ext_b3",
]
OSIRIS_PHA = [
    "p1x1",
    "p1x2",
    "p1x3",
    "p2x1",
    "p2x2",
    "p2x3",
    "p3x1",
    "p3x2",
    "p3x3",
    "gammax1",
    "gammax2",
    "gammax3",
]  # there may be more that I don't know
OSIRIS_ALL = (
    OSIRIS_DENSITY
    + OSIRIS_SPECIE_REPORTS
    + OSIRIS_SPECIE_REP_UDIST
    + OSIRIS_FLD
    + OSIRIS_PHA
)


def which_quantities():
    print("Available quantities:")
    print(OSIRIS_ALL)


class Diagnostic:
    """
    Class to handle diagnostics. This is the "base" class of the code. Diagnostics can be loaded from OSIRIS output files, but are also created when performing operations with other diagnostics.
    Post-processed quantities are also considered diagnostics. This way, we can perform operations with them as well.

    Parameters
    ----------
    species : str
        The species to handle the diagnostics.
    simulation_folder : str
        The path to the simulation folder. This is the path to the folder where the input deck is located.

    Attributes
    ----------
    species : str
        The species to handle the diagnostics.
    dx : np.ndarray(float) or float
        The grid spacing in each direction. If the dimension is 1, this is a float. If the dimension is 2 or 3, this is a np.ndarray.
    nx : np.ndarray(int) or int
        The number of grid points in each direction. If the dimension is 1, this is a int. If the dimension is 2 or 3, this is a np.ndarray.
    x : np.ndarray
        The grid points.
    dt : float
        The time step.
    grid : np.ndarray
        The grid boundaries.
    axis : dict
        The axis information. Each key is a direction and the value is a dictionary with the keys "name", "long_name", "units" and "plot_label".
    units : str
        The units of the diagnostic. This info may not be available for all diagnostics, ie, diagnostics resulting from operations and postprocessing.
    name : str
        The name of the diagnostic. This info may not be available for all diagnostics, ie, diagnostics resulting from operations and postprocessing.
    label : str
        The label of the diagnostic. This info may not be available for all diagnostics, ie, diagnostics resulting from operations and postprocessing.
    dim : int
        The dimension of the diagnostic.
    ndump : int
        The number of steps between dumps.
    maxiter : int
        The maximum number of iterations.
    tunits : str
        The time units.
    path : str
        The path to the diagnostic.
    simulation_folder : str
        The path to the simulation folder.
    all_loaded : bool
        If the data is already loaded into memory. This is useful to avoid loading the data multiple times.
    data : np.ndarray
        The diagnostic data. This is created only when the data is loaded into memory.

    Methods
    -------
    get_quantity(quantity)
        Get the data for a given quantity.
    load_all()
        Load all data into memory.
    load(index)
        Load data for a given index.
    __getitem__(index)
        Get data for a given index. Does not load the data into memory.
    __iter__()
        Iterate over the data. Does not load the data into memory.
    __add__(other)
        Add two diagnostics.
    __sub__(other)
        Subtract two diagnostics.
    __mul__(other)
        Multiply two diagnostics.
    __truediv__(other)
        Divide two diagnostics.
    __pow__(other)
        Power of a diagnostic.
    plot_3d(idx, scale_type="default", boundaries=None)
        Plot a 3D scatter plot of the diagnostic data.
    time(index)
        Get the time for a given index.
        
    """

    def __init__(self, simulation_folder=None, species=None, input_deck=None):
        self._species = species if species else None

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
        self._tunits = None

        if simulation_folder:
            self._simulation_folder = simulation_folder
            if not os.path.isdir(simulation_folder):
                raise FileNotFoundError(
                    f"Simulation folder {simulation_folder} not found."
                )
        else:
            self._simulation_folder = None

        # load input deck if available
        if input_deck:
            self._input_deck = input_deck
        else:
            self._input_deck = None

        self._all_loaded = False
        self._quantity = None

    def get_quantity(self, quantity):
        """
        Get the data for a given quantity.

        Parameters
        ----------
        quantity : str
            The quantity to get the data.
        """
        self._quantity = quantity

        if self._quantity not in OSIRIS_ALL:
            raise ValueError(
                f"Invalid quantity {self._quantity}. Use which_quantities() to see the available quantities."
            )
        if self._quantity in OSIRIS_SPECIE_REP_UDIST:
            if self._species is None:
                raise ValueError("Species not set.")
            self._get_moment(self._species.name, self._quantity)
        elif self._quantity in OSIRIS_SPECIE_REPORTS:
            if self._species is None:
                raise ValueError("Species not set.")
            self._get_density(self._species.name, self._quantity)
        elif self._quantity in OSIRIS_FLD:
            self._get_field(self._quantity)
        elif self._quantity in OSIRIS_PHA:
            if self._species is None:
                raise ValueError("Species not set.")
            self._get_phase_space(self._species.name, self._quantity)
        elif self._quantity == "n":
            if self._species is None:
                raise ValueError("Species not set.")
            self._get_density(self._species.name, "charge")
        else:
            raise ValueError(
                f"Invalid quantity {self._quantity}. Or it's not implemented yet (this may happen for phase space quantities)."
            )

    def _get_moment(self, species, moment):
        if self._simulation_folder is None:
            raise ValueError(
                "Simulation folder not set. If you're using CustomDiagnostic, this method is not available."
            )
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._file_template = glob.glob(f"{self._path}/*.h5")[0][:-9]
        self._maxiter = len(glob.glob(f"{self._path}/*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_field(self, field):
        if self._simulation_folder is None:
            raise ValueError(
                "Simulation folder not set. If you're using CustomDiagnostic, this method is not available."
            )
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._file_template = glob.glob(f"{self._path}/*.h5")[0][:-9]
        self._maxiter = len(glob.glob(f"{self._path}/*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_density(self, species, quantity):
        if self._simulation_folder is None:
            raise ValueError(
                "Simulation folder not set. If you're using CustomDiagnostic, this method is not available."
            )
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._file_template = glob.glob(f"{self._path}/*.h5")[0][:-9]
        self._maxiter = len(glob.glob(f"{self._path}/*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_phase_space(self, species, type):
        if self._simulation_folder is None:
            raise ValueError(
                "Simulation folder not set. If you're using CustomDiagnostic, this method is not available."
            )
        self._path = f"{self._simulation_folder}/MS/PHA/{type}/{species}/"
        self._file_template = glob.glob(f"{self._path}/*.h5")[0][:-9]
        self._maxiter = len(glob.glob(f"{self._path}/*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _load_attributes(
        self, file_template, input_deck
    ):  # this will be replaced by reading the input deck
        # This can go wrong! NDUMP
        # if input_deck is not None:
        #     self._dt = float(input_deck["time_step"][0]["dt"])
        #     self._dim = get_dimension_from_deck(input_deck)
        #     self._nx = np.array(list(map(int, input_deck["grid"][0][f"nx_p(1:{self._dim})"].split(','))))
        #     xmin = [deval(input_deck["space"][0][f"xmin(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     xmax = [deval(input_deck["space"][0][f"xmax(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     self._grid = np.array([[xmin[i], xmax[i]] for i in range(self._dim)])
        #     self._dx = (self._grid[:,1] - self._grid[:,0])/self._nx
        #     self._x = [np.arange(self._grid[i,0], self._grid[i,1], self._dx[i]) for i in range(self._dim)]

        self._ndump = int(input_deck["time_step"][0]["ndump"])
        
        try:
            # Try files 000001, 000002, etc. until one is found
            found_file = False
            for file_num in range(1, self._maxiter + 1):
                path_file = os.path.join(file_template + f"{file_num:06d}.h5")
                if os.path.exists(path_file):
                    dump = OsirisGridFile(path_file)
                    self._dx = dump.dx
                    self._nx = dump.nx
                    self._x = dump.x
                    self._dt = dump.dt
                    self._grid = dump.grid
                    self._axis = dump.axis
                    self._units = dump.units
                    self._name = dump.name
                    self._label = dump.label
                    self._dim = dump.dim
                    # self._iter = dump.iter
                    self._tunits = dump.time[1]
                    self._type = dump.type
                    found_file = True
                    break
            
            if not found_file:
                warnings.warn(f"No valid data files found in {self._path} to read metadata from.")
        except Exception as e:
            warnings.warn(f"Error loading diagnostic attributes: {str(e)}. Please verify it there's any file in the folder.")

    def _data_generator(self, index):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set.")
        file = os.path.join(self._file_template + f"{index:06d}.h5")
        data_object = OsirisGridFile(file)
        yield (
            data_object.data
            if self._quantity not in OSIRIS_DENSITY
            else self._species.rqm * data_object.data
        )

    def load_all(self):
        """
        Load all data into memory (all iterations).

        Returns
        -------
        data : np.ndarray
            The data for all iterations. Also stored in the attribute data.
        """
        # If data is already loaded, don't do anything
        if self._all_loaded and self._data is not None:
            print("Data already loaded.")
            return self._data

        # If this is a derived diagnostic without files
        if self._simulation_folder is None:
            # If it has a data generator but no direct files
            try:
                print(
                    "This appears to be a derived diagnostic. Loading data from generators..."
                )
                # Get the maximum size from the diagnostic attributes
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    size = self._maxiter
                else:
                    # Try to infer from a related diagnostic
                    if hasattr(self, "_diag") and hasattr(self._diag, "_maxiter"):
                        size = self._diag._maxiter
                    else:
                        # Default to a reasonable number if we can't determine
                        size = 100
                        print(
                            f"Warning: Could not determine timestep count, using {size}."
                        )

                # Load data for all timesteps using the generator - this may take a while
                self._data = np.stack(
                    [self[i] for i in tqdm.tqdm(range(size), desc="Loading data")]
                )
                self._all_loaded = True
                return self._data

            except Exception as e:
                raise ValueError(f"Could not load derived diagnostic data: {str(e)}")

        # Original implementation for file-based diagnostics
        print("Loading all data from files. This may take a while.")
        size = len(sorted(glob.glob(f"{self._path}/*.h5")))
        self._data = np.stack(
            [self[i] for i in tqdm.tqdm(range(size), desc="Loading data")]
        )
        self._all_loaded = True
        return self._data

    def unload(self):
        """
        Unload data from memory. This is useful to free memory when the data is not needed anymore.
        """
        print("Unloading data from memory.")
        if self._all_loaded == False:
            print("Data is not loaded.")
            return
        self._data = None
        self._all_loaded = False

    def load(self, index):
        """
        Load data for a given index into memory. Not recommended. Use load_all for all data or access via generator or index for better performance.
        """
        self._data = next(self._data_generator(index))

    def __getitem__(self, index):
        # For derived diagnostics with cached data
        if self._all_loaded and self._data is not None:
            return self._data[index]

        # For standard diagnostics with files
        if isinstance(index, int):
            if self._simulation_folder is not None and hasattr(self, "_data_generator"):
                return next(self._data_generator(index))

            # For derived diagnostics with custom generators
            if hasattr(self, "_data_generator") and callable(self._data_generator):
                return next(self._data_generator(index))

        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step

            if index.stop is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    stop = self._maxiter
                elif self._simulation_folder is not None and hasattr(self, "_path"):
                    stop = len(sorted(glob.glob(f"{self._path}/*.h5")))
                else:
                    stop = 100  # Default if we can't determine
                    print(
                        f"Warning: Could not determine iteration count for iteration, using {stop}."
                    )
            else:
                stop = index.stop

            indices = range(start, stop, step)
            if self._simulation_folder is not None and hasattr(self, "_data_generator"):
                return np.stack([next(self._data_generator(i)) for i in indices])
            elif hasattr(self, "_data_generator") and callable(self._data_generator):
                return np.stack([next(self._data_generator(i)) for i in indices])

        # If we get here, we don't know how to get data for this index
        raise ValueError(
            f"Cannot retrieve data for this diagnostic at index {index}. No data loaded and no generator available."
        )

    def __iter__(self):
        # If this is a file-based diagnostic
        if self._simulation_folder is not None:
            for i in range(len(sorted(glob.glob(f"{self._path}/*.h5")))):
                yield next(self._data_generator(i))

        # If this is a derived diagnostic and data is already loaded
        elif self._all_loaded and self._data is not None:
            for i in range(self._data.shape[0]):
                yield self._data[i]

        # If this is a derived diagnostic with custom generator but no loaded data
        elif hasattr(self, "_data_generator") and callable(self._data_generator):
            # Determine how many iterations to go through
            max_iter = self._maxiter
            if max_iter is None:
                if hasattr(self, "_diag") and hasattr(self._diag, "_maxiter"):
                    max_iter = self._diag._maxiter
                else:
                    max_iter = 100  # Default if we can't determine
                    print(
                        f"Warning: Could not determine iteration count for iteration, using {max_iter}."
                    )

            for i in range(max_iter):
                yield next(self._data_generator(i))

        # If we don't know how to handle this
        else:
            raise ValueError(
                "Cannot iterate over this diagnostic. No data loaded and no generator available."
            )

    def __add__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            # Make sure _maxiter is set even for derived diagnostics
            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " + " + str(other) if isinstance(other, (int, float)) else self._name + " + np.ndarray"

            if self._all_loaded:
                result._data = self._data + other
                result._all_loaded = True
            else:

                def gen_scalar_add(original_gen, scalar):
                    for val in original_gen:
                        yield val + scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_add(
                    original_generator(index), other
                )

            return result

        elif isinstance(other, Diagnostic):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " + " + str(other._name)

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
                result._data_generator = lambda index: gen_diag_add(
                    original_generator(index), other_generator(index)
                )

            return result

    def __sub__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " - " + str(other) if isinstance(other, (int, float)) else self._name + " - np.ndarray"

            if self._all_loaded:
                result._data = self._data - other
                result._all_loaded = True
            else:

                def gen_scalar_sub(original_gen, scalar):
                    for val in original_gen:
                        yield val - scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_sub(
                    original_generator(index), other
                )

            return result

        elif isinstance(other, Diagnostic):

            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " - " + str(other._name)

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
                result._data_generator = lambda index: gen_diag_sub(
                    original_generator(index), other_generator(index)
                )

            return result

    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " * " + str(other) if isinstance(other, (int, float)) else self._name + " * np.ndarray"

            if self._all_loaded:
                result._data = self._data * other
                result._all_loaded = True
            else:

                def gen_scalar_mul(original_gen, scalar):
                    for val in original_gen:
                        yield val * scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_mul(
                    original_generator(index), other
                )

            return result

        elif isinstance(other, Diagnostic):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " * " + str(other._name)

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
                result._data_generator = lambda index: gen_diag_mul(
                    original_generator(index), other_generator(index)
                )

            return result

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " / " + str(other) if isinstance(other, (int, float)) else self._name + " / np.ndarray"

            if self._all_loaded:
                result._data = self._data / other
                result._all_loaded = True
            else:

                def gen_scalar_div(original_gen, scalar):
                    for val in original_gen:
                        yield val / scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_div(
                    original_generator(index), other
                )

            return result

        elif isinstance(other, Diagnostic):

            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " / " + str(other._name)

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
                result._data_generator = lambda index: gen_diag_div(
                    original_generator(index), other_generator(index)
                )

            return result

    def __pow__(self, other):
        # power by scalar
        if isinstance(other, (int, float)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
                "_tunits",
                "_type",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = self._name + " ^(" + str(other) + ")"
            # result._label = self._label + rf"$ ^{other}$"

            if self._all_loaded:
                result._data = self._data**other
                result._all_loaded = True
            else:

                def gen_scalar_pow(original_gen, scalar):
                    for val in original_gen:
                        yield val**scalar

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_pow(
                    original_generator(index), other
                )

            return result

        # power by another diagnostic
        elif isinstance(other, Diagnostic):
            raise ValueError(
                "Power by another diagnostic is not supported. Why would you do that?"
            )

    def __radd__(self, other):
        return self + other

    def __rsub__(
        self, other
    ):  # I don't know if this is correct because I'm not sure if the order of the subtraction is correct
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):  # division is not commutative
        if isinstance(other, (int, float, np.ndarray)):
            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name = str(other) + " / " + self._name if isinstance(other, (int, float)) else "np.ndarray / " + self._name

            if self._all_loaded:
                result._data = other / self._data
                result._all_loaded = True
            else:

                def gen_scalar_rdiv(scalar, original_gen):
                    for val in original_gen:
                        yield scalar / val

                original_generator = self._data_generator
                result._data_generator = lambda index: gen_scalar_rdiv(
                    other, original_generator(index)
                )

            return result

        elif isinstance(other, Diagnostic):

            result = Diagnostic(species=self._species)

            for attr in [
                "_dx",
                "_nx",
                "_x",
                "_dt",
                "_grid",
                "_axis",
                "_dim",
                "_ndump",
                "_maxiter",
            ]:
                if hasattr(self, attr):
                    setattr(result, attr, getattr(self, attr))

            if not hasattr(result, "_maxiter") or result._maxiter is None:
                if hasattr(self, "_maxiter") and self._maxiter is not None:
                    result._maxiter = self._maxiter

            # result._name =  str(other._name) + " / " + self._name

            if self._all_loaded:
                other.load_all()
                result._data = other._data / self._data
                result._all_loaded = True
            else:

                def gen_diag_div(original_gen1, original_gen2):
                    for val1, val2 in zip(original_gen1, original_gen2):
                        yield val2 / val1

                original_generator = self._data_generator
                other_generator = other._data_generator
                result._data_generator = lambda index: gen_diag_div(
                    original_generator(index), other_generator(index)
                )

            return result

    def to_h5(self, folder, savename=None, index=None, all=False, verbose=False):
        """
        Save the diagnostic data to HDF5 files.

        Parameters
        ----------
        folder : str
            The folder to save the HDF5 files.
        savename : str, optional
            The name of the HDF5 file. If None, uses the diagnostic name.
        index : int, or list of ints, optional
            The index or indices of the data to save.
        all : bool, optional
            If True, save all data. Default is False.
        """
        if savename is None:
            print(f"No savename provided. Using {self._name}.")
            savename = self._name

        def savefile(filename, i):
            with h5py.File(filename, 'w') as f:
                # Create SIMULATION group with attributes
                sim_group = f.create_group("SIMULATION")
                sim_group.attrs.create("DT", [self._dt])
                sim_group.attrs.create("NDIMS", [self._dim])
                
                # Set file attributes
                f.attrs.create("TIME", [self.time(i)[0]])
                f.attrs.create("TIME UNITS", [np.bytes_(self.time(i)[1].encode()) if self.time(i)[1] else np.bytes_(b"")])
                f.attrs.create("ITER", [self._ndump * i])
                f.attrs.create("NAME", [np.bytes_(self._name.encode())]) 
                f.attrs.create("TYPE", [np.bytes_(self._type.encode())])
                f.attrs.create("UNITS", [np.bytes_(self._units.encode()) if self._units else np.bytes_(b"")])
                f.attrs.create("LABEL", [np.bytes_(self._label.encode()) if self._label else np.bytes_(b"")])
                
                # Create dataset with data (transposed to match convention)
                f.create_dataset(savename, data=self[i].T)
                
                # Create AXIS group
                axis_group = f.create_group("AXIS")
                
                # Create axis datasets
                axis_names = ["AXIS1", "AXIS2", "AXIS3"][:self._dim]
                axis_shortnames = [self._axis[i]["name"] for i in range(self._dim)]
                axis_longnames = [self._axis[i]["long_name"] for i in range(self._dim)]
                axis_units = [self._axis[i]["units"] for i in range(self._dim)]
                
                for i, axis_name in enumerate(axis_names):
                    # Create axis dataset
                    axis_dataset = axis_group.create_dataset(axis_name, data=np.array(self._grid[i]))
                    
                    # Set axis attributes
                    axis_dataset.attrs.create("NAME", [np.bytes_(axis_shortnames[i].encode())])
                    axis_dataset.attrs.create("UNITS", [np.bytes_(axis_units[i].encode())])
                    axis_dataset.attrs.create("LONG_NAME", [np.bytes_(axis_longnames[i].encode())])
                    axis_dataset.attrs.create("TYPE", [np.bytes_("linear".encode())])
                
                if verbose:
                    print(f"File created: {filename}")

        print(f"The savename of the diagnostic is {savename}. Files will be saves as {savename}-000001.h5, {savename}-000002.h5, etc.")
        
        print(f"If you desire a different name, please set it with the 'name' method (setter).")

        if self._name is None:
            raise ValueError("Diagnostic name is not set. Cannot save to HDF5.")
        if not os.path.exists(folder):
            print(f"Creating folder {folder}...")
            os.makedirs(folder)
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a directory.")

        if all == False:
            if isinstance(index, int):
                filename = folder + f"/{savename}-{index:06d}.h5"
                savefile(filename, index)
            elif isinstance(index, list) or isinstance(index, tuple):
                for i in index:
                    filename = folder + f"/{savename}-{i:06d}.h5"
                    savefile(filename, i)
        elif all == True:
            for i in range(self._maxiter):
                filename = folder + f"/{savename}-{i:06d}.h5"
                savefile(filename, i)
        else:
            raise ValueError("index should be an int, slice, or list of ints, or all should be True")
                    
    def plot_3d(
        self,
        idx,
        scale_type: Literal["zero_centered", "pos", "neg", "default"] = "default",
        boundaries: np.ndarray = None,
    ):
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
            try:
                boundaries = np.array(boundaries)
            except:
                boundaries = self._grid
                warnings.warn(
                    "boundaries cannot be accessed as a numpy array with shape (3, 2), using default instead"
                )

        if boundaries.shape != (3, 2):
            warnings.warn("boundaries should have shape (3, 2), using default instead")
            boundaries = self._grid

        # Load data
        if self._all_loaded:
            data = self._data[idx]
        else:
            data = self[idx]

        X, Y, Z = np.meshgrid(self._x[0], self._x[1], self._x[2], indexing="ij")

        # Flatten arrays for scatter plot
        (
            X_flat,
            Y_flat,
            Z_flat,
        ) = (
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        )
        data_flat = data.ravel()

        # Apply filter: Keep only chosen points
        mask = (
            (X_flat > boundaries[0][0])
            & (X_flat < boundaries[0][1])
            & (Y_flat > boundaries[1][0])
            & (Y_flat < boundaries[1][1])
            & (Z_flat > boundaries[2][0])
            & (Z_flat < boundaries[2][1])
        )
        X_cut, Y_cut, Z_cut, data_cut = (
            X_flat[mask],
            Y_flat[mask],
            Z_flat[mask],
            data_flat[mask],
        )

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
        ax.set_xlim(*self._grid[0])
        ax.set_ylim(*self._grid[1])
        ax.set_zlim(*self._grid[2])

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

        # Labels
        # TODO try to use a latex label instaead of _name
        cbar.set_label(r"${}$".format(self._name) + r"$\  [{}]$".format(self._units))
        ax.set_title(
            r"$t={:.2f}$".format(self.time(idx)[0])
            + r"$\  [{}]$".format(self.time(idx)[1])
        )
        ax.set_xlabel(
            r"${}$".format(self.axis[0]["long_name"])
            + r"$\  [{}]$".format(self.axis[0]["units"])
        )
        ax.set_ylabel(
            r"${}$".format(self.axis[1]["long_name"])
            + r"$\  [{}]$".format(self.axis[1]["units"])
        )
        ax.set_zlabel(
            r"${}$".format(self.axis[2]["long_name"])
            + r"$\  [{}]$".format(self.axis[2]["units"])
        )

        return fig, ax

    # Getters
    @property
    def data(self):
        if self._data is None:
            raise ValueError(
                "Data not loaded into memory. Use get_* method with load_all=True or access via generator/index."
            )
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
    def tunits(self):
        return self._tunits

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
    
    # @property
    # def iter(self):
    #     return self._iter

    @property
    def all_loaded(self):
        return self._all_loaded

    @property
    def maxiter(self):
        return self._maxiter

    @property
    def label(self):
        return self._label
    
    @property
    def type(self):
        return self._type

    @property
    def quantity(self):
        return self._quantity

    def time(self, index):
        return [index * self._dt * self._ndump, self._tunits]

    def attributes_to_save(self, index):
        """
        Prints the attributes of the diagnostic.
        """
        print(f"dt: {self._dt}\n"
              f"dim: {self._dim}\n"
              f"time: {self.time(index)[0]}\n"
              f"tunits: {self.time(index)[1]}\n"
              f"iter: {self._ndump * index}\n"
              f"name: {self._name}\n"
              f"type: {self._type}\n"
              f"label: {self._label}\n"
              f"units: {self._units}")

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

    @tunits.setter
    def tunits(self, value):
        self._tunits = value

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

    @quantity.setter
    def quantity(self, key):
        self._quantity = key

    @label.setter
    def label(self, value):
        self._label = value
