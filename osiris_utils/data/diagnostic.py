from __future__ import annotations

"""
The utilities on data.py are cool but not useful when you want to work with whole data of a simulation instead
of just a single file. This is what this file is for - deal with ''folders'' of data.

Took some inspiration from Diogo and Madox's work.

This would be awsome to compute time derivatives.
"""

import glob
import logging
import operator
import os
import warnings
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from .data import OsirisGridFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)8s │ %(message)s")
logger = logging.getLogger(__name__)

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
    "part_e3",
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
OSIRIS_ALL = OSIRIS_DENSITY + OSIRIS_SPECIE_REPORTS + OSIRIS_SPECIE_REP_UDIST + OSIRIS_FLD + OSIRIS_PHA

_ATTRS_TO_CLONE = [
    "_dx",
    "_nx",
    "_x",
    "_dt",
    "_grid",
    "_axis",
    "_units",
    "_name",
    "_label",
    "_dim",
    "_ndump",
    "_maxiter",
    "_tunits",
    "_type",
    "_simulation_folder",
    "_quantity",
]


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

    def __init__(self, simulation_folder: Optional[str] = None, species: Any = None, input_deck: Optional[str | None] = None) -> None:
        self._species = species if species else None

        self._dx: Optional[Union[float, np.ndarray]] = None  # grid spacing in each direction
        self._nx: Optional[Union[int, np.ndarray]] = None  # number of grid points in each direction
        self._x: Optional[np.ndarray] = None  # grid points
        self._dt: Optional[float] = None  # time step
        self._grid: Optional[np.ndarray] = None  # grid boundaries
        self._axis: Optional[Any] = None  # axis information
        self._units: Optional[str] = None  # units of the diagnostic
        self._name: Optional[str] = None
        self._label: Optional[str] = None
        self._dim: Optional[int] = None
        self._ndump: Optional[int] = None
        self._maxiter: Optional[int] = None
        self._tunits: Optional[str] = None  # time units

        if simulation_folder:
            self._simulation_folder = simulation_folder
            if not os.path.isdir(simulation_folder):
                raise FileNotFoundError(f"Simulation folder {simulation_folder} not found.")
        else:
            self._simulation_folder = None
        # load input deck if available
        if input_deck:
            self._input_deck = input_deck
        else:
            self._input_deck = None

        self._all_loaded: bool = False  # if the data is already loaded into memory
        self._quantity: Optional[str] = None

    #########################################
    #
    # Diagnostic metadata and attributes
    #
    #########################################

    def get_quantity(self, quantity: str) -> None:
        """
        Get the data for a given quantity.

        Parameters
        ----------
        quantity : str
            The quantity to get the data.
        """
        self._quantity = quantity

        if self._quantity not in OSIRIS_ALL:
            raise ValueError(f"Invalid quantity {self._quantity}. Use which_quantities() to see the available quantities.")
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

    def _scan_files(self, pattern: str) -> None:
        """Populate _file_list and related attributes from a glob pattern."""
        self._file_list = sorted(glob.glob(pattern))
        if not self._file_list:
            raise FileNotFoundError(f"No HDF5 files match {pattern}")
        self._file_template = self._file_list[0][:-9]  # keep old “template” idea
        self._maxiter = len(self._file_list)

    def _get_moment(self, species: str, moment: str) -> None:
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_field(self, field: str) -> None:
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_density(self, species: str, quantity: str) -> None:
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_phase_space(self, species: str, type: str) -> None:
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/PHA/{type}/{species}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _load_attributes(self, file_template: str, input_deck: Optional[dict]) -> None:  # this will be replaced by reading the input deck
        # This can go wrong! NDUMP
        # if input_deck is not None:
        #     self._dt = float(input_deck["time_step"][0]["dt"])
        #     self._nx = np.array(list(map(int, input_deck["grid"][0][f"nx_p(1:{self._dim})"].split(','))))
        #     xmin = [deval(input_deck["space"][0][f"xmin(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     xmax = [deval(input_deck["space"][0][f"xmax(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     self._grid = np.array([[xmin[i], xmax[i]] for i in range(self._dim)])
        #     self._dx = (self._grid[:,1] - self._grid[:,0])/self._nx
        #     self._x = [np.arange(self._grid[i,0], self._grid[i,1], self._dx[i]) for i in range(self._dim)]

        if input_deck is not None:
            self._ndump = int(input_deck["time_step"][0]["ndump"])
        elif input_deck is None:
            self._ndump = 1

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

    ##########################################
    #
    # Data loading and processing
    #
    ##########################################

    def _data_generator(self, index: int) -> None:
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set.")
        if self._file_list is None:
            raise RuntimeError("File list not initialized. Call get_quantity() first.")
        try:
            file = self._file_list[index]
        except IndexError:
            raise RuntimeError(f"File index {index} out of range (max {self._maxiter - 1}).")
        data_object = OsirisGridFile(file)
        yield (data_object.data if self._quantity not in OSIRIS_DENSITY else np.sign(self._species.rqm) * data_object.data)

    def load_all(self) -> np.ndarray:
        """
        Load all data into memory (all iterations), in a pre-allocated array.

        Returns
        -------
        data : np.ndarray
            The data for all iterations. Also stored in self._data.
        """
        if getattr(self, "_all_loaded", False) and self._data is not None:
            logger.debug("Data already loaded into memory.")
            return self._data

        size = getattr(self, "_maxiter", None)
        if size is None:
            raise RuntimeError("Cannot determine iteration count (no _maxiter).")

        try:
            first = self[0]
        except Exception as e:
            raise RuntimeError(f"Failed to load first timestep: {e}")
        slice_shape = first.shape
        dtype = first.dtype

        data = np.empty((size, *slice_shape), dtype=dtype)
        data[0] = first

        for i in tqdm.trange(1, size, desc="Loading data"):
            try:
                data[i] = self[i]
            except Exception as e:
                raise RuntimeError(f"Error loading timestep {i}: {e}")

        self._data = data
        self._all_loaded = True
        return self._data

    def unload(self) -> None:
        """
        Unload data from memory. This is useful to free memory when the data is not needed anymore.
        """
        logger.info("Unloading data from memory.")
        if self._all_loaded is False:
            logger.warning("Data is not loaded.")
            return
        self._data = None
        self._all_loaded = False

    ###########################################
    #
    # Data access and iteration
    #
    ###########################################

    def __len__(self) -> int:
        """Return the number of timesteps available."""
        return getattr(self, "_maxiter", 0)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Retrieve timestep data.

        Parameters
        ----------
        index : int or slice
            - If int, may be negative (Python-style).
            - If slice, supports start:stop:step. Zero-length slices return an empty array of shape (0, ...).

        Returns
        -------
        np.ndarray
            Array for that timestep (or stacked array for a slice).

        Raises
        ------
        IndexError
            If the index is out of range or no data generator is available.
        RuntimeError
            If loading a specific timestep fails.
        """
        # Quick path: all data already in memory
        if getattr(self, "_all_loaded", False) and self._data is not None:
            return self._data[index]

        # Data generator must exist
        data_gen = getattr(self, "_data_generator", None)
        if not callable(data_gen):
            raise IndexError(f"No data available for indexing; you did something wrong!")

        # Handle int indices (including negatives)
        if isinstance(index, int):
            if index < 0:
                index += self._maxiter
            if not (0 <= index < self._maxiter):
                raise IndexError(f"Index {index} out of range (0..{self._maxiter - 1})")
            try:
                gen = data_gen(index)
                return next(gen)
            except Exception as e:
                raise RuntimeError(f"Error loading data at index {index}: {e}")

        # Handle slice indices
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop if index.stop is not None else self._maxiter
            step = index.step or 1
            # Normalize negatives
            if start < 0:
                start += self._maxiter
            if stop < 0:
                stop += self._maxiter
            # Clip to bounds
            start = max(0, min(start, self._maxiter))
            stop = max(0, min(stop, self._maxiter))
            indices = range(start, stop, step)

            # Empty slice
            if not indices:
                # Determine single-step shape by peeking at index 0 (if possible)
                try:
                    dummy = next(data_gen(0))
                    empty_shape = (0,) + dummy.shape
                    return np.empty(empty_shape, dtype=dummy.dtype)
                except Exception:
                    return np.empty((0,))

            # Collect and stack
            data_list = []
            for i in indices:
                try:
                    data_list.append(next(data_gen(i)))
                except Exception as e:
                    raise RuntimeError(f"Error loading slice at index {i}: {e}")
            return np.stack(data_list)

        # Unsupported index type
        raise IndexError(f"Invalid index type {type(index)}; must be int or slice")

    def __iter__(self) -> Iterator[np.ndarray]:
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
                    logger.warning(f"Could not determine iteration count for iteration, using {max_iter}.")

            for i in range(max_iter):
                yield next(self._data_generator(i))

        # If we don't know how to handle this
        else:
            raise ValueError("Cannot iterate over this diagnostic. No data loaded and no generator available.")

    def _clone_meta(self) -> Diagnostic:
        """
        Create a new Diagnostic instance that carries over metadata only.
        No data is copied, and no constructor edits are required because we
        assign attributes dynamically.
        """
        clone = Diagnostic(species=getattr(self, "_species", None))  # keep species link
        for attr in _ATTRS_TO_CLONE:
            if hasattr(self, attr):
                setattr(clone, attr, getattr(self, attr))
        # If this diagnostic already discovered a _file_list via _scan_files,
        # copy it too (harmless for virtual diags).
        if hasattr(self, "_file_list"):
            clone._file_list = self._file_list
        return clone

    def _binary_op(self, other: Union["Diagnostic", int, float, np.ndarray], op_func: Callable) -> Diagnostic:
        """
        Universal helper for `self (op) other`.
        - If both operands are fully loaded, does eager numpy arithmetic.
        - Otherwise builds a lazy generator that applies op_func on each timestep.
        """
        # 1) Prepare the metadata clone
        result = self._clone_meta()
        result.created_diagnostic_name = "MISC"

        # 2) Determine iteration count
        if isinstance(other, Diagnostic):
            result._maxiter = min(self._maxiter, other._maxiter)
        else:
            result._maxiter = self._maxiter

        # 3) Eager path: both in RAM (or scalar/ndarray + self in RAM)
        self_loaded = getattr(self, "_all_loaded", False)
        other_loaded = isinstance(other, Diagnostic) and getattr(other, "_all_loaded", False)
        if self_loaded and (other_loaded or not isinstance(other, Diagnostic)):
            lhs = self._data
            rhs = other._data if other_loaded else other
            result._data = op_func(lhs, rhs)
            result._all_loaded = True
            return result

        def _wrap(arr):
            return lambda idx: (arr[idx],)

        gen1 = _wrap(self._data) if self_loaded else self._data_generator
        if isinstance(other, Diagnostic):
            gen2 = _wrap(other._data) if other_loaded else other._data_generator
        else:
            gen2 = other  # scalar or ndarray

        def _make_gen(idx):
            seq1 = gen1(idx)
            if callable(gen2):
                seq2 = gen2(idx)
                return (op_func(a, b) for a, b in zip(seq1, seq2))
            else:
                return (op_func(a, gen2) for a in seq1)

        result._data_generator = _make_gen
        result._all_loaded = False
        return result

    # Now define each operator in one line:

    def __add__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self + other

    def __sub__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        # swap args for reversed subtraction
        return self._binary_op(other, lambda x, y: operator.sub(y, x))

    def __mul__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self * other

    def __truediv__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        return self._binary_op(other, lambda x, y: operator.truediv(y, x))

    def __neg__(self) -> Diagnostic:
        # unary minus as multiplication by -1
        return self._binary_op(-1, operator.mul)

    def __pow__(self, other: Union["Diagnostic", int, float, np.ndarray]) -> Diagnostic:
        """
        Power operation. Raises the diagnostic data to the power of `other`.
        If `other` is a Diagnostic, it raises each timestep's data to the corresponding timestep's power.
        If `other` is a scalar or ndarray, it raises all data to that power.
        """
        return self._binary_op(other, operator.pow)

    def to_h5(
        self,
        savename: Optional[str] = None,
        index: Optional[Union[int, List[int]]] = None,
        all: bool = False,
        verbose: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """
        Save the diagnostic data to HDF5 files.

        Parameters
        ----------
        savename : str, optional
            The name of the HDF5 file. If None, uses the diagnostic name.
        index : int, or list of ints, optional
            The index or indices of the data to save.
        all : bool, optional
            If True, save all data. Default is False.
        verbose : bool, optional
            If True, print messages about the saving process.
        path : str, optional
            The path to save the HDF5 files. If None, uses the default save path (in simulation folder).
        """
        if path is None:
            path = self._simulation_folder
            self._save_path = path + f"/MS/MISC/{self._default_save}/{savename}"
        else:
            self._save_path = path
        # Check if is has attribute created_diagnostic_name or postprocess_name
        if savename is None:
            logger.warning(f"No savename provided. Using {self._name}.")
            savename = self._name

        if hasattr(self, "created_diagnostic_name"):
            self._default_save = self.created_diagnostic_name
        elif hasattr(self, "postprocess_name"):
            self._default_save = self.postprocess_name
        else:
            self._default_save = "DIR_" + self._name

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
            if verbose:
                logger.info(f"Created folder {self._save_path}")

        if verbose:
            logger.info(f"Save Path: {self._save_path}")

        def savefile(filename, i):
            with h5py.File(filename, "w") as f:
                # Create SIMULATION group with attributes
                sim_group = f.create_group("SIMULATION")
                sim_group.attrs.create("DT", [self._dt])
                sim_group.attrs.create("NDIMS", [self._dim])

                # Set file attributes
                f.attrs.create("TIME", [self.time(i)[0]])
                f.attrs.create(
                    "TIME UNITS",
                    [(np.bytes_(self.time(i)[1].encode()) if self.time(i)[1] else np.bytes_(b""))],
                )
                f.attrs.create("ITER", [self._ndump * i])
                f.attrs.create("NAME", [np.bytes_(self._name.encode())])
                f.attrs.create("TYPE", [np.bytes_(self._type.encode())])
                f.attrs.create(
                    "UNITS",
                    [(np.bytes_(self._units.encode()) if self._units else np.bytes_(b""))],
                )
                f.attrs.create(
                    "LABEL",
                    [(np.bytes_(self._label.encode()) if self._label else np.bytes_(b""))],
                )

                # Create dataset with data (transposed to match convention)
                f.create_dataset(savename, data=self[i].T)

                # Create AXIS group
                axis_group = f.create_group("AXIS")

                # Create axis datasets
                axis_names = ["AXIS1", "AXIS2", "AXIS3"][: self._dim]
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
                    logger.info(f"File created: {filename}")

        logger.info(
            f"The savename of the diagnostic is {savename}. Files will be saves as {savename}-000001.h5, {savename}-000002.h5, etc."
        )

        logger.info("If you desire a different name, please set it with the 'name' method (setter).")

        if self._name is None:
            raise ValueError("Diagnostic name is not set. Cannot save to HDF5.")
        if not os.path.exists(path):
            logger.info(f"Creating folder {path}...")
            os.makedirs(path)
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory.")

        if all is False:
            if isinstance(index, int):
                filename = self._save_path + f"/{savename}-{index:06d}.h5"
                savefile(filename, index)
            elif isinstance(index, list) or isinstance(index, tuple):
                for i in index:
                    filename = self._save_path + f"/{savename}-{i:06d}.h5"
                    savefile(filename, i)
        elif all is True:
            for i in range(self._maxiter):
                filename = self._save_path + f"/{savename}-{i:06d}.h5"
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
        *****************************************************************************************************
        THIS SHOULD BE REMOVED FROM THE BASE CLASS AND MOVED TO A SEPARATED CLASS DESIGNATED FOR THIS PURPOSE
        *****************************************************************************************************

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
            except Exception:
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
        ax.set_title(r"$t={:.2f}$".format(self.time(idx)[0]) + r"$\  [{}]$".format(self.time(idx)[1]))
        ax.set_xlabel(r"${}$".format(self.axis[0]["long_name"]) + r"$\  [{}]$".format(self.axis[0]["units"]))
        ax.set_ylabel(r"${}$".format(self.axis[1]["long_name"]) + r"$\  [{}]$".format(self.axis[1]["units"]))
        ax.set_zlabel(r"${}$".format(self.axis[2]["long_name"]) + r"$\  [{}]$".format(self.axis[2]["units"]))

        return fig, ax

    def __str__(self):
        """String representation of the diagnostic."""
        return f"Diagnostic: {self._name}, Species: {self._species}, Quantity: {self._quantity}"

    def __repr__(self):
        """Detailed string representation of the diagnostic."""
        return (
            f"Diagnostic(species={self._species}, name={self._name}, quantity={self._quantity}, "
            f"dim={self._dim}, maxiter={self._maxiter}, all_loaded={self._all_loaded})"
        )

    # Getters
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise ValueError("Data not loaded into memory. Use get_* method with load_all=True or access via generator/index.")
        return self._data

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def nx(self) -> int | np.ndarray:
        return self._nx

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def axis(self) -> list[dict]:
        return self._axis

    @property
    def units(self) -> str:
        return self._units

    @property
    def tunits(self) -> str:
        return self._tunits

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def path(self) -> str:
        return self._path

    @property
    def simulation_folder(self) -> str:
        return self._simulation_folder

    @property
    def ndump(self) -> int:
        return self._ndump

    @property
    def all_loaded(self) -> bool:
        return self._all_loaded

    @property
    def maxiter(self) -> int:
        return self._maxiter

    @property
    def label(self) -> str:
        return self._label

    @property
    def type(self) -> str:
        return self._type

    @property
    def quantity(self) -> str:
        return self._quantity

    @property
    def file_list(self) -> list[str] | None:
        """Return the cached list of HDF5 file paths (read-only)."""
        return self._file_list

    def time(self, index) -> list[float | str]:
        return [index * self._dt * self._ndump, self._tunits]

    def attributes_to_save(self, index: int = 0) -> None:
        """
        Prints the attributes of the diagnostic.
        """
        logger.info(
            f"dt: {self._dt}\n"
            f"dim: {self._dim}\n"
            f"time: {self.time(index)[0]}\n"
            f"tunits: {self.time(index)[1]}\n"
            f"iter: {self._ndump * index}\n"
            f"name: {self._name}\n"
            f"type: {self._type}\n"
            f"label: {self._label}\n"
            f"units: {self._units}"
        )

    @dx.setter
    def dx(self, value: float) -> None:
        self._dx = value

    @nx.setter
    def nx(self, value: int | np.ndarray) -> None:
        self._nx = value

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._x = value

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = value

    @grid.setter
    def grid(self, value: np.ndarray) -> None:
        self._grid = value

    @axis.setter
    def axis(self, value: list[dict]) -> None:
        self._axis = value

    @units.setter
    def units(self, value: str) -> None:
        self._units = value

    @tunits.setter
    def tunits(self, value: str) -> None:
        self._tunits = value

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    @ndump.setter
    def ndump(self, value: int) -> None:
        self._ndump = value

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value

    @quantity.setter
    def quantity(self, key: str) -> None:
        self._quantity = key

    @label.setter
    def label(self, value: str) -> None:
        self._label = value
