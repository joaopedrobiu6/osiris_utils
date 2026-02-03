from __future__ import annotations

import glob
import logging
import operator
import os
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Literal

import h5py
import numpy as np
import tqdm

from ..decks.decks import InputDeckIO
from ..decks.species import Species
from .data import OsirisGridFile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)8s │ %(message)s")
logger = logging.getLogger(__name__)

"""
The utilities on data.py are cool but not useful when you want to work with whole data of a simulation instead
of just a single file. This is what this file is for - deal with ''folders'' of data.

Took some inspiration from Diogo and Madox's work.

This would be awsome to compute time derivatives.
"""

OSIRIS_DENSITY = ["n"]
OSIRIS_SPECIE_REPORTS = ["ene", "charge", "q1", "q2", "q3", "j1", "j2", "j3"]
OSIRIS_SPECIE_REP_UDIST = [
    "vfl1",
    "vfl2",
    "vfl3",
    "ufl1",
    "ufl2",
    "ufl3",
    "P00",
    "P01",
    "P02",
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

__all__ = ["Diagnostic", "which_quantities"]


def which_quantities():
    print("Available quantities:")
    print(OSIRIS_ALL)


class Diagnostic:
    """Class to handle diagnostics. This is the "base" class of the code. Diagnostics can be loaded from OSIRIS output files,
    but are also created when performing operations with other diagnostics.
    Post-processed quantities are also considered diagnostics. This way, we can perform operations with them as well.

    Parameters
    ----------
    species : str
        The species to handle the diagnostics.
    simulation_folder : str
        The path to the simulation folder. This is the path to the folder where the input deck is located.
    input_deck : str or dict, optional
        The input deck to load the diagnostic attributes. If None, the attributes are loaded from the files.
        If a string is provided, it is assumed to be the path to the input deck file.
        If a dict is provided, it is assumed to be the parsed input deck.

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
        The axis information. Each key is a direction and the value is a dictionary with the keys "name",
        "long_name", "units" and "plot_label".
    units : str
        The units of the diagnostic. This info may not be available for all diagnostics, ie,
        diagnostics resulting from operations and postprocessing.
    name : str
        The name of the diagnostic. This info may not be available for all diagnostics, ie,
        diagnostics resulting from operations and postprocessing.
    label : str
        The label of the diagnostic. This info may not be available for all diagnostics, ie,
        diagnostics resulting from operations and postprocessing.
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

    def __init__(
        self,
        simulation_folder: str | None = None,
        species: Species = None,
        input_deck: InputDeckIO | None = None,
    ) -> None:
        self._species = species if species else None

        self._dx: float | np.ndarray | None = None  # grid spacing in each direction
        self._nx: int | np.ndarray | None = None  # number of grid points in each direction
        self._x: np.ndarray | None = None  # grid points
        self._dt: float | None = None  # time step
        self._grid: np.ndarray | None = None  # grid boundaries
        self._axis: Any | None = None  # axis information
        self._units: str | None = None  # units of the diagnostic
        self._name: str | None = None
        self._label: str | None = None
        self._dim: int | None = None
        self._ndump: int | None = None
        self._maxiter: int | None = None
        self._tunits: str | None = None  # time units

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
        self._quantity: str | None = None

    #########################################
    #
    # Diagnostic metadata and attributes
    #
    #########################################

    def get_quantity(self, quantity: str) -> None:
        """Get the data for a given quantity.

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
                f"Invalid quantity {self._quantity}. Or it's not implemented yet (this may happen for phase space quantities).",
            )

    def _scan_files(self, pattern: str) -> None:
        """Populate _file_list and related attributes from a glob pattern.

        Parameters
        ----------
        pattern : str
            The glob pattern to search for HDF5 files.

        """
        self._file_list = sorted(glob.glob(pattern))
        if not self._file_list:
            raise FileNotFoundError(f"No HDF5 files match {pattern}")
        self._file_template = self._file_list[0][:-9]  # keep old “template” idea
        self._maxiter = len(self._file_list)

    def _get_moment(self, species: str, moment: str) -> None:
        """Get the moment data for a given species and moment.

        Parameters
        ----------
        species : str
            The species to get the moment data.
        moment : str
            The moment to get the data.

        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_field(self, field: str) -> None:
        """Get the field data for a given field.

        Parameters
        ----------
        field : str
            The field to get the data.

        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_density(self, species: str, quantity: str) -> None:
        """Get the density data for a given species and quantity.

        Parameters
        ----------
        species : str
            The species to get the density data.
        quantity : str
            The quantity to get the data.

        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _get_phase_space(self, species: str, type: str) -> None:
        """Get the phase space data for a given species and type.

        Parameters
        ----------
        species : str
            The species to get the phase space data.
        type : str
            The type of phase space to get the data.

        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set. If you're using CustomDiagnostic, this method is not available.")
        self._path = f"{self._simulation_folder}/MS/PHA/{type}/{species}/"
        self._scan_files(os.path.join(self._path, "*.h5"))
        self._load_attributes(self._file_template, self._input_deck)

    def _load_attributes(self, file_template: str, input_deck: dict | None) -> None:  # this will be replaced by reading the input deck
        """Load diagnostic attributes from the first available file or input deck.

        Parameters
        ----------
        file_template : str
            The file template to load the attributes from.
            This is the path to the file without the iteration number and extension.
            (e.g., /path/to/diagnostic/000001.h5 -> /path/to/diagnostic/)
        input_deck : dict, optional
            The input deck to load the diagnostic attributes. If None, the attributes are loaded from the files.

        """
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
                    dump = OsirisGridFile(path_file, load_data=False)
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
                warnings.warn(f"No valid data files found in {self._path} to read metadata from.", stacklevel=2)
        except Exception as e:
            warnings.warn(f"Error loading diagnostic attributes: {e!s}. Please verify it there's any file in the folder.", stacklevel=2)

    ##########################################
    #
    # Data loading and processing
    #
    ##########################################

    def load_all(self, n_workers: int | None = None, use_parallel: bool | None = None) -> np.ndarray:
        """Load all data into memory (all iterations), in a pre-allocated array.

        Parameters
        ----------
        n_workers : int, optional
            Number of parallel workers for loading data. If None, uses CPU count.
            Only used if use_parallel=True.
        use_parallel : bool, optional
            If True, force parallel loading. If False, force sequential.
            If None (default), automatically choose based on data size.

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
            raise RuntimeError("Failed to load first timestep") from e
        slice_shape = first.shape
        dtype = first.dtype

        data = np.empty((size, *slice_shape), dtype=dtype)
        data[0] = first

        # Auto-detect whether to use parallel loading
        if use_parallel is None:
            # Use parallel for large datasets: >10 timesteps AND >1MB per file
            bytes_per_timestep = first.nbytes
            use_parallel = (size > 10) and (bytes_per_timestep > 1_000_000)

        # Parallel loading for significant performance improvement
        if use_parallel and size > 1:
            import concurrent.futures
            import os

            if n_workers is None:
                n_workers = min(os.cpu_count() or 4, size - 1)

            def load_single(i):
                """Helper to load a single timestep"""
                try:
                    return i, self[i]
                except Exception as e:
                    raise RuntimeError(f"Error loading timestep {i}") from e

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all loading tasks
                futures = {executor.submit(load_single, i): i for i in range(1, size)}

                # Collect results with progress bar
                with tqdm.tqdm(total=size - 1, desc="Loading data (parallel)") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        i, result = future.result()
                        data[i] = result
                        pbar.update(1)
        else:
            # Sequential loading (fallback or for small files)
            for i in tqdm.trange(1, size, desc="Loading data"):
                try:
                    data[i] = self[i]
                except Exception as e:
                    raise RuntimeError(f"Error loading timestep {i}") from e

        self._data = data
        self._all_loaded = True
        return self._data

    def unload(self) -> None:
        """Unload data from memory. This is useful to free memory when the data is not needed anymore."""
        logger.info("Unloading data from memory.")
        if self._all_loaded is False:
            logger.warning("Data is not loaded.")
            return
        self._data = None
        self._all_loaded = False

    def _data_generator(self, index: int, data_slice: tuple | None = None) -> Iterator[np.ndarray]:
        """Data generator for a given index or slice.

        Parameters
        ----------
        index : int
            The index of the file to load.
        data_slice : tuple, optional
            The slice to apply to the data. This is a tuple of slices, one for each dimension.

        Returns
        -------
        Iterator[np.ndarray]
            An iterator that yields the data for the given index or slice.

        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set.")
        if self._file_list is None:
            raise RuntimeError("File list not initialized. Call get_quantity() first.")
        try:
            file = self._file_list[index]  # try to get the file at the given index
        except IndexError as err:
            raise RuntimeError(
                f"File index {index} out of range (max {self._maxiter - 1}).",
            ) from err
        # Pass data_slice to OsirisGridFile - HDF5 will efficiently read only the requested slice from disk
        data_object = OsirisGridFile(file, data_slice=data_slice)  # This is were the data is actually being read from disk
        yield (data_object.data if self._quantity not in OSIRIS_DENSITY else np.sign(self._species.rqm) * data_object.data)

    def _read_index(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        """Read and return the array for a single time index (keeps lazy behavior).

        This helper centralizes the single-file read logic so callers avoid generator
        overhead while keeping the operation lazy (we only read requested files).
        """
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set.")
        if self._file_list is None:
            raise RuntimeError("File list not initialized. Call get_quantity() first.")
        try:
            file = self._file_list[index]
        except IndexError as err:
            raise RuntimeError(
                f"File index {index} out of range (max {self._maxiter - 1}).",
            ) from err

        data_object = OsirisGridFile(file, data_slice=data_slice)
        return data_object.data if self._quantity not in OSIRIS_DENSITY else np.sign(self._species.rqm) * data_object.data

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        """Return one timestep (lazy). Overridden by derived diagnostics."""
        return self._read_index(index, data_slice=data_slice)

    ###########################################
    #
    # Data access and iteration
    #
    ###########################################

    def __len__(self) -> int:
        """Return the number of timesteps available."""
        return getattr(self, "_maxiter", 0)

    def __getitem__(self, index: int | slice | tuple) -> np.ndarray:
        """Retrieve timestep data with optional spatial slicing.

        Parameters
        ----------
        index : int, slice, or tuple
            - If int, loads full spatial data for that timestep.
            - If slice, supports start:stop:step for time. Zero-length slices return an empty array of shape (0, ...).
            - If tuple, first element is time index/slice, remaining elements are spatial slices for each dimension.
              Example: diag[5, :, 100:200] loads timestep 5 with spatial slicing applied.
              **Efficient**: HDF5 reads only the requested spatial slice from disk, not the full array.

        Returns
        -------
        np.ndarray
            Array for that timestep (or stacked array for a slice), optionally spatially sliced.

        Raises
        ------
        IndexError
            If the index is out of range or no data generator is available.
        RuntimeError
            If loading a specific timestep fails.

        Examples
        --------
        >>> data = diag[5]              # Load full spatial data for timestep 5
        >>> data = diag[5, :, 100:200]  # Load timestep 5 with spatial slice (only reads slice from disk!)
        >>> data = diag[0:10]           # Load timesteps 0-9 (full spatial domain)
        >>> data = diag[0:10, :, 50:]   # Load timesteps 0-9 with spatial slice

        """
        # This part separates index into time_index (time) and data_slice (space)
        # In the case where index is a tuple, separate time index and spatial slices
        # e.g., diag[5, :, 100:200] -> time_index=5, data_slice=(slice(None), slice(100,200))
        data_slice = None
        if isinstance(index, tuple):
            if len(index) == 0:
                raise IndexError("Empty tuple index not supported")
            time_index = index[0]  # first element is time index/slice
            if len(index) > 1:
                data_slice = index[1:]  # remaining elements are spatial slices
        else:
            time_index = index  # index is just time index/slice

        # Quick path: all data already in memory
        if getattr(self, "_all_loaded", False) and self._data is not None:
            if data_slice is not None:
                # Apply spatial slicing to loaded data
                return self._data[(time_index,) + data_slice]
            return self._data[time_index]

        # Data generator must exist
        data_gen = getattr(self, "_data_generator", None)
        if not callable(data_gen):
            raise IndexError("No data available for indexing; you did something wrong!")

        # Handle int indices (including negatives)
        if isinstance(time_index, int):
            if time_index < 0:
                time_index += self._maxiter  # if negative, go from the end
            if not (0 <= time_index < self._maxiter):
                raise IndexError(f"Index {time_index} out of range (0..{self._maxiter - 1})")

            # Load data immediately with optional spatial slicing
            try:
                return self._frame(time_index, data_slice=data_slice)
            except Exception as e:
                raise RuntimeError(f"Error loading data at index {time_index}") from e

        # if time is a slice
        if isinstance(time_index, slice):
            # Use slice.indices to correctly handle negative steps and defaults
            start, stop, step = time_index.indices(getattr(self, "_maxiter", 0))
            indices = range(start, stop, step)

            # Empty slice in time: try to preserve spatial shape/dtype if possible
            if len(indices) == 0:
                try:
                    dummy = self._frame(0, data_slice=data_slice)
                    empty_shape = (0,) + dummy.shape
                    return np.empty(empty_shape, dtype=dummy.dtype)
                except Exception:
                    return np.empty((0,))

            # Preallocate output using first requested index to determine shape/dtype
            try:
                first_idx = indices[0]
                first_arr = self._frame(first_idx, data_slice=data_slice)
            except Exception as e:
                raise RuntimeError(f"Error loading timestep {first_idx}") from e

            n = len(indices)
            out = np.empty((n, *first_arr.shape), dtype=first_arr.dtype)

            # Fill all slots (including first) in order
            for pos, i in enumerate(indices):
                try:
                    out[pos] = self._frame(i, data_slice=data_slice)
                except Exception as e:
                    raise RuntimeError(f"Error loading timestep {i}") from e

            return out

        # If we don't know how to handle this
        raise ValueError("Cannot index this diagnostic. No data loaded and no generator available.")

    def _clone_meta(self) -> Diagnostic:
        """Create a new Diagnostic instance that carries over metadata only.
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

    def _binary_op(self, other: Diagnostic | float | np.ndarray, op_func: Callable) -> Diagnostic:
        """Universal helper for `self (op) other`.
        - If both operands are fully loaded, does eager numpy arithmetic.
        - Otherwise builds a lazy generator that applies op_func on each timestep.

        Parameters
        ----------
        other : Diagnostic, float, or np.ndarray
            The other operand.
        op_func : Callable
            The binary operation function (e.g., operator.add).
        """
        # nPrepare the metadata clone and set the name of the resulting diagnostic to "MISC"
        result = self._clone_meta()
        result.created_diagnostic_name = "MISC"

        # Determine iteration count - set as minimum of both diagnostics if both are Diagnostics
        if isinstance(other, Diagnostic):
            result._maxiter = min(self._maxiter, other._maxiter)
        else:
            result._maxiter = self._maxiter

        # SELF ALL_LOADED & OTHER ALL LOADED / SCALAR/NDARRAY
        self_loaded = getattr(self, "_all_loaded", False)  # check if self data is loaded
        other_loaded = isinstance(other, Diagnostic) and getattr(other, "_all_loaded", False)  # check is other data is loaded
        # if both are loaded, do eager operation
        if self_loaded and (other_loaded or not isinstance(other, Diagnostic)):
            lhs = self._data  # self data
            rhs = other._data if other_loaded else other  # data from the other Diagnostic if all_loaded, or scalar/ndarray
            result._data = op_func(lhs, rhs)  # compute the operation
            result._all_loaded = True  # new resulting Diagnostic is all loaded
            return result  # return the result as a new Diagnostic

        if isinstance(other, Diagnostic):

            def _frame(index: int, data_slice: tuple | None = None) -> np.ndarray:
                return op_func(self._frame(index, data_slice=data_slice), other._frame(index, data_slice=data_slice))

        else:

            def _frame(index: int, data_slice: tuple | None = None) -> np.ndarray:
                return op_func(self._frame(index, data_slice=data_slice), other)

        result._frame = _frame  # attach method dynamically
        result._all_loaded = False
        result._data = None
        return result

    # Now define each operator in one line:

    def __add__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self + other

    def __sub__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        # swap args for reversed subtraction
        return self._binary_op(other, lambda x, y: operator.sub(y, x))

    def __mul__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self * other

    def __truediv__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        return self._binary_op(other, lambda x, y: operator.truediv(y, x))

    def __neg__(self) -> Diagnostic:
        # unary minus as multiplication by -1
        return self._binary_op(-1, operator.mul)

    def __pow__(self, other: Diagnostic | float | np.ndarray) -> Diagnostic:
        """Power operation. Raises the diagnostic data to the power of `other`.
        If `other` is a Diagnostic, it raises each timestep's data to the corresponding timestep's power.
        If `other` is a scalar or ndarray, it raises all data to that power.
        """
        return self._binary_op(other, operator.pow)

    def to_h5(
        self,
        savename: str | None = None,
        index: int | list[int] | None = None,
        all: bool = False,
        verbose: bool = False,
        path: str | None = None,
    ) -> None:
        """Save the diagnostic data to HDF5 files.

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
                    axis_dataset.attrs.create("TYPE", [np.bytes_(b"linear")])

                if verbose:
                    logger.info(f"File created: {filename}")

        logger.info(f"The savename of the diagnostic is {savename}.")
        logger.info(f"Files will be saved as {savename}-000001.h5, {savename}-000002.h5, etc.")
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
        """**DEPRECATED**: Use `osiris_utils.vis.plot_3d` instead.

        Plots a 3D scatter plot of the diagnostic data (grid data).
        """
        _msg = (
            "Diagnostic.plot_3d is deprecated and will be removed in a future version."
            + "Please use osiris_utils.vis.plot_3d(diagnostic, idx, ...) instead."
        )
        warnings.warn(_msg, DeprecationWarning, stacklevel=2)
        from ..vis.plot3d import plot_3d

        return plot_3d(self, idx, scale_type, boundaries)

    def __str__(self):
        """String representation of the diagnostic."""
        return f"Diagnostic: {self._name}, Species: {self._species}, Quantity: {self._quantity}"

    def __repr__(self):
        """Detailed string representation of the diagnostic."""
        parts = [
            f"species={self._species}",
            f"name={self._name}",
            f"quantity={self._quantity}",
            f"dim={self._dim}",
            f"maxiter={self._maxiter}",
            f"all_loaded={self._all_loaded}",
        ]
        return f"Diagnostic({', '.join(parts)})"

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
    def x(self) -> np.ndarray | list[np.ndarray]:
        # Return the coordinate array(s).
        # For 1D data, return the single array directly to support plt.plot(diag.x, diag[0])
        # For >1D data, return the list/array of axes.
        if self._x is not None:
            # Check if it's a list/array of arrays
            try:
                if len(self._x) == 1 and (self._dim == 1 or self._dim is None):
                    return self._x[0]
            except (TypeError, IndexError):
                pass
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
        """Prints the attributes of the diagnostic."""
        logger.info(
            f"dt: {self._dt}\n"
            f"dim: {self._dim}\n"
            f"time: {self.time(index)[0]}\n"
            f"tunits: {self.time(index)[1]}\n"
            f"iter: {self._ndump * index}\n"
            f"name: {self._name}\n"
            f"type: {self._type}\n"
            f"label: {self._label}\n"
            f"units: {self._units}",
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
