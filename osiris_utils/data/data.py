from typing import Literal

import h5py
import numpy as np
import pandas as pd

from osiris_utils.utils import create_file_tags


class OsirisData:
    """
    Base class for handling OSIRIS simulation data files (HDF5 and HIST formats).

    This class provides common functionality for reading and managing basic attributes
    from OSIRIS output files. It serves as the parent class for specialized data handlers.

    Parameters
    ----------
    filename : str
        Path to the data file. Supported formats:
        - HDF5 files (.h5 extension)
        - HIST files (ending with _ene)

    Attributes
    ----------
    dt : float
        Time step of the simulation [simulation units]
    dim : int
        Number of dimensions in the simulation (1, 2, or 3)
    time : list[float, str]
        Current simulation time and units as [value, unit_string]
    iter : int
        Current iteration number
    name : str
        Name identifier of the data field
    type : str
        Type of data (e.g., 'grid', 'particles')
    verbose : bool
        Verbosity flag controlling diagnostic messages (default: False)
    """

    def __init__(self, filename):
        self._filename = str(filename)
        # self._file = None

        self._verbose = False

        if self._filename.endswith(".h5"):
            self._open_file_hdf5(self._filename)
            self._load_basic_attributes(self._file)
        elif self._filename.endswith("_ene"):
            self._open_hist_file(self._filename)
        else:
            raise ValueError("The file should be an HDF5 file with the extension .h5, or a HIST file ending with _ene.")

    def _load_basic_attributes(self, f: h5py.File) -> None:
        """Load common attributes from HDF5 file"""
        self._dt = float(f["SIMULATION"].attrs["DT"][0])
        self._dim = int(f["SIMULATION"].attrs["NDIMS"][0])
        self._time = [
            float(f.attrs["TIME"][0]),
            f.attrs["TIME UNITS"][0].decode("utf-8"),
        ]
        self._iter = int(f.attrs["ITER"][0])
        self._name = f.attrs["NAME"][0].decode("utf-8")
        self._type = f.attrs["TYPE"][0].decode("utf-8")

    def verbose(self, verbose: bool = True):
        """
        Set the verbosity of the class

        Parameters
        ----------
        verbose : bool, optional
            If True, the class will print messages, by default True when calling (False when not calling)
        """
        self._verbose = verbose

    def _open_file_hdf5(self, filename):
        """
        Open the OSIRIS output file. Usually an HDF5 file or txt.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        """
        if self._verbose:
            print(f"Opening file > {filename}")

        if filename.endswith(".h5"):
            self._file = h5py.File(filename, "r")
        else:
            raise ValueError("The file should be an HDF5 file with the extension .h5")

    def _open_hist_file(self, filename):
        self._df = pd.read_csv(filename, sep=r"\s+", comment="!", header=0, engine="python")

    def _close_file(self):
        """
        Close the HDF5 file.
        """
        if self._verbose:
            print("Closing file")
        if self._file:
            self._file.close()

    @property
    def dt(self):
        return self._dt

    @property
    def dim(self):
        return self._dim

    @property
    def time(self):
        return self._time

    @property
    def iter(self):
        return self._iter

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type


class OsirisGridFile(OsirisData):
    """
    Handles structured grid data from OSIRIS HDF5 simulations, including electromagnetic fields.

    Parameters
    ----------
    filename : str
        Path to OSIRIS HDF5 grid file (.h5 extension)

    Attributes
    ----------
    grid : np.ndarray
        Grid boundaries as ((x1_min, x1_max), (x2_min, x2_max), ...)
    nx : tuple
        Number of grid points per dimension (nx1, nx2, nx3)
    dx : np.ndarray
        Grid spacing per dimension (dx1, dx2, dx3)
    x : list[np.ndarray]
        Spatial coordinates arrays for each dimension
    axis : list[dict]
        Axis metadata with keys:
        - 'name': Axis identifier (e.g., 'x1')
        - 'units': Physical units (LaTeX formatted)
        - 'long_name': Descriptive name (LaTeX formatted)
        - 'type': Axis type (e.g., 'SPATIAL')
        - 'plot_label': Combined label for plotting
    data : np.ndarray
        Raw field data array (shape depends on simulation dimensions)
    units : str
        Field units (LaTeX formatted)
    label : str
        Field label/name (LaTeX formatted, e.g., r'$E_x$')
    FFTdata : np.ndarray
        Fourier-transformed data (available after calling FFT())
    """

    def __init__(self, filename):
        super().__init__(filename)

        variable_key = self._get_variable_key(self._file)

        self._units = self._file.attrs["UNITS"][0].decode("utf-8")
        self._label = self._file.attrs["LABEL"][0].decode("utf-8")
        self._FFTdata = None

        data = np.array(self._file[variable_key][:])

        axis = list(self._file["AXIS"].keys())
        if len(axis) == 1:
            self._grid = self._file["AXIS/" + axis[0]][()]
            self._nx = len(data)
            self._dx = (self.grid[1] - self.grid[0]) / self.nx
            self._x = np.arange(self.grid[0], self.grid[1], self.dx)
        else:
            grid = []
            for ax in axis:
                grid.append(self._file["AXIS/" + ax][()])
            self._grid = np.array(grid)
            self._nx = self._file[variable_key][()].transpose().shape
            self._dx = (self.grid[:, 1] - self.grid[:, 0]) / self.nx

            # There's an issue when the dimension is 3 and we want to plot a 2D phasespace. I believe this
            # is a problem for all cases where the dim != dim_of_phasespace
            self._x = [np.arange(self.grid[i, 0], self.grid[i, 1], self.dx[i]) for i in range(self.dim)]
            # self._x = [np.arange(self.grid[i, 0], self.grid[i, 1], self.dx[i]) for i in range(2)]

        self._axis = []
        for ax in axis:
            axis_data = {
                "name": self._file["AXIS/" + ax].attrs["NAME"][0].decode("utf-8"),
                "units": self._file["AXIS/" + ax].attrs["UNITS"][0].decode("utf-8"),
                "long_name": self._file["AXIS/" + ax].attrs["LONG_NAME"][0].decode("utf-8"),
                "type": self._file["AXIS/" + ax].attrs["TYPE"][0].decode("utf-8"),
                "plot_label": rf'${self._file["AXIS/" + ax].attrs["LONG_NAME"][0].decode("utf-8")}$ $[{self._file["AXIS/" + ax].attrs["UNITS"][0].decode("utf-8")}]$',
            }
            self._axis.append(axis_data)

        self._data = np.ascontiguousarray(data.T)

        self._close_file()

    def _load_basic_attributes(self, f: h5py.File) -> None:
        """Load common attributes from HDF5 file"""
        self._dt = float(f["SIMULATION"].attrs["DT"][0])
        self._dim = int(f["SIMULATION"].attrs["NDIMS"][0])
        self._time = [
            float(f.attrs["TIME"][0]),
            f.attrs["TIME UNITS"][0].decode("utf-8"),
        ]
        self._iter = int(f.attrs["ITER"][0])
        self._name = f.attrs["NAME"][0].decode("utf-8")
        self._type = f.attrs["TYPE"][0].decode("utf-8")

    def _get_variable_key(self, f: h5py.File) -> str:
        return next(k for k in f.keys() if k not in {"AXIS", "SIMULATION"})

    def _yeeToCellCenter1d(self, boundary):
        """
        Converts 1d EM fields from a staggered Yee mesh to a grid with field values centered on the Center of the cell
        """

        if self.name.lower() in ["b2", "b3", "e1"]:
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1) + self.data)
            else:
                return 0.5 * (self.data[1:] + self.data[:-1])
        elif self.name.lower() in ["b1", "e2", "e3"]:
            if boundary == "periodic":
                return self.data
            else:
                return self.data[1:]
        else:
            raise TypeError(f"This method expects magnetic or electric field grid data but received '{self.name}' instead")

    def _yeeToCellCenter2d(self, boundary):
        """
        Converts 2d EM fields from a staggered Yee mesh to a grid with field values centered on the Center of the cell
        """

        if self.name.lower() in ["e1", "b2"]:
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
            else:
                return 0.5 * (self.data[1:, 1:] + self.data[:-1, 1:])
        elif self.name.lower() in ["e2", "b1"]:
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1, axis=1) + self.data)
            else:
                return 0.5 * (self.data[1:, 1:] + self.data[1:, :-1])
        elif self.name.lower() in ["b3"]:
            if boundary == "periodic":
                return 0.5 * (
                    np.roll(
                        (0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)),
                        shift=1,
                        axis=1,
                    )
                    + (0.5 * (np.roll(self.data, shift=1, axis=0) + self.data))
                )
            else:
                return 0.25 * (self.data[1:, 1:] + self.data[:-1, 1:] + self.data[1:, :-1] + self.data[:-1, :-1])
        elif self.name.lower() in ["e3"]:
            if boundary == "periodic":
                return self.data
            else:
                return self.data[1:, 1:]
        else:
            raise TypeError(f"This method expects magnetic or electric field grid data but received '{self.name}' instead")

    def _yeeToCellCenter3d(self, boundary):
        """
        Converts 3d EM fields from a staggered Yee mesh to a grid with field values centered on the Center of the cell
        """
        if self.name.lower() == "b1":
            if boundary == "periodic":
                return 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self.data, shift=1, axis=1) + self.data),
                        shift=1,
                        axis=2,
                    )
                    + 0.5 * (np.roll(self.data, shift=1, axis=1) + self.data)
                )
            else:
                return 0.25 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[1:, 1:, :-1] + self.data[1:, :-1, :-1])
        elif self.name.lower() == "b2":
            if boundary == "periodic":
                return 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self.data, shift=1, axis=0) + self.data),
                        shift=1,
                        axis=2,
                    )
                    + 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
                )
            else:
                return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, 1:, :-1] + self.data[:-1, 1:, :-1])
        elif self.name.lower() == "b3":
            if boundary == "periodic":
                return 0.5 * (
                    0.5
                    * np.roll(
                        (np.roll(self.data, shift=1, axis=0) + self.data),
                        shift=1,
                        axis=1,
                    )
                    + 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
                )
            else:
                return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[:-1, :-1, 1:])
        elif self.name.lower() == "e1":
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
            else:
                return 0.5 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:])
        elif self.name.lower() == "e2":
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1, axis=1) + self.data)
            else:
                return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:])
        elif self.name.lower() == "e3":
            if boundary == "periodic":
                return 0.5 * (np.roll(self.data, shift=1, axis=2) + self.data)
            else:
                return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, 1:, :-1])
        else:
            raise TypeError(f"This method expects magnetic or electric field grid data but received '{self.name}' instead")

    def yeeToCellCenter(self, boundary: Literal["periodic", "default"] = "default"):
        """'
        Converts EM fields from a staggered Yee mesh to a grid with field values centered on the center of the cell.'
        Can be used for 1D, 2D and 3D simulations.'
        Creates a new attribute `data_centered` with the centered data.'
        """

        if boundary not in ("periodic", "default"):
            raise ValueError(f"Invalid boundary: {boundary}, choose 'periodic' or 'default' instead.")

        cases = {"b1", "b2", "b3", "e1", "e2", "e3"}
        if self.name not in cases:
            raise TypeError(f"This method expects magnetic or electric field grid data but received '{self.name}' instead")

        if self.dim == 1:
            self.data_centered = self._yeeToCellCenter1d(boundary)
            return self.data_centered
        elif self.dim == 2:
            self.data_centered = self._yeeToCellCenter2d(boundary)
            return self.data_centered
        elif self.dim == 3:
            self.data_centered = self._yeeToCellCenter3d(boundary)
            return self.data_centered
        else:
            raise ValueError(f"Dimension {self.dim} is not supported")

    def FFT(self, axis=(0,)):
        """
        Computes the Fast Fourier Transform of the data along the specified axis and shifts the zero frequency to the center.
        Transforms the data to the frequency domain. A(x, y, z) -> A(kx, ky, kz)
        """
        datafft = np.fft.fftn(self.data, axes=axis)
        self._FFTdata = np.fft.fftshift(datafft, axes=axis)

    # Getters
    @property
    def grid(self):
        return self._grid

    @property
    def nx(self):
        return self._nx

    @property
    def dx(self):
        return self._dx

    @property
    def x(self):
        return self._x

    @property
    def axis(self):
        return self._axis

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units

    @property
    def label(self):
        return self._label

    @property
    def FFTdata(self):
        if self._FFTdata is None:
            raise ValueError("The FFT of the data has not been computed yet. Compute it using the FFT method.")
        return self._FFTdata

    # Setters
    @data.setter
    def data(self, data):
        self._data = data

    def __str__(self):
        # write me a template to print with the name, label, units, time, iter, grid, nx, dx, axis, dt, dim in a logical way
        return (
            rf"{self.name}"
            + "\n"
            + rf"Time: [{self.time[0]} {self.time[1]}], dt = {self.dt}"
            + "\n"
            + f"Iteration: {self.iter}"
            + "\n"
            + f"Grid: {self.grid}"
            + "\n"
            + f"dx: {self.dx}"
            + "\n"
            + f"Dimensions: {self.dim}D"
        )

    def __array__(self):
        return np.asarray(self.data)


class OsirisRawFile(OsirisData):
    """
    Class to read the raw data from an OSIRIS HDF5 file.

    Parameters
    ----------
    filename : str
        Path to OSIRIS HDF5 track file (.h5 extension)

    Attributes:
    -----------
    axis : dict[str, dict[str, str]]
        Dictionary where each key is a dataset name, and each value is another dictionary containing:
            - 'name' (str): Short name of the quantity (e.g., 'x1', 'ene')
            - 'units' (str): Units (LaTeX formatted, e.g., 'c/\\omega_p', 'm_e c^2')
            - 'long_name' (str): Descriptive name (LaTeX formatted, e.g., 'x_1', 'En2')
    data : dict[str, np.ndarray]
        Dataset values indexed by dataset name (quants).
    dim : int
        Number of spatial dimensions.
    dt : float
        Time step between iterations.
    grid : np.ndarray
        Grid boundaries as ((x1_min, x1_max), (x2_min, x2_max), ...)
    iter : int
        Iteration number corresponding to the data.
    name : str
        Name of the species.
    time : list[float, str]
        Simulation time and its units (e.g., [12.5, '1/\\omega_p']).
    type : str
        Type of data (e.g., 'particles' for raw files).
    labels : list[str]
        Field labels/names (LaTeX formatted, e.g., 'x_1')
    quants : list[str]
        field names of the data
    units : list[str]
        Units of each field of the data (LaTeX formatted, e.g., 'c/\\omega_p')

    Example
    -------
        >>> import osiris_utils as ou
        >>> raw = ou.raw = ou.OsirisRawFile("path/to/raw/file.h5")
        >>> print(raw.data.keys())
        >>> # Access x1 position of first 10 particles
        >>> print(raw.data[\"x1\"][0:10])
        >>> # Write beautiful labels and units
        >>> print("${} = $".format(raw.labels[\"x1\"]) + "$[{}]$".format(track.units[\"x1\"]))
    """

    def __init__(self, filename):
        super().__init__(filename)

        self._grid = np.array(
            [
                self._file["SIMULATION"].attrs["XMIN"],
                self._file["SIMULATION"].attrs["XMAX"],
            ]
        ).T

        self._quants = [byte.decode("utf-8") for byte in self._file.attrs["QUANTS"][:]]
        units_list = [byte.decode("utf-8") for byte in self._file.attrs["UNITS"][:]]
        labels_list = [byte.decode("utf-8") for byte in self._file.attrs["LABELS"][:]]
        self._units = dict(zip(self._quants, units_list))
        self._labels = dict(zip(self._quants, labels_list))

        self._data = {}
        self._axis = {}
        for key in self._file.keys():
            if key == "SIMULATION":
                continue

            self.data[key] = np.array(self._file[key][()])

            idx = np.where(self._file.attrs["QUANTS"] == str(key).encode("utf-8"))
            axis_data = {
                "name": self._file.attrs["QUANTS"][idx][0].decode("utf-8"),
                "units": self._file.attrs["UNITS"][idx][0].decode("utf-8"),
                "long_name": self._file.attrs["LABELS"][idx][0].decode("utf-8"),
            }
            self._axis[key] = axis_data

    def raw_to_file_tags(self, filename, type: Literal["all", "random"] = "all", n_tags=10, mask=None):
        """
        Function to write a file_tags file from raw data.
        this file is used to choose particles for the OSIRIS track diagnostic.

        Parameters
        ----------
        filename : str
            Path to the output file where tags will be stored.
        type : {'all', 'random'}, optional
            Selection mode for tags:
            - 'all': Includes all available tags.
            - 'random': Randomly selects `n_tags` tags.
        n_tags : int, optional
            Number of tags to randomly select when `type` is 'random'. Default is 10.
        mask : np.ndarray, optional
            Boolean mask array applied to filter valid tags before selection.

        Returns
        ------
        A file_tags file with path \"filename\" to be used for the OSIRIS track diagnostic.

        Notes
        -----
            The first element of the tag of a particle that is already being tracked is negative,
            so we apply the absolute function when generating the file

        """

        if mask is not None:
            # Apply mask to select certain tags
            if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape[0] != self.data["tag"].shape[0]:
                raise ValueError("Mask must be a boolean NumPy array of the same length as 'tag'.")
            filtered_indices = np.where(mask)[0]
            filtered_tags = self.data["tag"][filtered_indices]
        else:
            filtered_tags = self.data["tag"]

        if type == "all":
            tags = filtered_tags
        elif type == "random":
            if len(filtered_tags) < n_tags:
                raise ValueError("Not enough tags to sample from.")
            random_indices = np.random.choice(len(filtered_tags), size=n_tags, replace=False)
            tags = filtered_tags[random_indices]
        else:
            raise TypeError("Invalid type", type)

        create_file_tags(filename, tags)
        print("Tag_file created: ", filename)

    # Getters
    @property
    def grid(self):
        return self._grid

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def quants(self):
        return self._quants

    @property
    def axis(self):
        return self._axis


class OsirisHIST(OsirisData):
    """'
    Class to read the data from an OSIRIS HIST file.'

    Input
    -----
    filename: the path to the HIST file

    Attributes
    ----------
    filename: the path to the file
        str
    df: the data in a pandas DataFrame
        pandas.DataFrame
    """

    def __init__(self, filename):
        super().__init__(filename)

    @property
    def df(self):
        return self._df


class OsirisTrackFile(OsirisData):
    """
    Handles structured track data from OSIRIS HDF5 simulations.

    Parameters
    ----------
    filename : str
        Path to OSIRIS HDF5 track file (.h5 extension)

    Attributes
    ----------
    data: numpy.ndarray of shape (num_particles, num_time_iter),
        dtype = [(field_name, float) for field_name in field_names]
        A structured numpy array with the track data
        Accessed as data[particles, time_iters][quant]
    grid : np.ndarray
        Grid boundaries as ((x1_min, x1_max), (x2_min, x2_max), ...)
    labels : list[str]
        Field labels/names (LaTeX formatted, e.g., 'x_1')
    num_particles : int
        Number of particlest tracked, they are accessed from 0 to num_particles-1
    num_time_iters : int
        Number of time iteratis, they are accessed from 0 to num_time_iters-1
    quants : list[str]
        field names of the data
    units : list[str]
        Units of each field of the data (LaTeX formatted, e.g., 'c/\\omega_p')

    Example
    -------
        >>> import osiris_utils as ou
        >>> track = ou.OsirisTrackFile(path/to/track_file.h5)
        >>> print(track.data[0:10, :]["x1"]) # Access x1 position of first 10 particles over all time steps
    """

    def __init__(self, filename):
        super().__init__(filename)

        self._grid = np.array(
            [
                self._file["SIMULATION"].attrs["XMIN"],
                self._file["SIMULATION"].attrs["XMAX"],
            ]
        ).T

        self._quants = [byte.decode("utf-8") for byte in self._file.attrs["QUANTS"][1:]]
        units_list = [byte.decode("utf-8") for byte in self._file.attrs["UNITS"][1:]]
        labels_list = [byte.decode("utf-8") for byte in self._file.attrs["LABELS"][1:]]
        self._units = dict(zip(self._quants, units_list))
        self._labels = dict(zip(self._quants, labels_list))

        self._num_particles = self._file.attrs["NTRACKS"][0]

        unordered_data = self._file["data"][:]
        itermap = self._file["itermap"][:]

        idxs = get_track_indexes(itermap, self._num_particles)
        self._data = reorder_track_data(unordered_data, idxs, self._quants)
        self._time = self._data[0][:]["t"]
        self._num_time_iters = np.shape(self._time.shape)
        self._close_file()

    def _load_basic_attributes(self, f: h5py.File) -> None:
        """Load common attributes from HDF5 file"""
        self._dt = float(f["SIMULATION"].attrs["DT"][0])
        self._dim = int(f["SIMULATION"].attrs["NDIMS"][0])
        self._time = None
        self._iter = None
        self._name = f.attrs["NAME"][0].decode("utf-8")
        self._type = f.attrs["TYPE"][0].decode("utf-8")

    # Getters
    @property
    def grid(self):
        return self._grid

    @property
    def data(self):
        return self._data

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def quants(self):
        return self._quants

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def num_time_iters(self):
        return self._num_time_iters

    # Setters
    @data.setter
    def data(self, data):
        self._data = data

    def __str__(self):
        # write me a template to print with the name, label, units, iter, grid, nx, dx, axis, dt, dim in a logical way
        return (
            rf"{self.name}"
            + "\n"
            + f"Iteration: {self.iter}"
            + "\n"
            + f"Grid: {self.grid}"
            + "\n"
            + f"dx: {self.dx}"
            + "\n"
            + f"Dimensions: {self.dim}D"
        )

    def __array__(self):
        return np.asarray(self.data)


def reorder_track_data(unordered_data, indexes, field_names):
    """
    Reorder data from HDF5 track file such data it can be accessed more intuitively

    Parameters
    ----------
    unordered_data: np.array
        The data from a HDF5 osiris track file

    indexes : list[list[int]]
        Output of get_track_indexes(), list with the indexes associated with each particle

    field_names: list[str]
        Names for the quantities on the output file.
        Recommended: field_names = [byte.decode('utf-8') for byte in file.attrs['QUANTS'][1:]]

    Returns
    -------
    data_sorted: numpy.ndarray of shape (num_particles, num_time_iter),
                    dtype = [(field_name, float) for field_name in field_names]
        A structured numpy array where data is reordered according to indexes.

    """
    # Initialize the sorted data structure
    num_particles = len(indexes)
    num_time_iter = len(indexes[0])
    data_sorted = np.empty((num_particles, num_time_iter), dtype=[(name, float) for name in field_names])

    # Fill the sorted data based on the indexes
    for particle in range(num_particles):
        for time_iter in range(num_time_iter):
            index = indexes[particle][time_iter]
            if len(unordered_data[index]) != len(field_names):
                raise ValueError(f"Data at index {index} has {len(unordered_data[index])} elements, but {len(field_names)} are expected.")
            data_sorted[particle, time_iter] = tuple(unordered_data[index])

    return data_sorted


def get_track_indexes(itermap, num_particles):
    """
    Returns the indexes for each particle to read track data directly from the hd5 file
    (before it is ordered)

    Parameters
    ----------
    itermap: np.array
        Itermap from a HDF5 osiris track file
    num_particles: int
        num of particles tracked, recomended file.attrs['NTRACKS'][0]

    Returns
    -------
    indexes : list[list[int]]
        Returns a list with the indexes associated with each particle
        shape(num_particles, num_time_iters)
    """

    itermapshape = itermap.shape
    for i in range(itermapshape[0]):
        part_number, npoints, nstart = itermap[i, :]
    track_indices = np.zeros(num_particles)

    data_index = 0
    indexes = [[] for _ in range(num_particles)]
    for i in range(itermapshape[0]):
        part_number, npoints, nstart = itermap[i, :]

        indexes[part_number - 1].extend(list(range(data_index, data_index + npoints)))

        data_index += npoints
        track_indices[part_number - 1] += npoints

    return indexes
