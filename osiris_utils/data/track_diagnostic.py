"""
Equivalent of diagnostic but for tracks files instead of grid files
"""

import numpy as np
import os
import glob

from .data import OsirisTrackFile
import tqdm
import matplotlib.pyplot as plt
import warnings
from typing import Literal
from ..decks.decks import InputDeckIO, deval
from ..data.diagnostic import OSIRIS_ALL, OSIRIS_SPECIE_REP_UDIST, OSIRIS_SPECIE_REPORTS, OSIRIS_FLD, OSIRIS_PHA, OSIRIS_DENSITY



class Track_Diagnostic:
    """
    Class to handle track diagnostics for OSIRIS HDF5 simulation files.


    Parameters
    ----------
    species : str
        The species for which the track diagnostics will be handled. 
    simulation_folder : str, optional
        The path to the simulation folder. This folder must contain the relevant OSIRIS output data. If not provided, it defaults to None.
    input_deck : str, optional
       The path to the simulation folder. This is the path to the folder where the input deck is located.

    Attributes
    ----------
    species : str
        The species to handle the diagnostics.
    dt : float
        The time step for the simulation.
    grid : np.ndarray
        The grid boundaries for the simulation.
    units : dict
        Units of each field of the data (LaTeX formatted, e.g., 'c/\\omega_p')
    name : str
        The name of the diagnostic.
    labels : list of str
        Field labels/names (LaTeX formatted, e.g., 'x_1')
    dim : int
        The number of spatial dimensions for the simulation.
    ndump : int
        The number of steps between data dumps.
    tunits : str
        The units of time in the simulation.
    path : str
        The path to the track file.
    simulation_folder : str
        The path to the simulation folder where the OSIRIS data is located.
    all_loaded : bool
        Flag indicating whether the data has been fully loaded into memory.
    data : np.ndarray
        The diagnostic data, stored after it has been loaded into memory.
    num_time_iters : int
        The number of time iterations in the track data.
    num_particles : int
        The number of particles tracked in the simulation.
    quants : list of str
        The names of the quantities available in the track data.

    Methods
    -------
    load_all()
        Loads all diagnostic data into memory. Returns the full dataset.
    unload()
        Unloads the diagnostic data from memory to free up resources.
    load()
        Same as `load_all`. Loads the full diagnostic data into memory.
    __getitem__(index)
        Retrieves the track data for a specific quantity found in quants.

    """

    def __init__(self, simulation_folder=None, species=None, input_deck=None):
        self._species = species if species else None

        self._dt = None
        self._grid = None
        self._units = None
        self._name = None
        self._labels = None
        self._dim = None
        self._ndump = None
        self._tunits = None
        self._data = None
        self._time = None
        self._path = None
        self._num_time_iters = None
        self._num_particles = None
        self._quants = None

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
        self._get_quantity()

    def _get_quantity(self):
        """
        Get the data for a given quantity.
        """
        self._quantity = "tracks"


        if self._species is None:
            raise ValueError("Species not set.")
        if self._simulation_folder is None:
            raise ValueError(
                "Simulation folder not set."
            )
        
        self._path = os.path.join(self._simulation_folder, f"MS/TRACKS/{self._species.name}-tracks.h5")
        self._load_attributes()




    def _load_attributes(
        self, 
    ):  # this will be replaced by reading the input deck
        # This can go wrong! NDUMP
        # if input_deck is not None:
        #     self._dt = float(input_deck["time_step"][0]["dt"])
        #     self._ndump = int(input_deck["time_step"][0]["ndump"])
        #     self._dim = get_dimension_from_deck(input_deck)
        #     self._nx = np.array(list(map(int, input_deck["grid"][0][f"nx_p(1:{self._dim})"].split(','))))
        #     xmin = [deval(input_deck["space"][0][f"xmin(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     xmax = [deval(input_deck["space"][0][f"xmax(1:{self._dim})"].split(',')[i]) for i in range(self._dim)]
        #     self._grid = np.array([[xmin[i], xmax[i]] for i in range(self._dim)])
        #     self._dx = (self._grid[:,1] - self._grid[:,0])/self._nx
        #     self._x = [np.arange(self._grid[i,0], self._grid[i,1], self._dx[i]) for i in range(self._dim)]

        try:
            dump = OsirisTrackFile(self._path)
            self._dt = dump.dt
            self._grid = dump.grid
            self._units = dump.units
            self._name = dump.name
            self._labels = dump.labels
            self._dim = dump.dim
            self._ndump = dump.iter
            self._tunits = self._units["t"]
            self._num_time_iters = dump.num_time_iters
            self._num_particles = dump.num_particles
            self._quants = dump.quants
        except:
            pass

    def _data_generator(self, index = None):
        if self._simulation_folder is None:
            raise ValueError("Simulation folder not set.")
        data_object = OsirisTrackFile(self._path)

        if index is None:
            yield (
                data_object.data
            )
        else:
            if index not in self._quants:
                raise ValueError(f"Quantity {index} is invalid, Options are {self._quants}.")
            else:
                yield (
                    data_object.data[index]
                )

    def load_all(self):
        """
        Load data into memory.

        Returns
        -------
        data : np.ndarray
            The data for all iterations. Also stored in the attribute data.
        """
        # If data is already loaded, don't do anything
        if self._all_loaded and self._data is not None:
            print("Data already loaded.")
            return self._data


        # Original implementation for file-based diagnostics
        print("Loading data from tracks file.")
        self._data = next(self._data_generator())
        self._time = self._data["t"][0]
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

    def load(self):
        """
        For tracks files it is the same as load_all

        Returns
        -------
        data : np.ndarray
            The data for all iterations. Also stored in the attribute data.
        """
        return self.load_all()


    def __getitem__(self, index):
        # For derived diagnostics with cached data
        if self._all_loaded and self._data is not None:
            return self._data[index]

        # For standard diagnostics with files
        if isinstance(index, str):
            if self._simulation_folder is not None and hasattr(self, "_data_generator"):
                return next(self._data_generator(index))

        # If we get here, we don't know how to get data for this index
        raise ValueError(
            f"Cannot retrieve data for this diagnostic at index {index}. No data loaded and no generator available."
        )


    # Getters
    @property
    def data(self):
        if self._data is None:
            raise ValueError(
                "Data not loaded into memory. Use get_* method with load_all=True or access via generator/index."
            )
        return self._data

    @property
    def time(self):
        if self._time is None:
            raise ValueError(
                "Data not loaded into memory. Use get_* method with load_all=True or access via generator/index."
            )
        return self._time

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        self._species = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    @property
    def ndump(self):
        return self._ndump

    @ndump.setter
    def ndump(self, value):
        self._ndump = value

    @property
    def tunits(self):
        return self._tunits

    @tunits.setter
    def tunits(self, value):
        self._tunits = value

    @property
    def simulation_folder(self):
        return self._simulation_folder

    @simulation_folder.setter
    def simulation_folder(self, value):
        self._simulation_folder = value

    @property
    def input_deck(self):
        return self._input_deck

    @input_deck.setter
    def input_deck(self, value):
        self._input_deck = value

    @property
    def quantity(self):
        return self._quantity

    @quantity.setter
    def quantity(self, value):
        self._quantity = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def num_time_iters(self):
        return self._num_time_iters

    @num_time_iters.setter
    def num_time_iters(self, value):
        self._num_time_iters = value

    @property
    def num_particles(self):
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value):
        self._num_particles = value

    @property
    def quants(self):
        return self._quants

    @quants.setter
    def quants(self, value):
        self._quants = value