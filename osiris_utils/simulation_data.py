"""
The utilities on data.py are cool but not useful when you want to work with whole data of a simulation instead
of just a single file. This is what this file is for - deal with ''folders'' of data.

Took some inspiration from Diogo and Madox's work.

This would be awsome to compute time derivatives. 
"""

import numpy as np
import os
from .data import OsirisGridFile, OsirisRawFile, OsirisHIST
import tqdm
import itertools

class OsirisSimulation:
    def __init__(self, simulation_folder):
        self._simulation_folder = simulation_folder
        if not os.path.isdir(simulation_folder):
            raise FileNotFoundError(f"Simulation folder {simulation_folder} not found.")
    
    def get_moment(self, species, moment):
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._load_attributes(self._file_template)
    
    def get_field(self, field, centered=False):
        if centered:
            self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._load_attributes(self._file_template)
        
    def get_density(self, species, quantity):
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._file_template = os.listdir(self._path)[0][:-9]
        self._load_attributes(self._file_template)

    def _load_attributes(self, file_template):
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
        self._ndump = dump1.iter
    
    def _data_generator(self, index):
            file = os.path.join(self._path, self._file_template + f"{index:06d}.h5")
            data_object = OsirisGridFile(file)
            if self._current_centered:
                data_object.yeeToCellCorner(boundary="periodic")
            yield data_object.data_centered if self._current_centered else data_object.data
    
    def load_all(self, centered=False):
        self._current_centered = centered
        files = sorted(os.listdir(self._path))
        size = len(files)
        self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])
    
    def load(self, index, centered=False):
        self._current_centered = centered
        self._data = next(self._data_generator(index))

    def __getitem__(self, index):
        return next(self._data_generator(index))
    
    def __iter__(self):
        for i in itertools.count():
            yield next(self._data_generator(i))
    
    # Getters
    @property
    def data(self):
        if self._data is None:
            raise ValueError("Data not loaded into memory. Use get_* method with load_all=True or access via generator/index.")
        return self._data
    
    @property
    def time(self):
        return self._time
    
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
    def path(self):
        return self
    
    @property
    def simulation_folder(self):
        return self._simulation_folder

        