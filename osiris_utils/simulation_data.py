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
import multiprocessing as mp

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
        self._dim = dump1.dim
        self._ndump = dump1.iter
    
    def _data_generator(self, index):
            file = os.path.join(self._path, self._file_template + f"{index:06d}.h5")
            data_object = OsirisGridFile(file)
            if self._current_centered:
                data_object.yeeToCellCorner(boundary="periodic")
            yield data_object.data_centered if self._current_centered else data_object.data
    
    def load_all(self, centered=False):
        self._current_centered = centered
        size = len(sorted(os.listdir(self._path)))
        self._data = np.stack([self[i] for i in tqdm.tqdm(range(size), desc="Loading data")])

    def load_all_parallel(self, centered=False, processes=None):
        self._current_centered = centered
        files = sorted(os.listdir(self._path))
        size = len(files)
        
        if processes is None:
            processes = mp.cpu_count()
            print(f"Using {processes} CPUs for parallel loading")
        
        with mp.Pool(processes=processes) as pool:
            data = list(tqdm.tqdm(pool.imap(self.__getitem__, range(size)), total=size, desc="Loading data"))
        
        self._data = np.stack(data)
    
    def load(self, index, centered=False):
        self._current_centered = centered
        self._data = next(self._data_generator(index))

    def __getitem__(self, index):
        return next(self._data_generator(index))
    
    def __iter__(self):
        for i in itertools.count():
            yield next(self._data_generator(i))

    def derivative(self, point, type, axis=None):
        if point == "all":
            if type == "t":
                self._deriv_t = np.gradient(self.data, self.dt, axis=0, edge_order=2)
            elif type == "x1":
                if self._dim == 1:
                    self._deriv_x1 = np.gradient(self.data, self.dx, axis=1, edge_order=2)
                else:
                    self._deriv_x1 = np.gradient(self.data, self.dx[0], axis=1, edge_order=2)
            elif type == "x2":
                self._deriv_x1 = np.gradient(self.data, self.dx[0], axis=2, edge_order=2)
            elif type == "x3":
                self._deriv_x2 = np.gradient(self.data, self.dx[0], axis=3, edge_order=2)
            elif type == "xx":
                if len(axis) != 2:
                    raise ValueError("Axis must be a tuple with two elements.")
                self._deriv_xx = np.gradient(np.gradient(self.data, self.dx[axis[0]], axis=axis[0], edge_order=2), self.dx[axis[1]], axis=axis[1], edge_order=2)
            elif type == "xt":
                if not isinstance(axis, int):
                    raise ValueError("Axis must be an integer.")
                self._deriv_xt = np.gradient(np.gradient(self.data, self.dt, axis=0, edge_order=2), self.dx[axis], axis=axis, edge_order=2)
            elif type == "tx":
                if not isinstance(axis, int):
                    raise ValueError("Axis must be an integer.")
                self._deriv_tx = np.gradient(np.gradient(self.data, self.dx[axis], axis=axis, edge_order=2), self.dt, axis=axis, edge_order=2)
            else:
                raise ValueError("Invalid type.")
        else:
            try:
                if type == "x1":
                    if self._dim == 1:
                        return np.gradient(self[point], self._dx, axis=0)
                    else:
                        return np.gradient(self[point], self._dx[0], axis=0)
                
                elif type == "x2":
                    return np.gradient(self[point], self._dx[1], axis=1)
                
                elif type == "x3":
                    return np.gradient(self[point], self._dx[2], axis=2)
                    
                elif type == "t":
                    if point == 0:
                        return (-3 * self[point] + 4 * self[point + 1] - self[point + 2]) / (2 * self._dt)
                    # derivate at last point not implemented yet
                    # elif self[point + 1] is None:
                    #     return (3 * self[point] - 4 * self[point - 1] + self[point - 2]) / (2 * self._dt)
                    else:
                        return (self[point + 1] - self[point - 1]) / (2 * self._dt)
                else:
                    raise ValueError("Invalid derivative type. Use 'x1', 'x2' or 't'.")
                    
            except Exception as e:
                raise ValueError(f"Error computing derivative at point {point}: {str(e)}")
    
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
    def deriv_t(self):
        return self._deriv_t
    
    @property
    def deriv_x1(self):
        return self._deriv_x1
    
    @property
    def deriv_x2(self):
        return self._deriv_x2
    
    @property
    def deriv_xx(self):
        return self._deriv_xx
    
    @property
    def deriv_xt(self):
        return self._deriv_xt
    
    @property
    def deriv_tx(self):
        return self._deriv_tx

        