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

class OsirisSimulation:
    def __init__(self, simulation_folder):
        self._simulation_folder = simulation_folder
        if not os.path.isdir(simulation_folder):
            raise FileNotFoundError(f"Simulation folder {simulation_folder} not found.")
    
    def get_moment(self, species, moment):
        self._path = f"{self._simulation_folder}/MS/UDIST/{species}/{moment}/"
        self._get_data_from_folder(self._path, centered=False)
    
    def get_field(self, field, centered=False):
        if centered:
            self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._path = f"{self._simulation_folder}/MS/FLD/{field}/"
        self._get_data_from_folder(self._path, centered)
        
    def get_density(self, species, quantity):
        self._path = f"{self._simulation_folder}/MS/DENSITY/{species}/{quantity}/"
        self._get_data_from_folder(self._path, centered=False)

    # def _get_data_from_folder(self, path, centered=False):
    #     files = os.listdir(path)
    #     iters = len(files)
    #     iter = []

    #     fname = files[0][:-9]

    #     data = []
    #     for i in tqdm.trange(iters, desc="Loading data"):
    #         if centered:
    #             aux = OsirisGridFile(f"{path}{fname}{i:06d}.h5")
    #             aux.yeeToCellCorner(boundary="periodic")
    #             data.append(aux)
    #         else:
    #             data.append(OsirisGridFile(f"{path}{fname}{i:06d}.h5"))

    #     self._dx = data[0].dx
    #     self._nx = data[0].nx
    #     self._x = data[0].x
    #     self._dt = data[0].dt
    #     self._grid = data[0].grid
    #     self._axis = data[0].axis
    #     self._units = data[0].units
    #     self._name = data[0].name
        
    #     if centered:
    #         self._data = np.array([d.data_centered for d in data])
    #     else:
    #         self._data = np.array([d.data for d in data])

    #     iter = np.array([d.iter for d in data])
    #     self._time = iter * self._dt
    
    def _get_data_from_folder(self, path, centered=False):
        files = sorted(
            [f for f in os.listdir(path) if f.endswith('.h5')],
            key=lambda x: int(x[-9:-3])  # Extracts 6-digit number from filename
        )
        if not files:
            raise ValueError("No valid .h5 files found in directory")

        data_list = []
        iterations = []
        first_data = None
        get_data = lambda d: d.data_centered if centered else d.data 

        for filename in tqdm.tqdm(files, desc="Loading data"):
            filepath = os.path.join(path, filename)
            data_object = OsirisGridFile(filepath)
            
            if centered:
                data_object.yeeToCellCorner(boundary="periodic")
            
            data_list.append(get_data(data_object))
            iterations.append(data_object.iter)

            if first_data is None:
                first_data = data_object
                self._dx = first_data.dx
                self._nx = first_data.nx
                self._x = first_data.x
                self._dt = first_data.dt
                self._grid = first_data.grid
                self._axis = first_data.axis
                self._units = first_data.units
                self._name = first_data.name

        self._data = np.array(data_list)
        self._time = np.array(iterations) * self._dt
        
    def derivative(self, type):
        if type == "t":
            self._data_t = np.gradient(self._data, self._dt, axis=0)
        if type == "x1":
            self._data_x1 = np.gradient(self._data, self._dx[0], axis=1)
        if type == "x2":
            self._data_x2 = np.gradient(self._data, self._dx[1], axis=2)
        if type == "x3":
            self._data_x3 = np.gradient(self._data, self._dx[2], axis=3)
    
    # Getters
    @property
    def data(self):
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
    def deriv_t(self):
        return self._data_t
    
    @property
    def deriv_x1(self):
        return self._data_x1
    
    @property
    def deriv_x2(self):
        return self._data_x2
    
    @property
    def path(self):
        return self
    
    @property
    def simulation_folder(self):
        return self._simulation_folder

        