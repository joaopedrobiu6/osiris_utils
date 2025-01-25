import numpy as np
import h5py

class OsirisGridFile():
    '''
    Class to read the grid data from an OSIRIS HDF5 file.
    
    Input:
        - filename: the path to the HDF5 file
        
    Attributes:
        - grid: the grid data ((x1.min, x1.max), (x2.min, x2.max), (x3.min, x3.max))
            numpy.ndarray
        - nx: the number of grid points (nx1, nx2, nx3)
            numpy.ndarray
        - dx: the grid spacing (dx1, dx2, dx3)
            numpy.ndarray
        - axis: the axis data [(name_x1, units_x1, long_name_x1, type_x1), ...]
            list of dictionaries
        - data: the data (numpy array) (it is transposed so the shape will be (nx3, nx2, nx1))
            numpy.ndarray
        - dt: the time step
            float
        - dim: the number of dimensions
            int
        - time: the time and its units
            list [time, units]
            list [float, str]
        - iter: the iteration number
            int
        - name: the name of the data
            str
        - units: the units of the data
            str
    '''
    def __init__(self, filename):
        with h5py.File(filename, 'r+') as f:
            # Get the data 
            known_keys = {"AXIS", "SIMULATION"}
            all_keys = set(f.keys())
            variable_key = (all_keys - known_keys).pop()
            
            # The data
            self.data = np.array(f[variable_key][:])

            keys = list(f.keys())
            # Now get the infos
            axis = list(f["AXIS"].keys())
            if len(axis) == 1:
                self.grid = f["AXIS/"+axis[0]][()]
                self.nx = len(self.data)
                self.dx = (self.grid[1] - self.grid[0] ) / self.nx
            else: 
                grid = []
                for ax in axis: grid.append(f["AXIS/"+ax][()])
                self.grid = np.array(grid)
                
                self.nx = f[variable_key][()].transpose().shape
                self.dx = (self.grid[:, 1] - self.grid[:, 0])/self.nx
            self.axis = []
            for ax in axis:
                self.axis_data = {
                    "name": f["AXIS/"+ax].attrs["NAME"][0].decode('utf-8'),
                    "units": f["AXIS/"+ax].attrs["UNITS"][0].decode('utf-8'),
                    "long_name": f["AXIS/"+ax].attrs["LONG_NAME"][0].decode('utf-8'),
                    "type": f["AXIS/"+ax].attrs["TYPE"][0].decode('utf-8'),
                }
                self.axis.append( self.axis_data )
                    
            # NOW WORK ON THE SIMULATION DATA
            self.dt = float(f["SIMULATION"].attrs["DT"][0])
            self.dim = int(f["SIMULATION"].attrs["NDIMS"][0])
            self.time = [float(f.attrs["TIME"][0]), f.attrs["TIME UNITS"][0].decode('utf-8')]
            self.iter = int(f.attrs["ITER"][0])
            self.name = f.attrs["NAME"][0].decode('utf-8')
            self.units = f.attrs["UNITS"][0].decode('utf-8')