import numpy as np
import h5py

class OsirisGridFile():
    '''
    Class to read the grid data from an OSIRIS HDF5 file.
    
    Input:
        - filename: the path to the HDF5 file
        
    Attributes:
        - grid - the grid data ((x1.min, x1.max), (x2.min, x2.max), (x3.min, x3.max))
            numpy.ndarray
        - nx - the number of grid points (nx1, nx2, nx3)
            numpy.ndarray
        - dx - the grid spacing (dx1, dx2, dx3)
            numpy.ndarray
        - axis - the axis data [(name_x1, units_x1, long_name_x1, type_x1), ...]
            list of dictionaries
        - data: the data (numpy array) with shape (nx1, nx2, nx3) (Transpose to use `plt.imshow`)
            numpy.ndarray
        - dt - the time step
            float
        - dim - the number of dimensions
            int
        - time - the time and its units
            list [time, units]
            list [float, str]
        - iter - the iteration number
            int
        - name - the name of the data
            str
        - units - the units of the data
            str
        - label - the label of the data (LaTeX formatted)
        
    '''
    def __init__(self, filename):
        with h5py.File(filename, 'r+') as f:
            self._load_basic_attributes(f)
            variable_key = self._get_variable_key(f)
            
            data = np.array(f[variable_key][:])

            axis = list(f["AXIS"].keys())
            if len(axis) == 1:
                self.grid = f["AXIS/" + axis[0]][()]
                self.nx = len(data)
                self.dx = (self.grid[1] - self.grid[0] ) / self.nx
                self.x = np.arange(self.grid[0], self.grid[1], self.dx)
            else: 
                grid = []
                for ax in axis: grid.append(f["AXIS/" + ax][()])
                self.grid = np.array(grid)
                self.nx = f[variable_key][()].transpose().shape
                self.dx = (self.grid[:, 1] - self.grid[:, 0])/self.nx
                self.x = [np.arange(self.grid[i, 0], self.grid[i, 1], self.dx[i]) for i in range(self.dim)]

            self.axis = []
            for ax in axis:
                axis_data = {
                    "name": f["AXIS/"+ax].attrs["NAME"][0].decode('utf-8'),
                    "units": f["AXIS/"+ax].attrs["UNITS"][0].decode('utf-8'),
                    "long_name": f["AXIS/"+ax].attrs["LONG_NAME"][0].decode('utf-8'),
                    "type": f["AXIS/"+ax].attrs["TYPE"][0].decode('utf-8'),
                    "plot_label": rf"${f["AXIS/"+ax].attrs["LONG_NAME"][0].decode('utf-8')}$ $[{f["AXIS/"+ax].attrs['UNITS'][0].decode('utf-8')}]$",
                }
                self.axis.append(axis_data)
            
                self.data = np.ascontiguousarray(data.T)

    def _load_basic_attributes(self, f: h5py.File) -> None:
        """Load common attributes from HDF5 file"""
        self.dt = float(f["SIMULATION"].attrs["DT"][0])
        self.dim = int(f["SIMULATION"].attrs["NDIMS"][0])
        self.time = [float(f.attrs["TIME"][0]), f.attrs["TIME UNITS"][0].decode('utf-8')]
        self.iter = int(f.attrs["ITER"][0])
        self.name = f.attrs["NAME"][0].decode('utf-8')
        self.units = f.attrs["UNITS"][0].decode('utf-8')
        self.label = f.attrs["LABEL"][0].decode('utf-8')
        self.type = f.attrs["TYPE"][0].decode('utf-8')

    def _get_variable_key(self, f: h5py.File) -> str:
        return next(k for k in f.keys() if k not in {"AXIS", "SIMULATION"})

    def _yeeToCellCorner1d(self, boundary):
        """
        Converts 1d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1] has coordinates [1])
        """

        if self.name.lower() in ["b2", "b3", "e1"]:
            if boundary == "periodic": return 0.5 * (np.roll(self.data, shift=1) + self.data) 
            else: return 0.5 * (self.data[1:] + self.data[:-1])
        elif self.name.lower() in ["b1", "e2", "e3"]:
            if boundary == "periodic": return self.data 
            else: return  self.data[1:]
        else: 
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
    

    def _yeeToCellCorner2d(self, boundary):
        """
        Converts 2d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1,1] has coordinates [1,1])
        """

        if self.name.lower() in ["e1", "b2"]:
            if boundary == "periodic": return 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
            else: return 0.5 * (self.data[1:, 1:] + self.data[:-1, 1:])
        elif self.name.lower() in ["e2", "b1"]:
            if boundary == "periodic": return 0.5 * (np.roll(self.data, shift=1, axis=1) + self.data)
            else: return 0.5 * (self.data[1:, 1:] + self.data[1:, :-1])
        elif self.name.lower() in ["b3"]:
            if boundary == "periodic": 
                # a1 = 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
               return 0.5 * (np.roll((0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)), shift=1, axis=1) + (0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)))
            else:
                return 0.25 * (self.data[1:, 1:] + self.data[:-1, 1:] + self.data[1:, :-1] + self.data[:-1, :-1])
        elif self.name.lower() in ["e3"]:
            if boundary == "periodic": return self.data
            else: return self.data[1:, 1:]
        else:
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        

    def _yeeToCellCorner3d(self, boundary):
        """
        Converts 3d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1,1,1] has coordinates [1,1,1])
        """
        if boundary == "periodic":
            raise ValueError("Centering field from 3D simulations considering periodic boundary conditions is not implemented yet")
        if self.name.lower() == "b1":
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[1:, 1:, :-1] + self.data[1:, :-1, :-1])
        elif self.name.lower() == "b2":
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, 1:, :-1] + self.data[:-1, 1:, :-1])
        elif self.name.lower() == "b3":
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[:-1, :-1, 1:])
        elif self.name.lower() == "e1":
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:])
        elif self.name.lower() == "e2":
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:])
        elif self.name.lower() == "e3":
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, 1:, :-1])
        else:
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        
    def yeeToCellCorner(self, boundary=None):
        """
        Converts EM fields from a staggered Yee mesh to a grid with field values centered on the corner
        of the cell (ex the corner of the cell [1,1,1] has coordinates [1,1,1])
        The dimension of the new data is smaller because it is not possible to calcualte the values on the corners 
        of the 0th cells

        Returns:
            - new_data: the data (numpy array) with shape (nx1-1), (nx1-1, nx2-1) or (nx1-1, nx2-1, nx3-1), depending on the dimension, with the fields defined on the corner of the grid, instead of the Yee mesh.
                numpy.ndarray
        """ 
        
        cases = {"b1", "b2", "b3", "e1", "e2", "e3"}
        if self.name not in cases:
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        
        if self.dim == 1:
            self.data_centered = self._yeeToCellCorner1d(boundary)
            return self.data_centered
        elif self.dim == 2:
            self.data_centered = self._yeeToCellCorner2d(boundary)
            return self.data_centered
        elif self.dim == 3:
            self.data_centered = self._yeeToCellCorner3d(boundary)
            return self.data_centered
        else:
            raise ValueError(f"Dimension {self.dim} is not supported")


class OsirisRawFile():
    '''
    Class to read the raw data from an OSIRIS HDF5 file.
    
    Input:
        - filename: the path to the HDF5 file
    
    Attributes:
        - axis - a dictionary where each key is a dataset name, and each value is another dictionary containing
            name (str): The name of the quantity (e.g., r'x1', r'ene').
            units (str): The units associated with that dataset in LaTeX (e.g., r'c/\\omega_p', r'm_e c^2').
            long_name (str): The name of the quantity in LaTeX (e.g., r'x_1', r'En2').
            dictionary of dictionaries
        - data - a dictionary where each key is a dataset name, and each value is the data
            dictionary of np.arrays
        - dim - the number of dimensions
            int
        - dt - the time step
            float
        - grid - maximum and minimum coordinates of the box, for each axis 
            numpy.ndarray(dim,2)
        - iter - the iteration number
            int
        - name - the name of the species
            str
        - time - the time and its units
            list [time, units]
            list [float, str]
        - type - type of data (particles in the case of raw files)
            str

    '''
    
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(self.filename, "r") as f: 

            self.dt = float(f["SIMULATION"].attrs["DT"][0])
            self.dim = int(f["SIMULATION"].attrs["NDIMS"][0])
            self.time = [float(f.attrs["TIME"][0]), f.attrs["TIME UNITS"][0].decode('utf-8')]
            self.iter = int(f.attrs["ITER"][0])
            self.name = f.attrs["NAME"][0].decode('utf-8')
            self.type = f.attrs["TYPE"][0].decode('utf-8')
            self.grid = np.array([f["SIMULATION"].attrs["XMIN"], f["SIMULATION"].attrs["XMAX"]]).T

            self.data = {}
            self.axis = {}
            for key in f.keys():
                if key == "SIMULATION": continue

                self.data[key] = np.array(f[key][()])

                idx = np.where(f.attrs["QUANTS"] == str(key).encode('utf-8'))
                axis_data = {
                    "name": f.attrs["QUANTS"][idx][0].decode('utf-8'),
                    "units": f.attrs["UNITS"][idx][0].decode('utf-8'),
                    "long_name": f.attrs["LABELS"][idx][0].decode('utf-8'),
                }
                self.axis[key] = axis_data
