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
                axis_data = {
                    "name": f["AXIS/"+ax].attrs["NAME"][0].decode('utf-8'),
                    "units": f["AXIS/"+ax].attrs["UNITS"][0].decode('utf-8'),
                    "long_name": f["AXIS/"+ax].attrs["LONG_NAME"][0].decode('utf-8'),
                    "type": f["AXIS/"+ax].attrs["TYPE"][0].decode('utf-8'),
                }
                self.axis.append(axis_data)
                    
            # NOW WORK ON THE SIMULATION DATA
            self.dt = float(f["SIMULATION"].attrs["DT"][0])
            self.dim = int(f["SIMULATION"].attrs["NDIMS"][0])
            self.time = [float(f.attrs["TIME"][0]), f.attrs["TIME UNITS"][0].decode('utf-8')]
            self.iter = int(f.attrs["ITER"][0])
            self.name = f.attrs["NAME"][0].decode('utf-8')
            self.units = f.attrs["UNITS"][0].decode('utf-8')
            self.label = f.attrs["LABEL"][0].decode('utf-8')
            self.type = f.attrs["TYPE"][0].decode('utf-8')
            self.data = self.data.T


    def __yeeToCellCorner1d(self, x):
        """
        ! Converts 1d EM fields from a staggered Yee mesh to a grid with field values centered on the corner
        ! of the cell (the corner of the cell [1] has coordinates [1])
        """


        def B1(B1, x):
            return(B1[x])
        
        def B2(B2, x):
            return(0.5 * (B2[x] + B2[x-1]))     

        def B3(B3, x):
            return(0.5 * (B3[x] + B3[x-1])) 

        def E1(E1, x):
            return(0.5 * (E1[x] + E1[x-1]))
        
        def E2(E2, x):
            return(E2[x])     

        def E3(E3, x):
            return(E3[x]) 
              
        def case_default(data, x):
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        
        cases = {
            'b1': B1,
            'b2': B2,
            'b3': B3,
            'e1': E1,
            'e2': E2,
            'e3': E3,
            'default': case_default,
        }

        return cases.get(self.name, cases['default'])(self.data, x)
    

    def __yeeToCellCorner2d(self, x, y):
        """
        ! Converts 2d EM fields from a staggered Yee mesh to a grid with field values centered on the corner
        ! of the cell (the corner of the cell [1,1] has coordinates [1,1])
        """


        def B1(B1, x, y):
            return(0.5 * (B1[x, y] + B1[x, y-1]))
        
        def B2(B2, x, y):
            return(0.5 * (B2[x, y] + B2[x-1, y]))
        
        def B3(B3, x, y):
            return(0.25 * (B3[x, y] + B3[x-1, y] + B3[x, y-1] + B3[x-1, y-1]))
        
        def E1(E1, x, y):
            return(0.5 * (E1[x, y] + E1[x-1, y]))
        
        def E2(E2, x, y):
            return(0.5 * (E2[x, y] + E2[x, y-1]))
        
        def E3(E3, x, y):
            return(E3[x, y])
        
        def case_default(data, x, y):
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        
        cases = {
            'b1': B1,
            'b2': B2,
            'b3': B3,
            'e1': E1,
            'e2': E2,
            'e3': E3,
            'default': case_default,
        }

        return cases.get(self.name, cases['default'])(self.data, x, y)
    

    def __yeeToCellCorner3d(self, x, y, z):
        """
        ! Converts 3d EM fields from a staggered Yee mesh to a grid with field values centered on the corner
        ! of the cell (the corner of the cell [1,1,1] has coordinates [1,1,1])
        """


        def B1(B1, x, y, z):
            return(0.25 * (B1[x, y, z] + B1[x, y-1, z] + B1[x, y, z-1] + B1[x, y-1, z-1]))
        
        def B2(B2, x, y, z):
            return(0.25 * (B2[x, y, z] + B2[x-1, y, z] + B2[x, y, z-1] + B2[x-1, y, z-1]))
        
        def B3(B3, x, y, z):
            return(0.25 * (B3[x, y, z] + B3[x-1, y, z] + B3[x, y-1, z] + B3[x-1, y-1, z]))
        
        def E1(E1, x, y, z):
            return(0.5 * (E1[x, y, z] + E1[x-1, y, z]))
        
        def E2(E2, x, y, z):
            return(0.5 * (E2[x, y, z] + E2[x, y-1, z]))
        
        def E3(E3, x, y, z):
            return(0.5 * (E3[x, y, z] + E3[x, y, z-1]))
        
        def case_default(data, x, y, z):
            raise TypeError(f"This method expects magnetic or electric field grid data but received \"{self.name}\" instead")
        
        cases = {
            'b1': B1,
            'b2': B2,
            'b3': B3,
            'e1': E1,
            'e2': E2,
            'e3': E3,
            'default': case_default,
        }

        return cases.get(self.name, cases['default'])(self.data, x, y, z)
        
        
    def yeeToCellCorner(self):
        """
        Converts EM fields from a staggered Yee mesh to a grid with field values centered on the corner
        of the cell (ex the corner of the cell [1,1,1] has coordinates [1,1,1])
        The dimension of the new data is smaller because it is not possible to calcualte the values on the corners 
        of the 0th cells

        Returns:
        - new_data: the data (numpy array) with shape (nx1-1), (nx1-1, nx2-1) or (nx1-1, nx2-1, nx3-1), 
            depending on the dimension, with the fields defined on the corner of the grid, instead of the Yee mesh.
            numpy.ndarray
        """ 
        
        
        if self.dim == 1:
            shape = np.shape(self.data)
            new_data = np.empty(shape-np.array([1]))
            for x in range(1, shape[0]):
                        new_data[x-1] = self.__yeeToCellCorner1d(x)

            return new_data

        elif self.dim == 2:
            shape = np.shape(self.data)
            new_data = np.empty(shape-np.array([1,1]))
            for x in range(1, shape[0]):
                for y in range(1, shape[1]):
                        new_data[x-1][y-1] = self.__yeeToCellCorner2d(x, y)

            return new_data

        elif self.dim == 3:
            shape = np.shape(self.data)
            new_data = np.empty(shape-np.array([1,1,1]))
            for x in range(1, shape[0]):
                for y in range(1, shape[1]):
                    for z in range(1, shape[2]):
                        new_data[x-1][y-1][z-1] = self.__yeeToCellCorner3d(x, y, z)

            return new_data
        
        else:
            raise ValueError("Invalid simulation dimension.")



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
