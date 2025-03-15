import numpy as np
import h5py

class OsirisData():
    '''
    Parent class for Osiris data files.

    Input:
    ------
        - filename: the path to the HDF5 file
    Attributes:
    ----------
        - filename - the path to the HDF5 file
            str
        - file - the HDF5 file
            h5py.File
        - verbose - if True, the class will print messages
            bool
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
        - type - the type of data
            str
    '''
    def __init__(self, filename):
        self._filename = filename
        self._file = None

        self._verbose = False

        self._open_file(self._filename)
        self._load_basic_attributes(self._file)
        
    def _load_basic_attributes(self, f: h5py.File) -> None:
        '''Load common attributes from HDF5 file'''
        self._dt = float(f['SIMULATION'].attrs['DT'][0])
        self._dim = int(f['SIMULATION'].attrs['NDIMS'][0])
        self._time = [float(f.attrs['TIME'][0]), f.attrs['TIME UNITS'][0].decode('utf-8')]
        self._iter = int(f.attrs['ITER'][0])
        self._name = f.attrs['NAME'][0].decode('utf-8')
        self._type = f.attrs['TYPE'][0].decode('utf-8')
    
    def verbose(self, verbose: bool = True):
        '''
        Set the verbosity of the class

        Parameters
        ----------
        verbose : bool, optional
            If True, the class will print messages, by default True when calling (False when not calling)
        '''
        self._verbose = verbose

    def _open_file(self, filename):
        '''
        Open the HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        '''
        if self._verbose: print(f'Opening file > {filename}')
        self._file = h5py.File(self._filename, 'r')

    def _close_file(self):
        '''
        Close the HDF5 file.
        '''
        if self._verbose: print('Closing file')
        if self._file:
            self._file.close()
        
    @property
    def dt(self):
        '''
        Returns
        -------
        float
            The time step
        '''
        return self._dt
    @property
    def dim(self):
        '''
        Returns
        -------
        int
            The number of dimensions
        '''
        return self._dim
    @property
    def time(self):
        '''
        Returns
        -------
        list
            The time and its units
        '''
        return self._time
    @property
    def iter(self):
        '''
        Returns
        -------
        int
            The iteration number
        '''
        return self._iter
    @property
    def name(self):
        '''
        Returns
        -------
        str
            The name of the data
        '''
        return self._name
    @property
    def type(self):
        '''
        Returns
        -------
        str
            The type of data
        '''
        return self._type

class OsirisGridFile(OsirisData):
    '''
    Class to read the grid data from an OSIRIS HDF5 file.
    
    Input:
    -----
        - filename: the path to the HDF5 file
        
    Attributes:
    ----------
        - filename - the path to the HDF5 file
            str
        - file - the HDF5 file
            h5py.File
        - verbose - if True, the class will print messages
            bool
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
        - type - the type of data
            str
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
        - units - the units of the data
            str
        - label - the label of the data (LaTeX formatted)
        
    '''
    def __init__(self, filename):
        super().__init__(filename)
            
        variable_key = self._get_variable_key(self._file)
        
        self._units = self._file.attrs['UNITS'][0].decode('utf-8')
        self._label = self._file.attrs['LABEL'][0].decode('utf-8')
        self._FFTdata = None
        
        data = np.array(self._file[variable_key][:])

        axis = list(self._file['AXIS'].keys())
        if len(axis) == 1:
            self._grid = self._file['AXIS/' + axis[0]][()]
            self._nx = len(data)
            self._dx = (self.grid[1] - self.grid[0] ) / self.nx
            self._x = np.arange(self.grid[0], self.grid[1], self.dx)
        else: 
            grid = []
            for ax in axis: grid.append(self._file['AXIS/' + ax][()])
            self._grid = np.array(grid)
            self._nx = self._file[variable_key][()].transpose().shape
            self._dx = (self.grid[:, 1] - self.grid[:, 0])/self.nx
            self._x = [np.arange(self.grid[i, 0], self.grid[i, 1], self.dx[i]) for i in range(self.dim)]

        self._axis = []
        for ax in axis:
            axis_data = {
                'name': self._file['AXIS/'+ax].attrs['NAME'][0].decode('utf-8'),
                'units': self._file['AXIS/'+ax].attrs['UNITS'][0].decode('utf-8'),
                'long_name': self._file['AXIS/'+ax].attrs['LONG_NAME'][0].decode('utf-8'),
                'type': self._file['AXIS/'+ax].attrs['TYPE'][0].decode('utf-8'),
                'plot_label': rf'${self._file["AXIS/"+ax].attrs["LONG_NAME"][0].decode("utf-8")}$ $[{self._file["AXIS/"+ax].attrs["UNITS"][0].decode("utf-8")}]$',
            }
            self._axis.append(axis_data)
        
        self._data = np.ascontiguousarray(data.T)

        self._close_file()

    def _load_basic_attributes(self, f: h5py.File) -> None:
        '''Load common attributes from HDF5 file'''
        self._dt = float(f['SIMULATION'].attrs['DT'][0])
        self._dim = int(f['SIMULATION'].attrs['NDIMS'][0])
        self._time = [float(f.attrs['TIME'][0]), f.attrs['TIME UNITS'][0].decode('utf-8')]
        self._iter = int(f.attrs['ITER'][0])
        self._name = f.attrs['NAME'][0].decode('utf-8')
        self._type = f.attrs['TYPE'][0].decode('utf-8')
            
    def _get_variable_key(self, f: h5py.File) -> str:
        return next(k for k in f.keys() if k not in {'AXIS', 'SIMULATION'})
    
    # Getters
    @property
    def grid(self):
        '''
        Returns
        -------
        numpy.ndarray
            The grid data ((x1.min, x1.max), (x2.min, x2.max), (x3.min, x3.max
        '''
        return self._grid
    @property
    def nx(self):
        '''
        Returns
        -------
        numpy.ndarray
            The number of grid points (nx1, nx2, nx3)
        '''
        return self._nx
    @property
    def dx(self):
        '''
        Returns
        -------
        numpy.ndarray
            The grid spacing (dx1, dx2, dx3)
        '''
        return self._dx
    @property
    def x(self):
        '''
        Returns
        -------
        numpy.ndarray
            The grid points in each axis
        '''
        return self._x
    @property
    def axis(self):
        '''
        Returns
        -------
        list of dictionaries
            The axis data [(name_x1, units_x1, long_name_x1, type_x1), ...]
        '''
        return self._axis   
    @property
    def data(self):
        '''
        Returns
        -------
        numpy.ndarray
            The data (numpy array) with shape (nx1, nx2, nx3) (Transpose to use `plt.imshow`)
        '''
        return self._data
    @property
    def units(self):
        '''
        Returns
        -------
        str
            The units of the data (LaTeX formatted)
        '''
        return self._units
    @property
    def label(self):
        '''
        Returns
        -------
        str
            The label of the data (LaTeX formatted)
        '''
        return self._label
    @property
    def FFTdata(self):
        '''
        Returns
        -------
        numpy.ndarray
            The FFT of the data
        '''
        if self._FFTdata is None:
            raise ValueError('The FFT of the data has not been computed yet. Compute it using the FFT method.')
        return self._FFTdata
    # Setters
    @data.setter
    def data(self, data):
        '''
        Set the data attribute
        '''
        self._data = data

    def __str__(self):
        # write me a template to print with the name, label, units, time, iter, grid, nx, dx, axis, dt, dim in a logical way
        return rf'{self.name}' + f'\n' + rf'Time: [{self.time[0]} {self.time[1]}], dt = {self.dt}' + f'\n' + f'Iteration: {self.iter}' + f'\n' + f'Grid: {self.grid}' + f'\n' + f'dx: {self.dx}' + f'\n' + f'Dimensions: {self.dim}D'
    

    def __array__(self):
        return np.asarray(self.data)
    

    def _yeeToCellCorner1d(self, boundary):
        '''
        Converts 1d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1] has coordinates [1])
        '''

        if self.name.lower() in ['b2', 'b3', 'e1']:
            if boundary == 'periodic': return 0.5 * (np.roll(self.data, shift=1) + self.data) 
            else: return 0.5 * (self.data[1:] + self.data[:-1])
        elif self.name.lower() in ['b1', 'e2', 'e3']:
            if boundary == 'periodic': return self.data 
            else: return  self.data[1:]
        else: 
            raise TypeError(f'This method expects magnetic or electric field grid data but received \'{self.name}\' instead')
    

    def _yeeToCellCorner2d(self, boundary):
        '''
        Converts 2d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1,1] has coordinates [1,1])
        '''

        if self.name.lower() in ['e1', 'b2']:
            if boundary == 'periodic': return 0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)
            else: return 0.5 * (self.data[1:, 1:] + self.data[:-1, 1:])
        elif self.name.lower() in ['e2', 'b1']:
            if boundary == 'periodic': return 0.5 * (np.roll(self.data, shift=1, axis=1) + self.data)
            else: return 0.5 * (self.data[1:, 1:] + self.data[1:, :-1])
        elif self.name.lower() in ['b3']:
            if boundary == 'periodic': 
               return 0.5 * (np.roll((0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)), shift=1, axis=1) + (0.5 * (np.roll(self.data, shift=1, axis=0) + self.data)))
            else:
                return 0.25 * (self.data[1:, 1:] + self.data[:-1, 1:] + self.data[1:, :-1] + self.data[:-1, :-1])
        elif self.name.lower() in ['e3']:
            if boundary == 'periodic': return self.data
            else: return self.data[1:, 1:]
        else:
            raise TypeError(f'This method expects magnetic or electric field grid data but received \'{self.name}\' instead')
        

    def _yeeToCellCorner3d(self, boundary):
        '''
        Converts 3d EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell (the corner of the cell [1,1,1] has coordinates [1,1,1])
        '''
        if boundary == 'periodic':
            raise ValueError('Centering field from 3D simulations considering periodic boundary conditions is not implemented yet')
        if self.name.lower() == 'b1':
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[1:, 1:, :-1] + self.data[1:, :-1, :-1])
        elif self.name.lower() == 'b2':
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, 1:, :-1] + self.data[:-1, 1:, :-1])
        elif self.name.lower() == 'b3':
            return 0.25 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:] + self.data[1:, :-1, 1:] + self.data[:-1, :-1, 1:])
        elif self.name.lower() == 'e1':
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[:-1, 1:, 1:])
        elif self.name.lower() == 'e2':
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, :-1, 1:])
        elif self.name.lower() == 'e3':
            return 0.5 * (self.data[1:, 1:, 1:] + self.data[1:, 1:, :-1])
        else:
            raise TypeError(f'This method expects magnetic or electric field grid data but received \'{self.name}\' instead')
        
    def yeeToCellCorner(self, boundary=None):
        ''''
        Converts EM fields from a staggered Yee mesh to a grid with field values centered on the corner of the cell.'
        Can be used for 1D, 2D and 3D simulations.'
        Creates a new attribute `data_centered` with the centered data.'
        '''
        
        cases = {'b1', 'b2', 'b3', 'e1', 'e2', 'e3'}
        if self.name not in cases:
            raise TypeError(f'This method expects magnetic or electric field grid data but received \'{self.name}\' instead')
        
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
            raise ValueError(f'Dimension {self.dim} is not supported')
        
    def FFT(self, axis=(0, )):
        '''
        Computes the Fast Fourier Transform of the data along the specified axis and shifts the zero frequency to the center.
        Transforms the data to the frequency domain. A(x, y, z) -> A(kx, ky, kz)
        '''
        datafft = np.fft.fftn(self.data, axes=axis)
        self._FFTdata = np.fft.fftshift(datafft, axes=axis)



class OsirisRawFile(OsirisData):
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
        super().__init__(filename)

        self.grid = np.array([self._file['SIMULATION'].attrs['XMIN'], self._file['SIMULATION'].attrs['XMAX']]).T

        self.data = {}
        self.axis = {}
        for key in self._file.keys():
            if key == 'SIMULATION': continue

            self.data[key] = np.array(self._file[key][()])

            idx = np.where(self._file.attrs['QUANTS'] == str(key).encode('utf-8'))
            axis_data = {
                'name': self._file.attrs['QUANTS'][idx][0].decode('utf-8'),
                'units': self._file.attrs['UNITS'][idx][0].decode('utf-8'),
                'long_name': self._file.attrs['LABELS'][idx][0].decode('utf-8'),
            }
            self.axis[key] = axis_data
