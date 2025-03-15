import numpy as np
import pandas as pd
import h5py

class OsirisData():
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
        self._filename = filename
        # self._file = None

        self._verbose = False

        if self._filename.endswith('.h5'):
            self._open_file_hdf5(self._filename)
            self._load_basic_attributes(self._file)
        elif self._filename.endswith('_ene'):
            self._open_hist_file(self._filename)
        else:
            raise ValueError('The file should be an HDF5 file with the extension .h5, or a HIST file ending with _ene.')
        
        
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

    def _open_file_hdf5(self, filename):
        '''
        Open the OSIRIS output file. Usually an HDF5 file or txt.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        '''
        if self._verbose: print(f'Opening file > {filename}')

        if filename.endswith('.h5'):
            self._file = h5py.File(filename, 'r')
        else:
            raise ValueError('The file should be an HDF5 file with the extension .h5')
            
    def _open_hist_file(self, filename):
        self._df = pd.read_csv(filename, sep=r'\s+', comment='!', header=0, engine='python')

    def _close_file(self):
        '''
        Close the HDF5 file.
        '''
        if self._verbose: print('Closing file')
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
            raise ValueError('The FFT of the data has not been computed yet. Compute it using the FFT method.')
        return self._FFTdata
    # Setters
    @data.setter
    def data(self, data):
        self._data = data

    def __str__(self):
        # write me a template to print with the name, label, units, time, iter, grid, nx, dx, axis, dt, dim in a logical way
        return rf'{self.name}' + f'\n' + rf'Time: [{self.time[0]} {self.time[1]}], dt = {self.dt}' + f'\n' + f'Iteration: {self.iter}' + f'\n' + f'Grid: {self.grid}' + f'\n' + f'dx: {self.dx}' + f'\n' + f'Dimensions: {self.dim}D'
    

    def __array__(self):
        return np.asarray(self.data)


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

class OsirisHIST(OsirisData):
    ''''
    Class to read the data from an OSIRIS HIST file.'

    Input:
        - filename: the path to the HIST file

    Attributes:
        - filename - the path to the file
            str
        - verbose - if True, the class will print messages
            bool
        - df - the data in a pandas DataFrame
            pandas.DataFrame
    '''
    def __init__(self, filename):
        super().__init__(filename)

    @property
    def df(self):
        """
        Returns the data in a pandas DataFrame
        """
        return self._df
