import numpy as np
import h5py  

def read_osiris_file(filename):
    f = h5py.File(filename, 'r+')
    atr = f.attrs
    k = [key for key in f.keys()]
    if "SIMULATION" in k:
        attrs1 = atr
        attrs2 = f["SIMULATION"].attrs
        attrs = {}
        for i in range(len(attrs1)):
            attrs[[key for key in attrs1][i]] = [value for value in attrs1.values()][i]
        for i in range(len(attrs2)):
            attrs[[key for key in attrs2][i]] = [value for value in attrs2.values()][i]
    ax = f.get([key for key in f.keys()][0])
    leanx = len(ax)
    axis = []
    for i in range(leanx):
        axis.append(ax.get([key for key in ax.keys()][i]))
    if "SIMULATION" in k:
        data = f.get([key for key in f.keys()][2])
        data.attrs["UNITS"] = attrs1["UNITS"]
        data.attrs["LONG_NAME"] = attrs1["LABEL"]
    else:
        data = f.get([key for key in f.keys()][1])
    
    return attrs, axis, data

def open1D(filename):
    """ 
    Open a 1D OSIRIS file and return the x axis and the data array.

    Parameters
    ----------
    filename : str
        The path to the file.
    
    Returns
    -------
    x : numpy.ndarray
        The x axis.
    data_array : numpy.ndarray
        The data array.
    """
    attrs, axes, data = read_osiris_file(filename)
    datash = data.shape
    ax1 = axes[0]
    x = np.linspace(ax1[0], ax1[1], datash[0])
    data_array = data[:]
    return x, data_array, [attrs, axes, data]

def open2D(filename):
    """
    Open a 2D OSIRIS file and return the x and y axes and the data array.

    Parameters
    ----------
    filename : str
        The path to the file.
    
    Returns
    -------
    x : numpy.ndarray
        The x axis.
    y : numpy.ndarray
        The y axis.
    data_array : numpy.ndarray
        The data array.
    """
    attrs, axes, data = read_osiris_file(filename)
    datash = data.shape
    ax1 = axes[0]
    ax2 = axes[1]
    x = np.linspace(ax1[0], ax1[1], datash[-1])
    y = np.linspace(ax2[0], ax2[1], datash[-2])
    data_array = data[:]
    return x, y, data_array, [attrs, axes, data]

def open3D(filename):
    """
    Open a 3D OSIRIS file and return the x, y and z axes and the data array.

    Parameters
    ----------
    filename : str
        The path to the file.

    Returns
    -------
    x : numpy.ndarray
        The x axis.
    y : numpy.ndarray
        The y axis.
    z : numpy.ndarray
        The z axis.
    data_array : numpy.ndarray
        The data array.
    """
    attrs, axes, data = read_osiris_file(filename)
    datash = data.shape
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    x = np.linspace(ax1[0], ax1[1], datash[-1])
    y = np.linspace(ax2[0], ax2[1], datash[-2])
    z = np.linspace(ax3[0], ax3[1], datash[-3])
    data_array = data[:], [attrs, axes, data]
    return x, y, z, data_array

def time_estimation(n_cells, ppc, push_time, t_steps, n_cpu, hours = False):
    """
    Estimate the simulation time.

    Parameters
    ----------
    n_cells : int
        The number of cells.
    ppc : int
        The number of particles per cell.
    push_time : float
        The time per push.
    t_steps : int
        The number of time steps.
    n_cpu : int
        The number of CPU's.
    hours : bool, optional
        If True, the output will be in hours. If False, the output will be in seconds. The default is False.

    Returns
    -------
    float
        The estimated time in seconds or hours.
    """
    time = (n_cells*ppc*push_time*t_steps)/n_cpu
    if hours:
        return time/3600
    else:
        return time
    
def filesize_estimation(n_gridpoints):
    return n_gridpoints*4/(1024**2)