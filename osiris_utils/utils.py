import numpy as np
import h5py  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import scipy 
import pandas as pd

def read_osiris_file(filename, pressure = False):
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
    if "SIMULATION" in k and pressure == False:
        data = f.get([key for key in f.keys()][2])
        data.attrs["UNITS"] = attrs1["UNITS"]
        data.attrs["LONG_NAME"] = attrs1["LABEL"]
    elif "SIMULATION" in k and pressure == True:
        data = f.get([key for key in f.keys()][1])
        data.attrs["UNITS"] = attrs1["UNITS"]
        data.attrs["LONG_NAME"] = attrs1["LABEL"]
    else:
        data = f.get([key for key in f.keys()][1])
    
    return attrs, axis, data

def open1D(filename, pressure = False):
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
    attrs, axes, data = read_osiris_file(filename, pressure)
    datash = data.shape
    ax1 = axes[0]
    x = np.linspace(ax1[0], ax1[1], datash[0])
    data_array = data[:]
    return x, data_array, [attrs, axes, data]

def open2D(filename, pressure = False):
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
    attrs, axes, data = read_osiris_file(filename, pressure)
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

def transverse_average(data):
    """
    Computes the transverse average of a 2D array.
    
    Parameters
    ----------
    data : numpy.ndarray
        Dim: 2D.
        The input data.
        
    Returns
    -------
    numpy.ndarray
        Dim: 1D.
        The transverse average.

    """

    if len(data.shape) != 2:
        raise ValueError("The input data must be a 2D array.")
    return np.mean(data, axis = 0)

def integrate(array, dx):
    """
    Integrate an the tranverse average from the left to the right. This may be changed in the future to allow 
    for integration in both directions or for other more general cases.

    Parameters
    ----------
    array : numpy.ndarray
        Dim: 1D.
        The input array.
    dx : float
        The spacing between points.

    Returns
    -------
    numpy.ndarray
        Dim: 1D.
        The integrated array.
    """

    if len(array.shape) != 1:
        raise ValueError(f"Array must be 1D\nFaz a transverse average antes de integrar...")
    flip_array = np.flip(array)
    int = -scipy.integrate.cumulative_trapezoid(flip_array, dx = dx, initial=0)
    return np.flip(int)

def compare_LHS_RHS(LHS, RHS, x, dx, **kwargs):
    """
    Compare the left hand side of the equation with the right hand side of the equation.

    Parameters
    ----------
    LHS : numpy.ndarray
        Dim: 1D.
        The left hand side of the equation.
    RHS : numpy.ndarray
        Dim: 1D.
        The right hand side of the equation.
    x : numpy.ndarray
        Dim: 1D.
        The x axis.
    dx : float
        The spacing between points.
    kwargs : dict
        Additional keyword arguments for plotting.
    """
    if len(LHS.shape) != 2 or len(RHS.shape) != 2:
        raise ValueError(f"LHS and RHS must be 2D")
    if len(LHS) != len(RHS):
        raise ValueError("LHS and RHS must have the same length.")
    LHS_avg = tranverse_average(LHS)
    RHS_avg = tranverse_average(RHS)
    LHS_int = integrate(LHS_avg, dx)
    RHS_int = integrate(RHS_avg, dx)
    plt.plot(x, LHS_int, label='LHS', **kwargs)
    plt.plot(x, RHS_int, label='RHS', **kwargs)
    plt.legend()
    plt.show()

def animate_2D(datafiles, frames, interval, fps, savename, **kwargs):
    """
    Animate 2D OSIRIS files.

    Parameters
    ----------
    datafiles : str
        The path to the files.
        Must be of the type "path/to/file_%06d.h5".
    kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation.
    """
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    im = 0

    def animate(i):
        # Assuming this returns (x, y, data, out) correctly
        x, y, data, _ = open2D(datafiles % i)
        ax.clear()
        # Display image data, make sure data shape is valid for imshow
        im = ax.imshow(data, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', origin='lower', **kwargs)

    # Creating the animation, and frames should be updated accordingly
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval)

    # Save the animation as a GIF (you can set the path and filename)
    ani.save(savename, writer='pillow', fps=fps)  # fps can be adjusted

    # Display the animation
    return ani

def save_data(x, LHS, RHS, savename, option="numpy"):
    """
    Save the data to a .txt file.

    Parameters
    ----------
    x : numpy.ndarray
        Dim: 1D.
        The x axis.
    LHS : numpy.ndarray
        Dim: 1D.
        The left hand side of the equation.
    RHS : numpy.ndarray
        Dim: 1D.
        The right hand side of the equation.
    """
    if option == "numpy":
        np.savetxt(savename, np.array([x, LHS, RHS]).T, header="x LHS RHS")
    elif option == "pandas":
        df = pd.DataFrame({"x": x, "LHS": LHS, "RHS": RHS})
        df.to_csv(savename, index=False)

def read_data(filename, option="numpy"):
    """
    Read the data from a .txt file.

    Parameters
    ----------
    filename : str
        The path to the file.

    Returns
    -------
    numpy.ndarray
        Dim: 2D.
        The data.
    """
    return np.loadtxt(filename) if option == "numpy" else pd.read_csv(filename).values

def mft_decomposition(filename, pressure =  False, xy = False):
    """
    Mean Field Theory decomposition of the data.
    Considering that A = ⟨A⟩ + δA with ⟨δA⟩ = 0
    This function returns ⟨A⟩ and δA from A. 
    
    Parameters
    ----------
    filename : str
        The path to the file.
        The data is 2D.
    pressure : bool, optional
        If True, the file is a pressure file. The default is False.
        
    Returns
    -------
    mean : numpy.ndarray
        Dim: 1D.
        The mean field ⟨A⟩.
    fluctuation : numpy.ndarray
        Dim: 2D.
        The fluctuation δA.
    """ 
    x, y, data, _ = open2D(filename, pressure)
    mean = transverse_average(data)
    fluctuation = data - mean
    if xy:
        return x, y, mean, fluctuation
    else:
        return mean, fluctuation