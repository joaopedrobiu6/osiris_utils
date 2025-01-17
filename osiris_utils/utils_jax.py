import jax.numpy as jnp
from jax import jit
import h5py  
import quadax


def courant2D_jax(dx, dy):
    """
    Compute the Courant number for a 2D simulation.

    Parameters
    ----------
    dx : float
        The spacing in the x direction.
    dy : float
        The spacing in the y direction.

    Returns
    -------
    float
        The limit for dt.
    """
    dt = 1/(jnp.sqrt(1/dx**2 + 1/dy**2))
    return dt

def read_osiris_file_jax(filename, pressure = False):
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
    
    return attrs, axis, jnp.array(data)

def open1D_jax(filename, pressure = False):
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
    attrs, axes, data = read_osiris_file_jax(filename, pressure)
    datash = data.shape
    ax1 = axes[0]
    x = jnp.linspace(ax1[0], ax1[1], datash[0])
    data_array = jnp.array(data[:])
    return x, data_array, [attrs, axes, data]

def open2D_jax(filename, pressure = False):
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
    attrs, axes, data = read_osiris_file_jax(filename, pressure)
    datash = data.shape
    ax1 = axes[0]
    ax2 = axes[1]
    x = jnp.linspace(ax1[0], ax1[1], datash[-1])
    y = jnp.linspace(ax2[0], ax2[1], datash[-2])
    data_array = jnp.array(data[:])
    return jnp.array(x), jnp.array(y), jnp.asarray(data_array), [attrs, axes, data]

def open3D_jax(filename, pressure = False):
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
    attrs, axes, data = read_osiris_file_jax(filename, pressure)
    datash = data.shape
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    x = jnp.linspace(ax1[0], ax1[1], datash[-1])
    y = jnp.linspace(ax2[0], ax2[1], datash[-2])
    z = jnp.linspace(ax3[0], ax3[1], datash[-3])
    data_array = jnp.array(data[:])
    return jnp.array(x), jnp.array(y), jnp.array(z), jnp.asarray(data_array), [attrs, axes, data]

def transverse_average_jax(data):
    """
    Compute the transverse average of a 2D array.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D array.

    Returns
    -------
    numpy.ndarray
        The transverse average.
    """
    if len(data.shape) != 2:
        raise ValueError("The input data must be a 2D array.")
    return jnp.mean(data, axis=0)

def integrate_jax(array, dx):
    """
    Integrate an the tranverse average from the left to the right. This may be changed in the future to allow 
    for integration in both directions or for other more general cases.

    Parameters
    ----------
    array : jax.numpy.ndarray
        Dim: 1D.
        The input array.
    dx : float
        The spacing between points.

    Returns
    -------
    jax.numpy.ndarray
        Dim: 1D.
        The integrated array.
    """
    if len(array.shape) != 1:
        raise ValueError("The input array must be 1D.")
    flip_array = jnp.flip(array)
    
    int = -jit(quadax.cumulative_trapezoid)(y=flip_array, dx=dx, initial=0)
    return jnp.flip(int)
    
def mft_decomposition_jax(filename, pressure = False, xy = False, data = False):
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
    xy : bool, optional
        If True, the function returns x and y axes. The default is False.
    data : bool, optional
        If True, the function returns the data (2D with no transformation). The default is False.
        
    Returns
    -------
    mean : numpy.ndarray
        Dim: 1D.
        The mean field ⟨A⟩.
    fluctuation : numpy.ndarray
        Dim: 2D.
        The fluctuation δA.
    x : numpy.ndarray
        Dim: 1D.
        The x axis.
    y : numpy.ndarray
        Dim: 1D.
        The y axis.
    data : numpy.ndarray
        Dim: 2D.
        The data.
    """   
    x, y, data, _ = open2D_jax(filename, pressure)
    mean = jnp.mean(data)
    fluctuation = data - mean
    if xy:
        return mean, fluctuation, x, y
    elif data:
        return mean, fluctuation, x, y, data
    else:
        return mean, fluctuation