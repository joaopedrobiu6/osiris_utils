import jax.numpy as jnp
import h5py  

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

def open1D_jax(filename):
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
    attrs, axes, data = read_osiris_file_jax(filename)
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
    return x, y, data_array, [attrs, axes, data]

def open3D_jax(filename):
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
    attrs, axes, data = read_osiris_file_jax(filename)
    datash = data.shape
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    x = jnp.linspace(ax1[0], ax1[1], datash[-1])
    y = jnp.linspace(ax2[0], ax2[1], datash[-2])
    z = jnp.linspace(ax3[0], ax3[1], datash[-3])
    data_array = jnp.array(data[:])
    return x, y, z, data_array, [attrs, axes, data]