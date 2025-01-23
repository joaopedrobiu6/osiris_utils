'''
This file contains methods to create datasets to train the neural network.
'''

import numpy as np
from .utils import *
import os
import h5py

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

def create_dataset(folder, pressure = False):
    data_list = []
    means = []
    fluctuations = []
    for filename in os.listdir(folder):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder, filename)
            x, y, data, _ = open2D(filepath, pressure=pressure)  # Adjust the parameters as needed
            data_mean = transverse_average(data)
            data_fluctuations = data - data_mean
            data_list.append(data)
            means.append(data_mean)
            fluctuations.append(data_fluctuations)
    return np.array(data_list), np.array(means), np.array(fluctuations)

