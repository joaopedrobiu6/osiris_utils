'''
This file contains methods to create datasets to train the neural network.
'''

import numpy as np
from .utils import *
import os

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

