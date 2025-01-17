'''
This file contains methods to create datasets to train the neural network.
'''

import jax.numpy as jnp
import jax
from .utils_jax import *
import h5py
import os

def create_dataset(folder, pressure = False):
    x_list = []
    label_list1 = []
    label_list2 = []
    for filename in os.listdir(folder):
        if filename.endswith('.h5'):
            filepath = os.path.join(folder, filename)
            x, y, data, _ = open2D_jax(filepath, pressure=pressure)  # Adjust the parameters as needed
            data_mean = jnp.mean(data, axis=0)
            data_fluctuations = data - data_mean
            x_list.append(data)
            label_list1.append(data_mean)
            label_list2.append(data_fluctuations)
    return jnp.array(x_list), jnp.array(label_list1), jnp.array(label_list2)

