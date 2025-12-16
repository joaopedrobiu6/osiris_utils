from __future__ import annotations

from datetime import datetime
from typing import cast

import h5py
import numpy as np
import pandas as pd
import scipy


def courant2D(dx: float, dy: float) -> float:
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
    dt = 1 / (np.sqrt(1 / dx**2 + 1 / dy**2))
    return cast(float, dt)


def time_estimation(n_cells: int, ppc: int, t_steps: int, n_cpu: int, push_time: float = 1e-7, hours: bool = False) -> float:
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
    time = (n_cells * ppc * push_time * t_steps) / n_cpu
    if hours:
        return time / 3600
    else:
        return time


def filesize_estimation(n_gridpoints: int) -> float:
    return n_gridpoints * 4 / (1024**2)


def transverse_average(data: np.ndarray) -> np.ndarray:
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
    return cast(np.ndarray, np.mean(data, axis=1))


def integrate(array: np.ndarray, dx: float) -> np.ndarray:
    """
    Integrate a 1D from the left to the right. This may be changed in the future to allow
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
        raise ValueError(f"Array must be 1D\n Array shape: {array.shape}")
    flip_array = np.flip(array)
    # int = -scipy.integrate.cumulative_trapezoid(flip_array, dx = dx, initial = flip_array[0])
    int = -scipy.integrate.cumulative_simpson(flip_array, dx=dx, initial=0)
    return cast(np.ndarray, np.flip(int))


def save_data(data: np.ndarray, savename: str, option: str = "numpy") -> None:
    """
    Save the data to a .txt (with Numpy) or .csv (with Pandas) file.

    Parameters
    ----------
    data : list
        The data to be saved.
    savename : str
        The path to the file.
    option : str, optional
        The option for saving the data. The default is 'numpy'. Can be 'numpy' or 'pandas'.
    """
    if option == "numpy":
        np.savetxt(savename, data)
    elif option == "pandas":
        pd.DataFrame(data).to_csv(savename, index=False)
    else:
        raise ValueError("Option must be 'numpy' or 'pandas'.")


def read_data(filename: str, option: str = "numpy") -> np.ndarray:
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


def convert_tracks(filename_in: str) -> str:
    """
    Converts a new OSIRIS track file aka IDL-formatted aka tracks-2 to an older format that is more human-readable.
    In the old format, each particle is stored in a separate folder, with datasets for each quantity.
    The function reads data from the input file, processes it, and writes it to a new file with the suffix "-v2".

     code from https://github.com/GoLP-IST/RaDi-x/blob/main/tools/convert_idl_tracks_py3.py

     Parameters
     ----------
     filename_in : str
         The path to the trackfile.

     Returns
     -------
     The output file will be in the same folder as the input file with the same name with \"v2\" added

    """

    try:
        file_in = h5py.File(filename_in, "r")
    except IOError:
        print("cannot open " + filename_in)
        exit()

    # read data from file
    data = file_in["data"][:]
    itermap = file_in["itermap"][:]
    ntracks = file_in.attrs["NTRACKS"][0]
    niter = file_in.attrs["NITER"][0]
    quants = file_in.attrs["QUANTS"][:]
    file_in_attr_keys = file_in.attrs.keys()
    sim_attr_keys = file_in["SIMULATION"].attrs.keys()
    nquants = len(quants)

    # construct file out for new format
    filename_out = filename_in[:-3] + "-v2" + filename_in[-3:]
    file_out = h5py.File(filename_out, "w")

    # copy attrs from file_in
    for item in file_in_attr_keys:
        file_out.attrs[item] = file_in.attrs[item]
    for item in sim_attr_keys:
        file_out.attrs[item] = file_in["SIMULATION"].attrs[item]

    # first pass -- find total size of each track
    # ----------------------------------------#
    sizes = np.zeros(ntracks)

    itermapshape = itermap.shape
    for i in range(itermapshape[0]):
        part_number, npoints, nstart = itermap[i, :]
        sizes[part_number - 1] += npoints

    # initialize ordered data buffer
    # ----------------------------------------#
    ordered_data = []
    for i in range(ntracks):
        ordered_data.append(np.zeros((int(sizes[i]), nquants)))
    # ----------------------------------------#

    # assign tracks to ordered data from file_in data
    # ----------------------------------------#
    track_indices = np.zeros(ntracks)
    data_index = 0

    for i in range(itermapshape[0]):
        part_number, npoints, nstart = itermap[i, :]
        track_index = int(track_indices[part_number - 1])

        ordered_data[part_number - 1][track_index : track_index + npoints, 0] = nstart + np.arange(npoints) * niter

        ordered_data[part_number - 1][track_index : track_index + npoints, 1:] = data[data_index : data_index + npoints, :]

        data_index += npoints
        track_indices[part_number - 1] += npoints

    # ----------------------------------------#

    # write to file out
    for i in range(ntracks):
        group = file_out.create_group(str(i + 1))
        for j in range(nquants):
            if j == 0:
                group.create_dataset(quants[j], data=np.array(ordered_data[i][:, j], dtype=int))
            else:
                group.create_dataset(quants[j], data=ordered_data[i][:, j])

    file_out.close()
    file_in.close()
    print("Track file converted to the old, more readable format: ", filename_out)
    return filename_out


def create_file_tags(filename: str, tags_array: np.ndarray) -> str:
    """
    Function to write a file_tags file from a (number_of_tags, 2) NumPy array of tags.
    this file is used to choose particles for the OSIRIS track diagnostic.

    Parameters
    ----------
    filename : str
        Path to the output file where tags will be stored.
    tags_array: np.ndarray
        shape (number_of_tags, 2), containing particle tags

    Returns
    -------
    file_tags file with path \"filename\" to be used for the OSIRIS track diagnostic.

    Notes
    ------
    The first element of the tag of a particle that is already being tracked is negative,
        so we apply the absolute function when generating the file

    """

    # In case the particles chosen were already being tracked
    tags_array[:, 0] = np.abs(tags_array[:, 0])
    tags_array = tags_array[np.lexsort((tags_array[:, 1], tags_array[:, 0]))]
    num_tags = tags_array.shape[0]

    with open(filename, "w") as file:
        file.write("! particle tag list\n")
        file.write(f"! generated on {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}\n")
        file.write("! number of tags\n")
        file.write(f"       {num_tags}\n")
        file.write("! particle tag list\n")

        for i in range(num_tags):
            file.write(f"         {tags_array[i, 0]:<6}{tags_array[i, 1]:>6}\n")
    return filename
