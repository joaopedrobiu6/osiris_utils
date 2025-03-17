import numpy as np
import h5py  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy 
import pandas as pd

def courant2D(dx, dy):
    '''
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
    '''
    dt = 1 / (np.sqrt(1/dx**2 + 1/dy**2))
    return dt

def time_estimation(n_cells, ppc, push_time, t_steps, n_cpu, hours = False):
    '''
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
    '''
    time = (n_cells*ppc*push_time*t_steps)/n_cpu
    if hours:
        return time/3600
    else:
        return time
    
def filesize_estimation(n_gridpoints):
    return n_gridpoints*4/(1024**2)

def transverse_average(data):
    '''
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

    '''

    if len(data.shape) != 2:
        raise ValueError('The input data must be a 2D array.')
    return np.mean(data, axis = 1)

def integrate(array, dx):
    '''
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
    '''

    if len(array.shape) != 1:
        raise ValueError(f'Array must be 1D\n Array shape: {array.shape}')
    flip_array = np.flip(array)
    # int = -scipy.integrate.cumulative_trapezoid(flip_array, dx = dx, initial = flip_array[0])
    int = -scipy.integrate.cumulative_simpson(flip_array, dx = dx, initial = 0)
    return np.flip(int)

def animate_2D(datafiles, frames, interval, fps, savename, **kwargs):
    '''
    Animate 2D OSIRIS files.

    Parameters
    ----------
    datafiles : str
        The path to the files.
        Must be of the type 'path/to/file_%06d.h5'.
    kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation.
    '''
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    im = 0

    def animate(i):
        # Assuming this returns (x, y, data, out) correctly
        x, y, data, _ = open2D(datafiles % i)
        ax.clear()
        # Display image data, make sure data shape is valid for imshow
        im = ax.imshow(-data, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', origin='lower', **kwargs)
        plt.xlabel(r'x [c/$\omega_p$]')
        plt.ylabel(r'y [c/$\omega_p$]')
        

    # Creating the animation, and frames should be updated accordingly
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval)

    # Save the animation as a GIF (you can set the path and filename)
    ani.save(savename, writer='pillow', fps=fps)  # fps can be adjusted

    # Display the animation
    return ani

def save_data(data, savename, option='numpy'):
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
    if option == 'numpy':
        np.savetxt(savename, data)
    elif option == 'pandas':
        pd.DataFrame(data).to_csv(savename, index=False)
    else:
        raise ValueError("Option must be 'numpy' or 'pandas'.")

def read_data(filename, option='numpy'):
    '''
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
    '''
    return np.loadtxt(filename) if option == 'numpy' else pd.read_csv(filename).values