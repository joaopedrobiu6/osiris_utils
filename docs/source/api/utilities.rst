Utilities API
=============

This document provides a reference to the osiris_utils utilities API.

Data Readers Structures
-----------------------

The package provides several classes for handling different types of OSIRIS data.

OsirisData
~~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisData
   :members: dt, dim, time, iter, name, type, verbose
   :special-members: __init__
   :noindex:
   
   Base class for all OSIRIS data types. All other data classes inherit from this class.
   
   **Key Attributes:**
   
   * ``dt`` - Time step of the simulation
   * ``dim`` - Dimensionality of the data
   * ``time`` - Physical time of the data
   * ``iter`` - Iteration number
   * ``name`` - Name of the dataset
   * ``type`` - Type of the data
   * ``verbose`` - Verbosity flag for logging

Grid-Based Data
~~~~~~~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisGridFile
   :members: grid, nx, dx, x, axis, data, units, label
   :inherited-members:

   Specialized class for handling grid-based field data such as electromagnetic fields.
   
   **Key Attributes:**
   
   * ``grid`` - Grid information
   * ``nx`` - Number of grid points
   * ``dx`` - Grid spacing
   * ``x`` - Grid coordinates
   * ``axis`` - Coordinate labels
   * ``data`` - The actual field data
   * ``units`` - Physical units of the data
   * ``label`` - Data labels for visualization

Particle Data
~~~~~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisRawFile  
   :members: raw_to_file_tags  
   :inherited-members:  

   **Description:**  
   This class is responsible for reading and structuring raw particle data from an OSIRIS HDF5 file. It provides direct access to raw simulation data, in a structured dictionary format.  

   **Key Attributes:**  

   * ``axis`` - Dictionary where each key is a dataset name, and each value is another dictionary containing
            name (str): The name of the quantity (e.g., r'x1', r'ene').
            units (str): The units associated with that dataset in LaTeX (e.g., r'c/\\omega_p', r'm_e c^2').
            long_name (str): The name of the quantity in LaTeX (e.g., r'x_1', r'En2').
            dictionary of dictionaries
   * ``data`` - Dictionary mapping dataset names to their corresponding NumPy arrays.  
   * ``dim`` - Number of spatial dimensions in the simulation.  
   * ``dt`` - Simulation time step.  
   * ``grid`` - Array specifying the min/max coordinates of the simulation box along each axis.  
   * ``iter`` - Iteration number of the simulation snapshot.  
   * ``name`` - Name of the particle species in the dataset.  
   * ``time`` - List containing the simulation time and its associated units.  
   * ``type`` - Type of the dataset (e.g., "particles" for raw particle data).  

   **Example Usage:**

   .. code-block:: python
  
      import osiris_utils as ou  
      raw = ou.raw = ou.OsirisRawFile("path/to/raw/file.h5")
      print(raw.data.keys())
      print(raw.data["x1"][0:10])  # Access x1 position of first 10 particles

   **Methods:**  

   * ``raw_to_file_tags(filename, type="all", n_tags=10, mask=None)``  

     - Converts raw particle data into a `file_tags` format, used for selecting particles in OSIRIS tracking diagnostics.  

     - **Parameters:**  

       - ``filename`` (str): Path to the output tag file.  
       - ``type`` (str, optional): Selection mode (`"all"` for all tags, `"random"` for a random subset).  
       - ``n_tags`` (int, optional): Number of tags to select when `type="random"`. Default is 10.  
       - ``mask`` (np.ndarray, optional): Boolean mask to filter valid particle tags before selection.  

     - **Output:**  

       - Generates a file storing selected particle tags to initialize OSIRIS track diagnostic.  
      
     - **Example Usage:**

      .. code-block:: python

         raw = ou.OsirisRawFile("path/to/raw/file/.../.h5")
         # Selecting 5 random tags from particles with energy>5
         mask = raw.data["ene"] >    5.
         raw_to_file_tags("output.tag", type="random", n_tags=5, mask=mask)



HIST Data
~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisHIST
   :inherited-members: grid, nx, dx, x, axis, data, units, label

   Processes HIST file from OSIRIS diagnostics.

TRACK Data
~~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisTrackFile  
   :members: grid, data, units, labels, quants, num_particles, num_time_iters, __array__, __str__  
   :inherited-members:  

   Specialized class for handling particle track data from OSIRIS HDF5 simulations. This class processes and organizes particle tracking data into a structured numpy array, providing an intuitive interface for accessing particle properties over time.  

   **Key Attributes:**  

   * ``grid`` - Grid boundaries of the simulation  
   * ``data`` - Structured numpy array with tracked particle data, accessed as ``data[particle, time_iter][quant]``  
   * ``units`` - List of units corresponding to each tracked field  
   * ``labels`` - LaTeX-formatted labels for each tracked quantity  
   * ``quants`` - Field names of the tracked data  
   * ``num_particles`` - Number of tracked particles, indexed from ``0`` to ``num_particles - 1``  
   * ``num_time_iters`` - Number of time iterations, indexed from ``0`` to ``num_time_iters - 1``  

   **Methods:**  

   * ``__array__()`` - Returns the ``data`` attribute as a standard numpy array.  
   * ``__str__()`` - Provides a string summary of the object, including simulation grid, iteration, and dimensions.  
   * ``_load_basic_attributes(f: h5py.File)`` - Loads essential attributes such as simulation time step (``dt``), dimensionality (``dim``), and file metadata.  

   **Example Usage:**

   .. code-block:: python
  
      import osiris_utils as ou
      track = ou.OsirisTrackFile(path/to/track_file.h5)
      print(track.data[0:10, :]["x1"]) # Access x1 position of first 10 particles over all time steps


Convert track file to the older more readable format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: convert_tracks(filename_in)

   **Description:**  
   Converts a new OSIRIS track file aka IDL-formatted aka tracks-2 to an older format that is more human-readable.  
   In the old format, each particle is stored in a separate folder, with datasets for each quantity.  
   The function reads data from the input file, processes it, and writes it to a new file with the suffix "-v2".

   **Parameters:**  
   - ``filename_in`` (str): The path to the input IDL-formatted track file.  

   **Output File:**  
   - A new HDF5 file is created with the same name as the input file, but with the suffix `-v2` added to the filename.  
   This file will be in the old, more readable format.

   **Example:**  

   .. code-block:: python
      
      >>> import osiris_utils as ou 
      >>> ou.utils.convert_tracks('path/to/input_trackfile.h5')
      >>> # The output will be saved as 'path/to/input_trackfile-v2.h5'

   **Notes:**  
   - The old format stores each particle's data in separate groups within the file.
   - This function assumes the input file follows the IDL (new) format as expected by OSIRIS.
   - Code from https://github.com/GoLP-IST/RaDi-x/blob/main/tools/convert_idl_tracks_py3.py 

Create `file_tags` file for track diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a tag file directly from raw data, see :class:`osiris_utils.data.data.OsirisRawFile.raw_to_file_tags()`.

.. function:: create_file_tags(filename, tags_array)

   **Description:**  
   Writes a `file_tags` file from a NumPy array containing particle tags.  
   The file is used to select specific particles for the OSIRIS track diagnostic.  
   The tags file includes the particle ID and the associated tag value.

   **Parameters:**  
   - ``filename`` (str): The path to the output `file_tags` file where the tags will be saved.  
   - ``tags_array`` (np.ndarray): A 2D NumPy array with shape `(number_of_tags, 2)`, with the tags.  

   **Output:**  
   - A `file_tags` file will be written to the specified `filename`, which can be used in OSIRIS diagnostics for particle tracking.

   **Example:**  

   .. code-block:: python
      
      import osiris_utils as ou 
      import numpy as np
      tags = np.array([[1, 12345], [2, 67890], [3, 11111]])  # Example tags
      ou.utils.create_file_tags('output.tag', tags)
      # This will generate a file 'output.tag' with the particle tags.

   **Notes:**  
   - The function generates the `file_tags` file in a format that can be used by the OSIRIS track diagnostic.

Physics & Analysis
------------------

Numerical Methods
~~~~~~~~~~~~~~~~~

.. automodule:: osiris_utils.utils
   :members: courant2D, transverse_average
   :noindex:

   Methods for common physics calculations:
   
   * ``courant2D`` - Calculate the Courant condition for 2D simulations
   * ``transverse_average`` - Compute averages along transverse directions

Data Utilities
-------------- 

File Operations
~~~~~~~~~~~~~~~

.. automodule:: osiris_utils.utils
   :members: 
       time_estimation,
       filesize_estimation,
       integrate,
       save_data,
       read_data
   :noindex:

   Utilities for data handling and file operations:
   
   * ``time_estimation`` - Estimate runtime for operations
   * ``filesize_estimation`` - Estimate file sizes
   * ``integrate`` - Numerical integration routines
   * ``save_data`` - Save data to disk
   * ``read_data`` - Read data from disk