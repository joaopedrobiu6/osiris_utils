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
   :members: data, axis
   :inherited-members: grid, nx, dx, x, axis, data, units, label

   Handles particle (raw) data files from OSIRIS simulations.
   
   **Key Attributes:**
   
   * ``data`` - Particle data array
   * ``axis`` - Coordinate labels
   * Inherits grid-related attributes from OsirisData

HIST Data
~~~~~~~~~

.. autoclass:: osiris_utils.data.data.OsirisHIST
   :members: df
   :inherited-members: grid, nx, dx, x, axis, data, units, label

   Processes HIST file from OSIRIS diagnostics.
   
   **Key Attributes:**
   
   * ``df`` - DataFrame containing the data

Visualization & GUI
-------------------

Interactive Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: osiris_utils.gui.gui.LAVA
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   LAVA (Lightweight Analysis and Visualization Application) provides interactive 
   visualization capabilities for OSIRIS data.

Animation Tools
~~~~~~~~~~~~~~~

.. automodule:: osiris_utils.utils
   :members: animate_2D
   :noindex:

   Tools for creating animations from simulation data:
   
   * ``animate_2D`` - Creates animations from 2D field data

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