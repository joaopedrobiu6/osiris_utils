API Reference
=============

Core Data Handling
------------------

Data Readers Module
~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.data_readers
   :members: open1D, open2D, open3D, read_osiris_file
   :show-inheritance:
   :noindex:

   Functions for reading OSIRIS data files in 1D, 2D, and 3D formats.

Core Data Structures
--------------------

OsirisData (Base Class)
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: osiris_utils.data.OsirisData
   :members: dt, dim, time, iter, name, type, verbose
   :special-members: __init__
   :noindex:
   
   Base class for all OSIRIS data types. Provides common attributes and methods for handling simulation metadata.

OsirisGridFile
~~~~~~~~~~~~~~
.. autoclass:: osiris_utils.data.OsirisGridFile
   :members: grid, nx, dx, x, axis, data, units, label
   :inherited-members:

   Specialized class for grid-based field data. Inherits from :class:`OsirisData`.

OsirisRawFile
~~~~~~~~~~~~~
.. autoclass:: osiris_utils.data.OsirisRawFile
   :members: data, axis
   :inherited-members: grid, nx, dx, x, axis, data, units, label

   Handles particle/raw data files. Inherits from :class:`OsirisData`.

OsirisHIST
~~~~~~~~~~
.. autoclass:: osiris_utils.data.OsirisHIST
   :members: df
   :inherited-members: grid, nx, dx, x, axis, data, units, label

   Processes HIST file time series data. Inherits from :class:`OsirisData`.

Mean Field Theory Module
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: osiris_utils.mean_field_theory.MeanFieldTheory
   :members: average, delta, derivative
   :inherited-members: grid, nx, dx, x, axis, data, units, label
   :noindex:

   Tools for calculating mean field quantities from simulation data.

Visualization & GUI
-------------------

GUI Component
~~~~~~~~~~~~~
.. autoclass:: osiris_utils.gui.LAVA
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Graphical user interface components for data visualization.

Visualization Utilities
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.utils
   :members: animate_2D
   :noindex:

   .. autofunction:: animate_2D

   Function for creating 2D animations from simulation data.

Physics & Analysis
------------------

Physical Calculations
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.utils
   :members: courant2D, transverse_average
   :noindex:

   .. autofunction:: courant2D
   .. autofunction:: transverse_average

   Core physics calculations and analysis routines.

Data Utilities
--------------

Data I/O & Processing
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.utils
   :members: 
       time_estimation,
       filesize_estimation,
       integrate,
       save_data,
       read_data
   :noindex:

   .. autofunction:: integrate
   .. autofunction:: save_data
   .. autofunction:: read_data

   Essential utilities for data processing and file operations.