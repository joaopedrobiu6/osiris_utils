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

Data Structures Module
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.data
   :members: 
       OsirisData, 
       OsirisGridFile,
       OsirisRawFile,
       OsirisHIST
   :special-members: __init__
   :exclude-members: 
       - FFTdata
       - yeeToCellCorner
       - FFT
   :noindex:

   Base classes for handling OSIRIS data structures and grid files.

   **Inherited Members** (from OsirisData):
   :members: dim, dt, iter, name, time, type, verbose

Mean Field Theory Module
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: osiris_utils.mean_field_theory.MeanFieldTheory
   :members: average, delta, derivative
   :inherited-members:
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