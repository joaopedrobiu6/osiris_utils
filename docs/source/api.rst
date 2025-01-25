API Reference
=============

Core Modules
------------

data_readers Module
~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.data_readers
   :members: create_dataset, open1D, open2D, open3D, read_osiris_file
   :show-inheritance:

data Module (Structures)
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: osiris_utils.data
   :members: OsirisGridFile
   :special-members: __init__
   :inherited-members:
   :undoc-members:

GUI Module
----------
.. autoclass:: osiris_utils.gui.LAVA
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. automethod:: __init__

Utilities Module
----------------
.. automodule:: osiris_utils.utils
   :members: 
       time_estimation,
       filesize_estimation,
       transverse_average,
       integrate,
       animate_2D,
       save_data,
       read_data,
       mft_decomposition,
       courant2D
   :show-inheritance:

Special Functions
-----------------
animate_2D
~~~~~~~~~~
.. autofunction:: osiris_utils.utils.animate_2D

courant2D
~~~~~~~~~
.. autofunction:: osiris_utils.utils.courant2D

Data Processing
---------------
mft_decomposition
~~~~~~~~~~~~~~~~~
.. autofunction:: osiris_utils.utils.mft_decomposition

transverse_average
~~~~~~~~~~~~~~~~~~
.. autofunction:: osiris_utils.utils.transverse_average