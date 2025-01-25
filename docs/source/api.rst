API Reference
=============

Core Modules
------------

data_readers Module
~~~~~~~~~~~~~~~~~~~
.. automodule:: your_package.data_readers
   :members: create_dataset, open1D, open2D, open3D, read_osiris_file
   :show-inheritance:

data Module (Structures)
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: your_package.data
   :members: OsirisGridFile
   :special-members: __init__

GUI Module
----------
.. autoclass:: your_package.gui.LAVA
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. automethod:: __init__

Utilities Module
----------------
.. automodule:: your_package.utils
   :members: 
    - time_estimation
    - filesize_estimation
    - transverse_average
    - integrate
    - animate_2D
    - save_data
    - read_data
    - mft_decomposition
    - courant2D
   :show-inheritance:

Special Functions
-----------------
animate_2D
~~~~~~~~~~
.. autofunction:: your_package.utils.animate_2D

courant2D
~~~~~~~~~
.. autofunction:: your_package.utils.courant2D

Data Processing
---------------
mft_decomposition
~~~~~~~~~~~~~~~~~
.. autofunction:: your_package.utils.mft_decomposition

transverse_average
~~~~~~~~~~~~~~~~~~
.. autofunction:: your_package.utils.transverse_average