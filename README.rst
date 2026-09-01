Utils
=====

|Pypi|

.. image:: https://github.com/joaopedrobiu6/osiris_utils/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/joaopedrobiu6/osiris_utils/actions
   :alt: CI status

.. image:: https://codecov.io/gh/joaopedrobiu6/osiris_utils/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/joaopedrobiu6/osiris_utils
   :alt: Coverage

.. image:: https://zenodo.org/badge/889119723.svg
  :target: https://doi.org/10.5281/zenodo.17382244

This package contains a set of utilities to open and analyze OSIRIS output files, using Python. All the methods implemented are fully integrated with `NumPy`, and use `np.ndarray` as the main data structure.
High-level functions are provided to manipulate data from OSIRIS, from reading the data of the diagnostics, to making post-processing calculations.

All code is written in Python. To contact the dev team, please send an email to João Biu: `joaopedrofbiu@tecnico.ulisboa.pt <mailto:joaopedrofbiu@tecnico.ulisboa.pt>`_.
The full dev team can be found below in the Authors and Contributors section.

How to install it?
------------------

To install this package, you can use `pip`::

    pip install osiris_utils

To install it from source, you can clone this repository and run (in the folder containing ``setup.py``)::

    git clone https://github.com/joaopedrobiu6/osiris_utils.git
    pip install .

Finally, you can install it in editor mode if you want to contribute to the code::
    
    git clone https://github.com/joaopedrobiu6/osiris_utils.git
    pip install -e .

Quick-start
-----------

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/joaopedrobiu6/osiris_utils/main?filepath=examples%2Fquick_start.ipynb
   :alt: Launch quick-start on Binder

.. code-block:: bash

   pip install osiris_utils              # from PyPI
   python -m pip install matplotlib      # plotting backend (optional)
   git clone https://github.com/joaopedrobiu6/osiris_utils
   cd osiris_utils
   python examples/scripts/01_simulation_and_diagnostics.py --plot

That first script needs no data -- it writes a small synthetic OSIRIS run for
itself.  Point ``examples/quick_start.py`` at a run of your own instead::

   python examples/quick_start.py path/to/os-stdin

Example scripts
---------------

``examples/scripts/`` holds a runnable, annotated tour of every public part of
the package -- diagnostics and the lazy pipeline, derivatives, FFTs, mean-field
theory and spatial filters, field centering and moment corrections, RAW and
track particles, parallel export, anomalous resistivity, the database builders
and the CLI. See `examples/scripts/README.md
<https://github.com/joaopedrobiu6/osiris_utils/blob/main/examples/scripts/README.md>`_
for the full index.

They need **no data**: with no ``--sim`` they build a small synthetic OSIRIS run
with the real output layout and HDF5 schema::

   python examples/scripts/run_all.py            # every example
   python examples/scripts/04_derivatives.py     # just one
   python examples/scripts/10_anomalous_resistivity.py --plot   # and its figures

Point any of them at your own run by passing its input deck::

   python examples/scripts/01_simulation_and_diagnostics.py --sim path/to/os-stdin

Command-Line Interface
----------------------

osiris_utils includes a command-line interface for common operations. After installation, the ``utils`` command becomes available::

   utils --version                     # Check version
   utils --help                        # Show available commands

**Available Commands:**

- ``utils info`` - Display metadata about OSIRIS files and simulations
- ``utils export`` - Convert data to CSV, JSON, or NumPy formats
- ``utils plot`` - Create quick visualizations
- ``utils validate`` - Check file integrity

**Examples:**

Show simulation information::

   utils info path/to/input.deck
   utils info path/to/file.h5 --brief

Export data to different formats::

   utils export file.h5 --format csv --output data.csv
   utils export diagnostic/dir --format npy --output data.npy

Generate quick plots::

   utils plot file.h5 --save plot.png
   utils plot file.h5 --save plot.png --title "Ez Field" --cmap viridis

Validate simulation data::

   utils validate path/to/input.deck
   utils validate path/to/input.deck --check-missing

For detailed help on any command::

   utils <command> --help


Documentation
-------------

The documentation is available at https://osiris-utils.readthedocs.io or via this link: `osiris-utils.readthedocs.io <https://osiris-utils.readthedocs.io>`_.

.. |Pypi| image:: https://img.shields.io/pypi/v/osiris-utils
    :target: https://pypi.org/project/osiris-utils/
    :alt: Pypi

.. _authors:

Author
------

- João Biu

Contributors
------------

- Diogo Carvalho
- João Cândido
- Margarida Pereira

