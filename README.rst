OSIRIS_UTILS
============
|Pypi|

.. image:: https://github.com/joaopedrobiu6/osiris_utils/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/joaopedrobiu6/osiris_utils/actions
   :alt: CI status
.. image:: https://codecov.io/gh/joaopedrobiu6/osiris_utils/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/joaopedrobiu6/osiris_utils
   :alt: Coverage



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
   git clone https://github.com/joaopedrobiu6/osiris_utils  # sample data
   cd osiris_utils
   python examples/quick_start.py examples/example_data/thermal.1d

Documentation
-------------

The documentation is available at https://osiris-utils.readthedocs.io or via this link: `osiris-utils.readthedocs.io <https://osiris-utils.readthedocs.io>`_.

.. |Pypi| image:: https://img.shields.io/pypi/v/osiris-utils
    :target: https://pypi.org/project/osiris-utils/
    :alt: Pypi

.. _authors:

Authors and Contributors
------------------------

- João Biu