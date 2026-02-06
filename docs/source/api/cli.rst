Command Line Interface
======================

The ``osiris_utils`` package provides a command-line interface (CLI) tool named ``utils`` for quick analysis and management of OSIRIS simulations.

Usage
-----

.. code-block:: bash

    utils <command> [options] arguments

Available Commands
------------------

info
~~~~

Display metadata about OSIRIS files or simulations.

.. code-block:: bash

    # Show simulation metadata
    utils info path/to/input.deck

    # Show single file info
    utils info path/to/file.h5

    # Show brief summary
    utils info path/to/input.deck --brief

validate
~~~~~~~~

Check integrity of OSIRIS data files and simulation structure.

.. code-block:: bash

    # Validate entire simulation
    utils validate path/to/input.deck

    # Validate single file
    utils validate file.h5

    # Check for missing timesteps
    utils validate path/to/input.deck --check-missing

export
~~~~~~

Export OSIRIS data to other formats (CSV, JSON, NumPy).

.. code-block:: bash

    # Export to CSV
    utils export file.h5 --format csv --output data.csv

    # Export to NumPy
    utils export file.h5 --format npy --output data.npy

plot
~~~~

Create quick visualizations of OSIRIS data.

.. code-block:: bash

    # Save plot to file
    utils plot file.h5 --save plot.png

    # Interactive display
    utils plot file.h5 --display

Notes
-----

- The CLI uses the input deck path (e.g., ``input.deck``, ``os-stdin``) to identify simulations, rather than the directory path.
- The command name is ``utils`` (previously ``osiris``).
