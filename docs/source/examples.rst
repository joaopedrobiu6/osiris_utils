Examples
========

This section contains examples and tutorials on how to use ``osiris_utils``.

Runnable scripts
----------------

``examples/scripts/`` in the repository holds an annotated tour of every public
part of the package, split by topic and runnable with no data of your own --
each script builds a small synthetic OSIRIS run if you do not pass ``--sim``:

.. code-block:: bash

   python examples/scripts/run_all.py             # all of them
   python examples/scripts/04_derivatives.py      # one topic
   python examples/scripts/11_databases.py --sim path/to/os-stdin   # your run

======================================  =========================================================
Script                                  Covers
======================================  =========================================================
``01_simulation_and_diagnostics.py``    ``Simulation``, ``Species_Handler``, ``Diagnostic``
``02_single_file_readers.py``           ``OsirisGridFile``, ``OsirisRawFile``, tracks, HIST, TIMINGS
``03_input_decks.py``                   ``InputDeckIO`` and ``Species``
``04_derivatives.py``                   ``Derivative_Simulation`` / ``Derivative_Diagnostic``
``05_fft.py``                           ``FFT_Simulation`` / ``FFT_Diagnostic``
``06_mft_and_filters.py``               mean-field theory and the ``SpatialFilter`` family
``07_field_centering_and_corrections``  Yee centering, pressure and heat-flux corrections
``08_raw_and_tracks.py``                RAW dumps, tag files, ``Track_Diagnostic``
``09_export_and_io.py``                 ``export_to_npy``, ``to_h5``, text I/O
``10_anomalous_resistivity.py``         ``AnomalousResistivity`` and its terms
``11_databases.py``                     ``DatabaseCreator``, burst dumps, Lorentz augmentation
``12_utils_profiling_vis.py``           run planning, array utilities, profiling, ``plot_3d``
``13_cli.py``                           the ``utils`` command-line tool
======================================  =========================================================

Notebooks
---------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   examples/quick_start
   examples/example_Simulation_Diagnostic
   examples/example_InputDeck
   examples/example_Derivatives
   examples/example_FFT
