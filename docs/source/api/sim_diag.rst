Simulation Interface
====================

.. _simulation-api:

The ``Simulation`` class provides a high-level interface for handling OSIRIS simulation data and accessing diagnostics through a consistent, dictionary-like API.

Simulation Class
----------------

.. autoclass:: osiris_utils.data.simulation.Simulation
   :members:
   :special-members: __init__, __getitem__
   :undoc-members:
   :noindex:

A wrapper class that manages access to multiple diagnostic quantities from a single OSIRIS run.

**What Simulation gives you**

- **Dictionary-style access** to diagnostics:

  - ``sim["e1"]`` returns a non-species diagnostic (fields, global quantities, etc.)
  - ``sim["electrons"]["charge"]`` returns a species diagnostic

- **Centralized metadata**:

  - simulation folder is inferred from the input deck path
  - species list comes from the parsed input deck

- **Caching of created diagnostics**:

  - repeated access returns the same wrapper object
  - you can delete cached diagnostics to free memory

**Key attributes**

- ``species``: list of species names found in the input deck
- ``loaded_diagnostics``: dictionary of cached non-species diagnostics

.. note::

   ``Simulation`` does not automatically load data. Diagnostics are lazy by default.
   Data is read only when you index a diagnostic or call ``load_all()``.


Usage examples
~~~~~~~~~~~~~~

Basic access (lazy)
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from osiris_utils.data import Simulation

   sim = Simulation("/path/to/osiris.inp")

   # Non-species diagnostic (no data read yet)
   e1 = sim["e1"]

   # Species diagnostic
   charge = sim["electrons"]["charge"]

Load specific timesteps and slices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Diagnostics support:

- time indexing: ``diag[i]`` and ``diag[i:j]``
- tuple indexing: ``diag[i, x_slice, y_slice, ...]`` (time + spatial slicing)

.. code-block:: python

   e1 = sim["e1"]

   # Single timestep (reads one file)
   arr_t10 = e1[10]

   # Multiple timesteps (reads only requested files)
   arr_t10_20 = e1[10:20]

   # Time index + spatial slicing (efficient: reads only slice from disk if supported)
   # Example for 2D: (time, x1, x2)
   arr_roi = e1[10, :, 100:200]

Loading everything into memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   e1 = sim["e1"]
   e1.load_all()          # loads all timesteps into memory
   full = e1.data         # numpy array with shape (t, ...)

Cache management
^^^^^^^^^^^^^^^^

.. code-block:: python

   # remove one diagnostic from the simulation cache
   sim.delete_diagnostic("e1")

   # remove all cached diagnostics
   sim.delete_all_diagnostics()


Integration with Diagnostics
----------------------------

The workflow is:

1. ``sim[quantity]`` or ``sim[species][quantity]`` creates a ``Diagnostic`` wrapper (lazy)
2. Accessing data (via indexing or ``load_all()``) reads from disk
3. After the first ``load_all()``, the ``Simulation`` caches the diagnostic so future access returns the same object
4. You can remove cached diagnostics explicitly to manage memory

This keeps interactive analysis fast while still scaling to large datasets.


.. _diagnostic-system:

Diagnostic System
=================

The ``Diagnostic`` class is the foundation of data handling in ``osiris_utils``. It represents a time series of grid data stored in OSIRIS output files, plus any derived quantities created through operations.

Diagnostic Base Class
---------------------

.. autoclass:: osiris_utils.data.diagnostic.Diagnostic
   :members:
   :special-members: __init__, __getitem__, __add__, __sub__, __mul__, __truediv__, __pow__
   :undoc-members:
   :noindex:

**Key features**

- **Lazy loading**: reads only the requested timestep(s) from disk
- **Time slicing**: supports ``diag[i]`` and ``diag[i:j]``
- **Spatial slicing**: supports tuple indexing (time + spatial slices) when backed by OSIRIS HDF5
- **Derived diagnostics**: arithmetic between diagnostics produces new diagnostics without immediately loading data
- **Metadata propagation**: grid/time metadata is preserved for derived quantities

**Common attributes**

- ``dx``: grid spacing (float for 1D, array-like for >1D)
- ``nx``: number of grid points
- ``x``: coordinate arrays
- ``dt``: simulation timestep (from file metadata)
- ``ndump``: dump interval (from input deck when available; defaults to 1 otherwise)
- ``grid``: physical bounds of each axis
- ``axis``: axis metadata (names/labels/units)
- ``dim``: dimensionality (1/2/3)
- ``maxiter``: number of timesteps available for this diagnostic
- ``name`` / ``label`` / ``units``: metadata when available
- ``data``: full in-memory array after calling ``load_all()``

Usage examples
~~~~~~~~~~~~~~

Create and access diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases you should use ``Simulation`` rather than creating ``Diagnostic`` manually:

.. code-block:: python

   from osiris_utils.data import Simulation

   sim = Simulation("/path/to/osiris.inp")

   e1 = sim["e1"]
   ne = sim["electrons"]["n"]

   # read a single timestep (lazy)
   e1_t10 = e1[10]

   # read a time slice (lazy, reads only needed files)
   ne_t0_20 = ne[0:20]

Derived diagnostics (lazy)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Operations between diagnostics return a new ``Diagnostic``-like object that can still be indexed lazily:

.. code-block:: python

   e1 = sim["e1"]
   e2 = sim["e2"]
   e3 = sim["e3"]

   e_mag = (e1**2 + e2**2 + e3**2) ** 0.5

   # computed on demand
   e_mag_t10 = e_mag[10]

   # can also load full result if desired
   e_mag.load_all()
   full = e_mag.data


Available Diagnostic Quantities in OSIRIS
-----------------------------------------

OSIRIS provides many diagnostics. In ``osiris_utils``, quantities are exposed via their OSIRIS-style names.

**Field quantities**

- ``e1``, ``e2``, ``e3`` — electric field components
- ``b1``, ``b2``, ``b3`` — magnetic field components
- (and optionally ``part_*`` / ``ext_*`` variants if present in the output)

**Species-dependent grid quantities**

- ``charge`` — charge density
- ``j1``, ``j2``, ``j3`` — current density components
- ``q1``, ``q2``, ``q3`` — charge flux components
- ``n`` — convenience alias for density (implemented via OSIRIS charge with sign convention)

**Velocity / moment quantities**

- ``vfl1``, ``vfl2``, ``vfl3`` — flow velocity components
- ``ufl1``, ``ufl2``, ``ufl3`` — momentum components
- pressure / temperature tensor components: ``P11``, ``P12``, ..., ``T11``, ...

**Phase space quantities**

- ``p1x1``, ``p1x2``, ... (depends on what was configured in the OSIRIS input)

To list what the package recognizes:

.. code-block:: python

   from osiris_utils.data.diagnostic import which_quantities
   which_quantities()


Memory-efficient processing
---------------------------

For large simulations, prefer lazy access patterns:

1. **Single timestep**: ``diag[i]``
2. **Time slice**: ``diag[i:j]``
3. **Spatial ROI**: ``diag[i, :, 100:200]`` (time + spatial slices)

Only use ``load_all()`` when you truly need the full time series in memory.


Derived diagnostics and metadata propagation
--------------------------------------------

Derived diagnostics created by arithmetic preserve important metadata:

- grid spacing and coordinates (``dx``, ``x``, ``grid``)
- dimensionality (``dim``) and timestep count (``maxiter``)
- time metadata (``dt``, ``ndump``)

This makes derived quantities “first-class” diagnostics that can be:

- indexed like any other diagnostic
- used in further operations
- passed into post-processing routines (FFT, derivatives, MFT, etc.)
