Simulation Interface
====================

.. _simulation-api:

The `Simulation` class provides a high-level interface for handling OSIRIS simulation data and accessing various diagnostics.

Simulation Class
----------------

.. autoclass:: osiris_utils.data.simulation.Simulation
   :members:
   :special-members: __init__, __getitem__
   :undoc-members:
   :noindex:

   A wrapper class that manages access to multiple diagnostic quantities from an OSIRIS simulation.
   
   The Simulation class simplifies working with multiple diagnostics from the same simulation by:
   
   * Providing dictionary-style access to diagnostics using field/quantity names
   * Managing the simulation path and species information centrally
   * Caching loaded diagnostics for efficient reuse
   
   **Key Attributes:**
   
   * ``species`` - The plasma species being analyzed
   * ``simulation_folder`` - Path to the OSIRIS simulation data
   * ``diagnostics`` - Dictionary of loaded diagnostic objects
   
   **Usage Examples:**
   
   Basic usage with lazy loading:
   
   .. code-block:: python
   
       from osiris_utils.data import Simulation
       
       # Create a simulation interface for electron data
       sim = Simulation('path/to/simulation', "name_of_input_deck")
       
       # Access the E1 field diagnostic (doesn't load data yet) - this is a Diagnostic object
       # Since it is a diagnostic not related with the species, the species argument is not needed
       sim['e1']
       
       # For diagnostics related with the species, the species argument is needed
       sim['electrons']['charge']

       # Load all timesteps for this diagnostic
       e1_diag.load_all()
       
       # The diagnostic is now cached in the simulation object
       # Access it again without recreating
       same_e1_diag = sim['e1']
   
   Working with multiple diagnostics:
   
   .. code-block:: python
   
       # Access several diagnostics
       e1 = sim['e1']
       e2 = sim['e2']
       b3 = sim['b3']
       
       # Load specific timesteps
       e1[10:20]  # Load timesteps 10-19
       
       # Clean up to free memory
       sim.delete_diagnostic('e1')
       sim.delete_all_diagnostics()

Integration with Diagnostics
----------------------------

The Simulation class works seamlessly with the :ref:`diagnostic-system` system:

1. When you request a diagnostic with ``sim[quantity_name]`` or ``sim[species_name][quantity_name]``, it creates a Diagnostic object
2. The first time you load data with ``.load_all()`` or indexing, the diagnostic is cached
3. Subsequent accesses return the cached diagnostic for efficiency
4. You can explicitly remove diagnostics from the cache to manage memory
5. ``.load_all()`` loads all timesteps for the diagnostic
6. To access a single iteration of the quantity, indexing gives you the data for the requested timesteps.

This approach allows for efficient workflow when analyzing large simulations with multiple fields and timesteps.

.. _diagnostic-system:

Diagnostic System
=================

The `Diagnostic` class is the foundation of osiris_utils data handling, providing access to OSIRIS simulation diagnostics and support for derived quantities.

Diagnostic Base Class
---------------------

.. autoclass:: osiris_utils.data.diagnostic.Diagnostic
   :members:
   :special-members: __init__, __getitem__, __add__, __sub__, __mul__, __truediv__, __pow__
   :undoc-members:
   :noindex:

   The core class for accessing and manipulating OSIRIS diagnostic data. This class handles both raw OSIRIS data files and derived data from operations.
   
   **Key Features:**
   
   * Lazy loading system that only reads data when needed
   * Support for mathematical operations between diagnostics
   * Generator-based access for memory-efficient processing
   * Automatic attribute handling for derived quantities
   
   **Key Attributes:**
   
   * ``species`` - The plasma species being analyzed (Specie object)
   * ``dx`` - Grid spacing in each direction
   * ``nx`` - Number of grid points in each direction
   * ``x`` - Grid coordinates
   * ``data`` - The actual diagnostic data (when loaded)
   * ``dim`` - Dimensionality of the data
   * ``units`` - Physical units of the data
   * ``dt`` - Time step between outputs
   * ``ndump`` - Number of steps between dumps
   * ``grid`` - Boundaries of the simulation grid in each direction (physical units)
   * ``name`` - Name of the diagnostic quantity (for quantities directly obtained from OSIRIS data)
   * ``label`` - LaTeX label for the quantity (for quantities directly obtained from OSIRIS data)
   * ``axis`` - Axis information
   * ``maxiter`` - Maximum number of iterations of the diagnostic
   * ``tunits`` - Time units of the simulation

   
   **Usage Examples:**
   
   Basic loading and access:
   
   .. code-block:: python
   
       # Create diagnostic for electron charge
       diag = Diagnostic("/path/to/simulation", species)
       diag.get_quantity("charge")
       
       # Access specific timestep (without loading all data)
       timestep_5 = diag[5]
       
       # Load all timesteps into memory
       diag.load_all()
       
       # Now data is available as an array
       print(diag.data.shape)
   
   Mathematical operations:
   
   .. code-block:: python
   
       # Operations between diagnostics
       sim = Simulation("/path/to/simulation", species)
       e1 = sim["e1"]
       vfl1 = sim["electron"]["vfl1"]
       
       # Create a derived diagnostic without loading data
       e_times_v = (e1**2 + vfl1**2)**0.5
       
       # Access specific timestep of the result (calculated on-demand)
       timestep_10 = e_times_v[10]

Available Diagnostic Quantities in OSIRIS
-----------------------------------------

The following diagnostic quantities are available for OSIRIS simulations:

**Field Quantities:**

* ``e1``, ``e2``, ``e3`` - Electric field components
* ``b1``, ``b2``, ``b3`` - Magnetic field components

**Particle Quantities:**

* ``n`` - Density
* ``charge`` - Charge density
* ``j1``, ``j2``, ``j3`` - Current density components
* ``q1``, ``q2``, ``q3`` - Charge flux components

**Velocity Distribution Quantities:**

* ``vfl1``, ``vfl2``, ``vfl3`` - Flow velocity components
* ``ufl1``, ``ufl2``, ``ufl3`` - Momentum components
* Various pressure and temperature tensor components

**Phase Space Quantities:**

* ``p1x1``, ``p1x2``, etc. - Phase space diagnostics

To see all available quantities:

.. code-block:: python

    from osiris_utils.data.diagnostic import which_quantities
    which_quantities()

Memory-Efficient Processing
---------------------------

The `Diagnostic` class provides several ways to work with large datasets without loading everything into memory:

1. **Item access with** ``diag[index]`` - Loads only the requested timestep
2. **Iteration with** ``for timestep in diag:`` - Processes one timestep at a time
3. **Generator-based operations** - Mathematical operations create new diagnostics without loading data

This lazy evaluation system allows you to work with large simulations that would otherwise exceed available memory.

Visualization Methods
---------------------

The `Diagnostic` class provides visualization methods for quick inspection of 3D data:

* ``plot_3d()`` - Creates 3D scatter plots of 3D field data

Derived Diagnostics
-------------------

One of the most powerful features of the Diagnostic system is that new diagnostics can be created through operations on existing ones. These derived diagnostics maintain all the benefits of the base class:

.. code-block:: python

    # Create base diagnostics
    sim = Simulation("/path/to/simulation", "file_input_deck")
    e1 = sim["e1"]
    e2 = sim["e2"] 
    e3 = sim["e3"]
    
    # Create derived diagnostics through operations - e_magnitude and normalized_1 are Diagnostic objects
    e_magnitude = (e1**2 + e2**2 + e3**2)**0.5  # E-field magnitude
    normalized_e1 = e1 / e_magnitude            # Normalized E1 component
    
    # These are full diagnostics objects that support:
    # - Lazy loading
    # - Indexing
    # - Further operations
    # - Visualization
    
    # Access specific timesteps (calculated on demand)
    timestep_10 = e_magnitude[10]
    
    # Apply mathematical functions
    import numpy as np
    log_e_magnitude = np.log10(e_magnitude)

**Automatic Attribute Inheritance**

When creating diagnostics through operations, metadata is intelligently propagated:

1. Grid information (``dx``, ``nx``, ``x``) is preserved
2. Axis information is maintained
3. Time information (``dt``, ``ndump``) is maintained
4. Dimension and array shapes remain consistent, as well as maximum iteration number

**Chaining Operations**

Operations can be chained to create complex derivations.
