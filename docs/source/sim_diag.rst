Simulation Interface
====================

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
       sim = Simulation('electrons', 'path/to/simulation')
       
       # Access the E1 field diagnostic (doesn't load data yet)
       e1_diag = sim['e1']
       
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

1. When you request a diagnostic with ``sim[quantity_name]``, it creates a Diagnostic object
2. The first time you load data with ``.load_all()`` or indexing, the diagnostic is cached
3. Subsequent accesses return the cached diagnostic for efficiency
4. You can explicitly remove diagnostics from the cache to manage memory
5. ``.load_all()`` loads all timesteps for the diagnostic. Not required since indexing also loads data for the requested timesteps.

This approach allows for efficient workflow when analyzing large simulations with multiple fields and timesteps.

.. _diagnostic-system:

Diagnostic System
=================

The `Diagnostic` class is the foundation of osiris_utils data handling, providing access to OSIRIS simulation output files and support for derived quantities.

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
   
   * ``species`` - The plasma species being analyzed
   * ``dx`` - Grid spacing in each direction
   * ``nx`` - Number of grid points in each direction
   * ``x`` - Grid coordinates
   * ``data`` - The actual diagnostic data (when loaded)
   * ``dim`` - Dimensionality of the data
   * ``units`` - Physical units of the data
   * ``dt`` - Time step between outputs
   * ``ndump`` - Number of steps between dumps
   
   **Usage Examples:**
   
   Basic loading and access:
   
   .. code-block:: python
   
       # Create diagnostic for electron charge
       diag = Diagnostic("electrons", "/path/to/simulation")
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
       sim = Simulation("electrons", "/path/to/simulation")
       e1 = sim["e1"]
       e2 = sim["e2"]
       
       # Create a derived diagnostic without loading data
       e_magnitude = (e1**2 + e2**2)**0.5
       
       # Access specific timestep of the result (calculated on-demand)
       timestep_10 = e_magnitude[10]

Available Diagnostic Quantities in OSIRIS
-----------------------------------------

The following diagnostic quantities are available for OSIRIS simulations:

**Field Quantities:**

* ``e1``, ``e2``, ``e3`` - Electric field components
* ``b1``, ``b2``, ``b3`` - Magnetic field components

**Particle Quantities:**

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
    sim = Simulation("electrons", "/path/to/simulation")
    e1 = sim["e1"]
    e2 = sim["e2"] 
    e3 = sim["e3"]
    
    # Create derived diagnostics through operations
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
2. Time information (``dt``, ``ndump``) is maintained
3. Units and labels are combined appropriately when possible
4. Dimension and array shapes remain consistent

**Chaining Operations**

Operations can be chained to create complex derivations:

Post-Processing Framework
=========================

The `osiris_utils` package provides a framework for post-processing OSIRIS simulation data through the `PostProcess` class.

PostProcess Base Class
----------------------

.. autoclass:: osiris_utils.postprocessing.postprocess.PostProcess
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

   Base class for all post-processing operations in osiris_utils.
   
   The PostProcess framework allows for applying custom transformations to OSIRIS diagnostics.
   It inherits from the Diagnostic class to ensure compatibility with the broader osiris_utils ecosystem.
   
   **Key Attributes:**
   
   * ``name`` - Name of the post-processing operation
   * ``species`` - The plasma species to analyze (if applicable)
   
   **Key Methods:**
   
   * ``process(diagnostic)`` - Apply the post-processing to a diagnostic
   
   **Usage Example:**
   
   .. code-block:: python
   
       from osiris_utils.postprocessing import CustomPostProcess
       from osiris_utils.data import OsirisData
       
       # Load some OSIRIS data
       data = OsirisData("path/to/data")
       
       # Create a post-processor
       processor = CustomPostProcess(name="my_analysis", species="electrons")
       
       # Apply processing
       result = processor.process(data)
       
       # Visualize or further analyze
       result.plot()

.. Creating Custom Post-Processors
.. -------------------------------

.. To create your own post-processing operation, subclass `PostProcess` and implement the `process` method:

.. .. code-block:: python

..     from osiris_utils.postprocessing import PostProcess
    
..     class MyCustomProcessor(PostProcess):
..         def __init__(self, name, threshold=0.5, species=None):
..             super().__init__(name, species)
..             self.threshold = threshold
            
..         def process(self, diagnostic):
..             # Get raw data from the diagnostic
..             raw_data = diagnostic.data
            
..             # Apply some transformation
..             processed_data = raw_data * (raw_data > self.threshold)
            
..             # Create and return a new object with processed data
..             result = diagnostic.copy()
..             result.data = processed_data
..             return result

Integration with OSIRIS Diagnostics
-----------------------------------

Post-processors are designed to work seamlessly with the OSIRIS diagnostic system:

* They can be chained together for complex analysis pipelines
* Results maintain compatibility with visualization tools
* Processing operations can be applied to any diagnostic type


Derivative Post-Processing
==========================

The `Derivative_Simulation` module provides tools for computing various types of derivatives from simulation diagnostics.

Derivative_Simulation Class
----------------

.. autoclass:: osiris_utils.postprocessing.derivative.Derivative_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Post-processor for computing derivatives of diagnostic data.
   
   The Derivative_Simulation class provides a convenient interface for calculating derivatives of various simulation quantities.
   It works as a wrapper around the `Derivative_Diagnostic` class, managing the creation and caching of derivative objects.
   
   **Derivative Types:**
   
   * ``t`` - Time derivative (d/dt)
   * ``x1`` - First spatial derivative (d/dx1)
   * ``x2`` - Second spatial derivative (d/dx2) 
   * ``x3`` - Third spatial derivative (d/dx3)
   * ``xx`` - Second-order spatial derivative (d2/dxidxj)
   * ``xt`` - Mixed space-time derivative (d2/dxdt)
   * ``tx`` - Mixed time-space derivative (d2/dtdx)
   
   **Usage Examples:**
   
   Basic usage with a simulation:
   
   .. code-block:: python
   
       from osiris_utils.data import Simulation
       from osiris_utils.postprocessing import Derivative_Simulation
       
       # Create a simulation interface
       sim = Simulation('electrons', '/path/to/simulation')
       
       # Create a derivative processor for x₁ derivatives
       dx1 = Derivative_Simulation(sim, 'x1')
       
       # Get the derivative of E1 with respect to x₁
       dE1_dx1 = dx1['e1']
       
       # Access specific timestep
       timestep_5 = dE1_dx1[5]
       
       # Load all timesteps
       dE1_dx1.load_all()


Derivative_Diagnostic Class
---------------------------

.. autoclass:: osiris_utils.postprocessing.derivative.Derivative_Diagnostic
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Specialized diagnostic that represents a derivative of another diagnostic.
   
   This class handles the actual computation of derivatives while maintaining the Diagnostic interface.
   It inherits from the base Diagnostic class, ensuring that all mathematical operations and visualization
   methods work consistently.
   
   **Key Features:**
   
   * Lazy evaluation - derivatives are computed on-demand
   * Memory-efficient - only requested timesteps are processed
   * Metadata preservation - grid information and other metadata is maintained
   * Full compatibility with other diagnostics - can be used in further operations

Implementation Details
----------------------

The derivative calculation uses NumPy's gradient function with appropriate handling of boundary conditions:

1. **Time Derivatives** (∂/∂t):
   * Uses centered differences for interior points
   * Uses forward/backward differences at boundaries
   * Accounts for the simulation time step and ndump parameter

2. **Spatial Derivatives** (∂/∂x):
   * Uses centered differences with edge_order=2 for higher accuracy
   * Properly handles grid spacing (dx) from the original diagnostic
   * Supports different dimensionality (1D, 2D, 3D)

3. **Higher-Order Derivatives** (∂²/∂x∂y):
   * Implemented through chained application of gradient
   * Requires specification of derivative axes

Performance Considerations
--------------------------

When working with large simulations, consider these performance tips:

1. **Selective Loading**:
   * Use indexing (`deriv['e1'][10]`) to compute derivatives for specific timesteps
   * Only call `load_all()` when you need all timesteps

2. **Memory Management**:
   * Clear unneeded derivatives with `derivative.delete('e1')` or `derivative.delete_all()`
   * For very large simulations, process data iteratively rather than loading everything

3. **Caching Behavior**:
   * Computed derivatives are cached for reuse
   * Initial access may be slower, but subsequent access is fast

Use Cases
---------

Common applications of the Derivative_Simulation post-processor include:

1. **Computing Gradients**:
   * Electric field gradients for energy analysis
   * Density gradients for instability studies

2. **Calculating Curl and Divergence**:
   * Curl of magnetic field using spatial derivatives
   * Divergence of electric field for charge density validation

3. **Growth Rate Analysis**:
   * Time derivatives to measure instability growth rates

4. **Phase Velocity Measurements**:
   * Mixed space-time derivatives for wave analysis

Example: Computing Curl of B
----------------------------

This example shows how to compute the z-component of curl(B):

.. code-block:: python

    from osiris_utils.data import Simulation
    from osiris_utils.postprocessing import Derivative
    
    # Setup
    sim = Simulation('electrons', '/path/to/simulation')
    dx1 = Derivative_Simulation(sim, 'x1')
    dx2 = Derivative_Simulation(sim, 'x2')

    # Calculate curl(B) = dB2/dx1 - dB1/dx2
    dB2_dx1 = dx1["b2"]
    dB1_dx2 = dx2["b1"]
    
    # Compute curl B (z-component)
    curl_B_z = dB2_dx1 - dB1_dx2

Spectral Analysis with Fast Fourier Transform
=============================================

The `FastFourierTransform_Simulation` module provides tools for performing spectral analysis on simulation diagnostics using the Fast Fourier Transform (FFT).

FastFourierTransform_Simulation Class
-------------------------------------

.. autoclass:: osiris_utils.postprocessing.fft.FastFourierTransform_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Post-processor for computing power spectra of diagnostic data.
   
   The FastFourierTransform class provides a convenient interface for calculating FFTs of various simulation quantities.
   It works as a wrapper around the `FFT_Diagnostic` class, managing the creation and caching of FFT objects.
   
   **Key Features:**
   
   * Transform along time axis (axis=0)
   * Transform along spatial axes (axis=1,2,3)
   * Support for multi-dimensional FFTs
   * Automatic windowing for improved spectral estimates
   * Caching of results for efficient reuse
   
   **Usage Examples:**
   
   Basic usage with a simulation:
   
   .. code-block:: python
   
       from osiris_utils.data import Simulation
       from osiris_utils.postprocessing import FastFourierTransform_Simulation
       
       # Create a simulation interface
       sim = Simulation('electrons', '/path/to/simulation')
       
       # Create an FFT processor for the first spatial dimension
       fft = FastFourierTransform_Simulation(sim, 1)
       
       # Get the power spectrum of E1
       e1_spectrum = fft['e1']

FFT_Diagnostic Class
--------------------

.. autoclass:: osiris_utils.postprocessing.fft.FFT_Diagnostic
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Specialized diagnostic that represents the Fourier transform of a diagnostic.
   
   This class handles the actual computation of FFTs while maintaining the Diagnostic interface.
   It inherits from the base Diagnostic class, ensuring that all mathematical operations and visualization
   methods work consistently.
   
   **Key Methods:**
   
   * ``load_all()`` - Computes the complete FFT for all timesteps
   * ``omega()`` - Returns the frequency/wavenumber array
   * ``kmax`` - Property returning the maximum wavenumber (Nyquist frequency)
   
   **FFT Properties:**
   
   * Power spectra are returned
   * Data is properly shifted to place zero frequency at the center
   * Hanning windows are applied to reduce spectral leakage

Implementation Details
----------------------

The FFT implementation includes several important features:

1. **Windowing**:
   * Hanning windows are applied by default to reduce spectral leakage
   * Different windows can be implemented by overriding `_get_window()`
   * Windows are properly shaped to match data dimensionality

2. **FFT Calculation**:
   * Single-axis FFTs use `np.fft.fft`
   * Multi-dimensional FFTs use `np.fft.fftn`
   * Results are properly shifted with `np.fft.fftshift`
   * Power spectra are returned

3. **Memory Management**:
   * Single-dimension spatial FFTs can be computed on-demand for individual timesteps
   * Time-axis FFTs require all data to be loaded (via `load_all()`)
   * Progress bars are displayed for long-running computations

Common Applications
-------------------

The FFT post-processor is valuable for many plasma physics analyses:

1. **Wave Analysis**:
   * Identify wave modes and their frequencies
   * Measure dispersion relations
   * Analyze wave growth and damping

2. **Instability Studies**:
   * Identify dominant wavenumbers in instabilities
   * Track growth of specific modes

3. **Energy Cascade**:
   * Examine energy distribution across scales
   * Study turbulence through k-space analysis

4. **Noise Identification**:
   * Separate physical signals from numerical noise
   * Identify grid-scale artifacts

Example: Computing a Dispersion Relation
----------------------------------------

This example shows how to compute and visualize a dispersion relation:

.. code-block:: python

    from osiris_utils.data import Simulation
    from osiris_utils.postprocessing import FastFourierTransform_Simulation
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Setup
    sim = Simulation('electrons', '/path/to/simulation')
    
    # Create FFT processor for both time and space
    # We'll do a 2D FFT - time (axis 0) and x1 (axis 1)
    fft_processor = FastFourierTransform_Simulation(sim, [0, 1])
    
    # Get E1 field and compute its FFT
    e1_fft = fft_processor['e1']
    e1_fft.load_all()  # Must load all for time-domain FFT
    
    # Get frequency and wavenumber arrays
    time_steps = sim['e1'].nx[0]
    omega = np.fft.fftfreq(time_steps, d=sim['e1'].dt * sim['e1'].ndump)
    omega = np.fft.fftshift(omega)
    
    k = np.fft.fftfreq(sim['e1'].nx[1], d=sim['e1'].dx[0])
    k = np.fft.fftshift(k)
    
    # Plot the dispersion relation
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(k, omega, np.log10(e1_fft.data), cmap='inferno', shading='auto')
    plt.colorbar(label='log₁₀(Power)')
    plt.xlabel('Wavenumber k')
    plt.ylabel('Frequency ω')
    plt.title('E1 Dispersion Relation')
    
    plt.tight_layout()
    plt.show()

Performance Considerations
--------------------------

For large datasets, consider these performance optimizations:

1. **Memory Usage**:
   * For large 3D simulations, compute FFTs along one dimension at a time
   * Use `delete()` and `delete_all()` to free memory when finished with results

2. **Computation Time**:
   * Multi-dimensional FFTs are computationally intensive
   * The `load_all()` method displays a progress bar for long calculations
   * For time-critical applications, consider downsampling data before FFT

3. **Storage Efficiency**:
   * FFT results can be larger than the original data (complex values)
   * For very large datasets, consider saving results to disk using NumPy's `save` function


Mean Field Theory Analysis
==========================

The Mean Field Theory (MFT) module provides tools for decomposing simulation data into average (mean) and fluctuation components, a fundamental approach in plasma physics analysis.

MeanFieldTheory_Simulation Class
--------------------------------

.. autoclass:: osiris_utils.postprocessing.mft.MeanFieldTheory_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Post-processor for performing mean field decomposition of diagnostic data.
   
   The MeanFieldTheory_Simulation class provides a convenient interface for separating plasma quantities into their average and fluctuating components along a specified axis.
   
   **Key Features:**
   
   * Decompose field or particle quantities into mean and fluctuating parts
   * Support for analysis along any spatial dimension
   * Lazy evaluation for memory-efficient processing
   * Full diagnostic interfaces for both components
   
   **Usage Examples:**
   
   Basic decomposition:
   
   .. code-block:: python
   
       from osiris_utils.data import Simulation
       from osiris_utils.postprocessing import MeanFieldTheory_Simulation
       
       # Create a simulation interface
       sim = Simulation('electrons', '/path/to/simulation')
       
       # Create MFT analyzer for x₁ direction (axis=1)
       mft = MeanFieldTheory_Simulation(sim, 1)
       
       # Get MFT decomposition of electric field E₁
       mft_e1 = mft['e1']
       
       # Access average component
       e1_avg = mft_e1['avg']
       
       # Access fluctuation component
       e1_delta = mft_e1['delta']
       
       # Load specific timesteps
       timestep_10_avg = e1_avg[10]
       timestep_10_delta = e1_delta[10]
       
       # Load all data
       e1_avg.load_all()
       e1_delta.load_all()

MFT_Diagnostic Class
--------------------

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Container class that manages both average and fluctuation components of a diagnostic.
   
   This class acts as a manager for the decomposition, providing access to both components
   through a dictionary-like interface.
   
   **Key Methods:**
   
   * ``__getitem__(key)`` - Access either 'avg' or 'delta' components
   * ``load_all()`` - Load both components into memory

MFT_Diagnostic_Average Class
----------------------------

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic_Average
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Specialized diagnostic that represents the average component.
   
   This class provides the average (mean) of the original diagnostic along the specified axis,
   maintaining the full Diagnostic interface.
   
   **Key Methods:**
   
   * ``load_all()`` - Compute and store the complete average dataset
   * ``__getitem__(index)`` - Compute average for a specific timestep on-demand

MFT_Diagnostic_Fluctuations Class
---------------------------------

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic_Fluctuations
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

   Specialized diagnostic that represents the fluctuation component.
   
   This class provides the fluctuations (deviations from the average) of the original diagnostic
   along the specified axis, maintaining the full Diagnostic interface.
   
   **Key Methods:**
   
   * ``load_all()`` - Compute and store the complete fluctuation dataset
   * ``__getitem__(index)`` - Compute fluctuations for a specific timestep on-demand

Mean Field Theory Concepts
--------------------------

Mean Field Theory is a fundamental approach in plasma physics that decomposes quantities into:

1. **Average Component** (⟨A⟩):
   * Represents large-scale, slowly varying background
   * Computed by averaging over a specific dimension
   * Contains systematic behavior of the system

2. **Fluctuation Component** (δA):
   * Represents small-scale, rapidly varying perturbations
   * Computed as δA = A - ⟨A⟩
   * Contains turbulence, waves, and other transient phenomena

This decomposition allows for:

* Separation of scales in multi-scale phenomena
* Analysis of energy transfer between scales
* Study of instability development and turbulence
* Identification of coherent structures

Implementation Details
----------------------

The MFT implementation includes several important features:

1. **Averaging Mechanism**:
   * Uses NumPy's `mean()` function along the specified axis
   * For on-demand calculation, properly reshapes arrays for broadcasting

2. **Memory Management**:
   * Both components can be calculated on-demand for specific timesteps
   * Complete datasets can be pre-computed using `load_all()`
   * Metadata is preserved from the original diagnostic

3. **Dimensional Handling**:
   * Works with 1D, 2D, and 3D data
   * Manages axis indexing differences between full arrays and individual timesteps

Performance Considerations
--------------------------

For large datasets, consider these performance optimizations:

1. **Memory Usage**:
   * Use on-demand calculation with indexing when analyzing individual timesteps
   * Only call `load_all()` when analyzing the full time evolution
   * Use `delete()` and `delete_all()` to free memory when finished with results

2. **Computation Efficiency**:
   * Averaging is computationally inexpensive compared to other operations
   * For 2D/3D data, consider which axis to average along based on your physics
   * For iterative analysis, calculate fluctuations only when needed