Post-Processing Framework
=========================

.. _postprocessing:

The ``osiris_utils`` package provides a lightweight framework for post-processing OSIRIS simulation data.
Post-processors are implemented as wrappers around the existing ``Simulation`` / ``Diagnostic`` interface so that:

- indexing works the same way everywhere (lazy, sliceable, tuple-sliceable)
- arithmetic between diagnostics keeps working (``+``, ``-``, ``*``, ``/``, etc.)
- post-processors can be chained when it makes sense (e.g., derivatives)

This page documents the framework and the main post-processors currently available.


PostProcess Base Class
----------------------

.. _postprocess-class:

.. autoclass:: osiris_utils.postprocessing.postprocess.PostProcess
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

The ``PostProcess`` class is the base class for all simulation-level post-processors.

**Design goals**

A post-processor should behave like a ``Simulation``:

- ``pp["e3"]`` returns a diagnostic-like object (usually a ``Diagnostic`` subclass)
- ``pp["electrons"]["n"]`` returns a species diagnostic
- it caches created wrappers so repeated access is cheap

Most post-processors follow the same pattern:

1. A simulation wrapper (``Something_Simulation``) inheriting from ``PostProcess``
2. A diagnostic wrapper (``Something_Diagnostic``) inheriting from ``Diagnostic``
3. An optional species handler (``Something_Species_Handler``) for species-dependent quantities


Integration with OSIRIS Diagnostics
-----------------------------------

Post-processors are designed to work seamlessly with the OSIRIS diagnostic system:

- **Chaining (when appropriate)**: some post-processors can wrap other post-processors
  (e.g., derivatives of already-derived quantities).
- **Compatibility with visualization**: results remain ``Diagnostic``-like and can be plotted the same way.
- **Uniform indexing contract**:

  - ``diag[i]`` returns a single timestep ``np.ndarray`` (shape ``(x, y, z)`` depending on dimension)
  - ``diag[i:j]`` returns a stacked time array (shape ``(t, x, y, z)``)
  - ``diag[i, :, 100:200]`` uses tuple indexing (time index + spatial slices)

**Important implementation note**

The base ``Diagnostic.__getitem__`` already implements:

- int and slice time indexing
- tuple indexing (time + spatial slices)
- fast path for in-memory data

So post-processed diagnostics should generally implement:

- ``load_all()`` to build ``self._data`` (shape includes time)
- ``_frame(index, data_slice=None)`` to return a single timestep lazily (shape excludes time)

You usually **do not** need to override ``__getitem__`` in post-processed diagnostics.


Derivative Post-Processing
==========================

.. _derivative-api:

The ``Derivative_Simulation`` module provides tools for computing time and spatial derivatives of diagnostics.

Derivative_Simulation Class
---------------------------

.. autoclass:: osiris_utils.postprocessing.derivative.Derivative_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

``Derivative_Simulation`` behaves like an operator acting on all diagnostics inside a simulation:

- wraps a ``Simulation`` (or a compatible post-process wrapper)
- returns derivative diagnostics on demand
- supports species and non-species diagnostics
- supports lazy evaluation and efficient spatial slicing
- may be chained (e.g., derivative of a derivative)

**Derivative types**

- ``t``  : time derivative :math:`\\partial/\\partial t`
- ``x1`` : spatial derivative :math:`\\partial/\\partial x_1`
- ``x2`` : spatial derivative :math:`\\partial/\\partial x_2`
- ``x3`` : spatial derivative :math:`\\partial/\\partial x_3`
- ``xx`` : second derivative in two spatial axes (mixed or repeated), e.g. :math:`\\partial^2/\\partial x_i\\partial x_j`
- ``xt`` : :math:`\\partial/\\partial x` of :math:`\\partial/\\partial t`
- ``tx`` : :math:`\\partial/\\partial t` of :math:`\\partial/\\partial x`

**Usage examples**

.. code-block:: python

   from osiris_utils.data import Simulation
   from osiris_utils.postprocessing import Derivative_Simulation

   sim = Simulation("/path/to/input/deck.inp")

   # Spatial derivative d/dx1 of every diagnostic
   dx1 = Derivative_Simulation(sim, "x1")

   dE1_dx1 = dx1["e1"]
   dvfl1_dx1 = dx1["electrons"]["vfl1"]

   # Single timestep
   arr = dE1_dx1[5]

   # Time slice
   arr_t = dE1_dx1[5:10]

   # Spatial slicing (efficient if underlying diagnostic supports it)
   arr_slice = dE1_dx1[5, :, 100:200]

   # Full in-memory derivative (use only when needed)
   dE1_dx1.load_all()

**Implementation details and accuracy**

- Spatial derivatives use finite differences (2nd or 4th order depending on configuration).
- Time derivatives must account for OSIRIS ``ndump``:

  effective :math:`\\Delta t = dt \\times ndump`

- For mixed derivatives (``tx``, ``xt``), the implementation applies the corresponding operator ordering.

**Performance tips**

- Prefer indexing (lazy computation) when exploring data.
- Use spatial slices to reduce IO and compute time.
- Call ``load_all()`` only when you truly need the full time history in memory.


Derivative_Diagnostic Class
---------------------------

.. autoclass:: osiris_utils.postprocessing.derivative.Derivative_Diagnostic
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

A derivative diagnostic is a ``Diagnostic`` wrapper that computes derivatives lazily using ``_frame`` and can compute eagerly using ``load_all``.
It preserves metadata (grid spacing, dimensions, etc.) from the original diagnostic and remains compatible with diagnostic arithmetic.


Spectral Analysis with Fast Fourier Transform
=============================================

.. _fft-api:

The ``FFT_Simulation`` module provides tools for computing power spectra using the Fast Fourier Transform (FFT).

FFT_Simulation Class
--------------------

.. autoclass:: osiris_utils.postprocessing.fft.FFT_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

``FFT_Simulation`` is a simulation wrapper that returns ``FFT_Diagnostic`` objects.

**Key points about FFT axes**

- OSIRIS diagnostics loaded with ``load_all()`` have shape ``(t, x1, x2, x3)`` (depending on dimension).
  In that case:
  - axis ``0`` is time
  - axes ``1,2,3`` are spatial

- A *single timestep* returned by indexing has no time dimension, only spatial:
  - axes become ``(x1, x2, x3)`` → numpy axes ``0,1,2``

**Important constraint**

- If the FFT includes the time axis (axis ``0``), you cannot compute it from a single timestep.
  Time-domain FFT requires the full time series, therefore it requires ``load_all()`` on the FFT diagnostic.

- Spatial FFTs *can* be computed per-timestep lazily.

**Usage example**

.. code-block:: python

   from osiris_utils.data import Simulation
   from osiris_utils.postprocessing import FFT_Simulation

   sim = Simulation("/path/to/input/deck.inp")

   # Spatial spectrum at each timestep (FFT along x1)
   fft_x1 = FFT_Simulation(sim, 1)
   e3_k = fft_x1["e3"]
   k_spectrum_t10 = e3_k[10]  # OK: spatial FFT at timestep 10

   # Dispersion-like spectrum (time + space) requires load_all
   fft_wk = FFT_Simulation(sim, (0, 1))
   e3_wk = fft_wk["e3"]
   e3_wk.load_all()           # required
   P = e3_wk.data             # power spectrum |FFT|^2


FFT_Diagnostic Class
--------------------

.. autoclass:: osiris_utils.postprocessing.fft.FFT_Diagnostic
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

A specialized diagnostic representing an FFT-based power spectrum.

**Behavior**

- ``load_all()`` computes the FFT over the requested axes (including time if axis 0 is included).
- ``_frame(index, data_slice=None)`` computes a per-timestep FFT only for spatial axes.
- The returned data is a **power spectrum** (typically :math:`|\\mathrm{FFT}|^2`) and is shifted using ``fftshift`` so that
  zero frequency / wavenumber is centered.

**Frequency / wavenumber arrays**

- Use ``omega()`` for the time-frequency axis (when axis 0 was transformed and data is loaded).
- Use ``k(axis=...)`` for spatial wavenumbers.
- ``kmax`` reports the Nyquist wavenumber :math:`\\pi/\\Delta x` for the relevant spatial axes.


Example: Dispersion relation (ω–k)
----------------------------------

.. code-block:: python

   from osiris_utils.data import Simulation
   from osiris_utils.postprocessing import FFT_Simulation
   import numpy as np
   import matplotlib.pyplot as plt

   sim = Simulation("/path/to/input/deck.inp")

   fft = FFT_Simulation(sim, (0, 1))
   e1_fft = fft["e1"]
   e1_fft.load_all()

   omega = e1_fft.omega()      # angular frequency axis (shifted)
   k1 = e1_fft.k(1)            # k for x1 axis (shifted)

   P = np.log10(e1_fft.data + 1e-30)  # safer for plotting

   plt.figure()
   plt.pcolormesh(k1, omega, P, shading="auto")
   plt.xlabel("k1")
   plt.ylabel("ω")
   plt.title("Dispersion-like spectrum: E1")
   plt.show()


Mean Field Theory Analysis
==========================

.. _mft-api:

The Mean Field Theory (MFT) module provides tools for decomposing diagnostics into an average (mean) component and fluctuations along a chosen axis.

MFT_Simulation Class
--------------------

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

The MFT decomposition returns a container-like diagnostic with two components:

- ``"avg"``   : mean field :math:`\\langle A \\rangle`
- ``"delta"`` : fluctuations :math:`\\delta A = A - \\langle A \\rangle`

Usage example:

.. code-block:: python

   from osiris_utils.data import Simulation
   from osiris_utils.postprocessing import MFT_Simulation

   sim = Simulation("/path/to/input/deck.inp")

   mft = MFT_Simulation(sim, 1)       # average along x1 direction
   e1 = mft["e1"]

   e1_avg = e1["avg"]
   e1_delta = e1["delta"]

   avg_t10 = e1_avg[10]
   flt_t10 = e1_delta[10]


MFT_Diagnostic Classes
----------------------

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic_Average
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

.. autoclass:: osiris_utils.postprocessing.mft.MFT_Diagnostic_Fluctuations
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:


Field Centering
===============

.. _field-centering-api:

The ``FieldCentering`` module provides tools to convert electromagnetic fields from the Yee mesh locations to cell centers.

FieldCentering_Simulation Class
-------------------------------

.. autoclass:: osiris_utils.postprocessing.field_centering.FieldCentering_Simulation
   :members:
   :special-members: __init__, __getitem__
   :show-inheritance:
   :noindex:

Field centering is a purely spatial operation and can be computed lazily per timestep.

Usage example:

.. code-block:: python

   from osiris_utils.data import Simulation
   from osiris_utils.postprocessing import FieldCentering_Simulation

   sim = Simulation("/path/to/input/deck.inp")

   centered = FieldCentering_Simulation(sim)

   e1c = centered["e1"]
   arr = e1c[10]              # centered field at timestep 10
   arr_slice = e1c[10, :, :]  # supports spatial slicing


FieldCentering_Diagnostic Class
-------------------------------

.. autoclass:: osiris_utils.postprocessing.field_centering.FieldCentering_Diagnostic
   :members:
   :special-members: __init__
   :show-inheritance:
   :noindex:

This diagnostic wraps a field diagnostic and returns its cell-centered version. The implementation assumes periodic boundaries when using shifts/rolls.
``
