Contributing to OSIRIS Utils
============================

Thank you for contributing!

This document contains guidelines (not strict rules) for contributing to
**osiris_utils**. If something here doesn't fit your use case, use your best
judgment — and feel free to open an issue to discuss improvements.

Reporting Bugs
~~~~~~~~~~~~~~

How do I submit a good bug report?
----------------------------------

Bugs are tracked as GitHub issues:
`https://github.com/joaopedrobiu6/osiris_utils/issues/`

To help maintainers reproduce (and fix) the problem quickly, please include:

- **A clear and descriptive title**.
- **Exact steps to reproduce**, in order.
- **A minimal example** (copy/pasteable snippet) whenever possible.
- **What you observed**, including the full traceback or error message.
- **What you expected to happen**, and why.
- **Relevant context**, such as:
  - OS / Python version
  - osiris_utils version or commit hash
  - input deck details if relevant
- **Plots** if the bug is about “wrong-looking” physics or unexpected results.

If the bug involves post-processing or arithmetic with diagnostics, it helps a
lot to include a small snippet like:

.. code-block:: python

   import osiris_utils as ou
   sim = ou.Simulation("path/to/input_deck.txt")

   # reproduce here
   diag = sim["electrons"]["n"]
   out = diag[0]
   print(out.shape, out.dtype)

Adding Post-Processing routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-processing routines in **osiris_utils** are implemented as wrappers around
the existing ``Simulation`` / ``Diagnostic`` interface.

**Core idea**

- Indexing must behave consistently across all diagnostics:

  - ``diag[i]`` returns a single timestep ``np.ndarray``
  - ``diag[i:j]`` returns a stacked array over time (shape ``(t, ...)``)
  - ``diag[i, :, 10:20]`` must work (tuple indexing with spatial slicing)

- Post-processed diagnostics should remain compatible with diagnostic operations:

  - ``diag + other``, ``diag * other``, etc.

- Spatial slicing should be efficient when possible: pass the ``data_slice`` down
  to the base diagnostic (so HDF5 can load only the requested region).

The recommended structure is:

1) ``NameOfPostProcess_Simulation`` (simulation wrapper)
2) ``NameOfPostProcess_Diagnostic`` (diagnostic wrapper)
3) ``NameOfPostProcess_Species_Handler`` (species wrapper)

1) Simulation-level wrapper: ``NameOfPostProcess_Simulation``
-------------------------------------------------------------

Implement a wrapper class that behaves like a ``Simulation``.

**Requirements**

- Inherits from ``PostProcess``
- Stores the wrapped simulation in ``self._simulation``
- Implements ``__getitem__`` that:

  - returns a *species handler* when ``key`` is a species name
  - returns a *post-processed diagnostic* when ``key`` is a quantity name

- Uses caches (dicts) to avoid rebuilding wrappers repeatedly

Typical skeleton:

.. code-block:: python

   class MyPost_Simulation(PostProcess):
       def __init__(self, simulation: Simulation, ...):
           super().__init__("MyPost", simulation)
           self._computed = {}
           self._species_handler = {}

       def __getitem__(self, key):
           if key in self._simulation._species:
               if key not in self._species_handler:
                   self._species_handler[key] = MyPost_Species_Handler(self._simulation[key], ...)
               return self._species_handler[key]

           if key not in self._computed:
               self._computed[key] = MyPost_Diagnostic(self._simulation[key], ...)
           return self._computed[key]

**Note on “chainable” post-processes**

Some post-processes (e.g. derivatives) should be chainable:

.. code-block:: python

   d1 = ou.Derivative_Simulation(sim, "x1")
   d2 = ou.Derivative_Simulation(d1, "t")

If your post-process needs to be chainable, avoid hard checks like:

- ``isinstance(simulation, Simulation)``

Instead, validate by capability:

- has ``__getitem__``
- has ``species`` or ``_species``

2) Diagnostic-level wrapper: ``NameOfPostProcess_Diagnostic``
-------------------------------------------------------------

This class performs the actual post-processing while still behaving like a
``Diagnostic``.

**Requirements**

- Inherits from ``Diagnostic``
- Wraps a base diagnostic in ``self._diag``
- Copies relevant metadata (e.g. ``_dt``, ``_dx``, ``_dim``, ``_maxiter``, etc.)
- Implements:

  - ``load_all()``: eager computation into ``self._data`` (shape ``(t, ...)``)
  - ``_frame(index, data_slice=None)``: lazy single-timestep computation

**Important:** you generally do **not** need to implement ``__getitem__``.
The base ``Diagnostic.__getitem__`` already supports:

- int indexing
- time slices
- tuple indexing ``(time_index, spatial_slices...)``

and it calls:

- ``self._frame(time_index, data_slice=data_slice)``

So the minimal contract is:

.. code-block:: python

   class MyPost_Diagnostic(Diagnostic):
       def load_all(self) -> np.ndarray:
           ...

       def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
           ...

**Shape conventions**

- ``load_all()`` returns ``(t, x, y, z)`` (or lower dimensions depending on ``dim``)
- ``_frame()`` returns one timestep ``(x, y, z)`` (or lower dimensions)

**Slicing support**

If your base diagnostic supports efficient disk slicing, pass it through:

.. code-block:: python

   f = self._diag._frame(index, data_slice=data_slice)

That ensures ``diag[10, :, 100:200]`` loads only the requested region.

3) Species handler: ``NameOfPostProcess_Species_Handler``
---------------------------------------------------------

Species handlers are thin wrappers for dictionary-like access.

**Requirements**

- Does **not** inherit from anything
- Stores the wrapped species handler in ``self._species_handler``
- Lazily builds post-processed diagnostics per key and caches them

Example:

.. code-block:: python

   class MyPost_Species_Handler:
       def __init__(self, species_handler, ...):
           self._species_handler = species_handler
           self._computed = {}

       def __getitem__(self, key):
           if key not in self._computed:
               self._computed[key] = MyPost_Diagnostic(self._species_handler[key], ...)
           return self._computed[key]

Rules and common pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Indexing should return only** ``np.ndarray``.
  If you need multiple arrays (e.g., MFT average + fluctuations), create
  auxiliary diagnostic classes and expose them via an additional layer:

  .. code-block:: python

     mft = ou.MFT_Simulation(sim, axis=1)["e3"]
     avg = mft["avg"][0]
     flt = mft["delta"][0]

- **Avoid overriding ``Diagnostic.__getitem__``**.
  If you override it, you must re-implement tuple slicing and time slicing
  correctly, or indexing will break in subtle ways.

- **Implement ``_frame(index, data_slice=None)``** for lazy access.
  This is what enables:

  - tuple indexing and spatial slicing
  - arithmetic operation diagnostics (``diag + other``)
  - consistent interaction with ``load_all``

- **Time FFT cannot be computed from a single timestep**.
  If your transform includes axis 0 (time), you must require ``load_all()``
  (or implement a different algorithm explicitly designed for streaming).
  A “per-timestep” FFT can only work for *spatial* axes.

Consistent usage examples
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import osiris_utils as ou

   sim = ou.Simulation("path/to/input_deck.txt")

   pp = MyPost_Simulation(sim, ...)

   # Non-species diagnostic
   arr = pp["e3"][10]

   # Species diagnostic
   arr = pp["electrons"]["n"][10]

   # Spatial slicing
   arr = pp["e3"][10, :, 100:200]
   arr = pp["electrons"]["n"][0:10, :, 50:]

Pull Requests
~~~~~~~~~~~~~

General expectations for PRs:

- Keep PRs focused: one feature/fix per PR if possible.
- Include an explanation of *why* the change is needed (not only what changed).
- If you introduce a new post-process:
  - follow the structure described above
  - include at least a small usage example
  - add/update documentation if user-facing behavior changes
- If the PR changes physics / normalization / conventions, include a plot or a
  small validation test case.

If you're unsure about design choices (especially around performance or API
consistency), open an issue first to discuss.

Contact
~~~~~~~

If you need help, open an issue or contact João Biu via email:
``joaopedrofbiu@tecnico.ulisboa.pt``
or GitHub:
``https://github.com/joaopedrobiu6``
