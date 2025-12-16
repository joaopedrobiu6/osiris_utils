Contributing to OSIRIS Utils
============================

Thank you for contributing!

The following is a set of guidelines for contributing to our code. These are
mostly guidelines, not rules, so feel free to use your best judgment!

Reporting Bugs
~~~~~~~~~~~~~~

How Do I Submit A (Good) Bug Report?
------------------------------------

Bugs are tracked as `GitHub issues <https://github.com/joaopedrobiu6/osiris_utils/issues/>`__.

Explain the problem and include additional details to help maintainers
reproduce the problem:

-  **Use a clear and descriptive title** for the issue to identify the
   problem.
-  **Describe the exact steps which reproduce the problem** in as many
   details as possible.
-  **Provide specific examples to demonstrate the steps**. Include links
   to files or copy/pasteable snippets.
-  **Describe the behavior you observed after following the steps** and
   point out what exactly is the problem with that behavior.
-  **Explain which behavior you expected to see instead and why.**
-  **Include plots** of results that you believe to be wrong.

Adding Post-Processing routines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How we contribute by adding post-processing classes? Here's a breakdown:

1. First, you should implement a class that operates on simulations, usually called - ``NameOfPostProcess_Simulation``:

* Inherits from ``PostProcess``
* It should have a ``__getitem__(self, key)`` with if statements to separate the species diagnostics from the general ones

2. A class to deal with the diagnostics - ``NameOfPostProcess_Diagnostic``

* Inherits from ``Diagnostic``
* Must have a ``load_all``, ``_data_generator`` and ``__getitem__``. They need to have these names to work well with ``Diagnostic`` objects. The first two are more specific to the process that you want, and you can understand from the already implemented what stuff you need to take into account (usually that `load_all` makes the data have `(t, x, y, z)` and `_data_generator` only yields `(x, y, z)`, and once `load_all` is used, `_data_generator` should use already read data)
* The access by indexing should only return ``np.ndarray``. If you need different stuff, for example, a list of ``np.ndarray``, implement it in auxiliary classes. 
* Eg.: The Mean Field Theory should give us access to the averaged field and the fluctuations for each quantity. Since this would be 2 arrays and not 1, MFT has two diagnostic classes, one for averages and another for fluctuations.

3. A class to deal with species-related diagnostics - ``NameOfProcess_Species_Handler``

* Does not inherit from anything
* Just works as a wrapper for the dictionary-like syntax to be consistent.

By implementing it and taking these steps into consideration, the integration with the rest of the package should be fairly straightforward. 
However, since post-processes may differ a lot from each other, some things may be more of a case-to-case situation. 
If any help is needed, please open an issue or contact João Biu via `email <joaopedrofbiu@tecnico.ulisboa.pt>`__ or `GitHub <https://github.com/joaopedrobiu6>`__.

To be consistent with the syntax of other classes:

.. code-block :: python
                                                                                                                                 
  sim = ou.Simulation("folder", "input_deck")
  PostProcessExample_for_sim = PostProcessExample_Simulation(sim, ...)
  PostProcessExample_for_sim["non_species_related_quantity"][index] # this should return the values of the post-processed quantity for that index
  PostProcessExample_for_sim["species"]["quantity"][index] # this should return the value of the post-processed species-related quantity at the specified index
