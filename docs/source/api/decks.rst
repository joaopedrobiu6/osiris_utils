Deck Handling
=============

This section documents the utilities for handling OSIRIS input decks.

InputDeckIO
-----------

.. autoclass:: osiris_utils.decks.decks.InputDeckIO
   :members:
   :special-members: __init__, __getitem__
   :undoc-members:
   :noindex:

   Class to handle parsing and modifying OSIRIS input decks.

   **Key Features:**

   * Parse input decks into python objects
   * Modify parameters programmatically
   * Save modified decks to new files
   * Automatic handling of OSIRIS syntax idiosyncrasies

   **Usage Example:**

   .. code-block:: python

       from osiris_utils import InputDeckIO

       # Load an input deck
       deck = InputDeckIO("osiris.inp")

       # Get simulation dimension
       print(f"Simulation dimension: {deck.dim}")

       # Modify a parameter
       # (Assuming there is a 'time_step' section with 'dt' parameter)
       deck.set_param("time_step", "dt", "0.05")

       # Save to a new file
       deck.print_to_file("osiris_modified.inp")

Coordinate Conversion
---------------------

.. function:: osiris_utils.decks.decks.deval

   .. autofunction:: osiris_utils.decks.decks.deval
