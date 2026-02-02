from __future__ import annotations

from ..data.simulation import Simulation

__all__ = ["PostProcess"]


class PostProcess(Simulation):
    """
    Base class for post-processing operations.
    Inherits from Diagnostic to ensure all operation overloads work.

    Parameters
    ----------
    name : str
        Name of the post-processing operation.
    species : str
        The species to analyze.
    """

    def __init__(self, name: str, species: str = None):
        # Initialize with the same interface as Diagnostic
        self._name = name
        self._all_loaded = False
        self._data = None

    def process(self, simulation: Simulation) -> Simulation:
        """
        Apply the post-processing to a diagnostic.
        Must be implemented by subclasses.

        Parameters
        ----------
        diagnostic : Diagnostic
            The diagnostic to process.

        Returns
        -------
        Diagnostic or PostProcess
            The processed diagnostic.
        """
        raise NotImplementedError("Subclasses must implement process method")


# PostProcessing_Simulation
# PostProcessing_Diagnostic
