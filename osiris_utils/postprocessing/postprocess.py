from ..data.diagnostic import Diagnostic


class PostProcess(Diagnostic):
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

    def __init__(self, name, species=None):
        # Initialize with the same interface as Diagnostic
        super().__init__(species)
        self._name = name
        self._all_loaded = False
        self._data = None

    def process(self, diagnostic):
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
