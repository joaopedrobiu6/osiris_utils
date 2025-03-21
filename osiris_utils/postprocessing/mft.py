from ..utils import *
from ..data.simulation import Simulation
from .postprocess import PostProcess
from ..data.diagnostic import Diagnostic

class MeanFieldTheory(PostProcess):
    """
    Class to compute the mean field theory of a diagnostic. Works as a wrapper for the MFT_Diagnostic class.
    Inherits from PostProcess to ensure all operation overloads work properly.

    Parameters
    ----------
    simulation : Simulation
        The simulation object.
    mft_axis : int
        The axis to compute the mean field theory.

    Example
    -------
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> mft = MeanFieldTheory(sim, 1)
    >>> mft_e1 = mft['e1']
    """

    def __init__(self, simulation, mft_axis=None):
        super().__init__(f"MeanFieldTheory({mft_axis})")
        if not isinstance(simulation, Simulation):
            raise ValueError("Simulation must be a Simulation object.")
        self._simulation = simulation
        self._mft_axis = mft_axis
        self._mft_computed = {}

    def __getitem__(self, key):
        if key not in self._mft_computed:
            self._mft_computed[key] = MFT_Diagnostic(self._simulation[key], self._mft_axis)
        return self._mft_computed[key]
    
    def delete_all(self):
        self._mft_computed = {}

    def delete(self, key):
        if key in self._mft_computed:
            del self._mft_computed[key]
        else:
            print(f"MeanFieldTheory {key} not found in simulation")

    def process(self, diagnostic):
        """Apply mean field theory to a diagnostic"""
        return MFT_Diagnostic(diagnostic, self._mft_axis)

class MFT_Diagnostic(Diagnostic):
    """
    Class to compute mean field theory of a diagnostic.
    Acts as a container for the average and fluctuation components.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute mean field theory along.

    Example
    -------
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> diag = sim['e1']
    >>> mft = MFT_Diagnostic(diag, 1)
    >>> avg = mft['avg']
    >>> delta = mft['delta']
    """

    def __init__(self, diagnostic, mft_axis):
        # Initialize using parent's __init__ with the same species
        if hasattr(diagnostic, '_species'):
            super().__init__(diagnostic._species)
        else:
            super().__init__(None)
            
        self._name = f"MFT[{diagnostic._name}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False
        
        # Components that will be lazily created
        self._components = {}
        
        # Copy all relevant attributes from diagnostic
        for attr in ['_dt', '_dx', '_ndump', '_axis', '_nx', '_x', '_grid', '_dim', '_maxiter']:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def __getitem__(self, key):
        """
        Get a component of the mean field theory.
        
        Parameters
        ----------
        key : str
            Either "avg" for average or "delta" for fluctuations.
        
        Returns
        -------
        Diagnostic
            The requested component.
        """
        if key == "avg":
            if "avg" not in self._components:
                self._components["avg"] = MFT_Diagnostic_Average(self._diag, self._mft_axis)
            return self._components["avg"]
        
        elif key == "delta":
            if "delta" not in self._components:
                self._components["delta"] = MFT_Diagnostic_Fluctuations(self._diag, self._mft_axis)
            return self._components["delta"]
        
        else:
            raise ValueError("Invalid MFT component. Use 'avg' or 'delta'.")
    
    def load_all(self):
        """Load both average and fluctuation components"""
        # This will compute both components at once for efficiency
        if "avg" not in self._components:
            self._components["avg"] = MFT_Diagnostic_Average(self._diag, self._mft_axis)
        
        if "delta" not in self._components:
            self._components["delta"] = MFT_Diagnostic_Fluctuations(self._diag, self._mft_axis)
        
        # Load both components
        self._components["avg"].load_all()
        self._components["delta"].load_all()
        
        # Mark this container as loaded
        self._all_loaded = True
        
        return self._components

class MFT_Diagnostic_Average(Diagnostic):
    """
    Class to compute the average component of mean field theory.
    Inherits from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute the mean field theory.

    Example
    -------
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> diag = sim['e1']
    >>> avg = MFT_Diagnostic_Average(diag, 1)
    """

    def __init__(self, diagnostic, mft_axis):
        # Initialize with the same species as the diagnostic
        if hasattr(diagnostic, '_species'):
            super().__init__(diagnostic._species)
        else:
            super().__init__(None)
            
        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        
        self._name = f"MFT_avg[{diagnostic._name}, {mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

        # Copy all relevant attributes from diagnostic
        for attr in ['_dt', '_dx', '_ndump', '_axis', '_nx', '_x', '_grid', '_dim', '_maxiter']:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def load_all(self):
        """Load all data and compute the average"""
        if self._data is not None:
            print("Data already loaded")
            return self._data
        
        if not hasattr(self._diag, '_data') or self._diag._data is None:
            self._diag.load_all()

        if self._mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        else:
            self._data = self._diag._data.mean(axis=self._mft_axis)

        self._all_loaded = True
        return self._data
    
    def _data_generator(self, index):
        """Generate average data for a specific index"""
        if self._mft_axis is not None:
            # Get the data for this index
            data = self._diag[index]
            # Compute the average (mean) along the specified axis
            # Note: When accessing a slice, axis numbering is 0-based
            avg = data.mean(axis=self._mft_axis-1)
            yield avg
        else:
            raise ValueError("Invalid axis for mean field theory.")
        
    def __getitem__(self, index):
        """Get average at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]
        
        # Otherwise compute on-demand
        return next(self._data_generator(index))
    
class MFT_Diagnostic_Fluctuations(Diagnostic):
    """
    Class to compute the fluctuation component of mean field theory.
    Inherits from Diagnostic to ensure all operation overloads work properly.

    Parameters
    ----------
    diagnostic : Diagnostic
        The diagnostic object.
    mft_axis : int
        The axis to compute the mean field theory.

    Example
    -------
    >>> sim = Simulation('electrons', 'path/to/simulation')
    >>> diag = sim['e1']
    >>> delta = MFT_Diagnostic_Fluctuations(diag, 1)
    """
    
    def __init__(self, diagnostic, mft_axis):
        # Initialize with the same species as the diagnostic
        if hasattr(diagnostic, '_species'):
            super().__init__(diagnostic._species)
        else:
            super().__init__(None)
            
        if mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        
        self._name = f"MFT_delta[{diagnostic._name}, {mft_axis}]"
        self._diag = diagnostic
        self._mft_axis = mft_axis
        self._data = None
        self._all_loaded = False

        # Copy all relevant attributes from diagnostic
        for attr in ['_dt', '_dx', '_ndump', '_axis', '_nx', '_x', '_grid', '_dim', '_maxiter']:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    def load_all(self):
        """Load all data and compute the fluctuations"""
        if self._data is not None:
            print("Data already loaded")
            return self._data
        
        if not hasattr(self._diag, '_data') or self._diag._data is None:
            self._diag.load_all()

        if self._mft_axis is None:
            raise ValueError("Mean field theory axis must be specified.")
        else:
            # Compute the average
            avg = self._diag._data.mean(axis=self._mft_axis)
            # Reshape avg for broadcasting
            broadcast_shape = list(self._diag._data.shape)
            broadcast_shape[self._mft_axis] = 1
            avg_reshaped = avg.reshape(broadcast_shape)
            # Compute the fluctuations
            self._data = self._diag._data - avg_reshaped

        self._all_loaded = True
        return self._data
    
    def _data_generator(self, index):
        """Generate fluctuation data for a specific index"""
        if self._mft_axis is not None:
            # Get the data for this index
            data = self._diag[index]
            # Compute the average (mean) along the specified axis
            # Note: When accessing a slice, axis numbering is 0-based
            avg = data.mean(axis=self._mft_axis-1)
            # Expand dimensions to enable broadcasting
            avg_reshaped = np.expand_dims(avg, axis=self._mft_axis-1)
            # Compute fluctuations
            delta = data - avg_reshaped
            yield delta
        else:
            raise ValueError("Invalid axis for mean field theory.")
        
    def __getitem__(self, index):
        """Get fluctuations at a specific index"""
        if self._all_loaded and self._data is not None:
            return self._data[index]
        
        # Otherwise compute on-demand
        return next(self._data_generator(index))