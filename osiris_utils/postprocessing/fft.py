import numpy as np
from ..data.diagnostic import Diagnostic
from tqdm import tqdm

class FastFourierTransform:
    """
    Class to handle the Fast Fourier Transform on data.

    Parameters
    ----------
    Diagnostic : Diagnostic
        The simulation object.
    axis : int
        The axis to compute the FFT.
    """
    def __init__(self, Diagnostic, axis):
        self.sim = Diagnostic
        self._compute_fft(axis=axis)

    def _compute_fft(self, axis):
        if not hasattr(self.sim, 'data') or self.sim.data is None:
            print("No data to compute the FFT. Loading all data.")
            self.sim.load_all()

        hanning_window = np.hanning(self.sim.data.shape[0]).reshape(-1, 1, 1)
        data_hanned = hanning_window * self.sim.data

        with tqdm(total=1, desc="FFT calculation") as pbar:
            data_fft = np.abs(np.fft.fftn(data_hanned, axes=axis))**2
            pbar.update(0.5)
            self._data_fft = np.fft.fftshift(data_fft, axes=axis)
            pbar.update(0.5)

        self._kmax = np.pi / self.sim.dx
        self._omega_max = np.pi / self.sim.dt

    # adaptar para que possa fazer so para um time step

    @property
    def fft(self):
        return self._data_fft
    
    @property
    def kmax(self):
        return self._kmax
    
    @property
    def omega_max(self):
        return self._omega_max
    


        
