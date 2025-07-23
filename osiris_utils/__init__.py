from importlib import metadata as _meta

try:
    # Use the version that was baked into the wheel / sdist at build time
    __version__: str = _meta.version(__name__)
except _meta.PackageNotFoundError:  # package is being run from a checkout
    # Fallback for `pip install -e .` or direct source use
    __version__ = "0.0.0.dev0"

from .data.data import (
    OsirisData,
    OsirisGridFile,
    OsirisHIST,
    OsirisRawFile,
    OsirisTrackFile,
)
from .data.diagnostic import Diagnostic
from .data.simulation import Simulation, Species_Handler
from .decks.decks import InputDeckIO
from .decks.species import Specie
from .postprocessing.derivative import Derivative_Diagnostic, Derivative_Simulation
from .postprocessing.fft import FFT_Diagnostic, FFT_Simulation
from .postprocessing.field_centering import (
    FieldCentering_Diagnostic,
    FieldCentering_Simulation,
)
from .postprocessing.heatflux_correction import (
    HeatfluxCorrection_Diagnostic,
    HeatfluxCorrection_Simulation,
)
from .postprocessing.mft import (
    MFT_Diagnostic,
    MFT_Diagnostic_Average,
    MFT_Diagnostic_Fluctuations,
    MFT_Simulation,
)
from .postprocessing.mft_for_gridfile import MFT_Single
from .postprocessing.postprocess import PostProcess
from .postprocessing.pressure_correction import (
    PressureCorrection_Diagnostic,
    PressureCorrection_Simulation,
)
from .utils import (
    courant2D,
    filesize_estimation,
    integrate,
    read_data,
    save_data,
    time_estimation,
    transverse_average,
)

__all__ = [
    # Data Singles
    "OsirisGridFile",
    "OsirisRawFile",
    "OsirisData",
    "OsirisHIST",
    "OsirisTrackFile",
    # Data Diagnostic and Simulation
    "Simulation",
    "Diagnostic",
    "Species_Handler",
    # Decks
    "Specie",
    "InputDeckIO",
    # PostProcessing
    "PostProcess",
    "Derivative_Simulation",
    "Derivative_Diagnostic",
    "FFT_Diagnostic",
    "FFT_Simulation",
    "MFT_Simulation",
    "MFT_Diagnostic",
    "MFT_Diagnostic_Average",
    "MFT_Diagnostic_Fluctuations",
    "FieldCentering_Simulation",
    "FieldCentering_Diagnostic",
    "PressureCorrection_Simulation",
    "PressureCorrection_Diagnostic",
    "HeatfluxCorrection_Simulation",
    "HeatfluxCorrection_Diagnostic",
    # Single file MFT
    "MFT_Single",
    # Utilities
    "time_estimation",
    "filesize_estimation",
    "transverse_average",
    "integrate",
    "save_data",
    "read_data",
    "courant2D",
]
