from .pymca import PyMcaFileDialogs
from .plotting import PyMca_Icons, PyMcaPrintPreview
from .plotting.PyMca_Icons import IconDict
from .pymca import QPyMcaMatplotlibSave1D, QPyMcaMatplotlibSave
from .misc import DoubleSlider, CalculationThread, SubprocessLogWidget, \
                  NumpyArrayTableWidget, FrameBrowser, CloseEventNotifyingWidget
from .plotting import PlotWidget, PlotWindow, MaskImageWidget, \
                      ColormapDialog, \
                      RGBCorrelatorGraph
from .physics import McaAdvancedFit, FastXRFLinearFitWindow, \
                     XASNormalizationWindow, XASSelfattenuationWindow, \
                     QPeriodicTable, ElementsInfo, PeakIdentifier
from .pymca import ScanWindow, ExternalImagesWindow
from .math.fitting import SpecfitGui, SimpleFitGui, SimpleFitBatchGui

from .pymca import StackPluginResultsWindow
from .math import FFTAlignmentWindow, NNMADialog, NNMAWindow, PCADialog, \
                  PCAWindow, SGWindow, SIFTAlignmentWindow, SNIPWindow, \
                  StripBackgroundWidget
from .pymca import StackPluginResultsWindow, ExternalImagesWindow

from .pymca import RGBImageCalculator
from .io import QSourceSelector
from .io import QEdfFileWidget
from .pymca import RGBCorrelator
from .physics.xrf import MaterialEditor
#try:
#    from .io.hdf5 import HDF5Widget, QNexusWidget
#except:
#    # HDF5 is not a forced dependency
#    pass
