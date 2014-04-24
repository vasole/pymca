#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
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
                  PCAWindow, SGWindow, SNIPWindow, \
                  StripBackgroundWidget
try:
    from .math import SIFTAlignmentWindow
except:
    # sift or PyOpenCL might not be installed
    pass
from .pymca import StackPluginResultsWindow, ExternalImagesWindow

from .pymca import RGBImageCalculator
from .io import QSourceSelector
from .io import QEdfFileWidget
from .pymca import RGBCorrelator
from .physics.xrf import MaterialEditor
# Should they be moved to files in this directory?
try:
    from .io.hdf5 import HDF5Widget, QNexusWidget
except:
#    # HDF5 is not a forced dependency
    pass

from .pymca import StackSimpleFitWindow
from .pymca import QStackWidget
