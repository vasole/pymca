#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
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
                  PCAWindow, SGWindow, SIFTAlignmentWindow, SNIPWindow, \
                  StripBackgroundWidget
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
