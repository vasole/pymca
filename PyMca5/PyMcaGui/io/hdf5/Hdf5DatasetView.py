#/*##########################################################################
# Copyright (C) 2004-2017 V.A. Sole, European Synchrotron Radiation Facility
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
"""The :class:`HDF5DatasetView` widget in this module aims to be used
instead of :class:`HDF5DatasetTable` in :class:`QNexusWidget` for
visualization of HDF5 datasets.

It uses the silx :class:`DataViewerFrame` widget with views modified
to handle plugins."""
__author__ = "P. Knobel - ESRF Data Analysis"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"


import os

import PyMca5
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import CloseEventNotifyingWidget

from PyMca5.PyMcaGui.PluginsToolButton import PluginsToolButton

from silx.gui.data.DataViewer import DataViewer
from silx.gui.data.DataViewerSelector import DataViewerSelector
from silx.gui.data import DataViews
from silx.gui.plot import Plot1D


PLUGINS_DIR = []
if os.path.exists(os.path.join(os.path.dirname(PyMca5.__file__), "PyMcaPlugins")):
    from PyMca5 import PyMcaPlugins
    PLUGINS_DIR.append(os.path.dirname(PyMcaPlugins.__file__))
else:
    directory = os.path.dirname(__file__)
    while True:
        if os.path.exists(os.path.join(directory, "PyMcaPlugins")):
            PLUGINS_DIR.append(os.path.join(directory, "PyMcaPlugins"))
            break
        directory = os.path.dirname(directory)
        if len(directory) < 5:
            break

userPluginsDirectory = PyMca5.getDefaultUserPluginsDirectory()
if userPluginsDirectory is not None:
    PLUGINS_DIR.append(userPluginsDirectory)


class Plot1DWithPlugins(Plot1D):
    def __init__(self, parent=None):
        Plot1D.__init__(self, parent)

        self._toolbar = qt.QToolBar(self)
        self.addToolBar(self._toolbar)
        pluginsToolButton = PluginsToolButton(plot=self)

        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin1DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)


class Plot1DViewWithPlugins(DataViews._Plot1dView):
    def createWidget(self, parent):
        return Plot1DWithPlugins(parent=parent)


class DataViewerFrameWithPlugins(qt.QWidget):
    """
    A silx  DataViewerFrame modified to replace
    the silx DataViewer widget with our DataViewerWithPlugins.

    See docstrings for attributes and methods on the original
    :class:`silx.gui.data.DataViewerFrame`.
    """
    displayedViewChanged = qt.pyqtSignal(object)

    dataChanged = qt.pyqtSignal()

    def __init__(self, parent=None):
        super(DataViewerFrameWithPlugins, self).__init__(parent)

        class _DataViewer(DataViewer):
            """Overwrite methods to avoid to create views while the instance
            is not created. `initializeViews` have to be called manually."""

            def _initializeViews(self):
                pass

            def initializeViews(self):
                """Avoid to create views while the instance is not created."""
                super(_DataViewer, self)._initializeViews()

        self.__dataViewer = _DataViewer(self)

        # initialize views when `self.__dataViewer` is set
        self.__dataViewer.initializeViews()
        # start of PyMca specific code
        self.__dataViewer.removeView(
            self.__dataViewer.getViewFromModeId(DataViews.PLOT1D_MODE))
        self.__dataViewer.addView(Plot1DViewWithPlugins(self))
        # end of PyMca specific code

        self.__dataViewer.setFrameShape(qt.QFrame.StyledPanel)
        self.__dataViewer.setFrameShadow(qt.QFrame.Sunken)
        self.__dataViewerSelector = DataViewerSelector(self, self.__dataViewer)
        self.__dataViewerSelector.setFlat(True)

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.__dataViewer, 1)
        layout.addWidget(self.__dataViewerSelector)
        self.setLayout(layout)

        self.__dataViewer.dataChanged.connect(self.__dataChanged)
        self.__dataViewer.displayedViewChanged.connect(self.__displayedViewChanged)

    def __dataChanged(self):
        self.dataChanged.emit()

    def __displayedViewChanged(self, view):
        self.displayedViewChanged.emit(view)

    def availableViews(self):
        return self.__dataViewer.availableViews()

    def currentAvailableViews(self):
        return self.__dataViewer.currentAvailableViews()

    def createDefaultViews(self, parent=None):
        return self.__dataViewer.createDefaultViews(parent)

    def addView(self, view):
        return self.__dataViewer.addView(view)

    def removeView(self, view):
        return self.__dataViewer.removeView(view)

    def setData(self, data):
        self.__dataViewer.setData(data)

    def data(self):
        return self.__dataViewer.data()

    def setDisplayedView(self, view):
        self.__dataViewer.setDisplayedView(view)

    def displayedView(self):
        return self.__dataViewer.displayedView()

    def displayMode(self):
        return self.__dataViewer.displayMode()

    def setDisplayMode(self, modeId):
        return self.__dataViewer.setDisplayMode(modeId)


class Hdf5DatasetView(CloseEventNotifyingWidget.CloseEventNotifyingWidget):
    """QWidget displaying data as raw values in a table widget, or as a
    curve, image or stack in a plot widget.

    The plot features depend on *silx*'s availability.
    """
    def __init__(self, parent=None):
        CloseEventNotifyingWidget.CloseEventNotifyingWidget.__init__(self,
                                                                     parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        # # This should be the proper way of changing a single view without
        # # redefining the complete DataViewer
        # self.viewWidget = DataViewerFrame(self)
        # # for view in self.viewWidget.availableViews():
        # #     if isinstance(view, DataViews._Plot1dView):
        # #         self.viewWidget.removeView(view)
        # # alternative:
        # self.viewWidget.removeView(
        #     self.viewWidget.getViewFromModeId(DataViews.PLOT1D_MODE))
        # self.viewWidget.addView(Plot1DViewWithPlugins(self))

        self.viewWidget = DataViewerFrameWithPlugins(self)

        self.mainLayout.addWidget(self.viewWidget)

    def setDataset(self, dataset):
        self.viewWidget.setData(dataset)



