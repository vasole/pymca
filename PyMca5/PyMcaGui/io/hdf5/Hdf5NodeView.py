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
"""The :class:`Hdf5NodeView` widget in this module aims to replace
:class:`HDF5DatasetTable` in :class:`QNexusWidget` for visualization of HDF5
datasets and groups, with support of NXdata groups as plot.

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

from silx.gui.data.DataViewerFrame import DataViewerFrame
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
        self._plotType = "SCAN"    # needed by legacy plugins

        self._toolbar = qt.QToolBar(self)
        self.addToolBar(self._toolbar)
        pluginsToolButton = PluginsToolButton(plot=self, parent=self)

        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin1DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)


class Plot1DViewWithPlugins(DataViews._Plot1dView):
    def createWidget(self, parent):
        return Plot1DWithPlugins(parent=parent)


class DataViewerFrameWithPlugins(DataViewerFrame):
    """Overloaded DataViewerFrame with the 1D view replaced by
    Plot1DViewWithPlugins"""
    def createDefaultViews(self, parent=None):
        views = list(DataViewerFrame.createDefaultViews(self, parent=parent))
        oldView = [v for v in views if v.modeId() == DataViews.PLOT1D_MODE][0]
        newView = Plot1DViewWithPlugins(parent=parent)
        views[views.index(oldView)] = newView
        return views


class Hdf5NodeView(CloseEventNotifyingWidget.CloseEventNotifyingWidget):
    """QWidget displaying data as raw values in a table widget, or as a
    curve, image or stack in a plot widget. It can also display information
    related to HDF5 groups (attributes, compression, ...) and interpret
    a NXdata group to plot its data.

    The plot features depend on *silx*'s availability.
    """
    def __init__(self, parent=None):
        CloseEventNotifyingWidget.CloseEventNotifyingWidget.__init__(self,
                                                                     parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

        # # This should be the proper way of changing a single view without
        # # overloading DataViewerFrame. See silx issue #1183.
        # self.viewWidget = DataViewerFrame(self)
        # self.viewWidget.removeView(
        #     self.viewWidget.getViewFromModeId(DataViews.PLOT1D_MODE))
        # self.viewWidget.addView(Plot1DViewWithPlugins(self))

        self.viewWidget = DataViewerFrameWithPlugins(self)

        self.mainLayout.addWidget(self.viewWidget)

    def setData(self, dataset):
        self.viewWidget.setData(dataset)



