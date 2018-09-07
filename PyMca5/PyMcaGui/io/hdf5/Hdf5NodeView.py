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

import silx
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data import DataViews
from silx.gui.data import NXdataWidgets
from silx.gui.plot import Plot1D, Plot2D
from silx.gui import icons


# Quick and dirty workaround for issue #173
if silx.version == '0.7.0':
    from silx.gui.plot.actions.control import ColormapAction
    from silx.gui.plot.ColormapDialog import ColormapDialog

    def _ColormapAction_createDialog(parent):
        dialog = ColormapDialog()
        dialog.setModal(False)
        return dialog

    ColormapAction._createDialog = staticmethod(_ColormapAction_createDialog)


if silx.version == '0.7.0':
    import sys
    if sys.version_info < (3,):
        import numpy
        def _get_attr_as_string(item, attr_name, default=None):
            attr = item.attrs.get(attr_name, default)
            if hasattr(attr, "encode"):
                # unicode
                return attr.encode("utf-8")
            elif isinstance(attr, numpy.ndarray) and not attr.shape and\
                    hasattr(attr[()], "decode"):
                # byte string as ndarray scalar
                return attr[()].decode("utf-8")
            elif isinstance(attr, numpy.ndarray) and len(attr.shape) and\
                    hasattr(attr[0], "decode"):
                # array of byte-strings
                return [element.decode("utf-8") for element in attr]
            elif isinstance(attr, numpy.ndarray) and not attr.shape:
                # convert array to list
                return attr[()]
            elif isinstance(attr, numpy.ndarray) and len(attr.shape):
                # convert array to list
                return [element for element in attr]
            else:
                return attr
        from silx.io import nxdata
        nxdata.get_attr_as_string = _get_attr_as_string

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
    """Add a plugin toolbutton to a Plot1D"""
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
    """Overload Plot1DView to use the widget with a
    :class:`PluginsToolButton`"""
    def createWidget(self, parent):
        return Plot1DWithPlugins(parent=parent)


class Plot2DWithPlugins(Plot2D):
    """Add a plugin toolbutton to a Plot2D"""
    def __init__(self, parent=None):
        Plot2D.__init__(self, parent)

        self._toolbar = qt.QToolBar(self)
        self.addToolBar(self._toolbar)
        pluginsToolButton = PluginsToolButton(plot=self, parent=self,
                                              method="getPlugin2DInstance")

        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin2DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)


class Plot2DViewWithPlugins(DataViews._Plot2dView):
    def createWidget(self, parent):
        widget = Plot2DWithPlugins(parent=parent)
        widget.setDefaultColormap(self.defaultColormap())
        widget.getColormapAction().setColorDialog(self.defaultColorDialog())
        widget.getIntensityHistogramAction().setVisible(True)
        widget.setKeepDataAspectRatio(True)
        widget.getXAxis().setLabel('X')
        widget.getYAxis().setLabel('Y')
        return widget


class ArrayCurvePlotWithPlugins(NXdataWidgets.ArrayCurvePlot):
    """Adds a plugin toolbutton to an ArrayCurvePlot widget"""
    def __init__(self, parent=None):
        NXdataWidgets.ArrayCurvePlot.__init__(self, parent)

        # patch the Plot1D to make it compatible with plugins
        self._plot._plotType = "SCAN"

        self._toolbar = qt.QToolBar(self)
        self._plot.addToolBar(self._toolbar)
        pluginsToolButton = PluginsToolButton(plot=self._plot,
                                              parent=self)
        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin1DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)


class NXdataCurveViewWithPlugins(DataViews._NXdataCurveView):
    """Use the widget with a :class:`PluginsToolButton`"""
    def createWidget(self, parent):
        return ArrayCurvePlotWithPlugins(parent=parent)


class ArrayImagePlotWithPlugins(NXdataWidgets.ArrayImagePlot):
    """Adds a plugin toolbutton to an ArrayImagePlot widget"""
    def __init__(self, parent=None):
        NXdataWidgets.ArrayImagePlot.__init__(self, parent)

        self._toolbar = qt.QToolBar(self)
        self.getPlot().addToolBar(self._toolbar)
        pluginsToolButton = PluginsToolButton(plot=self.getPlot(),
                                              parent=self,
                                              method="getPlugin2DInstance")
        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin2DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)


class NXdataImageViewWithPlugins(DataViews._NXdataImageView):
    """Use the widget with a :class:`PluginsToolButton`"""
    def createWidget(self, parent):
        widget = ArrayImagePlotWithPlugins(parent)
        widget.getPlot().setDefaultColormap(self.defaultColormap())
        widget.getPlot().getColormapAction().setColorDialog(self.defaultColorDialog())
        return widget


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

        self.viewWidget = DataViewerFrame(self)
        self.viewWidget.replaceView(DataViews.PLOT1D_MODE,
                                    Plot1DViewWithPlugins(self))
        self.viewWidget.replaceView(DataViews.PLOT2D_MODE,
                                    Plot2DViewWithPlugins(self))
        self.viewWidget.replaceView(DataViews.NXDATA_CURVE_MODE,
                                    NXdataCurveViewWithPlugins(self))
        self.viewWidget.replaceView(DataViews.NXDATA_IMAGE_MODE,
                                    NXdataImageViewWithPlugins(self))

        self.mainLayout.addWidget(self.viewWidget)

    def setData(self, dataset):
        self.viewWidget.setData(dataset)



