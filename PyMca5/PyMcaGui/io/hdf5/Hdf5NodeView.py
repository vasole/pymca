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
import numpy

import PyMca5
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import CloseEventNotifyingWidget

from PyMca5.PyMcaGui.PluginsToolButton import PluginsToolButton

from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data import DataViews
from silx.gui.plot import Plot1D
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui import icons


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


class ArrayCurvePlotWithPlugins(qt.QWidget):
    """Add a plugin toolbutton to an ArrayCurvePlot widget"""
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.__signal = None
        self.__signal_name = None
        self.__signal_errors = None
        self.__axis = None
        self.__axis_name = None
        self.__axis_errors = None
        self.__values = None

        self.__first_curve_added = False

        self._plot = Plot1D(self)
        self._plot.setDefaultColormap(   # for scatters
                {"name": "viridis",
                 "vmin": 0., "vmax": 1.,   # ignored (autoscale) but mandatory
                 "normalization": "linear",
                 "autoscale": True})

        self.selectorDock = qt.QDockWidget("Data selector", self._plot)
        # not closable
        self.selectorDock.setFeatures(qt.QDockWidget.DockWidgetMovable |
                                qt.QDockWidget.DockWidgetFloatable)
        self._selector = NumpyAxesSelector(self.selectorDock)
        self._selector.setNamedAxesSelectorVisibility(False)
        self.__selector_is_connected = False
        self.selectorDock.setWidget(self._selector)
        self._plot.addTabbedDockWidget(self.selectorDock)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot,  0, 0)

        self.setLayout(layout)

        # plugin support
        self._plot._plotType = "SCAN"    # attribute needed by legacy plugins

        self._toolbar = qt.QToolBar(self)

        self._plot.addToolBar(self._toolbar)

        pluginsToolButton = PluginsToolButton(plot=self._plot,
                                              parent=self)

        if PLUGINS_DIR:
            pluginsToolButton.getPlugins(
                    method="getPlugin1DInstance",
                    directoryList=PLUGINS_DIR)
        self._toolbar.addWidget(pluginsToolButton)

    def setCurveData(self, y, x=None, values=None,
                     yerror=None, xerror=None,
                     ylabel=None, xlabel=None, title=None):
        """

        :param y: dataset to be represented by the y (vertical) axis.
            For a scatter, this must be a 1D array and x and values must be
            1-D arrays of the same size.
            In other cases, it can be a n-D array whose last dimension must
            have the same length as x (and values must be None)
        :param x: 1-D dataset used as the curve's x values. If provided,
            its lengths must be equal to the length of the last dimension of
            ``y`` (and equal to the length of ``value``, for a scatter plot).
        :param values: Values, to be provided for a x-y-value scatter plot.
            This will be used to compute the color map and assign colors
            to the points.
        :param yerror: 1-D dataset of errors for y, or None
        :param xerror: 1-D dataset of errors for x, or None
        :param ylabel: Label for Y axis
        :param xlabel: Label for X axis
        :param title: Graph title
        """
        self.__signal = y
        self.__signal_name = ylabel or "Y"
        self.__signal_errors = yerror
        self.__axis = x
        self.__axis_name = xlabel
        self.__axis_errors = xerror
        self.__values = values

        if self.__selector_is_connected:
            self._selector.selectionChanged.disconnect(self._updateCurve)
            self.__selector_is_connected = False
        self._selector.setData(y)
        self._selector.setAxisNames([ylabel or "Y"])

        if len(y.shape) < 2:
            self.selectorDock.hide()
        else:
            self.selectorDock.show()

        self._plot.setGraphTitle(title or "")
        self._plot.getXAxis().setLabel(self.__axis_name or "X")
        self._plot.getYAxis().setLabel(self.__signal_name)
        self._updateCurve()

        if not self.__selector_is_connected:
            self._selector.selectionChanged.connect(self._updateCurve)
            self.__selector_is_connected = True

    def _updateCurve(self):
        y = self._selector.selectedData()
        x = self.__axis
        if x is None:
            x = numpy.arange(len(y))
        elif numpy.isscalar(x) or len(x) == 1:
            # constant axis
            x = x * numpy.ones_like(y)
        elif len(x) == 2 and len(y) != 2:
            # linear calibration a + b * x
            x = x[0] + x[1] * numpy.arange(len(y))
        legend = self.__signal_name + "["
        for sl in self._selector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        if self.__signal_errors is not None:
            y_errors = self.__signal_errors[self._selector.selection()]
        else:
            y_errors = None

        self._plot.remove(kind=("curve", "scatter"))

        # values: x-y-v scatter
        if self.__values is not None:
            self._plot.addScatter(x, y, self.__values,
                                  legend=legend,
                                  xerror=self.__axis_errors,
                                  yerror=y_errors)

        else:
            self._plot.addCurve(x, y, legend=legend,
                                xerror=self.__axis_errors,
                                yerror=y_errors)

        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__axis_name)
        self._plot.getYAxis().setLabel(self.__signal_name)

    def clear(self):
        self._plot.clear()


class NXdataCurveViewWithPlugins(DataViews._NXdataCurveView):
    """Use the widget with a :class:`PluginsToolButton`"""
    def createWidget(self, parent):
        return ArrayCurvePlotWithPlugins(parent=parent)


class NXdataViewWithPlugins(DataViews.CompositeDataView):
    """Re-implement DataViews._NXdataView to use the 1D view with
    a plugin toolbutton in the composite view."""
    def __init__(self, parent):
        super(NXdataViewWithPlugins, self).__init__(
            parent=parent,
            label="NXdata",
            icon=icons.getQIcon("view-nexus"))

        self.addView(DataViews._NXdataScalarView(parent))
        self.addView(NXdataCurveViewWithPlugins(parent))
        self.addView(DataViews._NXdataXYVScatterView(parent))
        self.addView(DataViews._NXdataImageView(parent))
        self.addView(DataViews._NXdataStackView(parent))


class DataViewerFrameWithPlugins(DataViewerFrame):
    """Overloaded DataViewerFrame with the 1D view replaced by
    Plot1DViewWithPlugins"""
    def createDefaultViews(self, parent=None):
        views = list(DataViewerFrame.createDefaultViews(self, parent=parent))

        # replace 1d view
        oldView = [v for v in views if v.modeId() == DataViews.PLOT1D_MODE][0]
        newView = Plot1DViewWithPlugins(parent=parent)
        views[views.index(oldView)] = newView

        # replace NXdataView
        oldView = [v for v in views if isinstance(v, DataViews._NXdataView)][0]
        newView = NXdataViewWithPlugins(parent=parent)
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



