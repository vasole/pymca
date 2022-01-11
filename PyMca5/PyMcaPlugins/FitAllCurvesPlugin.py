#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
"""This plugin allows to perform a fit on all curves in the plot.
A widget is provided to configure the fit parameters and to specify the
output file.
The fit results are saved as a NeXus HDF5 file, with one entry per
fitted curve."""

try:
    from PyMca5.PyMcaCore import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

from PyMca5.PyMcaGui import PyMcaQt as qt

from PyMca5.PyMcaGui.math.fitting.SimpleFitAllGui import SimpleFitAllGui


class FitAllCurvesPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self._configure_method = "Configure fit"
        self._fit_all_method = "Fit All Curves"
        self.widget = SimpleFitAllGui()

    def getMethods(self, plottype=None):
        return [self._configure_method, self._fit_all_method]

    def getMethodToolTip(self, methodName):
        if methodName == self._configure_method:
            return "Configure fit prior to fitting all curves"
        if methodName == self._fit_all_method:
            return "Open a fit window to run a fit on all curves"
        raise RuntimeError("Unrecognized method name '%s'" % methodName)

    def applyMethod(self, methodName):
        activeCurve = self.getActiveCurve()
        allCurves = self.getAllCurves()
        if not allCurves:
            msg = qt.QMessageBox()
            msg.setWindowTitle("No curves to be fitted")
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText("There are no curves to be fitted on this plot.")
            msg.setStandardButtons(qt.QMessageBox.Ok)
            msg.exec()
            return
        if activeCurve is None:
            activeCurve = allCurves[0]

        xmin, xmax = self.getGraphXLimits()
        self.widget.setSpectrum(activeCurve[0], activeCurve[1],
                                xmin=xmin, xmax=xmax)
        if methodName == self._configure_method:
            self.widget.configureButtonSlot()
        if methodName == self._fit_all_method:
            curves_x, curves_y, legends, xlabels, ylabels = [], [], [], [], []

            for x, y, legend, info in allCurves:
                curves_x.append(x)
                curves_y.append(y)
                legends.append(legend)
                xlabels.append(info["xlabel"])
                ylabels.append(info["ylabel"])
            self.widget.setSpectra(curves_x, curves_y,
                                   legends=legends, xlabels=xlabels, ylabels=ylabels)

            self.widget.show()


MENU_TEXT = "Fit all curves"


def getPlugin1DInstance(plotWindow, **kw):
    ob = FitAllCurvesPlugin(plotWindow)
    return ob
