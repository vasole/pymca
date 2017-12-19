#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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


from PyMca5.PyMcaCore import Plugin1DBase

from PyMca5.PyMcaGui.math.fitting.SpecfitGui import SpecfitGui


class FitAllPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)

        self.widget = SpecfitGui(parent=None, config=True, status=True, buttons=True)
        self.widget.guibuttons.EstimateButton.setToolTip("Estimate first curve")
        # redefine "Run fit" to estimate and fit all curves
        self.widget.guibuttons.StartfitButton.clicked.disconnect()
        self.widget.guibuttons.StartfitButton.clicked.connect(self.fitAllCurves)
        self.widget.guibuttons.StartfitButton.setToolTip(
                "Estimate and fit all curves using current config")

    # Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...
        """
        return ["Fit all"]

    def _testMethodName(self, name):
        if name not in self.getMethods():
            raise AttributeError("'%s' is not a valid method name" % name)

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        self._testMethodName(name)
        return "Fit all curves"

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        self._testMethodName(name)
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        self._testMethodName(name)
        self.initFitDialog()

    def initFitDialog(self):
        # initialize fit widget with active or first curve
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            raise ValueError("No curves to be fitted on this plot")
        x, y, legend0, info = activeCurve[:4]
        xmin, xmax = self.getGraphXLimits()
        idx = (x >= xmin) & (x <= xmax)
        x = x[idx]
        y = y[idx]

        self.widget.setdata(x, y)
        self.widget.show()
        self.widget.raise_()

    def fitAllCurves(self):
        allCurves = self.getAllCurves()
        xmin, xmax = self.getGraphXLimits()

        for curve in allCurves:
            x, y, legend0, info = curve[:4]
            idx = (x >= xmin) & (x <= xmax)
            x = x[idx]
            y = y[idx]
            self.widget.setdata(x, y)
            self.widget.estimate()
            self.widget.startfit()
            for paramdict in self.widget.specfit.paramlist:
                print(paramdict["name"], paramdict["fitresult"])


MENU_TEXT = "Fit all curves"
def getPlugin1DInstance(plotWindow, **kw):
    ob = FitAllPlugin(plotWindow)
    return ob
