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
"""
This plugin opens a scan window displaying a number of spectra.

The user can choose to display 10, 100, 1000 or all spectra.
"""
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from PyMca5 import StackPluginBase
from PyMca5.PyMca import ScanWindow
import numpy
import logging

_logger = logging.getLogger(__name__)


class ShowSpectra(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        function = self.showSpectra
        info = "Show 1D Plot with all spectra"
        icon = None
        self.methodDict["Show_All"] =[function,
                                       info,
                                       icon]
        function = self.showOneOver10
        info = "Show 1/10th of all spectra"
        icon = None
        self.methodDict["Show_10"] =[function,
                                       info,
                                       icon]

        function = self.showOneOver100
        info = "Show 1/100 th of all spectra"
        icon = None
        self.methodDict["Show_100"] =[function,
                                       info,
                                       icon]

        function = self.showOneOver1000
        info = "Show 1/1000 th of all spectra"
        icon = None
        self.methodDict["Show_1000"] =[function,
                                       info,
                                       icon]
        self.widget = None
    
    def stackUpdated(self):
        if self.widget is not None:
            self.widget.close()
        self.widget = None

    def getMethods(self, plottype=None):
        keys = list(self.methodDict.keys())
        keys.sort()
        return keys

    def getMethodToolTip(self, methodName):
        return self.methodDict[methodName][1]

    def applyMethod(self, methodName):
        delta = methodName.split("_")[1]
        if delta == "All":
            self.showSpectra()
        else:
            self.showSpectra(step=int(delta))

    def showOneOver10(self):
        return self.showSpectra(step=10)

    def showOneOver100(self):
        return self.showSpectra(step=100)

    def showOneOver1000(self):
        return self.showSpectra(step=1000)
    
    def showSpectra(self, step=1):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        activeCurve = self.getActiveCurve()
        if activeCurve in [None, []]:
            return
        x, spectrum, legend, info = activeCurve
        if self.widget is None:
            self.widget = ScanWindow.ScanWindow()
        data = stack.data
        replot = False
        if step in [None, 1]:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if (i==0)  and (j==0):
                        replace = True
                    else:
                        replace = False
                    self.widget.addCurve(x, data[i, j], legend="Row %03d Col %03d" % (i, j), replace=replace, replot=replot)
        else:
            counter = 0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if not counter % step:
                        if (i==0)  and (j==0):
                            replace = True
                        else:
                            replace = False
                        self.widget.addCurve(x, data[i, j],
                                legend="Row %03d Col %03d" % (i, j),
                                replace=replace, replot=replot)
                    counter += 1
        self.widget.resetZoom()
        self.widget.show()
        self.widget.raise_()

MENU_TEXT = "Show Spectra"
def getStackPluginInstance(plotWindow, **kw):
    ob = ShowSpectra(plotWindow)
    return ob
