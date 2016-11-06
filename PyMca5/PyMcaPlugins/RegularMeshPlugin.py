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
import numpy
from numpy import cos, sin
import sys
from PyMca5 import Plugin1DBase
import os
from PyMca5.PyMcaGui import PyMcaQt as qt

from PyMca5.PyMcaGui import MaskImageWidget

DEBUG = 0

class RegularMeshPlugins(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {}
        self.methodDict['Show Image'] = [self._convert,
                                             "Show mesh as image",
                                             None]

        self.imageWidget = None

    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...
        """
        names = list(self.methodDict.keys())
        names.sort()
        return names

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return self.methodDict[name][2]

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        if DEBUG:
                self.methodDict[name][0]()
        else:
            try:
                self.methodDict[name][0]()
            except:
                print(sys.exc_info())
                raise

    def _convert(self):
        x, y, legend, info = self.getActiveCurve()
        self._x = x[:]
        self._y = y[:]
        if 'Header' not in info:
            raise ValueError("Active curve does not seem to be a mesh scan")

        header = info['Header'][0]

        item = header.split()
        if item[2] not in ['mesh', 'hklmesh']:
            raise ValueError("Active curve does not seem to be a mesh scan")

        self._xLabel = self.getGraphXLabel()
        self._yLabel = self.getGraphYLabel()

        self._motor0Mne = item[3]
        self._motor1Mne = item[7]

        #print("Scanned motors are %s and %s" % (motor0Mne, motor1Mne))

        #Assume an EXACTLY regular mesh for both motors
        self._motor0 = numpy.linspace(float(item[4]), float(item[5]), int(item[6])+1)
        self._motor1 = numpy.linspace(float(item[8]), float(item[9]), int(item[10])+1)

        #Didier's contribution: Try to do something if scan has been interrupted
        if y.size < (int(item[6])+1) * (int(item[10])+1):
            print("WARNING: Incomplete mesh scan")
            self._motor1 = numpy.resize(self._motor1,(y.size/(int(item[6])+1),1))
            y = numpy.resize(y,((y.size/(int(item[6])+1)*(int(item[6])+1)),1))

        try:
            if xLabel.upper() == motor0Mne.upper():
                self._motor0 = self._x
                self._motor0Mne = self._xLabel
            elif xLabel.upper() == motor1Mne.upper():
                self._motor1 = self._x
                self._motor1Mne = self._xLabel
            elif xLabel == info['selection']['cntlist'][0]:
                self._motor0 = self._x
                self._motor0Mne = self._xLabel
            elif xLabel == info['selection']['cntlist'][1]:
                self._motor1 = self._x
                self._motor1Mne = self._xLabel
        except:
            if DEBUG:
                print("XLabel should be one of the scanned motors")

        self._legend = legend
        self._info = info
        yView = y[:]
        yView.shape = len(self._motor1), len(self._motor0)
        if self.imageWidget is None:
            self.imageWidget = MaskImageWidget.MaskImageWidget(\
                                        imageicons=False,
                                        selection=False,
                                        profileselection=True,
                                        scanwindow=self)
        deltaX = self._motor0[1] - self._motor0[0]
        deltaY = self._motor1[1] - self._motor1[0]
        self.imageWidget.setImageData(yView,
                                      xScale=(self._motor0[0],
                                              deltaX),
                                      yScale=(self._motor1[0],
                                              deltaY))
        self.imageWidget.setXLabel(self._motor0Mne)
        self.imageWidget.setYLabel(self._motor1Mne)
        self.imageWidget.show()

MENU_TEXT = "RegularMeshPlugins"
def getPlugin1DInstance(plotWindow, **kw):
    ob = RegularMeshPlugins(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca5.PyMcaGraph import Plot
    app = qt.QApplication([])
    DEBUG = 1
    x = numpy.arange(100.)
    y = x * x
    plot = Plot.Plot()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    plugin = getPlugin1DInstance(plot)
    for method in plugin.getMethods():
        print(method, ":", plugin.getMethodToolTip(method))
    plugin.applyMethod(plugin.getMethods()[0])
    curves = plugin.getAllCurves()
    for curve in curves:
        print(curve[2])
    print("LIMITS = ", plugin.getGraphYLimits())
