#/*##########################################################################
# Copyright (C) 2004-2020 European Synchrotron Radiation Facility
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
"""This plugin provide simple math functions, to derivate or invert the
active curve.
"""
__author__ = "V.A. Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
try:
    from PyMca5 import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

from PyMca5.PyMcaGui import PyMca_Icons
import PyMca5.PyMcaMath.SimpleMath as SimpleMath

swapsign = PyMca_Icons.swapsign
derive = PyMca_Icons.derive

class MathPlugins(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
       Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
       self.methodDict = {'Invert':[self.invert,
                                    "Multiply active curve by -1",
                                    swapsign],
                          'Derivate':[self.derivate,
                                      "Derivate zoomed active curve",
                                      derive],
                          'Derivate 3':[self.derivate3,
                                      "3-point Smoothed Derivative of zoomed active curve",
                                      derive],
                          'Derivate 5':[self.derivate5,
                                      "5-point Smoothed Derivative of zoomed active curve",
                                      derive],
                          }
       self.simpleMath = SimpleMath.SimpleMath()

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
        self.methodDict[name][0]()
        return

    def invert(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, y, legend, info = activeCurve [0:4]
        operations = info.get("operations", [])
        operations.append("Invert")
        info['operations'] = operations
        legend = "-("+legend+")"
        self.addCurve(x, -y, legend=legend, info=info, replot=True)

    def derivate(self, option="Single point"):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, y, legend, info = activeCurve [0:4]
        xlimits=self.getGraphXLimits()
        x, y = self.simpleMath.derivate(x, y, xlimits=xlimits, option=option)
        info['ylabel'] = info['ylabel'] + "'"
        operations = info.get("operations", [])
        operations.append("derivate")
        info['operations'] = operations
        info['plot_yaxis'] = "right"
        legend = legend+"'"
        self.addCurve(x, y, legend=legend, info=info, replot=True)

    def derivate3(self):
        return self.derivate(option="SG Smoothed 3 point")

    def derivate5(self):
        return self.derivate(option="SG Smoothed 5 point")

MENU_TEXT = "Built-in Math"
def getPlugin1DInstance(plotWindow, **kw):
    ob = MathPlugins(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca5.PyMcaGraph import Plot
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
