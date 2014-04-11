#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
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
                                    derive]}
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

    def derivate(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, y, legend, info = activeCurve [0:4]
        xlimits=self.getGraphXLimits()
        x, y = self.simpleMath.derivate(x, y, xlimits=xlimits)
        info['ylabel'] = info['ylabel'] + "'"
        operations = info.get("operations", [])
        operations.append("derivate")
        info['operations'] = operations
        legend = legend+"'"
        self.addCurve(x, y, legend=legend, info=info, replot=True)
        

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
