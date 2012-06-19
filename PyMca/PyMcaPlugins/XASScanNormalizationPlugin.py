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
__author__ = "V.A. Sole - ESRF Software Group"
import numpy
try:
    from PyMca import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

try:
    from PyMca import XASNormalization
    from PyMca import XASNormalizationWindow
    from PyMca import SpecfitFuns
except ImportError:
    print("XASScanNormalizationPlugin importing from somewhere else")
    import XASNormalization
    import XASNormalizationWindow
    import SpecfitFuns


class XASScanNormalizationPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {}
        text = "Configure normalization parameters."
        function = self.configure
        info = text
        icon = None
        self.methodDict["Configure"] =[function,
                                       info,
                                       icon]
        function = self.XASNormalize
        text = "Replace all curves by normalized ones."
        info = text
        icon = None
        self.methodDict["Normalize"] =[function,
                                       info,
                                       icon]
        self.widget = None
        self.parameters = None
        
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

    def configure(self):
        #get active curve
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            raise ValueError("Please select an active curve")
            return
        x, y, legend0, info = activeCurve
        if self.widget is None:
            self._createWidget(y, energy=x)

        ret = self.widget.exec_()
        if ret:
            self.parameters = self.widget.getParameters()

    def _createWidget(self, spectrum, energy=None):
        parent = None
        self.widget = XASNormalizationWindow.XASNormalizationDialog(parent,
                                                            spectrum,
                                                            energy=energy)
        self.parameters = self.widget.getParameters()

    def XASNormalize(self):
        #all curves
        curves = self.getAllCurves()
        nCurves = len(curves)
        if nCurves < 1:
            raise ValueError("At least one curve needed")
            return
        
        #get active curve
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            raise ValueError("Please select an active curve")
            return
        
        x, y, legend0, info = activeCurve

        #sort the values
        idx = numpy.argsort(x, kind='mergesort')
        x0 = numpy.take(x, idx)
        y0 = numpy.take(y, idx)
        xmin, xmax = self.getGraphXLimits()

        # get calculation parameters
        if self.widget is None:
            self._createWidget(y0, x0)

        parameters = self.parameters
        if parameters['auto_edge']:
            edge = None
        else:
            edge = parameters['edge_energy']
        energy = x
        pre_edge_regions = parameters['pre_edge']['regions']
        post_edge_regions = parameters['post_edge']['regions']
        algorithm ='polynomial'
        algorithm_parameters = {}
        algorithm_parameters['pre_edge_order'] = parameters['pre_edge']\
                                                         ['polynomial']
        algorithm_parameters['post_edge_order'] = parameters['post_edge']\
                                                         ['polynomial']
        i = 0
        lastCurve = None
        for curve in curves:            
            x, y, legend, info = curve[0:4]
            #take the portion ox x between limits
            idx = numpy.nonzero((x>=xmin) & (x<=xmax))[0]
            if not len(idx):
                #no overlap
                continue
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)

            idx = numpy.nonzero((x0>=x.min()) & (x0<=x.max()))[0]
            if not len(idx):
                #no overlap
                continue
            xi = numpy.take(x0, idx)
            yi = numpy.take(y0, idx)
            
            #perform interpolation
            xi.shape = -1, 1
            yw = SpecfitFuns.interpol([x], y, xi, yi.min())

            # try: ... except: here?
            yw.shape = -1
            xi.shape = -1
            x, y = XASNormalization.XASNormalization(yw,
                                energy=xi,
                                edge=edge,
                                pre_edge_regions=pre_edge_regions,
                                post_edge_regions=post_edge_regions,
                                algorithm=algorithm,
                                algorithm_parameters=algorithm_parameters)[0:2]
            #
            if i == 0:
                replace = True
                replot = True
                i = 1
            else:
                replot = False
                replace = False
            newLegend = " ".join(legend.split(" ")[:-2])
            if not newLegend.startswith('Norm.'):
                newLegend = "Norm. " + newLegend
            self.addCurve(x, y,
                          legend=newLegend,
                          info=info,
                          replot=replot,
                          replace=replace)
            lastCurve = [x, y, newLegend]
        self.addCurve(lastCurve[0],
                      lastCurve[1],
                      legend=lastCurve[2],                      
                      info=info,
                      replot=True,
                      replace=False)

MENU_TEXT = "XAS Normalization"
def getPlugin1DInstance(plotWindow, **kw):
    ob = XASScanNormalizationPlugin(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca import Plot1D
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D.Plot1D()
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
