#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
"""

A 1D plugin is a module that will be automatically added to the PyMca 1D windows
in order to perform user defined operations of the plotted 1D data.

These plugins will be compatible with any 1D-plot window that provides the functions:
    getActiveCurve
    getAllCurves
    addCurve
    getGraphXLimits
    getGraphYLimits
    getGraphXTitle
    getGraphYTitle
    removeCurve
    
"""
import weakref

class Plugin1DBase(object):
    def __init__(self, plotWindow, **kw):
        """
        plotWindow is the object instantiating the plugin.

        Unless one knows what (s)he is doing, only a proxy should be used.

        I pass the actual instance to keep all doors open.
        """        
        self._plotWindow = weakref.proxy(plotWindow)
        pass

    #Window related functions
    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        """
        return self._plotWindow.setActiveCurve(legend)
    
    def getActiveCurve(self):
        """
        Function to access the currently active curve.
        It returns None in case of not having an active curve.

        Output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.
        """        
        return self._plotWindow.getActiveCurve()

    def getAllCurves(self):
        """
        It returns a list of the form:
            [[xvalues0, xvalues1, ..., xvaluesn],
             [yvalues0, yvalues1, ..., yvaluesn],
             [legend0,  legend1,  ..., legendn ],
             [dict0,    dict1,    ..., dictn]]
        or just an empty list.
        """
        return self._plotWindow.getAllCurves()

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True):
        return self._plotWindow.addCurve(x, y, legend=legend,
                                         info=info,
                                         replace=replace,
                                         replot=replot)

    #Graph related functions
    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        return self._plotWindow.getGraphXLimits()

    def getGraphYLimits(self):
        """
        Get the graph Y limits. 
        """
        return self._plotWindow.getGraphYLimits()

    def removeCurve(self, legend, **kw):
        return self._plotWindow.removeCurve(legend, **kw) 

    def getGraphXTitle(self):
        return self._plotWindow.getGraphXTitle()

    def getGraphYTitle(self):
        return self._plotWindow.getGraphYTitle()

    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        print("getMethods not implemented")
        return []

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return None

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        print("applyMethod not implemented")
        return

        

def getPlugin1DInstance(plotWindow, **kw):
    ob = Plugin1DBase(plotWindow)
    return ob
