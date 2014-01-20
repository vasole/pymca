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
__license__ = "LGPL"
__doc__ = """

Any plot window willing to accept plugins should implement the methods
defined in this class.

That way the plot will respect the Plot backend interface besides additional
methods:
The plugins will be compatible with any plot window that provides the methods:
    getActiveCurve
    getActiveImage
    getAllCurves
    getCurve
    getMonotonicCurves
    hideCurve
    hideImage
    isCurveHidden
    isImageHidden
    setActiveCurve
    showCurve
    showImage

The simplest way to achieve that is to inherit from Plot
"""

import os
import sys
import glob
try:
    from numpy import argsort, nonzero, take
except ImportError:
    print("WARNING: numpy not present")

from . import PlotBackend
from . import PluginLoader

DEBUG = 0

class PlotBase(PlotBackend.PlotBackend, PluginLoader.PluginLoader):
    def __init__(self, parent=None):
        # This serves to define the plugins
        PluginLoader.PluginLoader.__init__(self)

        # And this to complete the interface
        PlotBackend.PlotBackend.__init__(self, parent)

    def getActiveCurve(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the active curve or list [x, y, legend, info]
        :rtype: string or list 
        Function to access the graph currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active curve (or None) is returned.
        """
        print("PlotBase getActiveCurve not implemented")
        return None

    def getActiveImage(self, just_legend=False):
        print("PlotBase getActiveImage not implemented")
        return None

    def getAllCurves(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the curves or list [[x, y, legend, info], ...]
        :rtype: list of strings or list of curves 

        It returns an empty list in case of not having any curve.
        If just_legend is False:
            It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
            or just an empty list.
        If just_legend is True:
            It returns a list of the form:
                [legend0, legend1, ..., legendn]
            or just an empty list.
        """
        print("PlotBase getAllCurves not implemented")
        return []

    def getCurve(self, legend):
        """
        :param legend: legend assiciated to the curve
        :type legend: boolean
        :return: list [x, y, legend, info]
        :rtype: list 
        Function to access the graph currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.
        """
        print("PlotBase getCurve not implemented")
        return []

    def getMonotonicCurves(self):
        """
        Convenience method that calls getAllCurves and makes sure that all of
        the X values are strictly increasing.
        :return: It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
        """
        allCurves = self.getAllCurves() * 1
        for i in range(len(allCurves)):
            x, y, legend, info = curve[0:4]
            if self.isCurveHidden(legend):
                continue
            # Sort
            idx = numpy.argsort(x, kind='mergesort')
            xproc = take(x, idx)
            yproc = take(y, idx)
            # Ravel, Increase
            xproc = xproc.ravel()
            idx = nonzero((xproc[1:] > xproc[:-1]))[0]
            xproc = take(xproc, idx)
            yproc = take(yproc, idx)
            allCurves[i][0:2] = x, y
        return allCurves

    def hideCurve(self, legend, replot=True):
        """
        Remove the curve associated to the legend form the graph.
        It is still kept in the list of curves.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the curve to be hidden
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBase hideCurve not implemented")
        return

    def hideImage(self, legend, replot=True):
        """
        Remove the image associated to the supplied legend from the graph.
        I is still kept in the list of curves.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the image to be hidden
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True        
        """
        print("PlotBase hideImage not implemented")
        return

    def isCurveHidden(self, legend):
        """
        :param legend: The legend associated to the curve
        :type legend: string or handle
        :return: True if the associated curve is hidden
        """
        if DEBUG:
            print("PlotBase isCurveHidden not implemented")
        return False
        
    def isImageHidden(self, legend):
        """
        :param legend: The legend associated to the image
        :type legend: string or handle
        :return: True if the associated image is hidden
        """
        if DEBUG:
            print("PlotBase isImageHidden not implemented")
        return False

        
    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        :param legend: The legend associated to the curve
        :type legend: string
        """
        print("setActiveCurve not implemented")
        return None

    def showCurve(self, legend, replot=True):
        """
        Show the curve associated to the legend in the graph.
        :param legend: The legend associated to the curve
        :type legend: string
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True        
        """
        print("PlotBase showCurve not implemented")
        return False

    def showImage(self, legend, replot=True):
        """
        Show the image associated to the legend in the graph.
        :param legend: The legend associated to the image
        :type legend: string
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBase showImage not implemented")
        return False
