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
__author__ = "V.A. Sole - ESRF Data Analysis"
"""
This module can be used for plugin testing purposes as well as for doing
the bookkeeping of actual plot windows.

It implements the Plot1DBase interface:

    addCurve(self, x, y, legend=None, info=None, replace=False, replot=True)
    getActiveCurve(self, just_legend=False)
    getAllCurves(self, just_legend=False)
    getGraphXLimits(self)
    getGraphYLimits(self)
    removeCurve(self, legend, replot=True)
    setActiveCurve(self)
"""
from PyMca import Plot1DBase


class Plot1D(Plot1DBase.Plot1DBase):
    def __init__(self):
        Plot1DBase.Plot1DBase.__init__(self)
        self.curveList = []
        self.curveDict = {}
        self.activeCurve = None

    def addCurve(self, x, y, legend=None, info=None, replace=False,
                 replot=True):
        """
        Add the 1D curve given by x an y to the graph.
        :param x: The data corresponding to the x axis
        :type x: list or numpy.ndarray
        :param y: The data corresponding to the y axis
        :type y: list or numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the curve
        :type info: dict or None
        :param replace: Flag to indicate if already existing curves are to be deleted
        :type replace: boolean default False
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'X')
        ylabel = info.get('ylabel', 'Y')
        info['xlabel'] = str(xlabel)
        info['ylabel'] = str(ylabel)

        if replace:
            self.curveList = []
            self.curveDict = {}

        if key in self.curveList:
            idx = self.curveList.index(key)
            self.curveList[idx] = key
        else:
            self.curveList.append(key)
        self.curveDict[key] = [x, y, key, info]
        if len(self.curveList) == 1:
            self.activeCurve = key
        return

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the curve to be deleted
        :type legend: string or None
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True        
        """
        if legend is None:
            return
        if legend in self.curveList:
            idx = self.curveList.index(legend)
            del self.curveList[idx]
        if legend in self.curveDict.keys():
            del self.curveDict[legend]
        return

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
        if self.activeCurve not in self.curveDict.keys():
            self.activeCurve = None
        if just_legend:
            return self.activeCurve
        if self.activeCurve is None:
            return None
        else:
            return self.curveDict[self.activeCurve]

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
        output = []
        keys = self.curveDict.keys()
        for key in self.curveList:
            if key in keys:
                if just_legend:
                    output.append(key)
                else:
                    output.append(self.curveDict[key])
        return output

    def _getAllLimits(self):
        keys = self.curveDict.keys()
        if not len(keys):
            return 0.0, 0.0, 100., 100.
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        for key in keys:
            x = self.curveDict[key][0]
            y = self.curveDict[key][1]
            if xmin is None:
                xmin = x.min()
            else:
                xmin = min(xmin, x.min())
            if ymin is None:
                ymin = y.min()
            else:
                ymin = min(ymin, y.min())
            if xmax is None:
                xmax = x.max()
            else:
                xmax = max(xmax, x.max())
            if ymax is None:
                ymax = y.max()
            else:
                ymax = max(ymax, y.max())
        return xmin, ymin, xmax, ymax

    def getGraphXLimits(self):
        """
        Set the graph X limits.
        :param xmin:  minimum value of the axis
        :type xmin: float
        :param xmax:  minimum value of the axis
        :type xmax: float
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default False
        """
        xmin, ymin, xmax, ymax = self._getAllLimits()
        return xmin, xmax

    def getGraphYLimits(self):
        """
        Set the graph Y (left) limits.
        :param ymin:  minimum value of the axis
        :type ymin: float
        :param ymax:  minimum value of the axis
        :type ymax: float
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default False
        """
        xmin, ymin, xmax, ymax = self._getAllLimits()
        return ymin, ymax

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified legend
        as the active curve.
        :param legend: The legend associated to the curve
        :type legend: string
        """
        key = str(legend)
        if key in self.curveDict.keys():
            self.activeCurve = key
        return self.activeCurve


def main():
    import numpy
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x + 100, -x * x)
    print("Active curve = ", plot.getActiveCurve())
    print("X Limits = ", plot.getGraphXLimits())
    print("Y Limits = ", plot.getGraphYLimits())
    print("All curves = ", plot.getAllCurves())
    plot.removeCurve("dummy")
    print("All curves = ", plot.getAllCurves())

if __name__ == "__main__":
    main()
