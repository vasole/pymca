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
__author__ = "Mauro Rovezzi - ID26, V.A. Sole - ESRF Data Analysis"
import sys
import os
import numpy
from matplotlib.mlab import griddata

from PyMca5 import Plugin1DBase
from PyMca5.PyMcaGui import MaskImageWidget
from PyMca5.PyMcaGui import PyMcaQt as qt

DEBUG = 0

class MultipleScanToMeshPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.methodDict = {}
        self.methodDict['Show RIXS Image'] = [self._rixsID26,
                                              "Show curves as RIXS image",
                                              None]
                           
        self._rixsWidget = None
        
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

    def _rixsID26(self):
        allCurves = self.getAllCurves()

        nCurves = len(allCurves)
        if  nCurves < 2:
            msg = "ID26 RIXS scans are built combining several single scans"
            raise ValueError(msg)

        self._xLabel = self.getGraphXLabel()
        self._yLabel = self.getGraphYLabel()

        if self._xLabel not in ["Spec.Energy", "arr_hdh_ene", "Mono.Energy"]:
            msg = "X axis does not correspond to an ID26 RIXS scan"
            raise ValueError(msg)

        if self._xLabel == "Spec.Energy":
            fixedMotorMne = "Mono.Energy"
        else:
            fixedMotorMne = "Spec.Energy"
        fixedMotorIndex = allCurves[0][3]["MotorNames"].index(fixedMotorMne)


        #get the min and max values of the curves
        if fixedMotorMne == "Mono.Energy":
            info = allCurves[0][3]
            xMin = info["MotorValues"][fixedMotorIndex]
            xMax = xMin
            nData = 0
            i = 0
            minValues = numpy.zeros((nCurves,), numpy.float64)
            for curve in allCurves:
                info = curve[3]
                tmpMin = info['MotorValues'][fixedMotorIndex]
                tmpMax = info['MotorValues'][fixedMotorIndex]
                minValues[i] = tmpMin
                if tmpMin < xMin:
                    xMin = tmpMin
                if tmpMax > xMax:
                    xMax =tmpMax
                nData += len(curve[0])
                i += 1
        else:
            xMin = allCurves[0][0][0] # ID26 data are already ordered
            xMax = allCurves[0][0][-1]

            minValues = numpy.zeros((nCurves,), numpy.float64)
            minValues[0] = xMin
            nData = len(allCurves[0][0])
            i = 0
            for curve in allCurves[1:]:
                i += 1
                tmpMin = curve[0][0]
                tmpMax = curve[0][-1]
                minValues[i] = tmpMin
                if tmpMin < xMin:
                    xMin = tmpMin
                if tmpMax > xMax:
                    xMax =tmpMax
                nData += len(curve[0])

        #sort the curves
        orderIndex = minValues.argsort()

        #print "ORDER INDEX = ", orderIndex
        # express data in eV
        if (xMax - xMin) < 5.0 :
            # it seems data need to be multiplied
            factor = 1000.
        else:
            factor = 1.0

        motor2Values = numpy.zeros((nCurves,), numpy.float64)
        xData = numpy.zeros((nData,), numpy.float32)
        yData = numpy.zeros((nData,), numpy.float32)
        zData = numpy.zeros((nData,), numpy.float32)
        start = 0
        for i in range(nCurves):
            idx = orderIndex[i]
            curve = allCurves[idx]
            info = curve[3]
            nPoints = max(curve[0].shape)
            end = start + nPoints
            x = curve[0]
            z = curve[1]
            x.shape = -1
            z.shape = -1
            if fixedMotorMne == "Mono.Energy":
                xData[start:end] = info["MotorValues"][fixedMotorIndex] * factor
                yData[start:end] = x * factor
            else:
                xData[start:end] = x * factor
                yData[start:end] = info["MotorValues"][fixedMotorIndex] * factor
            zData[start:end] = z
            start = end

        # construct the grid in steps of eStep eV
        eStep = 0.05
        n = (xMax - xMin) * (factor / eStep)
        grid0 = numpy.linspace(xMin * factor, xMax * factor, n)
        grid1 = numpy.linspace(yData.min(), yData.max(), n)
        
        # create the meshgrid
        xx, yy = numpy.meshgrid(grid0, grid1)

        # get the interpolated values
        zz = griddata(xData, yData, zData, xx, yy)

        if 0:
            # show them
            if self._rixsWidget is None:
                self._rixsWidget = MaskImageWidget.MaskImageWidget(\
                                            imageicons=False,
                                            selection=False,
                                            profileselection=True,
                                            scanwindow=self)                
            self._rixsWidget.setImageData(zz,
                                          xScale=(xx.min(), xx.max()),
                                          yScale=(yy.min(), yy.max()))
            self._rixsWidget.show()
        elif 1:
            etData = xData - yData
            grid3 = numpy.linspace(etData.min(), etData.max(), n) 
            # create the meshgrid
            xx, yy = numpy.meshgrid(grid0, grid3)

            # get the interpolated values
            zz = griddata(xData, etData, zData, xx, yy)

            if self._rixsWidget is None:
                self._rixsWidget = MaskImageWidget.MaskImageWidget(\
                                            imageicons=False,
                                            selection=False,
                                            profileselection=True,
                                            scanwindow=self)
                self._rixsWidget.setLineProjectionMode('X')
            #actualMax = zData.max()
            #actualMin = zData.min()
            #zz = numpy.where(numpy.isfinite(zz), zz, actualMax)
            self._rixsWidget.setImageData(zz,
                                          xScale=(xx.min(), xx.max()),
                                          yScale=(yy.min(), yy.max()))
            self._rixsWidget.setXLabel("Incident Energy (eV)")
            self._rixsWidget.setYLabel("Energy Transfer (eV)")
            self._rixsWidget.show()
        return

MENU_TEXT = "MultipleScanToMeshPlugin"
def getPlugin1DInstance(plotWindow, **kw):
    ob = MultipleScanToMeshPlugin(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca5.PyMcaGraph import Plot
    app = qt.QApplication([])
    #w = ConfigurationWidget()
    #w.exec_()
    #sys.exit(0)
    
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
