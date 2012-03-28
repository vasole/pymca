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
from PyMca import Plot1DWindowBase
from PyMca import QtBlissGraph
qt = QtBlissGraph.qt

class Plot1DQwt(Plot1DWindowBase.Plot1DWindowBase):
    def __init__(self, parent=None,**kw):
        Plot1DWindowBase.Plot1DWindowBase.__init__(self, parent, **kw)
        mainLayout = self.layout()
        if not ('usecrosscursor' in kw):
            kw['usecrosscursor'] = True
        self.graph = QtBlissGraph.QtBlissGraph(self, **kw)
        self.graph.canvas().setMouseTracking(1)
        self.graph.setCanvasBackground(qt.Qt.white)
        mainLayout.addWidget(self.graph)
        self.newCurve = self.addCurve
        self.setTitle = self.graph.setTitle
        self.activeCurve = None
        self._logY = False
        self.__toggleCounter = 0

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True, **kw):
        """
        Add the 1D curve given by x an y to the graph.
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'X')
        ylabel = info.get('ylabel', 'Y')
        if 'xlabel' in kw:
            info['xlabel'] = kw['xlabel'] 
        if 'ylabel' in kw:
            info['ylabel'] = kw['ylabel'] 
        Plot1DWindowBase.Plot1DWindowBase.addCurve(self, x, y, legend=legend,
                               info=info, replace=replace, replot=replot)        
        if replot:
            if replace:
                self.replot('REPLACE')
            else:
                self.replot('ADD')

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        Plot1DWindowBase.Plot1DWindowBase.removeCurve(self, legend, replot=replot)
        if legend in self.graph.curves.keys():
            self.graph.removeCurve(legend)
        self._updateActiveCurve()
        if replot:
            self.graph.replot()
        return
        
    def replot(self, mode=None):
        if mode in [None, 'REPLACE']:
            if mode == 'REPLACE':
                self.graph.clearCurves()
            for curve in self.curveList:
                x, y, legend, info = self.curveDict[curve] 
                self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self._updateActiveCurve()
            self.graph.replot()
            return
        
        if mode.upper() == 'ADD':
            for curve in self.curveList:
                x, y, legend, info = self.curveDict[curve] 
                self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self._updateActiveCurve()
            self.graph.replot()
            return

    def _updateActiveCurve(self):
        self.activeCurve = self.getActiveCurve(just_legend=True)
        if self.activeCurve not in self.curveList:
            self.activeCurve = None
        if self.activeCurve is None:
            if len(self.curveList):
                self.activeCurve = self.curveList[0]
        if self.activeCurve is not None:
            self.graph.setActiveCurve(self.activeCurve)            


    def _zoomReset(self):
        self.graph.zoomReset()

    def _yAutoScaleToggle(self):
        if self.graph.yAutoScale:
            self.graph.yAutoScale = False
            self.yAutoScaleButton.setDown(False)
            self.yAutoScaleButton.setChecked(False)
            self.graph.setY1AxisLimits(*self.graph.getY1AxisLimits())
            y2limits = self.graph.getY2AxisLimits()
            if y2limits is not None:self.graph.setY2AxisLimits(*y2limits)
        else:
            self.graph.yAutoScale = True
            self.yAutoScaleButton.setDown(True)
            self.graph.zoomReset()

    def _xAutoScaleToggle(self):
        if self.graph.xAutoScale:
            self.graph.xAutoScale = False
            self.xAutoScaleButton.setDown(False)
            self.xAutoScaleButton.setChecked(False)
            self.graph.setX1AxisLimits(*self.graph.getX1AxisLimits())
        else:
            self.graph.xAutoScale = True
            self.xAutoScaleButton.setDown(True)
            self._zoomReset()

    def _toggleLogY(self):
        if self._logY:
            self._logY = False
        else:
            self._logY = True
        activeCurve = self.graph.getActiveCurve(justlegend=1)
        self.activeCurve = activeCurve
        self.graph.clearCurves()    
        self.graph.toggleLogY()
        self.replot(mode="ADD")
        self.graph.setActiveCurve(activeCurve)

    def _togglePointsSignal(self):
        self.__toggleCounter = (self.__toggleCounter + 1) % 3
        if self.__toggleCounter == 1:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(True)
        elif self.__toggleCounter == 2:
            self.graph.setDefaultPlotPoints(True)
            self.graph.setDefaultPlotLines(False)
        else:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(False)
        self.graph.setActiveCurve(self.graph.getActiveCurve(justlegend=1))
        self.graph.replot()

    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        xmin, xmax = self.graph.getX1AxisLimits()
        return xmin, xmax
        
    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        ymin, ymax = self.graph.getY1AxisLimits()
        return ymin, ymax

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified
        legend as the active curve.
        It returns the active curve legend.
        """
        key = str(legend)
        if key in self.curveDict.keys():
            self.activeCurve = key
            self.graph.setActiveCurve(key)
        return self.activeCurve


if __name__ == "__main__":
    import numpy
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = Plot1DQwt(uselegendmenu=True)
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    print("Active curve = ", plot.getActiveCurve())
    print("X Limits = ",     plot.getGraphXLimits())
    print("Y Limits = ",     plot.getGraphYLimits())
    #print "All curves = ",   plot.getAllCurves()
    #plot.removeCurve("dummy")
    #print "All curves = ",   plot.getAllCurves()
    app.exec_()

    
