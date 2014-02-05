#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
import sys
import os
from . import Plot

#TODO check for PySide
from PyQt4 import QtCore, QtGui
if not hasattr(QtCore, "Signal"):
    QtCore.Signal = QtCore.pyqtSignal

DEBUG = 0
if DEBUG:
    Plot.DEBUG = DEBUG

class PlotWidget(QtGui.QMainWindow, Plot.Plot):
    sigPlotSignal = QtCore.Signal(object)

    def __init__(self, parent=None, backend=None,
                         legends=False, callback=None, **kw):
        QtGui.QMainWindow.__init__(self, parent)
        Plot.Plot.__init__(self, parent, backend=backend)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(QtCore.Qt.Widget)
        self.containerWidget = QtGui.QWidget()
        self.containerWidget.mainLayout = QtGui.QVBoxLayout(self.containerWidget)
        self.containerWidget.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.containerWidget.mainLayout.setSpacing(0)
        widget = self.getWidgetHandle()
        if widget is not None:
            self.containerWidget.mainLayout.addWidget(widget)
            self.setCentralWidget(self.containerWidget)
        else:
            print("WARNING: No backend. Using default.")
        if legends:
            print("Legends widget to be implemented")
        self.setGraphTitle("  ")
        self.setGraphXLabel("X")
        self.setGraphYLabel("Y")
        self.setCallback(callback)

    def showLegends(self, flag=True):
        print("Legends widget to be implemented")

    def graphCallback(self, ddict=None):
        if ddict is not None:
            Plot.Plot.graphCallback(self, ddict)
            self.sigPlotSignal.emit(ddict)

    def resizeEvent(self, event):
        super(PlotWidget, self).resizeEvent(event)
        #Should I reset the zoom or replot?
        #self.resetZoom()

    def replot(self):
        Plot.Plot.replot(self)
        # force update of the widget!!!
        # should this be made at the backend level?
        w = self.centralWidget()
        QtGui.qApp.postEvent(w, QtGui.QResizeEvent(w.size(),
                                                   w.size()))
        
if __name__ == "__main__":
    import time
    if "matplotlib" in sys.argv:
        from PyMca.plotting.backends.MatplotlibBackend import MatplotlibBackend as backend
        print("USING matplotlib")
        time.sleep(1)
    else:
        try:
            from PyMca.plotting.backends.PyQtGraphBackend import PyQtGraphBackend as backend
            print("USING PyQtGraph")
        except:
            from PyMca.plotting.backends.MatplotlibBackend import MatplotlibBackend as backend
            print("USING matplotlib")
        time.sleep(1)
    import numpy
    x = numpy.arange(100.)
    y = x * x
    app = QtGui.QApplication([])
    plot = PlotWidget(None, backend=backend, legends=True)
    plot.show()
    if 1:
        plot.addCurve(x, y, "dummy")
        plot.addCurve(x+100, x*x)
        plot.addCurve(x, -y, "dummy 2")
        print("Active curve = ", plot.getActiveCurve())
        print("X Limits = ",     plot.getGraphXLimits())
        print("Y Limits = ",     plot.getGraphYLimits())
        print("All curves = ",   plot.getAllCurves())
        #print("REMOVING dummy")
        #plot.removeCurve("dummy")
        plot.insertXMarker(50., draggable=True)
        #plot.insertYMarker(50., draggable=True)
    else:
        # insert a few curves
        cSin={}
        cCos={}
        nplots=50
        for i in range(nplots):
            # calculate 3 NumPy arrays
            x = numpy.arange(0.0, 10.0, 0.1)
            y = 10*numpy.sin(x+(i/10.0) * 3.14)
            z = numpy.cos(x+(i/10.0) * 3.14)
            #build a key
            a="%d" % i
            #plot the data
            cSin[a] = plot.addCurve(x, y, 'y = sin(x)' + a, replot=False)
            cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=False)
        cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=True)
        plot.insertXMarker(5., draggable=True)
        plot.insertYMarker(5., draggable=True)
    print("All curves = ", plot.getAllCurves(just_legend=True))
    plot.setYAxisLogarithmic(True)
    app.exec_()
