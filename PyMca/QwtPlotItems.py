#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
from PyMca import PyMcaQt as qt
from PyQt4 import Qwt5 as qwt

DEBUG = 0

class PolygonItem(qwt.QwtPlotItem):
    def __init__(self, title="Unnamed PolygonItem"):
        self._title = title
        qwt.QwtPlotItem.__init__(self, qwt.QwtText(self._title))
        self._x = []
        self._y = []

    def setData(self, x, y):
        """
        setData(self, x, y)

        Set the set of points defining the polygon (in world coordinates)

        """
        self._x = x
        self._y = y

    def rtti(self):
        return qwt.QwtPlotItem.Rtti_PlotUserItem

    def draw(self, painter, xMap, yMap, canvasQRect):
        if DEBUG:
            print("%s.draw called" % self._title)
        nPoints = len(self._x)
        if not nPoints:
            return

        plot = self.plot()

        #just a line
        if len(self._x) < 3:
            xPixel = [0] * nPoints
            yPixel = [0] * nPoints
            for i in range(nPoints):
                xPixel[i] = plot.transform(qwt.QwtPlot.xBottom, self._x[i])
                yPixel[i] = plot.transform(qwt.QwtPlot.yLeft, self._y[i])
            line = qt.QLineF(xPixel[0], yPixel[0],\
                             xPixel[1], yPixel[1])         
            painter.drawLine(line)
            return

        #a generic polygon
        qPoints = [0] * nPoints
        for i in range(nPoints):
            qPoints[i] = qt.QPointF(\
                        plot.transform(qwt.QwtPlot.xBottom, self._x[i]),
                        plot.transform(qwt.QwtPlot.yLeft, self._y[i]))
        polygon = qt.QPolygonF(qPoints)
        oldBrush = painter.brush()
        brush = qt.QBrush(oldBrush)
        brush.setStyle(qt.Qt.CrossPattern)
        painter.setBrush(brush)
        painter.drawPolygon(polygon, qt.Qt.OddEvenFill)
        painter.setBrush(oldBrush)

if __name__ == "__main__":
    DEBUG = 1
    from PyMca import QtBlissGraph
    app = qt.QApplication([])
    plot = QtBlissGraph.QtBlissGraph()
    item = PolygonItem("Dummy")
    item.setData(x=[10, 400, 600.], y=[200, 600, 800.])
    item.attach(plot)
    plot.replot()
    plot.show()
    app.exec_()
