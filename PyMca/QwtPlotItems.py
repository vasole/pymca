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
import sys
from PyMca import PyMcaQt as qt
if 'PyQt4.QtGui' in sys.modules:
    raise ImportError("QwtPlotItems only works under PyQt4")
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

class QImageItem(PolygonItem):
    def __init__(self, title="Unnamed ImageItem"):
        self._title = title
        PolygonItem.__init__(self, qwt.QwtText(self._title))
        self._qImage = None
        self._imageDict = {}
        self.imageList = []
        self._worldX = None
        self._worldY = None
        self._worldWidth  = None
        self._worldHeight = None
        self._alpha = 1.0
        

    def setQImageList(self, images, width, height, imagenames = None):
        nimages = len(images)
        if imagenames is None:
            self.imageNames = []
            for i in range(nimages):
                self.imageNames.append("%s %02d" % (self._title, i))
        else:
            self.imageNames = imagenames

        i = 0
        self._imageDict = {}
        for label in self.imageNames:
            self.setQImage(images[i], width, height)
            self._imageDict[label] = self.getQImage()            
            i += 1

    def setQImage(self, qimage, width, height):
        if (width != qimage.width()) or\
           (height != qimage.height()):
            self._qImage = qimage.scaled(qt.QSize(width, height),
                                         qt.Qt.IgnoreAspectRatio,
                                         qt.Qt.FastTransformation)
        else:
            self._qImage = qimage
        if self._qImage.format() != qt.QImage.Format_ARGB32:
            self._qImage = self._qImage.convertToFormat(qt.QImage.Format_ARGB32)
        self._worldX = 0.0
        self._worldY = 0.0
        self._worldWidth  = width
        self._worldHeight = height

    def getQImage(self):
        return self._qImage

    def setCurrentIndex(self, index):
        self._qImage = self._imageDict[self.imageNames[index]]

    def draw(self, painter, xMap, yMap, canvasQRect):
        if DEBUG:
            print("%s.draw called" % self._title)

        #xMap and yMap contain the world coordinates 
        #one should deal with logarithmic axes?
	#print "xlimits = ", xMap.s1(),xMap.s2()
        #print "ylimits = ", yMap.s1(),yMap.s2()

        #the canvasQRect contains the pixel coordinates to be drawn
        #canvasQRect.x(), canvasQRect.y(), canvasQRect.width(), canvasQRect.height()
        if self._qImage is None:
            return

        xMin = self._worldX
        xMax = self._worldX + self._worldWidth
        yMin = self._worldY
        yMax = self._worldY + self._worldHeight

        #get the plot instance
        plot = self.plot()
        
        #get the destination area in pixel coordinates
        x = plot.transform(qwt.QwtPlot.xBottom, xMin)
        xmax = plot.transform(qwt.QwtPlot.xBottom, xMax)
        y = plot.transform(qwt.QwtPlot.yLeft, yMin)
        ymax = plot.transform(qwt.QwtPlot.yLeft, yMax)
        width = xmax - x

        #take care of y origin
        height = y - ymax
        if height < 0:
            destination = qt.QRectF(x, y+height, width, -height)
        else:
            destination = qt.QRectF(x, y, width, height)

        #with a brush I only get part of the image
        #brush = qt.QBrush(painter.brush())
        #brush.setTextureImage(self._qImage)
        #transform = qt.QTransform(brush.transform())
        #transform.map(x, y, 0, 0)
        #transform.map(x+width, y+height,
        #              self._qImage.width(), self._qImage.height())
        #brush.setTransform(transform)
        #the methods below give the same result 
        #painter.drawRect(destination)
        #painter.setBrush(brush)
        #painter.fillRect(destination, brush)
        #this works
        painter.setOpacity(self._alpha)
        if DEBUG:
            #draw the rectangle around the image
            #just for debugging purposes
            painter.drawRect(destination)
        if height < 0:
            painter.drawImage(destination,
                             self._qImage.mirrored(0,1))
        else:
            painter.drawImage(destination,
                             self._qImage)
        
    def setData(self, x, y, width=None, height=None):
        """
        setData(self, x, y, width=None, height=None)

        Set the set of points defining the square containing the image (in world coordinates)

        """
        if self._qImage is None:
            return
        if width is None:
            width = self._qImage.width()
        if height is None:
            height = self._qImage.height()
        self._worldX = x
        self._worldY = y
        self._worldWidth  = width
        self._worldHeight = height

    def setAlpha(self, alpha):
        self._alpha = alpha

if __name__ == "__main__":
    DEBUG = 1
    import os
    from PyMca import QtBlissGraph
    app = qt.QApplication([])
    plot = QtBlissGraph.QtBlissGraph()
    rescaler = qwt.QwtPlotRescaler(plot.canvas())
    rescaler.setEnabled(True)
   
    item = PolygonItem("Dummy")
    item.setData(x=[10, 400, 600.], y=[200, 600, 800.])
    item.attach(plot)
    image = QImageItem("Dummy2")
    qImage = qt.QImage(os.path.join(os.path.dirname(__file__),"PyMcaSplashImage.png"))
    image.setQImageList([qImage], qImage.width(), qImage.height())
    image.setData(200, 600)
    image.setAlpha(0.5)
    image.attach(plot)
    if 0:
        plot.setY1AxisLimits(1000,0)
    else:
        plot.setY1AxisLimits(0, 1000)        
    plot.show()
    app.exec_()
