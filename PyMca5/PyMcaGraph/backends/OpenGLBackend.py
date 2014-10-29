# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
OpenGL backend
"""


# import ######################################################################

try:
    from PyMca5.PyMcaGui import PyMcaQt as qt
    QGLWidget = qt.QGLWidget
except ImportError:
    try:
        from PyQt4.QtOpenGL import QGLWidget
    except ImportError:
        from PyQt5.QtOpenGL import QGLWidget

from collections import OrderedDict
import numpy as np
import math

import OpenGL
if 1:  # Debug
    OpenGL.ERROR_ON_COPY = True
else:
    OpenGL.ERROR_CHECKING = False

from OpenGL.GL import *  # noqa
from OpenGL.GL.ARB.texture_rg import GL_R32F  # Core in OpenGL 3

from ..PlotBackend import PlotBackend
from ..Plot import colordict
from .GLSupport import *  # noqa


# shaders #####################################################################

_baseVertShd = """
   attribute vec2 position;
   uniform mat4 matrix;

   void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
   }
   """

_baseFragShd = """
    uniform vec4 color;

    void main(void) {
        gl_FragColor = color;
    }
    """

_texVertShd = """
   attribute vec2 position;
   attribute vec2 texCoords;
   uniform mat4 matrix;

   varying vec2 coords;

   void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        coords = texCoords;
   }
   """

_texFragShd = """
    uniform sampler2D tex;

    varying vec2 coords;

    void main(void) {
        gl_FragColor = texture2D(tex, coords);
    }
    """


_vertexSrc = """
    attribute vec2 position;
    attribute vec2 texCoords;
    uniform mat4 matrix;

    varying vec2 coords;

    void main(void) {
        coords = texCoords;
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """

_fragmentSrc = """
    #define CMAP_GRAY   0
    #define CMAP_R_GRAY 1
    #define CMAP_RED    2
    #define CMAP_GREEN  3
    #define CMAP_BLUE   4
    #define CMAP_TEMP   5

    uniform sampler2D data;
    uniform struct {
        int id;
        float min;
        float oneOverRange;
        float logMin;
        float oneOverLogRange;
    } cmap;

    varying vec2 coords;

    vec4 cmapGray(float normValue) {
        return vec4(normValue, normValue, normValue, 1.);
    }

    vec4 cmapReversedGray(float normValue) {
        float invValue = 1. - normValue;
        return vec4(invValue, invValue, invValue, 1.);
    }

    vec4 cmapRed(float normValue) {
        return vec4(normValue, 0., 0., 1.);
    }

    vec4 cmapGreen(float normValue) {
        return vec4(0., normValue, 0., 1.);
    }

    vec4 cmapBlue(float normValue) {
        return vec4(0., 0., normValue, 1.);
    }

    //red: 0.5->0.75: 0->1
    //green: 0.->0.25: 0->1; 0.75->1.: 1->0
    //blue: 0.25->0.5: 1->0
    vec4 cmapTemperature(float normValue) {
        float red = clamp(4. * normValue - 2., 0., 1.);
        float green = 1. - clamp(4. * abs(normValue - 0.5) - 1., 0., 1.);
        float blue = 1. - clamp(4. * normValue - 1., 0., 1.);
        return vec4(red, green, blue, 1.);
    }

    void main(void) {
        float value = texture2D(data, coords).r;
        if (cmap.oneOverRange != 0.) {
            value = clamp(cmap.oneOverRange * (value - cmap.min), 0., 1.);
        } else {
            value = clamp(cmap.oneOverLogRange * (log(value) - cmap.logMin),
                          0., 1.);
        }

        if (cmap.id == CMAP_GRAY) {
            gl_FragColor = cmapGray(value);
        } else if (cmap.id == CMAP_R_GRAY) {
            gl_FragColor = cmapReversedGray(value);
        } else if (cmap.id == CMAP_RED) {
            gl_FragColor = cmapRed(value);
        } else if (cmap.id == CMAP_GREEN) {
            gl_FragColor = cmapGreen(value);
        } else if (cmap.id == CMAP_BLUE) {
            gl_FragColor = cmapBlue(value);
        } else if (cmap.id == CMAP_TEMP) {
            gl_FragColor = cmapTemperature(value);
        }
    }
    """

_shaderColormapIds = {
    'gray': 0,
    'reversed gray': 1,
    'red': 2,
    'green': 3,
    'blue': 4,
    'temperature': 5
}


# utils #######################################################################

def _ticks(start, stop, step):
    """range for float (including stop)
    """
    while start <= stop:
            yield start
            start += step


def linesVertices(width, height, step):
    nbLines = int(math.ceil((width + height) / float(step)))
    vertices = np.empty((nbLines * 2, 2), dtype=np.float32)
    for line in range(nbLines):
        vertices[2 * line] = 0., step * line
        vertices[2 * line + 1] = step * line, 0.
    return vertices


# Interaction #################################################################

class Zoom(ClicOrDrag):
    def __init__(self, backend):
        self.backend = backend
        self.zoomStack = []
        super(Zoom, self).__init__()

    def _ensureAspectRatio(self, x0, y0, x1, y1):
        plotW, plotH = self.backend.plotSizeInPixels()
        try:
            plotRatio = plotW / float(plotH)
        except ZeroDivisionError:
            pass
        else:
            width, height = math.fabs(x1 - x0), math.fabs(y1 - y0)

            try:
                selectRatio = width / height
            except ZeroDivisionError:
                width, height = 1., 1.
            else:
                if selectRatio < plotRatio:
                    height = width / plotRatio
                else:
                    width = height * plotRatio
            x1 = x0 + np.sign(x1 - x0) * width
            y1 = y0 + np.sign(y1 - y0) * height
        return x1, y1

    def clic(self, x, y, btn):
        if btn == LEFT_BTN:
            xMin, xMax = self.backend.getGraphXLimits()
            yMin, yMax = self.backend.getGraphYLimits()
            self.zoomStack.append((xMin, xMax, yMin, yMax))
            self._zoom(x, y, 2)
        else:
            try:
                xMin, xMax, yMin, yMax = self.zoomStack.pop()
            except IndexError:
                self.backend.resetZoom()
            else:
                self.backend.setLimits(xMin, xMax, yMin, yMax)

    def beginDrag(self, x, y):
        self.x0, self.y0 = self.backend.pixelToDataCoords(x, y)

    def drag(self, x1, y1):
        x1, y1 = self.backend.pixelToDataCoords(x1, y1)
        if self.backend.isKeepDataAspectRatio():
            x1, y1 = self._ensureAspectRatio(self.x0, self.y0, x1, y1)

        self.backend.setSelectionArea((self.x0, self.y0),
                                      (self.x0, y1),
                                      (x1, y1),
                                      (x1, self.y0))

    def endDrag(self, startPos, endPos):
        xMin, xMax = self.backend.getGraphXLimits()
        yMin, yMax = self.backend.getGraphYLimits()
        self.zoomStack.append((xMin, xMax, yMin, yMax))

        self.backend.setSelectionArea()
        x0, y0 = self.backend.pixelToDataCoords(*startPos)
        x1, y1 = self.backend.pixelToDataCoords(*endPos)
        if self.backend.isKeepDataAspectRatio():
            x1, y1 = self._ensureAspectRatio(x0, y0, x1, y1)
        xMin, xMax = min(x0, x1), max(x0, x1)
        yMin, yMax = min(y0, y1), max(y0, y1)
        self.backend.setLimits(xMin, xMax, yMin, yMax)

    def onWheel(self, x, y, angle):
        scaleF = 1.1 if angle > 0 else 1./1.1
        self._zoom(x, y, scaleF)

    def _zoom(self, cx, cy, scaleF):
        xCenter, yCenter = self.backend.pixelToDataCoords(cx, cy)

        xMin, xMax = self.backend.getGraphXLimits()
        xOffset = (xCenter - xMin)/(xMax - xMin)
        xRange = (xMax - xMin) / scaleF

        yMin, yMax = self.backend.getGraphYLimits()
        yOffset = (yCenter - yMin)/(yMax - yMin)
        yRange = (yMax - yMin) / scaleF

        self.backend.setLimits(xCenter - xOffset * xRange,
                               xCenter + (1. - xOffset) * xRange,
                               yCenter - yOffset * yRange,
                               yCenter + (1. - yOffset) * yRange)


class Select(object):
    parameters = {}


class SelectPolygon(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto(SelectPolygon.Select, x, y)

    class Select(State):
        def enter(self, x, y):
            x, y = self.machine.backend.pixelToDataCoords(x, y)
            self.points = [(x, y), (x, y)]

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                x, y = self.machine.backend.pixelToDataCoords(x, y)
                self.points[-1] = (x, y)
                self.machine.backend.setSelectionArea(*self.points)
                if self.points[-2] != self.points[-1]:
                    self.points.append((x, y))

        def onMove(self, x, y):
            x, y = self.machine.backend.pixelToDataCoords(x, y)
            self.points[-1] = (x, y)
            self.machine.backend.setSelectionArea(*self.points)

        def onPress(self, x, y, btn):
            if btn == RIGHT_BTN:
                x, y = self.machine.backend.pixelToDataCoords(x, y)
                self.points[-1] = (x, y)
                if self.points[-2] == self.points[-1]:
                    self.points.pop()
                self.machine.backend.setSelectionArea()

                # Signal drawingFinished
                points = np.array(self.points, dtype=np.float32)
                transPoints = np.transpose(points)
                eventDict = {
                    'parameters': self.machine.parameters,
                    'points': points,
                    'xdata': transPoints[0],
                    'ydata': transPoints[1],
                    'type': 'polygon',
                    'event': 'drawingFinished'
                }
                self.machine.backend._callback(eventDict)
                self.goto(SelectPolygon.Idle)

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        super(SelectPolygon, self).__init__(SelectPolygon.Idle)


class Select2Points(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto(Select2Points.Start, x, y)

    class Start(State):
        def enter(self, x, y):
            self.machine.beginSelect(x, y)

        def onMove(self, x, y):
            self.goto(Select2Points.Select, x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto(Select2Points.Select, x, y)

    class Select(State):
        def enter(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto(Select2Points.Idle)

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        super(Select2Points, self).__init__(Select2Points.Idle)

    def beginSelect(self, x, y):
        pass

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass


class SelectRectangle(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToDataCoords(x, y)

    def select(self, x, y):
        x, y = self.backend.pixelToDataCoords(x, y)
        self.backend.setSelectionArea(self.startPt,
                                      (self.startPt[0], y),
                                      (x, y),
                                      (x, self.startPt[1]))

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        x, y = self.backend.pixelToDataCoords(x, y)

        # Signal drawingFinished
        points = np.array((self.startPt, (x, y)), dtype=np.float32)
        transPoints = np.transpose(points)
        eventDict = {
            'parameters': self.parameters,
            'points': points,
            'xdata': transPoints[0],
            'ydata': transPoints[1],
            'x': transPoints[0].min(),
            'y': transPoints[1].max(),
            'width': math.fabs(points[0][0] - points[1][0]),
            'height': math.fabs(points[0][1] - points[1][1]),
            'type': 'rectangle',
            'event': 'drawingFinished'
        }
        self.backend._callback(eventDict)


class SelectLine(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToDataCoords(x, y)

    def select(self, x, y):
        x, y = self.backend.pixelToDataCoords(x, y)
        self.backend.setSelectionArea(self.startPt, (x, y))

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        x, y = self.backend.pixelToDataCoords(x, y)

        # Signal drawingFinished
        points = np.array((self.startPt, (x, y)), dtype=np.float32)
        transPoints = np.transpose(points)
        eventDict = {
            'parameters': self.parameters,
            'points': points,
            'xdata': transPoints[0],
            'ydata': transPoints[1],
            'type': 'line',
            'event': 'drawingFinished'
        }
        self.backend._callback(eventDict)


class Select1Point(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto(Select1Point.Select, x, y)

    class Select(State):
        def enter(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto(Select1Point.Idle)

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        super(Select1Point, self).__init__(Select1Point.Idle)

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass


class SelectHLine(Select1Point):
    def _hLine(self, y):
        y = self.backend.pixelToDataCoords(yPixel=y)
        return (self.backend._xMin, y), (self.backend._xMax, y)

    def select(self, x, y):
        self.backend.setSelectionArea(*self._hLine(y))

    def endSelect(self, x, y):
        self.backend.setSelectionArea()

        # Signal drawingFinished
        points = np.array((self._hLine(y)), dtype=np.float32)
        transPoints = np.transpose(points)
        eventDict = {
            'parameters': self.parameters,
            'points': points,
            'xdata': transPoints[0],
            'ydata': transPoints[1],
            'type': 'hline',
            'event': 'drawingFinished'
        }
        self.backend._callback(eventDict)


class SelectVLine(Select1Point):
    def _vLine(self, x):
        x = self.backend.pixelToDataCoords(xPixel=x)
        return (x, self.backend._yMin), (x, self.backend._yMax)

    def select(self, x, y):
        self.backend.setSelectionArea(*self._vLine(x))

    def endSelect(self, x, y):
        self.backend.setSelectionArea()

        # Signal drawingFinished
        points = np.array(self._vLine(x), dtype=np.float32)
        transPoints = np.transpose(points)
        eventDict = {
            'parameters': self.parameters,
            'points': points,
            'xdata': transPoints[0],
            'ydata': transPoints[1],
            'type': 'vline',
            'event': 'drawingFinished'
        }
        self.backend._callback(eventDict)


# OpenGLBackend ###############################################################

class OpenGLBackend(PlotBackend, QGLWidget):
    def __init__(self, parent=None, **kw):
        self._xMin, self._xMax = 0., 1.
        self._yMin, self._yMax = 0., 1.
        self.keepDataAspectRatio(False)
        self._isYInverted = False
        self._title, self._xLabel, self._yLabel = '', '', ''

        self.winWidth, self.winHeight = 0, 0
        self._dataBBox = {'xMin': 0., 'xMax': 0., 'xStep': 1.,
                          'yMin': 0., 'yMax': 0., 'yStep': 1.}
        self._images = OrderedDict()
        self._items = OrderedDict()
        self._labels = []
        self._selectionArea = None

        self._margins = {'left': 50, 'right': 50, 'top': 50, 'bottom': 50}
        self._lineWidth = 1
        self._tickLen = 5

        self._axisDirtyFlag = True
        self._plotDirtyFlag = True

        self.eventHandler = None
        self._plotHasFocus = set()

        QGLWidget.__init__(self, parent)
        PlotBackend.__init__(self, parent, **kw)
        self.setMouseTracking(True)

    # Mouse events #

    def _mouseInPlotArea(self, x, y):
        xPlot = clamp(x, self._margins['left'],
                      self.winWidth - self._margins['right'])
        yPlot = clamp(y, self._margins['top'],
                      self.winHeight - self._margins['bottom'])
        return xPlot, yPlot

    def mousePressEvent(self, event):
        x, y = event.x(), event.y()
        if (self._mouseInPlotArea(x, y) == (x, y)):
            btn = event.button()
            self._plotHasFocus.add(btn)
            if self.eventHandler is not None:
                self.eventHandler.handleEvent('press', x, y, btn)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.eventHandler:
            x, y = self._mouseInPlotArea(event.x(), event.y())
            self.eventHandler.handleEvent('move', x, y)
        event.accept()

    def mouseReleaseEvent(self, event):
        btn = event.button()
        try:
            self._plotHasFocus.remove(btn)
        except KeyError:
            pass
        else:
            if self.eventHandler:
                x, y = self._mouseInPlotArea(event.x(), event.y())
                self.eventHandler.handleEvent('release', x, y, btn)
        event.accept()

    def wheelEvent(self, event):
        x, y = event.x(), event.y()
        if self.eventHandler and (self._mouseInPlotArea(x, y) == (x, y)):
            angle = event.delta() / 8.  # in degrees
            self.eventHandler.handleEvent('wheel', x, y, angle)
        event.accept()

    # Manage Plot #

    def setSelectionArea(self, *points):
        self._selectionArea = Shape2D(points) if points else None
        self.updateGL()

    def _updateDataBBox(self):
        xMin, xMax, xStep = float('inf'), -float('inf'), float('inf')
        yMin, yMax, yStep = float('inf'), -float('inf'), float('inf')
        for image in self._images.values():
            bbox = image['bBox']
            xMin = min(xMin, bbox['xMin'])
            xMax = max(xMax, bbox['xMax'])
            xStep = min(xStep, bbox['xStep'])
            yMin = min(yMin, bbox['yMin'])
            yMax = max(yMax, bbox['yMax'])
            yStep = min(yStep, bbox['yStep'])
        if xMin >= xMax:
                xMin, xMax = 0., 1.
        if yMin >= yMax:
                yMin, yMax = 0., 1.

        self._dataBBox = {'xMin': xMin, 'xMax': xMax, 'xStep': xStep,
                          'yMin': yMin, 'yMax': yMax, 'yStep': yStep}

    def updateAxis(self):
        self._axisDirtyFlag = True
        self._plotDirtyFlag = True

    def _updateAxis(self):
        if not self._axisDirtyFlag:
            return
        else:
            self._axisDirtyFlag = False

        # Check if window is large enough
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        # Plot frame
        xLeft = self._margins['left'] - .5 * self._lineWidth
        xRight = self.winWidth - self._margins['right'] + .5 * self._lineWidth
        yBottom = self.winHeight - self._margins['bottom'] + \
            .5 * self._lineWidth
        yTop = self._margins['top'] - .5 * self._lineWidth

        self._frameVertices = np.array(
            ((xLeft,  yBottom), (xLeft,  yTop),
             (xRight, yTop), (xRight, yBottom),
             (xLeft,  yBottom)), dtype=np.float32)

        # Ticks
        self._labels = []

        nTicks = 5

        xMin, xMax, xStep, xNbFrac = niceNumbers(self._xMin, self._xMax,
                                                 nTicks)
        xTicks = [x for x in _ticks(xMin, xMax, xStep)
                  if x >= self._xMin and x <= self._xMax]

        yMin, yMax, yStep, yNbFrac = niceNumbers(self._yMin, self._yMax,
                                                 nTicks)
        yTicks = [y for y in _ticks(yMin, yMax, yStep)
                  if y >= self._yMin and y <= self._yMax]

        self._tickVertices = np.empty((len(xTicks) + len(yTicks), 4, 2),
                                      dtype=np.float32)

        plotBottom = self.winHeight - self._margins['bottom']
        for index, xTick in enumerate(xTicks):
            tickText = ('{:.' + str(xNbFrac) + 'f}').format(xTick)
            xTick = self.dataToPixelCoords(xData=xTick)
            self._tickVertices[index][0] = xTick, plotBottom
            self._tickVertices[index][1] = xTick, plotBottom + self._tickLen
            self._tickVertices[index][2] = xTick, self._margins['top']
            self._tickVertices[index][3] = (xTick, self._margins['top'] -
                                            self._tickLen)

            self._labels.append((xTick, plotBottom + self._tickLen + 2,
                                 Text2D(tickText, align=CENTER, valign=TOP)))

        plotRight = self.winWidth - self._margins['right']
        for index, yTick in enumerate(yTicks, len(xTicks)):
            tickText = ('{:.' + str(yNbFrac) + 'f}').format(yTick)
            yTick = self.dataToPixelCoords(yData=yTick)
            self._tickVertices[index][0] = self._margins['left'], yTick
            self._tickVertices[index][1] = (self._margins['left'] -
                                            self._tickLen, yTick)
            self._tickVertices[index][2] = plotRight, yTick
            self._tickVertices[index][3] = plotRight + self._tickLen, yTick

            self._labels.append((self._margins['left'] - self._tickLen - 2,
                                 yTick,
                                 Text2D(tickText, align=RIGHT, valign=CENTER)))

        # Title, Labels
        if self._title:
            self._labels.append((self.winWidth // 2,
                                 self._margins['top'] - self._tickLen - 2,
                                 Text2D(self._title, align=CENTER,
                                        valign=BOTTOM)))
        if self._xLabel:
            self._labels.append((self.winWidth // 2,
                                self.winHeight - self._margins['bottom'] // 2,
                                Text2D(self._xLabel, align=CENTER,
                                       valign=TOP)))

        if self._yLabel:
            self._labels.append((self._margins['left'] // 2,
                                self.winHeight // 2,
                                Text2D(self._yLabel, align=CENTER,
                                       valign=CENTER, rotate=ROTATE_270)))

    def dataToPixelCoords(self, xData=None, yData=None):
        plotWidth, plotHeight = self.plotSizeInPixels()

        if xData is not None:
            xPixel = self._margins['left'] + \
                (xData - self._xMin) / (self._xMax - self._xMin) * plotWidth
        if yData is not None:
            yOffset = (yData - self._yMin) / (self._yMax - self._yMin)
            yOffset *= plotHeight
            if self._isYInverted:
                yPixel = self._margins['top'] + yOffset
            else:
                yPixel = self.winHeight - self._margins['bottom'] - yOffset

        if xData is None:
            try:
                return yPixel
            except NameError:
                return None
        elif yData is None:
            return xPixel
        else:
            return xPixel, yPixel

    def pixelToDataCoords(self, xPixel=None, yPixel=None):
        plotWidth, plotHeight = self.plotSizeInPixels()

        if xPixel is not None:
            if xPixel < self._margins['left'] or \
               xPixel > (self.winWidth - self._margins['right']):
                xData = None
            else:
                xData = (xPixel - self._margins['left']) + 0.5
                xData /= float(plotWidth)
                xData = self._xMin + xData * (self._xMax - self._xMin)

        if yPixel is not None:
            if yPixel < self._margins['top'] or \
               yPixel > self.winHeight - self._margins['bottom']:
                yData = None
            elif self._isYInverted:
                yData = yPixel - self._margins['top'] + 0.5
                yData /= float(plotHeight)
                yData = self._yMin + yData * (self._yMax - self._yMin)
            else:
                yData = self.winHeight - self._margins['bottom'] - yPixel - 0.5
                yData /= float(plotHeight)
                yData = self._yMin + yData * (self._yMax - self._yMin)

        if xPixel is None:
            try:
                return yData
            except NameError:
                return None
        elif yPixel is None:
            return xData
        else:
            return xData, yData

    def plotSizeInPixels(self):
        w = self.winWidth - self._margins['left'] - self._margins['right']
        h = self.winHeight - self._margins['top'] - self._margins['bottom']
        return w, h

    # QGLWidget API #

    def initializeGL(self):
        glClearColor(1., 1., 1., 1.)
        glClearStencil(0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Create basic program
        self._progBase = Program(_baseVertShd, _baseFragShd)

        # Create texture program
        self._progTex = Program(_texVertShd, _texFragShd)

        # Create image program
        self._progImg = Program(_vertexSrc, _fragmentSrc)

    def _paintGLDirect(self):
        self._renderPlot()
        self._renderSelection()

    def _paintGLFBO(self):
        if self._plotDirtyFlag or not hasattr(self, '_plotTex'):
            self._plotDirtyFlag = False
            self._plotVertices = np.array(((-1., -1., 0., 0.),
                                           (1., -1., 1., 0.),
                                           (-1., 1., 0., 1.),
                                           (1., 1., 1., 1.)),
                                          dtype=np.float32)
            if not hasattr(self, '_plotTex') or \
               self._plotTex.width != self.winWidth or \
               self._plotTex.height != self.winHeight:
                if hasattr(self, '_plotTex'):
                    self._plotTex.discard()
                self._plotTex = FBOTexture(GL_RGBA,
                                           self.winWidth, self.winHeight,
                                           minFilter=GL_NEAREST,
                                           magFilter=GL_NEAREST,
                                           wrapS=GL_CLAMP_TO_EDGE,
                                           wrapT=GL_CLAMP_TO_EDGE)
            with self._plotTex:
                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
                self._renderPlot()

        # Render plot in screen coords
        glViewport(0, 0, self.winWidth, self.winHeight)

        self._progTex.use()
        texUnit = 0

        glUniform1i(self._progTex.uniforms['tex'], texUnit)
        glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, GL_TRUE,
                           mat4Identity())

        stride = self._plotVertices.shape[-1] * self._plotVertices.itemsize
        glEnableVertexAttribArray(self._progTex.attributes['position'])
        glVertexAttribPointer(self._progTex.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, self._plotVertices)

        texCoordsPtr = c_void_p(self._plotVertices.ctypes.data +
                                2 * self._plotVertices.itemsize)  # Better way?
        glEnableVertexAttribArray(self._progTex.attributes['texCoords'])
        glVertexAttribPointer(self._progTex.attributes['texCoords'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, texCoordsPtr)

        self._plotTex.bind(texUnit)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(self._plotVertices))
        glBindTexture(GL_TEXTURE_2D, 0)

        self._renderSelection()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # Check if window is large enough
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        # self._paintGLDirect()
        self._paintGLFBO()

    def _renderSelection(self):
        # Render selection area
        if self._selectionArea is not None:
            plotWidth, plotHeight = self.plotSizeInPixels()

            # Render in plot area
            glScissor(self._margins['left'], self._margins['bottom'],
                      plotWidth, plotHeight)
            glEnable(GL_SCISSOR_TEST)

            glViewport(self._margins['left'], self._margins['right'],
                       plotWidth, plotHeight)

            # Matrix
            if self._isYInverted:
                matDataProj = mat4Ortho(self._xMin, self._xMax,
                                        self._yMax, self._yMin,
                                        1, -1)
            else:
                matDataProj = mat4Ortho(self._xMin, self._xMax,
                                        self._yMin, self._yMax,
                                        1, -1)
            self._progBase.use()

            # Render fill
            glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                               matDataProj)
            glUniform4f(self._progBase.uniforms['color'], 0., 0., 0., 0.5)

            posAttrib = self._progBase.attributes['position']
            self._selectionArea.prepareFillMask(posAttrib)

            matPlotScreenProj = mat4Ortho(0, plotWidth,
                                          plotHeight, 0,
                                          1, -1)
            glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                               matPlotScreenProj)
            vertices = linesVertices(plotWidth, plotHeight, 20)
            glVertexAttribPointer(self._progBase.attributes['position'],
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0, vertices)
            glDrawArrays(GL_LINES, 0, len(vertices))

            glDisable(GL_STENCIL_TEST)

            # Render stroke
            glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                               matDataProj)
            glUniform4f(self._progBase.uniforms['color'], 0., 0., 0., 1.)
            self._selectionArea.renderStroke(posAttrib)

            glDisable(GL_SCISSOR_TEST)

    def _renderPlot(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        # Render plot in screen coords
        glViewport(0, 0, self.winWidth, self.winHeight)
        matScreenProj = mat4Ortho(0, self.winWidth,
                                  self.winHeight, 0,
                                  1, -1)

        self._updateAxis()

        # Render Plot frame
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           matScreenProj)
        glUniform4f(self._progBase.uniforms['color'], 0., 0., 0., 1.)
        glVertexAttribPointer(self._progBase.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self._frameVertices)
        glLineWidth(self._lineWidth)
        glDrawArrays(GL_LINE_STRIP, 0, len(self._frameVertices))

        # Render Ticks
        glVertexAttribPointer(self._progBase.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self._tickVertices)

        glLineWidth(self._lineWidth)
        nbTicks, nbVertPerTick, _ = self._tickVertices.shape
        glDrawArrays(GL_LINES, 0, nbTicks * nbVertPerTick)

        # Render Text
        self._progTex.use()
        textTexUnit = 0
        glUniform1i(self._progTex.uniforms['tex'], textTexUnit)

        for x, y, label in self._labels:
            glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, GL_TRUE,
                               matScreenProj * mat4Translate(x, y, 0))
            label.render(self._progTex.attributes['position'],
                         self._progTex.attributes['texCoords'],
                         textTexUnit)

        # Render in plot area
        glScissor(self._margins['left'], self._margins['bottom'],
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

        glViewport(self._margins['left'], self._margins['right'],
                   plotWidth, plotHeight)

        # Matrix
        if self._isYInverted:
            matDataProj = mat4Ortho(self._xMin, self._xMax,
                                    self._yMax, self._yMin,
                                    1, -1)
        else:
            matDataProj = mat4Ortho(self._xMin, self._xMax,
                                    self._yMin, self._yMax,
                                    1, -1)
        # Render Images
        dataTexUnit = 0

        for image in self._images.values():
            try:
                texture = image['_texture']
            except KeyError:
                data = image['data']
                if len(data.shape) == 2:
                    height, width = data.shape
                    texture = Image(GL_R32F, width, height,
                                    format_=GL_RED, type_=GL_FLOAT,
                                    data=image['data'])
                    image['_texture'] = texture
                else:
                    height, width, depth = data.shape
                    format_=GL_RGBA if depth == 4 else GL_RGB
                    texture = Image(format_, width, height,
                                    format_=format_,
                                    type_=GL_UNSIGNED_BYTE
                                    if data.dtype == np.uint8 else GL_FLOAT,
                                    data=image['data'])
                    image['_texture'] = texture

            bbox = image['bBox']
            mat = matDataProj * mat4Translate(bbox['xMin'], bbox['yMin'])
            mat *= mat4Scale(
                float(bbox['xMax'] - bbox['xMin'])/texture.width,
                float(bbox['yMax'] - bbox['yMin'])/texture.height,
            )

            try:
                colormapName = image['colormapName']
            except KeyError:
                # Pixmap
                self._progTex.use()
                glUniform1i(self._progTex.uniforms['tex'], dataTexUnit)
                glUniformMatrix4fv(self._progTex.uniforms['matrix'],
                                   1, GL_TRUE, mat)
                texture.render(self._progTex.attributes['position'],
                               self._progTex.attributes['texCoords'],
                               dataTexUnit)
            else:
                # Colormap
                self._progImg.use()
                glUniform1i(self._progImg.uniforms['data'], dataTexUnit)
                glUniformMatrix4fv(self._progImg.uniforms['matrix'],
                                   1, GL_TRUE, mat)

                glUniform1i(self._progImg.uniforms['cmap.id'],
                            _shaderColormapIds[colormapName])
                if image['colormapIsLog']:
                    logVMin = math.log(image['vmin'])
                    glUniform1f(self._progImg.uniforms['cmap.logMin'], logVMin)
                    glUniform1f(self._progImg.uniforms['cmap.oneOverRange'],
                                0.)
                    glUniform1f(self._progImg.uniforms['cmap.oneOverLogRange'],
                                1./(math.log(image['vmax']) - logVMin))
                else:
                    glUniform1f(self._progImg.uniforms['cmap.min'],
                                image['vmin'])
                    glUniform1f(self._progImg.uniforms['cmap.oneOverRange'],
                                1./(image['vmax'] - image['vmin']))

                texture.render(self._progImg.attributes['position'],
                               self._progImg.attributes['texCoords'],
                               dataTexUnit)

        # Render Items
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           matDataProj)

        for item in self._items.values():
            try:
                shape2D = item['_shape2D']
            except KeyError:
                shape2D = Shape2D(zip(item['x'], item['y']),
                                  fill=item['fill'], stroke=not item['fill'])
                item['_shape2D'] = shape2D

            glUniform4f(self._progBase.uniforms['color'], *item['color'])
            shape2D.render(self._progBase.attributes['position'])

        glDisable(GL_SCISSOR_TEST)

    def resizeGL(self, width, height):
        self.winWidth, self.winHeight = width, height
        self.setLimits(self._xMin, self._xMax, self._yMin, self._yMax)

        self.updateAxis()
        self.updateGL()

    # PlotBackend API #

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=(0, 1), yScale=(0, 1), z=0,
                 selectable=False, draggable=False,
                 colormap=None, **kwargs):
        if info:
            print('addImage info ignored:', info)
        if z != 0 or selectable or draggable:
            raise NotImplementedError("z, selectable and draggable \
                                      not implemented")

        if replace:
            self.clearImages()

        oldImage = self._images.get(legend, None)

        height, width = data.shape[0:2]
        bbox = {'xMin': xScale[0],
                'xMax': xScale[0] + xScale[1] * width,
                'xStep': xScale[1],
                'yMin': yScale[0],
                'yMax': yScale[0] + yScale[1] * height,
                'yStep': yScale[1]}

        if len(data.shape) == 2:
            if colormap is None:
                colormap = self.getDefaultColormap()
            if colormap['normalization'] not in ('linear', 'log'):
                raise NotImplementedError(
                    "Normalisation: {0}".format(colormap['normalization']))
            if colormap['colors'] != 256:
                raise NotImplementedError(
                    "Colors: {0}".format(colormap['colors']))

            self._images[legend] = {
                'data': data,
                'colormapName': colormap['name'][:],
                'colormapIsLog': colormap['normalization'].startswith('log'),
                'vmin': data.min() if colormap['autoscale']
                else colormap['vmin'],
                'vmax': data.max() if colormap['autoscale']
                else colormap['vmax'],
                'bBox': bbox
            }

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert(data.shape[2] in (3, 4))
            assert(data.dtype == np.uint8 or
                   np.can_cast(data.dtype, np.float32))
            self._images[legend] = {'data': data, 'bBox': bbox}
        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        if oldImage is None or bbox != oldImage['bBox']:
            self._updateDataBBox()
            self.setLimits(self._dataBBox['xMin'], self._dataBBox['xMax'],
                           self._dataBBox['yMin'], self._dataBBox['yMax'])

        self._plotDirtyFlag = True

        if replot:
            self.replot()

    def removeImage(self, legend, replot=True):
        try:
            del self._images[legend]
        except KeyError:
            pass
        if replot:
            self.replot()

    def clearImages(self):
        self._images = OrderedDict()

    def addItem(self, xList, yList, legend=None, info=None,
                replace=False, replot=True,
                shape="polygon", fill=True, **kwargs):
        if info:
            raise NotImplementedError("info not implemented")

        if shape not in self._drawModes:
            raise NotImplementedError("Unsupported shape {0}".format(shape))

        if replace:
            self.clearItems()

        colorCode = kwargs.get('color', 'black')
        if colorCode[0] != '#':
            colorCode = colordict[colorCode]

        if shape == 'rectangle':
            xMin, xMax = xList
            xList = xMin, xMin, xMax, xMax
            yMin, yMax = yList
            yList = yMin, yMax, yMax, yMin

        self._items[legend] = {
            'shape': shape,
            'color': rgba(colorCode),
            'fill': fill,
            'x': xList,
            'y': yList
        }
        self._plotDirtyFlag = True

        if replot:
            self.replot()

    def removeItem(self, legend, replot=True):
        try:
            del self._items[legend]
        except KeyError:
            pass
        if replot:
            self.replot()

    def clearItems(self):
        self._items = OrderedDict()

    def clear(self):
        self.clearItems()

    def replot(self):
        self.updateGL()

    # Draw mode #

    def isDrawModeEnabled(self):
        return isinstance(self.eventHandler, Draw)

    _drawModes = {
        'polygon': SelectPolygon,
        'rectangle': SelectRectangle,
        'line': SelectLine,
        'vline': SelectVLine,
        'hline': SelectHLine,
    }

    def setDrawModeEnabled(self, flag=True, shape="polygon", label=None, **kw):
        eventHandlerClass = self._drawModes[shape]
        if flag:
            parameters = kw
            parameters['shape'] = shape
            parameters['label'] = label
            if not isinstance(self.eventHandler, eventHandlerClass):
                self.eventHandler = eventHandlerClass(self, parameters)
        elif isinstance(self.eventHandler, eventHandlerClass):
            self.eventHandler = None

    def getDrawMode(self):
        if self.isDrawModeEnabled():
            return self.eventHandler.parameters
        else:
            None

    # Zoom #

    def isZoomModeEnabled(self):
        return isinstance(self.eventHandler, Zoom)

    def setZoomModeEnabled(self, flag=True):
        if flag:
            if not isinstance(self.eventHandler, Zoom):
                self.eventHandler = Zoom(self)
        elif isinstance(self.eventHandler, Zoom):
            self.eventHandler = None

    def resetZoom(self):
        if self.isXAxisAutoScale() and self.isYAxisAutoScale():
            self.setLimits(self._dataBBox['xMin'], self._dataBBox['xMax'],
                           self._dataBBox['yMin'], self._dataBBox['yMax'])
        elif self.isXAxisAutoScale():
            self.setGraphXLimits(self._dataBBox['xMin'],
                                 self._dataBBox['xMax'])
        elif self.isYAxisAutoScale():
            self.setGraphYLimits(self._dataBBox['yMin'],
                                 self._dataBBox['yMax'])

    # Limits #

    def _ensureAspectRatio(self):
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        plotRatio = plotWidth / float(plotHeight)

        dataW, dataH = self._xMax - self._xMin, self._yMax - self._yMin
        dataRatio = dataW / float(dataH)

        if dataRatio < plotRatio:
            dataW = dataH * plotRatio
            xCenter = (self._xMin + self._xMax) / 2.
            self._xMin = xCenter - dataW / 2.
            self._xMax = xCenter + dataW / 2.
        else:
            dataH = dataW / plotRatio
            yCenter = (self._yMin + self._yMax) / 2.
            self._yMin = yCenter - dataH / 2.
            self._yMax = yCenter + dataH / 2.

    def _setGraphXLimits(self, xMin, xMax):
        xMin = clamp(xMin, self._dataBBox['xMin'], self._dataBBox['xMax'])
        xMax = clamp(xMax, self._dataBBox['xMin'], self._dataBBox['xMax'])
        if xMax - xMin < self._dataBBox['xStep'] * 2:
            xCenter = clamp((xMax + xMin) / 2.,
                            self._dataBBox['xMin'] + self._dataBBox['xStep'],
                            self._dataBBox['xMax'] - self._dataBBox['xStep'])
            xMin = xCenter - self._dataBBox['xStep']
            xMax = xCenter + self._dataBBox['xStep']
        self._xMin, self._xMax = xMin, xMax

    def _setGraphYLimits(self, yMin, yMax):
        yMin = clamp(yMin, self._dataBBox['yMin'], self._dataBBox['yMax'])
        yMax = clamp(yMax, self._dataBBox['yMin'], self._dataBBox['yMax'])
        if yMax - yMin < self._dataBBox['yStep'] * 2:
            yCenter = clamp((yMax + yMin) / 2.,
                            self._dataBBox['yMin'] + self._dataBBox['yStep'],
                            self._dataBBox['yMax'] - self._dataBBox['yStep'])
            yMin = yCenter - self._dataBBox['yStep']
            yMax = yCenter + self._dataBBox['yStep']
        self._yMin, self._yMax = yMin, yMax

    def isKeepDataAspectRatio(self):
        return self._keepDataAspectRatio

    def keepDataAspectRatio(self, flag=True):
        self._keepDataAspectRatio = flag

        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

            self.updateAxis()
            self.updateGL()

    def setGraphXLimits(self, xMin, xMax):
        self._setGraphXLimits(xMin, xMax)
        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()
        self.updateGL()

    def setGraphYLimits(self, yMin, yMax):
        self._setGraphYLimits(yMin, yMax)

        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()
        self.updateGL()

    def setLimits(self, xMin, xMax, yMin, yMax):
        self._setGraphXLimits(xMin, xMax)
        self._setGraphYLimits(yMin, yMax)

        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()
        self.updateGL()

    def invertYAxis(self, flag=True):
        if flag != self._isYInverted:
            self._isYInverted = flag
            self.updateAxis()
            self.updateGL()

    def isYAxisInverted(self):
        return self._isYInverted

    # Autoscale #

    def isXAxisAutoScale(self):
        return self._xAutoScale

    def isYAxisAutoScale(self):
        return self._yAutoScale

    def setXAxisAutoScale(self, flag=True):
        self._xAutoScale = flag

    def setYAxisAutoScale(self, flag=True):
        self._yAutoScale = flag

    # Title, Labels
    def setGraphTitle(self, title=""):
        self._title = title
        self.updateGL()

    def getGraphTitle(self):
        return self._title

    def setGraphXLabel(self, label="X"):
        self._xLabel = label
        self.updateGL()

    def getGraphXLabel(self):
        return self._xLabel

    def setGraphYLabel(self, label="Y"):
        self._yLabel = label
        self.updateGL()

    def getGraphYLabel(self):
        return self._yLabel


# main ########################################################################

if __name__ == "__main__":
    import sys
    from ..Plot import Plot

    try:
        from PyQt4.QtGui import QApplication
    except ImportError:
        from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    w = Plot(None, backend=OpenGLBackend)

    size = 4096
    data = np.arange(float(size)*size, dtype=np.dtype(np.float32))
    data.shape = size, size

    colormap = {'name': 'gray', 'normalization': 'linear',
                'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                'colors': 256}
    w.addImage(data, legend="image 1",
               xScale=(25, 1.0), yScale=(-1000, 1.0),
               replot=False, colormap=colormap)

    w.getWidgetHandle().show()
    sys.exit(app.exec_())
