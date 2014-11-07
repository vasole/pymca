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
OpenGL/Qt backend
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

import numpy as np
import math

import OpenGL
if 0:  # Debug
    OpenGL.ERROR_ON_COPY = True
else:
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_ON_COPY = False

from OpenGL.GL import *  # noqa
from OpenGL.GL.ARB.texture_rg import GL_R32F  # Core in OpenGL 3

from ..PlotBackend import PlotBackend
from ..Plot import colordict
from .GLSupport import *  # noqa


# OrderedDict #################################################################

class MiniOrderedDict(object):
    """Simple subset of OrderedDict for python 2.6 support"""

    def __init__(self):
        self._dict = {}
        self._orderedKeys = []

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if key not in self._orderedKeys:
            self._orderedKeys.append(key)
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]
        self._orderedKeys.remove(key)

    def values(self):
        return [self._dict[key] for key in self._orderedKeys]

    def get(self, key, default=None):
        return self._dict.get(key, default)


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
    uniform int hatchStep;

    void main(void) {
        if (hatchStep == 0 ||
            mod(gl_FragCoord.x - gl_FragCoord.y, hatchStep) == 0) {
            gl_FragColor = color;
        } else {
            gl_FragColor = vec4(0., 0., 0., 0.);
        }
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


# signals #####################################################################

def prepareDrawingSignal(event, type_, points, parameters={}):
    eventDict = {}
    eventDict['event'] = event
    eventDict['type'] = type_
    points = np.array(points, dtype=np.float32)
    points.shape = -1, 2
    eventDict['points'] = points
    eventDict['xdata'] = points[:, 0]
    eventDict['ydata'] = points[:, 1]
    if type_ in ('rectangle'):
        eventDict['x'] = eventDict['xdata'].min()
        eventDict['y'] = eventDict['ydata'].min()
        eventDict['width'] = eventDict['xdata'].max() - eventDict['x']
        eventDict['height'] = eventDict['ydata'].max() - eventDict['y']
    eventDict['parameters'] = parameters.copy()
    return eventDict


def prepareMouseMovedSignal(button, xData, yData, xPixel, yPixel):
    return {'event': 'mouseMoved',
            'x': xData,
            'y': yData,
            'xpixel': xPixel,
            'ypixel': yPixel,
            'button': button}


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
            self.backend.replot()

    def beginDrag(self, x, y):
        self.x0, self.y0 = self.backend.pixelToDataCoords(x, y)

    def drag(self, x1, y1):
        x1, y1 = self.backend.pixelToDataCoords(x1, y1)
        if self.backend.isKeepDataAspectRatio():
            x1, y1 = self._ensureAspectRatio(self.x0, self.y0, x1, y1)

        self.backend.setSelectionArea(((self.x0, self.y0),
                                       (self.x0, y1),
                                       (x1, y1),
                                       (x1, self.y0)), fill=None)
        self.backend.replot()

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
        self.backend.replot()

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
        self.backend.replot()


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

        def updateSelectionArea(self):
            self.machine.backend.setSelectionArea(self.points)
            self.machine.backend.replot()
            eventDict = prepareDrawingSignal('drawingProgress',
                                             'polygon',
                                             self.points,
                                             self.machine.parameters)
            self.machine.backend._callback(eventDict)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                x, y = self.machine.backend.pixelToDataCoords(x, y)
                self.points[-1] = (x, y)
                self.updateSelectionArea()
                if self.points[-2] != self.points[-1]:
                    self.points.append((x, y))

        def onMove(self, x, y):
            x, y = self.machine.backend.pixelToDataCoords(x, y)
            self.points[-1] = (x, y)
            self.updateSelectionArea()

        def onPress(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.backend.setSelectionArea()
                self.machine.backend.replot()

                x, y = self.machine.backend.pixelToDataCoords(x, y)
                self.points[-1] = (x, y)
                if self.points[-2] == self.points[-1]:
                    self.points.pop()
                self.points.append(self.points[0])

                eventDict = prepareDrawingSignal('drawingFinished',
                                                 'polygon',
                                                 self.points,
                                                 self.machine.parameters)
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
        self.backend.setSelectionArea((self.startPt,
                                      (self.startPt[0], y),
                                      (x, y),
                                      (x, self.startPt[1])))
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'rectangle',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        self.backend.replot()

        x, y = self.backend.pixelToDataCoords(x, y)
        eventDict = prepareDrawingSignal('drawingFinished',
                                         'rectangle',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)


class SelectLine(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToDataCoords(x, y)

    def select(self, x, y):
        x, y = self.backend.pixelToDataCoords(x, y)
        self.backend.setSelectionArea((self.startPt, (x, y)))
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'line',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        self.backend.replot()

        x, y = self.backend.pixelToDataCoords(x, y)
        eventDict = prepareDrawingSignal('drawingFinished',
                                         'line',
                                         (self.startPt, (x, y)),
                                         self.parameters)
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
        points = self._hLine(y)
        self.backend.setSelectionArea(points)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'hline',
                                         points,
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'hline',
                                         self._hLine(y),
                                         self.parameters)
        self.backend._callback(eventDict)


class SelectVLine(Select1Point):
    def _vLine(self, x):
        x = self.backend.pixelToDataCoords(xPixel=x)
        return (x, self.backend._yMin), (x, self.backend._yMax)

    def select(self, x, y):
        points = self._vLine(x)
        self.backend.setSelectionArea(points)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'vline',
                                         points,
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.setSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'vline',
                                         self._vLine(x),
                                         self.parameters)
        self.backend._callback(eventDict)


# OpenGLPlotCanvas ############################################################

class OpenGLPlotCanvas(PlotBackend):
    def __init__(self, parent=None, **kw):
        self._xMin, self._xMax = 0., 1.
        self._yMin, self._yMax = 0., 1.
        self._keepDataAspectRatio = False
        self._isYInverted = False
        self._title, self._xLabel, self._yLabel = '', '', ''

        self.winWidth, self.winHeight = 0, 0
        self._dataBBox = {'xMin': 0., 'xMax': 0., 'xStep': 1.,
                          'yMin': 0., 'yMax': 0., 'yStep': 1.}
        self._images = MiniOrderedDict()
        self._items = MiniOrderedDict()
        self._curves = MiniOrderedDict()
        self._labels = []
        self._selectionArea = None

        self._margins = {'left': 100, 'right': 50, 'top': 50, 'bottom': 50}
        self._lineWidth = 1
        self._tickLen = 5

        self._axisDirtyFlag = True
        self._plotDirtyFlag = True

        self.eventHandler = None
        self._plotHasFocus = set()

        PlotBackend.__init__(self, parent, **kw)

    def updateGL(self):
        raise NotImplementedError("This method must be provided by \
                                  subclass to trigger redraw")

    def _mouseInPlotArea(self, x, y):
        xPlot = clamp(x, self._margins['left'],
                      self.winWidth - self._margins['right'])
        yPlot = clamp(y, self._margins['top'],
                      self.winHeight - self._margins['bottom'])
        return xPlot, yPlot

    def onMousePress(self, xPixel, yPixel, btn):
        if (self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel)):
            self._plotHasFocus.add(btn)
            if self.eventHandler is not None:
                self.eventHandler.handleEvent('press', xPixel, yPixel, btn)

    def onMouseMove(self, xPixel, yPixel):
        # Signal mouse move event
        xData, yData = self.pixelToDataCoords(xPixel, yPixel)
        if xData is not None and yData is not None:
            eventDict = prepareMouseMovedSignal(None, xData, yData,
                                                xPixel, yPixel)
            self._callback(eventDict)

        if self.eventHandler:
            xPlot, yPlot = self._mouseInPlotArea(xPixel, yPixel)
            self.eventHandler.handleEvent('move', xPlot, yPlot)

    def onMouseRelease(self, xPixel, yPixel, btn):
        try:
            self._plotHasFocus.remove(btn)
        except KeyError:
            pass
        else:
            if self.eventHandler:
                xPixel, yPixel = self._mouseInPlotArea(xPixel, yPixel)
                self.eventHandler.handleEvent('release', xPixel, yPixel, btn)

    def onMouseWheel(self, xPixel, yPixel, angleInDegrees):
        if self.eventHandler and \
           self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self.eventHandler.handleEvent('wheel', xPixel, yPixel,
                                          angleInDegrees)

    # Manage Plot #

    def setSelectionArea(self, points=None, fill='hatch'):
        if points:
            self._selectionArea = Shape2D(points, fill=fill,
                                          fillColor=(0., 0., 0., 0.5),
                                          stroke=True,
                                          strokeColor=(0., 0., 0., 1.))
        else:
            self._selectionArea = None

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
        for curve in self._curves.values():
            bbox = curve['bBox']
            xMin = min(xMin, bbox['xMin'])
            xMax = max(xMax, bbox['xMax'])
            yMin = min(yMin, bbox['yMin'])
            yMax = max(yMax, bbox['yMax'])

        if xMin >= xMax:
                xMin, xMax = 0., 1.
        if yMin >= yMax:
                yMin, yMax = 0., 1.
        if xStep == float('inf'):
            xStep = 1.
        if yStep == float('inf'):
            yStep = 1.

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

        nbLinePairs = 2 + len(xTicks) + len(yTicks)
        self._frameVertices = np.empty((nbLinePairs, 4, 2), dtype=np.float32)

        plotBottom = self.winHeight - self._margins['bottom']
        for index, xTick in enumerate(xTicks):
            tickText = ('{:.' + str(xNbFrac) + 'f}').format(xTick)
            xTick = self.dataToPixelCoords(xData=xTick)
            self._frameVertices[index][0] = xTick, plotBottom
            self._frameVertices[index][1] = xTick, plotBottom - self._tickLen

            self._frameVertices[index][2] = xTick, self._margins['top']
            self._frameVertices[index][3] = (xTick, self._margins['top'] +
                                             self._tickLen)

            self._labels.append((xTick, plotBottom + self._tickLen,
                                 Text2D(tickText, align=CENTER, valign=TOP)))

        plotRight = self.winWidth - self._margins['right']
        for index, yTick in enumerate(yTicks, len(xTicks)):
            tickText = ('{:.' + str(yNbFrac) + 'f}').format(yTick)
            yTick = self.dataToPixelCoords(yData=yTick)
            self._frameVertices[index][0] = self._margins['left'], yTick
            self._frameVertices[index][1] = (self._margins['left'] +
                                             self._tickLen, yTick)

            self._frameVertices[index][2] = plotRight, yTick
            self._frameVertices[index][3] = plotRight - self._tickLen, yTick

            self._labels.append((self._margins['left'] - self._tickLen,
                                 yTick,
                                 Text2D(tickText, align=RIGHT, valign=CENTER)))

        self._frameVertices.shape = (4 * nbLinePairs, 2)

        # Plot frame
        xLeft = self._margins['left']
        xRight = self.winWidth - self._margins['right']
        yBottom = self.winHeight - self._margins['bottom']
        yTop = self._margins['top']

        self._frameVertices[-8] = xLeft, yBottom
        self._frameVertices[-7] = xLeft, yTop

        self._frameVertices[-6] = xLeft, yTop
        self._frameVertices[-5] = xRight, yTop

        self._frameVertices[-4] = xRight, yTop
        self._frameVertices[-3] = xRight, yBottom

        self._frameVertices[-2] = xRight, yBottom
        self._frameVertices[-1] = xLeft, yBottom

        # Title, Labels
        plotCenterX = self._margins['left'] + plotWidth // 2
        plotCenterY = self._margins['top'] + plotHeight // 2
        if self._title:
            self._labels.append((plotCenterX,
                                 self._margins['top'] - self._tickLen,
                                 Text2D(self._title, align=CENTER,
                                        valign=BOTTOM)))
        if self._xLabel:
            self._labels.append((plotCenterX,
                                self.winHeight - self._margins['bottom'] // 2,
                                Text2D(self._xLabel, align=CENTER,
                                       valign=TOP)))

        if self._yLabel:
            self._labels.append((self._margins['left'] // 4,
                                plotCenterY,
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

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Create basic program
        self._progBase = Program(_baseVertShd, _baseFragShd)

        # Create texture program
        self._progTex = Program(_texVertShd, _texFragShd)

        # Create image program
        self._progImg = Program(_vertexSrc, _fragmentSrc)

    def _paintGLDirect(self):
        self._renderPlotArea()
        self._renderPlotFrame()
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
                self._renderPlotArea()
                self._renderPlotFrame()

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
            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            self._selectionArea.render(posAttrib, colorUnif, hatchStepUnif)

            glDisable(GL_SCISSOR_TEST)

    def _renderPlotFrame(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        # Render plot in screen coords
        glViewport(0, 0, self.winWidth, self.winHeight)

        self._updateAxis()

        # Render Plot frame
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self.matScreenProj)
        glUniform4f(self._progBase.uniforms['color'], 0., 0., 0., 1.)
        glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        glVertexAttribPointer(self._progBase.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self._frameVertices)
        glLineWidth(self._lineWidth)
        glDrawArrays(GL_LINES, 0, len(self._frameVertices))

        # Render Text
        self._progTex.use()
        textTexUnit = 0
        glUniform1i(self._progTex.uniforms['tex'], textTexUnit)

        for x, y, label in self._labels:
            glUniformMatrix4fv(self._progTex.uniforms['matrix'], 1, GL_TRUE,
                               self.matScreenProj * mat4Translate(x, y, 0))
            label.render(self, self._progTex.attributes['position'],
                         self._progTex.attributes['texCoords'],
                         textTexUnit)

    def _renderPlotArea(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

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

        # sorted is stable: original order is preserved when key is the same
        for image in sorted(self._images.values(), key=lambda d: d['zOrder']):
            try:
                texture = image['_texture']
            except KeyError:
                data = image['data']
                if len(data.shape) == 2:
                    height, width = data.shape
                    texture = Image(GL_R32F, width, height,
                                    format_=GL_RED, type_=GL_FLOAT,
                                    data=image['data'], texUnit=dataTexUnit)
                    image['_texture'] = texture
                else:
                    height, width, depth = data.shape
                    format_ = GL_RGBA if depth == 4 else GL_RGB
                    if data.dtype == np.uint8:
                        type_ = GL_UNSIGNED_BYTE
                    else:
                        type_ = GL_FLOAT
                    texture = Image(format_, width, height,
                                    format_=format_, type_=type_,
                                    data=image['data'], texUnit=dataTexUnit)
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

        # Render Curves
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           matDataProj)

        for curve in self._curves.values():
            try:
                vbo = curve['_vbo']
            except KeyError:
                vbo = VertexBuffer(curve['data'], usage=GL_STATIC_DRAW)
                curve['_vbo'] = vbo

            glUniform4f(self._progBase.uniforms['color'], *curve['color'])
            glUniform1i(self._progBase.uniforms['hatchStep'], 0)
            posAttrib = self._progBase.attributes['position']

            glEnableVertexAttribArray(posAttrib)
            with vbo:
                glVertexAttribPointer(posAttrib,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      0, c_void_p(0))
            glLineWidth(curve['lineWidth'])
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glDrawArrays(GL_LINE_STRIP, 0, len(curve['data']))

        # Render Items
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           matDataProj)

        for item in self._items.values():
            try:
                shape2D = item['_shape2D']
            except KeyError:
                shape2D = Shape2D(zip(item['x'], item['y']),
                                  fill=item['fill'],
                                  fillColor=item['color'],
                                  stroke=True,
                                  strokeColor=item['color'])
                item['_shape2D'] = shape2D

            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            shape2D.render(posAttrib, colorUnif, hatchStepUnif)

        glDisable(GL_SCISSOR_TEST)

    def resizeGL(self, width, height):
        self.winWidth, self.winHeight = width, height
        self.matScreenProj = mat4Ortho(0, self.winWidth,
                                       self.winHeight, 0,
                                       1, -1)
        self.setLimits(self._xMin, self._xMax, self._yMin, self._yMax)

        self.updateAxis()
        self.replot()

    # PlotBackend API #

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=0,
                 selectable=False, draggable=False,
                 colormap=None, **kwargs):
        # info is ignored
        if selectable or draggable:
            raise NotImplementedError("selectable and draggable \
                                      not implemented")

        oldImage = self._images.get(legend, None)
        if oldImage is not None and oldImage['data'].shape != data.shape:
            oldImage = None

        if replace:
            self.clearImages()

        height, width = data.shape[0:2]
        if xScale is None:
            xScale = (0, 1)
        if yScale is None:
            yScale = (0, 1)
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
                'zOrder': z,
                'data': data,
                'colormapName': colormap['name'][:],
                'colormapIsLog': colormap['normalization'].startswith('log'),
                'vmin': data.min() if colormap['autoscale']
                else colormap['vmin'],
                'vmax': data.max() if colormap['autoscale']
                else colormap['vmax'],
                'bBox': bbox
            }
            if oldImage is not None and '_texture' in oldImage:
                # Reuse texture and update
                texture = oldImage['_texture']
                texture.updateAll(format_=GL_RED, type_=GL_FLOAT,
                                  data=data)
                self._images[legend]['_texture'] = texture

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert(data.shape[2] in (3, 4))
            assert(data.dtype == np.uint8 or
                   np.can_cast(data.dtype, np.float32))

            self._images[legend] = {'zOrder': z, 'data': data, 'bBox': bbox}
            if oldImage is not None and '_texture' in oldImage:
                # Reuse texture and update
                format_ = GL_RGBA if data.shape[2] == 4 else GL_RGB
                if data.dtype == np.uint8:
                    type_ = GL_UNSIGNED_BYTE
                else:
                    type_ = GL_FLOAT

                texture = oldImage['_texture']
                texture.updateAll(format_=format_, type_=type_,
                                  data=data)
                self._images[legend]['_texture'] = texture

        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        if oldImage is None or bbox != oldImage['bBox']:
            self._updateDataBBox()
            self.setLimits(self._dataBBox['xMin'], self._dataBBox['xMax'],
                           self._dataBBox['yMin'], self._dataBBox['yMax'])

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend  # This is the 'handle'

    def removeImage(self, legend, replot=True):
        try:
            del self._images[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearImages(self):
        self._images = MiniOrderedDict()
        self._plotDirtyFlag = True

    def addItem(self, xList, yList, legend=None, info=None,
                replace=False, replot=True,
                shape="polygon", fill=True, **kwargs):
        # info is ignored
        if shape not in self._drawModes:
            raise NotImplementedError("Unsupported shape {0}".format(shape))

        if replace:
            self.clearItems()

        colorCode = kwargs.get('color', 'black')

        if shape == 'rectangle':
            xMin, xMax = xList
            xList = xMin, xMin, xMax, xMax
            yMin, yMax = yList
            yList = yMin, yMax, yMax, yMin

        self._items[legend] = {
            'shape': shape,
            'color': rgba(colorCode, colordict),
            'fill': 'hatch' if fill else None,
            'x': xList,
            'y': yList
        }
        self._plotDirtyFlag = True

        if replot:
            self.replot()
        return legend  # this is the 'handle'

    def removeItem(self, legend, replot=True):
        try:
            del self._items[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearItems(self):
        self._items = MiniOrderedDict()
        self._plotDirtyFlag = True

    def addCurve(self, x, y, legend=None, info=None,
                 replace=False, replot=True, **kw):

        data = np.array((x, y), dtype=np.float32, order='F').T

        oldCurve = self._curves.get(legend, None)
        if oldCurve is not None and oldCurve['data'].shape != data.shape:
            oldCurve = None

        if replace:
            self.clearCurves()

        # Copied from MatplotlibBackend, can be common
        if info is None:
            info = {}
        color = info.get('plot_color', self._activeCurveColor)
        color = kw.get('color', color)
        # symbol = info.get('plot_symbol', None)
        # symbol = kw.get('symbol', symbol)
        # style = info.get('plot_line_style', '-')
        # style = info.get('line_style', style)
        lineWidth = 1
        # axisId = info.get('plot_yaxis', 'left')
        # axisId = kw.get('yaxis', axisId)
        # fill = info.get('plot_fill', False)

        bbox = {
            'xMin': min(x),
            'xMax': max(x),
            'yMin': min(y),
            'yMax': max(y)
        }

        self._curves[legend] = {
            'data': data,
            # 'lineStyle': style,
            'lineWidth': lineWidth,
            'color': rgba(color, colordict),
            # 'symbol': symbol,
            # 'axes': axisId,
            # 'fill': fill,
            'bBox': bbox
        }

        if oldCurve is not None and '_vbo' in oldCurve:
            # Reuse vbo and update
            vbo = oldCurve['_vbo']
            vbo.update(data)
            self._curves[legend]['_vbo'] = vbo

        if oldCurve is None: # or bbox != oldCurve['bBox']:
            self._updateDataBBox()
            self.setLimits(self._dataBBox['xMin'], self._dataBBox['xMax'],
                           self._dataBBox['yMin'], self._dataBBox['yMax'])

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend

    def removeCurve(self, legend, replot=True):
        try:
            del self._curves[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearCurves(self):
        self._curves = MiniOrderedDict()
        self._plotDirtyFlag = True

    def clear(self):
        self.clearItems()
        self.clearCurves()

    def replot(self):
        self.updateGL()

    # Draw mode #

    def isDrawModeEnabled(self):
        return isinstance(self.eventHandler, Select)

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
        self.replot()

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

        self.resetZoom()
        self.updateAxis()
        self.replot()

    def setGraphXLimits(self, xMin, xMax):
        self._setGraphXLimits(xMin, xMax)
        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()

    def setGraphYLimits(self, yMin, yMax):
        self._setGraphYLimits(yMin, yMax)

        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()

    def setLimits(self, xMin, xMax, yMin, yMax):
        self._setGraphXLimits(xMin, xMax)
        self._setGraphYLimits(yMin, yMax)

        if self._keepDataAspectRatio:
            self._ensureAspectRatio()

        self.updateAxis()

    def invertYAxis(self, flag=True):
        if flag != self._isYInverted:
            self._isYInverted = flag
            self.updateAxis()

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

    def getGraphTitle(self):
        return self._title

    def setGraphXLabel(self, label="X"):
        self._xLabel = label

    def getGraphXLabel(self):
        return self._xLabel

    def setGraphYLabel(self, label="Y"):
        self._yLabel = label

    def getGraphYLabel(self):
        return self._yLabel


# OpenGLBackend ###############################################################

class OpenGLBackend(QGLWidget, OpenGLPlotCanvas):
    def __init__(self, parent=None, **kw):
        QGLWidget.__init__(self, parent)
        self.setAutoFillBackground(False)
        self.setMinimumSize(300, 300)  # TODO better way ?
        self.setMouseTracking(True)

        OpenGLPlotCanvas.__init__(self, parent, **kw)

    # Mouse events #
    _MOUSE_BTNS = {1: 'left', 2: 'right', 4: 'middle'}

    def mousePressEvent(self, event):
        xPixel, yPixel = event.x(), event.y()
        btn = self._MOUSE_BTNS[event.button()]
        self.onMousePress(xPixel, yPixel, btn)
        event.accept()

    def mouseMoveEvent(self, event):
        xPixel, yPixel = event.x(), event.y()
        self.onMouseMove(xPixel, yPixel)
        event.accept()

    def mouseReleaseEvent(self, event):
        xPixel, yPixel = event.x(), event.y()
        btn = self._MOUSE_BTNS[event.button()]
        self.onMouseRelease(xPixel, yPixel, btn)
        event.accept()

    def wheelEvent(self, event):
        xPixel, yPixel = event.x(), event.y()
        angleInDegrees = event.delta() / 8.
        self.onMouseWheel(xPixel, yPixel, angleInDegrees)
        event.accept()


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
