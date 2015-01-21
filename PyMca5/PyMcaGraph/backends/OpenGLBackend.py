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

from collections import namedtuple
import math
import numpy as np
import time
import warnings
import sys
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue

try:
    from PyMca5.PyMcaGui import PyMcaQt as qt
    QGLWidget = qt.QGLWidget
    QGLContext = qt.QGLContext
    pyqtSignal = qt.pyqtSignal
except ImportError:
    try:
        from PyQt4.QtCore import pyqtSignal
        from PyQt4.QtOpenGL import QGLWidget, QGLContext
    except ImportError:
        from PyQt5.QtCore import pyqtSignal
        from PyQt5.QtOpenGL import QGLWidget, QGLContext

try:
    from ..PlotBackend import PlotBackend
except ImportError:
    from PyMca5.PyMcaGraph.PlotBackend import PlotBackend

from .GLSupport.gl import *  # noqa
from .GLSupport import *  # noqa


# OrderedDict #################################################################

class MiniOrderedDict(object):
    """Simple subset of OrderedDict for python 2.6 support"""

    _DEFAULT_ARG = object()

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

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._orderedKeys[:]

    def values(self):
        return [self._dict[key] for key in self._orderedKeys]

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def pop(self, key, default=_DEFAULT_ARG):
        value = self._dict.pop(key, self._DEFAULT_ARG)
        if value is not self._DEFAULT_ARG:
            self._orderedKeys.remove(key)
            return value
        elif default is self._DEFAULT_ARG:
            raise KeyError
        else:
            return default


# Bounds ######################################################################

class Range(namedtuple('Range', ('min_', 'max_'))):
    """Describes a 1D range"""

    @property
    def range_(self):
        return self.max_ - self.min_

    @property
    def center(self):
        return 0.5 * (self.min_ + self.max_)


class Bounds(object):
    """Describes plot bounds with 2 y axis"""

    def __init__(self, xMin, xMax, yMin, yMax, y2Min, y2Max):
        self._xAxis = Range(xMin, xMax)
        self._yAxis = Range(yMin, yMax)
        self._y2Axis = Range(y2Min, y2Max)

    def __repr__(self):
        return "x: %s, y: %s, y2: %s" % (repr(self._xAxis),
                                         repr(self._yAxis),
                                         repr(self._y2Axis))

    @property
    def xAxis(self):
        return self._xAxis

    @property
    def yAxis(self):
        return self._yAxis

    @property
    def y2Axis(self):
        return self._y2Axis


# Image writer ################################################################

def convertRGBDataToPNG(data):
    """Convert a RGB bitmap to PNG.

    It only supports RGB bitmap with one byte per channel stored as a 3D array.
    See `Definitive Guide <http://www.libpng.org/pub/png/book/>`_ and
    `Specification <http://www.libpng.org/pub/png/spec/1.2/>`_ for details.

    :param data: A 3D array (h, w, rgb) storing an RGB image
    :type data: numpy.ndarray of unsigned bytes
    :returns: The PNG encoded data
    :rtype: bytes
    """
    import struct
    import zlib

    height, width = data.shape[0], data.shape[1]
    depth = 8  # 8 bit per channel
    colorType = 2  # 'truecolor' = RGB
    interlace = 0  # No

    pngData = []

    # PNG signature
    pngData.append(b'\x89PNG\r\n\x1a\n')

    # IHDR chunk: Image Header
    pngData.append(struct.pack(">I", 13))  # length
    IHDRdata = struct.pack(">ccccIIBBBBB", b'I', b'H', b'D', b'R',
                           width, height, depth, colorType,
                           0, 0, interlace)
    pngData.append(IHDRdata)
    pngData.append(struct.pack(">I", zlib.crc32(IHDRdata) & 0xffffffff))  # CRC

    # Add filter 'None' before each scanline
    preparedData = b'\x00' + b'\x00'.join(line.tostring() for line in data)
    compressedData = zlib.compress(preparedData, 8)

    # IDAT chunk: Payload
    pngData.append(struct.pack(">I", len(compressedData)))
    IDATdata = struct.pack("cccc", b'I', b'D', b'A', b'T')
    IDATdata += compressedData
    pngData.append(IDATdata)
    pngData.append(struct.pack(">I", zlib.crc32(IDATdata) & 0xffffffff))  # CRC

    # IEND chunk: footer
    pngData.append(b'\x00\x00\x00\x00IEND\xaeB`\x82')
    return b''.join(pngData)


def saveImageToFile(data, fileName, fileFormat):
    """Save a RGB image to a file.

    :param data: A 3D array (h, w, 3) storing an RGB image
    :type data: numpy.ndarray with of unsigned bytes
    :param str fileName: The fileName where to write the image
    :param str fileType: The type of the file in: 'png', 'ppm', 'svg', 'tiff'
    """
    assert len(data.shape) == 3
    assert data.shape[2] == 3
    assert fileFormat in ('png', 'ppm', 'svg', 'tiff')

    if fileFormat == 'svg':
        import base64

        height, width = data.shape[:2]
        base64Data = base64.b64encode(convertRGBDataToPNG(data))

        with open(fileName, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            f.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n')
            f.write('  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
            f.write('<svg xmlns:xlink="http://www.w3.org/1999/xlink"\n')
            f.write('     xmlns="http://www.w3.org/2000/svg"\n')
            f.write('     version="1.1"\n')
            f.write('     width="%d"\n' % width)
            f.write('     height="%d">\n' % height)
            f.write('    <image xlink:href="data:image/png;base64,')
            f.write(base64Data.decode('ascii'))
            f.write('"\n')
            f.write('           x="0"\n')
            f.write('           y="0"\n')
            f.write('           width="%d"\n' % width)
            f.write('           height="%d"\n' % height)
            f.write('           id="image" />\n')
            f.write('</svg>')

    elif fileFormat == 'ppm':
        with open(fileName, 'w') as f:
            f.write('P6\n')
            f.write('%d %d\n' % (self.winWidth, self.winHeight))
            f.write('255\n')
            f.write(data.tostring())

    elif fileFormat == 'png':
        with open(fileName, 'wb') as f:
            f.write(convertRGBDataToPNG(data))

    elif fileFormat == 'tiff':
        from PyMca5.PyMcaIO.TiffIO import TiffIO
        tif = TiffIO(fileName, mode='wb+')
        tif.writeImage(data, info={'Title': 'PyMCA GL Snapshot'})


# shaders #####################################################################

_baseVertShd = """
    attribute vec2 position;
    uniform mat4 matrix;
    uniform bvec2 isLog;

    const float oneOverLog10 = 1.0 / log(10.0);

    void main(void) {
        vec2 posTransformed = position;
        if (isLog.x) {
            posTransformed.x = oneOverLog10 * log(position.x);
        }
        if (isLog.y) {
            posTransformed.y = oneOverLog10 * log(position.y);
        }
        gl_Position = matrix * vec4(posTransformed, 0.0, 1.0);
    }
    """

_baseFragShd = """
    uniform vec4 color;
    uniform int hatchStep;
    uniform float tickLen;

    void main(void) {
        if (tickLen != 0) {
            if (mod((gl_FragCoord.x + gl_FragCoord.y) / tickLen, 2.) < 1.) {
                gl_FragColor = color;
            } else {
                discard;
            }
        } else if (hatchStep == 0 ||
            mod(gl_FragCoord.x - gl_FragCoord.y, hatchStep) == 0) {
            gl_FragColor = color;
        } else {
            discard;
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


# utils #######################################################################

def _ticks(start, stop, step):
    """range for float (including stop)
    """
    while start <= stop:
            yield start
            start += step


# signals #####################################################################

def prepareDrawingSignal(event, type_, points, parameters=None):
    if parameters is None:
        parameters = {}

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


def prepareMouseSignal(eventType, button, xData, yData, xPixel, yPixel):
    assert eventType in ('mouseMoved', 'mouseClicked', 'mouseDoubleClicked')
    assert button in (None, 'left', 'right')

    return {'event': eventType,
            'x': xData,
            'y': yData,
            'xpixel': xPixel,
            'ypixel': yPixel,
            'button': button}


def prepareHoverSignal(label, type_, posData, posPixel, draggable, selectable):
    return {'event': 'hover',
            'label': label,
            'type': type_,
            'x': posData[0],
            'y': posData[1],
            'xpixel': posPixel[0],
            'ypixel': posPixel[1],
            'draggable': draggable,
            'selectable': selectable}


def prepareMarkerSignal(eventType, button, label, type_,
                        draggable, selectable,
                        posDataMarker,
                        posPixelCursor=None, posDataCursor=None):
    if eventType == 'markerClicked':
        assert posPixelCursor is not None
        assert posDataCursor is None

        posDataCursor = list(posDataMarker)
        if hasattr(posDataCursor[0], "__len__"):
            posDataCursor[0] = posDataCursor[0][-1]
        if hasattr(posDataCursor[1], "__len__"):
            posDataCursor[1] = posDataCursor[1][-1]

    elif eventType == 'markerMoving':
        assert posPixelCursor is not None
        assert posDataCursor is not None

    elif eventType == 'markerMoved':
        assert posPixelCursor is None
        assert posDataCursor is None

        posDataCursor = posDataMarker
    else:
        raise NotImplementedError("Unknown event type {0}".format(eventType))

    eventDict = {'event': eventType,
                 'button': button,
                 'label': label,
                 'type': type_,
                 'x': posDataCursor[0],
                 'y': posDataCursor[1],
                 'xdata': posDataMarker[0],
                 'ydata': posDataMarker[1],
                 'draggable': draggable,
                 'selectable': selectable}

    if eventType in ('markerMoving', 'markerClicked'):
        eventDict['xpixel'] = posPixelCursor[0]
        eventDict['ypixel'] = posPixelCursor[1]

    return eventDict


def prepareImageSignal(button, label, type_, col, row,
                       x, y, xPixel, yPixel):
    return {'event': 'imageClicked',
            'button': button,
            'label': label,
            'type': type_,
            'col': col,
            'row': row,
            'x': x,
            'y': y,
            'xpixel': xPixel,
            'ypixel': yPixel}


def prepareCurveSignal(button, label, type_, xData, yData,
                       x, y, xPixel, yPixel):
    return {'event': 'curveClicked',
            'button': button,
            'label': label,
            'type': type_,
            'xdata': xData,
            'ydata': yData,
            'x': x,
            'y': y,
            'xpixel': xPixel,
            'ypixel': yPixel}


# Interaction #################################################################

class Zoom(ClickOrDrag):
    _DOUBLE_CLICK_TIMEOUT = 0.4

    class ZoomIdle(ClickOrDrag.Idle):
        def onWheel(self, x, y, angle):
            scaleF = 1.1 if angle > 0 else 1./1.1
            self.machine._zoom(x, y, scaleF)

    def __init__(self, backend, color):
        self.backend = backend
        self.color = color
        self.zoomStack = []
        self._lastClick = 0., None

        states = {
            'idle': Zoom.ZoomIdle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

    # Selection area constrained by aspect ratio
    # def _ensureAspectRatio(self, x0, y0, x1, y1):
    #    plotW, plotH = self.backend.plotSizeInPixels()
    #    try:
    #        plotRatio = plotW / float(plotH)
    #    except ZeroDivisionError:
    #        pass
    #    else:
    #        width, height = math.fabs(x1 - x0), math.fabs(y1 - y0)
    #
    #        try:
    #            selectRatio = width / height
    #        except ZeroDivisionError:
    #            width, height = 1., 1.
    #        else:
    #            if selectRatio < plotRatio:
    #                height = width / plotRatio
    #            else:
    #                width = height * plotRatio
    #        x1 = x0 + np.sign(x1 - x0) * width
    #        y1 = y0 + np.sign(y1 - y0) * height
    #    return x1, y1

    def _areaWithAspectRatio(self, x0, y0, x1, y1):
        plotW, plotH = self.backend.plotSizeInPixels()

        if plotH != 0.:
            plotRatio = plotW / float(plotH)
            width, height = math.fabs(x1 - x0), math.fabs(y1 - y0)

            if height == 0. or width == 0.:
                areaX0, areaY0, areaX1, areaY1 = x0, y0, x1, y1
            else:
                if width / height > plotRatio:
                    areaHeight = width / plotRatio
                    areaX0, areaX1 = x0, x1
                    center = 0.5 * (y0 + y1)
                    areaY0 = center - np.sign(y1 - y0) * 0.5 * areaHeight
                    areaY1 = center + np.sign(y1 - y0) * 0.5 * areaHeight
                else:
                    areaWidth = height * plotRatio
                    areaY0, areaY1 = y0, y1
                    center = 0.5 * (x0 + x1)
                    areaX0 = center - np.sign(x1 - x0) * 0.5 * areaWidth
                    areaX1 = center + np.sign(x1 - x0) * 0.5 * areaWidth

        return areaX0, areaY0, areaX1, areaY1

    def click(self, x, y, btn):
        if btn == LEFT_BTN:
            lastClickTime, lastClickPos = self._lastClick

            # Signal mouse double clicked event first
            if (time.time() - lastClickTime) <= self._DOUBLE_CLICK_TIMEOUT:
                # Use position of first click
                eventDict = prepareMouseSignal('mouseDoubleClicked', 'left',
                                               *lastClickPos)
                self.backend._callback(eventDict)

                self._lastClick = 0., None
            else:
                # Signal mouse clicked event
                xData, yData = self.backend.pixelToDataCoords(x, y)
                assert xData is not None and yData is not None
                eventDict = prepareMouseSignal('mouseClicked', 'left',
                                               xData, yData,
                                               x, y)
                self.backend._callback(eventDict)

                self._lastClick = time.time(), (xData, yData, x, y)

            # Zoom-in centered on mouse cursor
            # xMin, xMax = self.backend.getGraphXLimits()
            # yMin, yMax = self.backend.getGraphYLimits()
            # y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
            # self.zoomStack.append((xMin, xMax, yMin, yMax, y2Min, y2Max))
            # self._zoom(x, y, 2)
        elif btn == RIGHT_BTN:
            try:
                xMin, xMax, yMin, yMax, y2Min, y2Max = self.zoomStack.pop()
            except IndexError:
                # Signal mouse clicked event
                xData, yData = self.backend.pixelToDataCoords(x, y)
                assert xData is not None and yData is not None
                eventDict = prepareMouseSignal('mouseClicked', 'right',
                                               xData, yData,
                                               x, y)
                self.backend._callback(eventDict)
            else:
                self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
            self.backend.replot()

    def beginDrag(self, x, y):
        self.x0, self.y0 = self.backend.pixelToDataCoords(x, y)

    def drag(self, x1, y1):
        x1, y1 = self.backend.pixelToDataCoords(x1, y1)

        # Selection area constrained by aspect ratio
        # if self.backend.isKeepDataAspectRatio():
        #    x1, y1 = self._ensureAspectRatio(self.x0, self.y0, x1, y1)

        if self.backend.isKeepDataAspectRatio():
            area = self._areaWithAspectRatio(self.x0, self.y0, x1, y1)
            areaX0, areaY0, areaX1, areaY1 = area
            areaPoints = ((areaX0, areaY0),
                          (areaX1, areaY0),
                          (areaX1, areaY1),
                          (areaX0, areaY1))
            areaColor = list(self.color)
            areaColor[3] *= 0.25
            self.backend.setSelectionArea(areaPoints,
                                          fill=None,
                                          color=areaColor,
                                          name="zoomedArea")

        points = ((self.x0, self.y0),
                  (self.x0, y1),
                  (x1, y1),
                  (x1, self.y0))
        self.backend.setSelectionArea(points,
                                      fill=None,
                                      color=self.color)
        self.backend.replot()

    def endDrag(self, startPos, endPos):
        xMin, xMax = self.backend.getGraphXLimits()
        yMin, yMax = self.backend.getGraphYLimits()
        y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
        self.zoomStack.append((xMin, xMax, yMin, yMax, y2Min, y2Max))

        x0, y0 = self.backend.pixelToDataCoords(*startPos)
        y2_0 = self.backend.pixelToDataCoords(yPixel=startPos[1], axis="right")
        x1, y1 = self.backend.pixelToDataCoords(*endPos)
        y2_1 = self.backend.pixelToDataCoords(yPixel=endPos[1], axis="right")

        # Selection area constrained by aspect ratio
        # if self.backend.isKeepDataAspectRatio():
        #     x1, y1 = self._ensureAspectRatio(x0, y0, x1, y1)

        xMin, xMax = min(x0, x1), max(x0, x1)
        yMin, yMax = min(y0, y1), max(y0, y1)
        y2Min, y2Max = min(y2_0, y2_1), max(y2_0, y2_1)

        if xMin != xMax and yMin != yMax and y2Min != y2Max:
            # Avoid null zoom area
            self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        self.backend.resetSelectionArea()
        self.backend.replot()

    def _newZoomRange(self, min_, max_, center, scale, isLog):
        if isLog:
            center = math.log10(center)
            oldMin, oldMax = math.log10(min_), math.log10(max_)
        else:
            oldMin, oldMax = min_, max_

        offset = (center - oldMin) / (oldMax - oldMin)
        range_ = (oldMax - oldMin) / scale
        newMin = center - offset * range_
        newMax = center + (1. - offset) * range_
        if isLog:
            try:
                newMin, newMax = 10. ** newMin, 10. ** newMax
            except OverflowError:  # Limit case
                newMin, newMax = min_, max_
            if newMin <= 0. or newMax <= 0.:  # Limit case
                newMin, newMax = min_, max_
        return newMin, newMax

    def _zoom(self, cx, cy, scaleF):
        xCenter = self.backend.pixelToDataCoords(xPixel=cx)
        xMin, xMax = self.backend.getGraphXLimits()
        xMin, xMax = self._newZoomRange(xMin, xMax, xCenter, scaleF,
                                        self.backend.isXAxisLogaritmic())

        yCenter = self.backend.pixelToDataCoords(yPixel=cy)
        yMin, yMax = self.backend.getGraphYLimits()
        yMin, yMax = self._newZoomRange(yMin, yMax, yCenter, scaleF,
                                        self.backend.isYAxisLogaritmic())

        y2Center = self.backend.pixelToDataCoords(yPixel=cy, axis="right")
        y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
        y2Min, y2Max = self._newZoomRange(y2Min, y2Max, y2Center, scaleF,
                                          self.backend.isYAxisLogaritmic())

        self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
        self.backend.replot()

    def cancel(self):
        if isinstance(self.state, self.states['drag']):
            self.backend.resetSelectionArea()
            self.backend.replot()


class Select(object):
    parameters = {}

    @property
    def color(self):
        return self.parameters.get('color', None)


class SelectPolygon(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
            x, y = self.machine.backend.pixelToDataCoords(x, y)
            self.points = [(x, y), (x, y)]

        def updateSelectionArea(self):
            self.machine.backend.setSelectionArea(self.points,
                                                  fill='hatch',
                                                  color=self.machine.color)
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
                return True

        def onMove(self, x, y):
            x, y = self.machine.backend.pixelToDataCoords(x, y)
            self.points[-1] = (x, y)
            self.updateSelectionArea()

        def onPress(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.backend.resetSelectionArea()
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
                self.goto('idle')

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        states = {
            'idle': SelectPolygon.Idle,
            'select': SelectPolygon.Select
        }
        super(SelectPolygon, self).__init__(states, 'idle')

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.backend.resetSelectionArea()
            self.backend.replot()


class Select2Points(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('start', x, y)
                return True

    class Start(State):
        def enter(self, x, y):
            self.machine.beginSelect(x, y)

        def onMove(self, x, y):
            self.goto('select', x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto('idle')

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        states = {
            'idle': Select2Points.Idle,
            'start': Select2Points.Start,
            'select': Select2Points.Select
        }
        super(Select2Points, self).__init__(states, 'idle')

    def beginSelect(self, x, y):
        pass

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass

    def cancelSelect(self):
        pass

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.cancelSelect()


class SelectRectangle(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToDataCoords(x, y)

    def select(self, x, y):
        x, y = self.backend.pixelToDataCoords(x, y)
        self.backend.setSelectionArea((self.startPt,
                                      (self.startPt[0], y),
                                      (x, y),
                                      (x, self.startPt[1])),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'rectangle',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        x, y = self.backend.pixelToDataCoords(x, y)
        eventDict = prepareDrawingSignal('drawingFinished',
                                         'rectangle',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class SelectLine(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToDataCoords(x, y)

    def select(self, x, y):
        x, y = self.backend.pixelToDataCoords(x, y)
        self.backend.setSelectionArea((self.startPt, (x, y)),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'line',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        x, y = self.backend.pixelToDataCoords(x, y)
        eventDict = prepareDrawingSignal('drawingFinished',
                                         'line',
                                         (self.startPt, (x, y)),
                                         self.parameters)
        self.backend._callback(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class Select1Point(StateMachine, Select):
    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('select', x, y)
                return True

    class Select(State):
        def enter(self, x, y):
            self.onMove(x, y)

        def onMove(self, x, y):
            self.machine.select(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endSelect(x, y)
                self.goto('idle')

    def __init__(self, backend, parameters):
        self.parameters = parameters
        self.backend = backend
        states = {
            'idle': Select1Point.Idle,
            'select': Select1Point.Select
        }
        super(Select1Point, self).__init__(states, 'idle')

    def select(self, x, y):
        pass

    def endSelect(self, x, y):
        pass

    def cancelSelect(self):
        pass

    def cancel(self):
        if isinstance(self.state, self.states['select']):
            self.cancelSelect()


class SelectHLine(Select1Point):
    def _hLine(self, y):
        y = self.backend.pixelToDataCoords(yPixel=y)
        xMin, xMax = self.backend.getGraphXLimits()
        return (xMin, y), (xMax, y)

    def select(self, x, y):
        points = self._hLine(y)
        self.backend.setSelectionArea(points, fill='hatch', color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'hline',
                                         points,
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'hline',
                                         self._hLine(y),
                                         self.parameters)
        self.backend._callback(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class SelectVLine(Select1Point):
    def _vLine(self, x):
        x = self.backend.pixelToDataCoords(xPixel=x)
        yMin, yMax = self.backend.getGraphYLimits()
        return (x, yMin), (x, yMax)

    def select(self, x, y):
        points = self._vLine(x)
        self.backend.setSelectionArea(points, fill='hatch', color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'vline',
                                         points,
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'vline',
                                         self._vLine(x),
                                         self.parameters)
        self.backend._callback(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class ItemsInteraction(ClickOrDrag):
    class Idle(ClickOrDrag.Idle):
        def __init__(self, *args, **kwargs):
            super(ItemsInteraction.Idle, self).__init__(*args, **kwargs)
            self._hoverMarker = None

        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                testBehaviors = set(('selectable', 'draggable'))

                marker = self.machine.backend.pickMarker(
                    x, y,
                    lambda marker: marker['behaviors'] & testBehaviors)
                if marker is not None:
                    self.goto('clickOrDrag', x, y)
                    return True

                else:
                    picked = self.machine.backend.pickImageOrCurve(
                        x,
                        y,
                        lambda item: item.info['behaviors'] & testBehaviors)
                    if picked is not None:
                        self.goto('clickOrDrag', x, y)
                        return True

            return False

        def onMove(self, x, y):
            marker = self.machine.backend.pickMarker(x, y)
            if marker is not None:
                posData = self.machine.backend.pixelToDataCoords(x, y)
                eventDict = prepareHoverSignal(
                    marker['label'], 'marker',
                    posData, (x, y),
                    'draggable' in marker['behaviors'],
                    'selectable' in marker['behaviors'])
                self.machine.backend._callback(eventDict)

            if marker != self._hoverMarker:
                self._hoverMarker = marker

                if marker is None:
                    self.machine.backend.setCursor()

                elif 'draggable' in marker['behaviors']:
                    if marker['x'] is None:
                        self.machine.backend.setCursor(CURSOR_SIZE_VER)
                    elif marker['y'] is None:
                        self.machine.backend.setCursor(CURSOR_SIZE_HOR)
                    else:
                        self.machine.backend.setCursor(CURSOR_SIZE_ALL)

                elif 'selectable' in marker['behaviors']:
                    self.machine.backend.setCursor(CURSOR_POINTING)

            return True

    def __init__(self, backend):
        self.backend = backend

        states = {
            'idle': ItemsInteraction.Idle,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        StateMachine.__init__(self, states, 'idle')

    def click(self, x, y, btn):
        if btn == LEFT_BTN:
            marker = self.backend.pickMarker(
                x, y, lambda marker: 'selectable' in marker['behaviors'])
            if marker is not None:
                # Mimic MatplotlibBackend signal
                xData, yData = marker['x'], marker['y']
                if xData is None:
                    xData = [0, 1]
                if yData is None:
                    yData = [0, 1]

                draggable = 'draggable' in marker['behaviors']
                selectable = 'selectable' in marker['behaviors']
                eventDict = prepareMarkerSignal('markerClicked',
                                                'left',
                                                marker['label'],
                                                'marker',
                                                draggable,
                                                selectable,
                                                (xData, yData),
                                                (x, y), None)
                self.backend._callback(eventDict)

                self.backend.replot()
            else:
                picked = self.backend.pickImageOrCurve(
                    x,
                    y,
                    lambda item: 'selectable' in item.info['behaviors'])

                if picked is None:
                    pass
                elif picked[0] == 'curve':
                    _, curve, indices = picked
                    xData, yData = self.backend.pixelToDataCoords(x, y)
                    eventDict = prepareCurveSignal('left',
                                                   curve.info['legend'],
                                                   'curve',
                                                   curve.xData[indices],
                                                   curve.yData[indices],
                                                   xData, yData, x, y)
                    self.backend._callback(eventDict)

                elif picked[0] == 'image':
                    _, image, posImg = picked

                    xData, yData = self.backend.pixelToDataCoords(x, y)
                    eventDict = prepareImageSignal('left',
                                                   image.info['legend'],
                                                   'image',
                                                   posImg[0], posImg[1],
                                                   xData, yData, x, y)
                    self.backend._callback(eventDict)

    def _signalMarkerMovingEvent(self, eventType, marker, x, y):
        assert marker is not None

        # Mimic MatplotlibBackend signal
        xData, yData = marker['x'], marker['y']
        if xData is None:
            xData = [0, 1]
        if yData is None:
            yData = [0, 1]

        posDataCursor = self.backend.pixelToDataCoords(x, y)

        eventDict = prepareMarkerSignal(eventType,
                                        'left',
                                        marker['label'],
                                        'marker',
                                        'draggable' in marker['behaviors'],
                                        'selectable' in marker['behaviors'],
                                        (xData, yData),
                                        (x, y),
                                        posDataCursor)
        self.backend._callback(eventDict)

    def beginDrag(self, x, y):
        self._lastPos = self.backend.pixelToDataCoords(x, y)
        self.image = None
        self.marker = self.backend.pickMarker(
            x, y, lambda marker: 'draggable' in marker['behaviors'])
        if self.marker is not None:
            self._signalMarkerMovingEvent('markerMoving', self.marker, x, y)
        else:
            picked = self.backend.pickImageOrCurve(
                x,
                y,
                lambda item: 'draggable' in item.info['behaviors'])
            if picked is None:
                self.image = None
                self.backend.setCursor()
            else:
                assert picked[0] == 'image'  # For now, only drag images
                self.image = picked[1]

    def drag(self, x, y):
        xData, yData = self.backend.pixelToDataCoords(x, y)
        if self.marker is not None:
            if self.marker['x'] is not None:
                self.marker['x'] = xData
            if self.marker['y'] is not None:
                self.marker['y'] = yData

            self._signalMarkerMovingEvent('markerMoving', self.marker, x, y)

            self.backend.replot()

        if self.image is not None:
            dx, dy = xData - self._lastPos[0], yData - self._lastPos[1]
            self.image.xMin += dx
            self.image.yMin += dy

            self.backend._plotDirtyFlag = True
            self.backend.replot()

        self._lastPos = xData, yData

    def endDrag(self, startPos, endPos):
        if self.marker is not None:
            posData = [self.marker['x'], self.marker['y']]
            # Mimic MatplotlibBackend signal
            if posData[0] is None:
                posData[0] = [0, 1]
            if posData[1] is None:
                posData[1] = [0, 1]

            eventDict = prepareMarkerSignal(
                'markerMoved',
                'left',
                self.marker['label'],
                'marker',
                'draggable' in self.marker['behaviors'],
                'selectable' in self.marker['behaviors'],
                posData)
            self.backend._callback(eventDict)

        del self.marker
        del self.image
        del self._lastPos

    def cancel(self):
        self.backend.setCursor()


class FocusManager(StateMachine):
    """Manages focus across multiple event handlers

    On press an event handler can acquire focus.
    By default it looses focus when all buttons are released.
    """
    class Idle(State):
        def onPress(self, x, y, btn):
            for eventHandler in self.machine.eventHandlers:
                requestFocus = eventHandler.handleEvent('press', x, y, btn)
                if requestFocus:
                    self.goto('focus', eventHandler, btn)
                    break

        def _processEvent(self, *args):
            for eventHandler in self.machine.eventHandlers:
                consumeEvent = eventHandler.handleEvent(*args)
                if consumeEvent:
                    break

        def onMove(self, x, y):
            self._processEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self._processEvent('release', x, y, btn)

        def onWheel(self, x, y, angle):
            self._processEvent('wheel', x, y, angle)

    class Focus(State):
        def enter(self, eventHandler, btn):
            self.eventHandler = eventHandler
            self.focusBtns = set((btn,))

        def onPress(self, x, y, btn):
            self.focusBtns.add(btn)
            self.eventHandler.handleEvent('press', x, y, btn)

        def onMove(self, x, y):
            self.eventHandler.handleEvent('move', x, y)

        def onRelease(self, x, y, btn):
            self.focusBtns.discard(btn)
            requestFocus = self.eventHandler.handleEvent('release', x, y, btn)
            if len(self.focusBtns) == 0 and not requestFocus:
                self.goto('idle')

        def onWheel(self, x, y, angleInDegrees):
            self.eventHandler.handleEvent('wheel', x, y, angleInDegrees)

    def __init__(self, eventHandlers=()):
        self.eventHandlers = list(eventHandlers)

        states = {
            'idle': FocusManager.Idle,
            'focus': FocusManager.Focus
        }
        super(FocusManager, self).__init__(states, 'idle')

    def cancel(self):
        for handler in self.eventHandlers:
            handler.cancel()


class ZoomAndSelect(FocusManager):
    def __init__(self, backend, color):
        eventHandlers = ItemsInteraction(backend), Zoom(backend, color)
        super(ZoomAndSelect, self).__init__(eventHandlers)


# OpenGLPlotCanvas ############################################################

(CURSOR_DEFAULT, CURSOR_POINTING, CURSOR_SIZE_HOR,
 CURSOR_SIZE_VER, CURSOR_SIZE_ALL) = range(5)


class OpenGLPlotCanvas(PlotBackend):
    """Implements PlotBackend API using OpenGL.

    WARNINGS:
    Unless stated otherwise, this API is NOT thread-safe and MUST be
    called from the main thread.
    When numpy arrays are passed as arguments to the API (through
    :func:`addCurve` and :func:`addImage`), they are copied only if
    required.
    So, the caller should not modify these arrays afterwards.
    """

    _PICK_OFFSET = 3

    def __init__(self, parent=None, **kw):
        self._plotDataBounds = Bounds(1., 100., 1., 100., 1., 100.)
        self._keepDataAspectRatio = False
        self._isYInverted = False
        self._title = ''
        self._xLabel = ''
        self._yLabel = ''
        self._isXLog = False
        self._isYLog = False

        self._grid = False
        self._activeCurve = None

        self._zoomColor = None

        self.winWidth, self.winHeight = 0, 0

        self._markers = MiniOrderedDict()
        self._items = MiniOrderedDict()
        self._zOrderedItems = MiniOrderedDict()  # For images and curves
        self._labels = []
        self._selectionAreas = MiniOrderedDict()
        self._glGarbageCollector = []
        self._graphsToSave = queue.Queue()

        self._margins = {'left': 100, 'right': 50, 'top': 50, 'bottom': 50}
        self._lineWidth = 1
        self._tickLen = 5

        self._axisDirtyFlag = True
        self._plotDirtyFlag = True

        self._hasRightYAxis = set()

        self._mousePosition = 0, 0
        self.eventHandler = ZoomAndSelect(self, (0., 0., 0., 1.))

        self._plotHasFocus = set()

        PlotBackend.__init__(self, parent, **kw)

    # Link with embedding toolkit #

    def postRedisplay(self):
        raise NotImplementedError("This method must be provided by \
                                  subclass to trigger redraw")

    def setCursor(self, cursor=CURSOR_DEFAULT):
        """Override this method in subclass to enable cursor shape changes
        """
        print('setCursor:', cursor)

    # User event handling #

    def _mouseInPlotArea(self, x, y):
        xPlot = clamp(x, self._margins['left'],
                      self.winWidth - self._margins['right'])
        yPlot = clamp(y, self._margins['top'],
                      self.winHeight - self._margins['bottom'])
        return xPlot, yPlot

    def onMousePress(self, xPixel, yPixel, btn):
        if self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self._plotHasFocus.add(btn)
            self.eventHandler.handleEvent('press', xPixel, yPixel, btn)

    def onMouseMove(self, xPixel, yPixel):
        inXPixel, inYPixel = self._mouseInPlotArea(xPixel, yPixel)
        isCursorInPlot = inXPixel == xPixel and inYPixel == yPixel

        if isCursorInPlot:
            # Signal mouse move event
            xData, yData = self.pixelToDataCoords(inXPixel, inYPixel)
            assert xData is not None
            assert yData is not None
            eventDict = prepareMouseSignal('mouseMoved', None,
                                           xData, yData,
                                           xPixel, yPixel)
            self._callback(eventDict)

        # Either button was pressed in the plot or cursor is in the plot
        if isCursorInPlot or self._plotHasFocus:
            self.eventHandler.handleEvent('move', inXPixel, inYPixel)

    def onMouseRelease(self, xPixel, yPixel, btn):
        try:
            self._plotHasFocus.remove(btn)
        except KeyError:
            pass
        else:
            xPixel, yPixel = self._mouseInPlotArea(xPixel, yPixel)
            self.eventHandler.handleEvent('release', xPixel, yPixel, btn)

    def onMouseWheel(self, xPixel, yPixel, angleInDegrees):
        if self._mouseInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self.eventHandler.handleEvent('wheel', xPixel, yPixel,
                                          angleInDegrees)

    # Picking #

    def pickMarker(self, x, y, test=None):
        if test is None:
            test = lambda marker: True
        for marker in reversed(self._markers.values()):
            if marker['x'] is not None:
                xMarker = self.dataToPixelCoords(xData=marker['x'])
                xDist = math.fabs(x - xMarker)
            else:
                xDist = 0

            if marker['y'] is not None:
                yMarker = self.dataToPixelCoords(yData=marker['y'])
                yDist = math.fabs(y - yMarker)
            else:
                yDist = 0

            if xDist <= self._PICK_OFFSET and yDist <= self._PICK_OFFSET:
                if test(marker):
                    return marker
        return None

    def pickImageOrCurve(self, x, y, test=None):
        if test is None:
            test = lambda item: True

        xPick, yPick = self.pixelToDataCoords(x, y)
        for item in sorted(self._zOrderedItems.values(),
                           key=lambda item: - item.info['zOrder']):
            if test(item):
                if isinstance(item, (GLColormap, GLRGBAImage)):
                    pickedPos = item.pick(xPick, yPick)
                    if pickedPos is not None:
                        return 'image', item, pickedPos

                elif isinstance(item, Curve2D):
                    offset = self._PICK_OFFSET
                    if item.marker is not None:
                        offset = max(item.markerSize / 2., offset)
                    if item.lineStyle is not None:
                        offset = max(item.lineWidth / 2., offset)

                    yAxis = item.info['yaxis']
                    xPick0, yPick0 = self.pixelToDataCoords(x - offset,
                                                            y - offset,
                                                            axis=yAxis)
                    xPick1, yPick1 = self.pixelToDataCoords(x + offset,
                                                            y + offset,
                                                            axis=yAxis)

                    if xPick0 < xPick1:
                        xPickMin, xPickMax = xPick0, xPick1
                    else:
                        xPickMin, xPickMax = xPick1, xPick0

                    if yPick0 < yPick1:
                        yPickMin, yPickMax = yPick0, yPick1
                    else:
                        yPickMin, yPickMax = yPick1, yPick0

                    pickedIndices = item.pick(xPickMin, yPickMin,
                                              xPickMax, yPickMax)
                    if pickedIndices:
                        return 'curve', item, pickedIndices
        return None

    # Manage Plot #

    def setSelectionArea(self, points, fill=None, color=None, name=None):
        """Set a polygon selection area overlaid on the plot.
        Multiple simultaneous areas are supported through the name parameter.

        :param points: The 2D coordinates of the points of the polygon
        :type points: An iterable of (x, y) coordinates
        :param str fill: The fill mode: 'hatch', 'solid' or None (default)
        :param color: RGBA color to use (default: black)
        :type color: list or tuple of 4 float in the range [0, 1]
        :param name: The key associated with this selection area
        """
        if color is None:
            color = (0., 0., 0., 1.)
        self._selectionAreas[name] = Shape2D(points, fill=fill,
                                             fillColor=color,
                                             stroke=True,
                                             strokeColor=color)

    def resetSelectionArea(self, name=None):
        """Remove the name selection area set by setSelectionArea.
        If name is None (the default), it removes all selection areas.

        :param name: The name key provided to setSelectionArea or None
        """
        if name is None:
            self._selectionAreas = MiniOrderedDict()
        elif name in self._selectionAreas:
            del self._selectionAreas[name]

    def updateAxis(self):
        self._axisDirtyFlag = True
        self._plotDirtyFlag = True

    def _axesTicksAndLabels(self):
        trXMin, trXMax = self.plotDataTransformedBounds.xAxis
        trYMin, trYMax = self.plotDataTransformedBounds.yAxis
        trY2Min, trY2Max = self.plotDataTransformedBounds.y2Axis

        vertices = []
        labels = []

        if trXMin != trXMax:
            plotBottom = self.winHeight - self._margins['bottom']

            if self._isXLog:
                xMin, xMax, xStep = niceNumbersForLog10(trXMin, trXMax)

                for xDataLog in _ticks(xMin, xMax, xStep):
                    if xDataLog >= trXMin and xDataLog <= trXMax:
                        xPixel = self.dataToPixelCoords(xData=10 ** xDataLog)

                        vertices.append((xPixel, plotBottom))
                        vertices.append((xPixel, plotBottom - self._tickLen))
                        vertices.append((xPixel, self._margins['top']))
                        vertices.append((xPixel,
                                         self._margins['top'] + self._tickLen))

                        text = '1e{0:+03d}'.format(xDataLog)
                        labels.append(Text2D(text=text,
                                             x=xPixel,
                                             y=plotBottom + self._tickLen,
                                             align=CENTER,
                                             valign=TOP))

            else:  # linear scale
                xMin, xMax, xStep, xNbFrac = niceNumbers(trXMin, trXMax)

                for xData in _ticks(xMin, xMax, xStep):
                    if xData >= trXMin and xData <= trXMax:
                        xPixel = self.dataToPixelCoords(xData=xData)

                        vertices.append((xPixel, plotBottom))
                        vertices.append((xPixel, plotBottom - self._tickLen))
                        vertices.append((xPixel, self._margins['top']))
                        vertices.append((xPixel,
                                         self._margins['top'] + self._tickLen))

                        if xNbFrac == 0:
                            text = '{0:g}'.format(xData)
                        else:
                            text = ('{0:.' + str(xNbFrac) + 'f}').format(xData)

                        labels.append(Text2D(text=text,
                                             x=xPixel,
                                             y=plotBottom + self._tickLen,
                                             align=CENTER,
                                             valign=TOP))

        if trYMin != trYMax:
            plotRight = self.winWidth - self._margins['right']

            if self._isYLog:
                yMin, yMax, yStep = niceNumbersForLog10(trYMin, trYMax)

                for yDataLog in _ticks(yMin, yMax, yStep):
                    if yDataLog >= trYMin and yDataLog <= trYMax:
                        yPixel = self.dataToPixelCoords(yData=10 ** yDataLog)

                        vertices.append((self._margins['left'], yPixel))
                        vertices.append((self._margins['left'] + self._tickLen,
                                         yPixel))
                        vertices.append((plotRight, yPixel))
                        vertices.append((plotRight - self._tickLen, yPixel))

                        text = '1e{0:+03d}'.format(yDataLog)
                        labels.append(Text2D(text=text,
                                             x=self._margins['left'] -
                                             self._tickLen,
                                             y=yPixel,
                                             align=RIGHT,
                                             valign=CENTER))

            else:  # linear scale
                yMin, yMax, yStep, yNbFrac = niceNumbers(trYMin, trYMax)

                for yData in _ticks(yMin, yMax, yStep):
                    if yData >= trYMin and yData <= trYMax:
                        yPixel = self.dataToPixelCoords(yData=yData)

                        vertices.append((self._margins['left'], yPixel))
                        vertices.append((self._margins['left'] + self._tickLen,
                                         yPixel))
                        vertices.append((plotRight, yPixel))
                        vertices.append((plotRight - self._tickLen, yPixel))

                        if yNbFrac == 0:
                            text = '{0:g}'.format(yData)
                        else:
                            text = ('{0:.' + str(yNbFrac) + 'f}').format(yData)

                        labels.append(Text2D(text=text,
                                             x=self._margins['left'] -
                                             self._tickLen,
                                             y=yPixel,
                                             align=RIGHT,
                                             valign=CENTER))

        if self._hasRightYAxis and trY2Min != trY2Max:
            plotRight = self.winWidth - self._margins['right']

            if self._isYLog:
                y2Min, y2Max, y2Step = niceNumbersForLog10(trY2Min, trY2Max)

                for y2DataLog in _ticks(y2Min, y2Max, y2Step):
                    if y2DataLog >= trY2Min and y2DataLog <= trY2Max:
                        y2Pixel = self.dataToPixelCoords(yData=10 ** y2DataLog,
                                                         axis="right")

                        vertices.append((self._margins['left'], y2Pixel))
                        vertices.append((self._margins['left'] + self._tickLen,
                                         y2Pixel))
                        vertices.append((plotRight, y2Pixel))
                        vertices.append((plotRight - self._tickLen, y2Pixel))

                        text = '1e{0:+03d}'.format(y2DataLog)
                        labels.append(Text2D(text=text,
                                             x=plotRight + self._tickLen,
                                             y=y2Pixel,
                                             align=LEFT,
                                             valign=CENTER))

            else:  # linear scale
                y2Min, y2Max, y2Step, y2NbFrac = niceNumbers(trY2Min, trY2Max)

                for y2Data in _ticks(y2Min, y2Max, y2Step):
                    if y2Data >= trY2Min and y2Data <= trY2Max:
                        y2Pixel = self.dataToPixelCoords(yData=y2Data,
                                                         axis="right")

                        vertices.append((self._margins['left'], y2Pixel))
                        vertices.append((self._margins['left'] + self._tickLen,
                                         y2Pixel))
                        vertices.append((plotRight, y2Pixel))
                        vertices.append((plotRight - self._tickLen, y2Pixel))

                        if y2NbFrac == 0:
                            text = '{0:g}'.format(y2Data)
                        else:
                            text = ('%.' + str(y2NbFrac) + 'f') % y2Data

                        labels.append(Text2D(text=text,
                                             x=plotRight + self._tickLen,
                                             y=y2Pixel,
                                             align=LEFT,
                                             valign=CENTER))
        nbMainTicks = len(vertices) // 4

        if trXMin != trXMax and self._isXLog and xStep == 1:
            for xDataLog in list(_ticks(xMin, xMax, xStep))[:-1]:
                xDataOrig = 10 ** xDataLog
                for index in range(2, 10):
                    plotXMin, plotXMax = self.plotDataBounds.xAxis
                    xData = xDataOrig * index
                    if xData >= plotXMin and xData <= plotXMax:
                        xPixel = self.dataToPixelCoords(xData=xData)

                        vertices.append((xPixel, plotBottom))
                        vertices.append((xPixel,
                                         plotBottom - 0.5 * self._tickLen))
                        vertices.append((xPixel, self._margins['top']))
                        vertices.append(
                            (xPixel,
                             self._margins['top'] + 0.5 * self._tickLen))

        if trYMin != trYMax and self._isYLog and yStep == 1:
            for yDataLog in list(_ticks(yMin, yMax, yStep))[:-1]:
                yDataOrig = 10 ** yDataLog
                for index in range(2, 10):
                    plotYMin, plotYMax = self.plotDataBounds.yAxis
                    yData = yDataOrig * index
                    if yData >= plotYMin and yData <= plotYMax:
                        yPixel = self.dataToPixelCoords(yData=yData)

                        vertices.append((self._margins['left'], yPixel))
                        vertices.append((self._margins['left'] +
                                         0.5 * self._tickLen, yPixel))
                        vertices.append((plotRight, yPixel))
                        vertices.append((plotRight - 0.5 * self._tickLen,
                                         yPixel))

        if (self._hasRightYAxis and trY2Min != trY2Max and
                self._isYLog and y2Step == 1):
            for y2DataLog in list(_ticks(y2Min, y2Max, y2Step))[:-1]:
                y2DataOrig = 10 ** y2DataLog
                for index in range(2, 10):
                    plotY2Min, plotY2Max = self.plotDataBounds.y2Axis
                    y2Data = y2DataOrig * index
                    if y2Data >= plotY2Min and y2Data <= plotY2Max:
                        y2Pixel = self.dataToPixelCoords(yData=y2Data,
                                                         axis="right")

                        vertices.append((self._margins['left'], y2Pixel))
                        vertices.append((self._margins['left'] +
                                         0.5 * self._tickLen, y2Pixel))
                        vertices.append((plotRight, y2Pixel))
                        vertices.append((plotRight - 0.5 * self._tickLen,
                                         y2Pixel))

        return vertices, labels, nbMainTicks

    def _updateAxis(self):
        # Check if window is large enough
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        if not self._axisDirtyFlag:
            return
        else:
            self._axisDirtyFlag = False

        # Ticks
        vertices, tickLabels, nbMainTicks = self._axesTicksAndLabels()
        self._labels = tickLabels

        # Plot frame
        xLeft = self._margins['left']
        xRight = self.winWidth - self._margins['right']
        yBottom = self.winHeight - self._margins['bottom']
        yTop = self._margins['top']

        vertices.append((xLeft, yBottom))
        vertices.append((xLeft, yTop))

        vertices.append((xLeft, yTop))
        vertices.append((xRight, yTop))

        vertices.append((xRight, yTop))
        vertices.append((xRight, yBottom))

        vertices.append((xRight, yBottom))
        vertices.append((xLeft, yBottom))

        # Build numpy array from ticks and frame vertices
        self._frameVertices = np.array(vertices, dtype=np.float32)
        self._frameVerticesNbMainTicks = nbMainTicks

        # Title, Labels
        plotCenterX = self._margins['left'] + plotWidth // 2
        plotCenterY = self._margins['top'] + plotHeight // 2
        if self._title:
            self._labels.append(Text2D(self._title,
                                       x=plotCenterX,
                                       y=self._margins['top'] - self._tickLen,
                                       align=CENTER,
                                       valign=BOTTOM))
        if self._xLabel:
            self._labels.append(Text2D(self._xLabel,
                                       x=plotCenterX,
                                       y=self.winHeight -
                                       self._margins['bottom'] // 2,
                                       align=CENTER,
                                       valign=TOP))

        if self._yLabel:
            self._labels.append(Text2D(self._yLabel,
                                       x=self._margins['left'] // 4,
                                       y=plotCenterY,
                                       align=CENTER,
                                       valign=CENTER,
                                       rotate=ROTATE_270))

    # Coordinate systems #

    @property
    def dataBounds(self):
        """Bounds of the currently loaded data
        Not including markers (TODO check consistency with MPLBackend)

        :type: Bounds
        """
        try:
            return self._dataBounds
        except AttributeError:
            xMin, xMax = float('inf'), -float('inf')
            yMin, yMax = float('inf'), -float('inf')
            y2Min, y2Max = float('inf'), -float('inf')
            for item in self._zOrderedItems.values():
                if item.xMin < xMin:
                    xMin = item.xMin
                if item.xMax > xMax:
                    xMax = item.xMax

                if item.info.get('yaxis') == 'right':
                    if item.yMin < y2Min:
                        y2Min = item.yMin
                    if item.yMax > y2Max:
                        y2Max = item.yMax
                else:
                    if item.yMin < yMin:
                        yMin = item.yMin
                    if item.yMax > yMax:
                        yMax = item.yMax

            if xMin >= xMax:
                xMin, xMax = 1., 100.
            if yMin >= yMax:
                yMin, yMax = 1., 100.
            if y2Min >= y2Max:
                y2Min, y2Max = 1., 100.

            self._dataBounds = Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)
            return self._dataBounds

    def _dirtyDataBounds(self):
        if hasattr(self, '_dataBounds'):
            del self._dataBounds

    @property
    def plotDataBounds(self):
        """Bounds of the displayed area in data coordinates

        :type: Bounds
        """
        return self._plotDataBounds

    @plotDataBounds.setter
    def plotDataBounds(self, bounds):
        if bounds != self.plotDataBounds:
            if self._isXLog and bounds.xAxis.min_ <= 0.:
                raise RuntimeError(
                    'Cannot use plot area with X <= 0 with X axis log scale')
            if self._isYLog and (bounds.yAxis.min_ <= 0. or
                                 bounds.y2Axis.min_ <= 0.):
                raise RuntimeError(
                    'Cannot use plot area with Y <= 0 with Y axis log scale')
            self._plotDataBounds = bounds
            self._dirtyPlotDataTransformedBounds()

    @property
    def plotDataTransformedBounds(self):
        """Bounds of the displayed area in transformed data coordinates
        (i.e., log scale applied if any)

        :type: Bounds
        """
        try:
            return self._plotDataTransformedBounds
        except AttributeError:
            if not self._isXLog and not self._isYLog:
                self._plotDataTransformedBounds = self.plotDataBounds
            else:
                xMin, xMax = self.plotDataBounds.xAxis
                yMin, yMax = self.plotDataBounds.yAxis
                y2Min, y2Max = self.plotDataBounds.y2Axis

                if self._isXLog:
                    try:
                        xMin = math.log10(xMin)
                    except ValueError:
                        print('xMin: warning log10({0})'.format(xMin))
                        xMin = 0.
                    try:
                        xMax = math.log10(xMax)
                    except ValueError:
                        print('xMax: warning log10({0})'.format(xMax))
                        xMax = 0.

                if self._isYLog:
                    try:
                        yMin = math.log10(yMin)
                    except ValueError:
                        print('yMin: warning log10({0})'.format(yMin))
                        yMin = 0.
                    try:
                        yMax = math.log10(yMax)
                    except ValueError:
                        print('yMax: warning log10({0})'.format(yMax))
                        yMax = 0.

                    try:
                        y2Min = math.log10(y2Min)
                    except ValueError:
                        print('yMin: warning log10({0})'.format(y2Min))
                        y2Min = 0.
                    try:
                        y2Max = math.log10(y2Max)
                    except ValueError:
                        print('yMax: warning log10({0})'.format(y2Max))
                        y2Max = 0.

                self._plotDataTransformedBounds = \
                    Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)

            return self._plotDataTransformedBounds

    def _dirtyPlotDataTransformedBounds(self):
        if hasattr(self, '_plotDataTransformedBounds'):
            del self._plotDataTransformedBounds
        self._dirtyMatrixPlotDataTransformedProj()

    @property
    def matrixPlotDataTransformedProj(self):
        """Orthographic projection matrix for rendering transformed data

        :type: numpy.matrix
        """
        try:
            return self._matrixPlotDataTransformedProj
        except AttributeError:
            xMin, xMax = self.plotDataTransformedBounds.xAxis
            yMin, yMax = self.plotDataTransformedBounds.yAxis

            if self._isYInverted:
                self._matrixPlotDataTransformedProj = mat4Ortho(xMin, xMax,
                                                                yMax, yMin,
                                                                1, -1)
            else:
                self._matrixPlotDataTransformedProj = mat4Ortho(xMin, xMax,
                                                                yMin, yMax,
                                                                1, -1)
            return self._matrixPlotDataTransformedProj

    @property
    def matrixY2PlotDataTransformedProj(self):
        """Orthographic projection matrix for rendering transformed data
        for the 2nd Y axis

        :type: numpy.matrix
        """
        try:
            return self._matrixY2PlotDataTransformedProj
        except AttributeError:
            xMin, xMax = self.plotDataTransformedBounds.xAxis
            y2Min, y2Max = self.plotDataTransformedBounds.y2Axis

            if self._isYInverted:
                self._matrixY2PlotDataTransformedProj = mat4Ortho(xMin, xMax,
                                                                  y2Max, y2Min,
                                                                  1, -1)
            else:
                self._matrixY2PlotDataTransformedProj = mat4Ortho(xMin, xMax,
                                                                  y2Min, y2Max,
                                                                  1, -1)
            return self._matrixY2PlotDataTransformedProj

    def _dirtyMatrixPlotDataTransformedProj(self):
        if hasattr(self, '_matrixPlotDataTransformedProj'):
            del self._matrixPlotDataTransformedProj
        if hasattr(self, '_matrixY2PlotDataTransformedProj'):
            del self._matrixY2PlotDataTransformedProj

    def dataToPixelCoords(self, xData=None, yData=None, axis="left"):
        assert axis in ("left", "right")

        plotWidth, plotHeight = self.plotSizeInPixels()

        trBounds = self.plotDataTransformedBounds

        if xData is None:
            xPixel = None
        else:
            if self._isXLog:
                if xData > 0.:
                    xData = math.log10(xData)
                else:
                    print('xData: warning log10({0})'.format(xData))
                    xData = 0.
            xPixel = int(self._margins['left'] +
                         plotWidth * (xData - trBounds.xAxis.min_) /
                         trBounds.xAxis.range_)

        if yData is None:
            yPixel = None
        else:
            if self._isYLog:
                if yData > 0.:
                    yData = math.log10(yData)
                else:
                    print('yData: warning log10({0})'.format(yData))
                    yData = 0.
            usedAxis = trBounds.yAxis if axis == "left" else trBounds.y2Axis
            yOffset = plotHeight * (yData - usedAxis.min_) / usedAxis.range_

            if self._isYInverted:
                yPixel = int(self._margins['top'] + yOffset)
            else:
                yPixel = int(self.winHeight - self._margins['bottom'] -
                             yOffset)

        if xData is None:
            return yPixel
        elif yData is None:
            return xPixel
        else:
            return xPixel, yPixel

    def pixelToDataCoords(self, xPixel=None, yPixel=None, axis="left"):
        assert axis in ("left", "right")

        plotWidth, plotHeight = self.plotSizeInPixels()

        trBounds = self.plotDataTransformedBounds

        if xPixel is not None:
            if xPixel < self._margins['left'] or \
               xPixel > (self.winWidth - self._margins['right']):
                xData = None
            else:
                xData = (xPixel - self._margins['left']) + 0.5
                xData /= float(plotWidth)
                xData = trBounds.xAxis.min_ + xData * trBounds.xAxis.range_
                if self._isXLog:
                    xData = pow(10, xData)

        if yPixel is not None:
            usedAxis = trBounds.yAxis if axis == "left" else trBounds.y2Axis
            if yPixel < self._margins['top'] or \
               yPixel > self.winHeight - self._margins['bottom']:
                yData = None
            elif self._isYInverted:
                yData = yPixel - self._margins['top'] + 0.5
                yData /= float(plotHeight)
                yData = usedAxis.min_ + yData * usedAxis.range_
                if self._isYLog:
                    yData = pow(10, yData)
            else:
                yData = self.winHeight - self._margins['bottom'] - yPixel - 0.5
                yData /= float(plotHeight)
                yData = usedAxis.min_ + yData * usedAxis.range_
                if self._isYLog:
                    yData = pow(10, yData)

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
        testGLExtensions()

        glClearColor(1., 1., 1., 1.)
        glClearStencil(0)

        glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE)

        # For lines
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # For points
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        glEnable(GL_POINT_SPRITE)  # OpenGL 2
        # glEnable(GL_PROGRAM_POINT_SIZE)

        # Create basic program
        self._progBase = Program(_baseVertShd, _baseFragShd)

        # Create texture program
        self._progTex = Program(_texVertShd, _texFragShd)

    def _paintDirectGL(self):
        self._renderPlotAreaGL()
        self._renderPlotFrameGL()
        self._renderMarkersGL()
        self._renderSelectionGL()

    def _paintFBOGL(self):
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
                self._renderPlotAreaGL()
                self._renderPlotFrameGL()

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

        self._renderMarkersGL()
        self._renderSelectionGL()

    def paintGL(self):
        # Release OpenGL resources
        for item in self._glGarbageCollector:
            item.discard()
        self._glGarbageCollector = []

        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        # Check if window is large enough
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        # self._paintDirectGL()
        self._paintFBOGL()

        self._deferredSaveGraphGL()

    def _renderMarkersGL(self):
        if len(self._markers) == 0:
            return

        plotWidth, plotHeight = self.plotSizeInPixels()

        # Render in plot area
        glScissor(self._margins['left'], self._margins['bottom'],
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

        glViewport(self._margins['left'], self._margins['bottom'],
                   plotWidth, plotHeight)

        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self.matrixPlotDataTransformedProj)
        glUniform2i(self._progBase.uniforms['isLog'],
                    self._isXLog, self._isYLog)
        glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)
        posAttrib = self._progBase.attributes['position']
        glEnableVertexAttribArray(posAttrib)

        labels = []
        pixelOffset = 2

        for marker in self._markers.values():
            xCoord, yCoord = marker['x'], marker['y']

            if marker['label'] is not None:
                if xCoord is None:
                    x = self.winWidth - self._margins['right'] - pixelOffset
                    y = self.dataToPixelCoords(yData=yCoord) - pixelOffset
                    label = Text2D(marker['label'], x, y, marker['color'],
                                   align=RIGHT, valign=BOTTOM)
                elif yCoord is None:
                    x = self.dataToPixelCoords(xData=xCoord) + pixelOffset
                    y = self._margins['top'] + pixelOffset
                    label = Text2D(marker['label'], x, y, marker['color'],
                                   align=LEFT, valign=TOP)
                else:
                    x, y = self.dataToPixelCoords(xCoord, yCoord)
                    x, y = x + pixelOffset, y + pixelOffset
                    label = Text2D(marker['label'], x, y, marker['color'],
                                   align=LEFT, valign=TOP)
                labels.append(label)

            glUniform4f(self._progBase.uniforms['color'], * marker['color'])

            xMin, xMax = self.plotDataBounds.xAxis
            yMin, yMax = self.plotDataBounds.yAxis

            if xCoord is None:
                vertices = np.array(((xMin, yCoord),
                                     (xMax, yCoord)),
                                    dtype=np.float32)
            elif yCoord is None:
                vertices = np.array(((xCoord, yMin),
                                    (xCoord, yMax)),
                                    dtype=np.float32)
            else:
                xPixel, yPixel = self.dataToPixelCoords(xCoord, yCoord)
                x0, y0 = self.pixelToDataCoords(xPixel - 2 * pixelOffset,
                                                yPixel - 2 * pixelOffset)
                x1, y1 = self.pixelToDataCoords(xPixel + 2 * pixelOffset + 1.,
                                                yPixel + 2 * pixelOffset + 1.)

                vertices = np.array(((x0, yCoord), (x1, yCoord),
                                     (xCoord, y0), (xCoord, y1)),
                                    dtype=np.float32)
            glVertexAttribPointer(posAttrib,
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0, vertices)
            glLineWidth(1)
            glDrawArrays(GL_LINES, 0, len(vertices))

        glViewport(0, 0, self.winWidth, self.winHeight)

        # Render marker labels
        for label in labels:
            label.render(self.matScreenProj)

        glDisable(GL_SCISSOR_TEST)

    def _renderSelectionGL(self):
        # Render selection area
        if self._selectionAreas:
            plotWidth, plotHeight = self.plotSizeInPixels()

            # Render in plot area
            glScissor(self._margins['left'], self._margins['bottom'],
                      plotWidth, plotHeight)
            glEnable(GL_SCISSOR_TEST)

            glViewport(self._margins['left'], self._margins['bottom'],
                       plotWidth, plotHeight)

            # Render fill
            self._progBase.use()
            glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                               self.matrixPlotDataTransformedProj)
            glUniform2i(self._progBase.uniforms['isLog'],
                        self._isXLog, self._isYLog)
            glUniform1f(self._progBase.uniforms['tickLen'], 0.)
            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            for shape in self._selectionAreas.values():
                shape.render(posAttrib, colorUnif, hatchStepUnif)

            glDisable(GL_SCISSOR_TEST)

    def _renderPlotFrameGL(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        # Render plot in screen coords
        glViewport(0, 0, self.winWidth, self.winHeight)

        self._updateAxis()

        # Render Plot frame
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self.matScreenProj)
        glUniform2i(self._progBase.uniforms['isLog'], 0, 0)
        glUniform4f(self._progBase.uniforms['color'], 0., 0., 0., 1.)
        glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)

        glVertexAttribPointer(self._progBase.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self._frameVertices)
        glLineWidth(self._lineWidth)
        glDrawArrays(GL_LINES, 0, len(self._frameVertices))

        # Render Text
        for label in self._labels:
            label.render(self.matScreenProj)

    def _renderPlotAreaGL(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        glScissor(self._margins['left'], self._margins['bottom'],
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

        # Render grid by reusing tick vertices and a stride
        if self._grid:
            self._updateAxis()

            # Render plot in screen coords
            glViewport(0, 0, self.winWidth, self.winHeight)
            self._progBase.use()
            glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                               self.matScreenProj)
            glUniform2i(self._progBase.uniforms['isLog'], 0, 0)
            glUniform4f(self._progBase.uniforms['color'], 0.5, 0.5, 0.5, 1.)
            glUniform1i(self._progBase.uniforms['hatchStep'], 0)
            glUniform1f(self._progBase.uniforms['tickLen'], 2.)

            stride = 2 * self._frameVertices.shape[-1] * \
                self._frameVertices.itemsize
            glEnableVertexAttribArray(self._progBase.attributes['position'])
            glVertexAttribPointer(self._progBase.attributes['position'],
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  stride, self._frameVertices)
            glLineWidth(self._lineWidth)

            if self._grid == 1:
                firstVertex = 0
                nbVertices = self._frameVerticesNbMainTicks * 2
            elif self._grid == 2:
                firstVertex = self._frameVerticesNbMainTicks * 2
                nbVertices = (len(self._frameVertices) - 8) // 2 - firstVertex
            else:
                firstVertex = 0
                nbVertices = (len(self._frameVertices) - 8) // 2

            glDrawArrays(GL_LINES, firstVertex, nbVertices)

        # Matrix
        trBounds = self.plotDataTransformedBounds
        if trBounds.xAxis.min_ == trBounds.xAxis.max_ or \
           trBounds.yAxis.min_ == trBounds.yAxis.max_:
            return

        glViewport(self._margins['left'], self._margins['bottom'],
                   plotWidth, plotHeight)

        # Render images and curves
        # sorted is stable: original order is preserved when key is the same
        for item in sorted(self._zOrderedItems.values(),
                           key=lambda item: item.info['zOrder']):
            if item.info.get('yaxis') == 'right':
                item.render(self.matrixY2PlotDataTransformedProj,
                            self._isXLog, self._isYLog)
            else:
                item.render(self.matrixPlotDataTransformedProj,
                            self._isXLog, self._isYLog)

        # Render Items
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self.matrixPlotDataTransformedProj)
        glUniform2i(self._progBase.uniforms['isLog'],
                    self._isXLog, self._isYLog)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)

        for item in self._items.values():
            try:
                shape2D = item['_shape2D']
            except KeyError:
                shape2D = Shape2D(tuple(zip(item['x'], item['y'])),
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

        xMin, xMax = self.plotDataBounds.xAxis
        yMin, yMax = self.plotDataBounds.yAxis
        y2Min, y2Max = self.plotDataBounds.y2Axis
        self.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)

        self.updateAxis()
        self.replot()

    # PlotBackend API #

    def insertMarker(self, x, y, legend=None, text=None, color='k',
                     selectable=False, draggable=False,
                     **kwargs):
        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        if x is not None and self._isXLog and x <= 0.:
            raise RuntimeError(
                'Cannot add marker with X <= 0 with X axis log scale')
        if y is not None and self._isYLog and y <= 0.:
            raise RuntimeError(
                'Cannot add marker with Y <= 0 with Y axis log scale')

        self._markers[legend] = {
            'x': x,
            'y': y,
            'label': text,
            'color': rgba(color, PlotBackend.COLORDICT),
            'behaviors': behaviors,
        }

        self._plotDirtyFlag = True

        return legend

    def insertXMarker(self, x, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kwargs):
        return self.insertMarker(x, None, legend, text, color,
                                 selectable, draggable, **kwargs)

    def insertYMarker(self, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kwargs):
        return self.insertMarker(None, y, legend, text, color,
                                 selectable, draggable, **kwargs)

    def removeMarker(self, legend, replot=True):
        try:
            del self._markers[legend]
        except KeyError:
            pass
        else:
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearMarkers(self):
        self._markers = MiniOrderedDict()
        self._plotDirtyFlag = True

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=0,
                 selectable=False, draggable=False,
                 colormap=None, **kwargs):
        if info is not None:
            warnings.warn("Ignore info parameter of addImage",
                          RuntimeWarning)

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        oldImage = self._zOrderedItems.get(('image', legend), None)
        if oldImage is not None:
            if oldImage.data.shape == data.shape:
                oldXScale = oldImage.xMin, oldImage.xScale
                oldYScale = oldImage.yMin, oldImage.yScale
            else:
                oldImage = None
                self.removeImage(legend)

        if replace:
            self.clearImages()

        if xScale is None:
            xScale = (0, 1)
        if yScale is None:
            yScale = (0, 1)

        if len(data.shape) == 2:
            if colormap is None:
                colormap = self.getDefaultColormap()

            if colormap['normalization'] not in ('linear', 'log'):
                raise NotImplementedError(
                    "Normalisation: {0}".format(colormap['normalization']))
            if colormap['colors'] != 256:
                raise NotImplementedError(
                    "Colors: {0}".format(colormap['colors']))

            colormapIsLog = colormap['normalization'].startswith('log')

            if colormap['autoscale']:
                cmapRange = None
            else:
                cmapRange = colormap['vmin'], colormap['vmax']
                assert cmapRange[0] <= cmapRange[1]

            if oldImage is not None:  # TODO check if benefit
                image = oldImage
                image.xMin = xScale[0]
                image.xScale = xScale[1]
                image.yMin = yScale[0]
                image.yScale = yScale[1]
                image.colormap = colormap['name'][:]
                image.cmapIsLog = colormapIsLog
                image.cmapRange = cmapRange
                image.updateData(data)
            else:
                image = GLColormap(data,
                                   xScale[0], xScale[1],
                                   yScale[0], yScale[1],
                                   colormap['name'][:],
                                   colormapIsLog,
                                   cmapRange)
            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }
            self._zOrderedItems[('image', legend)] = image

        elif len(data.shape) == 3:
            # For RGB, RGBA data
            assert data.shape[2] in (3, 4)
            assert data.dtype == np.uint8 or \
                np.can_cast(data.dtype, np.float32)

            if oldImage is not None:
                image = oldImage
                image.xMin = xScale[0]
                image.xScale = xScale[1]
                image.yMin = yScale[0]
                image.yScale = yScale[1]
                image.updateData(data)
            else:
                image = GLRGBAImage(data,
                                    xScale[0], xScale[1],
                                    yScale[0], yScale[1])

            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }

            if self._isXLog and image.xMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with X <= 0 with X axis log scale')
            if self._isYLog and image.yMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with Y <= 0 with Y axis log scale')

            self._zOrderedItems[('image', legend)] = image

        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        if oldImage is None or \
           oldXScale != xScale or \
           oldYScale != yScale:

            self._dirtyDataBounds()
            self.setLimits(self.dataBounds.xAxis.min_,
                           self.dataBounds.xAxis.max_,
                           self.dataBounds.yAxis.min_,
                           self.dataBounds.yAxis.max_,
                           self.dataBounds.y2Axis.min_,
                           self.dataBounds.y2Axis.max_)

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend  # This is the 'handle'

    def removeImage(self, legend, replot=True):
        try:
            image = self._zOrderedItems.pop(('image', legend))
        except KeyError:
            pass
        else:
            self._glGarbageCollector.append(image)
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearImages(self):
        # Copy keys as it removes items from the dict
        for type_, legend in list(self._zOrderedItems.keys()):
            if type_ == 'image':
                self.removeImage(legend, replot=False)
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
            xList = np.array((xMin, xMin, xMax, xMax))
            yMin, yMax = yList
            yList = np.array((yMin, yMax, yMax, yMin))
        else:
            xList = np.array(xList, copy=False)
            yList = np.array(yList, copy=False)

        if self._isXLog and xList.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with X <= 0 with X axis log scale')
        if self._isYLog and yList.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with Y <= 0 with Y axis log scale')

        self._items[legend] = {
            'shape': shape,
            'color': rgba(colorCode, PlotBackend.COLORDICT),
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
                 replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=1, selectable=True,
                 fill=None, **kw):
        if xerror is not None:
            warnings.warn("Ignore xerror parameter of addCurve",
                          RuntimeWarning)
        if yerror is not None:
            warnings.warn("Ignore yerror parameter of addCurve",
                          RuntimeWarning)

        x = np.array(x, dtype=np.float32, copy=False, order='C')
        y = np.array(y, dtype=np.float32, copy=False, order='C')

        behaviors = set()
        if selectable:
            behaviors.add('selectable')

        oldCurve = self._zOrderedItems.get(('curve', legend), None)
        if oldCurve is not None:
            self.removeCurve(legend)

        if replace:
            self.clearCurves()

        if color is None:
            color = self._activeCurveColor

        if isinstance(color, np.ndarray) and len(color) > 4:
            colorArray = color
            color = None
        else:
            colorArray = None
            color = rgba(color, PlotBackend.COLORDICT)

        if fill is None and info is not None:  # To make it run with Plot.py
            fill = info.get('plot_fill', False)

        curve = Curve2D(x, y, colorArray,
                        lineStyle=linestyle,
                        lineColor=color,
                        lineWidth=1,
                        marker=symbol,
                        markerColor=color,
                        fillColor=color if fill else None)
        curve.info = {
            'legend': legend,
            'zOrder': z,
            'behaviors': behaviors,
            'xLabel': xlabel,
            'yLabel': ylabel,
            'yaxis': 'left' if yaxis is None else yaxis,
        }

        if yaxis == "right":
            self._hasRightYAxis.add(curve)

        if self._isXLog and curve.xMin <= 0.:
            raise RuntimeError(
                'Cannot add curve with X <= 0 with X axis log scale')
        if self._isYLog and curve.yMin <= 0.:
            raise RuntimeError(
                'Cannot add curve with Y <= 0 with Y axis log scale')

        self._zOrderedItems[('curve', legend)] = curve

        if oldCurve is None or \
           oldCurve.xMin != curve.xMin or oldCurve.xMax != curve.xMax or \
           oldCurve.info['axis'] != curve.info['axis'] or \
           oldCurve.yMin != curve.yMin or oldCurve.yMax != curve.yMax:
            self._dirtyDataBounds()
            self.setLimits(self.dataBounds.xAxis.min_,
                           self.dataBounds.xAxis.max_,
                           self.dataBounds.yAxis.min_,
                           self.dataBounds.yAxis.max_,
                           self.dataBounds.y2Axis.min_,
                           self.dataBounds.y2Axis.max_)

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend

    def removeCurve(self, legend, replot=True):
        try:
            curve = self._zOrderedItems.pop(('curve', legend))
        except KeyError:
            pass
        else:
            self._hasRightYAxis.discard(curve)
            if not self._hasRightYAxis:
                self.updateAxis()
            self._glGarbageCollector.append(curve)
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearCurves(self):
        # Copy keys as dict is changed
        for type_, legend in list(self._zOrderedItems.keys()):
            if type_ == 'curve':
                self.removeCurve(legend, replot=False)
        self._plotDirtyFlag = True

    def setActiveCurve(self, legend, replot=True):
        if not self._activeCurveHandling:
            return

        curve = self._zOrderedItems.get(('curve', legend), None)
        if curve is None:
            raise KeyError("Curve %s not found" % legend)

        if curve.info['yaxis'] == "right":
            warnings.warn("Ignore setActiveCurve for curve on right Y axis",
                          RuntimeWarning)
            return

        if self._activeCurve is not None:
            inactiveState = self._activeCurve._inactiveState
            del self._activeCurve._inactiveState
            self._activeCurve.lineColor = inactiveState['lineColor']
            self._activeCurve.markerColor = inactiveState['markerColor']
            self._activeCurve.useColorVboData = inactiveState['useColorVbo']
            self.setGraphXLabel(inactiveState['xLabel'])
            self.setGraphYLabel(inactiveState['yLabel'])

        curve._inactiveState = {'lineColor': curve.lineColor,
                                'markerColor': curve.markerColor,
                                'useColorVbo': curve.useColorVboData,
                                'xLabel': self.getGraphXLabel(),
                                'yLabel': self.getGraphYLabel()}

        if curve.info['xLabel'] is not None:
            self.setGraphXLabel(curve.info['xLabel'])
        if curve.info['yLabel'] is not None:
            self.setGraphYLabel(curve.info['yLabel'])

        color = rgba(self._activeCurveColor, PlotBackend.COLORDICT)
        curve.lineColor = color
        curve.markerColor = color
        curve.useColorVboData = False
        self._activeCurve = curve

        if replot:
            self.replot()

    def clear(self):
        self.clearItems()
        self.clearCurves()

    def replot(self):
        self.postRedisplay()

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

    def setDrawModeEnabled(self, flag=True, shape="polygon", label=None,
                           color=None, **kw):
        eventHandlerClass = self._drawModes[shape]
        if flag:
            parameters = kw
            parameters['shape'] = shape
            parameters['label'] = label
            if color is not None:
                parameters['color'] = rgba(color, PlotBackend.COLORDICT)

            if not isinstance(self.eventHandler, eventHandlerClass):
                self.eventHandler.cancel()
                self.eventHandler = eventHandlerClass(self, parameters)
        elif isinstance(self.eventHandler, eventHandlerClass):
            self.eventHandler.cancel()
            self.eventHandler = ItemsInteraction(self)

    def getDrawMode(self):
        if self.isDrawModeEnabled():
            return self.eventHandler.parameters
        else:
            None

    # Zoom #

    def isZoomModeEnabled(self):
        return isinstance(self.eventHandler, ZoomAndSelect)

    def setZoomModeEnabled(self, flag=True, color=None):
        if flag:
            if color is not None:
                self._zoomColor = rgba(color, PlotBackend.COLORDICT)
            elif self._zoomColor is None:
                self._zoomColor = 0., 0., 0., 1.

            self.eventHandler.cancel()
            self.eventHandler = ZoomAndSelect(self, self._zoomColor)

        elif isinstance(self.eventHandler, ZoomAndSelect):
            self.eventHandler.cancel()
            self.eventHandler = ItemsInteraction(self)

    def resetZoom(self):
        if self.isXAxisAutoScale() and self.isYAxisAutoScale():
            self.setLimits(self.dataBounds.xAxis.min_,
                           self.dataBounds.xAxis.max_,
                           self.dataBounds.yAxis.min_,
                           self.dataBounds.yAxis.max_,
                           self.dataBounds.y2Axis.min_,
                           self.dataBounds.y2Axis.max_)

        elif self.isXAxisAutoScale():
            self.setGraphXLimits(self.dataBounds.xAxis.min_,
                                 self.dataBounds.xAxis.max_)

        elif self.isYAxisAutoScale():
            xMin, xMax = self.getGraphXLimits()
            self.setLimits(xMin, xMax,
                           self.dataBounds.yAxis.min_,
                           self.dataBounds.yAxis.max_,
                           self.dataBounds.y2Axis.min_,
                           self.dataBounds.y2Axis.max_)

        self.replot()

    # Limits #

    def _ensureAspectRatio(self):
        plotWidth, plotHeight = self.plotSizeInPixels()
        if plotWidth <= 2 or plotHeight <= 2:
            return

        plotRatio = plotWidth / float(plotHeight)

        dataW = self.plotDataBounds.xAxis.range_
        dataH = self.plotDataBounds.yAxis.range_
        if dataH == 0.:
            return

        dataRatio = dataW / float(dataH)

        xMin, xMax = self.plotDataBounds.xAxis
        yMin, yMax = self.plotDataBounds.yAxis
        y2Min, y2Max = self.plotDataBounds.y2Axis

        if dataRatio < plotRatio:
            dataW = dataH * plotRatio
            xCenter = self.plotDataBounds.xAxis.center
            xMin = xCenter - 0.5 * dataW
            xMax = xCenter + 0.5 * dataW

        else:
            dataH = dataW / plotRatio
            yCenter = self.plotDataBounds.yAxis.center
            yMin = yCenter - 0.5 * dataH
            yMax = yCenter + 0.5 * dataH

        self.plotDataBounds = Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)

    def isKeepDataAspectRatio(self):
        if self._isXLog or self._isYLog:
            return False
        else:
            return self._keepDataAspectRatio

    def keepDataAspectRatio(self, flag=True):
        if flag and (self._isXLog or self._isYLog):
            warnings.warn("KeepDataAspectRatio is ignored with log axes",
                          RuntimeWarning)

        self._keepDataAspectRatio = flag

        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio()

        self.resetZoom()
        self.updateAxis()
        self.replot()

    def getGraphXLimits(self):
        return self.plotDataBounds.xAxis.min_, self.plotDataBounds.xAxis.max_

    def setGraphXLimits(self, xMin, xMax):
        yMin, yMax = self.plotDataBounds.yAxis
        y2Min, y2Max = self.plotDataBounds.y2Axis
        self.plotDataBounds = Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)

        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio()

        self.updateAxis()

    def getGraphYLimits(self, axis="left"):
        assert axis in ("left", "right")
        if axis == "left":
            return (self.plotDataBounds.yAxis.min_,
                    self.plotDataBounds.yAxis.max_)
        else:
            return (self.plotDataBounds.y2Axis.min_,
                    self.plotDataBounds.y2Axis.max_)

    def setGraphYLimits(self, yMin, yMax, axis="left"):
        assert axis in ("left", "right")

        if axis == "left":
            y2Min, y2Max = self.plotDataBounds.y2Axis
        else:
            y2Min, y2Max = yMin, yMax
            yMin, yMax = self.plotDataBounds.yAxis

        xMin, xMax = self.plotDataBounds.xAxis
        self.plotDataBounds = Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)

        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio()

        self.updateAxis()

    def setLimits(self, xMin, xMax, yMin, yMax, y2Min=None, y2Max=None):
        if y2Min is None or y2Max is None:
            y2Min, y2Max = self.plotDataBounds.y2Axis
            if y2Min is None:
                y2Min = 1.
            if y2Max is None:
                y2Max = 100.
        self.plotDataBounds = Bounds(xMin, xMax, yMin, yMax, y2Min, y2Max)

        if self.isKeepDataAspectRatio():
            self._ensureAspectRatio()

        self.updateAxis()

    def invertYAxis(self, flag=True):
        if flag != self._isYInverted:
            self._isYInverted = flag
            self._dirtyMatrixPlotDataTransformedProj()
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

    # Log axis #

    def setXAxisLogarithmic(self, flag=True):
        if flag != self._isXLog:
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            if flag and self.dataBounds.xAxis.min_ <= 0.:
                raise RuntimeError(
                    'Cannot use log scale for X axis: Some data is <= 0.')
            self._isXLog = flag
            self._dirtyPlotDataTransformedBounds()

    def setYAxisLogarithmic(self, flag=True):
        if flag != self._isYLog:
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            if flag and (self.dataBounds.yAxis.min_ <= 0. or
                         self.dataBounds.y2Axis.min_ <= 0.):
                raise RuntimeError(
                    'Cannot use log scale for Y axis: Some data is <= 0.')
            self._isYLog = flag
            self._dirtyPlotDataTransformedBounds()

    def isXAxisLogaritmic(self):
        return self._isXLog

    def isYAxisLogaritmic(self):
        return self._isYLog

    # Title, Labels
    def setGraphTitle(self, title=""):
        self._title = title

    def getGraphTitle(self):
        return self._title

    def setGraphXLabel(self, label="X"):
        self._xLabel = label
        self.updateAxis()

    def getGraphXLabel(self):
        return self._xLabel

    def setGraphYLabel(self, label="Y"):
        self._yLabel = label
        self.updateAxis()

    def getGraphYLabel(self):
        return self._yLabel

    def showGrid(self, flag=True):
        self._grid = flag
        self._plotDirtyFlag = True
        self.replot()

    # Save
    def saveGraph(self, fileName, fileFormat='svg', dpi=None, **kw):
        """Save the graph as an image to a file.

        WARNING: The file will be created asynchronously.
        This method is thread-safe if :func:`postRedisplay` is thread-safe.
        """
        if dpi is not None:
            warnings.warn("saveGraph ignores dpi parameter",
                          RuntimeWarning)
        if kw:
            warnings.warn("saveGraph ignores additional parameters",
                          RuntimeWarning)

        if fileFormat not in ['png', 'ppm', 'svg', 'tiff']:
            raise NotImplementedError('Unsupported format: %s' % fileFormat)

        self._graphsToSave.put_nowait((fileName, fileFormat))
        self.replot()

    def _deferredSaveGraphGL(self):
        """This method MUST be called with an active OpenGL context.
        """
        if not self._graphsToSave.empty():
            data = np.empty((self.winHeight, self.winWidth, 3),
                            dtype=np.uint8, order='C')

            glPixelStorei(GL_PACK_ALIGNMENT, 1)
            glReadPixels(0, 0, self.winWidth, self.winHeight,
                         GL_RGB, GL_UNSIGNED_BYTE, data)

            # glReadPixels gives bottom to top,
            # while images are stored as top to bottom
            data = np.flipud(data)

            for i in range(self._graphsToSave.qsize()):
                try:
                    fileName, fileFormat = self._graphsToSave.get_nowait()
                except queue.Full:
                    break
                else:
                    saveImageToFile(data, fileName, fileFormat)


# OpenGLBackend ###############################################################

# Init GL context getter
setGLContextGetter(QGLContext.currentContext)


class OpenGLBackend(QGLWidget, OpenGLPlotCanvas):
    _signalRedisplay = pyqtSignal()  # PyQt binds it to instances

    def __init__(self, parent=None, **kw):
        QGLWidget.__init__(self, parent)
        self._signalRedisplay.connect(self.update)

        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        OpenGLPlotCanvas.__init__(self, parent, **kw)

    def postRedisplay(self):
        """Thread-safe call to QWidget.update."""
        self._signalRedisplay.emit()

    # Mouse events #
    _MOUSE_BTNS = {1: 'left', 2: 'right', 4: 'middle'}

    def sizeHint(self):
        return qt.QSize(8 * 80, 6 * 80)  # Mimic MatplotlibBackend

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

    _CURSORS = {
        CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        CURSOR_POINTING: qt.Qt.PointingHandCursor,
        CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setCursor(self, cursor=CURSOR_DEFAULT):
        cursor = self._CURSORS[cursor]
        super(OpenGLBackend, self).setCursor(qt.QCursor(cursor))

    # Widget

    def getWidgetHandle(self):
        return self


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
