# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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


def saveImageToFile(data, fileNameOrObj, fileFormat):
    """Save a RGB image to a file.

    :param data: A 3D array (h, w, 3) storing an RGB image.
    :type data: numpy.ndarray with of unsigned bytes.
    :param fileNameOrObj: Filename or object to use to write the image.
    :type fileNameOrObj: A str or a 'file-like' object with a 'write' method.
    :param str fileType: The type of the file in: 'png', 'ppm', 'svg', 'tiff'.
    """
    assert len(data.shape) == 3
    assert data.shape[2] == 3
    assert fileFormat in ('png', 'ppm', 'svg', 'tiff')

    if not hasattr(fileNameOrObj, 'write'):
        fileObj = open(fileNameOrObj, 'wb')
    else:  # Use as a file-like object
        fileObj = fileNameOrObj

    if fileFormat == 'svg':
        import base64

        height, width = data.shape[:2]
        base64Data = base64.b64encode(convertRGBDataToPNG(data))

        fileObj.write(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        fileObj.write('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n')
        fileObj.write(
            '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')
        fileObj.write('<svg xmlns:xlink="http://www.w3.org/1999/xlink"\n')
        fileObj.write('     xmlns="http://www.w3.org/2000/svg"\n')
        fileObj.write('     version="1.1"\n')
        fileObj.write('     width="%d"\n' % width)
        fileObj.write('     height="%d">\n' % height)
        fileObj.write('    <image xlink:href="data:image/png;base64,')
        fileObj.write(base64Data.decode('ascii'))
        fileObj.write('"\n')
        fileObj.write('           x="0"\n')
        fileObj.write('           y="0"\n')
        fileObj.write('           width="%d"\n' % width)
        fileObj.write('           height="%d"\n' % height)
        fileObj.write('           id="image" />\n')
        fileObj.write('</svg>')

    elif fileFormat == 'ppm':
        fileObj.write('P6\n')
        fileObj.write('%d %d\n' % (self.winWidth, self.winHeight))
        fileObj.write('255\n')
        fileObj.write(data.tostring())

    elif fileFormat == 'png':
        fileObj.write(convertRGBDataToPNG(data))

    elif fileFormat == 'tiff':
        if fileObj == fileNameOrObj:
            raise NotImplementedError(
                'Save TIFF to a file-like object not implemented')

        from PyMca5.PyMcaIO.TiffIO import TiffIO

        tif = TiffIO(fileNameOrObj, mode='wb+')
        tif.writeImage(data, info={'Title': 'PyMCA GL Snapshot'})

    if fileObj != fileNameOrObj:
        fileObj.close()


# shaders #####################################################################

_baseVertShd = """
    attribute vec2 position;
    uniform mat4 matrix;
    uniform bvec2 isLog;

    const float oneOverLog10 = 0.43429448190325176;

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
        if (tickLen != 0.) {
            if (mod((gl_FragCoord.x + gl_FragCoord.y) / tickLen, 2.) < 1.) {
                gl_FragColor = color;
            } else {
                discard;
            }
        } else if (hatchStep == 0 ||
            mod(gl_FragCoord.x - gl_FragCoord.y, float(hatchStep)) == 0.) {
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
                dataPos = self.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareMouseSignal('mouseClicked', 'left',
                                               dataPos[0], dataPos[1],
                                               x, y)
                self.backend._callback(eventDict)

                self._lastClick = time.time(), (dataPos[0], dataPos[1], x, y)

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
                dataPos = self.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareMouseSignal('mouseClicked', 'right',
                                               dataPos[0], dataPos[1],
                                               x, y)
                self.backend._callback(eventDict)
            else:
                self.backend.setLimits(xMin, xMax, yMin, yMax, y2Min, y2Max)
            self.backend.replot()

    def beginDrag(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None
        self.x0, self.y0 = dataPos

    def drag(self, x1, y1):
        dataPos = self.backend.pixelToData(x1, y1)
        assert dataPos is not None
        x1, y1 = dataPos

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

        dataPos = self.backend.pixelToData(*startPos)
        assert dataPos is not None
        x0, y0 = dataPos

        dataPos = self.backend.pixelToData(y=startPos[1], axis="right")
        assert dataPos is not None
        y2_0 = dataPos[1]

        dataPos = self.backend.pixelToData(*endPos)
        assert dataPos is not None
        x1, y1 = dataPos

        dataPos = self.backend.pixelToData(y=endPos[1], axis="right")
        assert dataPos is not None
        y2_1 = dataPos[1]

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
            if min_ > 0.:
                oldMin = np.log10(min_)
            else:
                # Happens when autoscale is off and switch to log scale
                # while displaying area < 0.
                oldMin = np.log10(np.nextafter(0, 1))

            if center > 0.:
                center = np.log10(center)
            else:
                center = np.log10(np.nextafter(0, 1))

            if max_ > 0.:
                oldMax = np.log10(max_)
            else:
                # Should not happen
                oldMax = 0.
        else:
            oldMin, oldMax = min_, max_

        offset = (center - oldMin) / (oldMax - oldMin)
        range_ = (oldMax - oldMin) / scale
        newMin = center - offset * range_
        newMax = center + (1. - offset) * range_
        if isLog:
            try:
                newMin, newMax = 10. ** float(newMin), 10. ** float(newMax)
            except OverflowError:  # Limit case
                newMin, newMax = min_, max_
            if newMin <= 0. or newMax <= 0.:  # Limit case
                newMin, newMax = min_, max_
        return newMin, newMax

    def _zoom(self, cx, cy, scaleF):
        dataCenterPos = self.backend.pixelToData(cx, cy)
        assert dataCenterPos is not None

        xMin, xMax = self.backend.getGraphXLimits()
        xMin, xMax = self._newZoomRange(xMin, xMax, dataCenterPos[0], scaleF,
                                        self.backend.isXAxisLogarithmic())

        yMin, yMax = self.backend.getGraphYLimits()
        yMin, yMax = self._newZoomRange(yMin, yMax, dataCenterPos[1], scaleF,
                                        self.backend.isYAxisLogarithmic())

        dataPos = self.backend.pixelToData(y=cy, axis="right")
        assert dataPos is not None
        y2Center = dataPos[1]
        y2Min, y2Max = self.backend.getGraphYLimits(axis="right")
        y2Min, y2Max = self._newZoomRange(y2Min, y2Max, y2Center, scaleF,
                                          self.backend.isYAxisLogarithmic())

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
            dataPos = self.machine.backend.pixelToData(x, y)
            assert dataPos is not None
            self.points = [dataPos, dataPos]

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
                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                self.points[-1] = dataPos
                self.updateSelectionArea()
                if self.points[-2] != self.points[-1]:
                    self.points.append(dataPos)
                return True

        def onMove(self, x, y):
            dataPos = self.machine.backend.pixelToData(x, y)
            assert dataPos is not None
            self.points[-1] = dataPos
            self.updateSelectionArea()

        def onPress(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.backend.resetSelectionArea()
                self.machine.backend.replot()

                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                self.points[-1] = dataPos
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
        self.startPt = self.backend.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        self.backend.setSelectionArea((self.startPt,
                                      (self.startPt[0], dataPos[1]),
                                      dataPos,
                                      (dataPos[0], self.startPt[1])),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'rectangle',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend._callback(eventDict)

    def cancelSelect(self):
        self.backend.resetSelectionArea()
        self.backend.replot()


class SelectLine(Select2Points):
    def beginSelect(self, x, y):
        self.startPt = self.backend.pixelToData(x, y)
        assert self.startPt is not None

    def select(self, x, y):
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        self.backend.setSelectionArea((self.startPt, dataPos),
                                      fill='hatch',
                                      color=self.color)
        self.backend.replot()

        eventDict = prepareDrawingSignal('drawingProgress',
                                         'line',
                                         (self.startPt, dataPos),
                                         self.parameters)
        self.backend._callback(eventDict)

    def endSelect(self, x, y):
        self.backend.resetSelectionArea()
        self.backend.replot()

        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None

        eventDict = prepareDrawingSignal('drawingFinished',
                                         'line',
                                         (self.startPt, dataPos),
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
        dataPos = self.backend.pixelToData(y=y)
        assert dataPos is not None

        xMin, xMax = self.backend.getGraphXLimits()
        return (xMin, dataPos[1]), (xMax, dataPos[1])

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
        dataPos = self.backend.pixelToData(x=x)
        assert dataPos is not None

        yMin, yMax = self.backend.getGraphYLimits()
        return (dataPos[0], yMin), (dataPos[0], yMax)

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
        def __init__(self, *args, **kw):
            super(ItemsInteraction.Idle, self).__init__(*args, **kw)
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
                dataPos = self.machine.backend.pixelToData(x, y)
                assert dataPos is not None
                eventDict = prepareHoverSignal(
                    marker['legend'], 'marker',
                    dataPos, (x, y),
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
                                                marker['legend'],
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
                    dataPos = self.backend.pixelToData(x, y)
                    assert dataPos is not None
                    eventDict = prepareCurveSignal('left',
                                                   curve.info['legend'],
                                                   'curve',
                                                   curve.xData[indices],
                                                   curve.yData[indices],
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    self.backend._callback(eventDict)

                elif picked[0] == 'image':
                    _, image, posImg = picked

                    dataPos = self.backend.pixelToData(x, y)
                    assert dataPos is not None
                    eventDict = prepareImageSignal('left',
                                                   image.info['legend'],
                                                   'image',
                                                   posImg[0], posImg[1],
                                                   dataPos[0], dataPos[1],
                                                   x, y)
                    self.backend._callback(eventDict)

    def _signalMarkerMovingEvent(self, eventType, marker, x, y):
        assert marker is not None

        # Mimic MatplotlibBackend signal
        xData, yData = marker['x'], marker['y']
        if xData is None:
            xData = [0, 1]
        if yData is None:
            yData = [0, 1]

        posDataCursor = self.backend.pixelToData(x, y)
        assert posDataCursor is not None

        eventDict = prepareMarkerSignal(eventType,
                                        'left',
                                        marker['legend'],
                                        'marker',
                                        'draggable' in marker['behaviors'],
                                        'selectable' in marker['behaviors'],
                                        (xData, yData),
                                        (x, y),
                                        posDataCursor)
        self.backend._callback(eventDict)

    def beginDrag(self, x, y):
        self._lastPos = self.backend.pixelToData(x, y)
        assert self._lastPos is not None

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
        dataPos = self.backend.pixelToData(x, y)
        assert dataPos is not None
        xData, yData = dataPos

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
                self.marker['legend'],
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
    _UNNAMED_ITEM = '__unnamed_item__'

    _PICK_OFFSET = 3

    _DEFAULT_COLORMAP = {'name': 'gray', 'normalization': 'linear',
                         'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                         'colors': 256}

    def __init__(self, parent=None, **kw):
        self._defaultColormap = self._DEFAULT_COLORMAP

        self._progBase = GLProgram(_baseVertShd, _baseFragShd)
        self._progTex = GLProgram(_texVertShd, _texFragShd)
        self._plotFBOs = {}

        self._plotDataBounds = Bounds(1., 100., 1., 100., 1., 100.)
        self._keepDataAspectRatio = False

        self._activeCurve = None

        self._zoomColor = None

        self.winWidth, self.winHeight = 0, 0

        self._markers = MiniOrderedDict()
        self._items = MiniOrderedDict()
        self._zOrderedItems = MiniOrderedDict()  # For images and curves
        self._selectionAreas = MiniOrderedDict()
        self._glGarbageCollector = []

        self._margins = {'left': 100, 'right': 50, 'top': 50, 'bottom': 50}
        self._lineWidth = 1
        self._tickLen = 5

        self._plotDirtyFlag = True

        self._hasRightYAxis = set()

        self._mousePosition = 0, 0
        self.eventHandler = ZoomAndSelect(self, (0., 0., 0., 1.))

        self._plotHasFocus = set()

        self._plotFrame = GLPlotFrame(self._margins)

        PlotBackend.__init__(self, parent, **kw)

    # Link with embedding toolkit #

    def makeCurrent(self):
        """Override this method to allow to set the current OpenGL context."""
        pass

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
            dataPos = self.pixelToData(inXPixel, inYPixel)
            assert dataPos is not None
            eventDict = prepareMouseSignal('mouseMoved', None,
                                           dataPos[0], dataPos[1],
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
            pixelPos = self.dataToPixel(marker['x'], marker['y'], check=False)

            if marker['x'] is not None:
                xMarker = pixelPos[0]
                xDist = math.fabs(x - xMarker)
            else:
                xDist = 0

            if marker['y'] is not None:
                yMarker = pixelPos[1]
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

        dataPos = self.pixelToData(x, y)
        assert dataPos is not None

        for item in sorted(self._zOrderedItems.values(),
                           key=lambda item: - item.info['zOrder']):
            if test(item):
                if isinstance(item, (GLPlotColormap, GLPlotRGBAImage)):
                    pickedPos = item.pick(*dataPos)
                    if pickedPos is not None:
                        return 'image', item, pickedPos

                elif isinstance(item, GLPlotCurve2D):
                    offset = self._PICK_OFFSET
                    if item.marker is not None:
                        offset = max(item.markerSize / 2., offset)
                    if item.lineStyle is not None:
                        offset = max(item.lineWidth / 2., offset)

                    yAxis = item.info['yAxis']

                    inAreaPos = self._mouseInPlotArea(x - offset, y - offset)
                    dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                               axis=yAxis)
                    assert dataPos is not None
                    xPick0, yPick0 = dataPos

                    inAreaPos = self._mouseInPlotArea(x + offset, y + offset)
                    dataPos = self.pixelToData(inAreaPos[0], inAreaPos[1],
                                               axis=yAxis)
                    assert dataPos is not None
                    xPick1, yPick1 = dataPos

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

    # Default colormap #

    def getSupportedColormaps(self):
        return GLPlotColormap.COLORMAPS

    def getDefaultColormap(self):
        return self._defaultColormap.copy()

    def setDefaultColormap(self, colormap=None):
        if colormap is None:
            self._defaultColormap = self._DEFAULT_COLORMAP
        else:
            assert colormap['name'] in self.getSupportedColormaps()
            if colormap['colors'] != 256:
                warnings.warn("Colormap 'colors' field is ignored",
                              RuntimeWarning)
            self._defaultColormap = colormap.copy()

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
        self._plotDirtyFlag = True

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
                if self._plotFrame.xAxis.isLog and hasattr(item, 'xMinPos'):
                    # Supports curve <= 0. and log
                    if item.xMinPos is not None and item.xMinPos < xMin:
                        xMin = item.xMinPos
                elif item.xMin < xMin:
                    xMin = item.xMin
                if item.xMax > xMax:
                    xMax = item.xMax

                if item.info.get('yAxis') == 'right':
                    if (self._plotFrame.y2Axis.isLog and
                            hasattr(item, 'yMinPos')):
                        # Supports curve <= 0. and log
                        if item.yMinPos is not None and item.yMinPos < y2Min:
                            y2Min = item.yMinPos
                    elif item.yMin < y2Min:
                        y2Min = item.yMin
                    if item.yMax > y2Max:
                        y2Max = item.yMax
                else:
                    if (self._plotFrame.yAxis.isLog and
                            hasattr(item, 'yMinPos')):
                        # Supports curve <= 0. and log
                        if item.yMinPos is not None and item.yMinPos < yMin:
                            yMin = item.yMinPos
                    elif item.yMin < yMin:
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
            # if self._plotFrame.xAxis.isLog and bounds.xAxis.min_ <= 0.:
            #    raise RuntimeError(
            #        'Cannot use plot area with X <= 0 with X axis log scale')
            # if self._plotFrame.yAxis.isLog and (bounds.yAxis.min_ <= 0. or
            #                                    bounds.y2Axis.min_ <= 0.):
            #    raise RuntimeError(
            #        'Cannot use plot area with Y <= 0 with Y axis log scale')
            self._plotDataBounds = bounds

            # Update plot frame bounds
            self._plotFrame.xAxis.dataRange = self.plotDataBounds.xAxis
            self._plotFrame.yAxis.dataRange = self.plotDataBounds.yAxis
            self._plotFrame.y2Axis.dataRange = self.plotDataBounds.y2Axis

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
            if not (self._plotFrame.xAxis.isLog or
                    self._plotFrame.yAxis.isLog):
                self._plotDataTransformedBounds = self.plotDataBounds
            else:
                xMin, xMax = self.plotDataBounds.xAxis
                yMin, yMax = self.plotDataBounds.yAxis
                y2Min, y2Max = self.plotDataBounds.y2Axis

                if self._plotFrame.xAxis.isLog:
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

                if self._plotFrame.yAxis.isLog:
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

            if self._plotFrame.isYAxisInverted:
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

            if self._plotFrame.isYAxisInverted:
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

    def dataToPixel(self, x=None, y=None, axis='left', check=True):
        """
        :param bool check: Toggle checking if data position is in displayed
                           area.
                           If False, this method never returns None.
        :raises: ValueError if x or y < 0. with log axis.
        """
        assert axis in ('left', 'right')

        trBounds = self.plotDataTransformedBounds

        if x is None:
            xDataTr = trBounds.xAxis.center
        else:
            if self._plotFrame.xAxis.isLog:
                if x <= 0.:
                    raise ValueError('Cannot convert x < 0 with log axis.')
                xDataTr = math.log10(x)
            else:
                xDataTr = x

        if y is None:
            if axis == 'left':
                yDataTr = trBounds.yAxis.center
            else:
                yDataTr = trBounds.y2Axis.center
        else:
            if self._plotFrame.yAxis.isLog:
                if y <= 0.:
                    raise ValueError('Cannot convert y < 0 with log axis.')
                yDataTr = math.log10(y)
            else:
                yDataTr = y

        if check and (xDataTr < trBounds.xAxis.min_ or
                      xDataTr > trBounds.xAxis.max_):
            if ((axis == 'left' and
                 (yDataTr < trBounds.yAxis.min_ or
                  yDataTr > trBounds.yAxis.max_)) or
                (yDataTr < trBounds.y2Axis.min_ or
                 yDataTr > trBounds.y2Axis.max_)):
                return None  # (xDataTr, yDataTr) is out of displaayed area

        plotWidth, plotHeight = self.plotSizeInPixels()

        xPixel = int(self._margins['left'] +
                     plotWidth * (xDataTr - trBounds.xAxis.min_) /
                     trBounds.xAxis.range_)

        usedAxis = trBounds.yAxis if axis == "left" else trBounds.y2Axis
        yOffset = plotHeight * (yDataTr - usedAxis.min_) / usedAxis.range_

        if self._plotFrame.isYAxisInverted:
            yPixel = int(self._margins['top'] + yOffset)
        else:
            yPixel = int(self.winHeight - self._margins['bottom'] -
                         yOffset)

        return xPixel, yPixel

    def pixelToData(self, x=None, y=None, axis="left", check=True):
        """
        :param bool check: Toggle checking if pixel is in plot area.
                           If False, this method never returns None.
        """
        assert axis in ("left", "right")

        if x is None:
            x = self.winWidth / 2.
        if y is None:
            y = self.winHeight / 2.

        if check and (x < self._margins['left'] or
                      x > (self.winWidth - self._margins['right']) or
                      y < self._margins['top'] or
                      y > self.winHeight - self._margins['bottom']):
            return None  # (x, y) is out of plot area

        plotWidth, plotHeight = self.plotSizeInPixels()

        trBounds = self.plotDataTransformedBounds

        xData = (x - self._margins['left']) + 0.5
        xData /= float(plotWidth)
        xData = trBounds.xAxis.min_ + xData * trBounds.xAxis.range_
        if self._plotFrame.xAxis.isLog:
            xData = pow(10, xData)

        usedAxis = trBounds.yAxis if axis == "left" else trBounds.y2Axis
        if self._plotFrame.isYAxisInverted:
            yData = y - self._margins['top'] + 0.5
            yData /= float(plotHeight)
            yData = usedAxis.min_ + yData * usedAxis.range_
            if self._plotFrame.yAxis.isLog:
                yData = pow(10, yData)
        else:
            yData = self.winHeight - self._margins['bottom'] - y - 0.5
            yData /= float(plotHeight)
            yData = usedAxis.min_ + yData * usedAxis.range_
            if self._plotFrame.yAxis.isLog:
                yData = pow(10, yData)

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

        # Building shader programs here failed on Mac OS X 10.7.5

    def _paintDirectGL(self):
        self._renderPlotAreaGL()
        self._plotFrame.render()
        self._renderMarkersGL()
        self._renderSelectionGL()

    def _paintFBOGL(self):
        context = getGLContext()
        plotFBOTex = self._plotFBOs.get(context)
        if (self._plotDirtyFlag or self._plotFrame.isDirty or
                plotFBOTex is None):
            self._plotDirtyFlag = False
            self._plotVertices = np.array(((-1., -1., 0., 0.),
                                           (1., -1., 1., 0.),
                                           (-1., 1., 0., 1.),
                                           (1., 1., 1., 1.)),
                                          dtype=np.float32)
            if plotFBOTex is None or \
               plotFBOTex.width != self.winWidth or \
               plotFBOTex.height != self.winHeight:
                if plotFBOTex is not None:
                    plotFBOTex.discard()
                plotFBOTex = FBOTexture(GL_RGBA,
                                        self.winWidth, self.winHeight,
                                        minFilter=GL_NEAREST,
                                        magFilter=GL_NEAREST,
                                        wrapS=GL_CLAMP_TO_EDGE,
                                        wrapT=GL_CLAMP_TO_EDGE)
                self._plotFBOs[context] = plotFBOTex

            with plotFBOTex:
                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
                self._renderPlotAreaGL()
                self._plotFrame.render()

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

        plotFBOTex.bind(texUnit)
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
                    self._plotFrame.xAxis.isLog, self._plotFrame.yAxis.isLog)
        glUniform1i(self._progBase.uniforms['hatchStep'], 0)
        glUniform1f(self._progBase.uniforms['tickLen'], 0.)
        posAttrib = self._progBase.attributes['position']
        glEnableVertexAttribArray(posAttrib)

        labels = []
        pixelOffset = 2

        for marker in self._markers.values():
            xCoord, yCoord = marker['x'], marker['y']

            if marker['text'] is not None:
                pixelPos = self.dataToPixel(xCoord, yCoord, check=False)

                xMin, xMax = self.plotDataBounds.xAxis
                yMin, yMax = self.plotDataBounds.yAxis

                if xCoord is None:
                    x = self.winWidth - self._margins['right'] - pixelOffset
                    y = pixelPos[1] - pixelOffset
                    label = Text2D(marker['text'], x, y,
                                   color=marker['color'],
                                   bgColor=(1., 1., 1., 0.5),
                                   align=RIGHT, valign=BOTTOM)

                    vertices = np.array(((xMin, yCoord),
                                         (xMax, yCoord)),
                                        dtype=np.float32)

                elif yCoord is None:
                    x = pixelPos[0] + pixelOffset
                    y = self._margins['top'] + pixelOffset
                    label = Text2D(marker['text'], x, y,
                                   color=marker['color'],
                                   bgColor=(1., 1., 1., 0.5),
                                   align=LEFT, valign=TOP)

                    vertices = np.array(((xCoord, yMin),
                                         (xCoord, yMax)),
                                        dtype=np.float32)

                else:
                    xPixel, yPixel = pixelPos

                    x, y = xPixel + pixelOffset, yPixel + pixelOffset
                    label = Text2D(marker['text'], x, y,
                                   color=marker['color'],
                                   bgColor=(1., 1., 1., 0.5),
                                   align=LEFT, valign=TOP)

                    x0, y0 = self.pixelToData(xPixel - 2 * pixelOffset,
                                              yPixel - 2 * pixelOffset,
                                              check=False)

                    x1, y1 = self.pixelToData(xPixel + 2 * pixelOffset + 1.,
                                              yPixel + 2 * pixelOffset + 1.,
                                              check=False)

                    vertices = np.array(((x0, yCoord), (x1, yCoord),
                                         (xCoord, y0), (xCoord, y1)),
                                        dtype=np.float32)

                labels.append(label)

            glUniform4f(self._progBase.uniforms['color'], * marker['color'])

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
                        self._plotFrame.xAxis.isLog,
                        self._plotFrame.yAxis.isLog)
            glUniform1f(self._progBase.uniforms['tickLen'], 0.)
            posAttrib = self._progBase.attributes['position']
            colorUnif = self._progBase.uniforms['color']
            hatchStepUnif = self._progBase.uniforms['hatchStep']
            for shape in self._selectionAreas.values():
                shape.render(posAttrib, colorUnif, hatchStepUnif)

            glDisable(GL_SCISSOR_TEST)

    def _renderPlotAreaGL(self):
        plotWidth, plotHeight = self.plotSizeInPixels()

        self._plotFrame.renderGrid()

        glScissor(self._margins['left'], self._margins['bottom'],
                  plotWidth, plotHeight)
        glEnable(GL_SCISSOR_TEST)

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
            if item.info.get('yAxis') == 'right':
                item.render(self.matrixY2PlotDataTransformedProj,
                            self._plotFrame.xAxis.isLog,
                            self._plotFrame.yAxis.isLog)
            else:
                item.render(self.matrixPlotDataTransformedProj,
                            self._plotFrame.xAxis.isLog,
                            self._plotFrame.yAxis.isLog)

        # Render Items
        self._progBase.use()
        glUniformMatrix4fv(self._progBase.uniforms['matrix'], 1, GL_TRUE,
                           self.matrixPlotDataTransformedProj)
        glUniform2i(self._progBase.uniforms['isLog'],
                    self._plotFrame.xAxis.isLog,
                    self._plotFrame.yAxis.isLog)
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
        self._plotFrame.size = width, height

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
                     **kw):
        if kw:
            warnings.warn("insertMarker ignores additional parameters",
                          RuntimeWarning)

        if legend is None:
            legend = self._UNNAMED_ITEM

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        if x is not None and self._plotFrame.xAxis.isLog and x <= 0.:
            raise RuntimeError(
                'Cannot add marker with X <= 0 with X axis log scale')
        if y is not None and self._plotFrame.yAxis.isLog and y <= 0.:
            raise RuntimeError(
                'Cannot add marker with Y <= 0 with Y axis log scale')

        self._markers[legend] = {
            'x': x,
            'y': y,
            'legend': legend,
            'text': text,
            'color': rgba(color, PlotBackend.COLORDICT),
            'behaviors': behaviors,
        }

        self._plotDirtyFlag = True

        return legend

    def insertXMarker(self, x, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kw):
        if kw:
            warnings.warn("insertXMarker ignores additional parameters",
                          RuntimeWarning)
        return self.insertMarker(x, None, legend, text, color,
                                 selectable, draggable, **kw)

    def insertYMarker(self, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False,
                      **kw):
        if kw:
            warnings.warn("insertYMarker ignores additional parameters",
                          RuntimeWarning)
        return self.insertMarker(None, y, legend, text, color,
                                 selectable, draggable, **kw)

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
                 colormap=None, **kw):
        if info is not None:
            warnings.warn("Ignore info parameter of addImage",
                          RuntimeWarning)
        if kw:
            warnings.warn("addImage ignores additional parameters",
                          RuntimeWarning)

        behaviors = set()
        if selectable:
            behaviors.add('selectable')
        if draggable:
            behaviors.add('draggable')

        if legend is None:
            legend = self._UNNAMED_ITEM

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
                image = GLPlotColormap(data,
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
                image = GLPlotRGBAImage(data,
                                        xScale[0], xScale[1],
                                        yScale[0], yScale[1])

            image.info = {
                'legend': legend,
                'zOrder': z,
                'behaviors': behaviors
            }

            if self._plotFrame.xAxis.isLog and image.xMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with X <= 0 with X axis log scale')
            if self._plotFrame.yAxis.isLog and image.yMin <= 0.:
                raise RuntimeError(
                    'Cannot add image with Y <= 0 with Y axis log scale')

            self._zOrderedItems[('image', legend)] = image

        else:
            raise RuntimeError("Unsupported data shape {0}".format(data.shape))

        if oldImage is None or \
           oldXScale != xScale or \
           oldYScale != yScale:
            self._dirtyDataBounds()

        self._resetZoom()

        self._plotDirtyFlag = True

        if replot:
            self.replot()

        return legend  # This is the 'handle'

    def removeImage(self, legend, replot=True):
        if legend is None:
            legend = self._UNNAMED_ITEM

        try:
            image = self._zOrderedItems.pop(('image', legend))
        except KeyError:
            pass
        else:
            self._glGarbageCollector.append(image)
            self._dirtyDataBounds()
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearImages(self):
        # Copy keys as it removes items from the dict
        for type_, legend in list(self._zOrderedItems.keys()):
            if type_ == 'image':
                self.removeImage(legend, replot=False)

    def addItem(self, xList, yList, legend=None, info=None,
                replace=False, replot=True,
                shape="polygon", fill=True, color=None, **kw):
        # info is ignored
        if shape not in self._drawModes:
            raise NotImplementedError("Unsupported shape {0}".format(shape))
        if kw:
            warnings.warn("addItem ignores additional parameters",
                          RuntimeWarning)

        if legend is None:
            legend = self._UNNAMED_ITEM

        if replace:
            self.clearItems()

        colorCode = color if color is not None else 'black'

        if shape == 'rectangle':
            xMin, xMax = xList
            xList = np.array((xMin, xMin, xMax, xMax))
            yMin, yMax = yList
            yList = np.array((yMin, yMax, yMax, yMin))
        else:
            xList = np.array(xList, copy=False)
            yList = np.array(yList, copy=False)

        if self._plotFrame.xAxis.isLog and xList.min() <= 0.:
            raise RuntimeError(
                'Cannot add item with X <= 0 with X axis log scale')
        if self._plotFrame.yAxis.isLog and yList.min() <= 0.:
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
        if legend is None:
            legend = self._UNNAMED_ITEM

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
        if kw:
            warnings.warn("addCurve ignores additional parameters",
                          RuntimeWarning)

        if legend is None:
            legend = self._UNNAMED_ITEM

        x = np.array(x, dtype=np.float32, copy=False, order='C')
        y = np.array(y, dtype=np.float32, copy=False, order='C')

        behaviors = set()
        if selectable:
            behaviors.add('selectable')

        wasActiveCurve = False
        oldCurve = self._zOrderedItems.get(('curve', legend), None)
        if oldCurve is not None:
            if oldCurve == self._activeCurve:
                wasActiveCurve = True
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

        curve = GLPlotCurve2D(x, y, colorArray,
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
            'yAxis': 'left' if yaxis is None else yaxis,
        }

        if yaxis == "right":
            self._hasRightYAxis.add(curve)
            self._plotFrame.isY2Axis = True

        self._zOrderedItems[('curve', legend)] = curve

        if oldCurve is None or \
           oldCurve.xMin != curve.xMin or oldCurve.xMax != curve.xMax or \
           oldCurve.info['axis'] != curve.info['axis'] or \
           oldCurve.yMin != curve.yMin or oldCurve.yMax != curve.yMax:
            self._dirtyDataBounds()

        self._resetZoom()

        self._plotDirtyFlag = True

        if wasActiveCurve:
            self.setActiveCurve(legend, replot=False)

        if replot:
            self.replot()

        return legend

    def removeCurve(self, legend, replot=True):
        if legend is None:
            legend = self._UNNAMED_ITEM

        try:
            curve = self._zOrderedItems.pop(('curve', legend))
        except KeyError:
            pass
        else:
            if curve == self._activeCurve:
                self._activeCurve = None

            self._hasRightYAxis.discard(curve)
            self._plotFrame.isY2Axis = self._hasRightYAxis

            self._glGarbageCollector.append(curve)
            self._dirtyDataBounds()
            self._plotDirtyFlag = True

        if replot:
            self.replot()

    def clearCurves(self):
        # Copy keys as dict is changed
        for type_, legend in list(self._zOrderedItems.keys()):
            if type_ == 'curve':
                self.removeCurve(legend, replot=False)

    def setActiveCurve(self, legend, replot=True):
        if not self._activeCurveHandling:
            return

        if legend is None:
            legend = self._UNNAMED_ITEM

        curve = self._zOrderedItems.get(('curve', legend), None)
        if curve is None:
            raise KeyError("Curve %s not found" % legend)

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
        if curve.info['yAxis'] == 'left' and curve.info['yLabel'] is not None:
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

    def _resetZoom(self):
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

    def resetZoom(self):
        self._resetZoom()
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
        if self._plotFrame.xAxis.isLog or self._plotFrame.yAxis.isLog:
            return False
        else:
            return self._keepDataAspectRatio

    def keepDataAspectRatio(self, flag=True):
        if flag and (self._plotFrame.xAxis.isLog or
                     self._plotFrame.yAxis.isLog):
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
        if flag != self._plotFrame.isYAxisInverted:
            self._plotFrame.isYAxisInverted = flag
            self._dirtyMatrixPlotDataTransformedProj()
            self.updateAxis()

    def isYAxisInverted(self):
        return self._plotFrame.isYAxisInverted

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
        if flag != self._plotFrame.xAxis.isLog:
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            self._plotFrame.xAxis.isLog = flag
            self._dirtyDataBounds()
            self._dirtyPlotDataTransformedBounds()

    def setYAxisLogarithmic(self, flag=True):
        if (flag != self._plotFrame.yAxis.isLog or
                flag != self._plotFrame.y2Axis.isLog):
            if flag and self._keepDataAspectRatio:
                warnings.warn("KeepDataAspectRatio is ignored with log axes",
                              RuntimeWarning)

            self._plotFrame.yAxis.isLog = flag
            self._plotFrame.y2Axis.isLog = flag

            self._dirtyDataBounds()
            self._dirtyPlotDataTransformedBounds()

    def isXAxisLogarithmic(self):
        return self._plotFrame.xAxis.isLog

    def isYAxisLogarithmic(self):
        return self._plotFrame.yAxis.isLog

    # Title, Labels
    def setGraphTitle(self, title=""):
        self._plotFrame.title = title

    def getGraphTitle(self):
        return self._plotFrame.title

    def setGraphXLabel(self, label="X"):
        self._plotFrame.xAxis.title = label
        self.updateAxis()

    def getGraphXLabel(self):
        return self._plotFrame.xAxis.title

    def setGraphYLabel(self, label="Y"):
        self._plotFrame.yAxis.title = label
        self.updateAxis()

    def getGraphYLabel(self):
        return self._plotFrame.yAxis.title

    def showGrid(self, flag=True):
        self._plotFrame.grid = flag
        self._plotDirtyFlag = True
        self.replot()

    # Save
    def saveGraph(self, fileName, fileFormat='svg', dpi=None, **kw):
        """Save the graph as an image to a file.

        WARNING: This method is performing some OpenGL calls.
        It must be called from the main thread.
        """
        if dpi is not None:
            warnings.warn("saveGraph ignores dpi parameter",
                          RuntimeWarning)
        if kw:
            warnings.warn("saveGraph ignores additional parameters",
                          RuntimeWarning)

        if fileFormat not in ['png', 'ppm', 'svg', 'tiff']:
            raise NotImplementedError('Unsupported format: %s' % fileFormat)

        self.makeCurrent()

        data = np.empty((self.winHeight, self.winWidth, 3),
                        dtype=np.uint8, order='C')

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadPixels(0, 0, self.winWidth, self.winHeight,
                     GL_RGB, GL_UNSIGNED_BYTE, data)

        # glReadPixels gives bottom to top,
        # while images are stored as top to bottom
        data = np.flipud(data)

        # fileName is either a file-like object or a str
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
