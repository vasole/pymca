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
This module provides classes to render 2D lines and scatter plots
"""


# import ######################################################################

from OpenGL.GL import *  # noqa
import numpy as np
from math import sqrt as math_sqrt
from collections import defaultdict
from .GLContext import getGLContext
from .GLSupport import Program
from .GLVertexBuffer import createVBOFromArrays

try:
    from ....ctools import minMax
except ImportError:
    from PyMca5.PyMcaGraph.ctools import minMax

_MPL_NONES = None, 'None', '', ' '


# line ########################################################################

SOLID, DASHED = '-', '--'


class _Lines2D(object):
    STYLES = SOLID, DASHED
    """Supported line styles (missing '-.' ':')"""

    _LINEAR, _LOG10_X, _LOG10_Y, _LOG10_X_Y = 0, 1, 2, 3

    _SHADER_SRCS = {
        'vertexTransforms': {
            _LINEAR: """
        vec4 transformXY(float x, float y) {
            return vec4(x, y, 0.0, 1.0);
        }
        """,
            _LOG10_X: """
        vec4 transformXY(float x, float y) {
            return vec4(log(x)/log(10.), y, 0.0, 1.0);
        }
        """,
            _LOG10_Y: """
        vec4 transformXY(float x, float y) {
            return vec4(x, log(y)/log(10.), 0.0, 1.0);
        }
        """,
            _LOG10_X_Y: """
        vec4 transformXY(float x, float y) {
            return vec4(log(x)/log(10.), log(y)/log(10.), 0.0, 1.0);
        }
        """
    },
        SOLID: {
            'vertex': (
            """
        #version 120

        uniform mat4 matrix;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;

        varying vec4 vColor;
        """,
            """
        void main(void) {
            gl_Position = matrix * transformXY(xPos, yPos);
            vColor = color;
        }
        """),
            'fragment': """
        #version 120

        varying vec4 vColor;

        void main(void) {
            gl_FragColor = vColor;
        }
        """
        },


        # Limitation: Dash using an estimate of distance in screen coord
        # to avoid computing distance when viewport is resized
        # results in inequal dashes when viewport aspect ratio is far from 1
        DASHED: {
            'vertex': (
            """
        #version 120

        uniform mat4 matrix;
        uniform vec2 halfViewportSize;
        attribute float xPos;
        attribute float yPos;
        attribute vec4 color;
        attribute float distance;

        varying float vDist;
        varying vec4 vColor;
        """,
            """
        void main(void) {
            gl_Position = matrix * transformXY(xPos, yPos);
            //Estimate distance in pixels
            vec2 probe = vec2(matrix * vec4(1., 1., 0., 0.)) *
                         halfViewportSize;
            float pixelPerDataEstimate = length(probe)/sqrt(2.);
            vDist = distance * pixelPerDataEstimate;
            vColor = color;
        }
        """),
            'fragment': """
        #version 120

        uniform float dashPeriod;

        varying float vDist;
        varying vec4 vColor;

        void main(void) {
            if (mod(vDist, dashPeriod) > 0.5 * dashPeriod) {
                discard;
            } else {
                gl_FragColor = vColor;
            }
        }
        """
        }
    }

    _programs = defaultdict(dict)

    def __init__(self, xVboData=None, yVboData=None,
                 colorVboData=None, distVboData=None,
                 style=SOLID, color=(0., 0., 0., 1.),
                 width=1, dashPeriod=20,
                 isXLog=False, isYLog=False):
        self.xVboData = xVboData
        self.yVboData = yVboData
        self.distVboData = distVboData
        self.colorVboData = colorVboData

        self.color = color
        self.width = width
        self.style = style
        self.dashPeriod = dashPeriod
        self.isXLog = isXLog
        self.isYLog = isYLog

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        if style in _MPL_NONES:
            self._style = None
            self.render = self._renderNone
        else:
            assert(style in self.STYLES)
            self._style = style
            if style == SOLID:
                self.render = self._renderSolid
            elif style == DASHED:
                self.render = self._renderDash

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        try:
            widthRange = self._widthRange
        except AttributeError:
            widthRange = glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE)
            # Shared among contexts, this should be enough..
            _Lines2D._widthRange = widthRange
        assert(width >= widthRange[0] and width <= widthRange[1])
        self._width = width

    @classmethod
    def _getProgram(cls, transform, style):
        context = getGLContext()
        programsForStyle = cls._programs[style]
        try:
            prgm = programsForStyle[context]
        except KeyError:
            sources = cls._SHADER_SRCS[style]
            vertexShdr = sources['vertex'][0] + \
                cls._SHADER_SRCS['vertexTransforms'][transform] + \
                sources['vertex'][1]
            prgm = Program(vertexShdr,
                           sources['fragment'])
            programsForStyle[context] = prgm
        return prgm

    @classmethod
    def init(cls):
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def _renderNone(self, matrix):
        pass

    render = _renderNone  # Overridden in style setter

    def _renderSolid(self, matrix):
        if self.isXLog:
            transform = self._LOG10_X_Y if self.isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if self.isYLog else self._LINEAR

        prog = self._getProgram(transform, SOLID)
        prog.use()

        glEnable(GL_LINE_SMOOTH)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)

        colorAttrib = prog.attributes['color']
        if self.colorVboData is not None:
            glEnableVertexAttribArray(colorAttrib)
            self.colorVboData.setVertexAttrib(colorAttrib)
        else:
            glDisableVertexAttribArray(colorAttrib)
            glVertexAttrib4f(colorAttrib, *self.color)

        xPosAttrib = prog.attributes['xPos']
        glEnableVertexAttribArray(xPosAttrib)
        self.xVboData.setVertexAttrib(xPosAttrib)

        yPosAttrib = prog.attributes['yPos']
        glEnableVertexAttribArray(yPosAttrib)
        self.yVboData.setVertexAttrib(yPosAttrib)

        glLineWidth(self.width)
        glDrawArrays(GL_LINE_STRIP, 0, self.xVboData.size)

        glDisable(GL_LINE_SMOOTH)

    def _renderDash(self, matrix):
        if self.isXLog:
            transform = self._LOG10_X_Y if self.isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if self.isYLog else self._LINEAR

        prog = self._getProgram(transform, DASHED)
        prog.use()

        glEnable(GL_LINE_SMOOTH)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        x, y, viewWidth, viewHeight = glGetFloat(GL_VIEWPORT)
        glUniform2f(prog.uniforms['halfViewportSize'],
                    0.5 * viewWidth, 0.5 * viewHeight)

        glUniform1f(prog.uniforms['dashPeriod'], self.dashPeriod)

        colorAttrib = prog.attributes['color']
        if self.colorVboData is not None:
            glEnableVertexAttribArray(colorAttrib)
            self.colorVboData.setVertexAttrib(colorAttrib)
        else:
            glDisableVertexAttribArray(colorAttrib)
            glVertexAttrib4f(colorAttrib, *self.color)

        distAttrib = prog.attributes['distance']
        glEnableVertexAttribArray(distAttrib)
        self.distVboData.setVertexAttrib(distAttrib)

        xPosAttrib = prog.attributes['xPos']
        glEnableVertexAttribArray(xPosAttrib)
        self.xVboData.setVertexAttrib(xPosAttrib)

        yPosAttrib = prog.attributes['yPos']
        glEnableVertexAttribArray(yPosAttrib)
        self.yVboData.setVertexAttrib(yPosAttrib)

        glLineWidth(self.width)
        glDrawArrays(GL_LINE_STRIP, 0, self.xVboData.size)

        glDisable(GL_LINE_SMOOTH)


def _lines2DFromArrays(xData, yData, cData=None, **kwargs):
    if cData is None:
        xAttrib, yAttrib = createVBOFromArrays((xData, yData))
        return _Lines2D(xAttrib, yAttrib, None, None, **kwargs)
    else:
        xAttrib, yAttrib, cAttrib = createVBOFromArrays((xData, yData, cData))
        return _Lines2D(xAttrib, yAttrib, cAttrib, None, **kwargs)


def _distancesFromArrays(xData, yData):
    dists = np.empty((xData.size, 2), dtype=np.float32)
    totalDist = 0.
    point = None
    # TODO make it better
    for index, (x, y) in enumerate(zip(xData, yData)):
        if point is not None:
            totalDist += math_sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        dists[index] = totalDist
        point = (x, y)
    return dists


def _dashedLines2DFromArrays(xData, yData, cData=None, **kwargs):
    dists = _distancesFromArrays(xData, yData)
    if cData is None:
        arrays = xData, yData, dists
        xAttrib, yAttrib, dAttrib = createVBOFromArrays(arrays)
        return _Lines2D(xAttrib, yAttrib, None, dAttrib,
                        style=DASHED, *args, **kwargs)

    else:
        arrays = xData, yData, cData, dists
        xAttrib, yAttrib, cAttrib, dAttrib = createVBOFromArrays(arrays)
        return _Lines2D(xAttrib, yAttrib, cAttrib, dAttrib,
                        style=DASHED, *args, **kwargs)


# points ######################################################################

DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL = \
    'd', 'o', 's', '+', 'x', '.', ','


class _Points2D(object):
    MARKERS = DIAMOND, CIRCLE, SQUARE, PLUS, X_MARKER, POINT, PIXEL

    _LINEAR, _LOG10_X, _LOG10_Y, _LOG10_X_Y = 0, 1, 2, 3

    _SHDRS = {
        'vertexTransforms': {
            _LINEAR: """
        vec4 transformXY(float x, float y) {
            return vec4(x, y, 0.0, 1.0);
        }
        """,
            _LOG10_X: """
        vec4 transformXY(float x, float y) {
            return vec4(log(x)/log(10.), y, 0.0, 1.0);
        }
        """,
            _LOG10_Y: """
        vec4 transformXY(float x, float y) {
            return vec4(x, log(y)/log(10.), 0.0, 1.0);
        }
        """,
            _LOG10_X_Y: """
        vec4 transformXY(float x, float y) {
            return vec4(log(x)/log(10.), log(y)/log(10.), 0.0, 1.0);
        }
        """
    },
        'vertex': (
        """
    #version 120

    uniform mat4 matrix;
    uniform int transform;
    uniform float size;
    attribute float xPos;
    attribute float yPos;
    attribute vec4 color;

    varying vec4 vColor;
    """,
        """
    void main(void) {
        gl_Position = matrix * transformXY(xPos, yPos);
        vColor = color;
        gl_PointSize = size;
    }
    """),

        'fragmentSymbols': {
            DIAMOND: """
        float alphaSymbol(vec2 coord, float size) {
            vec2 centerCoord = abs(coord - vec2(0.5, 0.5));
            float f = centerCoord.x + centerCoord.y;
            return clamp(size * (0.5 - f), 0., 1.);
        }
        """,
            CIRCLE: """
        float alphaSymbol(vec2 coord, float size) {
            float radius = 0.5;
            float r = distance(coord, vec2(0.5, 0.5));
            return clamp(size * (radius - r), 0., 1.);
        }
        """,
            SQUARE: """
        float alphaSymbol(vec2 coord, float size) {
            return 1.;
        }
        """,
            PLUS: """
        float alphaSymbol(vec2 coord, float size) {
            vec2 d = abs(size * (coord - vec2(0.5, 0.5)));
            if (min(d.x, d.y) < 0.5) {
                return 1.;
            } else {
                return 0.;
            }
        }
        """,
            X_MARKER: """
        float alphaSymbol(vec2 coord, float size) {
            float d1 = abs(coord.x - coord.y);
            float d2 = abs(coord.x + coord.y - 1.);
            if (min(d1, d2) < 0.5/size) {
                return 1.;
            } else {
                return 0.;
            }
        }
        """
        },

        'fragment': (
        """
    #version 120

    uniform float size;

    varying vec4 vColor;
    """,
        """
    void main(void) {
        float alpha = alphaSymbol(gl_PointCoord, size);
        if (alpha <= 0.) {
            discard;
        } else {
            gl_FragColor = vColor;
            gl_FragColor.a = mix(0., gl_FragColor.a, alpha);
        }
    }
    """)
    }

    _programs = defaultdict(dict)

    def __init__(self, xVboData=None, yVboData=None, colorVboData=None,
                 marker=SQUARE, color=(0., 0., 0., 1.), size=7,
                 isXLog=False, isYLog=False):
        self.color = color
        self.marker = marker
        self.size = size
        self.isXLog = isXLog
        self.isYLog = isYLog

        self.xVboData = xVboData
        self.yVboData = yVboData
        self.colorVboData = colorVboData

    @property
    def marker(self):
        return self._marker

    @marker.setter
    def marker(self, marker):
        if marker in _MPL_NONES:
            self._marker = None
            self.render = self._renderNone
        else:
            assert(marker in self.MARKERS)
            self._marker = marker
            self.render = self._renderMarkers

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        try:
            sizeRange = self._sizeRange
        except AttributeError:
            sizeRange = glGetFloatv(GL_POINT_SIZE_RANGE)
            # Shared among contexts, this should be enough..
            _Points2D._sizeRange = sizeRange
        assert(size >= sizeRange[0] and size <= sizeRange[1])
        self._size = size

    @classmethod
    def _getProgram(cls, transform, marker):
        context = getGLContext()
        if marker in (POINT, PIXEL):
            marker = SQUARE
        programs = cls._programs[(transform, marker)]
        try:
            prgm = programs[context]
        except KeyError:
            vertShdr = cls._SHDRS['vertex'][0] + \
                       cls._SHDRS['vertexTransforms'][transform] + \
                       cls._SHDRS['vertex'][1]
            fragShdr = cls._SHDRS['fragment'][0] + \
                       cls._SHDRS['fragmentSymbols'][marker] + \
                       cls._SHDRS['fragment'][1]
            prgm = Program(vertShdr, fragShdr)

            programs[context] = prgm
        return prgm

    @classmethod
    def init(cls):
        version = glGetString(GL_VERSION)
        majorVersion = int(version[0])
        assert(majorVersion >= 2)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # OpenGL 2
        glEnable(GL_POINT_SPRITE)  # OpenGL 2
        if majorVersion >= 3:  # OpenGL 3
            glEnable(GL_PROGRAM_POINT_SIZE)

    def _renderNone(self, matrix):
        pass

    render = _renderNone

    def _renderMarkers(self, matrix):
        if self.isXLog:
            transform = self._LOG10_X_Y if self.isYLog else self._LOG10_X
        else:
            transform = self._LOG10_Y if self.isYLog else self._LINEAR

        prog = self._getProgram(transform, self.marker)
        prog.use()
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        if self.marker in (POINT, PIXEL):
            size = 1
        else:
            size = self.size
        glUniform1f(prog.uniforms['size'], size)
        # glPointSize(self.size)

        cAttrib = prog.attributes['color']
        if self.colorVboData:
            glEnableVertexAttribArray(cAttrib)
            self.colorVboData.setVertexAttrib(cAttrib)
        else:
            glDisableVertexAttribArray(cAttrib)
            glVertexAttrib4f(cAttrib, *self.color)

        xAttrib = prog.attributes['xPos']
        glEnableVertexAttribArray(xAttrib)
        self.xVboData.setVertexAttrib(xAttrib)

        yAttrib = prog.attributes['yPos']
        glEnableVertexAttribArray(yAttrib)
        self.yVboData.setVertexAttrib(yAttrib)

        glDrawArrays(GL_POINTS, 0, self.xVboData.size)

        glUseProgram(0)


def _points2DFromArrays(xData, yData, cData=None, *kwargs):
    if cData is None:
        xAttrib, yAttrib = createVBOFromArrays((xData, yData))
        cAttrib = None
    else:
        xAttrib, yAttrib, cAttrib = createVBOFromArrays((xData, yData, cData))

    return _Points2D(xVboAttrib, yVboAttrib, **kwargs)


# curves ######################################################################

def _proxyProperty(*componentsAttributes):
    """Create a property to access an attribute of attribute(s).
    Useful for composition.
    Supports multiple components this way:
    getter returns the first found, setter sets all
    """
    def getter(self):
        for compName, attrName in componentsAttributes:
            try:
                component = getattr(self, compName)
            except AttributeError:
                pass
            else:
                return getattr(component, attrName)

    def setter(self, value):
        for compName, attrName in componentsAttributes:
            component = getattr(self, compName)
            setattr(component, attrName, value)
    return property(getter, setter)

class Curve2D(object):
    def __init__(self, xData, yData, colorData=None,
                 lineStyle=None, lineColor=None,
                 lineWidth=None, lineDashPeriod=None,
                 marker=None, markerColor=None, markerSize=None,
                 isXLog=False, isYLog=False):
        self.xData, self.yData, self.colorData = xData, yData, colorData
        self.xMin, self.xMax = minMax(xData)
        self.yMin, self.yMax = minMax(yData)

        kwargs = {
            'style': lineStyle,
            'isXLog': isXLog,
            'isYLog': isYLog
        }
        if lineColor is not None:
            kwargs['color'] = lineColor
        if lineWidth is not None:
            kwargs['width'] = lineWidth
        if lineDashPeriod is not None:
            kwargs['dashPeriod'] = lineDashPeriod
        self.lines = _Lines2D(**kwargs)

        kwargs = {
            'marker': marker,
            'isXLog': isXLog,
            'isYLog': isYLog
        }
        if markerColor is not None:
            kwargs['color'] = markerColor
        if markerSize is not None:
            kwargs['size'] = markerSize
        self.points = _Points2D(**kwargs)

    xVboData = _proxyProperty(('lines', 'xVboData'), ('points', 'xVboData'))

    yVboData = _proxyProperty(('lines', 'yVboData'), ('points', 'yVboData'))

    colorVboData = _proxyProperty(('lines', 'colorVboData'),
                                  ('points', 'colorVboData'))

    distVboData = _proxyProperty(('lines', 'distVboData'))

    lineStyle = _proxyProperty(('lines', 'style'))

    lineColor = _proxyProperty(('lines', 'color'))

    lineWidth = _proxyProperty(('lines', 'width'))

    lineDashPeriod = _proxyProperty(('lines', 'dashPeriod'))

    marker = _proxyProperty(('points', 'marker'))

    markerColor = _proxyProperty(('points', 'color'))

    markerSize = _proxyProperty(('points', 'size'))

    isXLog = _proxyProperty(('lines', 'isXLog'), ('points', 'isXLog'))

    isYLog = _proxyProperty(('lines', 'isYLog'), ('points', 'isYLog'))

    @classmethod
    def init(cls):
        _Lines2D.init()
        _Points2D.init()

    def prepare(self):
        if self.xVboData is None:
            xAttrib, yAttrib, cAttrib, dAttrib = None, None, None, None
            if self.lineStyle == '--':
                dists = _distancesFromArrays(self.xData, self.yData)
                if self.colorData is None:
                    xAttrib, yAttrib, dAttrib = createVBOFromArrays(
                        (self.xData, self.yData, dists))
                else:
                    xAttrib, yAttrib, cAttrib, dAttrib = createVBOFromArrays(
                        (self.xData, self.yData, self.colorData, dists))

            elif self.colorData is None:
                xAttrib, yAttrib = createVBOFromArrays(
                    (self.xData, self.yData))
            else:
                xAttrib, yAttrib, cAttrib = createVBOFromArrays(
                    (self.xData, self.yData, self.colorData))

            self.xVboData = xAttrib
            self.yVboData = yAttrib
            self.colorVboData = cAttrib
            self.distVboData = dAttrib

    def render(self, matrix):
        self.prepare()
        self.lines.render(matrix)
        self.points.render(matrix)

    def pick(self, xPickMin, yPickMin, xPickMax, yPickMax):
        if (self.marker is None and self.lineStyle is None) or \
           self.xMin > xPickMax or xPickMin > self.xMax or \
           self.yMin > yPickMax or yPickMin > self.yMax:
            return None

        elif self.lineStyle is not None:
            # Using Cohen-Sutherland algorithm for line clipping
            picked = []

            pt0Code = int('1111', 2)  # Initialisation trick

            for x1, y1 in zip(self.xData, self.yData):
                pt1Code = ((y1 > yPickMax) << 3) | \
                          ((y1 < yPickMin) << 2) | \
                          ((x1 > xPickMax) << 1) | \
                          (x1 < xPickMin)

                if pt0Code == 0:  # Already added
                    pass
                elif pt1Code == 0:  # pt1 inside
                    picked.append((x1, y1))
                elif (pt0Code & pt1Code) == 0:  # check for crossing
                    if y1 > yPickMax:
                        x = x0 + (x1 - x0) * (yPickMax - y0) / (y1 - y0)
                    elif y1 < yPickMin:
                        x = x0 + (x1 - x0) * (yPickMin - y0) / (y1 - y0)
                    else:
                        x = xPickMin

                    if x >= xPickMin and x <= xPickMax:  # Intersection
                        picked.append((x0, y0))

                    else:
                        if x1 > xPickMax:
                            y = y0 + (y1 - y0) * (xPickMax - x0) / (x1 - x0)
                        elif x1 < xPickMin:
                            y = y0 + (y1 - y0) * (xPickMin - x0) / (x1 - x0)
                        else:
                            y = yPickMin

                        if y >= yPickMin and y <= yPickMax:  # Intersection
                            picked.append((x0, y0))

                x0, y0, pt0Code = x1, y1, pt1Code

        else:
            picked = [
                (xData, yData)
                for xData, yData in zip(self.xData, self.yData)
                if xData >= xPickMin and xData <= xPickMax and
                yData >= yPickMin and yData <= yPickMax
            ]

        return picked if picked else None


# main ########################################################################


if __name__ == "__main__":
    from OpenGL.GLUT import *  # noqa
    from .GLSupport import mat4Ortho

    glutInit(sys.argv)
    glutInitDisplayString("double rgba stencil")
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(0, 0)
    glutCreateWindow('Line Plot Test')

    # GL init
    glClearColor(1., 1., 1., 1.)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    _Lines2D.init()
    _Points2D.init()

    # Plot data init
    xData1 = np.arange(10, dtype=np.float32) * 100
    xData1[3] -= 100
    yData1 = np.asarray(np.random.random(10) * 500, dtype=np.float32)
    yData1 = np.array((100, 100, 200, 400, 100, 100, 400, 400, 401, 400),
                      dtype=np.float32)
    curve1 = Curve2D(xData1, yData1, marker='o', lineStyle='--')

    xData2 = np.arange(1000, dtype=np.float32) * 1
    yData2 = np.asarray(500 + np.random.random(1000) * 500, dtype=np.float32)
    curve2 = Curve2D(xData2, yData2, lineStyle='', marker='s')

    projMatrix = mat4Ortho(0, 1000, 0, 1000, -1, 1)

    def display():
        glClear(GL_COLOR_BUFFER_BIT)
        curve1.render(projMatrix)
        curve2.render(projMatrix)
        glutSwapBuffers()

    def resize(width, height):
        glViewport(0, 0, width, height)

    def idle():
        glutPostRedisplay()

    glutDisplayFunc(display)
    glutReshapeFunc(resize)
    # glutIdleFunc(idle)

    sys.exit(glutMainLoop())
