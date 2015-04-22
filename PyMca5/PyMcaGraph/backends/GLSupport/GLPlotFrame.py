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
from __future__ import with_statement

__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This modules provides the rendering of plot titles, axes and grid.
"""

# TODO
# keep aspect ratio managed here?
# smarter dirty flag handling?


# import ######################################################################

import logging
import numpy as np
import math
import weakref
import warnings

from .gl import *  # noqa
from .GLSupport import mat4Ortho
from .GLProgram import GLProgram
from .GLText import Text2D, CENTER, BOTTOM, TOP, LEFT, RIGHT, ROTATE_270
from .LabelLayout import niceNumbersAdaptative, niceNumbersForLog10


# PlotAxis ####################################################################

class PlotAxis(object):
    """Represents a 1D axis of the plot.
    This class is intended to be used with :class:`GLPlotFrame`.
    """

    def __init__(self, plot,
                 tickLength=(0., 0.),
                 labelAlign=CENTER, labelVAlign=CENTER,
                 titleAlign=CENTER, titleVAlign=CENTER,
                 titleRotate=0, titleOffset=(0., 0.)):
        self._ticks = None

        self._plot = weakref.ref(plot)

        self._isLog = False
        self._dataRange = 1., 1.
        self._displayCoords = (0., 0.), (1., 0.)
        self._title = ''

        self._tickLength = tickLength
        self._labelAlign = labelAlign
        self._labelVAlign = labelVAlign
        self._titleAlign = titleAlign
        self._titleVAlign = titleVAlign
        self._titleRotate = titleRotate
        self._titleOffset = titleOffset

    @property
    def dataRange(self):
        """The range of the data represented on the axis as a tuple
        of 2 floats: (min, max)."""
        return self._dataRange

    @dataRange.setter
    def dataRange(self, dataRange):
        assert len(dataRange) == 2
        assert dataRange[0] <= dataRange[1]
        dataRange = tuple(dataRange)

        if dataRange != self._dataRange:
            self._dataRange = dataRange
            self._dirtyTicks()

    @property
    def isLog(self):
        """Whether the axis is using a log10 scale or not as a bool."""
        return self._isLog

    @isLog.setter
    def isLog(self, isLog):
        isLog = bool(isLog)
        if isLog != self._isLog:
            self._isLog = isLog
            self._dirtyTicks()

    @property
    def displayCoords(self):
        """The coordinates of the start and end points of the axis
        in display space (i.e., in pixels) as a tuple of 2 tuples of
        2 floats: ((x0, y0), (x1, y1)).
        """
        return self._displayCoords

    @displayCoords.setter
    def displayCoords(self, displayCoords):
        assert len(displayCoords) == 2
        assert len(displayCoords[0]) == 2
        assert len(displayCoords[1]) == 2
        displayCoords = tuple(displayCoords[0]), tuple(displayCoords[1])
        if displayCoords != self._displayCoords:
            self._displayCoords = displayCoords
            self._dirtyTicks()

    @property
    def title(self):
        """The text label associated with this axis as a str in latin-1."""
        return self._title

    @title.setter
    def title(self, title):
        if title != self._title:
            self._title = title

            plot = self._plot()
            if plot is not None:
                plot._dirty()

    @property
    def ticks(self):
        """Ticks as tuples: ((x, y) in display, dataPos, textLabel)."""
        if self._ticks is None:
            self._ticks = tuple(self._ticksGenerator())
        return self._ticks

    def getVerticesAndLabels(self):
        """Create the list of vertices for axis and associated text labels.

        :returns: A tuple: List of 2D line vertices, List of Text2D labels.
        """
        vertices = list(self.displayCoords)  # Add start and end points
        labels = []
        tickLabelsSize = [0., 0.]

        xTickLength, yTickLength = self._tickLength
        for (xPixel, yPixel), dataPos, text in self.ticks:
            if text is None:
                tickScale = 0.5
            else:
                tickScale = 1.

                label = Text2D(text=text,
                               x=xPixel - xTickLength,
                               y=yPixel - yTickLength,
                               align=self._labelAlign,
                               valign=self._labelVAlign)

                width, height = label.size
                if width > tickLabelsSize[0]:
                    tickLabelsSize[0] = width
                if height > tickLabelsSize[1]:
                    tickLabelsSize[1] = height

                labels.append(label)

            vertices.append((xPixel, yPixel))
            vertices.append((xPixel + tickScale * xTickLength,
                             yPixel + tickScale * yTickLength))

        (x0, y0), (x1, y1) = self.displayCoords
        xAxisCenter = 0.5 * (x0 + x1)
        yAxisCenter = 0.5 * (y0 + y1)

        xOffset, yOffset = self._titleOffset

        # Adaptative title positioning:
        # tickNorm = math.sqrt(xTickLength ** 2 + yTickLength ** 2)
        # xOffset = -tickLabelsSize[0] * xTickLength / tickNorm
        # xOffset -= 3 * xTickLength
        # yOffset = -tickLabelsSize[1] * yTickLength / tickNorm
        # yOffset -= 3 * yTickLength

        axisTitle = Text2D(text=self.title,
                           x=xAxisCenter + xOffset,
                           y=yAxisCenter + yOffset,
                           align=self._titleAlign,
                           valign=self._titleVAlign,
                           rotate=self._titleRotate)
        labels.append(axisTitle)

        return vertices, labels

    def _dirtyTicks(self):
        """Mark ticks as dirty and notify listener (i.e., background)."""
        self._ticks = None
        plot = self._plot()
        if plot is not None:
            plot._dirty()

    @staticmethod
    def _frange(start, stop, step):
        """range for float (including stop)."""
        while start <= stop:
            yield start
            start += step

    def _ticksGenerator(self):
        """Generator of ticks as tuples:
        ((x, y) in display, dataPos, textLabel).
        """
        dataMin, dataMax = self.dataRange
        if self.isLog and dataMin <= 0.:
            warnings.warn(
                'Getting ticks while isLog=True and dataRange[0]<=0.',
                RuntimeWarning)
            dataMin = 1.
            if dataMax < dataMin:
                dataMax = 1.

        if dataMin != dataMax:  # data range is not null
            (x0, y0), (x1, y1) = self.displayCoords

            if self.isLog:
                logMin, logMax = math.log10(dataMin), math.log10(dataMax)
                tickMin, tickMax, step = niceNumbersForLog10(logMin, logMax)

                xScale = (x1 - x0) / (logMax - logMin)
                yScale = (y1 - y0) / (logMax - logMin)

                for logPos in self._frange(tickMin, tickMax, step):
                    if logPos >= logMin and logPos <= logMax:
                        dataPos = 10 ** logPos
                        xPixel = x0 + (logPos - logMin) * xScale
                        yPixel = y0 + (logPos - logMin) * yScale
                        text = '1e%+03d' % logPos
                        yield ((xPixel, yPixel), dataPos, text)

                if step == 1:
                    ticks = list(self._frange(tickMin, tickMax, step))[:-1]
                    for logPos in ticks:
                        dataOrigPos = 10 ** logPos
                        for index in range(2, 10):
                            dataPos = dataOrigPos * index
                            if dataPos >= dataMin and dataPos <= dataMax:
                                logSubPos = math.log10(dataPos)
                                xPixel = x0 + (logSubPos - logMin) * xScale
                                yPixel = y0 + (logSubPos - logMin) * yScale
                                yield ((xPixel, yPixel), dataPos, None)

            else:
                xScale = (x1 - x0) / (dataMax - dataMin)
                yScale = (y1 - y0) / (dataMax - dataMin)

                nbPixels = math.sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))

                # Density of 1.3 label per 92 pixels
                # i.e., 1.3 label per inch on a 92 dpi screen
                tickMin, tickMax, step, nbFrac = niceNumbersAdaptative(
                    dataMin, dataMax, nbPixels, 1.3 / 92)

                for dataPos in self._frange(tickMin, tickMax, step):
                    if dataPos >= dataMin and dataPos <= dataMax:
                        xPixel = x0 + (dataPos - dataMin) * xScale
                        yPixel = y0 + (dataPos - dataMin) * yScale

                        if nbFrac == 0:
                            text = '%g' % dataPos
                        else:
                            text = ('%.' + str(nbFrac) + 'f') % dataPos
                        yield ((xPixel, yPixel), dataPos, text)


# GLPlotFrame #################################################################

class GLPlotFrame(object):

    _TICK_LENGTH_IN_PIXELS = 5
    _LINE_WIDTH = 1

    _SHADERS = {
        'vertex': """
    attribute vec2 position;
    uniform mat4 matrix;

    void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """,
        'fragment': """
    uniform vec4 color;
    uniform float tickFactor; /* = 1./tickLength or 0. for solid line */

    void main(void) {
        if (mod(tickFactor * (gl_FragCoord.x + gl_FragCoord.y), 2.) < 1.) {
            gl_FragColor = color;
        } else {
            discard;
        }
    }
    """
    }

    def __init__(self, margins):
        """
        :param margins: The margins around plot area for axis and labels.
        :type margins: dict with 'left', 'right', 'top', 'bottom' keys and
                       values as ints.
        """
        self._renderResources = None

        self._margins = dict(margins)

        self.xAxis = PlotAxis(self,
                              tickLength=(0., -5.),
                              labelAlign=CENTER, labelVAlign=TOP,
                              titleAlign=CENTER, titleVAlign=TOP,
                              titleRotate=0,
                              titleOffset=(0, self._margins['bottom'] // 2))
        self._x2AxisCoords = ()

        self.yAxis = PlotAxis(self,
                              tickLength=(5., 0.),
                              labelAlign=RIGHT, labelVAlign=CENTER,
                              titleAlign=CENTER, titleVAlign=BOTTOM,
                              titleRotate=ROTATE_270,
                              titleOffset=(-3 * self._margins['left'] // 4, 0))

        self.y2Axis = PlotAxis(self,
                               tickLength=(-5., 0.),
                               labelAlign=LEFT, labelVAlign=CENTER,
                               titleAlign=CENTER, titleVAlign=TOP,
                               titleRotate=ROTATE_270,
                               titleOffset=(3*self._margins['right'] // 4, 0))
        self._y2AxisCoords = ()

        self._grid = False
        self._isY2Axis = False
        self._isYAxisInverted = False
        self._size = 0., 0.
        self._title = ''

    @property
    def isDirty(self):
        """True if it need to refresh graphic rendering, False otherwise."""
        return self._renderResources is None

    GRID_NONE = 0
    GRID_MAIN_TICKS = 1
    GRID_SUB_TICKS = 2
    GRID_ALL_TICKS = (GRID_MAIN_TICKS + GRID_SUB_TICKS)

    @property
    def grid(self):
        """Grid display mode:
        - 0: No grid.
        - 1: Grid on main ticks.
        - 2: Grid on sub-ticks for log scale axes.
        - 3: Grid on main and sub ticks."""
        return self._grid

    @grid.setter
    def grid(self, grid):
        assert grid in (self.GRID_NONE, self.GRID_MAIN_TICKS,
                        self.GRID_SUB_TICKS, self.GRID_ALL_TICKS)
        if grid != self._grid:
            self._grid = grid
            self._dirty()

    @property
    def isY2Axis(self):
        """Whether to display the left Y axis or not."""
        return self._isY2Axis

    @isY2Axis.setter
    def isY2Axis(self, isY2Axis):
        isY2Axis = bool(isY2Axis)
        if isY2Axis != self._isY2Axis:
            self._isY2Axis = isY2Axis
            self._dirty()

    @property
    def isYAxisInverted(self):
        """Whether Y axes are inverted or not as a bool."""
        return self._isYAxisInverted

    @isYAxisInverted.setter
    def isYAxisInverted(self, value):
        value = bool(value)
        if value != self._isYAxisInverted:
            self._isYAxisInverted = value
            self._dirty()

    @property
    def size(self):
        """Size in pixels of the plot area including margins."""
        return self._size

    @size.setter
    def size(self, size):
        assert len(size) == 2
        size = tuple(size)
        if size != self._size:
            self._size = size
            self._dirty()

    @property
    def title(self):
        """Main title as a str in latin-1."""
        return self._title

    @title.setter
    def title(self, title):
        if title != self._title:
            self._title = title
            self._dirty()

        # In-place update
        # if self._renderResources is not None:
        #    self._renderResources[-1][-1].text = title

    def _dirty(self):
        # When Text2D require discard we need to handle it
        self._renderResources = None

    def _updateAxis(self):
        width, height = self.size

        xCoords = (self._margins['left'] - 0.5,
                   width - self._margins['right'] + 0.5)
        yCoords = (height - self._margins['bottom'] + 0.5,
                   self._margins['top'] - 0.5)

        self.xAxis.displayCoords = ((xCoords[0], yCoords[0]),
                                    (xCoords[1], yCoords[0]))

        self._x2AxisCoords = ((xCoords[0], yCoords[1]),
                              (xCoords[1], yCoords[1]))

        if self.isYAxisInverted:
            # Y axes are inverted, axes coordinates are inverted
            yCoords = yCoords[1], yCoords[0]

        self.yAxis.displayCoords = ((xCoords[0], yCoords[0]),
                                    (xCoords[0], yCoords[1]))

        self._y2AxisCoords = ((xCoords[1], yCoords[0]),
                              (xCoords[1], yCoords[1]))
        self.y2Axis.displayCoords = self._y2AxisCoords

    def _buildGridVertices(self):
        if self._grid == self.GRID_NONE:
            return []

        elif self._grid == self.GRID_MAIN_TICKS:
            test = lambda text: text is not None
        elif self._grid == self.GRID_SUB_TICKS:
            test = lambda text: text is None
        elif self._grid == self.GRID_ALL_TICKS:
            test = lambda text: True
        else:
            logging.warning('Wrong grid mode: %d' % self._grid)
            return []

        vertices = []

        for (xPixel, yPixel), xData, text in self.xAxis.ticks:
            if test(text):
                vertices.append((xPixel, yPixel))
                vertices.append((xPixel, self._margins['top']))

        for (xPixel, yPixel), xData, text in self.yAxis.ticks:
            if test(text):
                vertices.append((xPixel, yPixel))
                vertices.append((self.size[0] - self._margins['right'],
                                 yPixel))

        if self.isY2Axis:
            for (xPixel, yPixel), xData, text in self.y2Axis.ticks:
                if test(text):
                    vertices.append((xPixel, yPixel))
                    vertices.append((self._margins['left'], yPixel))

        return vertices

    def _buildVerticesAndLabels(self):
        self._updateAxis()

        # To fill with copy of axes lists
        vertices = []
        labels = []

        xVertices, xLabels = self.xAxis.getVerticesAndLabels()
        vertices += xVertices
        labels += xLabels

        vertices += self._x2AxisCoords

        yVertices, yLabels = self.yAxis.getVerticesAndLabels()
        vertices += yVertices
        labels += yLabels

        if self.isY2Axis:
            y2Vertices, y2Labels = self.y2Axis.getVerticesAndLabels()

            vertices += y2Vertices
            labels += y2Labels
        else:
            vertices += self._y2AxisCoords

        vertices = np.array(vertices, dtype=np.float32)

        # Add main title
        xTitle = (self.size[0] + self._margins['left'] -
                  self._margins['right']) // 2
        yTitle = self._margins['top'] - self._TICK_LENGTH_IN_PIXELS
        labels.append(Text2D(text=self.title,
                             x=xTitle,
                             y=yTitle,
                             align=CENTER,
                             valign=BOTTOM))

        # grid
        gridVertices = np.array(self._buildGridVertices(), dtype=np.float32)

        self._renderResources = (vertices, gridVertices, labels)

    _program = GLProgram(_SHADERS['vertex'], _SHADERS['fragment'])

    def render(self):
        if self._renderResources is None:
            self._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        width, height = self.size
        matProj = mat4Ortho(0, width, height, 0, 1, -1)

        glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        glLineWidth(self._LINE_WIDTH)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matProj)
        glUniform4f(prog.uniforms['color'], 0., 0., 0., 1.)
        glUniform1f(prog.uniforms['tickFactor'], 0.)

        glEnableVertexAttribArray(prog.attributes['position'])
        glVertexAttribPointer(prog.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, vertices)

        glDrawArrays(GL_LINES, 0, len(vertices))

        for label in labels:
            label.render(matProj)

    def renderGrid(self):
        if self._grid == self.GRID_NONE:
            return

        if self._renderResources is None:
            self._buildVerticesAndLabels()
        vertices, gridVertices, labels = self._renderResources

        width, height = self.size
        matProj = mat4Ortho(0, width, height, 0, 1, -1)

        glViewport(0, 0, width, height)

        prog = self._program
        prog.use()

        glLineWidth(self._LINE_WIDTH)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matProj)
        glUniform4f(prog.uniforms['color'], 0.5, 0.5, 0.5, 1.)
        glUniform1f(prog.uniforms['tickFactor'], 1/2.)  # 1/tickLen

        glEnableVertexAttribArray(prog.attributes['position'])
        glVertexAttribPointer(prog.attributes['position'],
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, gridVertices)

        glDrawArrays(GL_LINES, 0, len(gridVertices))
