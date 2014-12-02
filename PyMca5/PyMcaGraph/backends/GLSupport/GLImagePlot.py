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
This module provides a class to render 2D array as a colormap or RGB(A) image
"""


# import ######################################################################

from OpenGL.GL import *  # noqa
from OpenGL.GL.ARB.texture_rg import GL_R32F  # Core in OpenGL 3

import math
import numpy as np
from .GLContext import getGLContext
from .GLSupport import Program, mat4Translate, mat4Scale
from .GLTexture import Image

try:
    from ....ctools import minMax
except ImportError:
    from PyMca5.PyMcaGraph.ctools import minMax


# colormap ####################################################################

class _GL2DDataPlot(object):
    def __init__(self, data, xMin, xScale, yMin, yScale):
        self.data = data
        self.xMin = xMin
        self.xScale = xScale
        self.yMin = yMin
        self.yScale = yScale

    def pick(self, x, y):
        if self.xMin <= x and x <= self.xMax and \
           self.yMin <= y and y <= self.yMax:
            col = int((x - self.xMin) / self.xScale)
            row = int((y - self.yMin) / self.yScale)
            return col, row
        else:
            return None

    @property
    def xMax(self):
        return self.xMin + self.xScale * self.data.shape[1]

    @property
    def yMax(self):
        return self.yMin + self.yScale * self.data.shape[0]

    def discard(self):
        pass

    def prepare(self):
        pass

    def render(self, matrix):
        pass


class GLColormap(_GL2DDataPlot):

    _SHADERS = {
        'vertex': """
    #version 120

    attribute vec2 position;
    attribute vec2 texCoords;
    uniform mat4 matrix;

    varying vec2 coords;

    void main(void) {
        coords = texCoords;
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
        """,
        'fragment': """
    #version 120

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

    const float oneOverLog10 = 1./log(10.);

    void main(void) {
        float value = texture2D(data, coords).r;
        if (cmap.oneOverRange != 0.) {
            value = clamp(cmap.oneOverRange * (value - cmap.min), 0., 1.);
        } else {
            value = clamp(cmap.oneOverLogRange *
                          (oneOverLog10 * log(value) - cmap.logMin),
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
    }

    _SHADER_CMAP_IDS = {
        'gray': 0,
        'reversed gray': 1,
        'red': 2,
        'green': 3,
        'blue': 4,
        'temperature': 5
    }

    COLORMAPS = tuple(_SHADER_CMAP_IDS.keys())

    _DATA_TEX_UNIT = 0

    _programs = {}

    def __init__(self, data, xMin, xScale, yMin, yScale,
                 colormap, cmapIsLog=False, cmapRange=None):
        """Create a 2D colormap

        :param data: The 2D scalar data array to display
        :type data: numpy.ndarray with 2 dimensions (dtype=numpy.float32)
        :param float xMin: Min X coordinate of the data array
        :param float xScale: X scale of the data array
        :param float yMin: Min Y coordinate of the data array
        :param float yScale: Y scale of the data array
        :param str colormap: Name of the colormap to use
            TODO: Accept a 1D scalar array as the colormap
        :param bool cmapIsLog: If True, uses log10 of the data value
        :param cmapRange: The range of colormap or None for autoscale colormap
            For logarithmic colormap, the range is in the untransformed data
            TODO: check consistency with matplotlib
        :type cmapRange: (float, float) or None
        """
        super(GLColormap, self).__init__(data, xMin, xScale, yMin, yScale)
        self.colormap = colormap
        self.cmapIsLog = cmapIsLog
        self.cmapRangeIsAuto = cmapRange is None
        if cmapRange is not None:
            self.cmapRange = cmapRange
        else:
            self.cmapRange = minMax(data)

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture

    def updateData(self, data):
        oldData = self.data
        self.data = data

        if self.cmapRangeIsAuto:
            self.cmapRange = minMax(data)

        if hasattr(self, '_texture'):
            if self.data.shape != oldData.shape:
                self.discard()
            else:
                self._texture.updateAll(format_=GL_RED, type_=GL_FLOAT,
                                        data=data)

    @classmethod
    def _getProgram(cls):
        context = getGLContext()
        try:
            prog = cls._programs[context]
        except KeyError:
            prog = Program(cls._SHADERS['vertex'], cls._SHADERS['fragment'])

            # Done once forever for each program
            glUniform1i(prog.uniforms['data'], cls._DATA_TEX_UNIT)

            cls._programs[context] = prog
        return prog

    def prepare(self):
        if not hasattr(self, '_texture'):
            height, width = self.data.shape
            self._texture = Image(GL_R32F, width, height,
                                  format_=GL_RED, type_=GL_FLOAT,
                                  data=self.data, texUnit=self._DATA_TEX_UNIT)

    def render(self, matrix):
        self.prepare()

        prog = self._getProgram()
        prog.use()

        mat = matrix * mat4Translate(self.xMin, self.yMin)
        mat *= mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        glUniform1i(prog.uniforms['cmap.id'],
                    self._SHADER_CMAP_IDS[self.colormap])

        dataMin, dataMax = self.cmapRange
        if self.cmapIsLog:
            logVMin = math.log10(dataMin)
            glUniform1f(prog.uniforms['cmap.logMin'], logVMin)
            glUniform1f(prog.uniforms['cmap.oneOverRange'], 0.)
            glUniform1f(prog.uniforms['cmap.oneOverLogRange'],
                        1./(math.log10(dataMax) - logVMin))
        else:
            glUniform1f(prog.uniforms['cmap.min'], dataMin)
            glUniform1f(prog.uniforms['cmap.oneOverRange'],
                        1./(dataMax - dataMin))

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)


# image #######################################################################

class GLRGBAImage(_GL2DDataPlot):

    _SHADERS = {
        'vertex': """
    #version 120

   attribute vec2 position;
   attribute vec2 texCoords;
   uniform mat4 matrix;

   varying vec2 coords;

   void main(void) {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        coords = texCoords;
   }
   """,
        'fragment': """
    #version 120

    uniform sampler2D tex;

    varying vec2 coords;

    void main(void) {
        gl_FragColor = texture2D(tex, coords);
    }
    """
    }

    _DATA_TEX_UNIT = 0

    _programs = {}

    def __init__(self, data, xMin, xScale, yMin, yScale):
        """Create a 2D RGB(A) image from data

        :param data: The 2D image data array to display
        :type data: numpy.ndarray with 3 dimensions (dtype=numpy.float32)
        :param float xMin: Min X coordinate of the data array
        :param float xScale: X scale of the data array
        :param float yMin: Min Y coordinate of the data array
        :param float yScale: Y scale of the data array
        """
        super(GLRGBAImage, self).__init__(data, xMin, xScale, yMin, yScale)

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture

    @classmethod
    def _getProgram(cls):
        context = getGLContext()
        try:
            prog = cls._programs[context]
        except KeyError:
            prog = Program(cls._SHADERS['vertex'], cls._SHADERS['fragment'])

            # Done once forever for each program
            glUniform1i(prog.uniforms['tex'], cls._DATA_TEX_UNIT)

            cls._programs[context] = prog
        return prog

    def _convertNPToGLType(self, numpyType):
        if numpyType == np.uint8:
            return GL_UNSIGNED_BYTE
        elif numpyType == np.float32:
            return GL_FLOAT
        else:
            raise NotImplementedError(
                "Data type not supported {}".format(self.data.dtype))

    def updateData(self, data):
        oldData = self.data
        self.data = data

        if hasattr(self, '_texture'):
            if self.data.shape != oldData.shape:
                self.discard()
            else:
                # We should check that internal format is the same
                format_ = GL_RGBA if data.shape[2] == 4 else GL_RGB
                type_ = self._convertNPToGLType(data.dtype)

                self._texture.updateAll(format_=format_, type_=type_,
                                        data=data)

    def prepare(self):
        if not hasattr(self, '_texture'):
            height, width, depth = self.data.shape
            format_ = GL_RGBA if depth == 4 else GL_RGB
            type_ = self._convertNPToGLType(self.data.dtype)

            self._texture = Image(format_, width, height,
                                  format_=format_, type_=type_,
                                  data=self.data,
                                  texUnit=self._DATA_TEX_UNIT)

    def render(self, matrix):
        self.prepare()

        prog = self._getProgram()
        prog.use()

        mat = matrix * mat4Translate(self.xMin, self.yMin)
        mat *= mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)
