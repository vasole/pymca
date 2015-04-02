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
This module provides a class to render 2D array as a colormap or RGB(A) image
"""


# import ######################################################################

from .gl import *  # noqa

import math
from .GLSupport import mat4Translate, mat4Scale
from .GLProgram import GLProgram
from .GLTexture import Image

try:
    from ....ctools import minMax
except ImportError:
    from PyMca5.PyMcaGraph.ctools import minMax


# colormap ####################################################################

class _GLPlotData2D(object):
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


class GLPlotColormap(_GLPlotData2D):

    _SHADERS = {
        'linear': {
            'vertex': """
    #version 120

    uniform mat4 matrix;
    attribute vec2 texCoords;
    attribute vec2 position;

    varying vec2 coords;

    void main(void) {
        coords = texCoords;
        gl_Position = matrix * vec4(position, 0.0, 1.0);
    }
    """,
            'fragTransform': """
    vec2 textureCoords(void) {
        return coords;
    }
    """},

        'log': {
            'vertex': """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform bvec2 isLog;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (isLog.x) {
            dataPos.x = oneOverLog10 * log(dataPos.x);
        }
        if (isLog.y) {
            dataPos.y = oneOverLog10 * log(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            'fragTransform': """
    uniform bvec2 isLog;
    uniform struct {
        vec2 oneOverRange;
        vec2 minOverRange;
    } bounds;

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (isLog.x) {
            pos.x = pow(10., coords.x);
        }
        if (isLog.y) {
            pos.y = pow(10., coords.y);
        }
        return pos * bounds.oneOverRange - bounds.minOverRange;
        // TODO texture coords in range different from [0, 1]
    }
    """},

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

    %s

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

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        float value = texture2D(data, textureCoords()).r;
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

    _linearProgram = GLProgram(_SHADERS['linear']['vertex'],
                               _SHADERS['fragment'] %
                               _SHADERS['linear']['fragTransform'])

    _logProgram = GLProgram(_SHADERS['log']['vertex'],
                            _SHADERS['fragment'] %
                            _SHADERS['log']['fragTransform'])

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
        assert data.dtype in (np.float32, np.uint16, np.uint8)

        super(GLPlotColormap, self).__init__(data, xMin, xScale, yMin, yScale)
        self.colormap = colormap
        self.cmapIsLog = cmapIsLog
        self.cmapRange = cmapRange  # Init _cmapRange and _cmapRangeIsAuto

        self._textureIsDirty = False

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture
        self._textureIsDirty = False

    @property
    def cmapRange(self):
        if self._cmapRange is None:  # Lazy computation
            self._cmapRange = minMax(self.data)
        return self._cmapRange

    @cmapRange.setter
    def cmapRange(self, cmapRange):
        self._cmapRangeIsAuto = cmapRange is None
        if cmapRange is None:
            self._cmapRange = None
        else:
            self._cmapRange = tuple(cmapRange)

    def updateData(self, data):
        oldData = self.data
        self.data = data

        if self._cmapRangeIsAuto:  # Reset cmapRange cache as data is updated
            self._cmapRange = None

        if hasattr(self, '_texture'):
            if (self.data.shape != oldData.shape or
                    self.data.dtype != oldData.dtype):
                self.discard()
            else:
                self._textureIsDirty = True

    _INTERNAL_FORMATS = {
        np.dtype(np.float32): GL_R32F,
        # Use normalized integer for unsigned int formats
        np.dtype(np.uint16): GL_R16,
        np.dtype(np.uint8): GL_R8
    }

    def prepare(self):
        if not hasattr(self, '_texture'):
            internalFormat = self._INTERNAL_FORMATS[self.data.dtype]
            height, width = self.data.shape

            self._texture = Image(internalFormat, width, height,
                                  format_=GL_RED,
                                  type_=numpyToGLType(self.data.dtype),
                                  data=self.data,
                                  texUnit=self._DATA_TEX_UNIT)
        elif self._textureIsDirty:
            self._textureIsDirty = True
            self._texture.updateAll(format_=GL_RED,
                                    type_=numpyToGLType(self.data.dtype),
                                    data=self.data)

    def _setCMap(self, prog):
        glUniform1i(prog.uniforms['cmap.id'],
                    self._SHADER_CMAP_IDS[self.colormap])

        dataMin, dataMax = self.cmapRange

        if self.data.dtype in (np.uint16, np.uint8):
            # Using unsigned int as normalized integer in OpenGL
            # So normalize range
            maxInt = float(np.iinfo(self.data.dtype).max)
            dataMin, dataMax = dataMin / maxInt, dataMax / maxInt

        if self.cmapIsLog:
            logVMin, logVMax = math.log10(dataMin), math.log10(dataMax)
            glUniform1f(prog.uniforms['cmap.logMin'], logVMin)
            glUniform1f(prog.uniforms['cmap.oneOverRange'], 0.)
            glUniform1f(prog.uniforms['cmap.oneOverLogRange'],
                        1./(logVMax - logVMin))
        else:
            glUniform1f(prog.uniforms['cmap.min'], dataMin)
            glUniform1f(prog.uniforms['cmap.oneOverRange'],
                        1./(dataMax - dataMin))

    def _renderLinear(self, matrix):
        self.prepare()

        prog = self._linearProgram
        prog.use()

        glUniform1i(prog.uniforms['data'], self._DATA_TEX_UNIT)

        mat = matrix * mat4Translate(self.xMin, self.yMin) * \
            mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        self._setCMap(prog)

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)

    def _renderLog10(self, matrix, isXLog, isYLog):
        self.prepare()

        prog = self._logProgram
        prog.use()

        glUniform1i(prog.uniforms['data'], self._DATA_TEX_UNIT)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        mat = mat4Translate(self.xMin, self.yMin) * \
            mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matOffset'], 1, GL_TRUE, mat)

        glUniform2i(prog.uniforms['isLog'], isXLog, isYLog)

        xOneOverRange = 1. / (self.xMax - self.xMin)
        yOneOverRange = 1. / (self.yMax - self.yMin)
        glUniform2f(prog.uniforms['bounds.minOverRange'],
                    self.xMin * xOneOverRange, self.yMin * yOneOverRange)
        glUniform2f(prog.uniforms['bounds.oneOverRange'],
                    xOneOverRange, yOneOverRange)

        self._setCMap(prog)

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale")

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes['position']
        stride = vertices.shape[-1] * vertices.itemsize
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, vertices)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, matrix, isXLog, isYLog):
        if any((isXLog, isYLog)):
            self._renderLog10(matrix, isXLog, isYLog)
        else:
            self._renderLinear(matrix)


# image #######################################################################

class GLPlotRGBAImage(_GLPlotData2D):

    _SHADERS = {
        'linear': {
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
    """},

        'log': {
            'vertex': """
    #version 120

    attribute vec2 position;
    uniform mat4 matrix;
    uniform mat4 matOffset;
    uniform bvec2 isLog;

    varying vec2 coords;

    const float oneOverLog10 = 0.43429448190325176;

    void main(void) {
        vec4 dataPos = matOffset * vec4(position, 0.0, 1.0);
        if (isLog.x) {
            dataPos.x = oneOverLog10 * log(dataPos.x);
        }
        if (isLog.y) {
            dataPos.y = oneOverLog10 * log(dataPos.y);
        }
        coords = dataPos.xy;
        gl_Position = matrix * dataPos;
    }
    """,
            'fragment': """
    #version 120

    uniform sampler2D tex;
    uniform bvec2 isLog;
    uniform struct {
        vec2 oneOverRange;
        vec2 minOverRange;
    } bounds;

    varying vec2 coords;

    vec2 textureCoords(void) {
        vec2 pos = coords;
        if (isLog.x) {
            pos.x = pow(10., coords.x);
        }
        if (isLog.y) {
            pos.y = pow(10., coords.y);
        }
        return pos * bounds.oneOverRange - bounds.minOverRange;
        // TODO texture coords in range different from [0, 1]
    }

    void main(void) {
        gl_FragColor = texture2D(tex, textureCoords());
    }
    """}
    }

    _DATA_TEX_UNIT = 0

    _linearProgram = GLProgram(_SHADERS['linear']['vertex'],
                               _SHADERS['linear']['fragment'])

    _logProgram = GLProgram(_SHADERS['log']['vertex'],
                            _SHADERS['log']['fragment'])

    def __init__(self, data, xMin, xScale, yMin, yScale):
        """Create a 2D RGB(A) image from data

        :param data: The 2D image data array to display
        :type data: numpy.ndarray with 3 dimensions (dtype=numpy.float32)
        :param float xMin: Min X coordinate of the data array
        :param float xScale: X scale of the data array
        :param float yMin: Min Y coordinate of the data array
        :param float yScale: Y scale of the data array
        """
        super(GLPlotRGBAImage, self).__init__(data, xMin, xScale, yMin, yScale)
        self._textureIsDirty = False

    def __del__(self):
        self.discard()

    def discard(self):
        if hasattr(self, '_texture'):
            self._texture.discard()
            del self._texture
        self._textureIsDirty = False

    def updateData(self, data):
        oldData = self.data
        self.data = data

        if hasattr(self, '_texture'):
            if self.data.shape != oldData.shape:
                self.discard()
            else:
                self._textureIsDirty = True

    def prepare(self):
        if not hasattr(self, '_texture'):
            height, width, depth = self.data.shape
            format_ = GL_RGBA if depth == 4 else GL_RGB
            type_ = numpyToGLType(self.data.dtype)

            self._texture = Image(format_, width, height,
                                  format_=format_, type_=type_,
                                  data=self.data,
                                  texUnit=self._DATA_TEX_UNIT)
        elif self._textureIsDirty:
            self._textureIsDirty = False

            # We should check that internal format is the same
            format_ = GL_RGBA if self.data.shape[2] == 4 else GL_RGB
            type_ = numpyToGLType(self.data.dtype)
            self._texture.updateAll(format_=format_, type_=type_,
                                    data=self.data)

    def _renderLinear(self, matrix):
        self.prepare()

        prog = self._linearProgram
        prog.use()

        glUniform1i(prog.uniforms['tex'], self._DATA_TEX_UNIT)

        mat = matrix * mat4Translate(self.xMin, self.yMin)
        mat *= mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, mat)

        self._texture.render(prog.attributes['position'],
                             prog.attributes['texCoords'],
                             self._DATA_TEX_UNIT)

    def _renderLog(self, matrix, isXLog, isYLog):
        self.prepare()

        prog = self._logProgram
        prog.use()

        glUniform1i(prog.uniforms['tex'], self._DATA_TEX_UNIT)

        glUniformMatrix4fv(prog.uniforms['matrix'], 1, GL_TRUE, matrix)
        mat = mat4Translate(self.xMin, self.yMin) * \
            mat4Scale(self.xScale, self.yScale)
        glUniformMatrix4fv(prog.uniforms['matOffset'], 1, GL_TRUE, mat)

        glUniform2i(prog.uniforms['isLog'], isXLog, isYLog)

        xOneOverRange = 1. / (self.xMax - self.xMin)
        yOneOverRange = 1. / (self.yMax - self.yMin)
        glUniform2f(prog.uniforms['bounds.minOverRange'],
                    self.xMin * xOneOverRange, self.yMin * yOneOverRange)
        glUniform2f(prog.uniforms['bounds.oneOverRange'],
                    xOneOverRange, yOneOverRange)

        try:
            tiles = self._texture.tiles
        except AttributeError:
            raise RuntimeError("No texture, discard has already been called")
        if len(tiles) > 1:
            raise NotImplementedError(
                "Image over multiple textures not supported with log scale")

        texture, vertices, info = tiles[0]

        texture.bind(self._DATA_TEX_UNIT)

        posAttrib = prog.attributes['position']
        stride = vertices.shape[-1] * vertices.itemsize
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              stride, vertices)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(vertices))

    def render(self, matrix, isXLog, isYLog):
        if any((isXLog, isYLog)):
            self._renderLog(matrix, isXLog, isYLog)
        else:
            self._renderLinear(matrix)
