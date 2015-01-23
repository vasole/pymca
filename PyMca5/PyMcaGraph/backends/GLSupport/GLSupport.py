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
This module provides convenient classes for the OpenGL rendering backend
"""


# import ######################################################################

from .gl import *  # noqa
from ctypes import c_float
import numpy as np
import warnings


# utils #######################################################################

def _glGetActiveAttrib(program, index):
    """Wrap PyOpenGL glGetActiveAttrib as for glGetActiveUniform
    """
    bufSize = glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
    length = GLsizei()
    size = GLint()
    type_ = GLenum()
    name = (GLchar * bufSize)()

    glGetActiveAttrib(program, index, bufSize, length, size, type_, name)
    return name.value, size.value, type_.value


def clamp(value, min_=0., max_=1.):
    return min(max(value, min_), max_)


def rgba(color, colorDict={}):
    """Convert color code '#RRGGBB' and '#RRGGBBAA' to (R, G, B, A)

    :param str code: The color code to conver
    :param dict colorDict: A dictionary of color name conversion to color code
    :returns: RGBA colors as floats in [0., 1.]
    :rtype: tuple
    """
    if len(color) == 4:
        r, g, b, a = color
        if type(color[3]) in [type(1), np.uint8, np.int8]:
            return r / 255., g / 255., b / 255., a / 255.
        if type(color[3]) in [type(1.), np.float32, np.float64]:
            assert r >= 0. and r <= 1.
            assert g >= 0. and g <= 1.
            assert b >= 0. and b <= 1.
            assert a >= 0. and a <= 1.
            return r, g, b, a

    # We assume color is a string
    if not color.startswith('#'):
        color = colorDict[color]

    assert len(color) in (7, 9) and color[0] == '#'
    r = int(color[1:3], 16) / 255.
    g = int(color[3:5], 16) / 255.
    b = int(color[5:7], 16) / 255.
    a = int(color[7:9], 16) / 255. if len(color) == 9 else 1.
    return r, g, b, a


# program #####################################################################

class Program(object):
    """Wrap shader program

    Provides access to attributes and uniforms locations
    """
    def __init__(self, vertexShaderSrc, fragmentShaderSrc):
        self._prog = glCreateProgram()

        vertexShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertexShader, vertexShaderSrc)
        glCompileShader(vertexShader)
        if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vertexShader))
        glAttachShader(self._prog, vertexShader)
        glDeleteShader(vertexShader)

        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragmentShader, fragmentShaderSrc)
        glCompileShader(fragmentShader)
        if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(fragmentShader))
        glAttachShader(self._prog, fragmentShader)
        glDeleteShader(fragmentShader)

        glLinkProgram(self._prog)
        if glGetProgramiv(self._prog, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self._prog))

        glValidateProgram(self._prog)
        if glGetProgramiv(self._prog, GL_VALIDATE_STATUS) != GL_TRUE:
            warnings.warn(
                'Cannot validate program: ' + glGetProgramInfoLog(self._prog),
                RuntimeWarning)

        self.attributes = {}
        for index in range(glGetProgramiv(self._prog, GL_ACTIVE_ATTRIBUTES)):
            name = _glGetActiveAttrib(self._prog, index)[0]
            nameStr = name.decode('ascii')
            self.attributes[nameStr] = glGetAttribLocation(self._prog, name)

        self.uniforms = {}
        for index in range(glGetProgramiv(self._prog, GL_ACTIVE_UNIFORMS)):
            name = glGetActiveUniform(self._prog, index)[0]
            nameStr = name.decode('ascii')
            self.uniforms[nameStr] = glGetUniformLocation(self._prog, name)

    @property
    def prog_id(self):
        return self._prog

    def discard(self):
        try:
            prog = self._prog
        except AttributeError:
            raise RuntimeError("No OpenGL program resource, \
                               discard has already been called")
        else:
            if bool(glDeleteProgram):  # Test for __del__
                glDeleteProgram(prog)
            del self._prog

    def __del__(self):
        self.discard()

    def use(self):
        glUseProgram(self.prog_id)


# shape2D #####################################################################

def buildFillMaskIndices(nIndices):
    if nIndices <= np.iinfo(np.uint16).max + 1:
        dtype = np.uint16
    else:
        dtype = np.uint32

    lastIndex = nIndices - 1
    splitIndex = lastIndex // 2 + 1
    indices = np.empty(nIndices, dtype=dtype)
    indices[::2] = np.arange(0, splitIndex, step=1, dtype=dtype)
    indices[1::2] = np.arange(lastIndex, splitIndex - 1, step=-1,
                              dtype=dtype)
    return indices


class Shape2D(object):
    _NO_HATCH = 0
    _HATCH_STEP = 20

    def __init__(self, points, fill='solid', stroke=True,
                 fillColor=(0., 0., 0., 1.), strokeColor=(0., 0., 0., 1.)):
        self.vertices = np.array(points, dtype=np.float32, copy=False)

        self._indices = buildFillMaskIndices(len(self.vertices))

        tVertex = np.transpose(self.vertices)
        xMin, xMax = min(tVertex[0]), max(tVertex[0])
        yMin, yMax = min(tVertex[1]), max(tVertex[1])
        self.bboxVertices = np.array(((xMin, yMin), (xMin, yMax),
                                      (xMax, yMin), (xMax, yMax)),
                                     dtype=np.float32)
        self._bounds = xMin, yMin, xMax - xMin, yMax - yMin

        self.fill = fill
        self.fillColor = fillColor
        self.stroke = stroke
        self.strokeColor = strokeColor

    def prepareFillMask(self, posAttrib):
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self.vertices)

        glEnable(GL_STENCIL_TEST)
        glStencilMask(1)
        glStencilFunc(GL_ALWAYS, 1, 1)
        glStencilOp(GL_INVERT, GL_INVERT, GL_INVERT)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)

        glDrawElements(GL_TRIANGLE_STRIP, len(self._indices),
                       GL_UNSIGNED_SHORT, self._indices)

        glStencilFunc(GL_EQUAL, 1, 1)
        glStencilOp(GL_ZERO, GL_ZERO, GL_ZERO)  # Reset stencil while drawing
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def renderFill(self, posAttrib):
        self.prepareFillMask(posAttrib)

        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self.bboxVertices)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(self.bboxVertices))

        glDisable(GL_STENCIL_TEST)

    def renderStroke(self, posAttrib):
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              0, self.vertices)
        glLineWidth(1)
        glDrawArrays(GL_LINE_LOOP, 0, len(self.vertices))

    def render(self, posAttrib, colorUnif, hatchStepUnif):
        assert self.fill in ['hatch', 'solid', None]
        if self.fill is not None:
            glUniform4f(colorUnif, *self.fillColor)
            step = self._HATCH_STEP if self.fill == 'hatch' else self._NO_HATCH
            glUniform1i(hatchStepUnif, step)
            self.renderFill(posAttrib)

        if self.stroke:
            glUniform4f(colorUnif, *self.strokeColor)
            glUniform1i(hatchStepUnif, self._NO_HATCH)
            self.renderStroke(posAttrib)


# matrix ######################################################################

def mat4Ortho(left, right, bottom, top, near, far):
    """Orthographic projection matrix (row-major)"""
    return np.matrix((
        (2./(right - left), 0., 0., -(right+left)/float(right-left)),
        (0., 2./(top - bottom), 0., -(top+bottom)/float(top-bottom)),
        (0., 0., -2./(far-near),    -(far+near)/float(far-near)),
        (0., 0., 0., 1.)), dtype=np.float32)


def mat4Translate(x=0., y=0., z=0.):
    """Translation matrix (row-major)"""
    return np.matrix((
        (1., 0., 0., x),
        (0., 1., 0., y),
        (0., 0., 1., z),
        (0., 0., 0., 1.)), dtype=np.float32)


def mat4Scale(sx=1., sy=1., sz=1.):
    """Scale matrix (row-major)"""
    return np.matrix((
        (sx, 0., 0., 0.),
        (0., sy, 0., 0.),
        (0., 0., sz, 0.),
        (0., 0., 0., 1.)), dtype=np.float32)


def mat4Identity():
    """Identity matrix"""
    return np.matrix((
        (1., 0., 0., 0.),
        (0., 1., 0., 0.),
        (0., 0., 1., 0.),
        (0., 0., 0., 1.)), dtype=np.float32)


# main ########################################################################

if __name__ == "__main__":
    import sys
    try:
        from PyQt4.QtGui import QApplication
        from PyQt4.QtOpenGL import QGLWidget
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtOpenGL import QGLWidget

    # TODO a better test example
    class Test(QGLWidget):
        _vertexShaderSrc = """
            attribute vec2 position;

            void main(void) {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

        _fragmentShaderSrc = """
            uniform vec4 color;

            void main(void) {
                gl_FragColor = color;
            }
            """

        def initializeGL(self):
            glClearColor(1., 1., 1., 0.)

            self.glProgram = Program(self._vertexShaderSrc,
                                     self._fragmentShaderSrc)
            print("Attributes: {0}".format(self.glProgram.attributes))
            print("Uniforms: {0}".format(self.glProgram.uniforms))

            self.glProgram.use()

            w, h = 128, 128
            data = (c_float * (w * h * 3))()
            for i in range(w * h):
                data[3*i] = i/float(w*h)
                data[3*i+1] = i/float(w*h)
                data[3*i+2] = i/float(w*h)

            glUniform4f(self.glProgram.uniforms['color'], 1., 0., 0., 1.)

            positions = (c_float * (4 * 2))(
                0., 0.,   1., 0.,   0., 1.,   1., 1.)
            glEnableVertexAttribArray(self.glProgram.attributes['position'])
            glVertexAttribPointer(self.glProgram.attributes['position'],
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0, positions)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        def resizeGL(self, w, h):
            glViewport(0, 0, w, h)

    app = QApplication([])
    widget = Test()
    widget.show()
    sys.exit(app.exec_())
