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
from __future__ import with_statement

__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module provides minimalistic text support for OpenGL.
It provides Latin-1 (ISO8859-1) characters for one monospace font at one size.
"""


# import ######################################################################
import numpy as np
import math
from ctypes import c_void_p, sizeof, c_float
from OpenGL.GL import *  # noqa
from . import FontLatin1_12 as font

# TODO: Font should be configurable by the main program


# Text2D ######################################################################

LEFT, CENTER, RIGHT = 'left', 'center', 'right'
TOP, BASELINE, BOTTOM = 'top', 'baseline', 'bottom'
ROTATE_90, ROTATE_180, ROTATE_270 = 90, 180, 270


class Text2D(object):
    _textures = {}

    def __init__(self, text, align=LEFT, valign=BASELINE,
                 rotate=0):
        self._text = text

        if align not in (LEFT, CENTER, RIGHT):
            raise RuntimeError(
                "Horizontal alignment not supported: {0}".format(align))
        self._align = align

        if valign not in (TOP, CENTER, BASELINE, BOTTOM):
            raise RuntimeError(
                "Vertical alignment not supported: {0}".format(valign))
        self._valign = valign

        self._rotate = rotate

    @classmethod
    def _getTexture(cls, context):
        # Loaded once for all Text2D instances per OpenGL context
        # TODO proper support of multiple contexts
        return cls._textures.setdefault(context, font.loadTexture())

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        if self._text != text:
            del self._vertices
            self._text = text

    def getSize(self):
        return len(self._text) * font.cWidth, font.cHeight

    def getVertices(self):
        try:
            return self._vertices
        except AttributeError:
            self._vertices = np.empty((len(self.text), 4, 4), dtype='float32')

            if self._align == LEFT:
                xOrig = 0
            elif self._align == RIGHT:
                xOrig = - len(self._text) * font.cWidth
            else:  # CENTER
                xOrig = - (len(self._text) * font.cWidth) // 2

            if self._valign == BASELINE:
                yOrig = - font.bearingY
            elif self._valign == TOP:
                yOrig = 0
            elif self._valign == BOTTOM:
                yOrig = - font.cHeight
            else:  # CENTER
                yOrig = - font.cHeight // 2

            for index, char in enumerate(self.text):
                uMin, vMin, uMax, vMax = font.charTexCoords(char)
                vertices = ((xOrig + index * font.cWidth, yOrig + font.cHeight,
                             uMin, vMax),
                            (xOrig + index * font.cWidth, yOrig, uMin, vMin),
                            (xOrig + (index + 1) * font.cWidth,
                             yOrig + font.cHeight, uMax, vMax),
                            (xOrig + (index + 1) * font.cWidth, yOrig,
                             uMax, vMin))

                rotate = math.radians(self._rotate)
                cos, sin = math.cos(rotate), math.sin(rotate)
                self._vertices[index] = [
                    (cos * x - sin * y, sin * x + cos * y, u, v)
                    for x, y, u, v in vertices]

            return self._vertices

    def getStride(self):
        vertices = self.getVertices()
        return vertices.shape[-1] * vertices.itemsize

    def render(self, context, posAttrib, texAttrib, texUnit=0):
        self._getTexture(context).bind(texUnit)
        glEnableVertexAttribArray(posAttrib)
        glVertexAttribPointer(posAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              self.getStride(), self.getVertices())
        glEnableVertexAttribArray(texAttrib)
        glVertexAttribPointer(texAttrib,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              self.getStride(),
                              c_void_p(self.getVertices().ctypes.data +
                                       2 * sizeof(c_float))  # Other way?
                              )
        nbChar, nbVert, _ = self.getVertices().shape
        glDrawArrays(GL_TRIANGLE_STRIP, 0, nbChar * nbVert)

        glBindTexture(GL_TEXTURE_2D, 0)


# main ########################################################################

if __name__ == "__main__":
    import sys

    try:
        from PyQt4.QtGui import QApplication
        from PyQt4.QtOpenGL import QGLWidget
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtOpenGL import QGLWidget

    from .GLSupport import Program, mat4Ortho, mat4Translate

    class TestText(QGLWidget):
        _vertexShaderSrc = """
            uniform mat4 transform;
            attribute vec2 position;
            attribute vec2 texCoords;
            varying vec2 coords;

            void main(void) {
                gl_Position = transform * vec4(position, 0.0, 1.0);
                coords = texCoords;
            }
            """

        _fragmentShaderSrc = """
            uniform sampler2D texture;
            uniform vec4 color;
            varying vec2 coords;

            void main(void) {
                gl_FragColor = vec4(color.rgb,
                    color.a * texture2D(texture, coords).a);
            }
            """

        def __init__(self, parent=None):
            QGLWidget.__init__(self, parent)

        def initializeGL(self):
            glClearColor(1., 1., 1., 1.)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.prog = Program(self._vertexShaderSrc, self._fragmentShaderSrc)
            self.prog.use()

            self.matScreenProj = np.matrix((
                (1., 0., 0., 0.),
                (0., 1., 0., 0.),
                (0., 0., 1., 0.),
                (0., 0., 0., 1.)),
                dtype=np.float32)

            self.texUnit = 0
            glUniform1i(self.prog.uniforms['texture'], self.texUnit)

            self.lines = np.array((
                (100, 0), (100, 1000),
                (220, 0), (220, 1000),
                (340, 0), (340, 1000),
                (460, 0), (460, 1000),

                (10, 25), (550, 25),
                (10, 50), (550, 50),
                (10, 75), (550, 75),

                (10, 160), (550, 160),
                (10, 260), (550, 260),
                (10, 360), (550, 360),
                ), dtype=np.float32)
            self.texts = [
                (100, 25, Text2D('left_top', align=LEFT, valign=TOP)),
                (100, 50, Text2D('center_top', align=CENTER, valign=TOP)),
                (100, 75, Text2D('right_top', align=RIGHT, valign=TOP)),

                (220, 25, Text2D('left_center', align=LEFT, valign=CENTER)),
                (220, 50, Text2D('center_center', align=CENTER,
                                 valign=CENTER)),
                (220, 75, Text2D('right_center', align=RIGHT, valign=CENTER)),

                (340, 25, Text2D('left_baseline', align=LEFT,
                                 valign=BASELINE)),
                (340, 50, Text2D('center_baseline', align=CENTER,
                                 valign=BASELINE)),
                (340, 75, Text2D('right_baseline', align=RIGHT,
                                 valign=BASELINE)),

                (460, 25, Text2D('left_bottom', align=LEFT, valign=BOTTOM)),
                (460, 50, Text2D('center_bottom', align=CENTER,
                                 valign=BOTTOM)),
                (460, 75, Text2D('right_bottom', align=RIGHT, valign=BOTTOM)),

                (100, 160, Text2D('center_90', align=CENTER, valign=CENTER,
                                  rotate=ROTATE_90)),
                (220, 160, Text2D('center_180', align=CENTER, valign=CENTER,
                                  rotate=ROTATE_180)),
                (340, 160, Text2D('ctr_270', align=CENTER, valign=CENTER,
                                  rotate=ROTATE_270)),
                (460, 160, Text2D('center_45', align=CENTER, valign=CENTER,
                                  rotate=45)),

                (100, 260, Text2D('left_90', align=LEFT, valign=CENTER,
                                  rotate=ROTATE_90)),
                (220, 260, Text2D('left_180', align=LEFT, valign=CENTER,
                                  rotate=ROTATE_180)),
                (340, 260, Text2D('left_270', align=LEFT, valign=CENTER,
                                  rotate=ROTATE_270)),
                (460, 260, Text2D('left_45', align=LEFT, valign=CENTER,
                                  rotate=45)),

                (100, 260, Text2D('r_90', align=RIGHT, valign=CENTER,
                                  rotate=ROTATE_90)),
                (220, 260, Text2D('r_180', align=RIGHT, valign=CENTER,
                                  rotate=ROTATE_180)),
                (340, 260, Text2D('r_270', align=RIGHT, valign=CENTER,
                                  rotate=ROTATE_270)),
                (460, 260, Text2D('r_45', align=RIGHT, valign=CENTER,
                                  rotate=45)),

                (100, 360, Text2D('l_top_90', align=LEFT, valign=TOP,
                                  rotate=ROTATE_90)),
                (220, 360, Text2D('l_top_180', align=LEFT, valign=TOP,
                                  rotate=ROTATE_180)),
                (340, 360, Text2D('l_top', align=LEFT, valign=TOP,
                                  rotate=ROTATE_270)),
                (460, 360, Text2D('l_top_45', align=LEFT, valign=TOP,
                                  rotate=45)),

                (100, 360, Text2D('r_btm', align=RIGHT, valign=BOTTOM,
                                  rotate=ROTATE_90)),
                (220, 360, Text2D('r_btm_180', align=RIGHT, valign=BOTTOM,
                                  rotate=ROTATE_180)),
                (340, 360, Text2D('r_btm_270', align=RIGHT, valign=BOTTOM,
                                  rotate=ROTATE_270)),
                (460, 360, Text2D('r_btm_45', align=RIGHT, valign=BOTTOM,
                                  rotate=45)),
            ]

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT)

            glUniformMatrix4fv(self.prog.uniforms['transform'], 1, GL_TRUE,
                               self.matScreenProj)
            glUniform4f(self.prog.uniforms['color'], 1., 0., 0., 1.)
            glLineWidth(1.)
            glEnableVertexAttribArray(self.prog.attributes['position'])
            glVertexAttribPointer(self.prog.attributes['position'],
                                  2,
                                  GL_FLOAT,
                                  GL_FALSE,
                                  0, self.lines)
            glDisableVertexAttribArray(self.prog.attributes['texCoords'])
            glVertexAttrib2f(self.prog.attributes['texCoords'], 0., 0.)

            # Hack to avoiding using another shader
            glDisable(GL_BLEND)
            glDrawArrays(GL_LINES, 0, len(self.lines))
            glEnable(GL_BLEND)

            glUniform4f(self.prog.uniforms['color'], 0., 0., 0., 1.)
            for x, y, text in self.texts:
                glUniformMatrix4fv(self.prog.uniforms['transform'], 1, GL_TRUE,
                                   self.matScreenProj * mat4Translate(x, y, 0))
                text.render(self.context(),
                            self.prog.attributes['position'],
                            self.prog.attributes['texCoords'],
                            self.texUnit)

        def resizeGL(self, w, h):
            glViewport(0, 0, w, h)
            self.matScreenProj = mat4Ortho(0, w, h, 0, 1, -1)

    app = QApplication([])
    widget1 = TestText()
    widget1.show()
    widget2 = TestText()
    widget2.show()
    sys.exit(app.exec_())
