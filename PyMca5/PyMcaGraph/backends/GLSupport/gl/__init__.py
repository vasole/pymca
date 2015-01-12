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
This module loads PyOpenGL
"""


# import ######################################################################

import numpy as np

import OpenGL
if 0:  # Debug
    OpenGL.ERROR_ON_COPY = True
else:
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_ON_COPY = False

from OpenGL.GL import *  # noqa

from OpenGL.GL.ARB.framebuffer_object import *  # noqa Core in OpenGL 3
from OpenGL.GL.ARB.texture_rg import GL_R32F  # noqa Core in OpenGL 3

# PyOpenGL 3.0.1 does not define it
try:
    GLchar
except NameError:
    from ctypes import c_char
    GLchar = c_char


def testGLExtensions():
    from OpenGL.GL.ARB.framebuffer_object import glInitFramebufferObjectARB
    from OpenGL.GL.ARB.texture_rg import glInitTextureRgARB

    if not glInitFramebufferObjectARB():
        raise RuntimeError(
            "OpenGL GL_ARB_framebuffer_object extension required !")

    if not glInitTextureRgARB():
        raise RuntimeError("OpenGL GL_ARB_texture_rg extension required !")


# utils #######################################################################

_GL_TYPE_SIZES = {
    GL_FLOAT: 4,
    GL_INT: 4,
    GL_UNSIGNED_BYTE: 1,
    GL_UNSIGNED_SHORT: 2,
    GL_UNSIGNED_INT: 4,
}


def sizeofGLType(type_):
    """Returns the size in bytes of an element of type type_"""
    return _GL_TYPE_SIZES[type_]


_TYPE_CONVERTER = {
    np.dtype(np.float32): GL_FLOAT,
    np.dtype(np.int32): GL_INT,
    np.dtype(np.uint8): GL_UNSIGNED_BYTE,
    np.dtype(np.uint16): GL_UNSIGNED_SHORT,
    np.dtype(np.uint32): GL_UNSIGNED_INT,
}


def numpyToGLType(type_):
    """Returns the GL type corresponding the provided numpy type or dtype"""
    return _TYPE_CONVERTER[np.dtype(type_)]
