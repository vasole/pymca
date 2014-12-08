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
This module provides a class managing a vertex buffer
"""


# import ######################################################################

from OpenGL.GL import *  # noqa
from ctypes import c_void_p, c_int
import numpy as np


# VBO #########################################################################

class VertexBuffer(object):
    def __init__(self, data=None, sizeInBytes=None,
                 usage=None):
        if usage is None:
            usage = GL_STATIC_DRAW

        self._vboId = glGenBuffers(1)
        self.bind()
        if data is None:
            assert sizeInBytes is not None
            self._size = sizeInBytes
            glBufferData(GL_ARRAY_BUFFER,
                         self._size,
                         c_void_p(0),
                         usage)
        else:
            assert isinstance(data, np.ndarray) and data.flags['C_CONTIGUOUS']
            if sizeInBytes is not None:
                assert sizeInBytes <= data.nbytes

            self._size = sizeInBytes or data.nbytes
            glBufferData(GL_ARRAY_BUFFER,
                         self._size,
                         data,
                         usage)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    @property
    def vboId(self):
        """OpenGL Vertex Buffer Object ID
        :type: int
        """
        try:
            return self._vboId
        except AttributeError:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    @property
    def size(self):
        """Size in bytes of the Vertex Buffer Object
        :type: int
        """
        try:
            return self._size
        except AttributeError:
            raise RuntimeError("No OpenGL buffer resource, \
                               discard has already been called")

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vboId)

    def update(self, data, offsetInBytes=0, sizeInBytes=None):
        assert isinstance(data, np.ndarray) and data.flags['C_CONTIGUOUS']
        if sizeInBytes is None:
            sizeInBytes = data.nbytes
        assert offsetInBytes + sizeInBytes <= self.size
        with self:
            glBufferSubData(GL_ARRAY_BUFFER, offsetInBytes, sizeInBytes, data)

    def discard(self):
        if hasattr(self, '_vboId'):
            if bool(glDeleteBuffers):  # Test for __del__
                glDeleteBuffers(1, (c_int * 1)(self._vboId))
            del self._vboId
            del self._size

    def __del__(self):
        self.discard()

    # with statement

    def __enter__(self):
        self.bind()

    def __exit__(self, excType, excValue, traceback):
        glBindBuffer(GL_ARRAY_BUFFER, 0)


# VBOAttrib ###################################################################

class VBOAttrib(object):
    """Describes data stored in a VBO
    """

    _GL_TYPES = GL_FLOAT, GL_INT

    def __init__(self, vbo, type_,
                 size, dimension=1,
                 offset=0, stride=0):
        """
        :param VertexBuffer vbo: The VBO storing the data
        :param int type_: The OpenGL type of the data
        :param int size: The number of data elements stored in the VBO
        :param int dimension: The number of type_  in [1, 4]
        :param int offset: Start offset of data in the VBO
        :param int stride: Data stride in the VBO
        """
        self.vbo = vbo
        assert type_ in self._GL_TYPES
        self.type_ = type_
        self.size = size
        assert dimension >= 1 and dimension <= 4
        self.dimension = dimension
        self.offset = offset
        self.stride = stride

    def setVertexAttrib(self, attrib):
        with self.vbo:
            glVertexAttribPointer(attrib,
                                  self.dimension,
                                  self.type_,
                                  GL_FALSE,
                                  self.stride,
                                  c_void_p(self.offset))


def convertNumpyToGLType(type_):
    if type_ == np.float32:
        return GL_FLOAT
    else:
        raise RuntimeError("Cannot convert dtype {} to GL type".format(type_))

_GL_TYPE_SIZES = {
    GL_FLOAT: 4,
    GL_INT: 4
}


def createVBOFromArrays(arrays, usage=None):
    """
    Create a single VBO from multiple 1D or 2D numpy arrays
    :param arrays: Arrays of data to store
    :type arrays: An iterable of numpy.ndarray
    :param int usage: VBO usage hint or None for default
    :returns: A list of VBOAttrib objects sharing the same VBO
    """
    arraysInfo = []
    vboSize = 0
    for data in arrays:
        shape = data.shape
        assert len(shape) <= 2
        type_ = convertNumpyToGLType(data.dtype)
        size = shape[0]
        dimension = 1 if len(shape) == 1 else shape[1]
        sizeInBytes = size * dimension * _GL_TYPE_SIZES[type_]
        sizeInBytes = 4 * (((sizeInBytes) + 3) >> 2)  # 4 bytes alignment
        arraysInfo.append((data, type_, size, dimension, vboSize, sizeInBytes))
        vboSize += sizeInBytes

    vbo = VertexBuffer(sizeInBytes=vboSize, usage=usage)

    result = []
    for data, type_, size, dimension, offset, sizeInBytes in arraysInfo:
        vbo.update(data, offsetInBytes=offset, sizeInBytes=sizeInBytes)
        result.append(VBOAttrib(vbo, type_, size, dimension, offset, 0))
    return result
