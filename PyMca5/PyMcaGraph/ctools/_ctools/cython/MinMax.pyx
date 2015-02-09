#/*##########################################################################
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
#############################################################################*/
cimport cython
cimport numpy
import numpy

#from libc.stdint cimport int8_t, uint8_t
#from libc.stdint cimport int16_t, uint16_t
#from libc.stdint cimport int32_t, uint32_t
#from libc.stdint cimport int64_t, uint64_t

from MinMax cimport *

def _minMaxFloat(numpy.ndarray data):
    cdef float[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef float minValue
    cdef float maxValue
    getMinMaxFloat(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxDouble(numpy.ndarray data):
    cdef double[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef double minValue
    cdef double maxValue
    getMinMaxDouble(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxInt8(numpy.ndarray data):
    cdef char[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef char minValue
    cdef char maxValue
    getMinMaxInt8(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxUInt8(numpy.ndarray data):
    cdef unsigned char[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef unsigned char minValue
    cdef unsigned char maxValue
    getMinMaxUInt8(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


_minMaxFunctions = {
    numpy.dtype('float32'): _minMaxFloat,
    numpy.dtype('float64'): _minMaxDouble,
    numpy.dtype('int8'): _minMaxInt8,
    numpy.dtype('uint8'): _minMaxUInt8,
}


@cython.boundscheck(False)
@cython.wraparound(False)
def minMax(numpy.ndarray data):
    try:
        minMaxFunc = _minMaxFunctions[data.dtype]
    except KeyError:
        raise NotImplementedError(
            "Unsupported numpy.ndarray dtype: {}".format(data.dtype))
    return minMaxFunc(numpy.ravel(data, order='C'))
