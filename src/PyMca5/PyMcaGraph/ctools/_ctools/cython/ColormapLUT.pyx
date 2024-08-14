#/*##########################################################################
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
#############################################################################*/
cimport cython
from ColormapLUT cimport fillPixmapFromDouble
from ColormapLUT cimport fillPixmapFromFloat
from ColormapLUT cimport fillPixmapFromChar
from ColormapLUT cimport fillPixmapFromUChar
from ColormapLUT cimport fillPixmapFromShort
from ColormapLUT cimport fillPixmapFromUShort
from ColormapLUT cimport fillPixmapFromInt
from ColormapLUT cimport fillPixmapFromUInt
from ColormapLUT cimport fillPixmapFromLong
from ColormapLUT cimport fillPixmapFromULong

cimport numpy
import numpy

@cython.boundscheck(False)
def fillPixmap(points, colormap, mode="linear", auto=True, minValue=0, maxValue=1):
    """
    :param points: Contiguous array with the values to be mapped
    :type points: ndarray
    :param colormap: Array of uint8 of N, 4 with the colors to assign
    :type points: ndarray
    :param mode: linear or log
    :type border: string (default linear)
    :param auto: True to calculate the min and max values of the input data
    :param minValue: If auto is false, the value to be used as minimum
    :param maxValue: If auto is false, the value to be used as maximum
    :returns pixmap: Int32 of colormap values with the same shape as points
    :returns usedMin: Used minimum value
    :returns usedMax: Used maximum value
    """
    if isinstance(points, numpy.ndarray):
        """
        if points.dtype in [numpy.float64, numpy.float]:
            f = _fillPixmapDouble
    elif points.dtype in [numpy.float32, numpy.float16]:
        f = _fillPixmapFloat
        elif points.dtype in [numpy.int32]:
        f = _fillPixmapInt
    elif points.dtype in [numpy.uint32]:
        f = _fillPixmapUInt
    elif points.dtype in [numpy.int, numpy.int64]:
        f = _fillPixmapLong
    elif points.dtype in [numpy.uint, numpy.uint64]:
        f = _fillPixmapULong
        elif points.dtype in [numpy.int16]:
        f = _fillPixmapShort
    elif points.dtype in [numpy.uint16]:
        f = _fillPixmapUShort
    elif points.dtype in [numpy.int8]:
        f = _fillPixmapByte
    elif points.dtype in [numpy.uint8]:
        f = _fillPixmapUByte
        else:
        f = _fillPixmapDouble
    """
    f = _fillPixmapDouble
    return f(points, colormap, mode, auto, minValue, maxValue)

@cython.boundscheck(False)
def _fillPixmapDouble(data, colormap, mode, auto=True, minValue=0, maxValue=1):
    cdef double[:,:] c_data = numpy.ascontiguousarray(data,
                                                      dtype=numpy.float64)
    cdef char[:, :] c_colormap = numpy.ascontiguousarray(colormap,
                                              dtype=numpy.uint8)
    cdef long n_data = c_data.size
    cdef long n_colors = c_colormap.shape[1]
    cdef short c_mode
    cdef short c_auto
    cdef double min_data
    cdef double max_data
    cdef char[:, :] pixmap = numpy.empty((n_data, 4), dtype=numpy.uint8)

    if mode.lower() == "linear":
        c_mode = 0
    else:
        c_mode = 1

    if auto:
        c_auto = 1
    else:
        c_auto = 0

    with nogil:
        fillPixmapFromDouble(&c_data[0,0], n_data, &c_colormap[0,0], n_colors,\
                &pixmap[0, 0], c_mode, c_auto, &min_data, &max_data)
    return pixmap, min_data, max_data

