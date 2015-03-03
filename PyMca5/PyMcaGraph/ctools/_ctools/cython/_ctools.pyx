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
from InsidePolygonWithBounds cimport PointsInsidePolygon as _pnpoly
from InsidePolygonWithBounds cimport PointsInsidePolygonF as _pnpolyf
from InsidePolygonWithBounds cimport PointsInsidePolygonInt as _pnpolyInt32
cimport numpy
import numpy

# From numpy_common.pxi to avoid warnings while compiling C code
# See this thread:
# https://mail.python.org/pipermail//cython-devel/2012-March/002137.html
cdef extern from *:
    bint FALSE "0"
    void import_array()
    void import_umath()

if FALSE:
    import_array()
    import_umath()


include "MinMax.pyx"

include "Colormap.pyx"

@cython.boundscheck(False)
def pnpoly(vertices, points, bint border=True):
    """
    :param vertices: Array Nx2 with the coordenates of the polygon vertices
    :type vertices: ndarray
    :param points: Points to be checked out.
    :type points: ndarray Nx2 or list of [x, y] pairs
    :param border: Flag to indicate if a pointon a vertex is to be in or out
    :type border: boolean (default True)
    """
    if isinstance(points, numpy.ndarray):
        if points.dtype == numpy.float32:
            return _pnpolyFloat(vertices, points, border)
        elif points.dtype in [numpy.int32, numpy.int8, numpy.int16,
                              numpy.uint32, numpy.uint8, numpy.uint16]:
            return _pnpolyInt(vertices, points, border)
    return _pnpolyd(vertices, points, border)

@cython.boundscheck(False)
def _pnpolyd(vertices, points, bint border=True):
    cdef double[:,:] c_vertices = numpy.ascontiguousarray(vertices,
                                                          dtype=numpy.float64)
    cdef int n_vertices = c_vertices.shape[0]
    assert c_vertices.shape[1] == 2
    cdef double[:,:] c_points = numpy.ascontiguousarray(points,
                                                        dtype=numpy.float64)
    cdef int n_points = c_points.shape[0]
    assert c_points.shape[1] == 2

    cdef numpy.ndarray[numpy.uint8_t, ndim=1] mask = \
         numpy.zeros((n_points, ), dtype=numpy.uint8)
    with nogil:
        _pnpoly(&c_vertices[0,0], n_vertices, &c_points[0,0], n_points,
                 border, &mask[0])
    return mask

@cython.boundscheck(False)
def _pnpolyFloat(vertices, points, bint border=True):
    cdef double[:,:] c_vertices = numpy.ascontiguousarray(vertices,
                                                          dtype=numpy.float64)
    cdef int n_vertices = c_vertices.shape[0]
    assert c_vertices.shape[1] == 2
    cdef float[:,:] c_points = numpy.ascontiguousarray(points,
                                                        dtype=numpy.float32)
    cdef int n_points = c_points.shape[0]
    assert c_points.shape[1] == 2

    cdef numpy.ndarray[numpy.uint8_t, ndim=1] mask = \
         numpy.zeros((n_points, ), dtype=numpy.uint8)
    with nogil:
        _pnpolyf(&c_vertices[0,0], n_vertices, &c_points[0,0], n_points,
                 border, &mask[0])
    return mask

@cython.boundscheck(False)
def _pnpolyInt(vertices, points, bint border=True):
    cdef double[:,:] c_vertices = numpy.ascontiguousarray(vertices,
                                                          dtype=numpy.float64)
    cdef int n_vertices = c_vertices.shape[0]
    assert c_vertices.shape[1] == 2
    cdef int[:,:] c_points = numpy.ascontiguousarray(points,
                                                        dtype=numpy.int32)
    cdef int n_points = c_points.shape[0]
    assert c_points.shape[1] == 2

    cdef numpy.ndarray[numpy.uint8_t, ndim=1] mask = \
         numpy.zeros((n_points, ), dtype=numpy.uint8)
    with nogil:
        _pnpolyInt32(&c_vertices[0,0], n_vertices, &c_points[0,0], n_points,
                 border, &mask[0])
    return mask
