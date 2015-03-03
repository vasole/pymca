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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from InsidePolygonWithBounds cimport PointsInsidePolygon as _pnpoly
cimport numpy
import numpy

def pnpoly(vertices, points, bint border=True):
    """
    :param vertices: Array Nx2 with the coordenates of the polygon vertices
    :type vertices: ndarray
    :param points: Points to be checked out.
    :type points: ndarray Nx2 or list of [x, y] pairs
    :param border: Flag to indicate if a pointon a vertex is to be in or out
    :type border: boolean (default True)
    """
    cdef double[:,:] c_vertices = numpy.ascontiguousarray(vertices,
                                                          dtype=numpy.float64)
    cdef int n_vertices = c_vertices.shape[0]
    assert c_vertices.shape[1] == 2
    cdef double[:,:] c_points = numpy.ascontiguousarray(points,
                                                        dtype=numpy.float64)
    cdef int n_points = c_points.shape[0]
    assert c_points.shape[1] == 2

    cdef numpy.ndarray[numpy.uint8_t, ndim=2] mask = \
         numpy.zeros(c_vertices.shape, dtypes=numpy.uint8)

    _pnpoly(&c_vertices[0,0], n_vertices, &c_points[0,0], n_points,
            border, &mask[0, 0])
    return mask
