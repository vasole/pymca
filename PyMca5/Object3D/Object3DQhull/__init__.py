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
from __future__ import absolute_import, division, print_function, \
    unicode_literals

__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """Qhull library wrapper for Delaunay triangulation.

See http://www.qhull.org/.
Supports float32 and float64 formats depending on the Qhull library used.
"""


import numpy
import warnings

from . import _qhull64
try:
    from . import _qhull32
except ImportError:
    _qhull32 = None


__version__ = _qhull64.__version__
"""Version of the Qhull library."""


def delaunay(points, options=None, dtype=numpy.float64):
    """Delaunay triangulation using qhull library.

    :param points: Array-like set of points to triangulate.
                   Array must be of dimension 2 and of type float32 or float64.
    :param options: Options of the 'qhull d' command to run
                    or None (the default) for default flags.
                    See http://www.qhull.org/ for details.
    :type options: iterable of str.
    :param numpy.dtype dtype: The data type to use for computation.
                              numpy.float32 or numpy.float64 or None.
    :returns: Indices of corners of simplex facets.
    :rtype: numpy.ndarray of uint32 of shape (nbFacets, (points dim + 1)).
    """
    assert dtype in (None, numpy.float32, numpy.float64)

    if _qhull32 is None:
        # Force dtype if qhull float32 is not available
        if dtype == numpy.float32:
            warnings.warn('Qhull for float32 not available', RuntimeWarning)
        dtype = numpy.float64

    points = numpy.array(points, dtype=dtype, order='C', copy=False)

    if options is None:
        options = ['Qbb', 'QJ', 'Qc']
        if points.dtype == numpy.float32:
            options.append('Po')

    command = 'qhull d ' + ' '.join(options)

    if points.dtype == numpy.float32:
        return _qhull32.delaunay(points, command)
    elif points.dtype == numpy.float64:
        return _qhull64.delaunay(points, command)
    else:
        raise ValueError('Unsupported array dtype %s' % points.dtype)
