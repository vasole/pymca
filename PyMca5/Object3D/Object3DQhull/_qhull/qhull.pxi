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

import cython
import numpy
cimport numpy
import threading
cimport qhull as _qhull


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


# Version string
__version__ = _qhull.qh_version


_errors = {
    _qhull.qh_ERRinput: RuntimeError('qhull input inconsistency'),
    _qhull.qh_ERRsingular: ArithmeticError('qhull singluar input data'),
    _qhull.qh_ERRprec: ArithmeticError('qhull precision error'),
    _qhull.qh_ERRmem: MemoryError('qhull insufficient memory'),
    _qhull.qh_ERRqhull: RuntimeError('qhull internal error'),
}


cdef unsigned int nbDelaunayFacets():
    # Corresponding C code:
    # unsigned int get_num_delaunay_facets(void) {
    #     unsigned int nbFacets = 0;
    #     facetT * facet;
    #     FORALLfacets {
    #        if (!(facet->upperdelaunay)) {
    #            nbFacets++;
    #        }
    #    }
    #    return nbFacets;
    # }
    cdef _qhull.facetT * facet
    cdef unsigned int nbFacets = 0

    # FORALLfacets
    facet = _qhull.qh_qh.facet_list
    while facet != NULL and facet.next != NULL:
        if not facet.upperdelaunay:
            nbFacets += 1
        facet = facet.next

    return nbFacets


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void setDelaunayIndices(unsigned int[:] c_indices):
    # Corresponding C code
    # void set_delaunay_indices(unsigned int * indices) {
    #     facetT * facet;
    #     vertexT * vertex, **vertexp;
    #     FORALLfacets {
    #         if (!facet->upperdelaunay) {
    #             FOREACHvertex_(facet->vertices) {
    #                *(indices++) = qh_pointid(vertex->point);
    #             }
    #         }
    #     }
    # }
    cdef _qhull.facetT * facet
    cdef _qhull.vertexT * vertex
    cdef unsigned int indicesIndex = 0
    cdef unsigned int setIndex

    # FORALLfacets    
    facet = _qhull.qh_qh.facet_list
    while facet != NULL and facet.next != NULL:
        if not facet.upperdelaunay:
            # FOREACHvertex_
            if facet.vertices != NULL:
                for setIndex in range(facet.vertices.maxsize):
                    vertex = <_qhull.vertexT *>(facet.vertices.e[setIndex].p)
                    if vertex == NULL:
                        break
                    c_indices[indicesIndex] = _qhull.qh_pointid(vertex.point)
                    indicesIndex += 1
        facet = facet.next

# Avoid concurrent use of the qhull library
_qhull_lock = threading.Lock()

@cython.boundscheck(False)
@cython.wraparound(False)
def delaunay(numpy.ndarray[_qhull.realT, ndim=2, mode='c'] points, command):
    """Delaunay triangulation using qhull library.

    :param points: numpy.ndarray of points to triangulate.
                   Array must be of dimension 2.
                   The second dimension being the dimension of the space.
    :param str command: 'qhull d' command to run
    :returns: Index of simplex facets corners.
    :rtype: numpy.ndarray of uint32 of dimension: nbFacets x (points dim + 1).
    """
    cdef int dimension = points.shape[1]
    cdef _qhull.coordT[:] c_points = numpy.ravel(points)
    cdef char * c_command = command

    #_qhull_lock.acquire()
    cdef int result = _qhull.qh_new_qhull(dimension,
                                          len(points),
                                          &c_points[0],
                                          False,
                                          &c_command[0],
                                          NULL,
                                          _qhull.stderr)

    if result != _qhull.qh_ERRnone:
        _qhull.qh_freeqhull(_qhull.qh_ALL)  # Free qhull resources
        #_qhull_lock.release()
        raise _errors[result]

    # Get number of facets
    cdef unsigned int nbFacets = nbDelaunayFacets()
    if nbFacets == 0:
        _qhull.qh_freeqhull(_qhull.qh_ALL)  # Free qhull resources
        #_qhull_lock.release()
        return numpy.array((), dtype=numpy.uint32)

    # Get facets' indices
    indices = numpy.empty((nbFacets, dimension + 1), dtype=numpy.uint32)
    cdef unsigned int[:] c_indices = indices.ravel()
    setDelaunayIndices(c_indices)

    # Free qhull resources
    _qhull.qh_freeqhull(_qhull.qh_ALL)
    #_qhull_lock.release()

    return indices
