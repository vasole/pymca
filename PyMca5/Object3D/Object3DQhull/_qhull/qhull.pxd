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


cimport numpy

cdef extern from "stdio.h":
    ctypedef void FILE

    extern FILE * stderr


cdef extern from 'user.h':
    # Sync floating point type with qhull
    # See REALfloat macro in qhull/src/user.h
    IF REALfloat == 1:
        ctypedef numpy.float32_t realT
    ELSE:
        ctypedef numpy.float64_t realT


# Debian libqhull-dev renamed libqhull.h tp qhull.h
# include qhull_a.h instead to workaround this case
# cdef extern from 'libqhull.h':
cdef extern from 'qhull_a.h':
    ctypedef unsigned int boolT
    ctypedef realT coordT
    ctypedef unsigned int flagT
    ctypedef coordT pointT

    ctypedef struct vertexT:
        pointT * point

    ctypedef struct facetT:
        facetT * next
        setT * vertices
        flagT upperdelaunay

    ctypedef struct qhT:
        facetT * facet_list

    int qh_new_qhull(int dim, int numpoints, coordT * points, boolT ismalloc,
                     char * qhull_cmd, FILE * outfile, FILE * errfile) nogil

    void qh_freeqhull(boolT allmem) nogil

    int qh_pointid(pointT * point) nogil

    void qh_setdelaunay(int dim, int count, pointT * points) nogil

    # facetT * qh_findbestfacet(pointT * point, boolT bestoutside,
    #                           realT * bestdist, boolT * isoutside) nogil

    extern char * qh_version

    # WARNING sync type with #define qh_QHpointer in user.h
    IF qh_QHpointer == 0:
        extern qhT qh_qh
    ELSE:
        extern qhT * qh_qh

    boolT qh_ALL = True

    int qh_ERRnone = 0
    int qh_ERRinput = 1
    int qh_ERRsingular = 2
    int qh_ERRprec = 3
    int qh_ERRmem = 4
    int qh_ERRqhull = 5


cdef extern from 'qset.h':
    ctypedef union setelemT:
        void * p
        int i

    ctypedef struct setT:
        int maxsize
        setelemT e[1]


# cdef extern from 'geom.h':
#     pointT *qh_facetcenter(setT *vertices) nogil
