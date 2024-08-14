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
import numpy
cimport numpy
from polspl cimport polspl as _polspl
from bessel0 cimport j0Single, j0Multiple

def j0(x):
    if hasattr(x, "__len__"):
        return _besselMultiple(x)
    else:
        return _besselSingle(x)

def _besselMultiple(x):
    result = numpy.array(x, copy=True, dtype=numpy.float64)
    cdef double[:] c_x = result
    cdef int c_npts = c_x.size
    j0Multiple(&c_x[0], c_npts)
    return result

def _besselSingle(double x):
    return j0Single(x)

def polspl(x, y, w, npts, xl, xh, nr, nc):
    c = numpy.zeros((36,), dtype=numpy.float64)
    cdef double[:] c_c = c
    cdef double[:] c_x = numpy.ascontiguousarray(x,
                                                 dtype=numpy.float64)
    cdef double[:] c_y = numpy.ascontiguousarray(y,
                                                 dtype=numpy.float64)
    cdef double[:] c_w = numpy.ascontiguousarray(w,
                                                 dtype=numpy.float64)
    cdef int c_npts = npts
    cdef double[:] c_xl = numpy.ascontiguousarray(xl,
                                                 dtype=numpy.float64)
    cdef double[:] c_xh = numpy.ascontiguousarray(xh,
                                                 dtype=numpy.float64)
    cdef int c_nr = nr
    cdef int[:] c_nc = numpy.ascontiguousarray(nc,
                                                 dtype=numpy.int32)
    cdef int c_sizeC = c_c.size
    _polspl(&c_x[0], &c_y[0], &c_w[0], c_npts, \
            &c_xl[0], &c_xh[0], &c_nc[0], c_nr, &c_c[0], c_sizeC)
    return c

def polspl2(x,y,w,npts,xl0,xh0,nr,nc):

    # ;
    # ; few definitions
    # ;

    cdef numpy.ndarray[double, ndim=1, mode='c'] buffer_xl0 = \
		    numpy.ascontiguousarray(xl0, numpy.float64)
    cdef double * xl = <double *> buffer_xl0.data 
    cdef numpy.ndarray[double, ndim=1, mode='c'] buffer_xh0 = \
		    numpy.ascontiguousarray(xh0, numpy.float64)
    cdef double * xh = <double *> buffer_xh0.data 
    df = numpy.zeros(26)  
    a = numpy.zeros((36,37))  
    nbs = numpy.zeros(11,dtype=int)
    cdef double[:] xk0 = numpy.zeros(10)
    cdef double * xk = &xk0[0]
    c = numpy.zeros(36)  
    cdef int j=0 
    cdef int i=0  
    ne_idl=0 
    n = 0 
    cdef int k = 0 
    cdef int ibl = 0
    cdef int ns = 0  
    cdef int ns1 = 0

    nbs[1]=1
    for i in range(1,nr+1):
        n=n+int(nc[i])
        nbs[i+1]=n+1
        if xl[i] < xh[i]: 
            pass
        else:
            t=xl[i]
            xl[i]=xh[i]
            xh[i]=t

    n=n+2*(nr-1)
    n1=n+1
    xl[nr+1]=0.
    xh[nr+1]=0.

    # this loop ...
    for ibl in range(1,nr+1):
        xk[ibl]=.5*(xh[ibl]+xl[ibl+1])
        if (xl[ibl] > xl[ibl+1]):
            xk[ibl]=.5*(xl[ibl]+xh[ibl+1])
        ns=nbs[ibl]
        ne_idl=nbs[ibl+1]-1
        for i in range(1, npts+1):
            if((x[i] < xl[ibl]) or (x[i] > xh[ibl])): 
                pass
            else:
                df[ns]=1.0
                ns1=ns+1
                for j in range(ns1,ne_idl+1):
                    df[j]=df[j-1]*x[i]
                for j in range(ns,ne_idl+1):
                    for k in range(j,ne_idl+1): 
                        a[j,k]=a[j,k]+df[j]*df[k]*w[i]
                    a[j,n1]=a[j,n1]+df[j]*y[i]*w[i]
    # ... has to be faster
    
    ncol=nbs[nr+1]-1
    nk=nr-1

    if (nk == 0): 
        pass
    else:
        for ik in range(1,nk+1):
            ncol=ncol+1
            ns=nbs[ik]
            ne_idl=nbs[ik+1]-1
            a[ns,ncol]=-1.
            ns=ns+1
            for i in range(ns,ne_idl+1):
                a[i,ncol]=a[i-1,ncol]*xk[ik]
            ncol=ncol+1
            a[ns,ncol]=-1.
            ns=ns+1
            if (ns > ne_idl): 
                pass
            else:
                for i in range(ns,ne_idl+1):
                    a[i,ncol]=(ns-i-2)*numpy.power(xk[ik],(i-ns+1))
            ncol=ncol-1
            ns=nbs[ik+1]
            ne_idl=nbs[ik+2]-1
            a[ns,ncol]=1.0
            ns=ns+1
            for i in range(ns,ne_idl+1):
                a[i,ncol]=a[i-1,ncol]*xk[ik]
            ncol=ncol+1
            a[ns,ncol]=1.0
            ns=ns+1
            if (ns > ne_idl): 
                pass
            else:
                for i in range(ns,ne_idl+1): 
                    a[i,ncol]=(i-ns+2)*numpy.power(xk[ik],(i-ns+1))

    for i in range(1,n+1):
        i1=i-1
        for j in range(1,i1+1): 
            a[i,j]=a[j,i]
    nm1=n-1

    for i in range(1,nm1+1): 
        i1=i+1
        m=i
        t=numpy.abs(a[i,i])
        for j in range(i1,n+1): 
            if (t >= numpy.abs(a[j,i])):
                pass
            else:
                m=j
                t=numpy.abs(a[j,i])
        if (m == i): 
            pass
        else:
            for j in range(1,n1+1): 
                t=a[i,j]
                a[i,j]=a[m,j]
                a[m,j]=t
        for j in range(i1,n+1): 
            t=a[j,i]/a[i,i]
            for k in range(i1,n1+1): 
                a[j,k]=a[j,k]-t*a[i,k]
    c[n]=a[n,n1]/a[n,n]
    for i in range(1,nm1+1): 
        ni=n-i
        t=a[ni,n1]
        ni1=ni+1
        for j in range(ni1,n+1): 
            t=t-c[j]*a[ni,j]
        c[ni]=t/a[ni,ni]

    return c

