cimport cython

cdef extern from "polspl.h":
    void polspl(double *, double *, double *, int, double *, double *, int *, int, double *, int)
