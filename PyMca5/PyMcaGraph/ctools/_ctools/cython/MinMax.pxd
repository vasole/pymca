cdef extern from "MinMax.h":
    void getMinMax(void * data,
                   unsigned int type,
                   unsigned long length,
                   double * minOut,
                   double * minPosOut,
                   double * maxOut) nogil
