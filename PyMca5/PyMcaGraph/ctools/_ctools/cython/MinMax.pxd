cdef extern from "MinMax.h":
    void getMinMax(void * data,
                   unsigned int type,
                   unsigned int length,
                   double * minOut,
                   double * minPosOut,
                   double * maxOut) nogil
