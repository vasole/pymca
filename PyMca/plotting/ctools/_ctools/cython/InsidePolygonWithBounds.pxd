cdef extern from "InsidePolygonWithBounds.h":
    void PointsInsidePolygon(double *, int , \
                         double *, int , int , unsigned char *) nogil
    void PointsInsidePolygonF(double *, int , \
                         float *, int , int , unsigned char *) nogil
    void PointsInsidePolygonInt(double *, int , \
                         int *, int , int , unsigned char *) nogil

