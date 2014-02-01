cdef extern from "InsidePolygonWithBounds.h":
	void PointsInsidePolygon(double *, int , \
                         double *, int , int , unsigned char *) nogil

