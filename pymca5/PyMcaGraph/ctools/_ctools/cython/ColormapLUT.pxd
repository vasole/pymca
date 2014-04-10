cdef extern from "ColormapLUT.h":
    void fillPixmapFromDouble(double *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromFloat(float *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromChar(char *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromUChar(unsigned char *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromShort(short *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromUShort(unsigned short **, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromInt(char *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromUInt(unsigned int *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromLong(long *, long, char *, long, \
                          char *, short, short, double *, double *) nogil
    void fillPixmapFromULong(unsigned long *, long, char *, long, \
                          char *, short, short, double *, double *) nogil

