# check on Windows
# try:
#    from libc.stdint cimport uint8_t
# except ImportError:
#    ctypedef unsigned char uint8_t
ctypedef unsigned char uint8_t

cdef extern from "Colormap.h":
    void colormapFillPixmap(void * data,
                            unsigned int type,
                            unsigned int length,
                            double min,
                            double max,
                            uint8_t * RGBAColormap,
                            unsigned int colormapLength,
                            uint8_t * RGBAPixmapOut) nogil
