ctypedef unsigned char uint8_t  # TODO use libc.stdint when available

cdef extern from "Colormap.h":
    void colormapFillPixmap(void * data,
                            unsigned int type,
                            unsigned int length,
                            double min,
                            double max,
                            uint8_t * RGBAColormap,
                            unsigned int colormapLength,
                            uint8_t * RGBAPixmapOut) nogil
