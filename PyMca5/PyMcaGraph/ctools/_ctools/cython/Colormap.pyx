cimport cython
cimport numpy as np
import numpy as np


from Colormap cimport colormapFillPixmap, initFastLog10
from Colormap cimport fastLog10 as _fastLog10

from MinMax cimport getMinMax

# Init fastLog10 look-up table
initFastLog10()

# Convert numpy dtype array-protocol string to bit field to pass to C function
# 4th bit for type: 1: floating point, 0: integer
# 3rd bit for signedness (int only): 1: unsigned, 0: signed
# 1st and 2nd bits for size: 00: 8 bits, 01: 16 bits, 10: 32 bits, 11: 64 bits.
_NUMPY_TO_TYPE_DESC = {
    'f8': 0b1011,
    'f4': 0b1010,
    'i1': 0b0000,
    'u1': 0b0100,
    'i2': 0b0001,
    'u2': 0b0101,
    'i4': 0b0010,
    'u4': 0b0110,
    'i8': 0b0011,
    'u8': 0b0111,
}

@cython.boundscheck(False)
@cython.wraparound(False)
def dataToRGBAColormap(data,
                       np.ndarray[np.uint8_t, ndim=2, mode="c"] colormap,
                       startValue, endValue,
                       bint isLog10Mapping=False):
    """Compute a pixmap by applying a colormap to data.

    :param np.ndarray data: Array of data value to convert to pixmap.
    :param np.ndarray colormap: palette to use as colormap as an array of
                                RGBA color.
    :param startValue: The value to map to the first color of the colormap.
    :param endValue: The value to map to the last color of the colormap.
    :param bool isLog10Mapping: The mapping: False for linear, True for log10.
    :returns: The corresponding pixmap of RGBA pixels as an array of 4 uint8
              with same dimensions as data and used min and max.
    :rtype: A tuple : (pixmap , (usedMin, usedMax)).
    """
    #Convert float16 to float32
    if data.dtype.str[1:] == 'f2':
        data = np.asarray(data, dtype=np.float32)

    cdef np.ndarray c_data = np.ascontiguousarray(data)
    cdef void * c_dataPtr = c_data.data  # &c_data[0] needs dim
    cdef unsigned int c_dataSize = c_data.size
    cdef unsigned int c_dataItemSize = c_data.itemsize

    cdef unsigned char[:, :] c_colormap = colormap
    cdef unsigned int c_colormapLength = len(colormap)

    pixmap = np.empty((data.size, 4), dtype=np.uint8)
    cdef unsigned char[:, :] c_pixmap = pixmap

    cdef unsigned int c_type = _NUMPY_TO_TYPE_DESC[data.dtype.str[1:]]

    cdef double c_start, c_startExtra, c_end
    if startValue is None or endValue is None:
        if isLog10Mapping:
            with nogil:
                getMinMax(c_dataPtr, c_type, c_dataSize,
                          &c_startExtra, &c_start, &c_end)
        else:
            with nogil:
                getMinMax(c_dataPtr, c_type, c_dataSize,
                          &c_start, NULL, &c_end)

        if startValue is not None:
            c_start = startValue
        if endValue is not None:
            c_end = endValue
    else:
        c_start = startValue
        c_end = endValue

    with nogil:
        colormapFillPixmap(c_dataPtr,
                           c_type,
                           c_dataSize,
                           c_start,
                           c_end,
                           isLog10Mapping,
                           &c_colormap[0, 0],
                           c_colormapLength,
                           &c_pixmap[0, 0])

    pixmap.shape = data.shape + (4,)
    return pixmap, (c_start, c_end)

def fastLog10(double value):
    return _fastLog10(value)
