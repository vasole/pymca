cimport cython
cimport numpy as np
import numpy as np

from MinMax cimport getMinMax


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
def minMax(np.ndarray data, bint minPositive=False):
    """Get min, max and optionally min positive of data.

    :param np.ndarray data: Array of data
    :param bool minPositive: Wheither to compute min positive or not.
    :returns: (min, max) or (min, minPositive, max) if minPositive is True
              If all data < 0, minPositive is None.
    :rtype: tuple of float
    :raises: ValueError if data is empty
    """
    #Convert float16 to float32
    if data.dtype.str[1:] == 'f2':
        data = np.asarray(data, dtype=np.float32)

    cdef np.ndarray c_data = np.ascontiguousarray(data)
    cdef void * c_dataPtr = c_data.data
    cdef unsigned int c_dataSize = c_data.size

    if c_dataSize == 0:
        raise ValueError("zero-size array")

    cdef double c_dataMin, c_dataMinPos, c_dataMax

    cdef unsigned int c_type = _NUMPY_TO_TYPE_DESC[data.dtype.str[1:]]

    if minPositive:
        with nogil:
            getMinMax(c_dataPtr, c_type, c_dataSize,
                      &c_dataMin, &c_dataMinPos, &c_dataMax)
        if c_dataMinPos == 0:
            return c_dataMin, None, c_dataMax
        else:
            return c_dataMin, c_dataMinPos, c_dataMax
    else:
        with nogil:
            getMinMax(c_dataPtr, c_type, c_dataSize,
                      &c_dataMin, NULL, &c_dataMax)
        return c_dataMin, c_dataMax
