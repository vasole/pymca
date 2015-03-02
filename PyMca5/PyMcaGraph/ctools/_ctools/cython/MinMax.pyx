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
def minMax(np.ndarray data):
    cdef np.ndarray c_data = np.ascontiguousarray(data)
    cdef double dataMin, dataMax

    type_ = _NUMPY_TO_TYPE_DESC[data.dtype.str[1:]]

    getMinMax(c_data.data, type_, c_data.size, &dataMin, &dataMax)
    return dataMin, dataMax
