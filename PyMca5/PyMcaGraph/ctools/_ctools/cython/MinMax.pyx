cimport cython
cimport numpy
import numpy

#from libc.stdint cimport int8_t, uint8_t
#from libc.stdint cimport int16_t, uint16_t
#from libc.stdint cimport int32_t, uint32_t
#from libc.stdint cimport int64_t, uint64_t

from MinMax cimport *

def _minMaxFloat(numpy.ndarray data):
    cdef float[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef float minValue
    cdef float maxValue
    getMinMaxFloat(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxDouble(numpy.ndarray data):
    cdef double[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef double minValue
    cdef double maxValue
    getMinMaxDouble(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxInt8(numpy.ndarray data):
    cdef char[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef char minValue
    cdef char maxValue
    getMinMaxInt8(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


def _minMaxUInt8(numpy.ndarray data):
    cdef unsigned char[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef unsigned char minValue
    cdef unsigned char maxValue
    getMinMaxUInt8(&c_data[0], length, &minValue, &maxValue)
    return minValue, maxValue


_minMaxFunctions = {
    numpy.dtype('float32'): _minMaxFloat,
    numpy.dtype('float64'): _minMaxDouble,
    numpy.dtype('int8'): _minMaxInt8,
    numpy.dtype('uint8'): _minMaxUInt8,
}


@cython.boundscheck(False)
@cython.wraparound(False)
def minMax(numpy.ndarray data):
    try:
        minMaxFunc = _minMaxFunctions[data.dtype]
    except KeyError:
        raise NotImplementedError(
            "Unsupported numpy.ndarray dtype: {}".format(data.dtype))
    return minMaxFunc(numpy.ravel(data, order='C'))
