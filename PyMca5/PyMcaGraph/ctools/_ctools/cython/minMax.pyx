cimport cython
cimport numpy
import numpy

from libc.stdint cimport int8_t, uint8_t
from libc.stdint cimport int16_t, uint16_t
from libc.stdint cimport int32_t, uint32_t
from libc.stdint cimport int64_t, uint64_t

from minMax cimport getMinMaxFloat, getMinMaxDouble
from minMax cimport getMinMaxInt8, getMinMaxUInt8
from minMax cimport getMinMaxInt16, getMinMaxUInt16
from minMax cimport getMinMaxInt32, getMinMaxUInt32
from minMax cimport getMinMaxInt64, getMinMaxUInt64


def _minMaxFloat(numpy.ndarray[numpy.float32_t, mode="c"] data):
    cdef float[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef float minValue
    cdef float maxValue
    getMinMaxFloat(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxDouble(numpy.ndarray[numpy.float64_t, mode="c"] data):
    cdef double[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef double minValue
    cdef double maxValue
    getMinMaxDouble(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxInt8(numpy.ndarray[numpy.int8_t, mode="c"] data):
    cdef int8_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef int8_t minValue
    cdef int8_t maxValue
    getMinMaxInt8(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxUInt8(numpy.ndarray[numpy.uint8_t, mode="c"] data):
    cdef uint8_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef uint8_t minValue
    cdef uint8_t maxValue
    getMinMaxUInt8(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxInt16(numpy.ndarray[numpy.int16_t, mode="c"] data):
    cdef int16_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef int16_t minValue
    cdef int16_t maxValue
    getMinMaxInt16(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxUInt16(numpy.ndarray[numpy.uint16_t, mode="c"] data):
    cdef uint16_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef uint16_t minValue
    cdef uint16_t maxValue
    getMinMaxUInt16(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxInt32(numpy.ndarray[numpy.int32_t, mode="c"] data):
    cdef int32_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef int32_t minValue
    cdef int32_t maxValue
    getMinMaxInt32(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxUInt32(numpy.ndarray[numpy.uint32_t, mode="c"] data):
    cdef uint32_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef uint32_t minValue
    cdef uint32_t maxValue
    getMinMaxUInt32(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxInt64(numpy.ndarray[numpy.int64_t, mode="c"] data):
    cdef int64_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef int64_t minValue
    cdef int64_t maxValue
    getMinMaxInt64(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue


def _minMaxUInt64(numpy.ndarray[numpy.uint64_t, mode="c"] data):
    cdef uint64_t[:] c_data = data
    cdef unsigned int length = c_data.size
    cdef uint64_t minValue
    cdef uint64_t maxValue
    getMinMaxUInt64(& c_data[0], length, & minValue, & maxValue)
    return minValue, maxValue

_minMaxFunctions = {
    numpy.dtype('float32'): _minMaxFloat,
    numpy.dtype('float64'): _minMaxDouble,
    numpy.dtype('int8'): _minMaxInt8,
    numpy.dtype('uint8'): _minMaxUInt8,
    numpy.dtype('int16'): _minMaxInt16,
    numpy.dtype('uint16'): _minMaxUInt16,
    numpy.dtype('int32'): _minMaxInt32,
    numpy.dtype('uint32'): _minMaxUInt32,
    numpy.dtype('int64'): _minMaxInt64,
    numpy.dtype('uint64'): _minMaxUInt64,
}


@cython.boundscheck(False)
@cython.wraparound(False)
def minMax(numpy.ndarray data):
    try:
        minMaxFunc = _minMaxFunctions[data.dtype]
    except KeyError:
        raise NotImplementedError(
            "Unsupported numpy.ndarray dtype: {}".format(data.dtype))
    return minMaxFunc(data)
