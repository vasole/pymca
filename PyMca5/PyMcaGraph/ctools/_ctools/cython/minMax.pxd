from libc.stdint cimport int8_t, uint8_t
from libc.stdint cimport int16_t, uint16_t
from libc.stdint cimport int32_t, uint32_t
from libc.stdint cimport int64_t, uint64_t

cdef extern from "MinMax.h":
    void getMinMaxDouble(double * data, unsigned int length,
                         double * min, double * max)
    void getMinMaxFloat(float * data, unsigned int length,
                        float * min, float * max)
    void getMinMaxInt8(int8_t * data, unsigned int length,
                       int8_t * min, int8_t * max)
    void getMinMaxUInt8(uint8_t * data, unsigned int length,
                        uint8_t * min, uint8_t * max)
    void getMinMaxInt16(int16_t * data, unsigned int length,
                        int16_t * min, int16_t * max)
    void getMinMaxUInt16(uint16_t * data, unsigned int length,
                         uint16_t * min, uint16_t * max)
    void getMinMaxInt32(int32_t * data, unsigned int length,
                        int32_t * min, int32_t * max)
    void getMinMaxUInt32(uint32_t * data, unsigned int length,
                         uint32_t * min, uint32_t * max)
    void getMinMaxInt64(int64_t * data, unsigned int length,
                        int64_t * min, int64_t * max)
    void getMinMaxUInt64(uint64_t * data, unsigned int length,
                         uint64_t * min, uint64_t * max)
