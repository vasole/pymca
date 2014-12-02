#ifndef __minMax_H__
#define __minMax_H__

/* Make sure we work with 32 bit integers */
#if defined (_MSC_VER)
    /* Microsoft Visual Studio */
    #if _MSC_VER >= 1600
        /* Visual Studio 2010 and higher */
        #include <stdint.h>
    #else
        #ifndef int8_t
            #define int8_t char
        #endif
        #ifndef uint8_t
            #define uint8_t unsigned char
        #endif

        #ifndef int16_t
            #define int16_t short
        #endif
        #ifndef uint16_t
            #define uint16_t unsigned short
        #endif

        #ifndef int32_t
            #define int32_t int
        #endif
        #ifndef uint32_t
            #define uint32_t unsigned int
        #endif

        #ifndef int64_t
            #define int64_t long
        #endif
        #ifndef uint64_t
            #define uint64_t unsigned long
        #endif

    #endif
#else
    #include <stdint.h>
#endif


void
getMinMaxFloat(float * data, unsigned int length,
               float * min, float * max);

void
getMinMaxDouble(double * data, unsigned int length,
                double * min, double * max);

void
getMinMaxInt8(int8_t * data, unsigned int length,
              int8_t * min, int8_t * max);

void
getMinMaxUInt8(uint8_t * data, unsigned int length,
               uint8_t * min, uint8_t * max);

void
getMinMaxInt16(int16_t * data, unsigned int length,
               int16_t * min, int16_t * max);

void
getMinMaxUInt16(uint16_t * data, unsigned int length,
                uint16_t * min, uint16_t * max);

void
getMinMaxInt32(int32_t * data, unsigned int length,
               int32_t * min, int32_t * max);

void
getMinMaxUInt32(uint32_t * data, unsigned int length,
                uint32_t * min, uint32_t * max);

void
getMinMaxInt64(int64_t * data, unsigned int length,
               int64_t * min, int64_t * max);

void
getMinMaxUInt64(uint64_t* data, unsigned int length,
                uint64_t * min, uint64_t * max);

#endif /*__minMax_H__*/
