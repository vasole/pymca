#ifndef __Types_H__
#define __Types_H__

/* Defines sized types if they are not defined */
#if defined (_MSC_VER)
    /* Microsoft Visual Studio */
    #if _MSC_VER >= 1600
        /* Visual Studio 2010 and higher */
        #include <stdint.h>
    #else
        #include <limits.h>

        #ifndef int8_t
            #define int8_t signed char
            #define INT8_MIN SCHAR_MIN
            #define INT8_MAX SCHAR_MAX
        #endif
        #ifndef uint8_t
            #define uint8_t unsigned char
            #define UINT8_MAX UCHAR_MAX
        #endif

        #ifndef int16_t
            #define int16_t short
            #define INT16_MIN SHRT_MIN
            #define INT16_MAX SHRT_MAX
        #endif
        #ifndef uint16_t
            #define uint16_t unsigned short
            #define UINT16_MAX USHRT_MAX
        #endif

        #ifndef int32_t
            #define int32_t int
            #define INT32_MIN INT_MIN
            #define INT32_MAX INT_MAX
        #endif
        #ifndef uint32_t
            #define uint32_t unsigned int
            #define UINT32_MAX UINT_MAX
        #endif

        #ifndef int64_t
            #define int64_t long long
            #define INT32_MIN LLONG_MIN
            #define INT32_MAX LLONG_MAX
        #endif
        #ifndef uint64_t
            #define uint64_t unsigned long long
            #define UINT32_MAX ULLONG_MAX
        #endif

    #endif
#else
    #include <stdint.h>
#endif


/* Description of data type using as a bit field */
#define FLOATING (1 << 3) /**< flag for floating point types */
#define UNSIGNED (1 << 2) /**< flag for unsigned types (int only) */
#define SIZE_MASK 0x3 /**< Bit mask corresponding to size */
#define SIZE_8   (0) /**< 8 bits sized type */
#define SIZE_16  (1) /**< 16 bits sized type */
#define SIZE_32  (2) /**< 32 bits sized type */
#define SIZE_64  (3) /**< 64 bits sized type */

#endif /*__Types_H__*/
