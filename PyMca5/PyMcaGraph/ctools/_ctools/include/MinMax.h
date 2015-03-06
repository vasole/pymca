#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
#ifndef __MinMax_H__
#define __MinMax_H__

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

#endif /*__MinMax_H__*/
