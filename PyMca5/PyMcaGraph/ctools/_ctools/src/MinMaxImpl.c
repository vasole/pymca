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
#include "MinMax.h"

#define GET_MINMAX(NAME, TYPE) \
void NAME(TYPE * data, unsigned int length, TYPE * min, TYPE * max) \
{ \
    TYPE tmpMin = data[0]; \
    TYPE tmpMax = tmpMin; \
    TYPE * endPtr = &data[length]; \
    TYPE * curPtr; \
\
    for (curPtr = data; curPtr < endPtr; curPtr++) { \
        TYPE value = *curPtr; \
        if (value < tmpMin) { \
            tmpMin = value; \
        } \
        else if (value > tmpMax) { \
            tmpMax = value; \
        } \
    } \
    *min = tmpMin; \
    *max = tmpMax; \
}

GET_MINMAX(getMinMaxDouble, double)
GET_MINMAX(getMinMaxFloat, float)

GET_MINMAX(getMinMaxInt8, int8_t)
GET_MINMAX(getMinMaxUInt8, uint8_t)
GET_MINMAX(getMinMaxInt16, int16_t)
GET_MINMAX(getMinMaxUInt16, uint16_t)
GET_MINMAX(getMinMaxInt32, int32_t)
GET_MINMAX(getMinMaxUInt32, uint32_t)
GET_MINMAX(getMinMaxInt64, int64_t)
GET_MINMAX(getMinMaxUInt64, uint64_t)

