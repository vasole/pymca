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
#include <math.h>

#include "MinMax.h"
#include "Types.h"

#if (defined (_MSC_VER) && _MSC_VER < 1800)
#include <float.h>

#define isnan(v) _isnan(v)
#endif

/* To support NaN, for floating type, we skip all first NaN data
 * If all data is NaNs: min/max are NaNs
 * Else min/max are computed ignoring NaNs,
 * as NaN is never < or > to a number.
 */
#define INIT_SKIP_NAN(TYPE) \
    for (; index < length; index++) {\
        TYPE value = data[index];\
        if (!isnan(value)) {\
            tmpMin = value;\
            tmpMax = value;\
            break;\
        }\
    }

#define INIT_NOOP(TYPE)

#define GET_MINMAX_DEFINITION(TYPE, INIT_CODE)\
static void getMinMax_ ## TYPE(TYPE * data,\
                 unsigned long length,\
                 double * min,\
                 double * minPos,\
                 double * max)\
{\
    TYPE tmpMin = data[0];\
    TYPE tmpMax = tmpMin;\
    unsigned long index = 0;\
\
    INIT_CODE(TYPE)\
\
    if (minPos != 0) {\
        TYPE tmpMinPos = (TYPE) 0;\
\
        /* First loop until tmpMinPos is initialized */\
        for (; index < length; index++) {\
            TYPE value = data[index];\
            tmpMin = (value < tmpMin) ? value : tmpMin;\
            tmpMax = (value > tmpMax) ? value : tmpMax;\
            if (value > (TYPE) 0) {\
                tmpMinPos = value;\
                break;\
            }\
        }\
\
        /* Second loop with tmpMinPos initialized */\
        for (; index < length; index++) {\
            TYPE value = data[index];\
            tmpMin = (value < tmpMin) ? value : tmpMin;\
            tmpMax = (value > tmpMax) ? value : tmpMax;\
            tmpMinPos = (value > (TYPE) 0 && value < tmpMinPos) ? value : tmpMinPos;\
        }\
\
        *minPos = (double) tmpMinPos;\
    }\
    else {\
        for (; index < length; index++) {\
            TYPE value = data[index];\
            tmpMin = (value < tmpMin) ? value : tmpMin;\
            tmpMax = (value > tmpMax) ? value : tmpMax;\
        }\
    }\
\
    *min = (double) tmpMin;\
    *max = (double) tmpMax;\
}


GET_MINMAX_DEFINITION(float, INIT_SKIP_NAN)
GET_MINMAX_DEFINITION(double, INIT_SKIP_NAN)

GET_MINMAX_DEFINITION(int8_t, INIT_NOOP)
GET_MINMAX_DEFINITION(uint8_t, INIT_NOOP)

GET_MINMAX_DEFINITION(int16_t, INIT_NOOP)
GET_MINMAX_DEFINITION(uint16_t, INIT_NOOP)

GET_MINMAX_DEFINITION(int32_t, INIT_NOOP)
GET_MINMAX_DEFINITION(uint32_t, INIT_NOOP)

GET_MINMAX_DEFINITION(int64_t, INIT_NOOP)
GET_MINMAX_DEFINITION(uint64_t, INIT_NOOP)


#define CALL_GET_MINMAX(TYPE)\
    getMinMax_ ## TYPE((TYPE *) data,\
        length,\
        minOut,\
        minPosOut,\
        maxOut)


void
getMinMax(void * data,
          unsigned int type,
          unsigned long length,
          double * minOut,
          double * minPosOut,
          double * maxOut)
{
    switch (type) {
        case (FLOATING | SIZE_32): /*float*/
            CALL_GET_MINMAX(float);
            break;
        case (FLOATING | SIZE_64): /*double*/
            CALL_GET_MINMAX(double);
            break;

        case (SIZE_8): /*int8_t*/
            CALL_GET_MINMAX(int8_t);
            break;
        case (UNSIGNED | SIZE_8): /*uint8_t*/
           CALL_GET_MINMAX(uint8_t);
           break;

        case (SIZE_16): /*int16_t*/
            CALL_GET_MINMAX(int16_t);
            break;
        case (UNSIGNED | SIZE_16): /*uint16_t*/
            CALL_GET_MINMAX(uint16_t);
            break;

        case (SIZE_32): /*int32_t*/
            CALL_GET_MINMAX(int32_t);
            break;
        case (UNSIGNED | SIZE_32): /*uint32_t*/
            CALL_GET_MINMAX(uint32_t);
            break;

        case (SIZE_64): /*int64_t*/
            CALL_GET_MINMAX(int64_t);
            break;
        case (UNSIGNED | SIZE_64): /*uint64_t*/
            CALL_GET_MINMAX(uint64_t);
            break;
        default:
            break;
    }
}
