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
#include <stdio.h>
#include <limits.h>
#include <float.h>
#ifndef __GNUC__
	#include <math.h>
	#ifndef lrintf
		#define	lrint(dbl)	((int)(dbl))
		#define	lrintf(flt)	((int)(flt))
	#endif
#else
	#include <math.h>
#endif // __GNUC__

#include "../include/ColormapLUT.h"


#define FINDMINMAX(NAME, TYPE) \
void NAME(TYPE *data, long nValues, double *minValue, double *maxValue, double *minPositive, TYPE highestValue) \
{                                                               \
    register TYPE *c1;                                          \
    register long i;                                            \
    register TYPE tmpMinValue, tmpMaxValue, tmpMinPositive;     \
    int doMinPositive;                                          \
                                                                \
    c1 = data;                                                  \
    if (minPositive == NULL)                                    \
        doMinPositive = 0;                                      \
    else                                                        \
        doMinPositive = 1;                                      \
    tmpMaxValue = tmpMinValue = *c1;                            \
    tmpMinPositive = highestValue;                              \
    if (doMinPositive)                                          \
    {                                                           \
        for (i = nValues; i; i--, c1++) {                       \
            if (*c1 < tmpMinValue)                              \
                tmpMinValue = *c1;                              \
            else if (*c1 > tmpMaxValue)                         \
                tmpMaxValue = *c1;                              \
            if ((*c1 < tmpMinPositive) && (*c1 > 0))            \
                tmpMinPositive = *c1;                           \
        }                                                       \
    }                                                           \
    else                                                        \
    {                                                           \
        for (i = nValues; i; i--, c1++)                         \
        {                                                       \
            if (*c1 < tmpMinValue)                              \
                tmpMinValue = *c1;                              \
            else if (*c1 > tmpMaxValue)                         \
                tmpMaxValue = *c1;                              \
        }                                                       \
    }                                                           \
    *minValue = (double) tmpMinValue;                           \
    *maxValue = (double) tmpMaxValue;                           \
    *minPositive = (double) tmpMinPositive;                     \
}

FINDMINMAX(getMinMaxDouble, double)//, DBL_MAX)
FINDMINMAX(getMinMaxFloat, float) //, FLT_MAX)
FINDMINMAX(getMinMaxChar, char)// , SCHAR_MAX)
FINDMINMAX(getMinMaxUChar, unsigned char) //, UCHAR_MAX)
FINDMINMAX(getMinMaxShort, short) //, SHRT_MAX)
FINDMINMAX(getMinMaxUShort, unsigned short) //, USHRT_MAX)
FINDMINMAX(getMinMaxInt, int32_t)//, INT_MAX)
FINDMINMAX(getMinMaxUInt, uint32_t)//, UINT_MAX)
FINDMINMAX(getMinMaxLong, long) //, LONG_MAX)
FINDMINMAX(getMinMaxULong, unsigned long)//, ULONG_MAX)

/*
It is very likely an performance mistake to always calculate min and max as double instead of
using the supplied type.

The goal would be to avoid the conversion double to int.

One could probably just use double pointers for the float and double cases.

The use of the lrint and lrintf functions would be very limited because it seems the cast truncation
is as fast on intel 64 bit machines with SSE2 optimization enabled.

*/


#define FILLPIXMAP()\
    colormapInt32 = (uint32_t *) colormap;\
    pixmapInt32 = (uint32_t *) pixmap;\
    delta = usedMax - usedMin; \
    if (delta <= 0) \
        delta = 1.0; \
    for (i = 0; i < nValues; i++) \
    {\
        if(data[i] <= usedMin)\
        {\
            pixmapInt32[i] = colormapInt32[0];\
        }\
        else\
        {\
            if(data[i] >= usedMin)\
            {\
                pixmapInt32[i] = colormapInt32[nColors - 1];\
            }\
            else\
            {\
                idx = lrint(nColors * ((*data - usedMin)/delta));\
                if (idx < 0)\
                    idx = 0;\
                if (idx >= nColors)\
                    idx = nColors - 1;\
                pixmapInt32[i] = colormapInt32[idx];\
            }\
        }\
    }


void fillPixmapFromDouble(double *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxDouble(data, nValues, &usedMin, &usedMax, minPlusPointer, DBL_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxDouble(data, nValues, &usedMin, &usedMax, minPlusPointer, DBL_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */

    FILLPIXMAP()
}

void fillPixmapFromFloat(float *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxFloat(data, nValues, &usedMin, &usedMax, minPlusPointer, FLT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxFloat(data, nValues, &usedMin, &usedMax, minPlusPointer, FLT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromChar(char *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxChar(data, nValues, &usedMin, &usedMax, minPlusPointer, CHAR_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxChar(data, nValues, &usedMin, &usedMax, minPlusPointer, CHAR_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromUChar(unsigned char *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxUChar(data, nValues, &usedMin, &usedMax, minPlusPointer, UCHAR_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxUChar(data, nValues, &usedMin, &usedMax, minPlusPointer, UCHAR_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}


void fillPixmapFromShort(short *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxShort(data, nValues, &usedMin, &usedMax, minPlusPointer, SHRT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxShort(data, nValues, &usedMin, &usedMax, minPlusPointer, SHRT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromUShort(unsigned short *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }
    if (autoFlag)
    {
        getMinMaxUShort(data, nValues, &usedMin, &usedMax, minPlusPointer, USHRT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxUShort(data, nValues, &usedMin, &usedMax, minPlusPointer, USHRT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromInt(int *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxInt(data, nValues, &usedMin, &usedMax, minPlusPointer, INT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxInt(data, nValues, &usedMin, &usedMax, minPlusPointer, INT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromUInt(unsigned int *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxUInt(data, nValues, &usedMin, &usedMax, minPlusPointer, UINT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxUInt(data, nValues, &usedMin, &usedMax, minPlusPointer, UINT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromInt32(int32_t *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxInt(data, nValues, &usedMin, &usedMax, minPlusPointer, INT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxInt(data, nValues, &usedMin, &usedMax, minPlusPointer, INT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromUInt32(uint32_t *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxUInt(data, nValues, &usedMin, &usedMax, minPlusPointer, UINT_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxUInt(data, nValues, &usedMin, &usedMax, minPlusPointer, UINT_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromLong(long *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxLong(data, nValues, &usedMin, &usedMax, minPlusPointer, LONG_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxLong(data, nValues, &usedMin, &usedMax, minPlusPointer, LONG_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}

void fillPixmapFromULong(unsigned long *data, long nValues, unsigned char* colormap, long nColors, unsigned char*pixmap, short method, short autoFlag, double *minValue, double *maxValue)
{
    double usedMin, usedMax, usedMinPlus;
    double *minPlusPointer;
    long i, idx;
    double delta;
    int32_t *colormapInt32;
    int32_t *pixmapInt32;

    if (method == 2)
    {
        /* Shift logarithmic */
        minPlusPointer = &usedMinPlus;
    }
    else
    {
        minPlusPointer = NULL;
    }

    if (autoFlag)
    {
        getMinMaxULong(data, nValues, &usedMin, &usedMax, minPlusPointer, ULONG_MAX);
    }
    else
    {
        if((minValue == NULL) || (maxValue == NULL))
        {
            getMinMaxULong(data, nValues, &usedMin, &usedMax, minPlusPointer, ULONG_MAX);
            if(minValue != NULL)
                usedMin = *minValue;
            if(maxValue != NULL)
                usedMax = *maxValue;
        }
    }

    /* minValue and maxValue to be modified in case of log use ... */
    FILLPIXMAP()
}
