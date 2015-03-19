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

#include "Colormap.h"
#include "Types.h"

/* No lrint before Microsoft Visual Studio 2013*/
#if (defined (_MSC_VER) && _MSC_VER < 1800)
#include <float.h>

#define lrint(v) ((int) (v))
#define isnan(v) _isnan(v)
#define isfinite(v) _finite(v)

#define INFINITY (DBL_MAX+DBL_MAX)
#define NAN (INFINITY-INFINITY)
#endif

#ifdef _OPENMP
#define PRAGMA_OMP(ompString) _Pragma(ompString)
#else
#define PRAGMA_OMP(ompString)
#endif /* _OPEMMP */

/* OpenMP parallel for if test is length >= 1024^2.
 * This is an arbitrary value.
 */

/* Fast log ******************************************************************/

/* Implements fast log using the fact that double is stored as m * 2^n,
 * so log2(m * 2^n) = log2(m) + n and a look-up table for log2(m).
 *
 * See: Vinyals, O. and Friedland, G.
 * A Hardware-Independent Fast Logarithm Approximation with Adjustable Accuracy.
 * In Proc. Tenth IEEE International Symposium on Multimedia (ISM 2008).
 * IEEE. pp 61-65.
 * http://dx.doi.org/10.1109/ISM.2008.83
 *
 * We use frexp (C99 but available in Visual Studio 2008) to get exponent and
 * mantissa.
 */

#define LOG_LUT_SIZE (1 << 12) /* 4096 */

static double logLUT[LOG_LUT_SIZE + 1]; /* indexLUT can overflow of 1 ! */
const double oneOverLog_2 = 1.4426950408889634;
const double oneOverLog2_10 = 0.30102999566398114;

void
initFastLog10(void)
{
    unsigned int index;

    for (index=0; index<LOG_LUT_SIZE; index++) {
        /* normFrac in [0.5, 1) */
        double normFrac = 0.5 + ((double) index) / (2.0 * LOG_LUT_SIZE);
        logLUT[index] = oneOverLog_2 * log(normFrac);
    }

    /* Cope with indexLUT == 1 overflow */
    logLUT[LOG_LUT_SIZE] = logLUT[LOG_LUT_SIZE - 1];
}


double
fastLog10(double value)
{
    double result = NAN; /* if value < 0.0 or value == NAN */

    if (value <= 0.0 || ! isfinite(value)) {
        if (value == 0.0) {
            result = - HUGE_VAL;
        }
        else if (value > 0.0) { /* i.e., value = +INFINITY */
            result = value; /* i.e. +INFINITY */
        }
    }
    else {
        int exponent;
        double mantissa; /* in [0.5, 1) unless value == 0 NaN or +/-inf */
        int indexLUT;

        mantissa = frexp(value, &exponent);
        indexLUT = lrint(LOG_LUT_SIZE * 2 * (mantissa - 0.5));
        result = oneOverLog2_10 * ((double) exponent + logLUT[indexLUT]);
    }
    return result;
}


/* Fill pixmap for each type *************************************************/

/** Macro defining typed function filling pixmap from data with a colormap.
 *
 * See colormapFillPixmap in header for a description of generated functions
 * arguments.
 *
 * colormap and pixmapOut are uint32_t to copy the 4 RGBA uint8_t at once.
 *
 * The index in the colormap is computed using casting and not rounding.
 * It provides equally spaced bins even on the edges (as opposed to rounding).
 *
 * For log10 mapping:
 * - data with value <= 0 is supported and represented with the first color
 *   of the colormap.
 * - min and max MUST be > 0.
 * - For the sake of simplicity, if min or max <= 0, the pixmap is filled
 *   with the last color of the colormap.
 *
 * For floating types, the nanColor is used for NaNs.
 *
 * @param TYPE The type of the input data of the function
 * @param FIRST_IN_LOOP Allow to inject code at the beginning of each loop.
 *        Used to add isnan test for floating types.
 */
#define FILL_PIXMAP_DEFINITION(TYPE, FIRST_IN_LOOP)\
static void fillPixmap_ ## TYPE(\
    TYPE * data,\
    unsigned long length,\
    double startValue,\
    double endValue,\
    unsigned int isLog10Mapping,\
    uint32_t * colormap,\
    unsigned int colormapLength,\
    uint32_t nanColor,\
    uint32_t * pixmapOut)\
{\
    double min, max;\
    unsigned int cmapMax = colormapLength - 1;\
\
    min = (startValue < endValue) ? startValue : endValue;\
    max = (startValue < endValue) ? endValue : startValue;\
\
    if (isLog10Mapping) {\
        unsigned long index;\
        double startLog, endLog, scale;\
\
        if (startValue <= 0.0 || endValue <= 0.0) {\
            startValue = 0.0;\
            endValue = 0.0;\
            min = 0.0;\
            max = 0.0;\
            startLog = 0.0;\
            endLog = 0.0;\
        }\
        else {\
            startLog = fastLog10(startValue);\
            endLog = fastLog10(endValue);\
        }\
\
        if (startLog != endLog) {\
            scale = ((double) colormapLength) / (endLog - startLog);\
        }\
        else {\
            scale = 0.0; /* Should never be used */\
        }\
\
        PRAGMA_OMP("omp parallel for schedule(static) if (length > 1048576)")\
        for (index=0; index<length; index++) {\
            unsigned int cmapIndex;\
            double value = (double) data[index];\
\
            FIRST_IN_LOOP\
\
            if (value >= max) {\
                cmapIndex = cmapMax;\
            }\
            else if (value <= min) {\
                cmapIndex = 0;\
            }\
            else {\
                cmapIndex = (unsigned int) (scale * (fastLog10(value) - startLog));\
                if (cmapIndex > cmapMax) {\
                    cmapIndex = cmapMax;\
                }\
            }\
\
            pixmapOut[index] = colormap[cmapIndex];\
        }\
    }\
    else {\
        unsigned long index;\
        double scale;\
\
        if (startValue != endValue) {\
            scale = ((double) colormapLength) / (endValue - startValue);\
        }\
        else {\
            scale = 0.0; /* Should never be used */\
        }\
\
        PRAGMA_OMP("omp parallel for schedule(static) if (length > 1048576)")\
        for (index=0; index<length; index++) {\
            unsigned int cmapIndex;\
            double value = (double) data[index];\
\
            FIRST_IN_LOOP\
\
            if (value >= max) {\
                cmapIndex = cmapMax;\
            }\
            else if (value <= min) {\
                cmapIndex = 0;\
            }\
            else {\
                cmapIndex = (unsigned int) (scale * (value - startValue));\
                if (cmapIndex > cmapMax) {\
                    cmapIndex = cmapMax;\
                }\
            }\
\
            pixmapOut[index] = colormap[cmapIndex];\
        }\
    }\
}

/* Code to handle NaN color for floating type */
#define HANDLE_NAN \
    if (isnan(value)) {\
        pixmapOut[index] = nanColor;\
        continue;\
    }

#define NOOP

FILL_PIXMAP_DEFINITION(float, HANDLE_NAN)
FILL_PIXMAP_DEFINITION(double, HANDLE_NAN)

FILL_PIXMAP_DEFINITION(uint8_t, NOOP)
FILL_PIXMAP_DEFINITION(int8_t, NOOP)

FILL_PIXMAP_DEFINITION(uint16_t, NOOP)
FILL_PIXMAP_DEFINITION(int16_t, NOOP)

FILL_PIXMAP_DEFINITION(uint32_t, NOOP)
FILL_PIXMAP_DEFINITION(int32_t, NOOP)

FILL_PIXMAP_DEFINITION(uint64_t, NOOP)
FILL_PIXMAP_DEFINITION(int64_t, NOOP)


/* Fill pixmap with LUT ******************************************************/

/** Macro defining typed function filling pixmap with a look-up table.
 *
 * See colormapFillPixmap in header for a description of generated functions
 * arguments.
 *
 * Faster-way to fill pixmap from (u)int8_t and (u)int16_t for large data.
 *
 * First builds a color look-up table first and then fill the pixmap with it.
 * The look-up table is built using functions generating pixmaps to
 * ensure similar results.
 *
 * @param TYPE The type of the input data of the function
 * @param TYPE_MIN The min value of TYPE
 * @param TYPE_NBELEM The number of values TYPE can represent
 */
#define FILL_PIXMAP_WITH_LUT_DEFINITION(TYPE, TYPE_MIN, TYPE_NBELEM) \
static void \
fillPixmapWithLUT_ ## TYPE(TYPE * data,\
    unsigned long length,\
    double startValue,\
    double endValue,\
    unsigned int isLog10Mapping,\
    uint32_t * colormap,\
    unsigned int colormapLength,\
    uint32_t * pixmapOut)\
{\
    unsigned long index;\
    uint32_t colorLUT[TYPE_NBELEM];\
    TYPE indices[TYPE_NBELEM];\
\
    /* Fill look-up table using colormap and an indices array */\
    for (index=0; index<TYPE_NBELEM; index++) {\
        /* Offset signed types so that indices[0] = TYPE_MIN */\
        indices[index] = (TYPE) (index + TYPE_MIN);\
    }\
\
    fillPixmap_ ## TYPE(indices,\
        TYPE_NBELEM,\
        startValue,\
        endValue,\
        isLog10Mapping,\
        colormap,\
        colormapLength,\
        0, /* NaN color is useless */\
        colorLUT);\
\
    /* Fill pixmap using look-up table */\
    PRAGMA_OMP("omp parallel for schedule(static) if (length > 1048576)")\
    for (index=0; index<length; index++) {\
        /* Revert offset for signed types */\
        pixmapOut[index] = colorLUT[data[index] - TYPE_MIN];\
    }\
}


FILL_PIXMAP_WITH_LUT_DEFINITION(int8_t, INT8_MIN, 256)
FILL_PIXMAP_WITH_LUT_DEFINITION(uint8_t, 0, 256)

FILL_PIXMAP_WITH_LUT_DEFINITION(int16_t, INT16_MIN, 65536)
FILL_PIXMAP_WITH_LUT_DEFINITION(uint16_t, 0, 65536)


/* Public API ****************************************************************/

#define CALL_FILL_PIXMAP(TYPE)\
    fillPixmap_ ## TYPE((TYPE *) data,\
        length,\
        startValue,\
        endValue,\
        isLog10Mapping,\
        colormap,\
        colormapLength,\
        nanColor, \
        pixmapOut)

#define CALL_FILL_PIXMAP_WITH_LUT(TYPE)\
    fillPixmapWithLUT_ ## TYPE((TYPE *) data,\
        length,\
        startValue,\
        endValue,\
        isLog10Mapping,\
        colormap,\
        colormapLength,\
        pixmapOut)

void
colormapFillPixmap(void * data,
                   unsigned int type,
                   unsigned long length,
                   double startValue,
                   double endValue,
                   unsigned int isLog10Mapping,
                   uint8_t * RGBAColormap,
                   unsigned int colormapLength,
                   uint8_t * RGBANaNColor,
                   uint8_t * RGBAPixmapOut)
{
    /* Convert pointers to uint32_t to copy the 4 RGBA uint8_t at once. */
    uint32_t * colormap = (uint32_t *) RGBAColormap;
    uint32_t * pixmapOut = (uint32_t *) RGBAPixmapOut;

    /* Color for NaNs, only used for floating types */
    uint32_t nanColor = 0;

    /* Choose implementation according to type, length and isLog10Mapping */

    /* Look-up table-based pixmap filling for (u)int8 and (u)int16
     * Using number of elements as a rule of thumb to choose using it */
    if ((type & SIZE_MASK) == SIZE_8 && length > UINT8_MAX) {
        if ((type & UNSIGNED) != 0) {
            CALL_FILL_PIXMAP_WITH_LUT(uint8_t);
        }
        else {
            CALL_FILL_PIXMAP_WITH_LUT(int8_t);
        }
    }
    else if ((type & SIZE_MASK) == SIZE_16 && length > UINT16_MAX) {
        if ((type  & UNSIGNED) != 0) {
            CALL_FILL_PIXMAP_WITH_LUT(uint16_t);
        }
        else {
            CALL_FILL_PIXMAP_WITH_LUT(int16_t);
        }
    }
    else { /* Generic approach */
        switch (type) {
            case (FLOATING | SIZE_32): /*float*/
                /* If NaN color is NULL, use the first color of the colormap */
                nanColor = (RGBANaNColor == 0) ? *((uint32_t *) RGBAColormap) :
                                                 *((uint32_t *) RGBANaNColor);
                CALL_FILL_PIXMAP(float);
                break;
            case (FLOATING | SIZE_64): /*double*/
                /* If NaN color is NULL, use the first color of the colormap */
                nanColor = (RGBANaNColor == 0) ? *((uint32_t *) RGBAColormap) :
                                                 *((uint32_t *) RGBANaNColor);
                CALL_FILL_PIXMAP(double);
                break;

            case (SIZE_8): /*int8_t*/
                CALL_FILL_PIXMAP(int8_t);
                break;
            case (UNSIGNED | SIZE_8): /*uint8_t*/
                CALL_FILL_PIXMAP(uint8_t);
                break;

            case (SIZE_16): /*int16_t*/
                CALL_FILL_PIXMAP(int16_t);
                break;
            case (UNSIGNED | SIZE_16): /*uint16_t*/
                CALL_FILL_PIXMAP(uint16_t);
                break;

            case (SIZE_32): /*int32_t*/
                CALL_FILL_PIXMAP(int32_t);
                break;
            case (UNSIGNED | SIZE_32): /*uint32_t*/
                CALL_FILL_PIXMAP(uint32_t);
                break;

            case (SIZE_64): /*int64_t*/
                CALL_FILL_PIXMAP(int64_t);
                break;
            case (UNSIGNED | SIZE_64): /*uint64_t*/
                CALL_FILL_PIXMAP(uint64_t);
                break;
            default:
                break;
        }
    }
}
