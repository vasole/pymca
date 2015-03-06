#include <math.h>
#include <stdlib.h>

#include "Colormap.h"


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
const double oneOverLog2 = 0x1.71547652b82fep+0;
const double oneOverLog10 = 0x1.34413509f79fep-2;

void
initFastLog10(void)
{
    unsigned int index;

    for (index=0; index<LOG_LUT_SIZE; index++) {
        /* normFrac in [0.5, 1) */
        double normFrac = 0.5 + ((double) index) / (2.0 * LOG_LUT_SIZE);
        logLUT[index] = oneOverLog2 * log(normFrac);
    }

    /* Cope with indexLUT == 1 overflow */
    logLUT[LOG_LUT_SIZE] = logLUT[LOG_LUT_SIZE - 1];
}


double
fastLog10(double value)
{
    if (value <= 0.0 || ! isfinite(value)) {
        if (value < 0.0) {
            return NAN;
        }
        else if (value == 0.0) {
            return - HUGE_VAL;
        }
        else if (isinf(value)) {
            return INFINITY;
        }
        else if (isnan(value)) {
            return NAN;
        }
    }
    else {
        int exponent;
        double mantissa; /* in [0.5, 1) unless value == 0 NaN or +/-inf */
        int indexLUT;

        mantissa = frexp(value, &exponent);
        indexLUT = lrint(LOG_LUT_SIZE * 2 * (mantissa - 0.5));
        return oneOverLog10 * ((double) exponent + logLUT[indexLUT]);
    }
}


/* Colormap with linear mapping **********************************************/

/** Fill a RGBA pixmap from data using the colormap with linear mapping.
 *
 * This functions is defined for different types.
 * The index in the colormap is computed using casting and not rounding.
 * It provides equally spaced bins even on the edges (as opposed to rounding).
 *
 * Converts pixmap pointer to uint32_t to copy the 4 RGBA uint8_t at once.
 *
 * @param data Pointer to the data to convert to colormap.
 * @param length Number of elements in data.
 * @param min Data value to convert to the minimum of the colormap.
 * @param max Data value to convert to the maximum of the colormap.
 * @param colormap Pointer the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param pixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
#define FILL_PIXMAP_LINEAR_DEFINITION(TYPE)\
static void fillPixmapLinear_ ## TYPE(\
    TYPE * data,\
    unsigned int length,\
    TYPE min,\
    TYPE max,\
    uint32_t * colormap,\
    unsigned int colormapLength,\
    uint32_t * pixmapOut)\
{\
    unsigned int index;\
    double scale;\
    unsigned int cmapMax = colormapLength - 1;\
\
    if (max > min) {\
        scale = ((double) colormapLength) / ((double) (max - min));\
    }\
    else {\
        scale = 0.0; /* Should never be used */\
    }\
\
    for (index=0; index<length; index++) {\
        unsigned int cmapIndex;\
        TYPE value = data[index];\
\
        if (value >= max) {\
            cmapIndex = cmapMax;\
        }\
        else if (value <= min) {\
            cmapIndex = 0;\
        }\
        else {\
            cmapIndex = (unsigned int) (scale * ((double) (value - min)));\
            if (cmapIndex > cmapMax) {\
                cmapIndex = cmapMax;\
            }\
        }\
\
        pixmapOut[index] = colormap[cmapIndex];\
    }\
}


FILL_PIXMAP_LINEAR_DEFINITION(float)
FILL_PIXMAP_LINEAR_DEFINITION(double)

FILL_PIXMAP_LINEAR_DEFINITION(uint8_t)
FILL_PIXMAP_LINEAR_DEFINITION(int8_t)

FILL_PIXMAP_LINEAR_DEFINITION(uint16_t)
FILL_PIXMAP_LINEAR_DEFINITION(int16_t)

FILL_PIXMAP_LINEAR_DEFINITION(uint32_t)
FILL_PIXMAP_LINEAR_DEFINITION(int32_t)

FILL_PIXMAP_LINEAR_DEFINITION(uint64_t)
FILL_PIXMAP_LINEAR_DEFINITION(int64_t)


#define CALL_FILL_PIXMAP_LINEAR(TYPE)\
    fillPixmapLinear_ ## TYPE((TYPE *) data,\
        length,\
        (TYPE) min,\
        (TYPE) max,\
        colormap,\
        colormapLength,\
        pixmapOut)


static void
fillPixmapLinear(void * data,
                 unsigned int type,
                 unsigned int length,
                 double min,
                 double max,
                 uint32_t * colormap,
                 unsigned int colormapLength,
                 uint32_t * pixmapOut)
{
    switch (type) {
        case (FLOATING | SIZE_32): /*float*/
            CALL_FILL_PIXMAP_LINEAR(float);
            break;
        case (FLOATING | SIZE_64): /*double*/
            CALL_FILL_PIXMAP_LINEAR(double);
            break;

        case (SIZE_8): /*int8_t*/
            CALL_FILL_PIXMAP_LINEAR(int8_t);
            break;
        case (UNSIGNED | SIZE_8): /*uint8_t*/
           CALL_FILL_PIXMAP_LINEAR(uint8_t);
           break;

        case (SIZE_16): /*int16_t*/
            CALL_FILL_PIXMAP_LINEAR(int16_t);
            break;
        case (UNSIGNED | SIZE_16): /*uint16_t*/
            CALL_FILL_PIXMAP_LINEAR(uint16_t);
            break;

        case (SIZE_32): /*int32_t*/
            CALL_FILL_PIXMAP_LINEAR(int32_t);
            break;
        case (UNSIGNED | SIZE_32): /*uint32_t*/
            CALL_FILL_PIXMAP_LINEAR(uint32_t);
            break;

        case (SIZE_64): /*int64_t*/
            CALL_FILL_PIXMAP_LINEAR(int64_t);
            break;
        case (UNSIGNED | SIZE_64): /*uint64_t*/
            CALL_FILL_PIXMAP_LINEAR(uint64_t);
            break;
        default:
            break;
    }
}


/* Colormap with log10 mapping ***********************************************/

/** Fill a RGBA pixmap from data using the colormap with log10 mapping.
 *
 * This functions is defined for different types.
 * The index in the colormap is computed using casting and not rounding.
 * It provides equally spaced bins even on the edges (as opposed to rounding).
 *
 * data with value <= 0 is supported and represented with the first color
 * of the colormap.
 *
 * min and max MUST be > 0.
 * For the sake of simplicity, if min or max <= 0,
 * the pixmap is filled with the last color of the colormap.
 *
 * Converts pixmap pointer to uint32_t to copy the 4 RGBA uint8_t at once.
 *
 * @param data Pointer to the data to convert to colormap.
 * @param length Number of elements in data.
 * @param min Data value to convert to the minimum of the colormap.
 *        It MUST be strictly positive.
 * @param max Data value to convert to the maximum of the colormap.
 *        It MUST be strictly positive.
 * @param colormap Pointer to the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param pixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
#define FILL_PIXMAP_LOG10_DEFINITION(TYPE, LOG10_FUNC, LOG10_TYPE)\
static void fillPixmapLog10_ ## TYPE(\
    TYPE * data,\
    unsigned int length,\
    TYPE min,\
    TYPE max,\
    uint32_t * colormap,\
    unsigned int colormapLength,\
    uint32_t * pixmapOut)\
{\
    unsigned int index;\
    LOG10_TYPE minLog, maxLog, scale;\
    unsigned int cmapMax = colormapLength - 1;\
\
    if (min <= (TYPE) 0 || max <= (TYPE) 0) {\
        min = (TYPE) 0;\
        max = (TYPE) 0;\
        minLog = 0.0;\
        maxLog = 0.0;\
    }\
    else {\
        maxLog = LOG10_FUNC((LOG10_TYPE) max);\
        minLog = LOG10_FUNC((LOG10_TYPE) min);\
    }\
\
    if (maxLog > minLog) {\
        scale = ((LOG10_TYPE) colormapLength) / ((LOG10_TYPE)(maxLog-minLog));\
    }\
    else {\
        scale = 0.0; /* Should never be used */\
    }\
\
    for (index=0; index<length; index++) {\
        unsigned int cmapIndex;\
        TYPE value = data[index];\
\
        if (value >= max) {\
            cmapIndex = cmapMax;\
        }\
        else if (value <= min) {\
            cmapIndex = 0;\
        }\
        else {\
            cmapIndex = (unsigned int) (scale * (LOG10_FUNC((LOG10_TYPE) value) - minLog));\
            if (cmapIndex > cmapMax) {\
                cmapIndex = cmapMax;\
            }\
        }\
\
        pixmapOut[index] = colormap[cmapIndex];\
    }\
}

FILL_PIXMAP_LOG10_DEFINITION(float, fastLog10, double) //log10f, float)
FILL_PIXMAP_LOG10_DEFINITION(double, fastLog10, double)

FILL_PIXMAP_LOG10_DEFINITION(uint8_t, log10f, float)
FILL_PIXMAP_LOG10_DEFINITION(int8_t, log10f, float)

FILL_PIXMAP_LOG10_DEFINITION(uint16_t, log10f, float)
FILL_PIXMAP_LOG10_DEFINITION(int16_t, log10f, float)

FILL_PIXMAP_LOG10_DEFINITION(uint32_t, fastLog10, double)
FILL_PIXMAP_LOG10_DEFINITION(int32_t, fastLog10, double)

FILL_PIXMAP_LOG10_DEFINITION(uint64_t, fastLog10, double)
FILL_PIXMAP_LOG10_DEFINITION(int64_t, fastLog10, double)


#define CALL_FILL_PIXMAP_LOG10(TYPE)\
    fillPixmapLog10_ ## TYPE((TYPE *) data,\
        length,\
        (TYPE) min,\
        (TYPE) max,\
        colormap,\
        colormapLength,\
        pixmapOut)


static void
fillPixmapLog10(void * data,
                unsigned int type,
                unsigned int length,
                double min,
                double max,
                uint32_t * colormap,
                unsigned int colormapLength,
                uint32_t * pixmapOut)
{
    switch (type) {
        case (FLOATING | SIZE_32): /*float*/
            CALL_FILL_PIXMAP_LOG10(float);
            break;
        case (FLOATING | SIZE_64): /*double*/
            CALL_FILL_PIXMAP_LOG10(double);
            break;

        case (SIZE_8): /*int8_t*/
            CALL_FILL_PIXMAP_LOG10(int8_t);
            break;
        case (UNSIGNED | SIZE_8): /*uint8_t*/
           CALL_FILL_PIXMAP_LOG10(uint8_t);
           break;

        case (SIZE_16): /*int16_t*/
            CALL_FILL_PIXMAP_LOG10(int16_t);
            break;
        case (UNSIGNED | SIZE_16): /*uint16_t*/
            CALL_FILL_PIXMAP_LOG10(uint16_t);
            break;

        case (SIZE_32): /*int32_t*/
            CALL_FILL_PIXMAP_LOG10(int32_t);
            break;
        case (UNSIGNED | SIZE_32): /*uint32_t*/
            CALL_FILL_PIXMAP_LOG10(uint32_t);
            break;

        case (SIZE_64): /*int64_t*/
            CALL_FILL_PIXMAP_LOG10(int64_t);
            break;
        case (UNSIGNED | SIZE_64): /*uint64_t*/
            CALL_FILL_PIXMAP_LOG10(uint64_t);
            break;
        default:
            break;
    }
}


/* Faster path for uint8_t and uint16_t **************************************/

/** Fill a color look-up table from a colormap.
 *
 * Meant to be used for filling pixmap from uint8, uint16 data.
 *
 * @param min The data value to associate to the first color of the colormap.
 * @param max The data value to associate to the last color of the colormap.
 * @param isLog10Mapping True for log10 mapping, False for linear mapping.
 * @param colormap Pointer to the colormap array (4 bytes per color).
 * @param colormapLength Number of colors in the colormap.
 * @param colorLUTLength Number of entries in the LUT (4 bytes per entry).
 *        Typically 256 or 65536 entries.
 * @param colorLUTOut Pointer to the color LUT to fill.
 */
static void
fillColorLUT(double min,
             double max,
             unsigned int isLog10Mapping,
             uint32_t * colormap,
             unsigned int colormapLength,
             unsigned int colorLUTLength,
             uint32_t * colorLUTOut)
{
    int index;
    const unsigned int cmapMaxIndex = colormapLength - 1;

    /* Fill value below or equal min*/
    for (index=0; index<=(int) min; index++) {
        colorLUTOut[index] = colormap[0];
    }

    /* Fill value in ]min, max[ using colormap */
    if (max > min) {
        int maxIndex = (max >= (double) colorLUTLength) ?
                       colorLUTLength : (int) max; /* Avoid overflow */
        float value = (float) index; /* Avoid casting index during the loop */

        if (isLog10Mapping) {
            /* Log mapping */
            float minLog = log10f((float) min);
            float scale = ((float) colormapLength) /
                          (log10f((float) max) - minLog);

            for (; index<maxIndex; index++) {
                unsigned int cmapIndex =
                    (unsigned int) (scale * (log10f(value) - minLog));
                if (cmapIndex > cmapMaxIndex) {
                    cmapIndex = cmapMaxIndex;
                }
                colorLUTOut[index] = colormap[cmapIndex];

                value += 1.0;
            }
        }
        else {
            /* Linear mapping */
            float minF = (float) min;
            float scale = ((float) colormapLength) / ((float) (max - min));

            for (; index<maxIndex; index++) {
                unsigned int cmapIndex =
                    (unsigned int) (scale * (value - minF));
                if (cmapIndex > cmapMaxIndex) {
                    cmapIndex = cmapMaxIndex;
                }
                colorLUTOut[index] = colormap[cmapIndex];

                value += 1.0;
            }
        }
    }

    /* Fill value above or equal max */
    for (; index<colorLUTLength; index++) {
        colorLUTOut[index] = colormap[cmapMaxIndex];
    }
}

/** Faster-way to fill pixmap from uint8_t and uint16_t for large data.
 *
 * Builds a color look-up table first and then fill the pixmap with it.
 *
 * Using malloc/free rather than static array to allow multhreading.
 *
 * WARNING: Only supports uint8_t and uint16_t.
 */
#define FILL_PIXMAP_WITH_LUT_DEFINITION(TYPE) \
static void \
fillPixmapWithLUT_ ## TYPE(TYPE * data,\
    unsigned int length,\
    double min,\
    double max,\
    unsigned int isLog10Mapping,\
    uint32_t * colormap,\
    unsigned int colormapLength,\
    uint32_t * pixmapOut)\
{\
    uint32_t * colorLUT;\
    const unsigned int colorLUTLength = (1 << (sizeof(TYPE) * 8));\
\
    colorLUT = (uint32_t *) malloc(colorLUTLength * sizeof(uint32_t));\
    if (colorLUT == NULL) {\
        abort();\
    }\
\
    /* Fill look-up table using colormap */\
    fillColorLUT(min, max, isLog10Mapping,\
                 colormap, colormapLength,\
                 colorLUTLength, colorLUT);\
\
    /* Fill pixmap using look-up table */\
    {\
        unsigned int index;\
\
        for (index=0; index<length; index++) {\
            pixmapOut[index] = colorLUT[data[index]];\
        }\
    }\
\
    free(colorLUT);\
}

FILL_PIXMAP_WITH_LUT_DEFINITION(uint8_t)
FILL_PIXMAP_WITH_LUT_DEFINITION(uint16_t)


/* Public API ****************************************************************/

void
colormapFillPixmap(void * data,
                   unsigned int type,
                   unsigned int length,
                   double min,
                   double max,
                   uint8_t * RGBAColormap,
                   unsigned int colormapLength,
                   unsigned int isLog10Mapping,
                   uint8_t * RGBAPixmapOut)
{
    /* Convert pointers to uint32_t to copy the 4 RGBA uint8_t at once. */
    uint32_t * colormap = (uint32_t *) RGBAColormap;
    uint32_t * pixmap = (uint32_t *) RGBAPixmapOut;

    /* Look-up table-based pixmap filling for uint8 and uint16
     * Using number of elements as a rule of thumb to choose using it */
    if (type == (UNSIGNED | SIZE_8) && length > 256) { /* uint8_t */
        fillPixmapWithLUT_uint8_t((uint8_t *) data,
                                  length,
                                  min,
                                  max,
                                  isLog10Mapping,
                                  colormap,
                                  colormapLength,
                                  pixmap);
    }
    else if (type == (UNSIGNED | SIZE_16) && length > 65536) {
        fillPixmapWithLUT_uint16_t((uint16_t *) data,
                                   length,
                                   min,
                                   max,
                                   isLog10Mapping,
                                   colormap,
                                   colormapLength,
                                   pixmap);
    }
    else { /* Generic approach */
        if (isLog10Mapping) {
            fillPixmapLog10(data,
                            type,
                            length,
                            min,
                            max,
                            colormap,
                            colormapLength,
                            pixmap);
        }
        else {
            fillPixmapLinear(data,
                             type,
                             length,
                             min,
                             max,
                             colormap,
                             colormapLength,
                             pixmap);
        }
    }
}
