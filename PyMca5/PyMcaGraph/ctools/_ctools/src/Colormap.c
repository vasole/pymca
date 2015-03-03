#include <math.h>

#include "Colormap.h"


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
 * @param RGBAColormap Pointer the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param RGBPixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
#define FILL_PIXMAP_LINEAR_DEFINITION(TYPE)\
static void fillPixmapLinear_ ## TYPE(\
    TYPE * data,\
    unsigned int length,\
    TYPE min,\
    TYPE max,\
    uint8_t * RGBAColormap,\
    unsigned int colormapLength,\
    uint8_t * RGBAPixmapOut)\
{\
    unsigned int index;\
    double scale;\
    uint32_t * colormap = (uint32_t *) RGBAColormap;\
    uint32_t * pixmap = (uint32_t *) RGBAPixmapOut;\
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
        if (value <= min) {\
            cmapIndex = 0;\
        }\
        else if (value >= max) {\
            cmapIndex = cmapMax;\
        }\
        else {\
            cmapIndex = (unsigned int) (scale * ((double) (value - min)));\
            if (cmapIndex > cmapMax) {\
                cmapIndex = cmapMax;\
            }\
        }\
\
        pixmap[index] = colormap[cmapIndex];\
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
        RGBAColormap,\
        colormapLength,\
        RGBAPixmapOut)


static void
fillPixmapLinear(void * data,
                 unsigned int type,
                 unsigned int length,
                 double min,
                 double max,
                 uint8_t * RGBAColormap,
                 unsigned int colormapLength,
                 uint8_t * RGBAPixmapOut)
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

//TODO use float when appropriate? alternative log10 for int?

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
 * the pixmap is filled with the first color of the colormap.
 *
 * Converts pixmap pointer to uint32_t to copy the 4 RGBA uint8_t at once.
 *
 * @param data Pointer to the data to convert to colormap.
 * @param length Number of elements in data.
 * @param min Data value to convert to the minimum of the colormap.
 *        It MUST be strictly positive.
 * @param max Data value to convert to the maximum of the colormap.
 *        It MUST be strictly positive.
 * @param RGBAColormap Pointer the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param RGBPixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
#define FILL_PIXMAP_LOG10_DEFINITION(TYPE)\
static void fillPixmapLog10_ ## TYPE(\
    TYPE * data,\
    unsigned int length,\
    TYPE min,\
    TYPE max,\
    uint8_t * RGBAColormap,\
    unsigned int colormapLength,\
    uint8_t * RGBAPixmapOut)\
{\
    unsigned int index;\
    double minLog, maxLog, scale;\
    uint32_t * colormap = (uint32_t *) RGBAColormap;\
    uint32_t * pixmap = (uint32_t *) RGBAPixmapOut;\
    unsigned int cmapMax = colormapLength - 1;\
\
    if (min <= (TYPE) 0 || max <= (TYPE) 0) {\
        min = (TYPE) 0; /* So as to fill with first color */\
        max = (TYPE) 0;\
        minLog = 0.0;\
        maxLog = 0.0;\
    }\
    else {\
        maxLog = log10((double) max);\
        minLog = log10((double) min);\
    }\
\
    if (maxLog > minLog) {\
        scale = ((double) colormapLength) / ((double) (maxLog - minLog));\
    }\
    else {\
        scale = 0.0; /* Should never be used */\
    }\
\
    for (index=0; index<length; index++) {\
        unsigned int cmapIndex;\
        TYPE value = data[index];\
\
        if (value <= min) {\
            cmapIndex = 0;\
        }\
        else if (value >= max) {\
            cmapIndex = cmapMax;\
        }\
        else {\
            cmapIndex = (unsigned int) (scale * (log10((double) value) - minLog));\
            if (cmapIndex > cmapMax) {\
                cmapIndex = cmapMax;\
            }\
        }\
\
        pixmap[index] = colormap[cmapIndex];\
    }\
}

FILL_PIXMAP_LOG10_DEFINITION(float)
FILL_PIXMAP_LOG10_DEFINITION(double)

FILL_PIXMAP_LOG10_DEFINITION(uint8_t)
FILL_PIXMAP_LOG10_DEFINITION(int8_t)

FILL_PIXMAP_LOG10_DEFINITION(uint16_t)
FILL_PIXMAP_LOG10_DEFINITION(int16_t)

FILL_PIXMAP_LOG10_DEFINITION(uint32_t)
FILL_PIXMAP_LOG10_DEFINITION(int32_t)

FILL_PIXMAP_LOG10_DEFINITION(uint64_t)
FILL_PIXMAP_LOG10_DEFINITION(int64_t)


#define CALL_FILL_PIXMAP_LOG10(TYPE)\
    fillPixmapLog10_ ## TYPE((TYPE *) data,\
        length,\
        (TYPE) min,\
        (TYPE) max,\
        RGBAColormap,\
        colormapLength,\
        RGBAPixmapOut)


static void
fillPixmapLog10(void * data,
                unsigned int type,
                unsigned int length,
                double min,
                double max,
                uint8_t * RGBAColormap,
                unsigned int colormapLength,
                uint8_t * RGBAPixmapOut)
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
    if (isLog10Mapping) {
        fillPixmapLog10(data,
                        type,
                        length,
                        min,
                        max,
                        RGBAColormap,
                        colormapLength,
                        RGBAPixmapOut);
    }
    else {
        fillPixmapLinear(data,
                         type,
                         length,
                         min,
                         max,
                         RGBAColormap,
                         colormapLength,
                         RGBAPixmapOut);
    }
}
