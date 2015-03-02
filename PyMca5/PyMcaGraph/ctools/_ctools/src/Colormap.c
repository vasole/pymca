#include "Colormap.h"

/** Fill a RGBA pixmap from data using the provided colormap.
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
#define FILL_PIXMAP_DEFINITION(NAME, TYPE)\
static void NAME(\
    TYPE * data,\
    unsigned int length,\
    TYPE min,\
    TYPE max,\
    uint8_t * RGBAColormap,\
    unsigned int colormapLength,\
    uint8_t * RGBAPixmapOut)\
{\
    unsigned int index;\
    uint32_t * colormap = (uint32_t *) RGBAColormap;\
    uint32_t * pixmap = (uint32_t *) RGBAPixmapOut;\
    unsigned int cmapMax = colormapLength - 1;\
    double scale = ((double) colormapLength) / ((double) (max - min));\
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

FILL_PIXMAP_DEFINITION(fillPixmap_float, float);
FILL_PIXMAP_DEFINITION(fillPixmap_double, double);

FILL_PIXMAP_DEFINITION(fillPixmap_uint8_t, uint8_t);
FILL_PIXMAP_DEFINITION(fillPixmap_int8_t, int8_t);

FILL_PIXMAP_DEFINITION(fillPixmap_uint16_t, uint16_t);
FILL_PIXMAP_DEFINITION(fillPixmap_int16_t, int16_t);

FILL_PIXMAP_DEFINITION(fillPixmap_uint32_t, uint32_t);
FILL_PIXMAP_DEFINITION(fillPixmap_int32_t, int32_t);

FILL_PIXMAP_DEFINITION(fillPixmap_uint64_t, uint64_t);
FILL_PIXMAP_DEFINITION(fillPixmap_int64_t, int64_t);


#define CALL_FILL_PIXMAP(TYPE)\
    fillPixmap_ ## TYPE((TYPE *) data,\
        length,\
        (TYPE) min,\
        (TYPE) max,\
        RGBAColormap,\
        colormapLength,\
        RGBAPixmapOut)

void
colormapFillPixmap(void * data,
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
            CALL_FILL_PIXMAP(float);
            break;
        case (FLOATING | SIZE_64): /*double*/
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
