/** @file Colormap.h 
 *  Conversion of data to pixmap using a colormap.
 */
#ifndef __Colormap_H__
#define __Colormap_H__

#include "Types.h"

/** Initialize the look-up table used by fastLog10.
 *
 * This function MUST be called before any call to fastLog10.
 */
void
initFastLog10(void);

/** Compute a fast log10 approximation.
 *
 * If value is negative, returns NaN.
 * If value is zero, returns - HUGE_VAL
 * If value positive infinity, returns INFINITY
 * If value is NaN, returns NaN.
 *
 * This function uses lrint and expect rounding to be: FE_TONEAREST.
 * See lrint and fegetround man for more information.
 * Note: rounding mode impacts approximation error.
 *
 * @param value
 * @return An approximation of log10(value)
 */
double
fastLog10(double value);

/** Fill a RGBA pixmap from data using the provided colormap.
 *
 * The index in the colormap is computed using casting and not rounding.
 * It provides equally spaced bins even on the edges (as opposed to rounding).
 *
 * startValue and endValue can be equal or startValue > endValue
 *
 * @param data Pointer to the data to convert to colormap.
 * @param type Bit field describing the data type.
 * @param length Number of elements in data.
 * @param startValue Data value to convert to the first color of the colormap.
 * @param endValue Data value to convert to the last color of the colormap.
 * @param isLog10Mapping True for log10 mapping, False for linear mapping.
 * @param RGBAColormap Pointer the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param RGBPixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
void
colormapFillPixmap(void * data,
                   unsigned int type,
                   unsigned int length,
                   double startValue,
                   double endValue,
                   unsigned int isLog10Mapping,
                   uint8_t * RGBAColormap,
                   unsigned int colormapLength,
                   uint8_t * RGBAPixmapOut);

#endif /*__Colormap_H__*/
