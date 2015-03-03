/** @file Colormap.h 
 *  Conversion of data to pixmap using a colormap.
 */
#ifndef __Colormap_H__
#define __Colormap_H__

#include "Types.h"

/** Fill a RGBA pixmap from data using the provided colormap.
 *
 * The index in the colormap is computed using casting and not rounding.
 * It provides equally spaced bins even on the edges (as opposed to rounding).
 *
 * @param data Pointer to the data to convert to colormap.
 * @param type Bit field describing the data type.
 * @param length Number of elements in data.
 * @param min Data value to convert to the minimum of the colormap.
 * @param max Data value to convert to the maximum of the colormap.
 * @param RGBAColormap Pointer the RGBA colormap.
 *        It is a contiguous array of RGBA values (1 byte per channel).
 * @param colormapLength The number of values in the colormap.
 * @param isLog10Mapping True for log10 mapping, False for linear mapping.
 * @param RGBPixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
void
colormapFillPixmap(void * data,
                   unsigned int type,
                   unsigned int length,
                   double min,
                   double max,
                   uint8_t * RGBAColormap,
                   unsigned int colormapLength,
                   unsigned int isLog10Mapping,
                   uint8_t * RGBAPixmapOut);

#endif /*__Colormap_H__*/
