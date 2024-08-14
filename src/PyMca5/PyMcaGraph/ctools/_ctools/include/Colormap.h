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
 * @param RGBANaNColor Pointer to 4 bytes describing the RGBA color
 *        to use for NaNs.
 *        If NULL, then the first color of the colormap is used.
 * @param RGBPixmapOut Pointer to the pixmap to fill.
 *        It is a contiguous memory block of RGBA pixels (1 byte per channel).
 *        The size of the pixmap MUST be at least 4 * length bytes.
 */
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
                   uint8_t * RGBAPixmapOut);

#endif /*__Colormap_H__*/
