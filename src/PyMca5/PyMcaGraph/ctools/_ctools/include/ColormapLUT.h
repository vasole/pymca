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
/* Make sure we work with 32 bit integers */
#if defined (_MSC_VER)
    /* Microsoft Visual Studio */
    #if _MSC_VER >= 1600
        /* Visual Studio 2010 and higher */
        #include <stdint.h>
    #else
        #ifndef int32_t
            #define int32_t int
            #define uint32_t unsigned int
        #endif
    #endif
#else
    #include <stdint.h>
#endif

void getMinMaxDouble(double *data, long nValues, \
                     double *minValue, double *maxValue, double *minPositive, double highestValue);
void getMinMaxFloat(float *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, float highestValue);
void getMinMaxChar(char *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, char highestValue);
void getMinMaxUChar(unsigned char *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, unsigned char highestValue);
void getMinMaxShort(short *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, short highestValue);
void getMinMaxUShort(unsigned short *data, long nValues, \
                     double *minValue, double *maxValue, double *minPositive, unsigned short highestValue);
void getMinMaxInt32(int32_t *data, long nValues, \
                  double *minValue, double *maxValue, double *minPositive, int32_t highestValue);
void getMinMaxUInt32(uint32_t *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, uint32_t highestValue);
void getMinMaxInt(int *data, long nValues, \
                  double *minValue, double *maxValue, double *minPositive, int32_t highestValue);
void getMinMaxUInt(unsigned int *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, uint32_t highestValue);
void getMinMaxLong(long *data, long nValues,\
                    double *minValue, double *maxValue, double *minPositive, long highestValue);
void getMinMaxULong(unsigned long *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, unsigned long highestValue);

void fillPixmapFromDouble(double *data, long nValues, unsigned char *colormap, long nColors, \
                          unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromFloat(float *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromChar(char *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUChar(unsigned char *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromShort(short *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUShort(unsigned short *data, long nValues, unsigned char *colormap, long nColors, \
                          unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromInt(int *data, long nValues, unsigned char *colormap, long nColors, \
                       unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUInt(unsigned int *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromInt32(int32_t *data, long nValues, unsigned char *colormap, long nColors, \
                       unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUInt32(uint32_t *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromLong(long *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromULong(unsigned long *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
