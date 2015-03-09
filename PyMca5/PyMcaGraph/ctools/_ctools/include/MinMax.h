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
/** @file MinMax.h
 *  Get min and max from data.
 */
#ifndef __MinMax_H__
#define __MinMax_H__

/** Get the min and max values of data.
 *
 * Optionally get the min positive value of data.
 *
 * @param data Pointer to the data to get min and max from.
 * @param type Bit field describing the data type.
 * @param length Number of elements in data.
 * @param minOut Pointer where to store the min value.
 * @param minPositiveOut Pointer where to store the strictly positive min.
 *        If not required, set to NULL.
 *        If all values of data are < 0, set to 0.
 * @param maxOut Pointer where to store the max value.
 */
void
getMinMax(void * data,
          unsigned int type,
          unsigned long length,
          double * minOut,
          double * minPositiveOut,
          double * maxOut);

#endif /*__MinMax_H__*/
