#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
import numpy
try:
    import PyMca.SpecfitFuns as SpecfitFuns
except ImportError:
    print("SNIPModule importing SpecfitFuns directly")
    import SpecfitFuns

snip1d = SpecfitFuns.snip1d
snip2d = SpecfitFuns.snip2d


def getSpectrumBackground(spectrum, width, roi_min=None, roi_max=None, smoothing=1):
    if roi_min is None:
        roi_min = 0
    if roi_max is None:
        roi_max = len(spectrum)
    background = spectrum * 1
    background[roi_min:roi_max] = snip1d(spectrum[roi_min:roi_max], width, smoothing)
    return background

getSnip1DBackground = getSpectrumBackground

def subtractSnip1DBackgroundFromStack(stack, width, roi_min=None, roi_max=None,  smoothing=1):
    if roi_min is None:
        roi_min = 0
    if roi_max is None:
        roi_max = len(spectrum)
    mcaIndex = -1
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
        mcaIndex = stack.info.get('McaIndex', -1)
    else:
        data = stack
    if not isinstance(data, numpy.ndarray):
        raise TypeError("This Plugin only supports numpy arrays")
    oldShape = data.shape
    if mcaIndex in [-1, len(data.shape)-1]:
        data.shape = -1, oldShape[-1]
        if roi_min > 0:
            data[:, 0:roi_min] = 0
        if roi_max < oldShape[-1]:
            data[:, roi_max:] = 0
        for i in range(data.shape[0]):
            data[i,roi_min:roi_max] -= snip1d(data[i,roi_min:roi_max],
                                              width, smoothing)
        data.shape = oldShape

    elif mcaIndex == 0:
        data.shape = oldShape[0], -1
        for i in range(data.shape[-1]):
            data[roi_min:roi_max, i] -= snip1d(data[roi_min:roi_max, i],
                                               width, smoothing)
        data.shape = oldShape
    else:
        raise ValueError("Invalid 1D index %d" % mcaIndex)
    return

def replaceStackWithSnip1DBackground(stack, width, roi_min=None, roi_max=None,  smoothing=1):
    if roi_min is None:
        roi_min = 0
    if roi_max is None:
        roi_max = len(spectrum)
    mcaIndex = -1
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
        mcaIndex = stack.info.get('McaIndex', -1)
    else:
        data = stack
    if not isinstance(data, numpy.ndarray):
        raise TypeError("This Plugin only supports numpy arrays")
    oldShape = data.shape
    if mcaIndex in [-1, len(data.shape)-1]:
        data.shape = -1, oldShape[-1]
        if roi_min > 0:
            data[:, 0:roi_min] = 0
        if roi_max < oldShape[-1]:
            data[:, roi_max:] = 0
        for i in range(data.shape[0]):
            data[i,roi_min:roi_max] = snip1d(data[i,roi_min:roi_max],
                                              width, smoothing)
        data.shape = oldShape

    elif mcaIndex == 0:
        data.shape = oldShape[0], -1
        for i in range(data.shape[-1]):
            data[roi_min:roi_max, i] = snip1d(data[roi_min:roi_max, i],
                                               width, smoothing)
        data.shape = oldShape
    else:
        raise ValueError("Invalid 1D index %d" % mcaIndex)
    return


def getImageBackground(image, width, roi_min=None, roi_max=None, smoothing=1):
    if roi_min is None:
        roi_min = (0, 0)
    if roi_max is None:
        roi_max = image.shape
    background = image * 1
    background[roi_min[0]:roi_max[0],roi_min[1]:roi_max[1]]=\
             snip2d(image[roi_min[0]:roi_max[0],roi_min[1]:roi_max[1]],
                    width,
                    smoothing)
    return background

getSnip2DBackground = getImageBackground

def subtractSnip2DBackgroundFromStack(stack, width, roi_min=None, roi_max=None,  smoothing=1, index=None):
    """
    index is the dimension used to index the images
    """
    if roi_min is None:
        roi_min = (0, 0)
    if roi_max is None:
        roi_max = image.shape
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
        if index is None:
            index = stack.info.get('McaIndex', 0)
    else:
        data = stack
    if index is None:
        index = 2
    if not isinstance(data, numpy.ndarray):
        raise TypeError("This Plugin only supports numpy arrays")
    shape = data.shape
    if index == 0:
        if (roi_min[0] > 0) or (roi_min[1] > 0):
            data[:, 0:roi_min[0], 0:roi_min[1]] = 0
        if roi_max[0] < (shape[1]-1):
            if roi_max[1] < (shape[2]-1):
                data[:, roi_max[0]:, roi_max[1]:] = 0
            else:
                data[:, roi_max[0]:, :] = 0
        else:
            if roi_max[1] < (shape[2]-1):
                data[:, :, roi_max[1]:] = 0
        for i in range(shape[index]):
            data[i,roi_min[0]:roi_max[0],roi_min[1]:roi_max[1]] -=\
                snip2d(data[i,roi_min[0]:roi_max[0],roi_min[1]:roi_max[1]], width, smoothing)
        return
    if index == 1:
        if (roi_min[0] > 0) or (roi_min[1] > 0):
            data[0:roi_min[0], :, 0:roi_min[1]] = 0
        if roi_max[0] < (shape[0]-1):
            if roi_max[1] < (shape[2]-1):
                data[roi_max[0]:, :, roi_max[1]:] = 0
            else:
                data[roi_max[0]:, :, :] = 0
        else:
            if roi_max[1] < (shape[2]-1):
                data[:, :, roi_max[1]:] = 0
        for i in range(shape[index]):
            data[roi_min[0]:roi_max[0], i, roi_min[1]:roi_max[1]] -=\
                snip2d(data[roi_min[0]:roi_max[0], i, roi_min[1]:roi_max[1]], width, smoothing)
        return
    if index == 2:
        if (roi_min[0] > 0) or (roi_min[1] > 0):
            data[0:roi_min[0], 0:roi_min[1],:] = 0
        if roi_max[0] < (shape[0]-1):
            if roi_max[1] < (shape[1]-1):
                data[roi_max[0]:, roi_max[1]:, :] = 0
            else:
                data[roi_max[0]:, :, :] = 0
        else:
            if roi_max[1] < (shape[2]-1):
                data[:, roi_max[1]:, :] = 0
        for i in range(shape[index]):
            data[roi_min[0]:roi_max[0],roi_min[1]:roi_max[1], i] -=\
                snip2d(data[roi_min[0]:roi_max[0],roi_min[1]:roi_max[1], i], width, smoothing)
        return
