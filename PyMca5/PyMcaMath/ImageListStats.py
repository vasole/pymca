# /*##########################################################################
# Copyright (C) 2023-2024 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy


def arrayListMeanRatioAndMedianRatio(imageList, mask=None):
    # the input imageList can be a 3D array or a list of images
    # the mask accounts for selected pixels
    # non-finite values are excluded

    if mask is not None:
        mask = mask.flatten()
    result_mean = numpy.zeros((len(imageList), len(imageList)), dtype=numpy.float64)
    result_median = numpy.zeros((len(imageList), len(imageList)), dtype=numpy.float64)
    for i in range(len(imageList)):
        if mask is None:
            image0 = imageList[i].flatten()
        else:
            image0 = imageList[i].flatten()[mask > 0]
        for j in range(len(imageList)):
            if mask is None:
                image1 = imageList[j].flatten()
            else:
                image1 = imageList[j].flatten()[mask > 0]
            goodIndex = numpy.isfinite(image0) & numpy.isfinite(image1)
            image0 = image0[goodIndex]
            image1 = image1[goodIndex]
            mean_ratio = image0 / numpy.asarray(image1, dtype=numpy.float64)
            goodIndex = numpy.isfinite(mean_ratio)
            mean_ratio = mean_ratio[goodIndex]
            median_ratio = numpy.median(mean_ratio)
            mean_ratio = mean_ratio.sum() / mean_ratio.size
            result_mean[i, j] = mean_ratio
            result_median[i, j] = median_ratio
    return result_mean, result_median


def arrayListPearsonCorrelation(imageList, mask=None):
    # the input imageList can be a 3D array or a list of images
    # the mask accounts for selected pixels
    # non-finite values are excluded
    if mask is not None:
        mask = mask.flatten()
    correlation = numpy.zeros((len(imageList), len(imageList)), dtype=numpy.float64)
    for i in range(len(imageList)):
        if mask is None:
            image0 = imageList[i].flatten()
        else:
            image0 = imageList[i].flatten()[mask > 0]
        for j in range(len(imageList)):
            if mask is None:
                image1 = imageList[j].flatten()
            else:
                image1 = imageList[j].flatten()[mask > 0]
            goodIndex = numpy.isfinite(image0) & numpy.isfinite(image1)
            image0 = image0[goodIndex]
            image1 = image1[goodIndex]
            image0_mean = image0.sum(dtype=numpy.float64) / image0.size
            image1_mean = image1.sum(dtype=numpy.float64) / image0.size
            image0 = image0 - image0_mean
            image1 = image1 - image1_mean
            cov = numpy.sum(image0 * image1) / image0.size
            stdImage0 = (numpy.sum(image0 * image0) / image0.size) ** 0.5
            stdImage1 = (numpy.sum(image1 * image1) / image1.size) ** 0.5
            correlation[i, j] = cov / (stdImage0 * stdImage1)
    return correlation
