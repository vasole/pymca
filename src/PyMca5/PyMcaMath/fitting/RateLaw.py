#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
from numpy.linalg import inv
import sys

def linregress(x, y, sigmay=None, full_output=False):
    """
    Linear fit to a straight line following P.R. Bevington:
    
    "Data Reduction and Error Analysis for the Physical Sciences"

    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.

    sigmay : The uncertainty on the y values
        
    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    r_value : float
        correlation coefficient

    if full_output is true, an additional dictionary is returned with the keys

    sigma_slope: uncertainty on the slope

    sigma_intercept: uncertainty on the intercept

    stderr: float
        square root of the variance
    
    """
    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    y = numpy.asarray(y, dtype=numpy.float64).flatten()
    N = y.size
    if sigmay is None:
        sigmay = numpy.ones((N,), dtype=y.dtype)
    else:
        sigmay = numpy.asarray(sigmay, dtype=numpy.float64).flatten()
    w = 1.0 / (sigmay * sigmay + (sigmay == 0))

    n = S = w.sum()
    Sx = (w * x).sum()
    Sy = (w * y).sum()    
    Sxx = (w * x * x).sum()
    Sxy = ((w * x * y)).sum()
    Syy = ((w * y * y)).sum()
    # SSxx is identical to delta in Bevington book
    delta = SSxx = (S * Sxx - Sx * Sx)

    tmpValue = Sxx * Sy - Sx * Sxy
    intercept = tmpValue / delta
    SSxy = (S * Sxy - Sx * Sy)
    slope = SSxy / delta
    sigma_slope = numpy.sqrt(S /delta)
    sigma_intercept = numpy.sqrt(Sxx / delta)

    SSyy = (n * Syy - Sy * Sy)
    r_value = SSxy / numpy.sqrt(SSxx * SSyy)
    if r_value > 1.0:
        r_value = 1.0
    if r_value < -1.0:
        r_value = -1.0

    if not full_output:
        return slope, intercept, r_value

    ddict = {}
    # calculate the variance
    if N < 3:
        variance = 0.0
    else:
        variance = ((y - intercept - slope * x) ** 2).sum() / (N - 2)
    ddict["variance"] = variance
    ddict["stderr"] = numpy.sqrt(variance)
    ddict["slope"] = slope
    ddict["intercept"] = intercept
    ddict["r_value"] = r_value
    ddict["sigma_intercept"] = numpy.sqrt(Sxx / SSxx)
    ddict["sigma_slope"] = numpy.sqrt(S / SSxx)
    return slope, intercept, r_value, ddict
    
def rateLaw(x, y, sigmay=None, order=None, xmin=None, ymin=None, xmax=None, ymax=None):
    """
    Perform a fit to y following the specified rate law order

    If xmin is not None, x values will be modified by subtraction/addition to
    match the desired xmin.

    If xmax is not None, x values will be divided by their maximum value and
    multiplied by yxax

    If ymin is not None, y values will be modified by subtraction/addition to
    match the desired ymin.

    If ymax is not None, y values will be divided by the maximum value and
    multiplied by ymax
    """

    x = numpy.asarray(x, dtype=numpy.float64).flatten()
    y = numpy.asarray(y, dtype=numpy.float64).flatten()

    if xmin is not None:
        x = x - x.min() + xmin

    if ymin is not None:
        y = y - y.min() + ymin

    if xmax is not None:
        x = xmax * (x /x.max())

    if ymax is not None:
        y = ymax * (y /y.max())

    # we are going to perform a linear fit using different
    # transformations as function of the requested order.
    ddict = {}
    if order is None:
        orderList = [0, 1, 2]
    else:
        orderList = [order]
    labels = ["zero", "first", "second"]
    for orderNumber in orderList:
        label = labels[orderNumber]
        ddict["order"] = label
        if label == "zero":
            # [A] = [A]0 - kt
            yw = y
            xw = x
        elif label == "first":
            # [A] = [A]0 exp(-kt)
            # or
            # ln([A]) = ln([A]0) - kt
            idx = y > 0
            yw = numpy.log(y[idx])
            xw = x[idx]
        elif label == "second":
            # 1/[A] = 1/[A]0 + kt
            idx = (y != 0)
            yw = 1 / y[idx]
            xw = x[idx]
        else:
            raise ValueError("Unknown rate law order %s" % order)
        if yw.size < 2:
            # we cannot perform a linear fit with less than two points
            ddict[label] = None
        else:
            slope, intercept, r, full = linregress(xw, yw, full_output=True)
            ddict[label] = full
            ddict[label]["x"] = xw
            ddict[label]["y"] = yw
    if len(orderList) == 1:
        return slope, intercept, r
    else:
        return ddict

def main(argv=None):
    if argv is None:
        # first order, k = 4.820e-04
        x = [0, 600, 1200, 1800, 2400, 3000, 3600]
        y = [0.0365, 0.0274, 0.0206, 0.0157, 0.0117, 0.00860, 0.00640]
        order = "First"
        slope = "0.000482"
        print("Expected order: First")
        print("Expected slope: 0.000482")
        sigmay = None
        # second order, k = 1.3e-02
        #x = [0, 900, 1800, 3600, 6000]
        #y = [1.72e-2, 1.43e-2, 1.23e-2, 9.52e-3, 7.3e-3]        
        #order = "second"
        #slope = "0.013"
    elif len(argv) > 1:
        # assume we have got a two column csv file
        data = numpy.loadtxt(argv[1])
        x = data[:, 0]
        y = data[:, 1]
        if data.shape[1] > 2:
            sigmay = data[:, 2]
        else:
            sigmay = None
    else:
        print("RateLaw [csv_file_name]")
        return
    result = rateLaw(x, y, sigmay = sigmay)
    labels = ["Zero", "First", "Second"]
    for key in labels:
        print(key + " Order")
        print("Interceptt = ", result[key.lower()]["intercept"])
        print("Slope = ", result[key.lower()]["slope"])
        print("r value = ", result[key.lower()]["r_value"])
        print("stderr = ", result[key.lower()]["stderr"])

if __name__ == "__main__":
    main(sys.argv)
