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

    It tries to be an improved version of scipystats.linregress

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

def main(argv=None):
    if argv is None:
        # Bevington data of Table 6-2
        x = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
        y = [106, 80, 98, 75, 74, 73, 49, 38, 37, 22]
        sigmay = numpy.sqrt(numpy.array(y))
        slope, intercept, r, ddict = linregress(x, y, sigmay=sigmay, full_output=True)
        print("WEIGHTED DATA")
        print("LINREGRESS results")
        print("SLOPE = ", ddict["slope"], " +/- ", ddict["sigma_slope"])
        print("INTERCEPT = ", ddict["intercept"], " +/- ", ddict["sigma_intercept"])
        from PyMca5.PyMcaMath.linalg import lstsq
        derivatives = numpy.zeros((len(y), 2))
        derivatives[:, 0] = numpy.array(x, dtype=numpy.float64)
        derivatives[:, 1] = 1.0
        print("LEAST SQUARES RESULT")
        result = lstsq(derivatives, y, sigma_b=sigmay, weight=1, uncertainties=True)
        print("SLOPE = ", result[0][0], " +/- ", result[1][0])
        print("INTERCEPT = ", result[0][1], " +/- ", result[1][1])
        print("\n\n")

        # Bevington data of Table 6-1
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [15.6, 17.5, 36.6, 43.8, 58.2, 61.6, 64.2, 70.4, 98.8]
        print("UNWEIGHTED DATA")
        slope, intercept, r, ddict = linregress(x, y, sigmay=None, full_output=True)
        print("LINREGRESS results")
        print("SLOPE = ", ddict["slope"], " +/- ", ddict["sigma_slope"])
        print("INTERCEPT = ", ddict["intercept"], " +/- ", ddict["sigma_intercept"])
        derivatives = numpy.zeros((len(y), 2))
        derivatives[:, 0] = numpy.array(x, dtype=numpy.float64)
        derivatives[:, 1] = 1.0
        print("LEAST SQUARES RESULT")
        result = lstsq(derivatives, y, sigma_b=None, weight=0, uncertainties=True)
        print("SLOPE = ", result[0][0], " +/- ", result[1][0])
        print("INTERCEPT = ", result[0][1], " +/- ", result[1][1])            
        print("\n\n")
    elif len(argv) > 1:
        # assume we have got a two (or three) column csv file
        data = numpy.loadtxt(argv[1])
        x = data[:, 0]
        y = data[:, 1]
        if data.shape[1] > 2:
            sigmay = data[:, 2]
        else:
            sigmay = None
        slope, intercept, r, ddict = linregress(x, y,
                                                sigmay=sigmay,
                                                full_output=True)
        print("LINREGRESS results")
        print("SLOPE = ", ddict["slope"], " +/- ", ddict["sigma_slope"])
        print("INTERCEPT = ", ddict["intercept"], " +/- ", ddict["sigma_intercept"])
    else:
        print("RateLaw [csv_file_name]")
        return

if __name__ == "__main__":
    main(sys.argv)
