#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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
import logging
from . import SGModule


_logger = logging.getLogger(__name__)


class SimpleMath(object):
    derivateOptions = ["Single point",
                       "SG Smoothed 3 point",
                       "SG smoothed 5 point"]

    def derivate(self,xdata,ydata, xlimits=None, option=None):
        x=numpy.array(xdata, copy=False, dtype=numpy.float64)
        y=numpy.array(ydata, copy=False, dtype=numpy.float64)
        if xlimits is not None:
            i1=numpy.nonzero((xdata>=xlimits[0])&\
                               (xdata<=xlimits[1]))[0]
            x=numpy.take(x,i1)
            y=numpy.take(y,i1)

        # make sure data are strictly increasing
        deltax=x[1:] - x[:-1]
        i1=numpy.nonzero(abs(deltax)>0.0000001)[0]
        x=numpy.take(x, i1)
        y=numpy.take(y, i1)

        if option is None or option.startswith("SG"):
            minDelta = deltax[deltax > 0]
            if minDelta.size:
                minDelta = minDelta.min()
            else:
                # all points are equal
                minDelta = 1.0
            _logger.info("Using a delta between points of %f" % minDelta)
            xInter = numpy.arange(x[0]-minDelta,x[-1]+minDelta,minDelta)
            yInter = numpy.interp(xInter, x, y, left=y[0], right=y[-1])
            if len(yInter) > 499:
                _logger.info("Using 5 points interpolation")
                option = "SG smoothed 5 point"
            else:
                _logger.info("Using 3 points interpolation")
                option = "SG smoothed 3 point"
            npoints = int(option.split()[2])
            degree = 1
            order = 1
            coeff = SGModule.calc_coeff(npoints, degree, order)
            N = int(numpy.size(coeff-1)/2)
            yInterPrime = numpy.convolve(yInter, coeff, mode='valid')/minDelta
            i1 = numpy.nonzero((x>=xInter[N+1]) & (x <= xInter[-N]))[0]
            x = numpy.take(x, i1)
            result = numpy.interp(x, xInter[(N+1):-N],
                                  yInterPrime[1:],
                                  left=yInterPrime[1],
                                  right=yInterPrime[-1])
        else:
            # single point derivative
            result = numpy.zeros((len(y),), dtype=numpy.float64)
            # loop implementation
            #for i in range(1, len(y)-1):
            #    result[i] = 0.5 * \
            #                (((y[i] - y[i-1]) / (x[i] - x[i-1])) + \
            #                ((y[i+1] - y[i]) / (x[i+1] - x[i])))
            result[1:-1] = 0.5 * \
                           (((y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2])) + \
                           ((y[2:] - y[1:-1]) / (x[2:] - x[1:-1])))
            # repeat first and last value for the first and last point?
            result[0] = result[1]
            result[-1] = result[-2]
            # prefer to return what is actually defined?
            result = result[1:-1]
            x = x[1:-1]
        return x, result

    def average(self, xarr, yarr, x=None):
        """
        :param xarr : List containing x values in 1-D numpy arrays
        :param yarr : List containing y Values in 1-D numpy arrays
        :param x: x values of the final average spectrum (or None)
        :return: Average spectrum. In case of invalid input (None, None) tuple is returned.

        From the spectra given in xarr & yarr, the method determines the overlap in
        the x-range. For spectra with unequal x-ranges, the method interpolates all
        spectra on the values given in x if provided or the first curve and averages them.
        """
        if (len(xarr) != len(yarr)) or\
           (len(xarr) == 0) or (len(yarr) == 0):
            _logger.debug('specAverage -- invalid input!\n'
                          'Array lengths do not match or are 0')
            return None, None

        same = True
        if x == None:
            SUPPLIED = False
            x0 = xarr[0]
        else:
            SUPPLIED = True
            x0 = x
        for x in xarr:
            if len(x0) == len(x):
                if numpy.all(x0 == x):
                    pass
                else:
                    same = False
                    break
            else:
                same = False
                break

        xsort = []
        ysort = []
        for (x,y) in zip(xarr, yarr):
            if numpy.all(numpy.diff(x) > 0.):
                # All values sorted
                xsort.append(x)
                ysort.append(y)
            else:
                # Sort values
                mask = numpy.argsort(x)
                xsort.append(x.take(mask))
                ysort.append(y.take(mask))

        if SUPPLIED:
            xmin0 = x0.min()
            xmax0 = x0.max()
        else:
            xmin0 = xsort[0][0]
            xmax0 = xsort[0][-1]
        if (not same) or (not SUPPLIED):
            # Determine global xmin0 & xmax0
            for x in xsort:
                xmin = x.min()
                xmax = x.max()
                if xmin > xmin0:
                    xmin0 = xmin
                if xmax < xmax0:
                    xmax0 = xmax
            if xmax <= xmin:
                _logger.debug('specAverage -- \n'
                              'No overlap between spectra!')
                return numpy.array([]), numpy.array([])

        # make sure x0 is sorted
        mask = numpy.argsort(x0)
        x0 = numpy.take(x0, mask)

        # Clip xRange to maximal overlap in spectra
        mask = numpy.nonzero((x0 >= xmin0) &
                             (x0 <= xmax0))[0]
        xnew = numpy.take(x0, mask)
        ynew = numpy.zeros(len(xnew))

        # Perform average
        for (x, y) in zip(xsort, ysort):
            if same:
                ynew += y
            else:
                yinter = numpy.interp(xnew, x, y)
                ynew   += numpy.asarray(yinter)
        num = len(yarr)
        ynew /= num
        idx = numpy.isfinite(ynew)
        return xnew[idx], ynew[idx]

    def smooth(self, *var, **kw):
        """
        smooth(self,*vars,**kw)
        Usage: self.smooth(y)
               self.smooth(y=y)
               self.smooth()
        """
        if 'y' in kw:
            ydata=kw['y']
        elif len(var) > 0:
            ydata=var[0]
        else:
            ydata=self.y
        f=[0.25,0.5,0.25]
        result=numpy.array(ydata, copy=False, dtype=numpy.float64)
        if len(result) > 1:
            result[1:-1]=numpy.convolve(result,f,mode=0)
            result[0]=0.5*(result[0]+result[1])
            result[-1]=0.5*(result[-1]+result[-2])
        return result

if __name__ == "__main__":
    x = numpy.arange(100.)*0.25
    y = x*x + 2 * x
    a = SimpleMath()
    #print(a.average(x,y))
    xplot, yprime = a.derivate(x, y)
    print("Found:")
    for i in range(0,10):
        print("x = %f  y'= %f expected = %f" % (xplot[i], yprime[i], 2*xplot[i]+2))

