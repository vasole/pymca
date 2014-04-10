#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import numpy
try:
    from PyMca5 import SGModule
except ImportError:
    print("SimpleMath importing SGModule directly")
    import SGModule
    
class SimpleMath(object):
    def derivate(self,xdata,ydata, xlimits=None):
        x=numpy.array(xdata, copy=False, dtype=numpy.float)
        y=numpy.array(ydata, copy=False, dtype=numpy.float)
        if xlimits is not None:
            i1=numpy.nonzero((xdata>=xlimits[0])&\
                               (xdata<=xlimits[1]))[0]
            x=numpy.take(x,i1)
            y=numpy.take(y,i1)
        i1 = numpy.argsort(x)
        x=numpy.take(x,i1)
        y=numpy.take(y,i1)  
        deltax=x[1:] - x[:-1]
        i1=numpy.nonzero(abs(deltax)>0.0000001)[0]
        x=numpy.take(x, i1)
        y=numpy.take(y, i1)
        minDelta = deltax.min()
        xInter = numpy.arange(x[0]-minDelta,x[-1]+minDelta,minDelta)
        yInter = numpy.interp(xInter, x, y, left=y[0], right=y[-1])
        if len(yInter) > 499:
            npoints = 5
        else:
            npoints = 3
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
            if DEBUG:
                print('specAverage -- invalid input!')
                print('Array lengths do not match or are 0')
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
                if DEBUG:
                    print('specAverage -- ')
                    print('No overlap between spectra!')
                return numpy.array([]), numpy.array([])

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
        return xnew, ynew

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
        result=numpy.array(ydata, copy=False, dtype=numpy.float)
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

