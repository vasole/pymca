#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
import numpy

try:
    from PyMca import Plugin1DBase
except ImportError:
    print("WARNING:MedianFilterScanPlugin import from somewhere else")
    from . import Plugin1DBase

from PyMca import SpecfitFuns

class AlignmentScanPlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
        Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
        self.__randomization = True
        self.__methodKeys = []
        self.methodDict = {}
        text = "FFT based alignment\n"
        info = text
        icon = None
        function = self.fftAlignment
        method = "FFT Alignment"
        self.methodDict[method] = [function,
                                   info,
                                   icon]
        self.__methodKeys.append(method)
        
    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...        
        """
        if self.__randomization:
            return self.__methodKeys[0:1] +  self.__methodKeys[2:]
        else:
            return self.__methodKeys[1:]

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        return self.methodDict[name][0]()

    def fftAlignment(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        if nCurves < 2:
            raise ValueError("At least 2 curves needed")
            return

        # get active curve            
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            activeCurve = curves[0]

        # apply between graph limits
        x0 = activeCurve[0][:]
        y0 = activeCurve[1][:]
        xmin, xmax =self.getGraphXLimits()
        idx = numpy.nonzero((x0 >= xmin) & (x0 <= xmax))[0]
        x0 = numpy.take(x0, idx)
        y0 = numpy.take(y0, idx)

        #sort the values
        idx = numpy.argsort(x0, kind='mergesort')
        x0 = numpy.take(x0, idx)
        y0 = numpy.take(y0, idx)

        #remove duplicates
        x0 = x0.ravel()
        idx = numpy.nonzero((x0[1:] > x0[:-1]))[0]
        x0 = numpy.take(x0, idx)
        y0 = numpy.take(y0, idx)

        #make sure values are regularly spaced
        xi = numpy.linspace(x0[0], x0[-1], len(idx)).reshape(-1, 1)
        yi = SpecfitFuns.interpol([x0], y0, xi, y0.min())
        x0 = xi
        y0 = yi

        y0.shape = -1
        fft0 = numpy.fft.fft(y0)
        y0.shape = -1, 1
        x0.shape = -1, 1
        nChannels = x0.shape[0]

        # built a couple of temporary array of spectra for handy access
        tmpArray = numpy.zeros((nChannels, nCurves), numpy.float)
        fftList = []
        shiftList = []
        curveList = []
        i = 0
        for idx in range(nCurves):
            x, y, legend, info = curves[idx][0:4]
            #sort the values
            x = x[:]
            idx = numpy.argsort(x, kind='mergesort')
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)

            #take the portion of x between limits
            idx = numpy.nonzero((x>=xmin) & (x<=xmax))[0]
            if not len(idx):
                # no overlap
                continue
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)

            #remove duplicates
            x = x.ravel()
            idx = numpy.nonzero((x[1:] > x[:-1]))[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
            x.shape = -1, 1
            if numpy.allclose(x, x0):
                # no need for interpolation
                pass
            else:
                # we have to interpolate
                x.shape = -1
                y.shape = -1
                xi = x0[:]
                y = SpecfitFuns.interpol([x], y, xi, y0.min())
            y.shape = -1
            tmpArray[:, i] = y
            i += 1

            # now calculate the shift
            ffty = numpy.fft.fft(y)
            fftList.append(ffty)
            if 0:
                self.addCurve(x,
                      y,
                      legend="NEW Y%d" % i,
                      info=None,
                      replot=True,
                      replace=False)
            elif numpy.allclose(fft0, ffty):
                shiftList.append(0.0)
            else:
                shift = numpy.fft.ifft(fft0 * ffty.conjugate()).real
                shift2 = numpy.zeros(shift.shape, dtype=shift.dtype)
                m = shift2.size//2
                shift2[m:] = shift[:-m]
                shift2[:m] = shift[-m:]
                if 0:
                    self.addCurve(numpy.arange(len(shift2)),
                          shift2,
                          legend="SHIFT",
                          info=None,
                          replot=True,
                          replace=False)
                threshold = 0.50*shift2.max()
                #threshold = shift2.mean()
                idx = numpy.nonzero(shift2 > threshold)[0]
                #print("max indices = %d" % (m - idx))
                shift = (shift2[idx] * idx/shift2[idx].sum()).sum()
                #print("shift = ", shift - m, "in x units = ", (shift - m) * (x[1]-x[0]))

                # shift the curve
                shift = (shift - m) * (x[1]-x[0])
                x.shape = -1
                y = numpy.fft.ifft(numpy.exp(-2.0*numpy.pi*numpy.sqrt(numpy.complex(-1))*\
                                numpy.fft.fftfreq(len(x), d=x[1]-x[0])*shift)*numpy.fft.fft(y))
                y = y.real
                y.shape = -1
            curveList.append([x, y, legend + "SHIFT", False, False])
        tmpArray = None
        curveList[-1][-2] = True
        curveList[-1][-1] = False
        x, y, legend, replot, replace = curveList[0]
        self.addCurve(x, y, legend=legend, replot=True, replace=True)
        for i in range(1, len(curveList)):
            x, y, legend, replot, replace = curveList[i]
            self.addCurve(x,
                          y,
                          legend=legend,
                          info=None,
                          replot=replot,
                          replace=False)
        return

        
        # now get the final spectrum
        y = medianSpectra.sum(axis=1) / nCurves
        x0.shape = -1
        y.shape = x0.shape
        legend = "%d Median from %s to %s" % (width,
                                              curves[0][2],
                                              curves[-1][2])
        self.addCurve(x0,
                      y,
                      legend=legend,
                      info=None,
                      replot=True,
                      replace=True)

MENU_TEXT = "Alignment Plugin"
def getPlugin1DInstance(plotWindow, **kw):
    ob = AlignmentScanPlugin(plotWindow)
    return ob

if __name__ == "__main__":
    from PyMca import PyMcaQt as qt
    app = qt.QApplication([])
    from PyMca.Plot1DQwt import Plot1DQwt as Plot1D
    i = numpy.arange(1000.)
    y1 = 10.0 + 5000.0 * numpy.exp(-0.01*(i-50)**2)
    y2 = 10.0 + 5000.0 * numpy.exp(-((i-55)/5.)**2)
    plot = Plot1D()
    plot.addCurve(i, y1, "y1")
    plot.addCurve(i, y2, "y2")
    plugin = getPlugin1DInstance(plot)
    for method in plugin.getMethods():
        print(method, ":", plugin.getMethodToolTip(method))
    plugin.applyMethod(plugin.getMethods()[0])
    curves = plugin.getAllCurves()
    #for curve in curves:
    #    print(curve[2])
    print("LIMITS = ", plugin.getGraphYLimits())
    #app = qt.QApplication()
    plot.show()
    app.exec_()
