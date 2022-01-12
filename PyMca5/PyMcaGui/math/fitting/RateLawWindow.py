#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import sys
import numpy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGui import PlotWindow
from PyMca5.PyMcaMath.fitting import RateLaw

class RateLawWindow(qt.QMainWindow):
    def __init__(self, parent=None, backend=None):
        super(RateLawWindow, self).__init__(parent)
        self.setWindowTitle("RateLaw Window")
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)
        self.mdiArea = RateLawMdiArea(self, backend=backend)
        self.setCentralWidget(self.mdiArea)

        # connect
        #self.mdiArea.sigRateLawMdiAreaSignal.connect(self._update)

    def setSpectrum(self, x, y, **kw):
        self.mdiArea.setSpectrum(x, y, **kw)

class RateLawMdiArea(qt.QMdiArea):
    sigRateLawMdiAreaSignal = qt.pyqtSignal(object)
    def __init__(self, parent=None, backend=None):
        super(RateLawMdiArea, self).__init__(parent)
        self._windowDict = {}
        self._windowList = ["Original", "Zero",
                            "First", "Second"]
        self._windowList.reverse()
        for title in self._windowList:
            plot = PlotWindow.PlotWindow(self,
                                         position=True,
                                         backend=backend)
            plot.setWindowTitle(title)
            self.addSubWindow(plot)
            self._windowDict[title] = plot
            plot.setDataMargins(0, 0, 0.025, 0.025)
        self._windowList.reverse()
        self.setActivationOrder(qt.QMdiArea.StackingOrder)
        self.tileSubWindows()

    def setSpectrum(self, x, y, legend=None, sigmay=None,
                    xlabel=None, ylabel=None):
        for key in self._windowDict:
            self._windowDict[key].clearCurves()
        self._windowDict["Original"].addCurve(x,
                                              y,
                                              legend=legend,
                                              xlabel=xlabel,
                                              ylabel=ylabel,
                                              yerror=sigmay,
                                              symbol="o")
        self.update()

    def update(self):
        plot = self._windowDict["Original"]
        activeCurve = plot.getActiveCurve()
        if not len(activeCurve):
            return
        [x, y, legend, info] = activeCurve[:4]
        xmin, xmax = plot.getGraphXLimits()
        ymin, ymax = plot.getGraphYLimits()
        
        result = RateLaw.rateLaw(x, y, sigmay=None)
        labels = ["Zero", "First", "Second"]
        for key in labels:
            plot = self._windowDict[key]
            workingResult = result[key.lower()]
            if workingResult is None:
                # no fit was performed
                plot.clear()
                continue
            intercept = workingResult["intercept"]
            slope = workingResult["slope"]
            sigma_intercept = workingResult["sigma_intercept"]
            sigma_slope = workingResult["sigma_slope"]
            r_value = workingResult["r_value"]
            stderr = workingResult["stderr"]
            xw = workingResult["x"]
            yw = workingResult["y"]
            xlabel = info["ylabel"]
            ylabel = info["ylabel"]
            title = "r = %.5f slope = %.3E +/- %.2E" % (r_value, slope, sigma_slope)
            fit_legend = "%.3g * x + %.3g" % (slope, intercept) 
            if key == "First":
                ylabel = "log(%s)" % ylabel 
            elif key == "Second":
                ylabel = "1 / %s" %  ylabel
            plot.addCurve(xw, yw,
                          legend="Data",
                          replace=True, replot=False,
                          symbol="o",
                          linestyle=" ",
                          ylabel=ylabel)
            plot.setGraphTitle(title)
            plot.addCurve(xw, intercept + slope * xw,
                          legend=fit_legend,
                          replace=False,
                          replot=True,
                          symbol=None,
                          color="red",
                          ylabel=ylabel)
            plot.resetZoom()
        self.sigRateLawMdiAreaSignal.emit(result)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 2:
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
    w = RateLawWindow()
    w.show()
    w.setSpectrum(x, y, sigmay = sigmay)
    return w

if __name__ == "__main__":
    app = qt.QApplication([])
    w = main()
    app.exec()
