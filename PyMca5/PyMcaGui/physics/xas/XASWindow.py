#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
import os
import sys
import numpy
import traceback
import copy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGui import PlotWindow
from PyMca5.PyMca import XASClass

class XASMdiArea(qt.QMdiArea):
    def __init__(self, parent=None, analyzer=None):
        super(XASMdiArea, self).__init__(parent)
        if analyzer is None:
            analyzer = XASClass.XASClass()
        self.analyzer = analyzer
        #self.setActivationOrder(qt.QMdiArea.CreationOrder)
        self._windowDict = {}
        self._windowList = ["Spectrum", "Post-edge", "Signal", "FT"]
        self._windowList.reverse()
        for title in self._windowList:
            plot = PlotWindow.PlotWindow(self)
            plot.setWindowTitle(title)
            self.addSubWindow(plot)
            self._windowDict[title] = plot
        self._windowList.reverse()
        self.setActivationOrder(qt.QMdiArea.StackingOrder)
        self.tileSubWindows()
        #self.cascadeSubWindows()
        #for window in self.subWindowList():
        #    print(" window = ", window.windowTitle())

    def setSpectrum(self, energy, mu):
        for key in self._windowDict:
            self._windowDict[key].clearCurves()
        if energy[0] < 200:
            energy = energy * 1000.
        self._windowDict["Spectrum"].addCurve(energy,
                                              mu,
                                              legend="Spectrum",
                                              xlabel="Energy (eV)",
                                              ylabel="Absorption (a.u.)")
        return self.analyzer.setSpectrum(energy, mu)

    def update(self):
        ddict = self.analyzer.processSpectrum()
        plot = self._windowDict["Spectrum"]
        plot.addCurve(energy, mu, legend="Spectrum",
                      xlabel="Energy (eV)", ylabel="Absorption (a.u.)",
                      replot=False, replace=True)
        plot.addCurve(ddict["NormalizedEnergy"],
                      ddict["NormalizedMu"],
                      legend="Normalized",
                      xlabel="Energy (eV)",
                      ylabel="Absorption (a.u.)",
                      yaxis="right",
                      replot=False)
        plot.addCurve(ddict["NormalizedEnergy"],
               ddict["NormalizedSignal"], legend="Post", replot=False)
        plot.addCurve(ddict["NormalizedEnergy"],
               ddict["NormalizedBackground"], legend="Pre",replot=False)
        plot.resetZoom()
        plot = self._windowDict["Post-edge"]
        plot = self._windowDict["Signal"]
        idx = (ddict["EXAFSKValues"] >= ddict["KMin"]) & \
              (ddict["EXAFSKValues"] <= ddict["KMax"])
        plot.addCurve(ddict["EXAFSKValues"][idx],
                      ddict["EXAFSNormalized"][idx],
                      legend="Normalized EXAFS")
        plot = self._windowDict["FT"]
        plot.addCurve(ddict["FT"]["FTRadius"],
                      ddict["FT"]["FTAmplitude"],
                      legend="FT Module",
                      xlabel="R (Angstrom)",
                      ylabel="Arbitraty Units")
    
if __name__ == "__main__":
    app = qt.QApplication([])
    from PyMca5.PyMcaIO import specfilewrapper as specfile
    from PyMca5.PyMcaDataDir import PYMCA_DATA_DIR
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        fileName = os.path.join(PYMCA_DATA_DIR, "EXAFS_Ge.dat")
    data = specfile.Specfile(fileName)[0].data()[-2:, :]
    energy = data[0, :]
    mu = data[1, :]
    w = XASMdiArea()
    w.show()
    w.setSpectrum(energy, mu)
    w.update()
    app.exec_()
