#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
import os
import numpy
from PyMca5.PyMcaGraph.ctools import pnpoly

DEBUG = 0

from . import PlotWindow
qt = PlotWindow.qt
IconDict = PlotWindow.IconDict

class MaskScatterWidget(PlotWindow.PlotWindow):
    def __init__(self, parent=None, backend=None, plugins=False, newplot=False,
                 control=False, position=False, maxNRois=1, grid=False, **kw):
        super(MaskScatterWidget, self).__init__(parent=parent,
                                                backend=backend,
                                                plugins=plugins,
                                                newplot=newplot,
                                                control=control,
                                                position=position,
                                                grid=grid,
                                                **kw)
        self._selectionCurve = None
        self._selectionMask = None
        self._selectionColors = numpy.zeros((len(self.colorList), 4), numpy.uint8)
        for i in range(len(self.colorList)):
            self._selectionColors[i, 0] = eval("0x" + self.colorList[i][-2:])
            self._selectionColors[i, 1] = eval("0x" + self.colorList[i][3:-2])
            self._selectionColors[i, 2] = eval("0x" + self.colorList[i][1:3])
            self._selectionColors[i, 3] = 0xff
        self._maxNRois = maxNRois
        self._nRoi = 1
        self._zoomMode = True
        self._eraseMode = False
        self._brushMode = False
        self.setDrawModeEnabled(False)

    def setSelectionCurveData(self, x, y, legend="MaskScatterWidget", info=None, 
                 replot=True, replace=True, line_style=" ", color="r", symbol=None, **kw):
        self.enableActiveCurveHandling(False)
        if symbol is None:
            symbol = "o"
        self.addCurve(x=x, y=y, legend=legend, info=info,
                 replace=replace, replot=replot, line_style=line_style, color=color, symbol=symbol, **kw)
        self._selectionCurve = legend

    def setSelectionMask(self, mask=None):
        if self._selectionCurve is not None:
            selectionCurve = self.getCurve(self._selectionCurve)
        if selectionCurve in [[], None]:
            self._selectionCurve = None
            self._selectionMask = mask
        else:
            x, y = selectionCurve[0:2]
            x = numpy.array(x, copy=False)
            if hasattr(mask, "size"):
                if mask.size == x.size:
                    if self._selectionMask is None:
                        self._selectionMask = mask
                    elif self._selectionMask.size == mask.size:
                        # keep shape because we may refer to images
                        tmpView = self._selectionMask[:]
                        tmpView.shape = -1
                        tmpMask = mask[:]
                        tmpMask.shape = -1
                        tmpView[:] = tmpMask[:]
                    else:
                        self._selectionMask = mask
                else:
                    raise ValueError("Mask size = %d while data size = %d" % (mask.size(), x.size()))
        self._updatePlot()

    def getSelectionMask(self):
        # TODO: Deal with non-finite data like in MaskImageWidget
        return self._selectionMask

    def _updatePlot(self, replot=True, replace=True):
        if self._selectionCurve is None:
            return
        x, y, legend, info = self.getCurve(self._selectionCurve)
        x.shape = -1
        y.shape = -1
        mask = numpy.zeros(y.shape, dtype=numpy.uint8)
        colors = numpy.zeros((y.size, 4), dtype=numpy.uint8)
        if self._selectionMask is not None:
            tmpMask = self._selectionMask[:]
            tmpMask.shape = -1
            for i in range(1, self._maxNRois):
                colors[tmpMask == i, :] = self._selectionColors[i]
        self.setSelectionCurveData(x, y, legend=legend, info=info, color=colors, line_style=" ",
                                   replot=replot, replace=replace)

    def setActiveRoiNumber(self, intValue):
        if (intValue < 0) or (intValue > self._maxNRois):
            raise ValueError("Value %d outside the interval [0, %d]" % (intValue, self._maxNRois))
        self._nRoi = intValue

    def setPolygonSelectionMode(self):
        """
        Resets zoom mode and enters selection mode with the current active ROI index
        """
        self._zoomMode = False
        self._brushMode = False
        # one should be able to erase with a polygonal mask
        self._eraseMode = False
        self.setDrawModeEnabled(True, shape="polygon", label="mask")

if __name__ == "__main__":
    app = qt.QApplication([])
    x = numpy.arange(1000.)
    y = x * x
    w = MaskScatterWidget(maxNRois=8)
    w.setSelectionCurveData(x, y, color="k", symbol="o")
    import numpy.random
    w.setSelectionMask(numpy.random.permutation(1000) % 10)
    w.setPolygonSelectionMode()
    w.show()
    app.exec_()

