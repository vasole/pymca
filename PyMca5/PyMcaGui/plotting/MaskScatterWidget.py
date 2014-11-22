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

DEBUG = 1

from . import PlotWindow
qt = PlotWindow.qt
IconDict = PlotWindow.IconDict

class MaskScatterWidget(PlotWindow.PlotWindow):
    def __init__(self, parent=None, backend=None, plugins=False, newplot=False,
                 control=False, position=False, maxNRois=1, grid=False,
                 logx=False, logy=False, togglePoints=False, normal=True,
                 polygon=True, colormap=True, aspect=True,
                 imageIcons=True, **kw):
        super(MaskScatterWidget, self).__init__(parent=parent,
                                                backend=backend,
                                                plugins=plugins,
                                                newplot=newplot,
                                                control=control,
                                                position=position,
                                                grid=grid,
                                                logx=logx,
                                                logy=logy,
                                                togglePoints=togglePoints,
                                                normal=normal,
                                                aspect=aspect,
                                                colormap=colormap,
                                                imageIcons=imageIcons,
                                                polygon=polygon,
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
        self.setPlotViewMode("scatter")
        self.setDrawModeEnabled(False)

    def setPlotViewMode(self, mode="scatter", bins=None):
        if mode.upper() != "DENSITY":
            self._activateScatterPlotView()
        else:
            self._activateDensityPlotView(bins)

    def _activateScatterPlotView(self):
        for key in ["colormap", "brushSelection", "brush", "rectangle"]:
            self.setToolBarActionVisible(key, False)
        if hasattr(self, "eraseSelectionToolButton"):
            self.eraseSelectionToolButton.setToolTip("Set erase mode if checked")            
            self.eraseSelectionToolButton.setCheckable(True)
            if self._eraseMode:
                self.eraseSelectionToolButton.setChecked(True)
            else:
                self.eraseSelectionToolButton.setChecked(False)
        if hasattr(self, "polygonSelectionToolButton"):
            self.polygonSelectionToolButton.setCheckable(True)

    def _activateDensityPlotView(self, bins):
        for key in ["colormap", "brushSelection", "brush", "rectangle"]:
            self.setToolBarActionVisible(key, True)
        if hasattr(self, "eraseSelectionToolButton"):
            self.eraseSelectionToolButton.setCheckable(False)
        if hasattr(self, "polygonSelectionToolButton"):
            self.polygonSelectionToolButton.setCheckable(False)
        raise NotImplemented("Density plot view not implemented yet")

    def setSelectionCurveData(self, x, y, legend="MaskScatterWidget", info=None,
                 replot=True, replace=True, linestyle=" ", color="r", symbol=None, **kw):
        self.enableActiveCurveHandling(False)
        if symbol is None:
            symbol = "o"
        self.addCurve(x=x, y=y, legend=legend, info=info,
                 replace=replace, replot=replot, linestyle=linestyle, color=color, symbol=symbol, **kw)
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
        colors = numpy.zeros((y.size, 4), dtype=numpy.uint8)
        if self._selectionMask is not None:
            tmpMask = self._selectionMask[:]
            tmpMask.shape = -1
            for i in range(0, self._maxNRois):
                colors[tmpMask == i, :] = self._selectionColors[i]
        self.setSelectionCurveData(x, y, legend=legend, info=info,
                                   color=colors, linestyle=" ",
                                   replot=replot, replace=replace)

    def setActiveRoiNumber(self, intValue):
        if (intValue < 0) or (intValue > self._maxNRois):
            raise ValueError("Value %d outside the interval [0, %d]" % (intValue, self._maxNRois))
        self._nRoi = intValue


    def _eraseSelectionIconSignal(self):
        if self.eraseSelectionToolButton.isChecked():
            self._eraseMode = True
        else:
            self._eraseMode = False

    def _polygonIconSignal(self):
        if self.polygonSelectionToolButton.isChecked():
            self.setPolygonSelectionMode()
        else:
            self.setZoomModeEnabled(True)

    def setZoomModeEnabled(self, flag):
        super(MaskScatterWidget, self).setZoomModeEnabled(flag)
        if flag:
            if hasattr(self,"polygonSelectionToolButton"):
                self.polygonSelectionToolButton.setChecked(False)

    def _handlePolygonMask(self, points):
        if self._eraseMode:
            value = 0
        else:
            value = self._nRoi
        x, y, legend, info = self.getCurve(self._selectionCurve)
        x.shape = -1
        y.shape = -1
        currentMask = self.getSelectionMask()
        if currentMask is None:
            currentMask = numpy.zeros(y.shape, dtype=numpy.uint8)
            if value == 0:
                return
        Z = numpy.zeros((y.size, 2), numpy.float64)
        Z[:, 0] = x
        Z[:, 1] = y
        mask = pnpoly(points, Z, 1)
        mask.shape = currentMask.shape
        currentMask[mask > 0] = value        
        self.setSelectionMask(currentMask)

    def graphCallback(self, ddict):
        if DEBUG:
            print("MaskScatterWidget graphCallback", ddict)
        if ddict["event"] == "mouseClicked":
            print("mouseClicked")
        elif ddict["event"] == "drawingFinished":
            self._handlePolygonMask(ddict["points"])
            print("drawing")
        elif ddict["event"] == "mouseMoved":
            print("mouseMoved")
        # the base implementation handles ROIs, mouse poistion and activeCurve
        super(MaskScatterWidget, self).graphCallback(ddict)

    def setPolygonSelectionMode(self):
        """
        Resets zoom mode and enters selection mode with the current active ROI index
        """
        self._zoomMode = False
        self._brushMode = False
        # one should be able to erase with a polygonal mask
        self._eraseMode = False
        self.setDrawModeEnabled(True, shape="polygon", label="mask")
        self.setZoomModeEnabled(False)
        if hasattr(self,"polygonSelectionToolButton"):
            self.polygonSelectionToolButton.setChecked(True)

    def setEraseSelectionMode(self, erase=True):
        if erase:
            self._eraseMode = True
        else:
            self._eraseMode = False
        if hasattr(self, "eraseSelectionToolButton"):
            self.eraseSelectionToolButton.setCheckable(True)
            if erase:
                self.eraseSelectionToolButton.setChecked(True)
            else:
                self.eraseSelectionToolButton.setChecked(False)

if __name__ == "__main__":
    app = qt.QApplication([])
    x = numpy.arange(1000.)
    y = x * x
    w = MaskScatterWidget(maxNRois=10)
    w.setSelectionCurveData(x, y, color="k", symbol="o")
    import numpy.random
    w.setSelectionMask(numpy.random.permutation(1000) % 10)
    w.setPolygonSelectionMode()
    w.show()
    app.exec_()

