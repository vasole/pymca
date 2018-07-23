#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
___doc__ = """
    This module implements a scatter plot with selection capabilities.

    It is structured in three superposed layers:

    - First (deepest) layer containing the original points as they came.
    - Second layer containing the scatter plot density map.
    - Final layer containing the selected points with the selected colors.

"""
import numpy
import logging
from PyMca5.PyMcaGraph.ctools import pnpoly
_logger = logging.getLogger(__name__)

from . import MaskImageWidget
from . import MaskImageTools
from PyMca5.PyMcaGui import PyMcaQt as qt
from .MaskToolBar import MaskToolBar
from . import ColormapDialog
from .PyMca_Icons import IconDict

from silx.gui.plot import PlotWindow

if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str


class MaskScatterWidget(PlotWindow):
    sigMaskScatterWidgetSignal = qt.pyqtSignal(object)
    DEFAULT_COLORMAP_INDEX = 2
    DEFAULT_COLORMAP_LOG_FLAG = True

    def __init__(self, parent=None, backend=None, control=False,
                 position=False, maxNRois=1, grid=False, logScale=False,
                 curveStyle=False, resetzoom=True,
                 aspectRatio=True, imageIcons=True, polygon=True, bins=None):
        super(MaskScatterWidget, self).__init__(parent=parent,
                                                backend=backend,
                                                control=control,
                                                position=position,
                                                grid=grid,
                                                logScale=logScale,
                                                curveStyle=curveStyle,
                                                resetzoom=resetzoom,
                                                aspectRatio=aspectRatio,
                                                colormap=False,
                                                mask=False,
                                                yInverted=False,
                                                roi=False,
                                                copy=True,
                                                print_=False)
        if parent is None:
            self.setWindowTitle("MaskScatterWidget")
        self.setActiveCurveHandling(False)

        # No context menu by default, execute zoomBack on right click
        plotArea = self.getWidgetHandle()
        plotArea.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        plotArea.customContextMenuRequested.connect(self._zoomBack)

        self.colormapIcon = qt.QIcon(qt.QPixmap(IconDict["colormap"]))
        self.colormapToolButton = qt.QToolButton(self.toolBar())
        self.colormapToolButton.setIcon(self.colormapIcon)
        self.colormapToolButton.setToolTip('Change Colormap')
        self.colormapToolButton.clicked.connect(self._colormapIconSignal)
        self.colormapAction = self.toolBar().insertWidget(self.getSaveAction(),
                                                          self.colormapToolButton)

        self.maskToolBar = None
        if polygon or imageIcons:
            self.maskToolBar = MaskToolBar(parent=self,
                                           plot=self,
                                           imageIcons=imageIcons,
                                           polygon=polygon)
            self.addToolBar(self.maskToolBar)

        self._selectionCurve = None
        self._selectionMask = None
        self._alphaLevel = None
        self._xScale = None
        self._yScale = None

        self._maxNRois = maxNRois
        self._nRoi = 1
        self._zoomMode = True
        self._eraseMode = False
        self._brushMode = False
        self._brushWidth = 5
        self._bins = bins
        self._densityPlotWidget = None
        self._pixmap = None
        self._imageData = None
        self.colormapDialog = None
        self.colormap = None
        self.setPlotViewMode("scatter", bins=bins)

    def _colormapIconSignal(self):
        image = self.getActiveImage()
        if image is None:
            return

        if hasattr(image, "getColormap"):
            if self.colormapDialog is None:
                self._initColormapDialog(image.getData(),
                                         image.getColormap()._toDict())
            self.colormapDialog.show()
        else:
            # RGBA image
            _logger.info("No colormap to be handled")
            return

    def setPlotViewMode(self, mode="scatter", bins=None):
        if mode.lower() != "density":
            self._activateScatterPlotView()
        else:
            self._activateDensityPlotView(bins)

    def _activateScatterPlotView(self):
        self._plotViewMode = "scatter"
        self.colormapAction.setVisible(False)
        self._brushMode = False
        self.setInteractiveMode("select")

        if hasattr(self, "maskToolBar"):
            self.maskToolBar.activateScatterPlotView()

        self.clearImages()
        self._updatePlot()

    def _activateDensityPlotView(self, bins=None):
        self._plotViewMode = "density"
        self.colormapAction.setVisible(True)

        if hasattr(self, "maskToolBar"):
            self.maskToolBar.activateDensityPlotView()

        if _logger.getEffectiveLevel() == logging.DEBUG:
            if self._densityPlotWidget is None:
                self._densityPlotWidget = MaskImageWidget.MaskImageWidget(
                                imageicons=True,
                                selection=True,
                                profileselection=True,
                                aspect=True,
                                polygon=True)
                self._densityPlotWidget.sigMaskImageWidgetSignal.connect(self._densityPlotSlot)
            self._updateDensityPlot(bins)
            # only show it in debug mode
            self._densityPlotWidget.show()
        curve = self.getCurve(self._selectionCurve)
        if curve is None:
            return
        x, y, legend, info = curve[0:4]
        self.setSelectionCurveData(x, y, legend=legend, info=info)

    def getDensityData(self, bins=None):
        curve = self.getCurve(self._selectionCurve)
        if curve is None:
            return
        x, y, = curve[0:2]
        if bins is not None:
            if type(bins) == type(1):
                bins = (bins, bins)
            elif len(bins) == 0:
                bins = (bins[0], bins[0])
            else:
                bins = bins[0:2]
        elif self._bins is None:
            bins = [int(x.size / 10), int(y.size/10)]
            if bins[0] > 100:
                bins[0] = 100
            elif bins[0] < 2:
                bins[0] = 2
            if bins[1] > 100:
                bins[1] = 100
            elif bins[1] < 2:
                bins[1] = 2
        else:
            bins = self._bins
        x0 = x.min()
        y0 = y.min()
        image = numpy.histogram2d(y, x,
                                  bins=bins,
                                  #range=(binsY, binsX),
                                  normed=False)
        self._binsX = image[2]
        self._binsY = image[1]
        self._bins = bins
        #print("shape", image[0].shape, "image max  min ", image[0].max(), image[0].min())
        #print("deltaxmin and max", (self._binsX[1:] - self._binsX[:-1]).min(),
        #      (self._binsX[1:] - self._binsX[:-1]).max())
        deltaX = (self._binsX[1:] - self._binsX[:-1]).mean()
        deltaY = (self._binsY[1:] - self._binsY[:-1]).mean()
        self._xScale = (x0, deltaX)
        self._yScale = (y0, deltaY)
        return image[0]

    def _updateDensityPlot(self, bins=None):
        _logger.debug("_updateDensityPlot called")
        if self._densityPlotWidget is None:
            return
        curve = self.getCurve(self._selectionCurve)
        if curve is None:
            return
        x, y, legend, info = curve[0:4]
        if bins is not None:
            if type(bins) == type(1):
                bins = (bins, bins)
            elif len(bins) == 0:
                bins = (bins[0], bins[0])
            else:
                bins = bins[0:2]
        elif self._bins is None:
            bins = [int(x.size/ 10), int(y.size/10)]
            if bins[0] > 100:
                bins[0] = 100
            elif bins[0] < 2:
                bins[0] = 2
            if bins[1] > 100:
                bins[1] = 100
            elif bins[1] < 2:
                bins[1] = 2
        else:
            bins = self._bins
        x0 = x.min()
        y0 = y.min()
        deltaX = (x.max() - x0) / float(bins[0] - 1)
        deltaY = (y.max() - y0) / float(bins[1] - 1)
        self.xScale = (x0, deltaX)
        self.yScale = (y0, deltaY)
        binsX = numpy.arange(bins[0]) * deltaX
        binsY = numpy.arange(bins[1]) * deltaY
        image = numpy.histogram2d(y, x, bins=(binsY, binsX), normed=False)
        self._binsX = image[2]
        self._binsY = image[1]
        self._bins = bins
        if _logger.getEffectiveLevel() == logging.DEBUG:
            # this does not work properly
            # update mask levels
            if self._selectionMask is not None:
                weights = self._selectionMask[:]
                weights.shape = x.shape
                if self._maxNRois > 1:
                    _logger.debug("BAD PATH")
                    # this does not work properly yet
                    weightsSum = weights.sum(dtype=numpy.float64)
                    volume = (binsY[1] - binsY[0]) * (binsX[1] - binsX[0])
                    mask = numpy.round(numpy.histogram2d(y, x,
                                       bins=(binsY, binsX),
                                       weights=weights,
                                       normed=True)[0] * weightsSum * volume).astype(numpy.uint8)
                else:
                    #print("GOOD PATH")
                    mask = numpy.histogram2d(y, x,
                                             bins=(binsY, binsX),
                                             weights=weights,
                                             normed=False)[0]
                    mask[mask > 0] = 1
                #print(mask.min(), mask.max())
                self._densityPlotWidget.setSelectionMask(mask, plot=False)
        self._densityPlotWidget.graphWidget.graph.setGraphXLabel(self.getGraphXLabel())
        self._densityPlotWidget.graphWidget.graph.setGraphYLabel(self.getGraphYLabel())
        self._densityPlotWidget.setImageData(image[0],
                                             clearmask=False,
                                             xScale=self.xScale,
                                             yScale=self.yScale)

        # do not overlay plot (yet)
        pixmap = self._densityPlotWidget.getPixmap() * 1
        #pixmap[:, :, 3] = 128
        #self.addImage(pixmap,
        #              legend=legend+" density",
        #              xScale=(x0, deltaX), yScale=(y0, deltaY), z=10)
        self._pixmap = pixmap
        self._imageData = image[0]
        #raise NotImplemented("Density plot view not implemented yet")

    def _initColormapDialog(self, imageData, colormap=None):
        """Set-up the colormap dialog default values.

        :param numpy.ndarray imageData: data used to init dialog.
        :param dict colormap: Description of the colormap as a dict.
                              See :class:`PlotBackend` for details.
                              If None, use default values.
        """
        goodData = imageData[numpy.isfinite(imageData)]
        if goodData.size > 0:
            maxData = goodData.max()
            minData = goodData.min()
        else:
            qt.QMessageBox.critical(self, "No Data",
                "Image data does not contain any real value")
            return

        self.colormapDialog = ColormapDialog.ColormapDialog(self)

        if colormap is None:
            colormapIndex = self.DEFAULT_COLORMAP_INDEX
            if colormapIndex == 6:
                colormapIndex = 1
            self.colormapDialog.setColormap(colormapIndex)
            self.colormapDialog.setDataMinMax(minData, maxData)
            self.colormapDialog.setAutoscale(1)
            self.colormapDialog.setColormap(self.colormapDialog.colormapIndex)
            # linear or logarithmic
            self.colormapDialog.setColormapType(self.DEFAULT_COLORMAP_LOG_FLAG,
                                                update=False)
        else:
            # Set-up colormap dialog from provided colormap dict
            cmapList = ColormapDialog.colormapDictToList(colormap)
            index, autoscale, vMin, vMax, dataMin, dataMax, cmapType = cmapList
            self.colormapDialog.setColormap(index)
            self.colormapDialog.setAutoscale(autoscale)
            self.colormapDialog.setMinValue(vMin)
            self.colormapDialog.setMaxValue(vMax)
            self.colormapDialog.setDataMinMax(minData, maxData)
            self.colormapDialog.setColormapType(cmapType, update=False)

        self.colormap = self.colormapDialog.getColormap()  # Is it used?
        self.colormapDialog.setWindowTitle("Colormap Dialog")
        self.colormapDialog.sigColormapChanged.connect(
                    self.updateActiveImageColormap)
        self.colormapDialog._update()

    def updateActiveImageColormap(self, colormap):
        if len(colormap) == 1:
            colormap = colormap[0]
        # TODO: Once everything is ready to work with dict instead of
        # list, we can remove this translation
        plotBackendColormap = ColormapDialog.colormapListToDict(colormap)
        self.setDefaultColormap(plotBackendColormap)

        image = self.getActiveImage()
        if image is None:
            if self.colormapDialog is not None:
                self.colormapDialog.hide()
            return

        if not hasattr(image, "getColormap"):
            if self.colormapDialog is not None:
                self.colormapDialog.hide()
            return
        pixmap = MaskImageTools.getPixmapFromData(image.getData(), colormap)
        self.addImage(image.getData(), legend=image.getLegend(),
                      info=image.getInfo(),
                      pixmap=pixmap)

    def setSelectionCurveData(self, x, y, legend=None, info=None,
                              replace=True, linestyle=" ", resetzoom=True,
                              color=None, symbol=None, selectable=None,
                              **kw):
        if "replot" in kw:
            _logger.warning("MaskScatterWidget.setSelectionCurveData: deprecated replot parameter")
            resetzoom = kw["replot"] and resetzoom
        self.setActiveCurveHandling(False)
        if legend is None:
            legend = "MaskScatterWidget"
        if symbol is None:
            if x.size < 1000:
                # circle
                symbol = "o"
            elif x.size < 1.0e5:
                # dot
                symbol = "."
            else:
                # pixel
                symbol = ","
        #if selectable is None:
        #    if symbol == ",":
        #        selectable = False
        #    else:
        #        selectable = True

        # the basic curve is drawn
        self.addCurve(x=x, y=y, legend=legend, info=info,
                      replace=replace, resetzoom=False, linestyle=linestyle,
                      color=color, symbol=symbol, selectable=selectable,
                      z=0, **kw)
        self._selectionCurve = legend

        # if view mode, draw the image
        if self._plotViewMode == "density":
            # get the binned data
            imageData = self.getDensityData()
            # get the associated pixmap
            if self.colormapDialog is None:
                self._initColormapDialog(imageData)
            cmap = self.colormapDialog.getColormap()
            pixmap = MaskImageTools.getPixmapFromData(imageData,
                                                      colormap=cmap)
            origin, scale = (0., 0.), (1., 1.)
            if self._xScale is not None and self._yScale is not None:
                origin = self._xScale[0], self._yScale[0]
                scale = self._xScale[1], self._yScale[1]

            self.addImage(imageData, legend=legend + "density",
                          origin=origin, scale=scale,
                          z=0,
                          pixmap=pixmap,
                          resetzoom=False)
            self._imageData = imageData
            self._pixmap = pixmap

        # draw the mask as a set of curves
        hasMaskedData = False
        if self._selectionMask is not None:
            if self._selectionMask.max():
                hasMaskedData = True

        if hasMaskedData or not replace:
            self._updatePlot(resetzoom=False)

        # update the limits if it was requested
        if resetzoom:
            self.resetZoom()

        if 0 :#or self._plotViewMode == "density":
            # get the binned data
            imageData = self.getDensityData()
            # get the associated pixmap
            pixmap = MaskImageTools.getPixmapFromData(imageData)
            if 0:
                self.addImage(imageData, legend=legend + "density",
                              xScale=self._xScale,
                              yScale=self._yScale,
                              z=0,
                              pixmap=pixmap,
                              resetzoom=True)
            if _logger.getEffectiveLevel() == logging.DEBUG:
                if self._densityPlotWidget is None:
                    self._densityPlotWidget = MaskImageWidget.MaskImageWidget(
                                    imageicons=True,
                                    selection=True,
                                    profileselection=True,
                                    aspect=True,
                                    polygon=True)
                self._updateDensityPlot()
                _logger.debug("CLOSE = %s", numpy.allclose(imageData, self._imageData))
                _logger.debug("CLOSE PIXMAP = %s", numpy.allclose(pixmap, self._pixmap))
            self._imageData = imageData
            self._pixmap = pixmap
        #self._updatePlot()

    def setSelectionMask(self, mask=None, plot=True):
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
                    raise ValueError("Mask size = %d while data size = %d" % (mask.size, x.size))
        if plot:
            self._updatePlot()

    def getSelectionMask(self):
        if self._selectionMask is None:
            if self._selectionCurve is not None:
                x, y = self.getCurve(self._selectionCurve)[0:2]
                self._selectionMask = numpy.zeros(x.shape, numpy.uint8)
        return self._selectionMask

    def _updatePlot(self, resetzoom=False, replace=True):
        if self._selectionCurve is None:
            return
        x0, y0, legend, info = self.getCurve(self._selectionCurve)[0:4]
        # make sure we work with views
        x = x0[:]
        y = y0[:]
        x.shape = -1
        y.shape = -1
        if 0:
            colors = numpy.zeros((y.size, 4), dtype=numpy.uint8)
            colors[:, 3] = 255
            if self._selectionMask is not None:
                tmpMask = self._selectionMask[:]
                tmpMask.shape = -1
                for i in range(0, self._maxNRois + 1):
                    colors[tmpMask == i, :] = self.maskToolBar._selectionColors[i]
                self.setSelectionCurveData(x, y, legend=legend, info=info,
                                           #color=colors,
                                           color="k",
                                           linestyle=" ",
                                           resetzoom=resetzoom, replace=replace)
        else:
            if self._selectionMask is None:
                for i in range(1, self._maxNRois + 1):
                    self.removeCurve(legend=legend + " %02d" % i)
            else:
                tmpMask = self._selectionMask[:]
                tmpMask.shape = -1
                if self._plotViewMode == "density":
                    useAlpha = True
                    if self._alphaLevel is None:
                        self._initializeAlpha()
                else:
                    useAlpha = False
                for i in range(1, self._maxNRois + 1):
                    xMask = x[tmpMask == i]
                    yMask = y[tmpMask == i]
                    if xMask.size < 1:
                        self.removeCurve(legend=legend + " %02d" % i)
                        continue
                    color = self.maskToolBar._selectionColors[i].copy()
                    if useAlpha:
                        if len(color) == 4:
                            if type(color[3]) in [numpy.uint8, numpy.int]:
                                color[3] = self._alphaLevel
                    # a copy of the input info is needed in order not
                    # to set the main curve to that color

                    self.addCurve(xMask, yMask, legend=legend + " %02d" % i,
                                  info=info.copy(), color=color,
                                  ylabel=legend + " %02d" % i,
                                  linestyle=" ", symbol="o",
                                  selectable=False,
                                  z=1,
                                  resetzoom=False, replace=False)
                if resetzoom:
                    self.resetZoom()

    def setActiveRoiNumber(self, intValue):
        if (intValue < 0) or (intValue > self._maxNRois):
            raise ValueError("Value %d outside the interval [0, %d]" % (intValue, self._maxNRois))
        self._nRoi = intValue

    def _handlePolygonMask(self, points):
        _logger.debug("_handlePolygonMask called")
        if self._eraseMode:
            value = 0
        else:
            value = self._nRoi
        x, y = self.getCurve(self._selectionCurve)[0:2]
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
        self.setSelectionMask(currentMask, plot=True)
        self._emitMaskChangedSignal()

    def setMouseText(self, text=""):
        try:
            if text:
                qt.QToolTip.showText(self.cursor().pos(),
                                     text, self, qt.QRect())
            else:
                qt.QToolTip.hideText()
        except:
            _logger.warning("Error trying to show mouse text <%s>" % text)

    def graphCallback(self, ddict):
        _logger.debug("MaskScatterWidget graphCallback %s", ddict)
        if ddict["event"] == "drawingFinished":
            if ddict["parameters"]["shape"].lower() == "rectangle":
                points = numpy.zeros((5, 2), dtype=ddict["points"].dtype)
                points[0] = ddict["points"][0]
                points[1, 0] = ddict["points"][0, 0]
                points[1, 1] = ddict["points"][1, 1]
                points[2] = ddict["points"][1]
                points[3, 0] = ddict["points"][1, 0]
                points[3, 1] = ddict["points"][0, 1]
                points[4] = ddict["points"][0]
                self._handlePolygonMask(points)
            else:
                self._handlePolygonMask(ddict["points"])
        elif ddict['event'] in ["mouseMoved", "MouseAt", "mouseClicked"]:
            if (self._plotViewMode == "density") and \
               (self._imageData is not None):
                shape = self._imageData.shape
                row, column = MaskImageTools.convertToRowAndColumn(
                                                      ddict['x'],
                                                      ddict['y'],
                                                      shape,
                                                      xScale=self._xScale,
                                                      yScale=self._yScale,
                                                      safe=True)

                halfWidth = 0.5 * self._brushWidth   #in (row, column) units
                halfHeight = 0.5 * self._brushWidth  #in (row, column) units

                columnMin = max(column - halfWidth, 0)
                columnMax = min(column + halfWidth, shape[1])

                rowMin = max(row - halfHeight, 0)
                rowMax = min(row + halfHeight, shape[0])

                rowMin = min(int(round(rowMin)), shape[0] - 1)
                rowMax = min(int(round(rowMax)), shape[0])
                columnMin = min(int(round(columnMin)), shape[1] - 1)
                columnMax = min(int(round(columnMax)), shape[1])

                if rowMin == rowMax:
                    rowMax = rowMin + 1
                elif (rowMax - rowMin) > self._brushWidth:
                    # python 3 implements banker's rounding
                    # test case ddict['x'] = 23.3 gives:
                    # i1 = 22 and i2 = 24 in python 3
                    # i1 = 23 and i2 = 24 in python 2
                    rowMin = rowMax - self._brushWidth

                if columnMin == columnMax:
                    columnMax = columnMin + 1
                elif (columnMax - columnMin) > self._brushWidth:
                    # python 3 implements banker's rounding
                    # test case ddict['x'] = 23.3 gives:
                    # i1 = 22 and i2 = 24 in python 3
                    # i1 = 23 and i2 = 24 in python 2
                    columnMin = columnMax - self._brushWidth

                #To show array coordinates:
                #x = self._xScale[0] + columnMin * self._xScale[1]
                #y = self._yScale[0] + rowMin * self._yScale[1]
                #self.setMouseText("%g, %g, %g" % (x, y, self.__imageData[rowMin, columnMin]))
                #To show row and column:
                #self.setMouseText("%g, %g, %g" % (row, column, self.__imageData[rowMin, columnMin]))
                #To show mouse coordinates:
                #self.setMouseText("%g, %g, %g" % (ddict['x'], ddict['y'], self.__imageData[rowMin, columnMin]))
                if self._xScale is not None and self._yScale is not None:
                    x = self._xScale[0] + column * self._xScale[1]
                    y = self._yScale[0] + row * self._yScale[1]
                else:
                    x = column
                    y = row
                self.setMouseText("%g, %g, %g" % (x, y, self._imageData[row, column]))

            if self._brushMode:
                if self.getInteractiveMode()['mode'] == 'zoom':
                    return
                if ddict['button'] != "left":
                    return
                selectionMask = numpy.zeros(self._imageData.shape,
                                            numpy.uint8)
                if self._eraseMode:
                    selectionMask[rowMin:rowMax, columnMin:columnMax] = 1
                else:
                    selectionMask[rowMin:rowMax, columnMin:columnMax] = \
                                                                self._nRoi
                self._setSelectionMaskFromDensityMask(selectionMask,
                                                      update=True)
        #if emitsignal:
        #    #should this be made by the parent?
        #    self.plotImage(update = False)
        #
        #    #inform the other widgets
        #    self._emitMaskChangedSignal()
        # the base implementation handles ROIs, mouse position and activeCurve
        super(MaskScatterWidget, self).graphCallback(ddict)

    def setEraseSelectionMode(self, erase=True):     # TODO: unused?
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

    def _emitMaskChangedSignal(self):
        #inform the other widgets
        ddict = {}
        ddict['event'] = "selectionMaskChanged"
        ddict['current'] = self._selectionMask * 1
        ddict['id'] = id(self)
        self.emitMaskScatterWidgetSignal(ddict)

    def emitMaskScatterWidgetSignal(self, ddict):
        self.sigMaskScatterWidgetSignal.emit(ddict)

    def _resetSelection(self, owncall=True):
        _logger.debug("_resetSelection")

        if self._selectionMask is None:
            _logger.info("Selection mask is None, doing nothing")
            return
        else:
            self._selectionMask[:] = 0

        self._updatePlot()

        #inform the others
        if owncall:
            ddict = {}
            ddict['event'] = "resetSelection"
            ddict['id'] = id(self)
            self.emitMaskScatterWidgetSignal(ddict)

    def _invertSelection(self):
        if self._selectionMask is None:
            return
        mask = numpy.ones(self._selectionMask.shape, numpy.uint8)
        mask[self._selectionMask > 0] = 0
        self.setSelectionMask(mask, plot=True)
        self._emitMaskChangedSignal()

    def _getSelectionMinMax(self):
        if self.colormap is None:
            goodData = self._imageData[numpy.isfinite(self._imageData)]
            maxValue = goodData.max()
            minValue = goodData.min()
        else:
            minValue = self.colormap[2]
            maxValue = self.colormap[3]
        return minValue, maxValue

    def _selectMax(self):
        if (self._plotViewMode != "density") or \
           (self._imageData is None):
            return

        selectionMask = numpy.zeros(self._imageData.shape, numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self._imageData, copy=True)
        tmpData[True - numpy.isfinite(self._imageData)] = minValue
        selectionMask[tmpData >= maxValue] = self._nRoi
        self._setSelectionMaskFromDensityMask(selectionMask)
        self._emitMaskChangedSignal()

    def _selectMiddle(self):
        if (self._plotViewMode != "density") or \
           (self._imageData is None):
            return
        selectionMask = numpy.zeros(self._imageData.shape, numpy.uint8)
        selectionMask[:] = self._nRoi
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self._imageData, copy=True)
        tmpData[True - numpy.isfinite(self._imageData)] = minValue
        selectionMask[tmpData >= maxValue] = 0
        selectionMask[tmpData <= minValue] = 0
        self._setSelectionMaskFromDensityMask(selectionMask)
        self._emitMaskChangedSignal()

    def _selectMin(self):
        if (self._plotViewMode != "density") or \
           (self._imageData is None):
            return
        selectionMask = numpy.zeros(self._imageData.shape, numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self._imageData, copy=True)
        tmpData[True - numpy.isfinite(self._imageData)] = maxValue
        selectionMask[tmpData <= minValue] = self._nRoi
        self._setSelectionMaskFromDensityMask(selectionMask)
        self._emitMaskChangedSignal()

    def _setSelectionMaskFromDensityMask(self, densityPlotMask, update=None):
        _logger.debug("_setSelectionMaskFromDensityMask called")
        curve = self.getCurve(self._selectionCurve)
        if curve is None:
            return
        x, y = curve[0:2]
        bins = self._bins
        x0 = x.min()
        y0 = y.min()
        deltaX = (x.max() - x0)/float(bins[0])
        deltaY = (y.max() - y0)/float(bins[1])
        columns = numpy.digitize(x, self._binsX, right=True)
        columns[columns >= densityPlotMask.shape[1]] = \
                                                   densityPlotMask.shape[1] - 1
        rows = numpy.digitize(y, self._binsY, right=True)
        rows[rows>=densityPlotMask.shape[0]] = densityPlotMask.shape[0] - 1
        values = densityPlotMask[rows, columns]
        values.shape = -1

        if self._selectionMask is None:
            view = numpy.zeros(x.size, dtype=numpy.uint8)
            view[:] = values[:]
        elif update:
            view = self._selectionMask.copy()
            if self._eraseMode:
                view[values > 0] = 0
            else:
                view[values > 0] = values[values > 0]
        else:
            view = numpy.zeros(self._selectionMask.size,
                               dtype=self._selectionMask.dtype)
            view[:] = values[:]
        if self._selectionMask is not None:
            view.shape = self._selectionMask.shape
        self.setSelectionMask(view, plot=True)

    def _densityPlotSlot(self, ddict):
        _logger.debug("_densityPlotSlot called")
        if ddict["event"] == "resetSelection":
            self.__resetSelection()
            return
        if ddict["event"] not in ["selectionMaskChanged"]:
            return
        densityPlotMask = ddict["current"]
        curve = self.getCurve(self._selectionCurve)
        if curve is None:
            return
        x, y = curve[0:2]
        bins = self._bins
        x0 = x.min()
        y0 = y.min()
        deltaX = (x.max() - x0)/float(bins[0])
        deltaY = (y.max() - y0)/float(bins[1])
        if _logger.getEffectiveLevel() == logging.DEBUG:
            if self._selectionMask is None:
                view = numpy.zeros(x.size, dtype=numpy.uint8)
            else:
                view = numpy.zeros(self._selectionMask.size, dtype=self._selectionMask.dtype)
            # this works even on unordered data
            for i in range(x.size):
                row = int((y[i] - y0) /deltaY)
                column = int((x[i] - x0) /deltaX)
                try:
                    value = densityPlotMask[row, column]
                except:
                    if row >= densityPlotMask.shape[0]:
                        row = densityPlotMask.shape[0] - 1
                    if column >= densityPlotMask.shape[1]:
                        column = densityPlotMask.shape[1] - 1
                    value = densityPlotMask[row, column]
                if value:
                    view[i] = value
            if self._selectionMask is not None:
                view.shape = self._selectionMask.shape
            self.setSelectionMask(view)

        if self._selectionMask is None:
            view2 = numpy.zeros(x.size, dtype=numpy.uint8)
        else:
            view2 = numpy.zeros(self._selectionMask.size, dtype=self._selectionMask.dtype)
        columns = numpy.digitize(x, self._binsX, right=True)
        columns[columns>=densityPlotMask.shape[1]] = densityPlotMask.shape[1] - 1
        rows = numpy.digitize(y, self._binsY, right=True)
        rows[rows>=densityPlotMask.shape[0]] = densityPlotMask.shape[0] - 1
        values = densityPlotMask[rows, columns]
        values.shape = -1
        view2[:] = values[:]
        if self._selectionMask is not None:
            view2.shape = self._selectionMask.shape
        if _logger.getEffectiveLevel() == logging.DEBUG:
            if not numpy.allclose(view, view2):
                a = view[:]
                b = view2[:]
                a.shape = -1
                b.shape = -1
                c = 0
                for i in range(a.size):
                    if a[i] != b[i]:
                        _logger.debug("%d a = %s, b = %s, (x, y) = (%s, %s)",
                                      i, a[i], b[i], x[i], y[i])
                        c += 1
                        if c > 10:
                            break
            else:
                _logger.debug("OK!!!")
        self.setSelectionMask(view2)

    def _initializeAlpha(self):
        self._alphaLevel = 128

    def _increaseMaskAlpha(self):
        if self._alphaLevel is None:
            self._initializeAlpha()
        self._alphaLevel *= 4
        if self._alphaLevel > 255:
            self._alphaLevel = 255
        self._alphaLevel
        self._updatePlot()

    def _decreaseMaskAlpha(self):
        if self._alphaLevel is None:
            self._initializeAlpha()
        self._alphaLevel /= 4
        if self._alphaLevel < 2:
            self._alphaLevel = 2
        self._updatePlot()

    def setPolygonSelectionMode(self):
        """
        Resets zoom mode and enters selection mode with the current active ROI index
        """
        self.maskToolBar.setPolygonSelectionMode()

    def _zoomBack(self, pos):
        self.getLimitsHistory().pop()

if __name__ == "__main__":
    backend = "matplotlib"
    #backend = "opengl"
    app = qt.QApplication([])
    def receivingSlot(ddict):
        print("Received: ", ddict)
    x = numpy.arange(100.)
    y = x * 1
    w = MaskScatterWidget(maxNRois=10, bins=(100, 100), backend=backend,
                          control=True)
    w.setSelectionCurveData(x, y, color="k", selectable=False)
    import numpy.random
    w.setSelectionMask(numpy.random.permutation(100) % 10)
    w.setPolygonSelectionMode()
    w.sigMaskScatterWidgetSignal.connect(receivingSlot)
    w.show()
    app.exec_()

