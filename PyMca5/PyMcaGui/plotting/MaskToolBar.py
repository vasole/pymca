#/*##########################################################################
# Copyright (C) 2004-2017 V.A. Sole, European Synchrotron Radiation Facility
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
"""This module implements a plot toolbar with buttons to draw and erase masks.
"""

__author__ = "P. Knobel"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy

from PyMca5.PyMcaGui import PyMcaQt as qt
from .PyMca_Icons import IconDict
from PyMca5.PyMcaGraph import Colors

if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str


_COLORDICT = Colors.COLORDICT
_COLORLIST = [_COLORDICT['black'],
              _COLORDICT['blue'],
              _COLORDICT['red'],
              _COLORDICT['green'],
              _COLORDICT['pink'],
              _COLORDICT['yellow'],
              _COLORDICT['brown'],
              _COLORDICT['cyan'],
              _COLORDICT['magenta'],
              _COLORDICT['orange'],
              _COLORDICT['violet'],
              #_COLORDICT['bluegreen'],
              _COLORDICT['grey'],
              _COLORDICT['darkBlue'],
              _COLORDICT['darkRed'],
              _COLORDICT['darkGreen'],
              _COLORDICT['darkCyan'],
              _COLORDICT['darkMagenta'],
              _COLORDICT['darkYellow'],
              _COLORDICT['darkBrown']]


class MaskToolBar(qt.QToolBar):
    sigIconSignal = qt.pyqtSignal(object)
    colorList = _COLORLIST

    def __init__(self, parent=None, plot=None, title="Mask tools",
                 imageIcons=True, polygon=True):
        super(MaskToolBar, self).__init__(title, parent)
        assert plot is not None
        assert imageIcons or polygon,\
            "It makes no sense to build an empty mask toolbar"
        self.plot = plot

        self.polygonIcon = qt.QIcon(qt.QPixmap(IconDict["polygon"]))
        self.imageIcon = qt.QIcon(qt.QPixmap(IconDict["image"]))
        self.eraseSelectionIcon = qt.QIcon(qt.QPixmap(IconDict["eraseselect"]))
        self.rectSelectionIcon = qt.QIcon(qt.QPixmap(IconDict["boxselect"]))
        self.brushSelectionIcon = qt.QIcon(qt.QPixmap(IconDict["brushselect"]))
        self.brushIcon = qt.QIcon(qt.QPixmap(IconDict["brush"]))
        self.additionalIcon = qt.QIcon(qt.QPixmap(IconDict["additionalselect"]))

        self.polygonSelectionToolButton = qt.QToolButton(self)
        self.imageToolButton = qt.QToolButton(self)
        self.eraseSelectionToolButton = qt.QToolButton(self)
        self.rectSelectionToolButton = qt.QToolButton(self)
        self.brushSelectionToolButton = qt.QToolButton(self)
        self.brushToolButton = qt.QToolButton(self)
        self.additionalSelectionToolButton = qt.QToolButton(self)

        self.polygonSelectionToolButton.setIcon(self.polygonIcon)
        self.imageToolButton.setIcon(self.imageIcon)
        self.eraseSelectionToolButton.setIcon(self.eraseSelectionIcon)
        self.rectSelectionToolButton.setIcon(self.rectSelectionIcon)
        self.brushSelectionToolButton.setIcon(self.brushSelectionIcon)
        self.brushToolButton.setIcon(self.brushIcon)
        self.additionalSelectionToolButton.setIcon(self.additionalIcon)

        self.polygonSelectionToolButton.setToolTip('Polygon selection\n'
                                                   'Right click to finish')
        self.imageToolButton.setToolTip('Reset')
        self.eraseSelectionToolButton.setToolTip('Erase Selection')
        self.rectSelectionToolButton.setToolTip('Rectangular Selection')
        self.brushSelectionToolButton.setToolTip('Brush Selection')
        self.brushToolButton.setToolTip('Select Brush')
        self.additionalSelectionToolButton.setToolTip('Additional Selections Menu')

        self.eraseSelectionToolButton.setCheckable(True)

        self.imageAction = self.addWidget(self.imageToolButton)
        self.eraseSelectionAction = self.addWidget(self.eraseSelectionToolButton)
        self.rectSelectionAction = self.addWidget(self.rectSelectionToolButton)
        self.brushSelectionAction = self.addWidget(self.brushSelectionToolButton)
        self.brushAction = self.addWidget(self.brushToolButton)
        self.polygonSelectionAction = self.addWidget(self.polygonSelectionToolButton)
        self.additionalSelectionAction = self.addWidget(self.additionalSelectionToolButton)

        if not imageIcons:
            self.imageAction.setVisible(False)
            self.eraseSelectionAction.setVisible(False)
            self.rectSelectionAction.setVisible(False)
            self.brushSelectionAction.setVisible(False)
            self.brushAction.setVisible(False)
            self.polygonSelectionAction.setVisible(False)
            self.additionalSelectionAction.setVisible(False)

        if not polygon:
            self.polygonSelectionAction.setVisible(False)

        self._buildAdditionalSelectionMenuDict()

        self._selectionColors = numpy.zeros((len(self.colorList), 4), numpy.uint8)
        for i in range(len(self.colorList)):
            self._selectionColors[i, 0] = eval("0x" + self.colorList[i][-2:])
            self._selectionColors[i, 1] = eval("0x" + self.colorList[i][3:-2])
            self._selectionColors[i, 2] = eval("0x" + self.colorList[i][1:3])
            self._selectionColors[i, 3] = 0xff

    def activateScatterPlotView(self):
        self.brushSelectionAction.setVisible(False)
        self.brushAction.setVisible(False)
        self.eraseSelectionAction.setToolTip("Set erase mode if checked")
        self.eraseSelectionAction.setCheckable(True)

        self.eraseSelectionAction.setChecked(self.plot._eraseMode)

        self.polygonSelectionAction.setCheckable(True)
        self.rectSelectionAction.setCheckable(True)

        self.brushSelectionAction.setChecked(False)

    def activateDensityPlotView(self):
        self.brushSelectionAction.setVisible(True)
        self.brushAction.setVisible(True)
        self.rectSelectionAction.setVisible(True)

        self.eraseSelectionAction.setCheckable(True)
        self.brushSelectionAction.setCheckable(True)
        self.polygonSelectionAction.setCheckable(True)
        self.rectSelectionAction.setCheckable(True)

    def _imageIconSignal(self):
        self.plot._resetSelection(owncall=True)

    def _eraseSelectionIconSignal(self):
        self.plot._eraseMode = self.eraseSelectionAction.isChecked()

    def _getSelectionColor(self):
        color = self._selectionColors[self.plot._nRoi]
        # make sure the selection is made with a non transparent color
        if len(color) == 4:
            if type(color[-1]) in [numpy.uint8, numpy.int8]:
                color = color.copy()
                color[-1] = 255

    def _polygonIconSignal(self):
        if self.polygonSelectionAction.isChecked():
            self.plot.setInteractiveMode("draw", shape="polygon",
                                         label="mask",
                                         color=self._getSelectionColor())
            self.plot.setPolygonSelectionMode()
            self.plot._zoomMode = False
            self.plot._brushMode = False

            self.brushSelectionAction.setChecked(False)
            self.rectSelectionAction.setChecked(False)
            self.polygonSelectionAction.setChecked(True)
        else:
            self.plot.setZoomModeEnabled(True)
            self.polygonSelectionAction.setChecked(False)
            self.brushSelectionAction.setChecked(False)

    def _rectSelectionIconSignal(self):
        if self.rectSelectionAction.isChecked():
            self.plot.setInteractiveMode("draw", shape="rectangle",
                                         label="mask")
            self.plot._zoomMode = False
            self.plot._brushMode = False
            self.brushSelectionAction.setChecked(False)
            self.polygonSelectionAction.setChecked(False)
            self.rectSelectionAction.setChecked(True)

            self.setInteractiveMode("draw",
                                    shape="rectangle",
                                    label="mask",
                                    color=self._getSelectionColor())
        else:
            self.plot.setZoomModeEnabled(True)
            self.polygonSelectionAction.setChecked(False)
            self.brushSelectionAction.setChecked(False)

    def _brushSelectionIconSignal(self):
        self.polygonSelectionAction.setChecked(False)
        if self.brushSelectionAction.isChecked():
            self.plot._brushMode = True
            self.plot.setInteractiveMode('select')
        else:
            self._brushMode = False
            self.plot.setInteractiveMode('zoom')

    def _brushIconSignal(self):
        if self._brushMenu is None:
            self._brushMenu = qt.QMenu()
            self._brushMenu.addAction(QString(" 1 Image Pixel Width"),
                                      self._setBrush1)
            self._brushMenu.addAction(QString(" 2 Image Pixel Width"),
                                      self._setBrush2)
            self._brushMenu.addAction(QString(" 3 Image Pixel Width"),
                                      self._setBrush3)
            self._brushMenu.addAction(QString(" 5 Image Pixel Width"),
                                      self._setBrush4)
            self._brushMenu.addAction(QString("10 Image Pixel Width"),
                                      self._setBrush5)
            self._brushMenu.addAction(QString("20 Image Pixel Width"),
                                      self._setBrush6)
        self._brushMenu.exec_(self.cursor().pos())

    def _setBrush1(self):
        self.plot._brushWidth = 1

    def _setBrush2(self):
        self.plot._brushWidth = 2

    def _setBrush3(self):
        self.plot._brushWidth = 3

    def _setBrush4(self):
        self.plot._brushWidth = 5

    def _setBrush5(self):
        self.plot._brushWidth = 10

    def _setBrush6(self):
        self.plot._brushWidth = 20

    def _buildAdditionalSelectionMenuDict(self):
        self._additionalSelectionMenu = {}
        #scatter view menu
        menu = qt.QMenu()
        menu.addAction(QString("Density plot view"), self.__setDensityPlotView)
        menu.addAction(QString("Reset Selection"), self.__resetSelection)
        menu.addAction(QString("Invert Selection"), self.plot._invertSelection)
        self._additionalSelectionMenu["scatter"] = menu

        # density view menu
        menu = qt.QMenu()
        menu.addAction(QString("Scatter plot view"), self.__setScatterPlotView)
        menu.addAction(QString("Reset Selection"), self.__resetSelection)
        menu.addAction(QString("Invert Selection"), self.plot._invertSelection)
        menu.addAction(QString("I >= Colormap Max"), self.plot._selectMax)
        menu.addAction(QString("Colormap Min < I < Colormap Max"),
                               self.plot._selectMiddle)
        menu.addAction(QString("I <= Colormap Min"), self.plot._selectMin)
        menu.addAction(QString("Increase mask alpha"), self.plot._increaseMaskAlpha)
        menu.addAction(QString("Decrease mask alpha"), self.plot._decreaseMaskAlpha)

        self._additionalSelectionMenu["density"] = menu

    def __setScatterPlotView(self):
        self.plot.setPlotViewMode(mode="scatter")

    def __setDensityPlotView(self):
        self.plot.setPlotViewMode(mode="density")

    def __resetSelection(self):
        self.plot._resetSelection(owncall=True)

    def _additionalIconSignal(self):
        if self.plot._plotViewMode == "density":   # and imageData is not none ...
            self._additionalSelectionMenu["density"].exec_(self.cursor().pos())
        else:
            self._additionalSelectionMenu["scatter"].exec_(self.cursor().pos())

    def emitIconSignal(self, key, event="iconClicked"):
        ddict = {"key": key,
                 "event": event}
        self.sigIconSignal.emit(ddict)


