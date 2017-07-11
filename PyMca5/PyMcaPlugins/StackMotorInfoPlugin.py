# /*#########################################################################
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
# ###########################################################################*/
"""Plugin opening a stack plot and a another window displaying motor positions
at the current mouse position."""

__authors__ = ["P. Knobel"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"

import numpy

from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.plotting import SilxMaskImageWidget
from . import MotorInfoWindow


class StackMotorInfoPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow)
        self.methodDict = {'Show motor positions':
                               [self._showWidgets, "Show motor positions in a popup window"], }
        self.__methodKeys = ['Show motor positions']
        self.maskImageWidget = None
        self.motorPositionsWindow = None

    def stackClosed(self):
        if self.maskImageWidget is not None:
            self.maskImageWidget.close()
        if self.motorPositionsWindow is not None:
            self.motorPositionsWindow.close()

    def _getStackOriginDelta(self):
        """Return (originX, originY) and (deltaX, deltaY)
        """
        info = self.getStackInfo()

        xscale = info.get("xScale", [0.0, 1.0])
        yscale = info.get("yScale", [0.0, 1.0])

        origin = xscale[0], yscale[0]
        delta = xscale[1], yscale[1]

        return origin, delta

    def stackUpdated(self):
        if self.maskImageWidget is None:
            return
        if self.maskImageWidget.isHidden():
            return
        images, names = self.getStackROIImagesAndNames()

        image_shape = list(self.getStackOriginalImage().shape)
        origin, delta = self._getStackOriginDelta()

        h = delta[1] * image_shape[0]
        w = delta[0] * image_shape[1]

        self.maskImageWidget.setImages(images, labels=names,
                                       origin=origin, width=w, height=h)

        self.maskImageWidget.setSelectionMask(self.getStackSelectionMask())

    def selectionMaskUpdated(self):
        if self.maskImageWidget is None:
            return
        mask = self.getStackSelectionMask()
        if not self.maskImageWidget.isHidden():
            self.maskImageWidget.setSelectionMask(mask)

    def onWidgetSignal(self, ddict):
        """triggered by self.widget.sigMaskImageWidget"""
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict["current"])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] in ["addImageClicked", "replaceImageClicked"]:
            if ddict['event'] == "addImageClicked":
                self.addImage(ddict['image'], ddict['title'])
            elif ddict['event'] == "replaceImageClicked":
                self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)

    def _showWidgets(self):
        if self.maskImageWidget is None:
            self.maskImageWidget = SilxMaskImageWidget.SilxMaskImageWidget()
            self.maskImageWidget.sigMaskImageWidget.connect(self.onWidgetSignal)

        if self.motorPositionsWindow is None:
            legends = ["Stack"]
            motorValues = {}
            self.motorPositionsWindow = MotorInfoWindow.MotorInfoDialog(None,
                                                                        legends,
                                                                        [motorValues])
            self.maskImageWidget.plot.sigPlotSignal.connect(self._updateMotors)

        # Show
        self.motorPositionsWindow.show()
        self.motorPositionsWindow.raise_()

        self.maskImageWidget.show()
        self.maskImageWidget.raise_()

        self.stackUpdated()    # fixme: is this necessary?

    def _updateMotors(self, ddict):
        if not ddict["event"] == "mouseMoved":
            return
        nRows, nCols = self.getStackOriginalImage().shape
        r, c = SilxMaskImageWidget.convertToRowAndColumn(
                ddict["x"], ddict["y"],
                shape=(nRows, nCols),
                xScale=self.getStackInfo().get("xScale"),
                yScale=self.getStackInfo().get("yScale"),
                safe=True)

        positioners = self.getStackInfo().get("positioners", {})
        motorsValuesAtCursor = {}
        for motorName, motorValues in positioners.items():
            if numpy.isscalar(motorValues) or (hasattr(motorValues, "ndim") and
                                               motorValues.ndim == 0):
                # scalar
                motorsValuesAtCursor[motorName] = motorValues
            else:
                # must be a numpy array
                assert hasattr(motorValues, "ndim") and \
                       hasattr(motorValues, "shape")
                if motorValues.ndim == 2:
                    # image
                    assert motorValues.shape == (nRows, nCols), \
                        "wrong shape for motor values frame"
                    motorsValuesAtCursor[motorName] = motorValues[r, c]
                elif motorValues.ndim == 1:
                    # 1D array
                    nPixels = nRows * nCols
                    assert len(motorValues) == nPixels, \
                        "wrong number of motor values in array"
                    motorIndex = r * nCols + c
                    motorsValuesAtCursor[motorName] = motorValues[motorIndex]

        self.motorPositionsWindow.table.updateTable(
                legList=["Stack"],
                motList=[motorsValuesAtCursor])

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        if len(self.methodDict[name]) < 3:
            return None
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()


MENU_TEXT = "Stack motor positions"


def getStackPluginInstance(stackWindow, **kw):
    ob = StackMotorInfoPlugin(stackWindow)
    return ob
