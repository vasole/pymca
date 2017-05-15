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
"""Plugin to apply a median filter on the ROI stack.

The mask of the plot widget is synchronized with the master stack widget.
"""
# TODO: docstring

__authors__ = ["V.A. Sole", "P. Knobel"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"


from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.pymca import SilxStackROIWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons


class SilxRoiStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow)
        self.methodDict = {'Show': [self._showWidget,
                                    "Show ROIs",
                                    PyMca_Icons.brushselect]}
        self.__methodKeys = ['Show']
        self.widget = None

    def stackUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        images, names = self.getStackROIImagesAndNames()
        self.widget.setImages(images, labels=names)

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def stackROIImageListUpdated(self):
        self.stackUpdated()

    def mySlot(self, ddict):
        if ddict['event'] == "selectionMaskChanged":
            mask = ddict["current"]
            # The mask is temporarily reset when image is changed,
            # on ROI selection. Ignore this.
            if hasattr(mask, "shape") and sum(mask.shape) > 0:
                self.setStackSelectionMask(mask)
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _showWidget(self):
        if self.widget is None:
            self.widget = SilxStackROIWindow.SilxStackROIWindow()
            self.widget.sigExternalImagesWindowSignal.connect(self.mySlot)

        # Show
        self.widget.show()
        self.widget.raise_()

        self.stackUpdated()


MENU_TEXT = "Silx Alternative ROI Options"


def getStackPluginInstance(stackWindow):
    ob = SilxRoiStackPlugin(stackWindow)
    return ob
