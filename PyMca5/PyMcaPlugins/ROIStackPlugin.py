#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
"""
This plugin opens a stack ROI window providing alternative views:

 - Usual sum of counts in the region
 - Channel/Energy at data Max in the region.
 - Channel/Energy at data Min in the region.
 - Map of the counts at the first channel of the region
 - Map of the counts at the middle cahnnel of the region
 - Map of the counts at the last channel of the region
 - Background counts

This window also provides a median filter tool, with a configurable filter
width, to get rid of outlier pixel.

The mask of this plot widget is synchronized with the master stack widget.

"""
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import logging
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.pymca import StackROIWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons

_logger = logging.getLogger(__name__)


class ROIStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show':[self._showWidget,
                                   "Show ROIs",
                                   PyMca_Icons.brushselect]}
        self.__methodKeys = ['Show']
        self.roiWindow = None

    def stackUpdated(self):
        _logger.debug("ROIStackPlugin.stackUpdated() called")
        if self.roiWindow is None:
            return
        if self.roiWindow.isHidden():
            return
        images, names = self.getStackROIImagesAndNames()
        self.roiWindow.setImageList(images, imagenames=names, dynamic=False)
        mask = self.getStackSelectionMask()
        self.roiWindow.setSelectionMask(mask)

    def selectionMaskUpdated(self):
        if self.roiWindow is None:
            return
        if self.roiWindow.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.roiWindow.setSelectionMask(mask)

    def stackClosed(self):
        if self.roiWindow is not None:
            self.roiWindow.close()

    def stackROIImageListUpdated(self):
        self.stackUpdated()

    def mySlot(self, ddict):
        _logger.debug("mySlot %s %s", ddict['event'], ddict.keys())
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
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
        if self.roiWindow is None:
            self.roiWindow = StackROIWindow.StackROIWindow(parent=None,
                                                        crop=False,
                                                        rgbwidget=None,
                                                        selection=True,
                                                        colormap=True,
                                                        imageicons=True,
                                                        standalonesave=True,
                                                        profileselection=True)
            self.roiWindow.setSelectionMode(True)
            qt = StackROIWindow.qt
            self.roiWindow.sigMaskImageWidgetSignal.connect(self.mySlot)

        #Show
        self.roiWindow.show()
        self.roiWindow.raise_()

        #update ROIs
        self.stackUpdated()


MENU_TEXT = "Alternative ROI Options"
def getStackPluginInstance(stackWindow, **kw):
    ob = ROIStackPlugin(stackWindow)
    return ob
