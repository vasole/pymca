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
"""
This plugin open a new stack window with a configurable median filter
applied to the data.

The user can select the frame and modify the filter width.

He can also choose to apply a conditional median filter instead
of the regular median filter.
With a regular median filter, each pixel value is replaced with the median
value of all pixels in the window. With the conditional option enabled,
the pixel value is only replaced if it is the minimum or the maximum value
in the smoothing window.
"""
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import logging
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.pymca import Median2DBrowser
from PyMca5.PyMcaGui import PyMca_Icons

_logger = logging.getLogger(__name__)


class MedianFilterStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show':[self._showWidget,
                                   "Image Browser with Median Filter",
                                   PyMca_Icons.brushselect]}
        self.__methodKeys = ['Show']
        self.widget = None

    def stackUpdated(self):
        _logger.debug("StackBrowserPlugin.stackUpdated() called")
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        stack = self.getStackDataObject()
        self.widget.setStackDataObject(stack, stack_name="Stack Index")
        self.widget.setBackgroundImage(self._getBackgroundImage())
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def _getBackgroundImage(self):
        images, names = self.getStackROIImagesAndNames()
        B = None
        for key in names:
            if key.endswith("ackground"):
                B = images[names.index(key)]
        return B

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

    def stackROIImageListUpdated(self):
        if self.widget is None:
            return
        self.widget.setBackgroundImage(self._getBackgroundImage())

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
        if self.widget is None:
            self.widget = Median2DBrowser.Median2DBrowser(parent=None,
                                                    rgbwidget=None,
                                                    selection=True,
                                                    colormap=True,
                                                    imageicons=True,
                                                    standalonesave=True,
                                                    profileselection=True)
            self.widget.setKernelWidth(1)
            self.widget.setSelectionMode(True)
            qt = Median2DBrowser.qt
            self.widget.sigMaskImageWidgetSignal.connect(self.mySlot)

        #Show
        self.widget.show()
        self.widget.raise_()

        #update
        self.stackUpdated()


MENU_TEXT = "Image Browser with Median Filter"
def getStackPluginInstance(stackWindow, **kw):
    ob = MedianFilterStackPlugin(stackWindow)
    return ob
