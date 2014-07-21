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
"""

A Stack plugin is a module that will be automatically added to the PyMca stack windows
in order to perform user defined operations on the data stack.

These plugins will be compatible with any stack window that provides the functions:
    #data related
    getStackDataObject
    getStackData
    getStackInfo
    setStack

    #images related
    addImage
    removeImage
    replaceImage

    #mask related
    setSelectionMask
    getSelectionMask

    #displayed curves
    getActiveCurve
    getGraphXLimits
    getGraphYLimits

    #information method
    stackUpdated
    selectionMaskUpdated
"""
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.pymca import StackROIWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons

DEBUG = 0

class ROIStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show':[self._showWidget,
                                   "Show ROIs",
                                   PyMca_Icons.brushselect]}
        self.__methodKeys = ['Show']
        self.roiWindow = None

    def stackUpdated(self):
        if DEBUG:
            print("ROIStackPlugin.stackUpdated() called")
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

    def stackROIImageListUpdated(self):
        self.stackUpdated()

    def mySlot(self, ddict):
        if DEBUG:
            print("mySlot ", ddict['event'], ddict.keys())
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
