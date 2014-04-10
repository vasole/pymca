#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
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
try:
    from PyMca5 import StackPluginBase
except ImportError:
    from . import StackPluginBase

from PyMca5.PyMcaGui import StackROIWindow
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
            qt.QObject.connect(self.roiWindow,
                   qt.SIGNAL('MaskImageWidgetSignal'),
                   self.mySlot)

        #Show
        self.roiWindow.show()
        self.roiWindow.raise_()        

        #update ROIs
        self.stackUpdated()


MENU_TEXT = "Alternative ROI Options"
def getStackPluginInstance(stackWindow, **kw):
    ob = ROIStackPlugin(stackWindow)
    return ob
