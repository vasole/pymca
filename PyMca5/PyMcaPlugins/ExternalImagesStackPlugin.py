#/*##########################################################################
# Copyright (C) 2004-2019 V.A. Sole, European Synchrotron Radiation Facility
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
This plugin open a file selection dialog to open one or more images in a
new window. Usual image data formats are supported, as well as standard
image formats (JPG, PNG).

The tool is meant to view an alternative view of the data, such as a
photograph of the sample or a different type of scientific measurement
of the same sample.

The window offer a cropping tool, to crop the image to the current visible
zoomed area and then resize it to fit the original size.

The mask of this plot widget is synchronized with the master stack widget.
"""
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import os
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.pymca import ExternalImagesWindow
from PyMca5.PyMcaGui.pymca import ExternalImagesStackPluginBase
from PyMca5.PyMcaGui.pymca import StackPluginResultsWindow
from PyMca5.PyMcaGui.plotting import PyMca_Icons as PyMca_Icons

_logger = logging.getLogger(__name__)

class ExternalImagesStackPlugin( \
    ExternalImagesStackPluginBase.ExternalImagesStackPluginBase):

    def __init__(self, stackWindow, **kw):
        ExternalImagesStackPluginBase.ExternalImagesStackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Load': [self._loadImageFiles,
                                    "Load Images",
                                    PyMca_Icons.fileopen],
                           'Show': [self._showWidget,
                                    "Show Image Browser",
                                    PyMca_Icons.brushselect]}
        self.__methodKeys = ['Load', 'Show']
        self.widget = None

    def stackUpdated(self):
        self.widget = None

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)

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
        if self.widget is None:
            return [self.__methodKeys[0]]
        else:
            return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _createStackPluginWindow(self, imagenames, imagelist):
        self.widget = StackPluginResultsWindow.StackPluginResultsWindow(parent=None,
                                                usetab=False)
        self.widget.buildAndConnectImageButtonBox()
        self.widget.sigMaskImageWidgetSignal.connect(self.mySlot)
        self.widget.setStackPluginResults(imagelist,
                                          image_names=imagenames)
        self._showWidget()

    def _createStackPluginWindowQImage(self, imagenames, imagelist):
        self.widget = ExternalImagesWindow.ExternalImagesWindow(parent=None,
                                                rgbwidget=None,
                                                selection=True,
                                                colormap=True,
                                                imageicons=True,
                                                standalonesave=True)
        self.widget.buildAndConnectImageButtonBox()
        self.widget.sigMaskImageWidgetSignal.connect(self.mySlot)
        self.widget.setImageData(None)
        shape = self._requiredShape
        self.widget.setQImageList(imagelist, shape[1], shape[0],
                                  clearmask=False,
                                  data=None,
                                  imagenames=imagenames)
                                  #data=self.__stackImageData)
        self._showWidget()

    def _showWidget(self):
        if self.widget is None:
            return
        self.widget.show()
        self.widget.raise_()
        self.selectionMaskUpdated()

    @property
    def _dialogParent(self):
        return self.widget


MENU_TEXT = "External Images Tool"
def getStackPluginInstance(stackWindow, **kw):
    ob = ExternalImagesStackPlugin(stackWindow)
    return ob
