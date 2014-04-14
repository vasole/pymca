#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""

A Stack plugin is a module that will be automatically added to the PyMca
stack windows in order to perform user defined operations on the data stack.

These plugins will be compatible with any stack window that provides the
functions:
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
from PyMca5.PyMcaGui import SGWindow
from PyMca5.PyMcaGui import SNIPWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons

import numpy

DEBUG = 0


class BackgroundStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        SGtext = "Replace current stack by a\n"
        SGtext += "Savitsky-Golay treated one."
        SNIP1Dtext = "Replace current stack by a\n"
        SNIP1Dtext += "SNIP1D treated one."
        SNIP2Dtext = "Replace current stack by a\n"
        SNIP2Dtext += "SNIP2D treated one."
        self.methodDict = {}
        function = self.replaceStackWithSavitzkyGolayFiltering
        info = SGtext
        icon = PyMca_Icons.substract
        self.methodDict["Savitzky-Golay Filtering"] = [function, info, icon]
        function = self.subtract1DSnipBackground
        info = SNIP1Dtext
        self.methodDict["Subtract SNIP 1D Background"] = [function, info, icon]
        function = self.replaceWith1DSnipBackground
        info = "Smooth and replace current stack\n"
        info += "by its SNIP1D background."
        self.methodDict["Deglitch with SNIP 1D Background"] = [function,
                                                              info,
                                                              PyMca_Icons.smooth]
        function = self.subtract2DSnipBackground
        info = SNIP2Dtext
        self.methodDict["Subtract SNIP 2D Background"] = [function, info, icon]

        function = self.subtractActiveCurve
        info = "Replace current stack by one in which\nthe active curve has been subtracted"
        self.methodDict["Subtract active curve"] = [function, info, icon]

        self.__methodKeys = ["Savitzky-Golay Filtering",
                             "Deglitch with SNIP 1D Background",
                             "Subtract SNIP 1D Background",
                             "Subtract SNIP 2D Background",
                             "Subtract active curve"]

    def stackUpdated(self):
        self.dialogWidget = None

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def replaceStackWithSavitzkyGolayFiltering(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SGWindow.SGDialog(None,
                                           spectrum, x=x)
        snipWindow.graph.setGraphXLabel(info['xlabel'])
        snipWindow.graph.setGraphYLabel(info['ylabel'])
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)

    def subtract1DSnipBackground(self, smooth=False):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        x, spectrum, legend, info = activeCurve
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           spectrum, x=x, smooth=smooth)
        snipWindow.graph.setGraphXLabel(info['xlabel'])
        snipWindow.graph.setGraphYLabel(info['ylabel'])
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)

    def replaceWith1DSnipBackground(self):
        return self.subtract1DSnipBackground(smooth=True)

    def subtract2DSnipBackground(self):
        imageList = self.getStackROIImagesAndNames()
        if imageList is None:
            return
        imageList, imageNames = imageList
        if not len(imageList):
            return
        snipWindow = SNIPWindow.SNIPDialog(None,
                                           imageList[0] * 1)
        #snipWindow.setModal(True)
        snipWindow.show()
        ret = snipWindow.exec_()
        if ret:
            snipParametersDict = snipWindow.getParameters()
            snipWindow.close()
            function = snipParametersDict['function']
            arguments = snipParametersDict['arguments']
            stack = self.getStackDataObject()
            function(stack, *arguments)
            self.setStack(stack)

    def subtractActiveCurve(self):
        curve = self.getActiveCurve()
        if curve is None:
            raise ValueError("No active curve")
        x, y = curve[0:2]
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        mcaIndex = stack.info.get('McaIndex', -1)
        if mcaIndex in [-1, 2]:
            for i in range(stack.data.shape[-1]):
                stack.data[:, :, i] -= y[i]
        elif mcaIndex == 0:
            for i in range(stack.data.shape[0]):
                stack.data[i] -= y[i]
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        self.setStack(stack) 

MENU_TEXT = "Stack Filtering Options"

def getStackPluginInstance(stackWindow, **kw):
    ob = BackgroundStackPlugin(stackWindow)
    return ob
