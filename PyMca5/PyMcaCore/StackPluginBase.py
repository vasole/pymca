#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
__doc__="""

A Stack plugin is a module that will be automatically added to the PyMca stack windows
in order to perform user defined operations on the data stack. It has to
inherit the StackPluginBase.StackPluginBase class and implement the methods:

    - getMethods
    - getMethodToolTip (optional but convenient)
    - getMethodPixmap (optional)
    - applyMethod

and modify the static module variable MENU_TEXT and the static module function
getStackPluginInstance according to the defined plugin.

These plugins will be compatible with any stack window that provides the functions:

    #data related

    - getStackDataObject
    - getStackData
    - getStackInfo
    - setStack
    - getStackROIImagesAndNames
    - isStackFinite

    #mask related

    - setStackSelectionMask
    - getStackSelectionMask

    #displayed curves

    - getActiveCurve
    - getGraphXLimits
    - getGraphYLimits
    - getGraphXLabel
    - getGraphYLabel

    #images

    - addImage
    - removeImage
    - replaceImage

    #information method

    - stackUpdated
    - selectionMaskUpdated

"""
import weakref
DEBUG = 0

class StackPluginBase(object):
    def __init__(self, stackWindow, **kw):
        """
        stackWindow is the object instantiating the plugin.

        Unless one knows what (s)he is doing, only a proxy should be used.

        I pass the actual instance to keep all doors open.
        """
        self._stackWindow = weakref.proxy(stackWindow)
        pass

    #stack related functions
    def isStackFinite(self):
        return self._stackWindow.isStackFinite()

    def getStackROIImagesAndNames(self):
        return self._stackWindow.getStackROIImagesAndNames()

    def getStackDataObject(self):
        return self._stackWindow.getStackDataObject()

    def getStackDataObjectList(self):
        return self._stackWindow.getStackDataObjectList()

    def getStackData(self):
        return self._stackWindow.getStackData()

    def getStackOriginalImage(self):
        return self._stackWindow.getStackOriginalImage()

    def getStackInfo(self):
        return self._stackWindow.getStackInfo()

    def getStackSelectionMask(self):
        return self._stackWindow.getSelectionMask()

    def setStackSelectionMask(self, mask, instance_id=None):
        if instance_id is None:
            instance_id = id(self)
        return self._stackWindow.setSelectionMask(mask,
                                                  instance_id=instance_id)

    def setStack(self, *var, **kw):
        return self._stackWindow.setStack(*var, **kw)

    def addImage(self, image, name):
        return self._stackWindow.addImage(image, name)

    def removeImage(self, name):
        return self._stackWindow.removeImage(name)

    def replaceImage(self, image, name):
        return self._stackWindow.replaceImage(image, name)

    #Plot window related functions
    def getActiveCurve(self):
        """
        Function to access the currently active curve.
        It returns None in case of not having an active curve.

        Output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.
        """
        return self._stackWindow.getActiveCurve()

    def getGraphXLimits(self):
        """
        Get the graph X limits.
        """
        return self._stackWindow.getGraphXLimits()

    def getGraphYLimits(self):
        """
        Get the graph Y limits.
        """
        return self._stackWindow.getGraphYLimits()

    def getGraphXLabel(self):
        """
        Get the graph X label
        """
        return self._stackWindow.getGraphXLabel()

    def getGraphYLabel(self):
        """
        Get the graph Y label
        """
        return self._stackWindow.getGraphYLabel()

    def stackUpdated(self):
        if DEBUG:
            print("stackUpdated(self) not implemented")

    def stackROIImageListUpdated(self):
        if DEBUG:
            print("stackROIImageListUpdated(self) not implemented")
        return

    def selectionMaskUpdated(self):
        if DEBUG:
            print("selectionMaskUpdated(self) not implemented")

    #Methods to be implemented by the plugin
    def getMethods(self):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified stack.
        """
        print("BASE STACK getMethods not implemented")
        return []

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return None

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        print("applyMethod not implemented")
        return

MENU_TEXT = "StackPluginBase"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackPluginBase(stackWindow)
    return ob
