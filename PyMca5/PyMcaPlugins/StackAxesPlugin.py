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
import numpy
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui import PyMcaFileDialogs
from PyMca5.PyMcaGui import PyMca_Icons

DEBUG = 0

class StackAxesPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        text = "Replace current 1D axis by list of numbers found in ASCII file"
        self.methodDict = {}
        function = self.replace1DAxisWithASCII
        info = text
        icon = None
        self.methodDict["1D axis from ASCII file"] = [function, info, icon]
        self.__methodKeys = ["1D axis from ASCII file"]

        function = self.replace1DAxisWithActiveCurveXValues
        text = "Replace current 1D axis by X values in current MCA curve"
        info = text
        icon = None
        self.methodDict["1D axis from MCA curve X values"] = [function, info, icon]
        self.__methodKeys.append("1D axis from MCA curve X values")

        function = self.replace1DAxisWithActiveCurveYValues
        text = "Replace current 1D axis by Y values in current MCA curve"
        info = text
        icon = None
        self.methodDict["1D axis from MCA curve Y values"] = [function, info, icon]
        self.__methodKeys.append("1D axis from MCA curve Y values")

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def replace1DAxisWithASCII(self):
        stack = self.getStackDataObject()
        mcaIndex = stack.info.get('McaIndex', -1)
        nPoints = stack.data.shape[mcaIndex]
        fileList = PyMcaFileDialogs.getFileList(None,
                                           filetypelist=["ASCII files (*)"],
                                           message="Select ASCII file",
                                           mode="OPEN",
                                           getfilter=False,
                                           single=True)
        if not len(fileList):
            return

        filename = fileList[0]
        data = numpy.loadtxt(filename)
        data.shape = -1
        if data.size != nPoints:
            raise ValueError("Number of read values not equal to %d" % nPoints)
        else:
            stack.x = [data]
            self.setStack(stack, mcaindex=mcaIndex)

    def replace1DAxisWithActiveCurveYValues(self):
        stack = self.getStackDataObject()
        mcaIndex = stack.info.get('McaIndex', -1)
        nPoints = stack.data.shape[mcaIndex]
        curve = self.getActiveCurve()
        data = curve[1]
        data.shape = -1
        if data.size != nPoints:
            raise ValueError("Number of read values not equal to %d" % nPoints)
        else:
            stack.x = [data]
            self.setStack(stack, mcaindex=mcaIndex)

    def replace1DAxisWithActiveCurveXValues(self):
        stack = self.getStackDataObject()
        mcaIndex = stack.info.get('McaIndex', -1)
        nPoints = stack.data.shape[mcaIndex]
        curve = self.getActiveCurve()
        data = curve[0]
        data.shape = -1
        if data.size != nPoints:
            raise ValueError("Number of read values not equal to %d" % nPoints)
        else:
            stack.x = [data]
            self.setStack(stack, mcaindex=mcaIndex)

MENU_TEXT = "Stack Axes Options"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackAxesPlugin(stackWindow)
    return ob
