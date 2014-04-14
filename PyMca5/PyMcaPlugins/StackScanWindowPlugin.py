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
import numpy

from PyMca5.PyMcaGui import ScanWindow
from PyMca5 import StackPluginBase

DEBUG = 0

class StackScanWindowPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        text  = "Add active curve to plugin scan window\n"
        function = self.addActiveCurve
        info = text
        icon = None
        self.methodDict["ADD"] =[function,
                                 info,
                                 icon]
        text  = "Replace scan window curves with current active curve\n"
        function = self.replaceByActiveCurve
        info = text
        icon = None
        self.methodDict["REPLACE"] =[function,
                                     info,
                                     icon]
        self.__methodKeys = ["ADD",
                             "REPLACE"]
        self.widget = None
        
    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def addActiveCurve(self):
        self._add(replace=False)

    def replaceByActiveCurve(self):
        self._add(replace=True)

    def _add(self, replace=False):
        curve = self.getActiveCurve()
        if curve is None:
            text = "Please make sure to have an active curve"
            raise TypeError(text)
        x, y, legend, info = self.getActiveCurve()
        if self.widget is None:
            self.widget = ScanWindow.ScanWindow()
        self.widget.addCurve(x, y, legend=legend, replot=True, replace=replace)
        self.widget.show()
        self.widget.raise_()

MENU_TEXT = "Stack Scan Window Plugin"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackScanWindowPlugin(stackWindow)
    return ob
