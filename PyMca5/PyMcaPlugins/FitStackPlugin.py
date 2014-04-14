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
try:
    from PyMca5 import StackPluginBase
    from PyMca5.PyMcaGui import StackSimpleFitWindow
    from PyMca5.PyMcaGui import PyMca_Icons
except ImportError:
    print("FitStackPlugin importing from somewhere else")

DEBUG = 0

class FitStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        function = self.fitStack
        info = "Fit stack with user defined functions"
        icon = PyMca_Icons.fit
        self.methodDict["Fit Stack"] =[function,
                                       info,
                                       icon]
        self.__methodKeys = ["Fit Stack"]
        self.simpleFitWindow = None
                                     
    def stackUpdated(self):
        if self.simpleFitWindow is None:
            return
        self.__updateOwnData()

    def selectionMaskUpdated(self):
        if self.simpleFitWindow is None:
            return
        self.simpleFitWindow.setMask(self.getStackSelectionMask())
    
    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def __updateOwnData(self):
        activeCurve = self.getActiveCurve()
        if activeCurve is None:
            return
        #this can be problematic if a fit is going on...
        x, spectrum, legend, info = activeCurve
        xlabel = info['xlabel']
        ylabel = info['ylabel']
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits()
        mcaIndex = self.getStackInfo()['McaIndex']
        self.simpleFitWindow.setSpectrum(x,
                                         spectrum,
                                         xmin=xmin,
                                         xmax=xmax)
        self.simpleFitWindow.setData(x,
                                     self.getStackData(),
                                     data_index=mcaIndex,
                                     mask=self.getStackSelectionMask())

    def fitStack(self):
        if self.simpleFitWindow is None:
            self.simpleFitWindow = StackSimpleFitWindow.StackSimpleFitWindow()
        self.__updateOwnData()
        self.simpleFitWindow.show()

MENU_TEXT = "Stack Simple Fitting"
def getStackPluginInstance(stackWindow, **kw):
    ob = FitStackPlugin(stackWindow)
    return ob
