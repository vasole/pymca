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
This stack plugin provides filtering and background subtraction methods:

  - Savitzky-Golay Filtering (smoothing)
  - Deglitch with SNIP 1D Background
  - Subtract SNIP 1D Background
  - Subtract SNIP 2D Background
  - Subtract active curve
"""

__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import logging
from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui import SGWindow
from PyMca5.PyMcaGui import SNIPWindow
from PyMca5.PyMcaGui import PyMca_Icons as PyMca_Icons

import numpy

_logger = logging.getLogger(__name__)


class BackgroundStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
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
        ret = snipWindow.exec()
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
        ret = snipWindow.exec()
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
        ret = snipWindow.exec()
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
        x, y, legend, info = curve[:4]
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        x0, y0, legend0, info0 = self.getStackOriginalCurve()[:4]
        if self.getGraphXLabel().upper() not in ["CHANNEL", "POINTS"]:
            # get the used calibration
            a, b, c = info.get("McaCalib", [0.0, 1,0, 0.0])
            x0 = a + b * x0 + c * x0 * x0 
        interpolate = True
        if x0.size == x.size:
            if numpy.allclose(x, x0):
                interpolate = False
        if interpolate:
            xmin = max(x.min(), x0.min())
            xmax = min(x.max(), x0.max())
            selection = (x0 >= xmin) & (x0 <= xmax)
            idx = numpy.nonzero(selection)[0]
            if not idx.size:
                raise ValueError("Curves do not overlap")
            xwork = x0[idx]
            # we have got the final x values, interpolate into the
            # curve to be subtracted
            ywork = numpy.interp(xwork, x, y)
            #proceed to subtract the data
            notIdx = numpy.nonzero(selection == False)[0]
            mcaIndex = stack.info.get('McaIndex', -1)
            if mcaIndex in [-1, 2]:
                #for i in range(stack.data.shape[-1]):
                stack.data[:, :, idx] = stack.data[:, :, idx] - ywork
                stack.data[:, :, notIdx] = 0.0
            elif mcaIndex == 0:
                counter = 0
                for i in range(stack.data.shape[0]):
                    if i in idx:
                        stack.data[i] -= ywork[counter]
                        counter += 1
                    else:
                        stack.data[i] = 0.0
            else:
                raise ValueError("Invalid 1D index %d" % mcaIndex)
        else:
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
