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
"""This plugin opens a scan window the first time it is called.
The user can then send the current active curve to it, for further
analysis.
"""

__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy
import logging

from PyMca5.PyMcaGui import ScanWindow
from PyMca5 import StackPluginBase

_logger = logging.getLogger(__name__)


class StackScanWindowPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
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
