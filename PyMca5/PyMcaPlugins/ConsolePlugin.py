#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
import numpy
try:
    from PyMca5 import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

from PyMca5.PyMcaGui.misc  import QIPythonWidget


class ConsolePlugin(Plugin1DBase.Plugin1DBase):
    def __init__(self, plotWindow, **kw):
       Plugin1DBase.Plugin1DBase.__init__(self, plotWindow, **kw)
       self.methodDict = {}
       self.methodDict["console"] = [self._embed,
                                     "Open IPython console",
                                     None]
       self._widget = None

    #Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """
        A list with the NAMES  associated to the callable methods
        that are applicable to the specified plot.

        Plot type can be "SCAN", "MCA", None, ...
        """
        names = list(self.methodDict.keys())
        names.sort()
        return names

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.
        """
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        """
        Returns the pixmap associated to the particular method name or None.
        """
        return self.methodDict[name][2]

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        self.methodDict[name][0]()
        return

    def _embed(self):
        if self._widget is None:
            try:
                banner = "%s Console 1D Window.\n" % self.windowTitle()
            except:
                banner = "%s Console 1D Window.\n" % self.windowTitle()
            banner += "Use plt to access the plot.\n"
            banner += "Use plugin to access the plugin interface.\n"
            self._widget = QIPythonWidget.QIPythonWidget(customBanner=banner)
            self._widget.pushVariables({"plt": self._plotWindow,
                                        "plugin": self})
        self._widget.show()
        self._widget.raise_()

MENU_TEXT = "Interactive Console"
def getPlugin1DInstance(plotWindow, **kw):
    ob = ConsolePlugin(plotWindow)
    return ob
