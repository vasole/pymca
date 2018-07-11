#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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
"""This plugin replaces all curves with a normalized and shifted
curve.
The minimum is subtracted, than the data is normalized to the maximum, and
finally it is shifted up by i*0.1 (i being the curve index).
"""

__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"


from PyMca5 import Plugin1DBase

class Shifting(Plugin1DBase.Plugin1DBase):

    def getMethods(self, plottype=None):
        return ["Shift"]
    def getMethodToolTip(self, methodName):
        if methodName != "Shift":
            raise KeyError("Method %s not valid" % methodName)
        return "Subtract minimum, normalize to maximum, and shift up by 0.1"

    def applyMethod(self, methodName):
        if methodName != "Shift":
            raise ValueError("Method %s not valid" % methodName)
        allCurves = self.getAllCurves()
        increment = 0.1
        for i in range(len(allCurves)):
            x, y, legend, info = allCurves[i][:4]
            delta = float(y.max() - y.min())
            if delta < 1.0e-15:
                delta = 1.0
            y = (y - y.min())/delta + i * increment
            if i == (len(allCurves) - 1):
                replot = True
            else:
                replot = False
            if i == 0:
                replace = True
            else:
                replace = False
            self.addCurve(x, y, legend=legend + " %.2f" % (i * increment),
                          info=info, replace=replace, replot=replot)
     
MENU_TEXT="Simple Vertical Shift"
def getPlugin1DInstance(plotWindow, **kw):
        ob = Shifting(plotWindow)
        return ob
