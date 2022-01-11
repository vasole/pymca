#/*##########################################################################
# Copyright (C) 2004-2014 T. Rueter, European Synchrotron Radiation Facility
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
"""This plugin opens a widget displaying values of various motors associated
with each spectrum, if the curve originates from a file whose format provides
this information.
"""

__author__ = "Tonn Rueter"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
try:
    from PyMca5 import Plugin1DBase
except ImportError:
    from . import Plugin1DBase

try:
    from PyMca5.PyMcaPlugins import MotorInfoWindow
except ImportError:
    try:
        # Frozen version
        from PyMcaPlugins import MotorInfoWindow
    except:
        print("MotorInfoPlugin importing from somewhere else")
        import MotorInfoWindow

import logging
_logger = logging.getLogger(__name__)


class MotorInfo(Plugin1DBase.Plugin1DBase):
    def __init__(self,  plotWindow,  **kw):
        Plugin1DBase.Plugin1DBase.__init__(self,  plotWindow,  **kw)
        self.methodDict = {}
        text = 'Show values of various motors.'
        function = self.showMotorInfo
        icon = None
        info = text
        self.methodDict["Show Motor Info"] =[function, info, icon]
        self.widget = None

    def getMethods(self, plottype=None):
        names = list(self.methodDict.keys())
        names.sort()
        return names

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        self.methodDict[name][0]()
        return

    def showMotorInfo(self):
        legendList,  motorValuesList = self._getLists()
        if self.widget is None:
            self._createWidget(legendList,  motorValuesList)
        else:
            self.widget.table.updateTable(legendList,  motorValuesList)
        self.widget.show()
        self.widget.raise_()

    def _getLists(self):
        curves = self.getAllCurves()
        nCurves = len(curves)
        _logger.debug("Received %d curve(s)..", nCurves)
        legendList = [leg for (xvals, yvals,  leg,  info) in curves]
        infoList = [info for (xvals, yvals,  leg,  info) in curves]
        motorValuesList = self._convertInfoDictionary(infoList)
        return legendList,  motorValuesList

    def _convertInfoDictionary(self,  infosList):
        ret = []
        for info in infosList :
            motorNames = info.get('MotorNames',  None)
            if motorNames is not None:
                if type(motorNames) == str:
                    namesList = motorNames.split()
                elif type(motorNames) == list:
                    namesList = motorNames
                else:
                    namesList = []
            else:
                namesList = []
            motorValues = info.get('MotorValues',  None)
            if motorNames is not None:
                if type(motorValues) == str:
                    valuesList = motorValues.split()
                elif type(motorValues) == list:
                    valuesList = motorValues
                else:
                    valuesList = []
            else:
                valuesList = []
            if len(namesList) == len(valuesList):
                ret.append( dict( zip( namesList,  valuesList ) ) )
            else:
                print("Number of motors and values does not match!")
        return ret

    def _createWidget(self,  legendList,  motorValuesList):
        parent = None
        self.widget = MotorInfoWindow.MotorInfoDialog(parent,
                                                      legendList,
                                                      motorValuesList)
        self.widget.buttonUpdate.clicked.connect(self.showMotorInfo)
        self.widget.updateShortCut.activated.connect(self.showMotorInfo)

MENU_TEXT = "Motor Info"
def getPlugin1DInstance(plotWindow,  **kw):
    ob = MotorInfo(plotWindow)
    return ob

if __name__ == "__main__":
    # Basic test setup
    import numpy
    from PyMca5.PyMcaGraph import Plot
    from PyMca5.PyMcaGui import PyMcaQt as qt
    app = qt.QApplication([])
    x = numpy.arange(100.)
    y = numpy.arange(100.)
    plot = Plot.Plot()
    plot.addCurve(x, y, "Curve1", {'MotorNames': "foo bar",  'MotorValues': "3.14 2.97"})
    plot.addCurve(x+100, y, "Curve2", {'MotorNames': "baz",  'MotorValues': "6.28"})
    plugin = getPlugin1DInstance(plot)
    plugin.applyMethod(plugin.getMethods()[0])

    app.exec()
