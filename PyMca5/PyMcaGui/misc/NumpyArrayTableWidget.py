#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
from PyMca5.PyMcaGui import PyMcaQt as qt
from . import FrameBrowser
from . import NumpyArrayTableView

class BrowserContainer(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)

class NumpyArrayTableWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QTableWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.browserContainer = BrowserContainer(self)
        self._widgetList = []
        for i in range(4):
            browser = FrameBrowser.HorizontalSliderWithBrowser(self.browserContainer)
            self.browserContainer.mainLayout.addWidget(browser)
            self._widgetList.append(browser)
            browser.valueChanged.connect(self.browserSlot)
            if i == 0:
                browser.setEnabled(False)
                browser.hide()
        self.view = NumpyArrayTableView.NumpyArrayTableView(self)
        self.mainLayout.addWidget(self.browserContainer)
        self.mainLayout.addWidget(self.view)

    def setArrayData(self, data):
        self._array = data
        nWidgets = len(self._widgetList)
        nDimensions = len(self._array.shape) 
        if nWidgets > (nDimensions - 2):
            for i in range((nDimensions - 2), nWidgets):
                browser = self._widgetList[i]
                self._widgetList[i].setEnabled(False)
                self._widgetList[i].hide()
        else:
            for i in range(nWidgets, nDimensions - 2):
                browser = FrameBrowser.HorizontalSliderWithBrowser(self.browserContainer)
                self.browserContainer.mainLayout.addWidget(browser)
                self._widgetList.append(browser)
                browser.valueChanged.connect(self.browserSlot)
                browser.setEnabled(False)
                browser.hide()
        for i in range(nWidgets):
            browser = self._widgetList[i]
            if (i + 2 ) < nDimensions:
                browser.setEnabled(True)
                if browser.isHidden():
                    browser.show()
                browser.setRange(1, self._array.shape[i])
            else:
                browser.setEnabled(False)
                browser.hide()
        self.view.setArrayData(self._array)

    def browserSlot(self, value):
        if len(self._array.shape) == 3:
            self.view.setCurrentArrayIndex(value - 1)
            self.view.reset()
        else:
            index = []
            for browser in self._widgetList:
                if browser.isEnabled():
                    index.append(browser.value() - 1)
            self.view.setCurrentArrayIndex(index)
            self.view.reset()

if __name__ == "__main__":
    import numpy
    import sys
    a = qt.QApplication([])
    d = numpy.random.normal(0,1, (4, 5, 1000,1000))
    for j in range(4):
        for i in range(5):
            d[j, i, :, :] += i + 10 * j 
    #m = NumpyArrayTableModel(numpy.arange(100.), fmt="%.5f")
    #m = NumpyArrayTableModel(numpy.ones((100,20)), fmt="%.5f")
    w = NumpyArrayTableWidget()
    if "2" in sys.argv:
        print("sending a single image")
        w.setArrayData(d[0,0])
    elif "3" in sys.argv:
        print("sending a 5 images ")
        w.setArrayData(d[0])
    else:
        print("sending a 4 * 5 images ")
        w.setArrayData(d)
    #m.setCurrentIndex(4)
    #m.setArrayData(numpy.ones((100,100)))
    w.show()
    a.exec()
