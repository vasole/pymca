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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from PyMca5.PyMcaGui import PyMcaQt as qt
from . import FrameBrowser
from . import NumpyArrayTableView

class NumpyArrayTableWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QTableWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.browser = FrameBrowser.FrameBrowser(self)
        self.view = NumpyArrayTableView.NumpyArrayTableView(self)
        self.mainLayout.addWidget(self.browser)
        self.mainLayout.addWidget(self.view)
        self.browser.sigIndexChanged.connect(self.browserSlot)

    def setArrayData(self, data):
        self._array = data
        self.view.setArrayData(self._array)
        if len(self._array.shape) > 2:
            self.browser.setNFrames(self._array.shape[0])
        else:
            self.browser.setNFrames(1)

    def browserSlot(self, ddict):
        if ddict['event'] == "indexChanged":
            if len(self._array.shape) == 3:
                self.view.setCurrentArrayIndex(ddict['new']-1)
                self.view.reset()

if __name__ == "__main__":
    import numpy
    a = qt.QApplication([])
    d = numpy.random.normal(0,1, (5, 1000,1000))
    for i in range(5):
        d[i, :, :] += i
    #m = NumpyArrayTableModel(numpy.arange(100.), fmt="%.5f")
    #m = NumpyArrayTableModel(numpy.ones((100,20)), fmt="%.5f")
    w = NumpyArrayTableWidget()
    w.setArrayData(d)
    #m.setCurrentIndex(4)
    #m.setArrayData(numpy.ones((100,100)))
    w.show()
    a.exec_()
