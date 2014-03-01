#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
from PyQt4 import QtCore, QtGui
from PyMca import FrameBrowser
from PyMca import NumpyArrayTableView

class NumpyArrayTableWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QTableWidget.__init__(self, parent)
        self.mainLayout = QtGui.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.browser = FrameBrowser.FrameBrowser(self)
        self.view = NumpyArrayTableView.NumpyArrayTableView(self)
        self.mainLayout.addWidget(self.browser)
        self.mainLayout.addWidget(self.view)
        self.connect(self.browser,
                     QtCore.SIGNAL("indexChanged"),
                     self.browserSlot)                     

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
    a = QtGui.QApplication([])
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
