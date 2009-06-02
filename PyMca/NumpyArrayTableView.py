#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
from PyQt4 import QtCore, QtGui
import NumpyArrayTableModel

class HorizontalHeader(QtCore.QAbstractItemModel):
    def __init__(self, parent=None):
        QtGui.QHeaderView.__init__(self, parent)

    def columnCount(self, modelIndex):
        return self.parent().columnCount()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant("%d" % section)
        return QtCore.QVariant()

class VerticalHeader(QtCore.QAbstractItemModel):
    def __init__(self, parent=None):
        QtGui.QHeaderView.__init__(self, parent)

    def rowCount(self, modelIndex):
        return self.parent().rowCount()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant("%d" % section)
        return QtCore.QVariant()

class NumpyArrayTableView(QtGui.QTableView):
    def __init__(self, parent=None):
        QtGui.QTableView.__init__(self, parent)
        self._model = NumpyArrayTableModel.NumpyArrayTableModel(self)
        self.setModel(self._model)
        horizontalHeaderModel = HorizontalHeader(self._model)
        verticalHeaderModel = VerticalHeader(self._model)
        self.horizontalHeader().setModel(horizontalHeaderModel)
        self.verticalHeader().setModel(verticalHeaderModel)

    def setArrayData(self, data):
        self._model.setArrayData(data)

    def setCurrentArrayIndex(self, index):
        return self._model.setCurrentArrayIndex(index)

if __name__ == "__main__":
    import numpy
    a = QtGui.QApplication([])
    d = numpy.random.normal(0,1, (5, 1000,1000))
    for i in range(5):
        d[i, :, :] += i
    w = NumpyArrayTableView()
    w.setArrayData(d)
    w.show()
    a.exec_()
