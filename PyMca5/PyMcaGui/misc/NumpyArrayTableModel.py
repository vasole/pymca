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
import numpy
from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, 'QStringList'):
    MyQVariant = qt.QVariant
else:
    def MyQVariant(x=None):
        return x

class NumpyArrayTableModel(qt.QAbstractTableModel):
    def __init__(self, parent=None, narray=None, fmt="%g", perspective=0):
        qt.QAbstractTableModel.__init__(self, parent)
        if narray is None:
            narray = numpy.array([])
        self._array  = narray
        self._format = fmt
        self._index  = 0
        self.assignDataFunction(perspective)

    def rowCount(self, parent=None):
        return self._rowCount(parent)

    def columnCount(self, parent=None):
        return self._columnCount(parent)

    def data(self, index, role=qt.Qt.DisplayRole):
        return self._data(index, role)

    def _rowCount1D(self, parent=None):
        return 1

    def _columnCount1D(self, parent=None):
        return self._array.shape[0]

    def _data1D(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                # row = 0
                col = index.column()
                return MyQVariant(self._format % self._array[col])
        return MyQVariant()

    def _rowCount2D(self, parent=None):
        return self._array.shape[0]

    def _columnCount2D(self, parent=None):
        return self._array.shape[1]

    def _data2D(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return MyQVariant(self._format % self._array[row, col])
        return MyQVariant()

        
    def _rowCount3DIndex0(self, parent=None):
        return self._array.shape[1]

    def _columnCount3DIndex0(self, parent=None):
        return self._array.shape[2]            

    def _rowCount3DIndex1(self, parent=None):
        return self._array.shape[0]

    def _columnCount3DIndex1(self, parent=None):
        return self._array.shape[2]            

    def _rowCount3DIndex2(self, parent=None):
        return self._array.shape[0]

    def _columnCount3DIndex2(self, parent=None):
        return self._array.shape[1]            

    def _data3DIndex0(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return MyQVariant(self._format % self._array[self._index,
                                                                  row,
                                                                  col])
        return MyQVariant()

    def _data3DIndex1(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return MyQVariant(self._format % self._array[row,
                                                                  self._index,
                                                                  col])
        return MyQVariant()

    def _data3DIndex2(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return MyQVariant(self._format % self._array[row,
                                                                  col,
                                                                  self._index])
        return MyQVariant()

    def setArrayData(self, data, perspective=0):
        """
        setStackData(self, data, perspective=0)
        data is a 3D array
        dimension is the array dimension acting as index of images
        """
        self.reset()
        self._array = data
        self.assignDataFunction(perspective)
        self._index = 0
        
    def assignDataFunction(self, dimension):
        shape = self._array.shape
        if len(shape) == 2:
            self._rowCount = self._rowCount2D
            self._columnCount = self._columnCount2D
            self._data = self._data2D
        elif len(shape) == 1:
            self._rowCount = self._rowCount1D
            self._columnCount = self._columnCount1D
            self._data = self._data1D
        else:
            if dimension == 1:
                self._rowCount = self._rowCount3DIndex1
                self._columnCount = self._columnCount3DIndex1
                self._data = self._data3DIndex1
            elif dimension == 2:
                self._rowCount = self._rowCount3DIndex2
                self._columnCount = self._columnCount3DIndex2
                self._data = self._data3DIndex1
            else:
                self._rowCount = self._rowCount3DIndex0
                self._columnCount = self._columnCount3DIndex0
                self._data = self._data3DIndex0
            self._dimension = dimension

    def setCurrentArrayIndex(self, index):
        """
        This method is ignored if the current array does not
        not a 3-dimensional array.
        """
        shape = self._array.shape
        if len(shape) != 3:
            return
        shape = self._array.shape[self._dimension]
        if (index < 0) or (index >= shape):
            raise ValueError("Index must be an integer lower than %d" % shape)
        self._index = index

    def setFormat(self, fmt):
        self._format = fmt

if __name__ == "__main__":
    a = qt.QApplication([])
    w = qt.QTableView()
    d = numpy.random.normal(0,1, (5, 1000,1000))
    for i in range(5):
        d[i, :, :] += i
    #m = NumpyArrayTableModel(fmt="%.5f")
    #m = NumpyArrayTableModel(None, numpy.arange(100.), fmt="%.5f")
    #m = NumpyArrayTableModel(None, numpy.ones((100,20)), fmt="%.5f")
    m = NumpyArrayTableModel(None, d, fmt = "%.5f")
    w.setModel(m)
    m.setCurrentArrayIndex(4)
    #m.setArrayData(numpy.ones((100,)))
    w.show()
    a.exec_()
