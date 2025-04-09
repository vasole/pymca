#/*##########################################################################
# Copyright (C) 2024 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "M. Spitoni"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import h5py

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaCore.NexusTools import getStartingPositionerValues

from . import HDF5Info


class NexusMotorInfoWidget(qt.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self.label = qt.QLabel(self)
        self.label.setText("Number of motors: 0")

        column_names = ["Name", "Value", "Units"]
        self._column_names = column_names

        self.table = qt.QTableWidget(self)
        self.table.setColumnCount(len(column_names))
        for i in range(len(column_names)):
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(column_names[i], qt.QTableWidgetItem.Type)
            item.setText(column_names[i])
            self.table.setHorizontalHeaderItem(i, item)
        self.table.setSortingEnabled(True)

        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.table)

    def setInfoDict(self, ddict):
        if "motors" in ddict:
            self._setInfoDict(ddict["motors"])
        else:
            self._setInfoDict(ddict)

    def _setInfoDict(self, ddict):
        nrows = len(ddict.get(self._column_names[0], []))
        self.label.setText("Number of motors: %d" % nrows)
        self.table.setRowCount(nrows)

        if not nrows:
            self.hide()
            return

        for row in range(nrows):
            for col, label in enumerate(self._column_names):
                text = str(ddict[label][row])
                item = self.table.item(row, col)
                if item is None:
                    item = qt.QTableWidgetItem(text, qt.QTableWidgetItem.Type)
                    item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
                    self.table.setItem(row, col, item)
                else:
                    item.setText(text)

        for col in range(len(self._column_names)):
            self.table.resizeColumnToContents(col)


class NexusInfoWidget(HDF5Info.HDF5InfoWidget):

    def __init__(self, parent=None, info=None, nxclass=None):
        self._nxclass = nxclass
        super().__init__(parent=parent, info=info)

    def _build(self):
        super()._build()
        if self._nxclass in ("NXentry", b"NXentry"):
            self.motorInfoWidget = NexusMotorInfoWidget(self)
            self.addTab(self.motorInfoWidget, "Motors")

    def setInfoDict(self, ddict):
        super().setInfoDict(ddict)
        if self._nxclass in ("NXentry", b"NXentry"):
            self.motorInfoWidget.setInfoDict(ddict)


def getInfo(hdf5File, node):
    """
    hdf5File is and HDF5 file-like insance
    node is the posix path to the node
    """
    info = HDF5Info.getInfo(hdf5File, node)
    info["motors"] = get_motor_positions(hdf5File, node)
    return info


def get_motor_positions(hdf5File, node):
    node = hdf5File[node]

    nxentry_name = node.name.split("/")[1]
    if not nxentry_name:
        return dict()

    if hasattr(node, "file"):
        source_file = node.file
    else:
        source_file = hdf5File

    nxentry = source_file[nxentry_name]
    if not isinstance(nxentry, h5py.Group):
        return dict()

    positions = getStartingPositionerValues(source_file, nxentry_name)
    column_names = "Name", "Value", "Units"
    return dict(zip(column_names, zip(*positions)))
