#/*##########################################################################
# Copyright (C) 2004-2020 European Synchrotron Radiation Facility
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
__author__ = "E. Papillon, V.A. Sole - ESRF Software Group"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt

QTVERSION = qt.qVersion()

_logger = logging.getLogger(__name__)


class SpecFileMcaTable(qt.QWidget):
    sigMcaDeviceSelected = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.l = qt.QVBoxLayout(self)
        self.table= qt.QTableWidget(self)
        self.table.setColumnCount(1)
        self.table.setRowCount(0)

        item = self.table.horizontalHeaderItem(0)
        if item is None:
            item = qt.QTableWidgetItem("No MCA for the selected scan",
                                           qt.QTableWidgetItem.Type)

        self.table.setHorizontalHeaderItem(0,item)
        self.table.resizeColumnToContents(0)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(qt.QAbstractItemView.MultiSelection)
        self.l.addWidget(self.table)

        #self.table.cellActivated[int, int].connect(self._cellActivated)
        self.table.cellClicked[int, int].connect(self._cellClicked)
        self.table.cellDoubleClicked[int, int].connect(self._cellDoubleClicked)

        self.table._hHeader = self.table.horizontalHeader()
        self.table._hHeader.sectionClicked[int].connect(self._horizontalHeaderClicked)
        self.table._hHeader.menu = qt.QMenu()
        self.table._hHeader.menu.addAction('ADD Image')
        self.table._hHeader.menu.addAction('REMOVE Image')
        self.table._hHeader.menu.addAction('REPLACE Image')
        self.table._hHeader.menu.addAction('ADD Stack')

    def _horizontalHeaderClicked(self, value):
        if value < 0:
            return
        item = self.table.horizontalHeaderItem(value)
        text = str(item.text())
        if text.startswith("No MCA for"):
            return
        action = self.table._hHeader.menu.exec_(self.cursor().pos())
        if action is None:
            return
        txt = str(action.text())
        ddict = {}
        ddict['event'] = 'McaDeviceSelected'
        ddict['mca']   = value
        ddict['action'] = txt
        self.sigMcaDeviceSelected.emit(ddict)

    def build(self, info):
        if info['NbMca'] > 0:
            ncol = int(info['NbMcaDet'])
        else:
            ncol = 1
        nrow = info['NbMca'] // ncol
        self.table.setColumnCount(ncol)
        self.table.setRowCount(nrow)
        if nrow == 0:
            item = self.table.horizontalHeaderItem(0)
            item.setText("No MCA for the selected scan")
            self.table.resizeColumnToContents(0)
            return

        for c in range(ncol):
            text = "Mca %d" % (c+1)
            item = self.table.horizontalHeaderItem(c)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.table.setHorizontalHeaderItem(c,item)
            else:
                item.setText(text)
            self.table.resizeColumnToContents(c)
        if nrow == 1:
            if ncol == 1:
                item = self.table.item(0, 0)
                if item is None:
                    item = qt.QTableWidgetItem('',
                            qt.QTableWidgetItem.Type)
                    self.table.setItem(0, 0, item)
                item.setSelected(True)

    def _toggleCell(self, row, col):
        item = self.table.item(row, col)
        if item is None:
            item = qt.QTableWidgetItem('X',
                qt.QTableWidgetItem.Type)
            self.table.setItem(row, col, item)
            return
        text = str(item.text())
        if text == "X":
            item.setText("")
        else:
            item.setText("X")

    def _cellClicked(self, row, col):
        _logger.debug("_cellClicked %d %d ", row, col)
        item = self.table.item(row, col)
        if item is None:
            item = qt.QTableWidgetItem('',qt.QTableWidgetItem.Type)
            self.table.setItem(row, col, item)

    def _cellDoubleClicked(self, row, col):
        _logger.debug("_cellDoubleClicked %d %d", (row, col))
        #self._toggleCell(row, col)
        pass

    def getCurrentlySelectedMca(self):
        mca = []
        for item in self.table.selectedItems():
            row = self.table.row(item)
            col = self.table.column(item)
            mca.append("%d.%d" % (row+1, col+1))
        return mca

    def getSelectedMca(self):
        mca = self.getCurrentlySelectedMca() # They may be not X marked
        for r in range(self.table.rowCount()):
            for c in range(self.table.ColumnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    text = str(item.text)
                    if text == "X":
                        new = "%d.%d" % (r+1, c+1)
                        if new not in mca:
                            mca.append(new)
        return mca

    def setSelectedMca(self, mcalist):
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                new = "%d.%d" % (r+1, c+1)
                if item is not None:
                    if new not in mcalist:
                        item.setText("")
                    else:
                        item.setText("X")
                else:
                    if new in mcalist:
                        self._toggleCell(r, c)

def test():
    import sys
    from PyMca5.PyMcaCore import SpecFileLayer
    app = qt.QApplication([])
    tab = SpecFileMcaTable()
    d = SpecFileLayer.SpecFileLayer()
    if len(sys.argv) > 1:
        d.SetSource(sys.argv[1])
    else:
        d.SetSource('03novs060sum.mca')
    info, data = d.LoadSource('1.1')
    tab.build(info)
    tab.setSelectedMca(["1.1"])
    tab.show()
    app.exec()

if __name__ == "__main__":
    test()
