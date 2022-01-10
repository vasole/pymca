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
__author__ = "V.A. Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import posixpath
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
safe_str = qt.safe_str

_logger = logging.getLogger(__name__)


class HDF5CounterTable(qt.QTableWidget):

    sigHDF5CounterTableSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QTableWidget.__init__(self, parent)
        self.cntList      = []
        self.aliasList    = []
        self.mcaList      = []
        self.xSelection   = []
        self.ySelection   = []
        self.monSelection = []
        self.__oldSelection = self.getCounterSelection()
        self.__is3DEnabled = False
        self.__is2DEnabled = False
        labels = ['Dataset', 'Axes', 'Signals', 'Monitor', 'Alias']
        self.setColumnCount(len(labels))
        for i in range(len(labels)):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(labels[i],
                                           qt.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.setHorizontalHeaderItem(i,item)

        """
        #the cell is not the same as the check box
        #but I wonder about the checkboxes being destroyed
        """
        self.cellChanged[int, int].connect(self._aliasSlot)

    def build(self, cntlist, aliaslist=None, selection=None):
        _logger.debug("build cntlist = %s aliaslist = %s" % (cntlist, aliaslist))
        self.__building = True
        if selection is None:
            if len(cntlist):
                if len(self.cntList):
                    self.__oldSelection = self.getCounterSelection()
        else:
            _logger.info("received selection %s", selection)
            self.__oldSelection = selection
        if aliaslist is None:
            aliaslist = []
            for item in cntlist:
                aliaslist.append(posixpath.basename(item))
        if len(cntlist) != len(aliaslist):
            raise ValueError("Alias list and counter list must have same length")
        self.cntList = cntlist
        self.aliasList = aliaslist
        n = len(cntlist)
        self.setRowCount(n)
        if n > 0:
            self.setRowCount(n)
            rheight = self.horizontalHeader().sizeHint().height()
            # check if we need the complete description
            useFullPath = []
            for i in range(n):
                iName = posixpath.basename(cntlist[i])
                for j in range(i+1, n):
                    if posixpath.basename(cntlist[j]) == iName:
                        if i not in useFullPath:
                            useFullPath.append(i)
                        if j not in useFullPath:
                            useFullPath.append(j)
            for i in range(n):
                self.setRowHeight(i, rheight)
                if i in useFullPath:
                    self.__addLine(i, cntlist[i])
                else:
                    self.__addLine(i, posixpath.basename(cntlist[i]))
                for j in range(1, 4, 1):
                    widget = self.cellWidget(i, j)
                    widget.setEnabled(True)
        else:
            self.setRowCount(0)

        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
        self.resizeColumnToContents(3)
        self.setCounterSelection(self.__oldSelection)
        self.__building = False

    def __addLine(self, i, cntlabel):
        #the counter name
        item = self.item(i, 0)
        if item is None:
            item = qt.QTableWidgetItem(cntlabel,
                                       qt.QTableWidgetItem.Type)
            item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
            self.setItem(i, 0, item)
        else:
            item.setText(cntlabel)

        #item is just enabled (not selectable)
        item.setFlags(qt.Qt.ItemIsEnabled)

        #the checkboxes
        for j in range(1, 4, 1):
            widget = self.cellWidget(i, j)
            if widget is None:
                widget = CheckBoxItem(self, i, j)
                self.setCellWidget(i, j, widget)
                widget.sigCheckBoxItemSignal.connect(self._mySlot)
            else:
                pass

        #the alias
        item = self.item(i, 4)
        alias = self.aliasList[i]
        if item is None:
            item = qt.QTableWidgetItem(alias,
                                       qt.QTableWidgetItem.Type)
            item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
            self.setItem(i, 4, item)
        else:
            item.setText(alias)

    def set3DEnabled(self, value, emit=True):
        if value:
            self.__is3DEnabled = True
            self.__is2DEnabled = True
        else:
            self.__is3DEnabled = False
            if len(self.xSelection) > 1:
                self.xSelection = self.xSelection[-1:]
        self._update(emit=emit)

    def set2DEnabled(self, value, emit=True):
        if value:
            self.__is2DEnabled = True
            self.__is3DEnabled = False
            if len(self.xSelection) > 2:
                self.xSelection = self.xSelection[-2:]
        else:
            self.__is2DEnabled = False
            if len(self.xSelection) > 1:
                self.xSelection = self.xSelection[-1:]
        self._update(emit=emit)

    def _aliasSlot(self, row, col):
        if self.__building:
            return
        if col != 4:
            return
        item = self.item(row, 4)
        self.aliasList[row] = safe_str(item.text())

    def _mySlot(self, ddict):
        row = ddict["row"]
        col = ddict["col"]
        if col == 1:
            if ddict["state"]:
                if row not in self.xSelection:
                    self.xSelection.append(row)
            else:
                if row in self.xSelection:
                    del self.xSelection[self.xSelection.index(row)]
            if self.__is3DEnabled:
                if len(self.xSelection) > 3:
                    self.xSelection = self.xSelection[-3:]
            elif self.__is2DEnabled:
                if len(self.xSelection) > 2:
                    self.xSelection = self.xSelection[-2:]
            else:
                if len(self.xSelection) > 1:
                    self.xSelection = self.xSelection[-1:]
        if col == 2:
            if ddict["state"]:
                if row not in self.ySelection:
                    self.ySelection.append(row)
            else:
                if row in self.ySelection:
                    del self.ySelection[self.ySelection.index(row)]
        if col == 3:
            if ddict["state"]:
                if row not in self.monSelection:
                    self.monSelection.append(row)
            else:
                if row in self.monSelection:
                    del self.monSelection[self.monSelection.index(row)]
            if len(self.monSelection) > 1:
                self.monSelection = self.monSelection[-1:]
        self._update()

    def _update(self, emit=True):
        _logger.debug("_update called with emit = %s" % emit)
        axisLabels = ['X', 'Y', 'Z']
        for i in range(self.rowCount()):
            j = 1
            widget = self.cellWidget(i, j)
            if i in self.xSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
                widget.setText(axisLabels[self.xSelection.index(i)])
            else:
                if widget.isChecked():
                    widget.setChecked(False)
                widget.setText("")
            j = 2
            widget = self.cellWidget(i, j)
            if i in self.ySelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
            j = 3
            widget = self.cellWidget(i, j)
            if i in self.monSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
        if emit:
            ddict = {}
            ddict["event"] = "updated"
            self.sigHDF5CounterTableSignal.emit(ddict)

    def getCounterSelection(self):
        ddict = {}
        ddict['cntlist'] = self.cntList * 1
        ddict['aliaslist'] = self.aliasList * 1
        ddict['x'] = self.xSelection * 1
        ddict['y'] = self.ySelection * 1
        ddict['m'] = self.monSelection * 1
        return ddict

    def setCounterSelection(self, ddict):
        _logger.debug("HDF5CounterTable.setCounterSelection %s", ddict)
        keys = ddict.keys()
        if 'cntlist' in keys:
            cntlist = ddict['cntlist']
        else:
            cntlist = self.cntList * 1

        # no selection based on aliaslist or counterlist (yet?)
        if 0:
            if 'aliaslist' in keys:
                aliaslist = ddict['aliaslist']
            elif len(self.aliasList) == len(cntlist):
                aliaslist = self.aliasList * 1
            else:
                aliaslist = self.cntList * 1

        if 'x' in keys:
            x = ddict['x']
        else:
            x = []

        if 'y' in keys:
            y = ddict['y']
        else:
            y = []

        if 'm' in keys:
            monitor = ddict['m']
        else:
            monitor = []

        self.xSelection = []
        for item in x:
            if item < len(cntlist):
                counter = cntlist[item]
                if counter in self.cntList:
                    # counter name based selection
                    self.xSelection.append(self.cntList.index(counter))
                elif item < len(self.cntList):
                    # index based selection
                    self.xSelection.append(item)

        self.ySelection = []
        for item in y:
            if item < len(cntlist):
                counter = cntlist[item]
                if counter in self.cntList:
                    self.ySelection.append(self.cntList.index(counter))

        self.monSelection = []
        for item in monitor:
            if item < len(cntlist):
                counter = cntlist[item]
                if counter in self.cntList:
                    self.monSelection.append(self.cntList.index(counter))
        self._update()

class CheckBoxItem(qt.QCheckBox):

    sigCheckBoxItemSignal = qt.pyqtSignal(object)

    def __init__(self, parent, row, col):
        qt.QCheckBox.__init__(self, parent)
        self.__row = row
        self.__col = col
        self.clicked[bool].connect(self._mySignal)

    def _mySignal(self, value=None):
        ddict = {}
        ddict["event"] = "clicked"
        if value is None:
            value = self.isChecked()
        ddict["state"] = value
        ddict["row"] = self.__row * 1
        ddict["col"] = self.__col * 1
        self.sigCheckBoxItemSignal.emit(ddict)

def main():
    app = qt.QApplication([])
    tab = HDF5CounterTable()
    tab.build(["Cnt1", "Cnt2", "Cnt3"])
    tab.setCounterSelection({'x':[1, 2], 'y':[4],
                        'cntlist':["dummy", "Cnt0", "Cnt1", "Cnt2", "Cnt3"]})
    tab.show()
    app.exec()

if __name__ == "__main__":
    main()

