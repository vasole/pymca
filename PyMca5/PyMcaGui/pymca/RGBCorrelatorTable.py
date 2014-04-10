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
__author__ = "V.A. Sole - ESRF Software Group"
from PyMca5 import PyMcaQt as qt
DEBUG = 0

class RGBCorrelatorTable(qt.QTableWidget):
    def __init__(self, parent=None):
        qt.QTableWidget.__init__(self, parent)
        self.elementList      = []
        self.rSelection   = []
        self.gSelection   = []
        self.bSelection = []
        labels = ['Element', 'R', 'G', 'B', 'Data Min', "Data Max"]
        self.setColumnCount(len(labels))
        for i in range(len(labels)):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(labels[i],
                                           qt.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.setHorizontalHeaderItem(i,item)
        rheight = self.horizontalHeader().sizeHint().height()
        self.setMinimumHeight(5*rheight)
        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
        self.resizeColumnToContents(3)

        """
        qt.QObject.connect(self,
                     qt.SIGNAL("cellClicked(int, int)"),
                     self._mySlot)
        """

    def build(self, elementlist):
        self.elementList = elementlist
        n = len(elementlist)
        self.setRowCount(n)
        if n > 0:
            rheight = self.horizontalHeader().sizeHint().height()
            for i in range(n):
                self.setRowHeight(i, rheight)
                self._addLine(i, elementlist[i])

        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
        self.resizeColumnToContents(3)
        
    def _addLine(self, i, cntlabel):
        #the counter name
        item = self.item(i, 0)
        if item is None:
            item = qt.QTableWidgetItem(cntlabel,
                                       qt.QTableWidgetItem.Type)
            item.setTextAlignment(qt.Qt.AlignHCenter | qt.Qt.AlignVCenter)
            self.setItem(i, 0, item)
        else:
            item.setText(cntlabel)
        #item is enabled and selectable
        item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable)

        #the checkboxes
        for j in range(1, 4, 1):
            widget = self.cellWidget(i, j)
            if widget is None:
                widget = CheckBoxItem(self, i, j)
                self.setCellWidget(i, j, widget)
                qt.QObject.connect(widget,
                                   qt.SIGNAL('CheckBoxItemSignal'),
                                   self._mySlot)
            else:
                pass

    def _mySlot(self, ddict):
        row = ddict["row"]
        col = ddict["col"]
        if col == 1:
            if ddict["state"]:
                if row not in self.rSelection:
                    self.rSelection.append(row)
            else:
                if row in self.rSelection:
                    del self.rSelection[self.rSelection.index(row)]
            if len(self.rSelection) > 1:
                self.rSelection = self.rSelection[-1:]
        if col == 2:
            if ddict["state"]:
                if row not in self.gSelection:
                    self.gSelection.append(row)
            else:
                if row in self.gSelection:
                    del self.gSelection[self.gSelection.index(row)]
            if len(self.gSelection) > 1:
                self.gSelection = self.gSelection[-1:]
        if col == 3:
            if ddict["state"]:
                if row not in self.bSelection:
                    self.bSelection.append(row)
            else:
                if row in self.bSelection:
                    del self.bSelection[self.bSelection.index(row)]
            if len(self.bSelection) > 1:
                self.bSelection = self.bSelection[-1:]
        #I should check if there is a change ...
        self._update()

    def _emitSignal(self):
        ddict = self.getElementSelection()
        ddict['event'] = "updated"
        self.emit(qt.SIGNAL('RGBCorrelatorTableSignal'),
                  ddict)

    def _update(self):
        for i in range(self.rowCount()):
            j = 1
            widget = self.cellWidget(i, j)
            if i in self.rSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
            j = 2
            widget = self.cellWidget(i, j)
            if i in self.gSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
            j = 3
            widget = self.cellWidget(i, j)
            if i in self.bSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
        self._emitSignal()        

    def getElementSelection(self):
        ddict = {}
        ddict['elementlist'] = self.elementList * 1
        n = len(self.elementList)
        if n == 0:
            self.rSelection = []
            self.gSelection = []
            self.bSelection = []
        if len(self.rSelection):
            if self.rSelection[0] >= n:
                self.rSelection = []
        if len(self.gSelection):
            if self.gSelection[0] >= n:
                self.gSelection = []
        if len(self.bSelection):
            if self.bSelection[0] >= n:
                self.bSelection = []
        ddict['r'] = self.rSelection * 1
        ddict['g'] = self.gSelection * 1
        ddict['b'] = self.bSelection * 1
        return ddict

    def setElementSelection(self, ddict):
        keys = ddict.keys()
        if 'elementlist' in keys: elementlist = ddict['elementlist']
        else: elementlist = self.elementList * 1

        if 'r' in keys: x = ddict['r']
        else: x = []

        if 'g' in keys: y = ddict['g']
        else: y = []

        if 'b' in keys: monitor = ddict['b']
        else: monitor = []

        self.rSelection = []
        for item in x:
            if item < len(elementlist):
                counter = elementlist[item]
                if 0:
                    if counter in self.elementList:
                        self.rSelection.append(self.elementList.index(counter))
                else:
                    self.rSelection.append(item)

        self.gSelection = []
        for item in y:
            if item < len(elementlist):
                counter = elementlist[item]
                if counter in self.elementList:
                    self.gSelection.append(self.elementList.index(counter))
        self.bSelection = []
        for item in monitor:
            if item < len(elementlist):
                counter = elementlist[item]
                if counter in self.elementList:
                    self.bSelection.append(self.elementList.index(counter))
        self._update()
        

class CheckBoxItem(qt.QCheckBox):
    def __init__(self, parent, row, col):
        qt.QCheckBox.__init__(self, parent)
        self.__row = row
        self.__col = col
        qt.QObject.connect(self, qt.SIGNAL("clicked(bool)"), self._mySignal)

    def _mySignal(self, value):
        ddict = {}
        ddict["event"] = "clicked"
        ddict["state"] = value
        ddict["row"] = self.__row * 1
        ddict["col"] = self.__col * 1
        self.emit(qt.SIGNAL('CheckBoxItemSignal'), ddict)

def main():
    app = qt.QApplication([])
    def slot(ddict):
        print("received dict = ", ddict)
    tab = RGBCorrelatorTable()
    app.connect(tab,
                qt.SIGNAL('RGBCorrelatorTableSignal'),
                slot)

    tab.build(["Cnt1", "Cnt2", "Cnt3"])
    tab.setElementSelection({'r':[1], 'g':[4], 'elementlist':["dummy", "Ca K", "Fe K", "Pb M", "U l"]})
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main()
    
