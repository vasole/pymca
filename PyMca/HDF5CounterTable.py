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
import sys
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
DEBUG = 0

if 'Object3D' in sys.modules:
    OBJECT3D = True
else:
    OBJECT3D = False

class HDF5CounterTable(QtGui.QTableWidget):
    def __init__(self, parent=None):
        QtGui.QTableWidget.__init__(self, parent)
        self.cntList      = []
        self.mcaList      = []
        self.xSelection   = []
        self.ySelection   = []
        self.monSelection = []
        self.__is3DEnabled = False
        labels = ['Counter', 'X    ', 'Y    ', 'Mon', 'Alias']
        self.setColumnCount(len(labels))
        for i in range(len(labels)):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = QtGui.QTableWidgetItem(labels[i],
                                           QtGui.QTableWidgetItem.Type)
            item.setText(labels[i])
            self.setHorizontalHeaderItem(i,item)

        """
        #the cell is not the same as the check box
        #but I wonder about the checkboxes being destroyed
        """
        QtCore.QObject.connect(self,
                     QtCore.SIGNAL("cellChanged(int, int)"),
                     self._aliasSlot)

    def build(self, cntlist, aliaslist):
        self.__building = True
        nmca = 0
        if len(cntlist) != len(aliaslist):
            raise ValueError, "Alias list and counter list must have same length"
        self.cntList = cntlist
        self.aliasList = aliaslist
        n = len(cntlist)
        self.setRowCount(n)
        if n > 0:
            self.setRowCount(n)
            rheight = self.horizontalHeader().sizeHint().height()
            for i in range(n):
                self.setRowHeight(i, rheight)
                self.__addLine(i, cntlist[i])
                for j in range(1, 4, 1):
                    widget = self.cellWidget(i, j)
                    widget.setEnabled(True)
        else:
            self.setRowCount(0)

        self.resizeColumnToContents(1)
        self.resizeColumnToContents(2)
        self.resizeColumnToContents(3)
        self.__building = False
        
    def __addLine(self, i, cntlabel):
        #the counter name
        item = self.item(i, 0)
        if item is None:
            item = QtGui.QTableWidgetItem(cntlabel,
                                       QtGui.QTableWidgetItem.Type)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.setItem(i, 0, item)
        else:
            item.setText(cntlabel)

        #item is just enabled (not selectable)
        item.setFlags(QtCore.Qt.ItemIsEnabled)

        #the checkboxes
        for j in range(1, 4, 1):
            widget = self.cellWidget(i, j)
            if widget is None:
                widget = CheckBoxItem(self, i, j)
                self.setCellWidget(i, j, widget)
                QtCore.QObject.connect(widget,
                                   QtCore.SIGNAL('CheckBoxItemSignal'),
                                   self._mySlot)
            else:
                pass

        #the alias
        item = self.item(i, 4)
        alias = self.aliasList[i]
        if item is None:
            item = QtGui.QTableWidgetItem(alias,
                                       QtGui.QTableWidgetItem.Type)
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.setItem(i, 4, item)
        else:
            item.setText(alias)

    def set3DEnabled(self, value):
        if value:
            self.__is3DEnabled = True
        else:
            if len(self.xSelection) > 1:
                self.xSelection = [1 * self.xSelection[0]]
        self._update()

    def _aliasSlot(self, row, col):
        if self.__building:
            return
        if col != 4:
            return
        item = self.item(row, 4)
        self.aliasList[row] = str(item.text())

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
            if (not OBJECT3D) or (not self.__is3DEnabled):
                if len(self.xSelection) > 2:
                    #that is to support mesh plots
                    self.xSelection = self.xSelection[-2:]
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

    def _update(self):
        for i in range(self.rowCount()):
            j = 1
            widget = self.cellWidget(i, j)
            if i in self.xSelection:
                if not widget.isChecked():
                    widget.setChecked(True)
            else:
                if widget.isChecked():
                    widget.setChecked(False)
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
        ddict = {}
        ddict["event"] = "updated"
        self.emit(QtCore.SIGNAL('HDF5CounterTableSignal'), ddict)        
        

    def getCounterSelection(self):
        ddict = {}
        ddict['cntlist'] = self.cntList * 1
        ddict['aliaslist'] = self.aliasList * 1
        ddict['x']       = self.xSelection * 1
        ddict['y']       = self.ySelection * 1
        ddict['m'] = self.monSelection * 1        
        return ddict

    def setCounterSelection(self, ddict):
        keys = ddict.keys()
        if 'cntlist' in keys:
            cntlist = ddict['cntlist']
        else:
            cntlist = self.cntList * 1

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
                if 0:
                    if counter in self.cntList:
                        self.xSelection.append(self.cntList.index(counter))
                else:
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
        

class CheckBoxItem(QtGui.QCheckBox):
    def __init__(self, parent, row, col):
        QtGui.QCheckBox.__init__(self, parent)
        self.__row = row
        self.__col = col
        QtCore.QObject.connect(self, QtCore.SIGNAL("clicked(bool)"), self._mySignal)

    def _mySignal(self, value):
        ddict = {}
        ddict["event"] = "clicked"
        ddict["state"] = value
        ddict["row"] = self.__row * 1
        ddict["col"] = self.__col * 1
        self.emit(QtCore.SIGNAL('CheckBoxItemSignal'), ddict)

def main():
    app = QtGui.QApplication([])
    tab = HDF5CounterTable()
    tab.build(["Cnt1", "Cnt2", "Cnt3"])
    tab.setCounterSelection({'x':[1, 2], 'y':[4], 'cntlist':["dummy", "Cnt0", "Cnt1", "Cnt2", "Cnt3"]})
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main()
    
