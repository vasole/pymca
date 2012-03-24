#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
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
from PyMca import PyMcaQt as qt

QTVERSION = qt.qVersion()

DEBUG = 0

if QTVERSION < '4.0.0':
    if qt.qVersion() < '3.0.0':
        from PyMca import Myqttable as qttable
    else:
        from PyMca import qttable

    class SpecFileMcaTable(qt.QWidget):
        def __init__(self, parent=None, name=None, fl=0):
            qt.QWidget.__init__(self, parent, name, fl)

            self.mainLayout= qt.QVBoxLayout(self)

            self.table= qttable.QTable(self)
            self.table.setSelectionMode(qttable.QTable.Multi)
            if qt.qVersion() >= '3.0.0':
                self.table.setFocusStyle(qttable.QTable.FollowStyle)
                self.table.setReadOnly(1)
            else:
                if DEBUG:
                    print("Methods to be implemented")
            self.table.verticalHeader().setResizeEnabled(0, -1)
            self.table.horizontalHeader().setResizeEnabled(0, -1)

            # --- selection in case of a single MCA by scan
            self.firstMca= qttable.QTableSelection()
            self.firstMca.init(0, 0)
            self.firstMca.expandTo(0, 0)

            self.mainLayout.addWidget(self.table)

            #self.connect(self.table, qt.SIGNAL("selectionChanged()"), self.__selectionChanged)
            # XXX: do not work correctly anymore !!! (after qt.Qt upgrade)
            self.connect(self.table, qt.SIGNAL("doubleClicked(int,int,int,const QPoint&)"), self.__doubleClicked)
            self.reset()

        #
        # data management
        #
        def setData(self, specfiledata):
            self.data= specfiledata
            self.reset()

        def setDataSource(self, specfiledata):
            self.data= specfiledata
            self.reset()

        def setScan(self, scankey):
            if type(scankey)==type([]):
                if len(scankey): scankey= scankey[0]
                else: scankey=None
            if scankey is None:
                self.reset()
            else:
                self.info= self.data.GetSourceInfo(scankey)
                self.refresh()

        def markMcaSelected(self, mcalist):
            if DEBUG:
                print("markMcaSelected(self, mcalist) called")
                print("mcalist = ",mcalist)
            scankey = ""
            if mcalist != []:
                if len(mcalist):
                    stringsplit=string.split(mcalist[0],".")
                    if len(stringsplit) == 2:
                        scankey = ""
                    else:
                        scankey = stringsplit[0]+"."+stringsplit[1]+"."
                for row in range(self.pts):
                    for col in range(self.det):
                        if ("%s%d.%d" % (scankey,row+1, col+1)) in mcalist:
                            self.table.setText(row, col, "X")
                        else:
                            self.table.setText(row, col, "")
            else:
                for row in range(self.pts):
                    for col in range(self.det):
                       self.table.setText(row, col, "") 
                
        def selectMcaList(self, mcalist):
            for mcano in mcalist:
                self.selectMcaNo(mcano)

        def selectMcaNo(self, mcano):
            (row, col)= self.__getRowCol(mcano)
            selection= qttable.QTableSelection()
            selection.init(row, col)
            selection.expandTo(row, col)
            self.table.addSelection(selection)
            #self.__selectionChanged()

        def selectFirstMca(self):
            self.table.addSelection(self.firstMca)
            #self.__selectionChanged()

        def refresh(self):
            mca= self.info["NbMca"] or 0
            pts= self.info["Lines"] or 0

            if mca==0:
                self.reset()
            else:
                # --- try to compute number of detectors
                if pts>0 and mca%pts==0:
                    self.det= mca/pts
                    self.pts= pts
                else:
                    self.det= mca
                    self.pts= 1

                self.table.setNumCols(self.det)
                self.table.setNumRows(self.pts)
                Hheader= self.table.horizontalHeader()
                for idx in range(self.det):
                    Hheader.setLabel(idx, "mca.%d"%(idx+1), -1)
                    self.table.adjustColumn(idx)

                for row in range(self.pts):
                    for col in range(self.det):
                        self.table.setText(row, col, "")

                #if self.info["Lines"]==pts:
                #    print "AddNorm", self.info["AllLabels"]

                if mca==1:
                    self.selectFirstMca()

        def clear(self):
            self.table.clearSelection(1)

        def reset(self):
            self.table.clearSelection(1)
            self.table.setNumCols(1)
            self.table.setNumRows(0)
            self.table.horizontalHeader().setLabel(0, "NO MCA for the selected scan", -1)
            self.table.adjustColumn(0)
            self.det= 0
            self.pts= 0

        def getSelection(self):
            if self.det==0:    return []
            selection= []
            for row in range(self.pts):
                for col in range(self.det):
                    if self.table.isSelected(row, col):
                        #selection.append("%s.%d"%(self.info["Key"], self.__getMcaNo(row,col)))
                        selection.append("%s.%d.%d"%(self.info["Key"], row+1,col+1))
                    elif self.table.text(row, col) != "":
                        selection.append("%s.%d.%d"%(self.info["Key"], row+1,col+1))

            return selection

        def getCurrentlySelectedMca(self):
            if self.det==0:    return []
            selection= []
            for row in range(self.pts):
                for col in range(self.det):
                    if self.table.isSelected(row, col):
                        #selection.append("%s.%d"%(self.info["Key"], self.__getMcaNo(row,col)))
                        selection.append("%d.%d" %(row+1,col+1))
            return selection

        def getSelectedMca(self):
            if self.det==0:    return []
            selection= []
            for row in range(self.pts):
                for col in range(self.det):
                    if self.table.isSelected(row, col):
                        #selection.append("%s.%d"%(self.info["Key"], self.__getMcaNo(row,col)))
                        selection.append("%d.%d" %(row+1,col+1))
                    elif self.table.text(row, col) != "":
                        selection.append("%d.%d"%(row+1,col+1))
            return selection

        #
        # signal / slot handling
        #
        def __selectionChanged(self):
            if self.det==0: return
            else:
                self.emit(qt.PYSIGNAL("mcaSelection"), (self.getSelection,))

        def __doubleClicked(self, *args):
            #self.emit(qt.PYSIGNAL("mcaDoubleClicked"), (["%s.%d"%(self.info["Key"], self.__getMcaNo(args[0], args[1]))],))
            self.emit(qt.PYSIGNAL("mcaDoubleClicked"), (["%s.%d.%d"%(self.info["Key"], args[0]+1, args[1]+1)],))

        #
        # (mcano) <--> (row,col) operations
        #
        def __getMcaNo(self, row, col):
            return (row*self.det+col+1)

        def __getRowCol(self, mcano):
            mcano= mcano-1
            row= int(mcano/self.det)
            col= mcano-(row*self.det)
            return (row, col)


else:
    class SpecFileMcaTable(qt.QWidget):
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

            #self.connect(self.table,
            #             qt.SIGNAL("cellActivated( int, int)"),
            #             self._cellActivated)
            self.connect(self.table,
                         qt.SIGNAL("cellClicked(int, int)"),
                         self._cellClicked)
            self.connect(self.table,
                         qt.SIGNAL("cellDoubleClicked(int, int)"),
                         self._cellDoubleClicked)
            
            self.table._hHeader = self.table.horizontalHeader()
            self.connect(self.table._hHeader,
                     qt.SIGNAL('sectionClicked(int)'),
                     self._horizontalHeaderClicked)
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
            self.emit(qt.SIGNAL("McaDeviceSelected"), ddict)

        def build(self, info):
            if info['NbMca'] > 0:
                ncol = int(info['NbMcaDet'])
            else:
                ncol = 1
            nrow = info['NbMca']/ncol
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
                    self.table.setItemSelected(item, True)

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
            if DEBUG:
                print("_cellClicked %d %d " % (row, col))
            item = self.table.item(row, col)
            if item is None:
                item = qt.QTableWidgetItem('',qt.QTableWidgetItem.Type)
                self.table.setItem(row, col, item)

        def _cellDoubleClicked(self, row, col):
            if DEBUG:
                print("_cellDoubleClicked %d %d" % (row, col))
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
    from PyMca import SpecFileLayer
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
    app.exec_()

if __name__ == "__main__":
    test()
