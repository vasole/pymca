#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
import PyMcaQt as qt
QTVERSION = qt.qVersion()
DEBUG = 0

if 'Object3D' in sys.modules:
    OBJECT3D = True
else:
    OBJECT3D = False

if QTVERSION < '4.0.0':
    if QTVERSION < '3.0.0':
        import Myqttable as qttable
    else:
        import qttable
    
    class SpecFileCntTable(qt.QWidget):
        def __init__(self, parent=None, name=None, fl=0):
            qt.QWidget.__init__(self, parent, name, fl)

            self.mainLayout= qt.QVBoxLayout(self)

            self.table= qttable.QTable(self)
            self.table.setSelectionMode(qttable.QTable.Multi)
            if qt.qVersion() > '3.0.0':
                self.table.setFocusStyle(qttable.QTable.FollowStyle)
                #self.table.setFocusStyle(qttable.QTable.SpreadSheet)
                #self.table.setReadOnly(1)
                self.table.setColumnReadOnly(0,1)
                self.table.setColumnReadOnly(2,1)
            self.table.verticalHeader().setResizeEnabled(0, -1)
            self.table.horizontalHeader().setResizeEnabled(0, -1)
            self.cnt=0
            self.cntlist=[]

            # --- selection in case of a single MCA by scan
            self.firstCnt= qttable.QTableSelection()
            self.firstCnt.init(0, 0)
            self.firstCnt.expandTo(0, 0)

            self.mainLayout.addWidget(self.table)

            #do not use double click
            #self.connect(self.table, qt.SIGNAL("doubleClicked(int,int,int,const qt.QPoint&)"), self.__doubleClicked)
            self.connect(self.table,qt.SIGNAL("clicked(int,int,int,const QPoint&)"),
                                    self.__Clicked)
            self.connect(self.table,qt.SIGNAL("selectionChanged()"),
                                     self.__selectionChanged)
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
            if type(scankey)==types.ListType: 
                if len(scankey): scankey= scankey[0]
                else: scankey=None
            if scankey is None:
                self.reset()
            else:
                self.info= self.data.GetSourceInfo(scankey)
                self.refresh()

        def markCntSelected(self, cntdict):
            if DEBUG:
                print("markCntSelected(self, cntdict)")
                print("cntdict = ", cntdict)
                print("self.cntlist = ",self.cntlist)
            if (cntdict != {}):
                for cnt in self.cntlist:
                    row =self.cntlist.index(cnt)
                    if row in cntdict["x"]:
                        self.table.xcheck[row].setChecked(1)
                    else:
                        self.table.xcheck[row].setChecked(0)
                    if row in cntdict["y"]:
                        self.table.ycheck[row].setChecked(1)
                    else:
                        self.table.ycheck[row].setChecked(0)
                    if row in cntdict["m"]:
                        self.table.mcheck[row].setChecked(1)
                    else:
                        self.table.mcheck[row].setChecked(0)
                
        def selectCntList(self, cntlist, numbers = True):
            if DEBUG:
                print("selectCntList(self, cntlist)")
                print("cntlist = ",cntlist)
            if not numbers:
                for cntname in cntlist:
                    self.selectCntName(cntname)

        def selectCntName(self, cntname):
            (row, col)= self.__getRowCol(cntname)
            selection= qttable.QTableSelection()
            selection.init(row, col)
            selection.expandTo(row, col)
            self.table.addSelection(selection)
            #self.__selectionChanged()

        def selectFirstCnt(self):
            self.table.addSelection(self.firstCnt)
            #self.__selectionChanged()

        def refresh(self):
            #self.cntlist= self.info["AllLabels"] or []
            self.cntlist= self.info["LabelNames"] or []
            pts= self.info["Lines"] or 0

            if self.cntlist==[]: 
                self.reset()
            else:
                # --- try to compute number of counters
                self.cnt=len(self.cntlist)
                self.table.setNumCols(4)
                self.table.setNumRows(self.cnt)
                Hheader= self.table.horizontalHeader()
                Hheader.setLabel(0, "Label", -1)
                Hheader.setLabel(1, "X", -1)
                self.table.adjustColumn(1)
                Hheader.setLabel(2, "Y", -1)
                self.table.adjustColumn(2)
                Hheader.setLabel(3, "M", -1)
                self.table.adjustColumn(3)
                #for idx in range(4):
                #    Hheader.setLabel(idx, "mca.%d"%(idx+1), -1)
                self.table.xcheck=[]
                self.table.ycheck=[]
                self.table.mcheck=[]
                for row in range(self.cnt):
                    for col in range(4):
                        if col==1:
                            #self.table.setText(row, col, "")
                            self.table.xcheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                            self.table.setItem(row,col,self.table.xcheck[row])
                        elif col == 2:
                            #self.table.setText(row, col, "")
                            self.table.ycheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                            self.table.setItem(row,col,self.table.ycheck[row])
                        elif col == 3:
                            self.table.mcheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                            self.table.setItem(row,col,self.table.mcheck[row])
                        else:
                            self.table.setText(row, col, self.cntlist[row])
                self.table.adjustColumn(0)

                #if mca==1:
                #    self.selectFirstMca()

        def clear(self):
            self.table.clearSelection(1)

        def reset(self):
            self.table.clearSelection(1)
            self.table.setNumCols(1)
            self.table.setNumRows(0)
            self.table.horizontalHeader().setLabel(0, "NO Counters for the selected scan", -1)
            self.table.adjustColumn(0)
            self.cnt= 0
            self.pts= 0

        def getSelection(self, numbers=True):
            if self.cnt==0:    return []
            selection= []
            Xcnt=[]
            Ycnt=[]
            Mcnt=[]
            for row in range(self.cnt):
                #if self.table.isSelected(row, 1):
                if self.table.xcheck[row].isChecked():
                    #Xnt
                    if numbers:
                        Xcnt.append(row)
                    else:
                        Xcnt.append(self.cntlist[row])
                if self.table.ycheck[row].isChecked():
                    #Ycnt
                    if numbers:
                        Ycnt.append(row)
                    else:
                        Ycnt.append(self.cntlist[row])
                #if self.table.isSelected(row, 3):
                if self.table.mcheck[row].isChecked():
                    #Mcnt
                    if numbers:
                        Mcnt.append(row)
                    else:
                        Mcnt.append(self.cntlist[row])

            for cnt in Ycnt:
                selection.append({"Key":"%s"%(self.info["Key"]),
                                  "scan":"%s"%(self.info["Key"]),
                                  "x":Xcnt,
                                  "y":cnt,
                                  "m":Mcnt,
                                  "cntlist":self.cntlist*1})
            return selection

        def getCounterSelection(self, numbers=True):
            ddict = {}
            ddict['cntlist'] = 1 * self.cntlist
            ddict['x']       = []
            ddict['y']       = []
            ddict['m'] = []
            for row in range(self.cnt):
                if self.table.xcheck[row].isChecked():
                    #Xnt
                    if numbers:
                        ddict['x'].append(row)
                    else:
                        ddict['x'].append(self.cntlist[row])
                if self.table.ycheck[row].isChecked():
                    #Ycnt
                    if numbers:
                        ddict['y'].append(row)
                    else:
                        ddict['y'].append(self.cntlist[row])
                #if self.table.isSelected(row, 3):
                if self.table.mcheck[row].isChecked():
                    #Mcnt
                    if numbers:
                        ddict['m'].append(row)
                    else:
                        ddict['m'].append(self.cntlist[row])
                
            return ddict

            
        #
        # signal / slot handling
        #
        def __selectionChanged(self):
            if self.cnt==0: return
            else:
                for row in range(self.cnt):
                    if self.table.isSelected(row, 2):
                        self.table.ycheck[row].setChecked(1)
                    else:
                        self.table.ycheck[row].setChecked(0)
                self.emit(qt.PYSIGNAL("cntSelection"), (self.getSelection,))

        def __doubleClicked(self, *args):
            self.emit(qt.PYSIGNAL("cntDoubleClicked"), (["%s.%d"%(self.info["Key"], self.__getCntNo(args[0], args[1]))],))


        def __Clicked(self, *args):
            row=args[0]
            col=args[1]
            if col == 0:
                pass
            elif col==1:
                #xclick
                maxclicked=1
                for i in range(self.cnt):
                    if i != row:
                        self.table.xcheck[i].setChecked(0)                
            elif col==2:
                #yclick
                pass
            elif col==3:
                #mclixk
                for i in range(self.cnt):
                    if i != row:
                        self.table.mcheck[i].setChecked(0) 
        #
        # (mcano) <--> (row,col) operations
        #
        def __getCntNo(self, row, col):
            return (row*self.cnt+col+1)
            
        def __getRowCol(self, mcano):
            mcano= mcano-1
            row= int(mcano/self.cnt)
            col= mcano-(row*self.cnt)
            return (row, col)

else:
    class SpecFileCntTable(qt.QTableWidget):
        def __init__(self, parent=None):
            qt.QTableWidget.__init__(self, parent)
            self.cntList      = []
            self.mcaList      = []
            self.xSelection   = []
            self.ySelection   = []
            self.monSelection = []
            self.__is3DEnabled = False
            labels = ['Counter', 'X    ', 'Y    ', 'Mon']
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
            qt.QObject.connect(self,
                         qt.SIGNAL("cellClicked(int, int)"),
                         self._mySlot)
            """

        def build(self, cntlist, nmca=None):
            if not OBJECT3D:
                nmca = 0
            if nmca is None:
                nmca = 0 
            self.cntList = cntlist
            self.mcaList = []
            n = len(cntlist)
            self.setRowCount(n)
            if n > 0:
                self.setRowCount(n + nmca)
                rheight = self.horizontalHeader().sizeHint().height()
                for i in range(n):
                    self.setRowHeight(i, rheight)
                    self.__addLine(i, cntlist[i])
                    for j in range(1, 4, 1):
                        widget = self.cellWidget(i, j)
                        widget.setEnabled(True)
                for j in range(nmca):
                    row = n+j
                    self.setRowHeight(n+j, rheight)
                    mca = "Mca %d" % (j+1)
                    self.mcaList.append(mca)
                    self.__addLine(n+j, self.mcaList[j])
                    #the x checkbox
                    widget = self.cellWidget(row, 1)
                    widget.setChecked(False)
                    widget.setEnabled(False)
                    #the y checkbox
                    widget = self.cellWidget(row, 2)
                    widget.setChecked(False)
                    widget.setEnabled(True)
                    #the Monitor checkbox
                    widget = self.cellWidget(row, 3)
                    widget.setChecked(False)
                    widget.setEnabled(False)
            else:
                self.setRowCount(0)

            self.resizeColumnToContents(1)
            self.resizeColumnToContents(2)
            self.resizeColumnToContents(3)
            
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
                    qt.QObject.connect(widget,
                                       qt.SIGNAL('CheckBoxItemSignal'),
                                       self._mySlot)
                else:
                    pass

        def set3DEnabled(self, value):
            if value:
                self.__is3DEnabled = True
                if len(self.xSelection) > 3:
                    self.xSelection = self.xSelection[-3:]
            else:
                self.__is3DEnabled = False
                if len(self.xSelection) > 1:
                    self.xSelection = [1 * self.xSelection[0]]                    
            self._update()

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
                elif len(self.xSelection) > 3:
                    self.xSelection = self.xSelection[-3:]

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
            self.emit(qt.SIGNAL('SpecCntTableSignal'), ddict)        
            

        def getCounterSelection(self):
            ddict = {}
            ddict['cntlist'] = self.cntList * 1
            ddict['mcalist'] = self.mcaList * 1
            ddict['x']       = self.xSelection * 1
            ddict['y']       = self.ySelection * 1
            ddict['m'] = self.monSelection * 1        
            return ddict

        def setCounterSelection(self, ddict):
            keys = ddict.keys()
            if 'cntlist' in keys: cntlist = ddict['cntlist']
            else: cntlist = self.cntList * 1

            if 'x' in keys: x = ddict['x']
            else: x = []

            if 'y' in keys: y = ddict['y']
            else: y = []

            if 'm' in keys: monitor = ddict['m']
            else: monitor = []

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
    tab = SpecFileCntTable()
    tab.build(["Cnt1", "Cnt2", "Cnt3"])
    tab.setCounterSelection({'x':[1, 2], 'y':[4], 'cntlist':["dummy", "Cnt0", "Cnt1", "Cnt2", "Cnt3"]})
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main()
    
