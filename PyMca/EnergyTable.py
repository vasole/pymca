#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
__revision__ = "$Revision: 1.10 $"
__author__="V.A. Sole - ESRF BLISS Group"
import sys
import Numeric
import QXTube
from   QXTube import qt as qt
if qt.qVersion() < '3.0.0':
    import Myqttable as qttable
elif qt.qVersion() < '4.0.0':
    import qttable

DEBUG=0
class EnergyTab(qt.QWidget):
    def __init__(self,parent=None, name="Energy Tab"):
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        hbox = qt.QWidget(self)
        self.hbox = qt.QHBoxLayout(hbox)
        self.tube = QXTube.QXTube(hbox)
        self.table  = EnergyTable(hbox)
        self.hbox.addWidget(self.tube)
        self.hbox.addWidget(self.table)
        self.tube.plot()
        self.tube.hide()
        self.tubeButton = qt.QPushButton(self)
        self.tubeButton.setText("I have an x-ray tube!")        
        layout.addWidget(self.tubeButton)
        layout.addWidget(hbox)
        self.connect(self.tubeButton,
                     qt.SIGNAL("clicked()"),
                     self.tubeButtonClicked)

        self.__calculating = 0
        if qt.qVersion() < '4.0.0':
            self.connect(self.tube, qt.PYSIGNAL("QXTubeSignal"), self.__tubeUpdated)
        else:
            self.connect(self.tube, qt.SIGNAL("QXTubeSignal"), self.__tubeUpdated)

    def tubeButtonClicked(self):
        if self.tube.isHidden():
            self.tube.show()
            self.tubeButton.setText("I don't have an x-ray tube!")
        else:
            self.tube.hide()
            self.tubeButton.setText("I have an x-ray tube!")

    def __tubeUpdated(self, d):
        if    self.__calculating:return
        else: self.__calculating = 1
        self.table.setParameters(d["energylist"],
                                 d["weightlist"],
                                 d["flaglist"],
                                 d["scatterlist"])
        self.__calculating = 0
        self.tubeButtonClicked()


if qt.qVersion() < '4.0.0':
    QTable = qttable.QTable
else:
    QTable = qt.QTableWidget

class EnergyTable(QTable):
    def __init__(self, parent=None, name="Energy Table",
                     energylist=None, weightlist=None, flaglist=None,offset=None,scatterlist=None):
        QTable.__init__(self, parent)
        if energylist is  None:energylist=[]
        if weightlist is  None:weightlist  =[]
        if flaglist   is  None:flaglist  =[]
        if scatterlist   is  None:scatterlist  =[]
        if offset is None:offset = 0
        self.energyList  = energylist
        self.weightList  = weightlist
        self.flagList    = flaglist
        self.offset      = offset
        self.scatterList = scatterlist
        self.verticalHeader().hide()
        self.dataColumns = 7
        if qt.qVersion() < '4.0.0':
            self.setLeftMargin(0)
            self.setFrameShape(qttable.QTable.NoFrame)
            #self.setFrameShadow(qttable.QTable.Sunken)
            self.setSelectionMode(qttable.QTable.Single)
            self.setNumCols(3 * self.dataColumns)
            if qt.qVersion() > '3.0.0':
                self.setFocusStyle(qttable.QTable.FollowStyle)
        else:
                if DEBUG:
                    print "margin"
                    print "frame shape"
                    print "selection mode"
                    print "focus style"
                    print "all of them missing"
                self.setColumnCount(3 * self.dataColumns)

        labels = []
        for i in range(self.dataColumns):
            labels.append("Use" + i * " ")
            labels.append("Energy" + i * " ")
            labels.append("Weight" + i * " ")
        if qt.qVersion() < '4.0.0':
            for label in labels:
                self.horizontalHeader().setLabel(labels.index(label),label)
        else:
            if DEBUG:
                print "margin to addjust"
                print "focus style"
            self.setFrameShape(qt.QTableWidget.NoFrame)
            self.setSelectionMode(qt.QTableWidget.NoSelection)
            self.setColumnCount(len(labels))
            for i in range(len(labels)):
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(labels[i],qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i,item)
                
        self.__rows = 20
        self.__build(self.dataColumns * 20)
        for i in range(self.dataColumns):
            if qt.qVersion() < '4.0.0':
                self.adjustColumn(0 + 3*i)
            else:
                if DEBUG:
                    print "column adjustment missing"
        if qt.qVersion() < '4.0.0':
            self.connect(self, qt.SIGNAL("valueChanged(int,int)"),self.mySlot)
        else:
            self.connect(self, qt.SIGNAL("cellChanged(int, int)"),self.mySlot)

    def _itemSlot(self, *var):
        self.mySlot(self.currentRow(), self.currentColumn())

    def __build(self,nrows=None):
        #self.setNumRows(int(nrows/2))
        if nrows is None: nrows = self.__rows *self.dataColumns
        if qt.qVersion() < '4.0.0':
            self.setNumRows(int(nrows/self.dataColumns))
        else:
            self.setRowCount(int(nrows/self.dataColumns))
        if qt.qVersion() > '4.0.0':
            rheight = self.horizontalHeader().sizeHint().height()
            for idx in range(self.rowCount()):
                self.setRowHeight(idx, rheight)
                
        coloffset = 0
        rowoffset = 0
        for idx in range(nrows):
            text = "Energy %3d" % (idx)
            if idx >= (nrows/self.dataColumns):
                rowoffset= (-int(idx/self.__rows))*(nrows/self.dataColumns)
                coloffset=  3*int(idx/self.__rows)
            r = idx + rowoffset
            color = qt.Qt.white
            if len(self.scatterList):
                if idx < len(self.scatterList):
                    if (self.scatterList[idx] is not None)and \
                       (self.scatterList[idx] != "None"):
                        if self.scatterList[idx]:color = qt.QColor(255, 20, 147)
            elif idx == 0:
                color = qt.QColor(255, 20, 147)
            if qt.qVersion() < '4.0.0':
                #item= qttable.QCheckTableItem(self, text)
                if qt.qVersion() < '3.0.0':
                    if DEBUG:
                        print "background  color to implement in qt 2.3.0"
                else:
                    self.viewport().setPaletteBackgroundColor(color)
                item= ColorQTableItem(self, text, color)                
                self.setItem(r, 0+coloffset, item)
            else:
                item= ColorQTableItem(self, text, color)
                self.setCellWidget(r, 0+coloffset, item)
                item.setText(text)
                self.connect(item, qt.SIGNAL("stateChanged(int)"),self._itemSlot)
            if idx < len(self.energyList):
                item.setChecked(self.flagList[idx])
                if (self.energyList[idx] is not None) and \
                   (self.energyList[idx] != "None"):
                    self.setText(r, 1+coloffset,
                                 "%f" % self.energyList[idx])
                else:
                    self.setText(r, 1+coloffset,"")
            else:
                self.setText(r, 1+coloffset,"")
            if idx < len(self.weightList):
                self.setText(r, 2+coloffset,"%f" % self.weightList[idx])
            else:
                self.setText(r, 2+coloffset,"")

    def setParameters(self, energylist, weightlist, flaglist, scatterlist=None):
        if type(energylist) == Numeric.ArrayType:self.energyList=energylist.tolist()
        elif type(energylist) != type([]):self.energyList=[energylist]
        else: self.energyList =energylist

        if   type(weightlist) == Numeric.ArrayType:self.weightList=weightlist.tolist()
        elif type(weightlist) != type([]):self.energyList=[weightlist]
        else: self.weightList =weightlist
        
        if   type(flaglist) == Numeric.ArrayType:self.flagList=flaglist.tolist()
        elif type(flaglist) != type([]):self.flagList=[flaglist]
        else: self.flagList =flaglist

        
        if scatterlist is None:
            scatterlist = Numeric.zeros(len(self.energyList)).tolist()
            scatterlist[0] = 1
        if type(scatterlist) == Numeric.ArrayType:self.scatterList=scatterlist.tolist()
        elif type(scatterlist) != type([]):self.scatterList=[scatterlist]
        else: self.scatterList =scatterlist
        self.__fillTable()

    def getParameters(self):
        if qt.qVersion() < '4.0.0':
            nrows = self.numRows()*self.dataColumns
        else:
            nrows = self.rowCount() * self.dataColumns
        coloffset   = 0
        rowoffset   = 0
        energyList  = []
        weightList  = []
        flagList    = []
        scatterList = []
        for idx in range(nrows):
            if idx >= (nrows/self.dataColumns):
                rowoffset= (-int(idx/self.__rows))*(nrows/self.dataColumns)
                coloffset=  3*int(idx/self.__rows)
            r = idx + rowoffset
            if qt.qVersion() < '4.0.0':
                item = self.item(r,0+coloffset)
                energyflag = int(item.isChecked())
            else:
                item = self.cellWidget(r,0+coloffset)
                energyflag = int(item.isChecked())
            if item.color == qt.Qt.white:
                scatterflag = 0
            else:
                scatterflag = 1
            text = str(self.text(r,1+coloffset))
            text=text.replace(" ","")
            if len(text):
                try:
                    energy = float(text)
                except:
                    energyflag = 0
                    energy = None
            else:
                energyflag = 0
                energy = None
            text = str(self.text(r,2+coloffset))
            text=text.replace(" ","")
            if len(text):
                try:
                    energyweight = float(text)
                except:
                    energyflag  = 0
                    energyweight= 0.0
            else:
                energyflag = 0
                energyweight = 0.0
            energyList.append(energy)
            weightList.append(energyweight)
            flagList.append(energyflag)
            scatterList.append(scatterflag)
        return energyList, weightList, flagList, scatterList

    def __fillTable(self):
        self.__build(max(self.__rows*self.dataColumns,len(self.energyList)))
        for i in range(self.dataColumns):
            if qt.qVersion() < '4.0.0':
                self.adjustColumn(0 + 3*i)
            else:
                if DEBUG:
                    print "column adjustment missing"

        ddict = self._getDict()
        if ddict != {}:
            ddict['event'] = "TableFilled"
            ddict['row']   = 0
            ddict['col']   = 0
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL("EnergyTableSignal"), (ddict,))
            else:
                self.emit(qt.SIGNAL("EnergyTableSignal"), (ddict))

    def mySlot(self,row,col):
        if DEBUG:
            print "Value changed row = ",row,"col = ",col
            print "Text = ", self.text(row,col)
        if (col != 0) and (col !=3) and (col != 6) and (col != 9):
            try:
                s = str(self.text(row, col))
                s=s.replace(" ","")
                if len(s):
                    v=float(s)
            except:
                msg = qt.QMessageBox(self)       
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Invalid Float")
                if qt.qVersion() < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return 
        ddict = self._getDict()
        if ddict != {}:
            ddict['event'] = "ValueChanged"
            ddict['row']   = row
            ddict['col']   = col
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL("EnergyTableSignal"),(ddict,))
            else:
                self.emit(qt.SIGNAL("EnergyTableSignal"),(ddict))
            
    def text(self, row, col):
        if qt.qVersion() < '4.0.0':
            return qttable.QTable.text(self, row, col)
        else:
            if (col % 3) in [1,2]:
                item = self.item(row , col)
                if item is not None:
                    return item.text()
                else:
                    return ''

    def setText(self, row, col, text):
        if qt.qVersion() < "4.0.0":
            QTable.setText(self, row, col, text)
        else:
            #ncol = self.columnCount()
            if (col % 3) in [1,2]:
                item = self.item(row, col)
                if item is None:
                    item = qt.QTableWidgetItem(text,
                                               qt.QTableWidgetItem.Type)
                else:
                    item.setText(text)
                self.setItem(row, col, item)
            else:
                if DEBUG:
                    print "checkbox can be called?"
                pass

    def _getDict(self):
        dict ={}
        if qt.qVersion() < '4.0.0':
            n = self.numRows()
        else:
            n = self.rowCount()
        dict['energy'] = []
        dict['rate']   = []
        dict['flag']   = []
        dict['scatterflag']   = []
        for i in range(n * self.dataColumns):
                if i >= (n*self.__rows/self.dataColumns):
                    rowoffset= (-int(i/self.__rows))*(self.__rows)
                    r = i + rowoffset
                    coffset=  3*int(i/self.__rows)
                else:
                    r = i
                    coffset= 0
                try:
                #if 1:
                    s = str(self.text(r, 1+coffset))
                    s=s.replace(" ","")
                    if len(s):
                        ene=float(s)
                        if qt.qVersion() < '4.0.0':
                            selfitem = self.item(r,0+coffset)
                        else:
                            selfitem = self.cellWidget(r, 0+coffset)
                        if selfitem.isChecked():
                            flag = 1
                        else:
                            flag = 0
                        if selfitem.color != qt.Qt.white:
                            scatterflag = 1
                        else:
                            scatterflag = 0
                        s = str(self.text(r, 2+coffset))
                        s=s.replace(" ","")
                        if len(s):
                            rate = float(s)
                            dict['flag'].append(flag)
                            dict['energy'].append(ene)
                            dict['rate'].append(rate)
                            dict['scatterflag'].append(scatterflag)
                except:
                #else:
                    msg = qt.QMessageBox(self)       
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("EnergyTable: Error on energy %d" % i)
                    if qt.qVersion() < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return {}
        return dict

if qt.qVersion() < '4.0.0':
    class ColorQTableItem(qttable.QCheckTableItem):
             def __init__(self, table, text,color=qt.Qt.white,bold=0):
                qttable.QCheckTableItem.__init__(self, table, text)
                self.color = color
                self.bold  = bold

             def paint(self, painter, colorgroup, rect, selected):
                painter.font().setBold(self.bold)
                cg = qt.QColorGroup()
                #colorgroup)
                cg.setColor(qt.QColorGroup.Base, self.color)
                cg.setColor(qt.QColorGroup.Foreground, self.color)
                cg.setColor(qt.QColorGroup.HighlightedText, self.color)
                qttable.QCheckTableItem.paint(self,painter, cg, rect, selected)
                painter.font().setBold(0)

else:
    class ColorQTableItem(qt.QCheckBox):
             def __init__(self, table, text, color=qt.Qt.white,bold=0):
                qt.QCheckBox.__init__(self, table)
                self.color = color
                self.bold  = bold
                self.setText(text)
                #this is the critical line
                self.setAutoFillBackground(1)

             def paintEvent(self, painter):
                palette = self.palette()
                role = self.backgroundRole()
                palette.setColor(role, self.color)
                self.setPalette(palette)
                return qt.QCheckBox.paintEvent(self, painter)
            
def main(args):
    app=qt.QApplication(args)
    #tab = AttenuatorsTableWidget(None)
    def dummy(ddict):
        print "dict =",ddict
    tab = EnergyTable(None)
    energy = Numeric.arange(100.).astype(Numeric.Float)+ 1.5
    weight = Numeric.ones(len(energy), Numeric.Float)
    flag  = Numeric.zeros(len(energy)).tolist()
    scatterlist = Numeric.zeros(len(energy))
    scatterlist[0:10] = 1
    tab.setParameters(energy, weight, flag, scatterlist)
    if qt.qVersion() < '4.0.0':
        qt.QObject.connect(tab,qt.PYSIGNAL('EnergyTableSignal'),dummy)
        tab.show()
        app.setMainWidget( tab )
        app.exec_loop()
    else:
        qt.QObject.connect(tab,qt.SIGNAL('EnergyTableSignal'),dummy)
        tab.show()
        app.exec_()

                            

if __name__=="__main__":
    main(sys.argv)	
