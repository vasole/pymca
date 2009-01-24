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
import qt
import qttable
DEBUG = 0
class QTable(qttable.QTable):
    def setItem(self,row,col,item):
        try:
            qttable.QTable.setItem(self,row,col,item)
        except:
            self.setCellWidget(row,col,item)

class QTableItem(qttable.QTableItem):
    pass

class QTableSelection(qttable.QTableSelection):
    pass

class QComboTableItem(qttable.QTableItem):
    def __init__(self, table,list):
        qttable.QTableItem.__init__(self,table,qttable.QTableItem.Always,"")
        self.setReplaceable(0)
        self.table=table
        self.list=list
        self.createEditor()
        if len(list):
            self.setCurrentItem(self.list[0])

    def setCurrentItem(self,cur):
        if (type(cur) == type("")) or (type(cur) == type(qt.QString(""))):
            if DEBUG:
                print "String type",cur
            for i in range (len(self.list)):
                if str(cur)==str(self.list[i]):
                    self.cb.setCurrentItem(i)
                    return
            if DEBUG:
                print "string",cur,"not in options list" 
        else:
            if DEBUG:
                print "int type"
            if cur in range(len(self.list)):
                self.cb.setCurrentItem(cur)
                


    def createEditor(self):
        self.cb=qt.QComboBox(self.table.viewport())
        self.cb.insertStringList(self.list)
        self.cb.mySlot = self.mySlot
        self.cb.connect(self.cb,qt.SIGNAL("activated(int)"),self.mySlot)
        return self.cb

    def mySlot (self,index):
        if DEBUG:
            print "passing mySlot(index), index = ",index
        self.table.setText(self.row(),self.col(),self.currentText())
        self.table.emit(qt.SIGNAL("valueChanged(int,int)"),(self.row(),self.col()))        

    def currentItem(self):
        return self.cb.currentItem()

    def currentText(self):
        return self.cb.currentText()

    def setStringList(self,slist):
        self.list = slist
        self.cb.clear()
        self.cb.insertStringList(slist)

    def setEditable(self,option):
        self.cb.setEditable(option)
    
class QCheckTableItem(qttable.QTableItem):
    def __init__(self, table,txt):
        qttable.QTableItem.__init__(self,table,qttable.QTableItem.Always,txt)
        self.setReplaceable(0)
        self.table=table
        self.createEditor()
        self.setText(txt)

    def createEditor(self,text=qt.QString()):
        wid = qt.QHBox(self.table.viewport())
        self.cb=qt.QCheckBox(wid)
        #self.cb.setText(self.f)
        self.cb.mySlot = self.mySlot
        self.cb.connect(self.cb,qt.SIGNAL("stateChanged(int)"),self.mySlot)
        return self.cb

    def mySlot (self,value):
        if DEBUG:
            print "passing mySlot(index), value = ",value
        self.table.emit(qt.SIGNAL("valueChanged(int,int)"),(self.row(),self.col()))
        
    def isChecked(self):
        return self.cb.isChecked()

    def setChecked(self,value):
        return self.cb.setChecked(value)

    def setText(self,text):
        self.cb.setText(qt.QString(text))

    def text(self):
        return self.cb.text()

class ColorQTableItem(qttable.QTableItem):
         def __init__(self, table, edittype, text,color=qt.Qt.white,bold=None,font=None):
                 if font is not None:self.setFont(font)
                 if color is None:color = qt.Qt.white
                 if bold is None: bold=0
                 self.color = color
                 self.bold  = bold
                 qttable.QTableItem.__init__(self, table, edittype, text)

         def paint(self, painter, colorgroup, rect, selected):
            painter.font().setBold(self.bold)
            cg = qt.QColorGroup(colorgroup)
            cg.setColor(qt.QColorGroup.Base, self.color)
            qttable.QTableItem.paint(self,painter, cg, rect, selected)
            painter.font().setBold(0)

