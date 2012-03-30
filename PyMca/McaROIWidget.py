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
import sys
import os

from PyMca import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
    
QTVERSION = qt.qVersion()

from PyMca import PyMcaDirs
from PyMca import ConfigDict

if QTVERSION < '4.0.0':
    from PyMca import qttable
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount    = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
            self.resizeColumnToContents = self.adjustColumn
        
else:
    QTable = qt.QTableWidget

DEBUG = 0
class McaROIWidget(qt.QWidget):
    def __init__(self, parent=None, name=None, fl=0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self, parent, name,fl)
            if name is not None:self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            if name is not None:self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)
        ##############
        self.headerlabel = qt.QLabel(self)
        self.headerlabel.setAlignment(qt.Qt.AlignHCenter)       
        self.setheader('<b>Channel ROIs of XXXXXXXXXX<\b>')
        layout.addWidget(self.headerlabel)
        ##############
        self.mcaROITable     = McaROITable(self)
        if QTVERSION < '4.0.0':
            self.mcaROITable.setMinimumHeight(4*self.mcaROITable.sizeHint().height())
            self.mcaROITable.setMaximumHeight(4*self.mcaROITable.sizeHint().height())
        else:
            rheight = self.mcaROITable.horizontalHeader().sizeHint().height()
            self.mcaROITable.setMinimumHeight(4*rheight)
            #self.mcaROITable.setMaximumHeight(4*rheight)
        self.fillfromroidict = self.mcaROITable.fillfromroidict
        self.addroi          = self.mcaROITable.addroi
        self.getroilistanddict=self.mcaROITable.getroilistanddict
        layout.addWidget(self.mcaROITable)
        self.roiDir = None
        #################


        
        hbox = qt.QWidget(self)
        hboxlayout = qt.QHBoxLayout(hbox)
        hboxlayout.setMargin(0)
        hboxlayout.setSpacing(0)

        hboxlayout.addWidget(HorizontalSpacer(hbox))
        
        self.addbutton = qt.QPushButton(hbox)
        self.addbutton.setText("Add ROI")
        self.delbutton = qt.QPushButton(hbox)
        self.delbutton.setText("Delete ROI")        
        self.resetbutton = qt.QPushButton(hbox)
        self.resetbutton.setText("Reset")

        hboxlayout.addWidget(self.addbutton)
        hboxlayout.addWidget(self.delbutton)
        hboxlayout.addWidget(self.resetbutton)
        hboxlayout.addWidget(HorizontalSpacer(hbox))

        if QTVERSION > '4.0.0':
            self.loadButton = qt.QPushButton(hbox)
            self.loadButton.setText("Load")
            self.saveButton = qt.QPushButton(hbox)
            self.saveButton.setText("Save")
            hboxlayout.addWidget(self.loadButton)
            hboxlayout.addWidget(self.saveButton)
            layout.setStretchFactor(self.headerlabel, 0)
            layout.setStretchFactor(self.mcaROITable, 1)
            layout.setStretchFactor(hbox, 0)

        layout.addWidget(hbox)

        self.connect(self.addbutton,  qt.SIGNAL("clicked()"), self.__add)
        self.connect(self.delbutton,  qt.SIGNAL("clicked()"), self.__del)
        self.connect(self.resetbutton,qt.SIGNAL("clicked()"), self.__reset)

        if QTVERSION < '4.0.0':
            self.connect(self.mcaROITable,  qt.PYSIGNAL('McaROITableSignal') ,self.__forward)
        else:
            self.connect(self.loadButton,qt.SIGNAL("clicked()"), self._load)
            self.connect(self.saveButton,qt.SIGNAL("clicked()"), self._save)
            self.connect(self.mcaROITable,  qt.SIGNAL('McaROITableSignal') ,self.__forward)

    def __add(self):
        if DEBUG:
            print("McaROIWidget.__add")
        ddict={}
        ddict['event']   = "AddROI"
        roilist,roidict  = self.mcaROITable.getroilistanddict()
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'),ddict)
        
        
    def __del(self):
        row = self.mcaROITable.currentRow()
        if row >= 0:
            index = self.mcaROITable.labels.index('Type')
            if QTVERSION < '4.0.0':
                text = str(self.mcaROITable.text(row, index))
            else:
                text = str(self.mcaROITable.item(row, index).text())
                
            if text.upper() != 'DEFAULT':
                index = self.mcaROITable.labels.index('ROI')
                if QTVERSION < '4.0.0':
                    key = str(self.mcaROITable.text(row, index))
                else:
                    key = str(self.mcaROITable.item(row, index).text())
            else:return
            roilist,roidict    = self.mcaROITable.getroilistanddict()
            row = roilist.index(key)
            del roilist[row]
            del roidict[key]
            self.mcaROITable.fillfromroidict(roilist=roilist,
                                             roidict=roidict,
                                             currentroi=roilist[0])
            ddict={}
            ddict['event']      = "DelROI"
            ddict['roilist']    = roilist
            ddict['roidict']    = roidict
            #ddict['currentrow'] = self.mcaROITable.currentRow()
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,))
            else:
                self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict)
        
    
    def __forward(self,ddict):
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,)) 
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict) 
        
    
    def __reset(self):
        ddict={}
        ddict['event']   = "ResetROI"
        roilist0,roidict0  = self.mcaROITable.getroilistanddict()
        index = 0
        for key in roilist0:
            if roidict0[key]['type'].upper() == 'DEFAULT':
                index = roilist0.index(key)
                break
        roilist=[]
        roilist.append(roilist0[index])
        roidict = {}
        roidict[roilist[0]] = {}
        roidict[roilist[0]].update(roidict0[roilist[0]])
        self.mcaROITable.fillfromroidict(roilist=roilist,roidict=roidict)
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,))
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict)


    def _load(self):        
        if self.roiDir is None:
            self.roiDir = PyMcaDirs.inputDir
        elif not os.path.isdir(self.roiDir):
            self.roiDir = PyMcaDirs.inputDir
        outfile = qt.QFileDialog(self)
        outfile.setFilter('PyMca  *.ini')
        outfile.setFileMode(outfile.ExistingFile)
        outfile.setDirectory(self.roiDir)
        ret = outfile.exec_()
        if not ret:
            outfile.close()
            del outfile
            return
        outputFile = str(outfile.selectedFiles()[0])
        outfile.close()
        del outfile
        self.roiDir = os.path.dirname(outputFile)
        self.load(outputFile)

    def load(self, filename):
        d = ConfigDict.ConfigDict()
        d.read(filename)
        current = ""
        if self.mcaROITable.rowCount():
            row = self.mcaROITable.currentRow()
            item = self.mcaROITable.item(row, 0)
            if item is not None:
                current = str(item.text())
        self.fillfromroidict(roilist=d['ROI']['roilist'],
                             roidict=d['ROI']['roidict'])
        if current in d['ROI']['roidict'].keys():
            if current in d['ROI']['roilist']:
                row = d['ROI']['roilist'].index(current, 0)
                self.mcaROITable.setCurrentCell(row, 0)
                self.mcaROITable._cellChangedSlot(row, 2)
                return            
        self.mcaROITable.setCurrentCell(0, 0)
        self.mcaROITable._cellChangedSlot(0, 2)

    def _save(self):
        if self.roiDir is None:
            self.roiDir = PyMcaDirs.outputDir
        elif not os.path.isdir(self.roiDir):
            self.roiDir = PyMcaDirs.outputDir
        outfile = qt.QFileDialog(self)
        outfile.setFilter('PyMca  *.ini')
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(qt.QFileDialog.AcceptSave)
        outfile.setDirectory(self.roiDir)
        ret = outfile.exec_()
        if not ret:
            outfile.close()
            del outfile
            return
        outputFile = str(outfile.selectedFiles()[0])
        extension = ".ini"
        outfile.close()
        del outfile
        if len(outputFile) < len(extension[:]):
            outputFile += extension[:]
        elif outputFile[-4:] != extension[:]:
            outputFile += extension[:]
        if os.path.exists(outputFile):
            try:
                os.remove(outputFile)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_()
                return
        self.roiDir = os.path.dirname(outputFile)
        self.save(outputFile)
        
    def save(self, filename):
        d= ConfigDict.ConfigDict()
        d['ROI'] = {}
        d['ROI'] = {'roilist': self.mcaROITable.roilist * 1,
                    'roidict':{}}
        d['ROI']['roidict'].update(self.mcaROITable.roidict)
        d.write(filename)
        
    def setdata(self,*var,**kw):
        self.info ={}
        if 'legend' in kw:
            self.info['legend'] = kw['legend']
            del kw['legend']
        else:
            self.info['legend'] = 'Unknown Type'
        if 'xlabel' in kw:
            self.info['xlabel'] = kw['xlabel']
            del kw['xlabel']
        else:
            self.info['xlabel'] = 'X'
        if 'rois' in kw:
            rois = kw['rois']
            self.mcaROITable.fillfromrois(rois)
        self.setheader(text="%s ROIs of %s" % (self.info['xlabel'],
                                               self.info['legend']))

    def setheader(self,*var,**kw):
        if len(var):
            text = var[0]
        elif 'text' in kw:
            text = kw['text']
        elif 'header' in kw:
            text = kw['header']
        else:
            text = ""
        self.headerlabel.setText("<b>%s<\b>" % text)

class McaROITable(QTable):
    def __init__(self, *args,**kw):
        QTable.__init__(self, *args)
        self.setRowCount(1)
        self.labels=['ROI','Type','From','To','Raw Counts','Net Counts']
        self.setColumnCount(len(self.labels))
        i=0
        if QTVERSION < '4.0.0':
            if 'labels' in kw:
                for label in kw['labels']:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i = i + 1
            else:
                for label in self.labels:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i = i + 1
        else:
            if QTVERSION > '4.2.0':self.setSortingEnabled(False)
            if 'labels' in kw:
                for label in kw['labels']:
                    item = self.horizontalHeaderItem(i)
                    if item is None:
                        item = qt.QTableWidgetItem(label,
                                               qt.QTableWidgetItem.Type)
                    item.setText(label)
                    self.setHorizontalHeaderItem(i,item)
                    i = i + 1
            else:
                for label in self.labels:
                    item = self.horizontalHeaderItem(i)
                    if item is None:
                        item = qt.QTableWidgetItem(label,
                                               qt.QTableWidgetItem.Type)
                    item.setText(label)
                    self.setHorizontalHeaderItem(i,item)
                    i=i+1
                
        self.roidict={}
        self.roilist=[]
        if 'roilist' in kw:
            self.roilist = kw['roilist']
        if 'roidict' in kw:
            self.roidict.update(kw['roilist'])
        self.building = False
        self.build()
        #self.connect(self,qt.SIGNAL("currentChanged(int,int)"),self.myslot)
        if QTVERSION < '4.0.0':
            self.connect(self,qt.SIGNAL("valueChanged(int,int)"),self.nameSlot)
            self.connect(self,qt.SIGNAL("selectionChanged()"),self._myslot)
        else:
            self.connect(self,qt.SIGNAL("cellClicked(int, int)"),self._myslot)
            self.connect(self,qt.SIGNAL("cellChanged(int, int)"),self._cellChangedSlot)
            #self.connect(self,qt.SIGNAL("itemSelectionChanged()"),self._myslot)
        #self.connect(self,qt.SIGNAL("pressed(int,int,QPoint())"),self.myslot)
        if QTVERSION > '2.3.0':
            if QTVERSION < '4.0.0':
                self.setSelectionMode(QTable.SingleRow)
            else:
                if DEBUG:
                    print("Qt4 selection mode?")
                #self.setMinimumHeight(8*self.horizontalHeader().sizeHint().height())
                #self.setMaximumHeight(8*self.horizontalHeader().sizeHint().height())
    

    def build(self):
        self.fillfromroidict(roilist=self.roilist,roidict=self.roidict)
    
    def fillfromroidict(self,roilist=[],roidict={},currentroi=None):
        self.building = True
        line0  = 0
        self.roilist = []
        self.roidict = {}
        for key in roilist:
            if key in roidict.keys():
                roi = roidict[key]
                self.roilist.append(key)
                self.roidict[key] = {}
                self.roidict[key].update(roi)
                line0 = line0 + 1
                nlines=self.rowCount()
                if (line0 > nlines):
                    self.setRowCount(line0)
                line = line0 -1
                self.roidict[key]['line'] = line
                ROI = key
                roitype = QString("%s" % roi['type'])
                fromdata= QString("%6g" % (roi['from']))
                todata  = QString("%6g" % (roi['to']))
                if 'rawcounts' in roi:
                    rawcounts= QString("%6g" % (roi['rawcounts']))
                else:
                    rawcounts = " ?????? "
                if 'netcounts' in roi:
                    netcounts= QString("%6g" % (roi['netcounts']))
                else:
                    netcounts = " ?????? "
                fields  = [ROI,roitype,fromdata,todata,rawcounts,netcounts]
                col = 0 
                for field in fields:
                    if QTVERSION < '4.0.0':
                        if (ROI.upper() == 'ICR') or (ROI.upper() == 'DEFAULT'):
                            key2=qttable.QTableItem(self,qttable.QTableItem.Never,field)
                        else:
                            if col == 0:
                                key2=qttable.QTableItem(self,qttable.QTableItem.OnTyping,field)                        
                            else:
                                key2=qttable.QTableItem(self,qttable.QTableItem.Never,field)
                    else:
                        key2 = self.item(line, col)
                        if key2 is None:
                            key2 = qt.QTableWidgetItem(field,
                                                       qt.QTableWidgetItem.Type)
                            self.setItem(line,col,key2)
                        else:
                            key2.setText(field)
                        if (ROI.upper() == 'ICR') or (ROI.upper() == 'DEFAULT'):
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled) 
                        else:
                            if col in [0, 2, 3]:
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled|
                                              qt.Qt.ItemIsEditable)                        
                            else:
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled) 
                    col=col+1
        self.setRowCount(line0)
        i = 0
        for label in self.labels:
            self.resizeColumnToContents(i)
            i=i+1
        if QTVERSION < '4.0.0':
            self.sortColumn(2,1,1)
        else:
            self.sortByColumn(2,qt.Qt.AscendingOrder)
        for i in range(len(self.roilist)):
            if QTVERSION < '4.0.0':
                key = str(self.text(i, 0))
            else:
                key = str(self.item(i, 0).text())
                
            self.roilist[i] = key
            self.roidict[key]['line'] = i
        if len(self.roilist) == 1:
            if QTVERSION > '2.3.0':
                self.selectRow(0)
            else:
                if DEBUG:
                    print("Method not implemented, just first cell")
                self.setCurrentCell(0,0)
        else:
            if currentroi in self.roidict.keys():
                if QTVERSION < '3.0.0':
                    self.setCurrentCell(self.roidict[currentroi]['line'],0)
                else:
                    self.selectRow(self.roidict[currentroi]['line'])
                if QTVERSION < '4.0.0':
                    self.ensureCellVisible(self.roidict[currentroi]['line'],0)
                else:
                    if DEBUG:
                        print("Qt4 ensureCellVisible to be implemented")
        self.building = False                

    def addroi(self,roi,key=None):
        nlines=self.numRows()
        self.setNumRows(nlines+1)
        line = nlines
        if key is None:
            key = "%d " % line
        self.roidict[key] = {}
        self.roidict[key]['line'] = line
        self.roidict[key]['type'] = roi['type']
        self.roidict[key]['from'] = roi['from']
        self.roidict[key]['to']   = roi['to']
        ROI = key
        roitype = QString("%s" % roi['type'])
        fromdata= QString("%6g" % (roi['from']))
        todata  = QString("%6g" % (roi['to']))
        if 'rawcounts' in roi:
            rawcounts= QString("%6g" % (roi['rawcounts']))
        else:
            rawcounts = " ?????? "
        self.roidict[key]['rawcounts']   = rawcounts
        if 'netcounts' in roi:
            netcounts= QString("%6g" % (roi['netcounts']))
        else:
            netcounts = " ?????? "
        self.roidict[key]['netcounts']   = netcounts
        fields  = [ROI,roitype,fromdata,todata,rawcounts,netcounts]
        col = 0 
        for field in fields:
            if (ROI == 'ICR') or (ROI.upper() == 'DEFAULT'):
                key=qttable.QTableItem(self,qttable.QTableItem.Never,field)
            else:
                if col == 0:
                    key=qttable.QTableItem(self,qttable.QTableItem.OnTyping,field)                
                else:
                    key=qttable.QTableItem(self,qttable.QTableItem.Never,field)                
            self.setItem(line,col,key)
            col=col+1
        if QTVERSION < '4.0.0':
            self.sortColumn(2,1,1)
        else:
            self.sortByColumn(2, qt.Qt.AscendingOrder)
        for i in range(len(self.roilist)):
            nkey = str(self.text(i,0))
            self.roilist[i] = nkey
            self.roidict[nkey]['line'] = i
        self.selectRow(self.roidict[key]['line'])
        self.ensureCellVisible(self.roidict[key]['line'],0)

    def getroilistanddict(self):
        return self.roilist,self.roidict 

    def _myslot(self, *var, **kw):
        #selection changed event
        #get the current selection
        row = self.currentRow()
        col = self.currentColumn()
        if row >= 0:
            ddict = {}
            ddict['event'] = "selectionChanged"
            ddict['row'  ] = row
            ddict['col'  ] = col
            if row >= len(self.roilist):
                if DEBUG:
                    print("deleting???")
                return
                row = 0
            if QTVERSION < '4.0.0':
                text = str(self.text(row, 0))
            else:
                item = self.item(row, 0)
                if item is None:text=""
                else:text = str(item.text())
            self.roilist[row] = text
            ddict['roi'  ] = self.roidict[self.roilist[row]]
            ddict['key']   = self.roilist[row]
            ddict['colheader'] = self.labels[col]
            ddict['rowheader'] = "%d" % row
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaROITableSignal'), (ddict,))
            else:
                self.emit(qt.SIGNAL('McaROITableSignal'), ddict)

    def _cellChangedSlot(self, row, col):
        if DEBUG:
            print("_cellChangedSlot(%d, %d)" % (row, col))
        if  self.building:return
        if col == 0:
            self.nameSlot(row, col)
        else:
            self._valueChanged(row, col)

    def _valueChanged(self, row, col):
        if col not in [2, 3]: return
        item = self.item(row, col)
        if item is None:return
        text = str(item.text())
        try:
            value = float(text)
        except:
            return
        if row >= len(self.roilist):
            if DEBUG:
                print("deleting???")
            return
        if QTVERSION < '4.0.0':
            text = str(self.text(row, 0))
        else:
            item = self.item(row, 0)
            if item is None:text=""
            else:text = str(item.text())
        if not len(text):return
        if col == 2:
            self.roidict[text]['from'] = value
        elif col ==3:              
            self.roidict[text]['to'] = value
        ddict = {}
        ddict['event'] = "selectionChanged"
        ddict['row'  ] = row
        ddict['col'  ] = col
        ddict['roi'  ] = self.roidict[self.roilist[row]]
        ddict['key']   = self.roilist[row]
        ddict['colheader'] = self.labels[col]
        ddict['rowheader'] = "%d" % row
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROITableSignal'), (ddict,))
        else:
            self.emit(qt.SIGNAL('McaROITableSignal'), ddict)



    def nameSlot(self, row, col):
        if col != 0: return
        if row >= len(self.roilist):
            if DEBUG:
                print("deleting???")
            return
        if QTVERSION < '4.0.0':
            text = str(self.text(row, col))
        else:
            item = self.item(row, col)
            if item is None:text=""
            else:text = str(item.text())
        if len(text) and (text not in self.roilist):
            old = self.roilist[row]
            self.roilist[row] = text
            self.roidict[text] = {}
            self.roidict[text].update(self.roidict[old])
            del self.roidict[old]
            ddict = {}
            ddict['event'] = "selectionChanged"
            ddict['row'  ] = row
            ddict['col'  ] = col
            ddict['roi'  ] = self.roidict[self.roilist[row]]
            ddict['key']   = self.roilist[row]
            ddict['colheader'] = self.labels[col]
            ddict['rowheader'] = "%d" % row
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaROITableSignal'), (ddict,))
            else:
                self.emit(qt.SIGNAL('McaROITableSignal'), ddict)

    def myslot(self,*var,**kw):
        if len(var) == 0:
            self._myslot()
            return
        if len(var) == 2:
            ddict={}
            row = var[0]
            col = var[1]
            if col == 0:
                if row >= len(self.roilist):
                    if DEBUG:
                        print("deleting???")
                    return
                    row = 0
                if QTVERSION < '4.0.0':
                    text = str(self.text(row, col))
                else:
                    item = self.item(row, col)
                    if item is None:text=""
                    else:text = str(item.text())
                if len(text) and (text not in self.roilist):
                    old = self.roilist[row]
                    self.roilist[row] = text
                    self.roidict[text] = {}
                    self.roidict[text].update(self.roidict[old])
                    del self.roidict[old]
                    ddict = {}
                    ddict['event'] = "selectionChanged"
                    ddict['row'  ] = row
                    ddict['col'  ] = col
                    ddict['roi'  ] = self.roidict[self.roilist[row]]
                    ddict['key']   = self.roilist[row]
                    ddict['colheader'] = self.labels[col]
                    ddict['rowheader'] = "%d" % row
                    if QTVERSION < '4.0.0':
                        self.emit(qt.PYSIGNAL('McaROITableSignal'), (ddict,))
                    else:
                        self.emit(qt.SIGNAL('McaROITableSignal'), ddict)
                else:
                    if QTVERSION < '4.0.0':
                        self.setText(row, col, self.roilist[row])
                    else:
                        if item is None:
                            item = qt.QTableWidgetItem(text,
                                       qt.QTableWidgetItem.Type)
                        else:
                            item.setText(text)
                    self._myslot()

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)

        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                             qt.QSizePolicy.Fixed))

class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)

        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                            qt.QSizePolicy.Expanding))

class SimpleComboBox(qt.QComboBox):
        def __init__(self,parent = None,name = None,fl = 0,options=['1','2','3']):
            qt.QComboBox.__init__(self,parent)
            self.setoptions(options) 
            
        def setoptions(self,options=['1','2','3']):
            self.clear()    
            self.insertStrList(options)
            
        def getcurrent(self):
            return   self.currentItem(),str(self.currentText())
             
if __name__ == '__main__':
    app = qt.QApplication([])
    demo = McaROIWidget()
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        app.exec_loop()
    else:
        demo.show()
        app.exec_()

