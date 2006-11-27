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
import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
    except:
        import qt
else:
    import qt
    
QTVERSION = qt.qVersion()

if QTVERSION < '3.0.0':
    import Myqttable as qttable
elif QTVERSION < '4.0.0':
    import qttable

if QTVERSION < '4.0.0':
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
        self.mcaroitable     = McaROITable(self)
        if QTVERSION < '4.0.0':
            self.mcaroitable.setMinimumHeight(4*self.mcaroitable.sizeHint().height())
            self.mcaroitable.setMaximumHeight(4*self.mcaroitable.sizeHint().height())
        else:
            rheight = self.mcaroitable.horizontalHeader().sizeHint().height()
            self.mcaroitable.setMinimumHeight(4*rheight)
            self.mcaroitable.setMaximumHeight(4*rheight)
        self.fillfromroidict = self.mcaroitable.fillfromroidict
        self.addroi          = self.mcaroitable.addroi
        self.getroilistanddict=self.mcaroitable.getroilistanddict
        layout.addWidget(self.mcaroitable)
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

        layout.addWidget(hbox)

        self.connect(self.addbutton,  qt.SIGNAL("clicked()"), self.__add)
        self.connect(self.delbutton,  qt.SIGNAL("clicked()"), self.__del)
        self.connect(self.resetbutton,qt.SIGNAL("clicked()"), self.__reset)
        if QTVERSION < '4.0.0':
            self.connect(self.mcaroitable,  qt.PYSIGNAL('McaROITableSignal') ,self.__forward)
        else:
            self.connect(self.mcaroitable,  qt.SIGNAL('McaROITableSignal') ,self.__forward)

    def __add(self):
        ddict={}
        ddict['event']   = "AddROI"
        roilist,roidict  = self.mcaroitable.getroilistanddict()
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'),ddict)
        
        
    def __del(self):
        row = self.mcaroitable.currentRow()
        if row >= 0:
            index = self.mcaroitable.labels.index('Type')
            if qt.qVersion() < '4.0.0':
                text = str(self.mcaroitable.text(row, index))
            else:
                text = str(self.mcaroitable.item(row, index).text())
                
            if text.upper() != 'DEFAULT':
                index = self.mcaroitable.labels.index('ROI')
                if qt.qVersion() < '4.0.0':
                    key = str(self.mcaroitable.text(row, index))
                else:
                    key = str(self.mcaroitable.item(row, index).text())
            else:return
            roilist,roidict    = self.mcaroitable.getroilistanddict()
            row = roilist.index(key)
            del roilist[row]
            del roidict[key]
            self.mcaroitable.fillfromroidict(roilist=roilist,
                                             roidict=roidict,
                                             currentroi=roilist[0])
            ddict={}
            ddict['event']      = "DelROI"
            ddict['roilist']    = roilist
            ddict['roidict']    = roidict
            #ddict['currentrow'] = self.mcaroitable.currentRow()
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,))
            else:
                self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict)
        
    
    def __forward(self,ddict):
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,)) 
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict) 
        
    
    def __reset(self):
        ddict={}
        ddict['event']   = "ResetROI"
        roilist0,roidict0  = self.mcaroitable.getroilistanddict()
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
        self.mcaroitable.fillfromroidict(roilist=roilist,roidict=roidict)
        ddict['roilist'] = roilist
        ddict['roidict'] = roidict
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('McaROIWidgetSignal'), (ddict,))
        else:
            self.emit(qt.SIGNAL('McaROIWidgetSignal'), ddict)
        
    def setdata(self,*var,**kw):
        self.info ={}
        if kw.has_key('legend'):
            self.info['legend'] = kw['legend']
            del kw['legend']
        else:
            self.info['legend'] = 'Unknown Type'
        if kw.has_key('xlabel'):
            self.info['xlabel'] = kw['xlabel']
            del kw['xlabel']
        else:
            self.info['xlabel'] = 'X'
        if kw.has_key('rois'):
            self.mcaroitable.fillfromrois(rois)
        self.setheader(text="%s ROIs of %s" % (self.info['xlabel'],
                                               self.info['legend']))

    def setheader(self,*var,**kw):
        if len(var):
            text = var[0]
        elif kw.has_key('text'):
            text = kw['text']
        elif kw.has_key('header'):
            text = kw['header']
        else:
            text = ""
        self.headerlabel.setText("<b>%s<\b>" % text)

class McaROITable(QTable):
    def __init__(self, *args,**kw):
        apply(QTable.__init__, (self, ) + args)
        self.setRowCount(1)
        self.labels=['ROI','Type','From','To','Raw Counts','Net Counts']
        self.setColumnCount(len(self.labels))
        i=0
        if qt.qVersion() < '4.0.0':
            if kw.has_key('labels'):
                for label in kw['labels']:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i = i + 1
            else:
                for label in self.labels:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i = i + 1
        else:
            if kw.has_key('labels'):
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
        if kw.has_key('roilist'):
            self.roilist = kw['roilist']
        if kw.has_key('roidict'):
            self.roidict.update(kw['roilist'])
        self.build()
        #self.connect(self,qt.SIGNAL("currentChanged(int,int)"),self.myslot)
        if QTVERSION < '4.0.0':
            self.connect(self,qt.SIGNAL("valueChanged(int,int)"),self.myslot)
            self.connect(self,qt.SIGNAL("selectionChanged()"),self.myslot)
        else:
            self.connect(self,qt.SIGNAL("cellClicked(int, int)"),self.myslot)
            self.connect(self,qt.SIGNAL("cellChanged(int, int)"),self.myslot)
            self.connect(self,qt.SIGNAL("itemSelectionChanged()"),self.myslot)
        #self.connect(self,qt.SIGNAL("pressed(int,int,QPoint())"),self.myslot)
        if qt.qVersion() > '2.3.0':
            if qt.qVersion() < '4.0.0':
                self.setSelectionMode(QTable.SingleRow)
            else:
                if DEBUG:print "Qt4 selection mode?"
                #self.setMinimumHeight(8*self.horizontalHeader().sizeHint().height())
                #self.setMaximumHeight(8*self.horizontalHeader().sizeHint().height())
    

    def build(self):
        self.fillfromroidict(roilist=self.roilist,roidict=self.roidict)
    
    def fillfromroidict(self,roilist=[],roidict={},currentroi=None):
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
                roitype = qt.QString("%s" % roi['type'])
                fromdata= qt.QString("%6g" % (roi['from']))
                todata  = qt.QString("%6g" % (roi['to']))
                if roi.has_key('rawcounts'):
                    rawcounts= qt.QString("%6g" % (roi['rawcounts']))
                else:
                    rawcounts = " ?????? "
                if roi.has_key('netcounts'):
                    netcounts= qt.QString("%6g" % (roi['netcounts']))
                else:
                    netcounts = " ?????? "
                fields  = [ROI,roitype,fromdata,todata,rawcounts,netcounts]
                col = 0 
                for field in fields:
                    if qt.qVersion() < '4.0.0':
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
                        else:
                            key2.setText(field)
                        if (ROI.upper() == 'ICR') or (ROI.upper() == 'DEFAULT'):
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled) 
                        else:
                            if col == 0:
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled|
                                              qt.Qt.ItemIsEditable)                        
                            else:
                                key2.setFlags(qt.Qt.ItemIsSelectable|
                                              qt.Qt.ItemIsEnabled) 
                    self.setItem(line,col,key2)
                    col=col+1
        self.setRowCount(line0)
        i = 0
        for label in self.labels:
            self.resizeColumnToContents(i)
            i=i+1
        if qt.qVersion() < '4.0.0':
            self.sortColumn(2,1,1)
        else:
            self.sortByColumn(2)
        for i in range(len(self.roilist)):
            if qt.qVersion() < '4.0.0':
                key = str(self.text(i, 0))
            else:
                key = str(self.item(i, 0).text())
                
            self.roilist[i] = key
            self.roidict[key]['line'] = i
        if len(self.roilist) == 1:
            if qt.qVersion() > '2.3.0':
                self.selectRow(0)
            else:
                if DEBUG:
                    print "Method not implemented, just first cell"
                self.setCurrentCell(0,0)
        else:
            if currentroi in self.roidict.keys():
                if qt.qVersion < '3.0.0':
                    self.setCurrentCell(self.roidict[currentroi]['line'],0)
                else:
                    self.selectRow(self.roidict[currentroi]['line'])
                if qt.qVersion() < '4.0.0':
                    self.ensureCellVisible(self.roidict[currentroi]['line'],0)
                else:
                    if DEBUG:
                        print "Qt4 ensureCellVisible to be implemented"
                

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
        roitype = qt.QString("%s" % roi['type'])
        fromdata= qt.QString("%6g" % (roi['from']))
        todata  = qt.QString("%6g" % (roi['to']))
        if roi.has_key('rawcounts'):
            rawcounts= qt.QString("%6g" % (roi['rawcounts']))
        else:
            rawcounts = " ?????? "
        self.roidict[key]['rawcounts']   = rawcounts
        if roi.has_key('netcounts'):
            netcounts= qt.QString("%6g" % (roi['netcounts']))
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
        self.sortColumn(2,1,1)
        for i in range(len(self.roilist)):
            nkey = str(self.text(i,0))
            self.roilist[i] = nkey
            self.roidict[nkey]['line'] = i
        self.selectRow(self.roidict[key]['line'])
        self.ensureCellVisible(self.roidict[key]['line'],0)

    def getroilistanddict(self):
        return self.roilist,self.roidict 


    def myslot(self,*var,**kw):
        if len(var) == 0:
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
                        print "deleting???"
                    row = 0
                ddict['roi'  ] = self.roidict[self.roilist[row]]
                ddict['key']   = self.roilist[row]
                ddict['colheader'] = self.labels[col]
                ddict['rowheader'] = "%d" % row
                if qt.qVersion() < '4.0.0':
                    self.emit(qt.PYSIGNAL('McaROITableSignal'), (ddict,))
                else:
                    self.emit(qt.SIGNAL('McaROITableSignal'), ddict)
        else:
            if len(var) == 2:
                ddict={}
                row = var[0]
                col = var[1]
                if col == 0:
                    if row >= len(self.roilist):
                        if DEBUG:
                            print "deleting???"
                    else:
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
                            if qt.qVersion() < '4.0.0':
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
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        app.exec_loop()
    else:
        demo.show()
        app.exec_()

