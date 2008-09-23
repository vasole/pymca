###########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
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
#############################################################################
from QSelectorWidget import qt
import QSelectorWidget
import SpecFileDataInfo
import SpecFileCntTable
import SpecFileMcaTable
import sys
import os

QTVERSION = qt.qVersion()

DEBUG = 0

#class QSpecFileWidget(qt.QWidget):
class QSpecFileWidget(QSelectorWidget.QSelectorWidget):
    def __init__(self, parent=None, autoreplace=False):
        self.autoReplace = autoreplace
        if self.autoReplace:
            self.autoAdd     = False
        else:
            self.autoAdd     = True
        self._oldCntSelection = None
        QSelectorWidget.QSelectorWidget.__init__(self, parent)
        self.dataInfoWidgetDict = {}

    def _build(self):        
        #self.layout= qt.QVBoxLayout(self)
        if QTVERSION < '4.0.0':
            self.list= qt.QListView(self, "ScanList")
            self.list.setSelectionMode(qt.QListView.Extended)
            self.mainTab = qt.QTabWidget(self)
        else:
            self.splitter = qt.QSplitter(self)
            self.splitter.setOrientation(qt.Qt.Vertical)
            self.list  = qt.QTreeWidget(self.splitter)
            self.list.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
            self.mainTab = qt.QTabWidget(self.splitter)

        self.cntTable = SpecFileCntTable.SpecFileCntTable()
        self.mcaTable = SpecFileMcaTable.SpecFileMcaTable()

        self.mainTab.addTab(self.cntTable,str("Counters"))
        self.mainTab.addTab(self.mcaTable,str("MCA"))
        if QTVERSION < '4.0.0':
            self.mainTab.setCurrentPage(self.mainTab.indexOf(self.mcaTable))
        else:
            self.mainTab.setCurrentWidget(self.mcaTable)
        autoBox = qt.QWidget(self)
        autoBoxLayout = qt.QHBoxLayout(autoBox)
        autoBoxLayout.setMargin(0)
        autoBoxLayout.setSpacing(0)
        self.autoOffBox = qt.QCheckBox(autoBox)
        self.autoOffBox.setText("Auto OFF")
        self.autoAddBox = qt.QCheckBox(autoBox)
        self.autoAddBox.setText("Auto ADD")
        self.autoReplaceBox = qt.QCheckBox(autoBox)
        self.autoReplaceBox.setText("Auto REPLACE")
        if self.autoReplace:
            self.autoAddBox.setChecked(False)
            self.autoReplaceBox.setChecked(True)
        else:
            self.autoAddBox.setChecked(True)
            self.autoReplaceBox.setChecked(False)

        autoBoxLayout.addWidget(self.autoOffBox)
        autoBoxLayout.addWidget(self.autoAddBox)
        autoBoxLayout.addWidget(self.autoReplaceBox)

        if QTVERSION < '4.0.0':
            self.mainLayout.addWidget(self.list)
            self.mainLayout.addWidget(self.mainTab)
        else:
            self.mainLayout.addWidget(self.splitter)
        self.mainLayout.addWidget(autoBox)


        # --- list headers
        if QTVERSION < '4.0.0':
            self.list.addColumn("X")
            self.list.addColumn("S#")
            self.list.addColumn("Command")
            self.list.addColumn("Pts")
            self.list.addColumn("Mca")
            self.list.header().setResizeEnabled(0, 0)
            self.list.header().setResizeEnabled(0, 1)
            self.list.header().setResizeEnabled(1, 2)
            self.list.header().setResizeEnabled(0, 3)
            self.list.header().setResizeEnabled(0, 4)
            self.list.header().setClickEnabled(0,-1)
            self.list.setSorting(-1)

            # --- list selection options
            self.list.setAllColumnsShowFocus(1)

            # --- signal handling
            self.connect(self.list, qt.SIGNAL("selectionChanged()"), self.__selectionChanged)
            if QTVERSION > '3.0.0':
                self.connect(self.list,
                    qt.SIGNAL("contextMenuRequested(QListViewItem *, const QPoint &, int)"),
                    self.__contextMenu)
            else:
                self.connect(self.list,
                    qt.SIGNAL("rightButtonPressed(QListViewItem *, const QPoint &, int)"),
                    self.__contextMenu)
            self.connect(self.list, qt.SIGNAL("doubleClicked(QListViewItem *)"),
                    self.__doubleClicked)
            """
            self.connect(self.cntTable,
                         qt.PYSIGNAL('SpecCntTableSignal'),
                         self._cntSignal)
            """

            # --- context menu
            self.menu= qt.QPopupMenu(self.list)
            idd= self.menu.insertItem("Show scan header")
            self.menu.connectItem(idd, self.__showScanInfo)
        else:
            labels = ["X", "S#", "Command", "Points", "Nb. Mca"]
            ncols  = len(labels)
            self.list.setColumnCount(ncols)
            self.list.setHeaderLabels(labels)
            #size=50
            #self.list.header().resizeSection(0, size)
            #self.list.header().resizeSection(1, size)
            #self.list.header().resizeSection(2, 4 * size)
            #self.list.header().resizeSection(3, size)
            #self.list.header().resizeSection(4, size)

            self.list.header().setStretchLastSection(False)
            if QTVERSION < '4.2.0':
                self.list.header().setResizeMode(0, qt.QHeaderView.Stretch)
                self.list.header().setResizeMode(1, qt.QHeaderView.Stretch)
                self.list.header().setResizeMode(2, qt.QHeaderView.Interactive)
                self.list.header().setResizeMode(3, qt.QHeaderView.Stretch)
                self.list.header().setResizeMode(4, qt.QHeaderView.Stretch)
            else:
                self.list.header().setResizeMode(0, qt.QHeaderView.ResizeToContents)
                self.list.header().setResizeMode(1, qt.QHeaderView.ResizeToContents)
                self.list.header().setResizeMode(2, qt.QHeaderView.Interactive)
                self.list.header().setResizeMode(3, qt.QHeaderView.ResizeToContents)
                self.list.header().setResizeMode(4, qt.QHeaderView.ResizeToContents)

            # --- signal handling
            self.connect(self.list, qt.SIGNAL("itemSelectionChanged()"),
                         self.__selectionChanged)
            self.list.setContextMenuPolicy(qt.Qt.CustomContextMenu)
            self.connect(self.list,
                         qt.SIGNAL("customContextMenuRequested(const QPoint &)"),
                         self.__contextMenu)
            self.connect(self.list,
                         qt.SIGNAL("itemDoubleClicked(QTreeWidgetItem *, int)"),
                         self.__doubleClicked)
            self.connect(self.cntTable,
                         qt.SIGNAL('SpecCntTableSignal'),
                         self._cntSignal)

        self.connect(self.autoOffBox, qt.SIGNAL("clicked()"),
                     self._setAutoOff)
        self.connect(self.autoAddBox, qt.SIGNAL("clicked()"),
                     self._setAutoAdd)
        self.connect(self.autoReplaceBox, qt.SIGNAL("clicked()"),
                     self._setAutoReplace)

        if QTVERSION < '4.0.0':
            self.connect(self.mainTab,
                         qt.SIGNAL('currentChanged(QWidget*)'),
                         self._tabChanged)
        else:
            self.connect(self.mainTab,
                         qt.SIGNAL('currentChanged(int)'),
                         self._tabChanged)

        self.disableMca    = 0 #(type=="scan")
        self.disableScan   = 0 #(type=="mca")

        # --- context menu        
        self.data= None
        self.scans= []


    def _setAutoOff(self):
        self.autoAddBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoOffBox.setChecked(True)

    def _setAutoAdd(self):
        self.autoOffBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoAddBox.setChecked(True)

    def _setAutoReplace(self):
        self.autoOffBox.setChecked(False)
        self.autoAddBox.setChecked(False)
        self.autoReplaceBox.setChecked(True)

    # 
    # Data management
    #
    #NEW data management
    def setDataSource(self, datasource):
        self.data = datasource
        self.refresh()
        if QTVERSION < '4.0.0':return

        if not self.autoAddBox.isChecked(): return
        #If there is only one mca containing scan
        # and we are in auto add mode, I plot it.
        if len(self.scans) == 1:
            item = self.list.itemAt(qt.QPoint(0,0))
            if item is not None:
                item.setSelected(True)
                self.__selectionChanged()
    
    #OLD data management
    def setData(self, specfiledata):
        if DEBUG:
            print "setData(self, specfiledata) called"
            print "specfiledata = ",specfiledata
        self.data= specfiledata
        self.refresh()

    def refresh(self):
        self.list.clear()
        if self.data is None: return
        try:
            if self.data.sourceName is None: return        
        except:
            if self.data.SourceName is None: return
        try:
            #new
            info= self.data.getSourceInfo()
        except:
            #old
            info= self.data.GetSourceInfo()
        self.scans= []
        after= None
        if QTVERSION < '4.0.0':
            for (sn, cmd, pts, mca) in zip(info["KeyList"], info["Commands"], info["NumPts"], info["NumMca"]):
                if after is not None:
                    item= qt.QListViewItem(self.list, after, "", sn, cmd, str(pts), str(mca))
                else:
                    item= qt.QListViewItem(self.list, "", sn, cmd, str(pts), str(mca))
                if (self.disableMca and not mca) or (self.disableScan and not pts):
                    item.setSelectable(0)
                    #XXX: not possible to put in italic: other solutions ??
                self.list.insertItem(item)
                self.scans.append(sn)
                after= item
        else:
            i = 0
            for (sn, cmd, pts, mca) in zip(info["KeyList"], info["Commands"], info["NumPts"], info["NumMca"]):
                if after is not None:
                    #print "after is not none"
                    #item= qt.QTreeWidgetItem(self.list, [after, "", sn, cmd, str(pts), str(mca)])
                    item= qt.QTreeWidgetItem(self.list, ["", sn, cmd, str(pts), str(mca)])
                else:
                    item= qt.QTreeWidgetItem(self.list, ["", sn, cmd, str(pts), str(mca)])
                if (self.disableMca and not mca) or (self.disableScan and not pts):
                    item.setSelectable(0)
                    #XXX: not possible to put in italic: other solutions ??
                self.scans.append(sn)
                after= item
                i = i + 1

    def clear(self):
        self.list.clear()
        self.data= None
        self.scans= []

    def markScanSelected(self, scanlist):
        if QTVERSION > '3.0.0':
            for sn in self.scans:
                item= self.list.findItem(sn, 1)
                if item is not None:
                    if sn in scanlist:
                        item.setText(0, "X")
                    else:
                        item.setText(0, "")
        else:
            item = self.list.firstChild()
            while item:
                    if str(item.text(1)) in scanlist:
                        item.setText(0, "X")
                    else:
                        item.setText(0, "")
                    item = item.nextSibling()

    def _autoReplace(self, scanlist):
        if DEBUG:print "autoreplace called with ",scanlist
        if self.autoReplaceBox.isChecked():
            self._replaceClicked()
        elif self.autoAddBox.isChecked():
            self._addClicked()

    #
    # signal/slot handling
    #
    if QTVERSION < '4.0.0':
        """
        def _cntSignal(self, ddict):
            if ddict["event"] == " updated":
                if QTVERSION > '3.0.0':
                    sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
                else:
                    sel = []
                    item = self.list.firstChild()
                    while item:
                        if item.isSelected():
                            sel.append(str(item.text(1)))
                        item=item.nextSibling()
                self._autoReplace(sel)
        """

        def __selectionChanged(self):
            if DEBUG:print "__selectionChanged"
            if QTVERSION > '3.0.0':
                sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
            else:
                sel = []
                item = self.list.firstChild()
                while item:
                    if item.isSelected():
                        sel.append(str(item.text(1)))
                    item=item.nextSibling()
            info = self.data.getKeyInfo(sel[0])
            self.cntTable.info = info
            self.cntTable.refresh()
            if self._oldCntSelection is not None:
                if len(self._oldCntSelection['y']):
                    self.cntTable.markCntSelected(self._oldCntSelection)
            self.mcaTable.info = info
            self.mcaTable.refresh()
            self.emit(qt.PYSIGNAL("scanSelection"), (sel,))
            self._autoReplace(sel)

    else:
        def _cntSignal(self, ddict):
            if ddict["event"] == "updated":                
                itemlist = self.list.selectedItems()
                sel = [str(item.text(1)) for item in itemlist]
                self._autoReplace(sel)

        
        def __selectionChanged(self):
            if DEBUG:print "__selectionChanged"
            itemlist = self.list.selectedItems()
            sel = [str(item.text(1)) for item in itemlist]
            if DEBUG: print "selection = ",sel
            if not len(sel):return
            info = self.data.getKeyInfo(sel[0])
            self.mcaTable.build(info)
            self.cntTable.build(info['LabelNames'])
            if (info['Lines'] > 0) and len(info['LabelNames']):
                if self._oldCntSelection is not None:
                    if len(self._oldCntSelection['y']):
                        self.cntTable.setCounterSelection(self._oldCntSelection)
                    else:
                        if len(self.cntTable.cntList):
                            self.cntTable.setCounterSelection({'x':[0],
                                                               'y':[-1],
                                                'cntlist':info['LabelNames']*1})
                else:
                    if len(self.cntTable.cntList):
                        self.cntTable.setCounterSelection({'x':[0],
                                                           'y':[-1],
                                            'cntlist':info['LabelNames']*1})
                
            self.emit(qt.SIGNAL("scanSelection"), (sel))
            if (info['NbMca'] > 0) and (info['Lines'] > 0):
                pass
            elif (info['NbMca'] > 0) and (info['Lines'] == 0):
                self.mainTab.setCurrentWidget(self.mcaTable)
            elif (info['NbMca'] == 0) and (info['Lines'] > 0):
                self.mainTab.setCurrentWidget(self.cntTable)
            else:
                pass
            self._autoReplace(sel)

    def __doubleClicked(self, item):
        if DEBUG:print "__doubleClicked"
        if item is not None:
            sn  = str(item.text(1))
            ddict={}
            ddict['Key']      = sn
            ddict['Command']  = str(item.text(2))
            ddict['NbPoints'] = int(str(item.text(3)))
            ddict['NbMca']    = int(str(item.text(4)))
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("scanDoubleClicked"), (ddict,))
            else:
                self.emit(qt.SIGNAL("scanDoubleClicked"), ddict)
            #shortcut selec + remove?
            #for the time being just add
            self._addClicked()

    if QTVERSION < '4.0.0':        
        def __contextMenu(self, item, point, col=None):
            if DEBUG:print "__contextMenu"
            if item is not None:
                sn= str(item.text(1))
                self.menu.setItemParameter(self.menu.idAt(0), self.scans.index(sn))
                self.menu.popup(point)
    else:
        def __contextMenu(self, point):
            if DEBUG:print "__contextMenu",point
            item = self.list.itemAt(point)
            if item is not None:
                sn= str(item.text(1))
                self.menu= qt.QMenu()
                self.menu.addAction("Show scan header", self.__showScanInfo)
                self.menu_idx = self.scans.index(sn)
                self.menu.popup(self.cursor().pos())

    def __showScanInfo(self, idx = None):
        if idx is None:
            if QTVERSION > '4.0.0': 
                idx = self.menu_idx
        if DEBUG:
            print "Scan information:"
        try:
            info = self.data.getDataObject(self.scans[idx]).info
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "Error: %s\n accessing scan information." % (sys.exc_info()[1])
            msg.setText(text)
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
            
        dataInfoWidget= SpecFileDataInfo.SpecFileDataInfo(info)
        if info.has_key("Header"):
            if QTVERSION > '4.0.0':
                dataInfoWidget.setWindowTitle(info['Header'][0])
            else:
                dataInfoWidget.setCaption(info['Header'][0])
        dataInfoWidget.show()
        wid = id(dataInfoWidget)
        self.dataInfoWidgetDict[wid] = dataInfoWidget
        if QTVERSION < '4.0.0':
            self.connect(dataInfoWidget,
                     qt.PYSIGNAL('SpecFileDataInfoSignal'),
                     self._dataInfoClosed)
        else:
            self.connect(dataInfoWidget,
                     qt.SIGNAL('SpecFileDataInfoSignal'),
                     self._dataInfoClosed)

    def _dataInfoClosed(self, ddict):
        if ddict['event'] == "SpecFileDataInfoClosed":
            del self.dataInfoWidgetDict[ddict['id']]

    def _addClicked(self):
        if DEBUG: print "Overwritten _addClicked method"

        #get selected scan keys
        if QTVERSION < '4.0.0':
            if QTVERSION > '3.0.0':
                scan_sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
            else:
                scan_sel = []
                item = self.list.firstChild()
                while item:
                    if item.isSelected():
                        scan_sel.append(str(item.text(1)))
                    item=item.nextSibling()
        else:
            itemlist = self.list.selectedItems()
            scan_sel = [str(item.text(1)) for item in itemlist]

        #get selected counter keys
        cnt_sel = self.cntTable.getCounterSelection()
        if len(cnt_sel['cntlist']):
            if len(cnt_sel['y']): self._oldCntSelection = cnt_sel
        mca_sel = self.mcaTable.getCurrentlySelectedMca()

        sel_list = []
        #build the appropriate selection for mca's
        for scan in scan_sel:
            for mca in mca_sel:
                sel = {}
                sel['SourceName'] = self.data.sourceName
                sel['SourceType'] = self.data.sourceType
                sel['Key'] = scan
                sel['Key'] += "."+mca
                sel['selection'] = None #for the future
                #sel['scanselection']  = False
                sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                sel_list.append(sel)
            if len(cnt_sel['cntlist']):
                if len(cnt_sel['y']): #if there is something to plot
                    sel = {}
                    sel['SourceName'] = self.data.sourceName
                    sel['SourceType'] = self.data.sourceType
                    sel['Key'] = scan
                    sel['selection'] = {}
                    sel['scanselection']  = True
                    sel['selection']['x'] = cnt_sel['x'] 
                    sel['selection']['y'] = cnt_sel['y'] 
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    sel_list.append(sel)

        if QTVERSION < '4.0.0':
            if len(sel_list): self.emit(qt.PYSIGNAL("addSelection"), (sel_list,))
        else:
            if len(sel_list): self.emit(qt.SIGNAL("addSelection"), sel_list)
        

    def _removeClicked(self):
        if DEBUG: print "Overwritten _removeClicked method"


        #get selected scan keys
        if QTVERSION < '4.0.0':
            if QTVERSION > '3.0.0':
                scan_sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
            else:
                scan_sel = []
                item = self.list.firstChild()
                while item:
                    if item.isSelected():
                        scan_sel.append(str(item.text(1)))
                    item=item.nextSibling()
        else:
            itemlist = self.list.selectedItems()
            scan_sel = [str(item.text(1)) for item in itemlist]

        #get selected counter keys
        cnt_sel = self.cntTable.getCounterSelection()
        mca_sel = self.mcaTable.getCurrentlySelectedMca()

        sel_list = []
        #build the appropriate selection for mca's
        for scan in scan_sel:
            for mca in mca_sel:
                sel = {}
                sel['SourceName'] = self.data.sourceName
                sel['SourceType'] = self.data.sourceType
                sel['Key'] = scan
                sel['Key'] += "."+mca
                sel['selection'] = None #for the future
                #sel['scanselection']  = False
                sel['legend'] = os.path.basename(sel['SourceName'][0]) +" "+sel['Key']
                sel_list.append(sel)
            if len(cnt_sel['cntlist']):
                if len(cnt_sel['y']): #if there is something to plot
                    sel = {}
                    sel['SourceName'] = self.data.sourceName
                    sel['SourceType'] = self.data.sourceType
                    sel['Key'] = scan
                    sel['selection'] = {}
                    sel['scanselection']  = True
                    sel['selection']['x'] = cnt_sel['x'] 
                    sel['selection']['y'] = cnt_sel['y'] 
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    sel_list.append(sel)
            
            
        if len(sel_list): 
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("removeSelection"), (sel_list,))            
            else:
                self.emit(qt.SIGNAL("removeSelection"), sel_list)

    def _replaceClicked(self):
        if DEBUG: print "Overwritten _replaceClicked method"
        #get selected scan keys
        if QTVERSION < '4.0.0':
            if QTVERSION > '3.0.0':
                scan_sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
            else:
                scan_sel = []
                item = self.list.firstChild()
                while item:
                    if item.isSelected():
                        scan_sel.append(str(item.text(1)))
                    item=item.nextSibling()
        else:
            itemlist = self.list.selectedItems()
            scan_sel = [str(item.text(1)) for item in itemlist]

        #get selected counter keys
        cnt_sel = self.cntTable.getCounterSelection()
        if len(cnt_sel['cntlist']):
            if len(cnt_sel['y']): self._oldCntSelection = cnt_sel
        mca_sel = self.mcaTable.getCurrentlySelectedMca()

        sel_list = []
        #build the appropriate selection for mca's
        for scan in scan_sel:
            for mca in mca_sel:
                sel = {}
                sel['SourceName'] = self.data.sourceName
                sel['SourceType'] = self.data.sourceType
                sel['Key'] = scan
                sel['Key'] += "."+mca
                sel['selection'] = None #for the future
                #sel['scanselection']  = False #This could also be MCA
                sel['legend'] = os.path.basename(sel['SourceName'][0]) +" "+sel['Key']
                sel_list.append(sel)
            if len(cnt_sel['cntlist']):
                sel = {}
                sel['SourceName'] = self.data.sourceName
                sel['SourceType'] = self.data.sourceType
                sel['Key'] = scan
                if len(cnt_sel['y']): #if there is something to plot
                    sel['scanselection']  = True #This could also be SCAN
                    sel['selection'] = {}
                    sel['selection']['x'] = cnt_sel['x'] 
                    sel['selection']['y'] = cnt_sel['y'] 
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    sel_list.append(sel)

        if len(sel_list): 
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("replaceSelection"), (sel_list,))            
            else:
                self.emit(qt.SIGNAL("replaceSelection"), sel_list)

    def _tabChanged(self, value):
        if DEBUG:print "self._tabChanged(value), value =  ",value
        if QTVERSION < '4.0.0':
            #is not an index but a widget
            index = self.mainTab.indexOf(value)
            text = str(self.mainTab.label(index))
        else:
            text = str(self.mainTab.tabText(value))
        if self.data is None: return

        ddict = {}
        ddict['SourceName'] = self.data.sourceName
        ddict['SourceType'] = self.data.sourceType
        ddict['event'] = "SelectionTypeChanged"
        ddict['SelectionType'] = text
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("otherSignals"), (ddict,))
        else:
            self.emit(qt.SIGNAL("otherSignals"), ddict)
            
def test():
    import DataSource
    a = qt.QApplication(sys.argv)
    w = QSpecFileWidget()
    if len(sys.argv) > 1:
        d = DataSource.DataSource(sys.argv[1])
    else:
        if os.path.exists('03novs060sum.mca'):
            d = DataSource.DataSource('03novs060sum.mca')
        else:
            print "Usage:"
            print "      python QSpecFileWidget.py filename"
            a.quit()
            sys.exit(0)
    w.setData(d) 
    w.show()
    qt.QObject.connect(a, qt.SIGNAL("lastWindowClosed()"),
                       a, qt.SLOT("quit()"))

    if QTVERSION < '4.0.0':
        a.exec_loop()
    else:
        a.exec_()




if __name__=="__main__":
    test()
