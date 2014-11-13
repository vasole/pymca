#/*##########################################################################
# Copyright (C) 2004-2014 E. Papillon, V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "E. Papillon, V.A. Sole - ESRF Software Group"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.io import QSelectorWidget
from PyMca5.PyMcaGui.io import SpecFileDataInfo
from PyMca5.PyMcaGui.io import SpecFileCntTable
OBJECT3D = SpecFileCntTable.OBJECT3D
from PyMca5.PyMcaGui.io import SpecFileMcaTable

QTVERSION = qt.qVersion()

DEBUG = 0
if QTVERSION > '4.2.0':
    class MyQTreeWidgetItem(qt.QTreeWidgetItem):
        def __lt__(self, other):
            c = self.treeWidget().sortColumn()
            if  c == 0:
                return False
            if c !=  2:
                return (float(self.text(c)) <  float(other.text(c)))
            return (self.text(c) < other.text(c))
else:
    MyQTreeWidgetItem = qt.QTreeWidgetItem

#class QSpecFileWidget(qt.QWidget):
class QSpecFileWidget(QSelectorWidget.QSelectorWidget):
    sigAddSelection = qt.pyqtSignal(object)
    sigRemoveSelection = qt.pyqtSignal(object)
    sigReplaceSelection = qt.pyqtSignal(object)
    sigOtherSignals = qt.pyqtSignal(object)
    sigScanSelection = qt.pyqtSignal(object)
    sigScanDoubleClicked = qt.pyqtSignal(object)
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
        self.splitter = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Vertical)
        self.list  = qt.QTreeWidget(self.splitter)
        self.list.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.mainTab = qt.QTabWidget(self.splitter)

        self.cntTable = SpecFileCntTable.SpecFileCntTable()
        self.mcaTable = SpecFileMcaTable.SpecFileMcaTable()

        self.mainTab.addTab(self.cntTable, str("Counters"))
        self.mainTab.addTab(self.mcaTable, str("MCA"))
        self.mainTab.setCurrentWidget(self.mcaTable)

        autoBox = qt.QWidget(self)
        autoBoxLayout = qt.QGridLayout(autoBox)
        autoBoxLayout.setContentsMargins(0, 0, 0, 0)
        autoBoxLayout.setSpacing(0)
        self.autoOffBox = qt.QCheckBox(autoBox)
        self.autoOffBox.setText("Auto OFF")
        self.autoAddBox = qt.QCheckBox(autoBox)
        self.autoAddBox.setText("Auto ADD")
        self.autoReplaceBox = qt.QCheckBox(autoBox)
        self.autoReplaceBox.setText("Auto REPLACE")

        row = 0
        autoBoxLayout.addWidget(self.autoOffBox, row, 0)
        autoBoxLayout.addWidget(self.autoAddBox, row, 1)
        autoBoxLayout.addWidget(self.autoReplaceBox, row, 2)

        if self.autoReplace:
            self.autoAddBox.setChecked(False)
            self.autoReplaceBox.setChecked(True)
        else:
            self.autoAddBox.setChecked(True)
            self.autoReplaceBox.setChecked(False)
        row += 1

        if OBJECT3D:
            self.object3DBox = qt.QCheckBox(autoBox)
            self.object3DBox.setText("3D On")
            autoBoxLayout.addWidget(self.object3DBox, row, 0)
            self.mcaTable.sigMcaDeviceSelected.connect(self.mcaDeviceSelected)

        self.meshBox = qt.QCheckBox(autoBox)
        self.meshBox.setText("Mesh")
        self.meshBox.setToolTip("Consider selection as a regular mesh")
        autoBoxLayout.addWidget(self.meshBox, row, 1)


        self.forceMcaBox = qt.QCheckBox(autoBox)
        self.forceMcaBox.setText("Force MCA")
        autoBoxLayout.addWidget(self.forceMcaBox, row, 2)

        self.mainLayout.addWidget(self.splitter)
        self.mainLayout.addWidget(autoBox)


        # --- list headers
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
        if QTVERSION > '5.0.0':
            self.list.header().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
            self.list.header().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
            self.list.header().setSectionResizeMode(2, qt.QHeaderView.Interactive)
            self.list.header().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
            self.list.header().setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        elif QTVERSION < '4.2.0':
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
        self.list.itemSelectionChanged.connect(self.__selectionChanged)
        self.list.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self.__contextMenu)
        self.list.itemDoubleClicked[qt.QTreeWidgetItem, int].connect( \
                     self.__doubleClicked)
        self.cntTable.sigSpecFileCntTableSignal.connect(self._cntSignal)

        if QTVERSION > '4.2.0':
            self.list.setSortingEnabled(False)
            self.list.header().sectionDoubleClicked[int].connect( \
                         self.__headerSectionDoubleClicked)
        if OBJECT3D:
            self.object3DBox.clicked.connect(self._setObject3DBox)
        if hasattr(self, 'meshBox'):
            self.meshBox.clicked.connect(self._setMeshBox)

        self.autoOffBox.clicked.connect(self._setAutoOff)
        self.autoAddBox.clicked.connect(self._setAutoAdd)
        self.autoReplaceBox.clicked.connect(self._setAutoReplace)

        self.forceMcaBox.clicked.connect(self._setForcedMca)

        self.mainTab.currentChanged[int].connect(self._tabChanged)

        self.disableMca    = 0 #(type=="scan")
        self.disableScan   = 0 #(type=="mca")

        # --- context menu
        self.data= None
        self.scans= []


    def _setObject3DBox(self):
        self.autoAddBox.setChecked(False)
        self.meshBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoOffBox.setChecked(False)
        self.cntTable.set3DEnabled(True)
        self.object3DBox.setChecked(True)

    def _setMeshBox(self):
        self.autoAddBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoOffBox.setChecked(False)
        self.cntTable.set2DEnabled(True)
        if hasattr(self, "object3DBox"):
            self.object3DBox.setChecked(False)
        self.meshBox.setChecked(True)

    def _setAutoOff(self):
        if OBJECT3D:
            self.cntTable.set3DEnabled(False)
            self.object3DBox.setChecked(False)
        if hasattr(self, "meshBox"):
            self.cntTable.set2DEnabled(False)
            self.meshBox.setChecked(False)
        self.autoAddBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoOffBox.setChecked(True)

    def _setAutoAdd(self):
        if OBJECT3D:
            self.cntTable.set3DEnabled(False)
            self.object3DBox.setChecked(False)
        if hasattr(self, "meshBox"):
            self.meshBox.setChecked(False)
            self.cntTable.set2DEnabled(False)
        self.autoOffBox.setChecked(False)
        self.autoReplaceBox.setChecked(False)
        self.autoAddBox.setChecked(True)

    def _setAutoReplace(self):
        if OBJECT3D:
            self.cntTable.set3DEnabled(False)
            self.object3DBox.setChecked(False)
        if hasattr(self, "meshBox"):
            self.cntTable.set2DEnabled(False)
            self.meshBox.setChecked(False)
        self.autoOffBox.setChecked(False)
        self.autoAddBox.setChecked(False)
        self.autoReplaceBox.setChecked(True)

    def _setForcedMca(self):
        if self.forceMcaBox.isChecked():
            if OBJECT3D:
                self.object3DBox.setChecked(False)
                self.object3DBox.setEnabled(False)
            if hasattr(self, "meshBox"):
                self.meshBox.setChecked(False)
                self.meshBox.setEnabled(False)
        else:
            if OBJECT3D:
                self.object3DBox.setEnabled(True)
            if hasattr(self, "meshBox"):
                self.meshBox.setEnabled(True)

    #
    # Data management
    #
    #NEW data management
    def setDataSource(self, datasource):
        if DEBUG:
            print("setDataSource(self, datasource) called")
            print("datasource = ", datasource)
        self.data = datasource
        self.refresh()

        if not self.autoAddBox.isChecked():
            return
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
            print("setData(self, specfiledata) called")
            print("specfiledata = ",specfiledata)
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
            if not hasattr(self.data, "GetSourceInfo"):
                raise
            info= self.data.GetSourceInfo()
        self.scans= []
        after= None
        i = 0
        for (sn, cmd, pts, mca) in zip(info["KeyList"], info["Commands"], info["NumPts"], info["NumMca"]):
            if after is not None:
                #print "after is not none"
                #item= qt.QTreeWidgetItem(self.list, [after, "", sn, cmd, str(pts), str(mca)])
                item= MyQTreeWidgetItem(self.list, ["", sn, cmd, str(pts), str(mca)])
            else:
                item= MyQTreeWidgetItem(self.list, ["", sn, cmd, str(pts), str(mca)])
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
        for sn in self.scans:
            item= self.list.findItem(sn, 1)
            if item is not None:
                if sn in scanlist:
                    item.setText(0, "X")
                else:
                    item.setText(0, "")

    def _autoReplace(self, scanlist=None):
        if DEBUG:
            print("autoreplace called with ",scanlist)
        if self.autoReplaceBox.isChecked():
            self._replaceClicked()
        elif self.autoAddBox.isChecked():
            self._addClicked()

    #
    # signal/slot handling
    #
    def _cntSignal(self, ddict):
        if ddict["event"] == "updated":
            itemlist = self.list.selectedItems()
            sel = [str(item.text(1)) for item in itemlist]
            self._autoReplace(sel)


    def __selectionChanged(self):
        if DEBUG:
            print("__selectionChanged")
        itemlist = self.list.selectedItems()
        sel = [str(item.text(1)) for item in itemlist]
        if DEBUG:
            print("selection = ",sel)
        if not len(sel):
            return
        info = self.data.getKeyInfo(sel[0])
        self.mcaTable.build(info)
        if False:
            # This does not work properly yet
            # TODO: mca as function of other parameter
            NbMca = info.get('NbMcaDet', 0)
            self.cntTable.build(info['LabelNames'], nmca=NbMca)
        else:
            self.cntTable.build(info['LabelNames'], nmca=0)

        autoReplaceCall = True
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
            # That already emitted a signal, no need to repeat with
            # autoreplace
            autoReplaceCall = False

        # Emit this signal for the case someone else uses it ...
        self.sigScanSelection.emit((sel))
        if (info['NbMca'] > 0) and (info['Lines'] > 0):
            pass
        elif (info['NbMca'] > 0) and (info['Lines'] == 0):
            self.mainTab.setCurrentWidget(self.mcaTable)
        elif (info['NbMca'] == 0) and (info['Lines'] > 0):
            self.mainTab.setCurrentWidget(self.cntTable)
        else:
            pass
        # Next call is needed to handle the direct opening of MCAs
        # when using a single scan, single mca file.
        if autoReplaceCall:
            self._autoReplace(sel)

    def __headerSectionDoubleClicked(self, index):
        if index == 0:
            return
        else:
            self.list.sortItems(index, qt.Qt.AscendingOrder)
            #print "index = ", index


    def __doubleClicked(self, item):
        if DEBUG:
            print("__doubleClicked")
        if item is not None:
            sn  = str(item.text(1))
            ddict={}
            ddict['Key']      = sn
            ddict['Command']  = str(item.text(2))
            ddict['NbPoints'] = int(str(item.text(3)))
            ddict['NbMca']    = int(str(item.text(4)))
            self.sigScanDoubleClicked.emit(ddict)
            #shortcut selec + remove?
            #for the time being just add
            self._addClicked()

    def __contextMenu(self, point):
        if DEBUG:
            print("__contextMenu",point)
        item = self.list.itemAt(point)
        if item is not None:
            sn= str(item.text(1))
            self.menu= qt.QMenu()
            self.menu.addAction("Show scan header", self.__showScanInfo)
            self.menu_idx = self.scans.index(sn)
            self.menu.popup(self.cursor().pos())

    def mcaDeviceSelected(self, ddict):
        action, actiontype = ddict['action'].split()
        mca = ddict['mca'] + 1
        sel_list = []
        itemlist = self.list.selectedItems()
        scan_sel = [str(item.text(1)) for item in itemlist]
        for scan in scan_sel:
            sel = {}
            sel['SourceName'] = self.data.sourceName
            sel['SourceType'] = self.data.sourceType
            sel['Key'] = "%s.%d"% (scan, mca)
            #sel['selection'] = None
            sel['selection'] = {}
            if actiontype.upper() == "STACK":
                sel['selection']['selectiontype'] = "STACK"
                sel['imageselection'] = False
            else:
                sel['selection']['selectiontype'] = "2D"
                sel['imageselection'] = True
            sel['scanselection'] = False
            sel['mcaselection']  = False
            sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
            sel_list.append(sel)
        if len(scan_sel):
            if action == 'ADD':
                self.sigAddSelection.emit(sel_list)
            elif action == 'REMOVE':
                self.sigRemoveSelection.emit(sel_list)
            elif action == 'REPLACE':
                self.sigReplaceSelection.emit(sel_list)

    def __showScanInfo(self, idx = None):
        if idx is None:
            if QTVERSION > '4.0.0':
                idx = self.menu_idx
        if DEBUG:
            print("Scan information:")

        try:
            info = self.data.getDataObject(self.scans[idx]).info
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "Error: %s\n accessing scan information." % (sys.exc_info()[1])
            msg.setText(text)
            msg.exec_()
            if DEBUG:
                raise
            return

        dataInfoWidget= SpecFileDataInfo.SpecFileDataInfo(info)
        if "Header" in info:
            if info['Header'] is not None:
                dataInfoWidget.setWindowTitle(info['Header'][0])
        dataInfoWidget.show()
        wid = id(dataInfoWidget)
        self.dataInfoWidgetDict[wid] = dataInfoWidget
        dataInfoWidget.notifyCloseEventToWidget(self)

    def _dataInfoClosed(self, ddict):
        if ddict['event'] == "SpecFileDataInfoClosed":
            del self.dataInfoWidgetDict[ddict['id']]

    def customEvent(self, event):
        if hasattr(event, 'dict'):
            ddict = event.dict
            self._dataInfoClosed(ddict)

    def _addClicked(self, emit=True):
        if DEBUG:
            print("Overwritten _addClicked method")

        #get selected scan keys
        if QTVERSION < '4.0.0':
            scan_sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
        else:
            itemlist = self.list.selectedItems()
            scan_sel = [str(item.text(1)) for item in itemlist]

        #get selected counter keys
        cnt_sel = self.cntTable.getCounterSelection()
        if len(cnt_sel['cntlist']):
            if len(cnt_sel['y']):
                self._oldCntSelection = cnt_sel
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
                    if self.forceMcaBox.isChecked():
                        sel['scanselection']  = "MCA"
                    else:
                        sel['scanselection']  = True
                    sel['selection']['x'] = cnt_sel['x']
                    if len(sel['selection']['x']) == 2:
                        if self.meshBox.isChecked():
                            sel['selection']['selectiontype'] = "2D"
                    sel['selection']['y'] = cnt_sel['y']
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    if cnt_sel['y'][0] >= len(cnt_sel['cntlist']):
                        if 'mcalist' in cnt_sel:
                            sel['selection']['mcalist'] = cnt_sel['mcalist']
                        else:
                            # I could rise the exception here
                            # but I let the data source to rise it.
                            pass
                    sel_list.append(sel)
        if emit:
            if len(sel_list):
                self.sigAddSelection.emit(sel_list)
        else:
            return sel_list

    def currentSelectionList(self):
        return self._addClicked(emit=False)


    def _removeClicked(self):
        if DEBUG:
            print("Overwritten _removeClicked method")

        #get selected scan keys
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
                    if self.forceMcaBox.isChecked():
                        sel['scanselection']  = "MCA"
                    else:
                        sel['scanselection']  = True
                    sel['selection']['x'] = cnt_sel['x']
                    if len(sel['selection']['x']) == 2:
                        if self.meshBox.isChecked():
                            sel['selection']['selectiontype'] = "2D"
                    sel['selection']['y'] = cnt_sel['y']
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    sel_list.append(sel)

        if len(sel_list):
            self.sigRemoveSelection.emit(sel_list)

    def _replaceClicked(self):
        if DEBUG:
            print("Overwritten _replaceClicked method")
        #get selected scan keys
        itemlist = self.list.selectedItems()
        scan_sel = [str(item.text(1)) for item in itemlist]

        #get selected counter keys
        cnt_sel = self.cntTable.getCounterSelection()
        if len(cnt_sel['cntlist']):
            if len(cnt_sel['y']):
                self._oldCntSelection = cnt_sel
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
                    if self.forceMcaBox.isChecked():
                        sel['scanselection']  = "MCA"
                    else:
                        sel['scanselection']  = True #This could also be SCAN
                    sel['selection'] = {}
                    sel['selection']['x'] = cnt_sel['x']
                    if len(sel['selection']['x']) == 2:
                        if self.meshBox.isChecked():
                            sel['selection']['selectiontype'] = "2D"
                    sel['selection']['y'] = cnt_sel['y']
                    sel['selection']['m'] = cnt_sel['m']
                    sel['selection']['cntlist'] = cnt_sel['cntlist']
                    sel['legend']    = os.path.basename(sel['SourceName'][0]) +" "+ sel['Key']
                    if cnt_sel['y'][0] >= len(cnt_sel['cntlist']):
                        if 'mcalist' in cnt_sel:
                            sel['selection']['mcalist'] = cnt_sel['mcalist']
                        else:
                            # I could rise the exception here
                            # but I let the data source to rise it.
                            pass
                    sel_list.append(sel)
        if len(sel_list):
            self.sigReplaceSelection.emit(sel_list)

    def _tabChanged(self, value):
        if DEBUG:
            print("self._tabChanged(value), value =  ",value)
        text = str(self.mainTab.tabText(value))
        if self.data is None: return

        ddict = {}
        ddict['SourceName'] = self.data.sourceName
        ddict['SourceType'] = self.data.sourceType
        ddict['event'] = "SelectionTypeChanged"
        ddict['SelectionType'] = text
        self.sigOtherSignals.emit(ddict)

def test():
    from PyMca5.PyMcaGui.pymca import QDataSource
    a = qt.QApplication(sys.argv)
    w = QSpecFileWidget()
    if len(sys.argv) > 1:
        d = QDataSource.QDataSource(sys.argv[1])
    else:
        if os.path.exists('03novs060sum.mca'):
            d = QDataSource.QDataSource('03novs060sum.mca')
        else:
            print("Usage:")
            print("      python QSpecFileWidget.py filename")
            a.quit()
            sys.exit(0)
    w.setData(d)
    w.show()
    def mySlot(selection):
        print(selection)
        try:
            # this is only for "addSelection"
            print(d.getDataObject(selection[0]['Key'], selection[0]['selection']))
        except:
            pass
        return
    w.sigAddSelection.connect(mySlot)
    w.sigRemoveSelection.connect(mySlot)
    w.sigReplaceSelection.connect(mySlot)
    a.lastWindowClosed.connect(a.quit)

    a.exec_()


if __name__=="__main__":
    test()
