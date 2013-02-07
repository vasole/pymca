#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
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
import posixpath
import weakref
import gc
import h5py
try:
    from PyMca import PyMcaQt as qt
    safe_str = qt.safe_str
except ImportError:
    # for people using this widget without PyMca installed
    import PyQt4.Qt as qt
    safe_str = str

if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = str

from PyMca import HDF5Widget
from PyMca import HDF5Info
from PyMca import HDF5CounterTable
from PyMca import HDF5DatasetTable
from PyMca import ConfigDict
if "PyMcaDirs" in sys.modules:
    from PyMca import PyMcaDirs

DEBUG=0

class Buttons(qt.QWidget):
    def __init__(self, parent=None, options=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.buttonGroup = qt.QButtonGroup(self)
        self.buttonList = []
        i = 0
        if options is None:
            optionList = ['SCAN', 'MCA']
        else:
            optionList = options
        actionList = ['ADD', 'REMOVE', 'REPLACE']
        for option in optionList:
            row = optionList.index(option)
            for action in actionList:
                col = actionList.index(action)
                button = qt.QPushButton(self)
                button.setText(action + " " + option)
                self.mainLayout.addWidget(button, row, col)
                self.buttonGroup.addButton(button)
                self.buttonList.append(button)
        self.connect(self.buttonGroup,
                     qt.SIGNAL('buttonClicked(QAbstractButton *)'),
                     self.emitSignal)

    def emitSignal(self, button):
        ddict={}
        ddict['event'] = 'buttonClicked'
        ddict['action'] = safe_str(button.text())
        self.emit(qt.SIGNAL('ButtonsSignal'), ddict)

class QNexusWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.data = None
        self._dataSourceList = []
        self._oldCntSelection = None
        self._cntList = []
        self._aliasList = []
        self._defaultModel = HDF5Widget.FileModel()
        self.getInfo = HDF5Info.getInfo
        self._modelDict = {}
        self._widgetDict = {}
        self._lastWidgetId = None
        self._dir = None
        self._lastAction = None
        self.build()

    def build(self):
        self.mainLayout = qt.QVBoxLayout(self)
        self.splitter = qt.QSplitter(self)
        self.splitter.setOrientation(qt.Qt.Vertical)
        self.hdf5Widget = HDF5Widget.HDF5Widget(self._defaultModel,
                                                self.splitter)
        self.hdf5Widget.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.cntTable = HDF5CounterTable.HDF5CounterTable(self.splitter)
        self.mainLayout.addWidget(self.splitter)
        #Enable 3D
        if ('PyMca.Object3D' in sys.modules) or \
           ('Object3D' in sys.modules):
            self.buttons = Buttons(self, options=['SCAN', 'MCA', '2D', '3D'])
            self.cntTable.set3DEnabled(True)
        else:
            self.buttons = Buttons(self, options=['SCAN', 'MCA', '2D'])
            self.cntTable.set3DEnabled(False)
        self.mainLayout.addWidget(self.buttons)
        self.connect(self.hdf5Widget,
                     qt.SIGNAL('HDF5WidgetSignal'),
                     self.hdf5Slot)
        self.connect(self.cntTable,
                     qt.SIGNAL('customContextMenuRequested(QPoint)'),
                     self._counterTableCustomMenuSlot)
        self.connect(self.buttons,
                     qt.SIGNAL('ButtonsSignal'),
                     self.buttonsSlot)


        # Some convenience functions to customize the table
        # They could have been included in other class inheriting
        # this one.
        self.cntTable.setContextMenuPolicy(qt.Qt.CustomContextMenu)

        self._cntTableMenu = qt.QMenu(self)
        self._cntTableMenu.addAction(QString("Load"),
                                    self._loadCounterTableConfiguration)
        self._cntTableMenu.addAction(QString("Merge"),
                                    self._mergeCounterTableConfiguration)
        self._cntTableMenu.addAction(QString("Save"),
                                    self._saveCounterTableConfiguration)
        self._cntTableMenu.addSeparator()
        self._cntTableMenu.addAction(QString("Delete All"),
                                    self._deleteAllCountersFromTable)
        self._cntTableMenu.addAction(QString("Delete Selected"),
                                    self._deleteSelectedCountersFromTable)

    def _counterTableCustomMenuSlot(self, qpoint):
        self.getWidgetConfiguration()
        self._cntTableMenu.exec_(qt.QCursor.pos())

    def _getConfigurationFromFile(self, fname):
        ddict = ConfigDict.ConfigDict()
        ddict.read(fname)

        keys = ddict.keys
        if 'PyMca' in keys():
            ddict = ddict['PyMca']
        
        if 'HDF5' not in ddict.keys():
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("File does not contain HDF5 configuration")
            msg.exec_()
            return None

        if 'WidgetConfiguration' not in ddict['HDF5'].keys():
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("File does not contain HDF5 WidgetConfiguration")
            msg.exec_()
            return None

        ddict =ddict['HDF5']['WidgetConfiguration'] 
        keys = ddict.keys()

        if ('counters' not in keys) or\
           ('aliases' not in keys):
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("File does not contain HDF5 counters information")
            msg.exec_()
            return None

        if len(ddict['counters']) != len(ddict['aliases']):
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Number of counters does not match number of aliases")
            msg.exec_()
            return None
        
        return ddict


    def _loadCounterTableConfiguration(self):
        fname = self.getInputFilename()
        if not len(fname):
            return

        ddict = self._getConfigurationFromFile(fname)
        if ddict is not None:
            self.setWidgetConfiguration(ddict)

    def _mergeCounterTableConfiguration(self):
        fname = self.getInputFilename()
        if not len(fname):
            return

        ddict = self._getConfigurationFromFile(fname)
        if ddict is None:
            return

        current = self.getWidgetConfiguration()
        cntList = ddict['counters']
        aliasList = ddict['aliases']
        for i in range(len(cntList)):
            cnt = cntList[i]
            if cnt not in current['counters']:
                current['counters'].append(cnt)
                current['aliases'].append(aliasList[i])

        self.setWidgetConfiguration(current)

    def _saveCounterTableConfiguration(self):
        fname = self.getOutputFilename()
        if not len(fname):
            return
        if not fname.endswith('.ini'):
            fname += '.ini'

        ddict = ConfigDict.ConfigDict()
        if "PyMcaDirs" in sys.modules:
            ddict['PyMca'] = {}
            ddict['PyMca']['HDF5'] = {'WidgetConfiguration':\
                                      self.getWidgetConfiguration()}
        else:
            ddict['HDF5'] ={'WidgetConfiguration':\
                             self.getWidgetConfiguration()}
        ddict.write(fname)

    def _deleteAllCountersFromTable(self):
        current = self.cntTable.getCounterSelection()
        current['x'] = []
        current['y'] = []
        current['m'] = []
        self.cntTable.setCounterSelection(current)
        self.setWidgetConfiguration(None)

    def _deleteSelectedCountersFromTable(self):
        itemList = self.cntTable.selectedItems()
        rowList = []
        for item in itemList:
            row = item.row()
            if row not in rowList:
                rowList.append(row)
                
        rowList.sort()
        rowList.reverse()
        current = self.cntTable.getCounterSelection()
        for row in rowList:
            for key in ['x', 'y', 'm']:
                if row in current[key]:
                    idx = current[key].index(row)
                    del current[key][idx]

        ddict = {}
        ddict['counters'] = []
        ddict['aliases'] = []
        for i in range(self.cntTable.rowCount()):
            if i not in rowList:
                name = safe_str(self.cntTable.item(i, 0).text())
                alias = safe_str(self.cntTable.item(i, 4).text())
                ddict['counters'].append(name)
                ddict['aliases'].append(alias)

        self.setWidgetConfiguration(ddict)
        self.cntTable.setCounterSelection(current)

    def getInputFilename(self):
        if self._dir is None:
            if "PyMcaDirs" in sys.modules:
                inidir = PyMcaDirs.inputDir
            else:
                inidir = os.getcwd()
        else:
            inidir = self._dir

        if not os.path.exists(inidir):
            inidir = os.getcwd()

        ret = safe_str(qt.QFileDialog.getOpenFileName(self,
                                         "Select a .ini file",
                                         inidir,
                                         "*.ini"))
        if len(ret):
            self._dir = os.path.dirname(ret)
            if "PyMcaDirs" in sys.modules:
                PyMcaDirs.inputDir = os.path.dirname(ret)                
        return ret

    def getOutputFilename(self):
        if self._dir is None:
            if "PyMcaDirs" in sys.modules:
                inidir = PyMcaDirs.outputDir
            else:
                inidir = os.getcwd()
        else:
            inidir = self._dir

        if not os.path.exists(inidir):
            inidir = os.getcwd()

        ret = safe_str(qt.QFileDialog.getSaveFileName(self,
                                         "Select a .ini file",
                                         inidir,
                                         "*.ini"))
        if len(ret):
            self._dir = os.path.dirname(ret)
            if "PyMcaDirs" in sys.modules:
                PyMcaDirs.outputDir = os.path.dirname(ret)                
        return ret

    def getWidgetConfiguration(self):
        cntSelection = self.cntTable.getCounterSelection()
        ddict = {}
        ddict['counters'] = cntSelection['cntlist']
        ddict['aliases'] = cntSelection['aliaslist']
        return ddict

    def setWidgetConfiguration(self, ddict=None):
        if ddict is None:
            self._cntList = []
            self._aliasList = []
        else:
            self._cntList = ddict['counters']
            self._aliasList = ddict['aliases']
            if type(self._cntList) == type(""):
                self._cntList = [ddict['counters']]
            if type(self._aliasList) == type(""):
                self._aliasList = [ddict['aliases']]
        self.cntTable.build(self._cntList, self._aliasList)

    def setDataSource(self, dataSource):
        self.data = dataSource
        if self.data is None:
            self.hdf5Widget.collapseAll()
            self.hdf5Widget.setModel(self._defaultModel)
            return
        def dataSourceDestroyed(weakrefReference):
            idx = self._dataSourceList.index(weakrefReference)
            del self._dataSourceList[idx]
            del self._modelDict[weakrefReference]
            return
        ref = weakref.ref(dataSource, dataSourceDestroyed)
        if ref not in self._dataSourceList:
            self._dataSourceList.append(ref)
            self._modelDict[ref] = HDF5Widget.FileModel()
        model = self._modelDict[ref]
        self.hdf5Widget.setModel(model)
        for source in self.data._sourceObjectList:
            self.hdf5Widget.model().appendPhynxFile(source, weakreference=True)

    def setFile(self, filename):
        self._data = self.hdf5Widget.model().openFile(filename, weakreference=True)

    def showInfoWidget(self, filename, name, dset=False):
        self._checkWidgetDict()
        #this solution seems more robust
        if 1:
            useInstance = True
        else:
            if h5py.version.version < '2.0':
                useInstance = True
            else:
                useInstance = False
        if useInstance:
            fileIndex = self.data.sourceName.index(filename)
            phynxFile  = self.data._sourceObjectList[fileIndex]
        else:
            phynxFile  = h5py.File(filename, 'r')
        info = self.getInfo(phynxFile, name)
        widget = HDF5Info.HDF5InfoWidget()
        widget.notifyCloseEventToWidget(self)
        title = os.path.basename(filename)
        title += " %s" % name
        widget.setWindowTitle(title)
        wid = id(widget)
        if self._lastWidgetId is not None:
            try:
                width = self._widgetDict[self._lastWidgetId].width()
                height = self._widgetDict[self._lastWidgetId].height()
                widget.resize(max(150, width), max(150, height))
            except:
                pass
        self._lastWidgetId = wid
        self._widgetDict[wid] = widget
        if useInstance:
            def sourceObjectDestroyed(weakrefReference):
                if wid == self._lastWidgetId:
                    self._latWidgetId = None
                if wid in self._widgetDict:
                    del self._widgetDict[wid]
            widget._sourceObjectWeakReference = weakref.ref(phynxFile,
                                                 sourceObjectDestroyed)
        widget.setInfoDict(info)
        if dset:
            dataset = phynxFile[name]
            if isinstance(dataset, h5py.Dataset):
                if len(dataset.shape):
                    #0 length datasets do not need a table
                    widget.w = HDF5DatasetTable.HDF5DatasetTable(widget)
                    try:
                        widget.w.setDataset(dataset)
                    except:
                        print("Error filling table")
                    widget.addTab(widget.w, 'TableView')
        widget.show()
        return widget

    def itemRightClickedSlot(self, ddict):
        filename = ddict['file']
        name = ddict['name']
        if ddict['dtype'].startswith('|S'):
            #handle a right click on a dataset of string type
            return self.showInfoWidget(filename, name, False)
            pass
        elif ddict['dtype'] == '':
            #handle a right click on a group
            return self.showInfoWidget(filename, name, False)
        elif 0:
            #should I show the option menu?
            self.showInfoWidget(filename, name, True)
            return
        else:
            #handle a right click on a numeric dataset
            _hdf5WidgetDatasetMenu = qt.QMenu(self)
            _hdf5WidgetDatasetMenu.addAction(QString("Add to selection table"),
                                        self._addToSelectionTable)

            if 0:
                #these two options can be combined into one for the time being
                _hdf5WidgetDatasetMenu.addAction(QString("Open"),
                                            self._openDataset)
                _hdf5WidgetDatasetMenu.addAction(QString("Show Properties"),
                                            self._showDatasetProperties)
            else:
                _hdf5WidgetDatasetMenu.addAction(QString("Show Information"),
                                        self._showInfoWidgetSlot)
                self._lastDatasetDict= ddict
                _hdf5WidgetDatasetMenu.exec_(qt.QCursor.pos())
                self._lastDatasetDict= None
            return

    def _addToSelectionTable(self, ddict=None):
        if ddict is None:
            ddict = self._lastDatasetDict
        #handle as a double click
        ddict['event'] = "itemDoubleClicked"
        self.hdf5Slot(ddict)

    def _showInfoWidgetSlot(self, ddict=None):
        if ddict is None:
            ddict = self._lastDatasetDict
        filename = ddict['file']
        name = ddict['name']
        return self.showInfoWidget(filename, name, True)

    def _openDataset(self, ddict=None):
        if ddict is None:
            ddict = self._lastDatasetDict
        filename = ddict['file']
        name = ddict['name']
        self._checkWidgetDict()
        fileIndex = self.data.sourceName.index(filename)
        phynxFile  = self.data._sourceObjectList[fileIndex]        
        dataset = phynxFile[name]
        widget = HDF5DatasetTable.HDF5DatasetTable()
        title = os.path.basename(filename)
        title += " %s" % name
        widget.setWindowTitle(title)
        widget.setDataset(dataset)
        if self._lastWidgetId is not None:
            ids = self._widgetDict.keys()
            if len(ids):
                if self._lastWidgetId in ids:
                    try:
                        width = self._widgetDict[self._lastWidgetId].width()
                        height = self._widgetDict[self._lastWidgetId].height()
                        widget.resize(max(150, width), max(300, height))
                    except:
                        pass
                else:
                    try:
                        width = self._widgetDict[ids[-1]].width()
                        height = self._widgetDict[ids[-1]].height()
                        widget.resize(max(150, width), max(300, height))
                    except:
                        pass
        widget.notifyCloseEventToWidget(self)
        wid = id(widget)
        self._lastWidgetId = wid
        self._widgetDict[wid] = widget
        widget.show()

    def _showDatasetProperties(self, ddict=None):
        if ddict is None:
            ddict = self._lastDatasetDict
        filename = ddict['file']
        name = ddict['name']
        return self.showInfoWidget(filename, name)
                
    def hdf5Slot(self, ddict):
        if ddict['event'] == 'itemClicked':
            if ddict['mouse'] == "right":
                return self.itemRightClickedSlot(ddict)
        if ddict['event'] == "itemDoubleClicked":
            if ddict['type'] in ['Dataset']:
                if ddict['dtype'].startswith('|S'):
                    print("string")
                else:
                    root = ddict['name'].split('/')
                    root = "/" + root[1]
                    if len(ddict['name']) == len(root):
                        cnt = ddict['name']
                    else:
                        cnt  = ddict['name'][len(root):]
                    if cnt not in self._cntList:
                        self._cntList.append(cnt)
                        basename = posixpath.basename(cnt)
                        if basename not in self._aliasList:
                            self._aliasList.append(basename)
                        else:
                            self._aliasList.append(cnt)
                        self.cntTable.build(self._cntList, self._aliasList)
            elif ddict['type'] in ['NXentry', 'Entry']:
                if self._lastAction is None:
                    return
                action, selectionType = self._lastAction.split()
                if action == 'REMOVE':
                    action = 'ADD'
                ddict['action'] = "%s %s" % (action, selectionType)
                self.buttonsSlot(ddict)
            else:
                if self.data is not None:
                    name = ddict['name']
                    filename = ddict['file']
                    fileIndex = self.data.sourceName.index(filename)
                    phynxFile  = self.data._sourceObjectList[fileIndex]
                    dataset = phynxFile[name]
                    if isinstance(dataset, h5py.Dataset):
                        root = ddict['name'].split('/')
                        root = "/" + root[1]
                        cnt  = ddict['name'].split(root)[-1]
                        if cnt not in self._cntList:
                            if DEBUG:
                                print("USING SECOND WAY")
                            self._cntList.append(cnt)
                            basename = posixpath.basename(cnt)
                            if basename not in self._aliasList:
                                self._aliasList.append(basename)
                            else:
                                self._aliasList.append(cnt)
                            self.cntTable.build(self._cntList, self._aliasList)
                        return
                if DEBUG:
                    print("Unhandled item type: %s" % ddict['dtype'])

    def buttonsSlot(self, ddict):
        if self.data is None:
            return
        action, selectionType = ddict['action'].split()
        entryList = self.getSelectedEntries()
        if not len(entryList):
            return
        cntSelection = self.cntTable.getCounterSelection()
        self._aliasList = cntSelection['aliaslist']
        selectionList = []
        for entry, filename in entryList:
            if not len(cntSelection['cntlist']):
                continue
            if not len(cntSelection['y']):
                #nothing to plot
                continue
            for yCnt in cntSelection['y']:
                sel = {}
                sel['SourceName'] = self.data.sourceName * 1
                sel['SourceType'] = "HDF5"
                fileIndex = self.data.sourceName.index(filename)
                phynxFile  = self.data._sourceObjectList[fileIndex]
                entryIndex = list(phynxFile["/"].keys()).index(entry[1:])
                sel['Key']        = "%d.%d" % (fileIndex+1, entryIndex+1)
                sel['legend']     = os.path.basename(sel['SourceName'][0])+\
                                    " " + posixpath.basename(entry) #it was sel['Key']
                sel['selection'] = {}
                sel['selection']['sourcename'] = filename
                #deal with the case the "entry" is a dataset hunging at root level
                if isinstance(phynxFile[entry], h5py.Dataset):
                    entry = "/" 
                sel['selection']['entry'] = entry
                sel['selection']['key'] = "%d.%d" % (fileIndex+1, entryIndex+1)
                sel['selection']['x'] = cntSelection['x']
                sel['selection']['y'] = [yCnt]
                sel['selection']['m'] = cntSelection['m']
                sel['selection']['cntlist'] = cntSelection['cntlist']
                sel['selection']['LabelNames'] = cntSelection['aliaslist']
                #sel['selection']['aliaslist'] = cntSelection['aliaslist']
                sel['selection']['selectiontype'] = selectionType
                if selectionType.upper() == "SCAN":
                    sel['scanselection'] = True
                    sel['mcaselection']  = False
                elif selectionType.upper() == "MCA":
                    sel['scanselection'] = False
                    sel['mcaselection']  = True
                else:
                    sel['scanselection'] = False
                    sel['mcaselection']  = False
                aliases = cntSelection['aliaslist'] 
                if len(cntSelection['x']) and len(cntSelection['m']):
                    addLegend = " (%s/%s) vs %s" % (aliases[yCnt],
                                                   aliases[cntSelection['m'][0]],
                                                   aliases[cntSelection['x'][0]])
                elif len(cntSelection['x']):
                    addLegend = " %s vs %s" % (aliases[yCnt],
                                               aliases[cntSelection['x'][0]])
                elif len(cntSelection['m']):
                    addLegend = " (%s/%s)" % (aliases[yCnt],
                                            aliases[cntSelection['m'][0]])
                else:
                    addLegend = " %s" % aliases[yCnt]
                sel['legend'] += addLegend
                selectionList.append(sel)
        self._lastAction = "%s" % ddict['action']
        if len(selectionList):
            if selectionType.upper() in ["SCAN", "MCA"]:
                ddict = {}
                ddict['event'] = "SelectionTypeChanged"
                ddict['SelectionType'] = selectionType.upper()
                self.emit(qt.SIGNAL('otherSignals'), ddict)
            if action.upper() == "ADD":
                self.emit(qt.SIGNAL("addSelection"), selectionList)
            if action.upper() == "REMOVE":
                self.emit(qt.SIGNAL("removeSelection"), selectionList)
            if action.upper() == "REPLACE":
                self.emit(qt.SIGNAL("replaceSelection"), selectionList)

    def getSelectedEntries(self):
        return self.hdf5Widget.getSelectedEntries()

    def closeEvent(self, event):
        keyList = self._widgetDict.keys()
        for key in keyList:
            self._widgetDict[key].close()
            del self._widgetDict[key]
        return qt.QWidget.closeEvent(self, event)

    def _checkWidgetDict(self):
        keyList = self._widgetDict.keys()
        for key in keyList:
            if self._widgetDict[key].isHidden():
                del self._widgetDict[key]

    def customEvent(self, event):
        if hasattr(event, 'dict'):
            ddict = event.dict
            if 'event' in ddict:
                if ddict['event'] == "closeEventSignal":
                    if ddict['id'] in self._widgetDict:
                        if DEBUG:
                            try:
                                widget = self._widgetDict[ddict['id']] 
                                print("DELETING %s" % widget.windowTitle())
                            except:
                                pass
                        del self._widgetDict[ddict['id']]
    
if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    try:
        #this is to add the 3D buttons ...
        from PyMca import Object3D
    except:
        pass
    w = QNexusWidget()
    if 0:
        w.setFile(sys.argv[1])
    else:
        from PyMca import NexusDataSource
        dataSource = NexusDataSource.NexusDataSource(sys.argv[1:])
        w.setDataSource(dataSource)
    def addSelection(sel):
        print(sel)
    def removeSelection(sel):
        print(sel)
    def replaceSelection(sel):
        print(sel)
    w.show()
    qt.QObject.connect(w, qt.SIGNAL("addSelection"),     addSelection)
    qt.QObject.connect(w, qt.SIGNAL("removeSelection"),  removeSelection)
    qt.QObject.connect(w, qt.SIGNAL("replaceSelection"), replaceSelection)
    sys.exit(app.exec_())
