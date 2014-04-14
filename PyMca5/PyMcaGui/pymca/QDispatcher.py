#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
import traceback
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()
from PyMca5.PyMcaGui.io import QSourceSelector
from . import QDataSource
#import weakref

DEBUG = 0

class QDispatcher(qt.QWidget):
    def __init__(self, parent=None, pluginsIcon=False):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.sourceList = []
        fileTypeList = ["Spec Files (*mca)",
                        "Spec Files (*dat)",
                        "Spec Files (*spec)",
                        "SPE Files (*SPE *spe)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "TIFF Files (*.tif *.tiff *.TIF *.TIFF)",
                        "CSV Files (*csv)"]
        if QDataSource.NEXUS:
            fileTypeList.append("HDF5 Files (*.nxs *.hdf *.h5 *.hdf5)")
        fileTypeList.append("All Files (*)")
        
        self.sourceSelector = QSourceSelector.QSourceSelector(self,
                                    filetypelist=fileTypeList,
                                    pluginsIcon=pluginsIcon)
        if pluginsIcon:
            self.sourceSelector.pluginsButton.clicked.connect(self._pluginsClicked)
            self.pluginsCallback = None
        self.selectorWidget = {}
        self.tabWidget = qt.QTabWidget(self)
        
        #for the time being just files
        for src_widget in QDataSource.source_widgets.keys():
            self.selectorWidget[src_widget] = QDataSource.source_widgets[src_widget]()
            self.tabWidget.addTab(self.selectorWidget[src_widget], src_widget)
            self.selectorWidget[src_widget].sigAddSelection.connect( \
                            self._addSelectionSlot)                                                 
            self.selectorWidget[src_widget].sigRemoveSelection.connect( \
                         self._removeSelectionSlot)
            self.selectorWidget[src_widget].sigReplaceSelection.connect( \
                         self._replaceSelectionSlot)
            if src_widget not in ['EdfFile']:
                self.selectorWidget[src_widget].sigOtherSignals.connect( \
                         self._otherSignalsSlot)
        
        self.mainLayout.addWidget(self.sourceSelector)
        self.mainLayout.addWidget(self.tabWidget)
        self.sourceSelector.sigSourceSelectorSignal.connect( \
                    self._sourceSelectorSlot)
        self.tabWidget.currentChanged[int].connect(self._tabChanged)

    def _addSelectionSlot(self, sel_list, event=None):
        if DEBUG:
            print("QDispatcher._addSelectionSlot")
            print("sel_list = ",sel_list)

        if event is None:
            event = "addSelection"
        for sel in sel_list:
            #The dispatcher should be a singleton to work properly
            #implement a patch
            targetwidgetid = sel.get('targetwidgetid', None)
            if targetwidgetid not in [None, id(self)]:
                continue
            #find the source
            sourcelist = sel['SourceName']
            for source in self.sourceList:
                selectionList = []
                if source.sourceName == sourcelist:
                    ddict = {}
                    ddict.update(sel)
                    ddict["event"]  = event
                    #we have found the source  
                    #this recovers the data and the info
                    if True:
                        #this creates a data object that is passed to everybody so
                        #there is only one read out.
                        #I should create a weakref to it in order to be informed
                        #about its deletion.
                        if source.sourceType != "SPS":
                            if DEBUG:
                                dataObject = source.getDataObject(sel['Key'],
                                                      selection=sel['selection'])
                            else:
                                try:
                                    dataObject = source.getDataObject(sel['Key'],
                                                          selection=sel['selection'])
                                except:
                                    error = sys.exc_info()
                                    text = "Failed to read data source.\n"
                                    text += "Source: %s\n" % source.sourceName
                                    text += "Key: %s\n"  % sel['Key']
                                    text += "Error: %s" % error[1]
                                    if QTVERSION < '4.0.0':
                                        qt.QMessageBox.critical(self,
                                                                "%s" % error[0],
                                                                text)
                                    else:
                                        msg = qt.QMessageBox(self)
                                        msg.setWindowTitle('Source Error')
                                        msg.setIcon(qt.QMessageBox.Critical)
                                        msg.setInformativeText(text)
                                        msg.setDetailedText(\
                                            traceback.format_exc())
                                    continue
                        else:
                            dataObject = source.getDataObject(sel['Key'],
                                                      selection=sel['selection'], 
                                                      poll=False)
                            if dataObject is not None:
                                dataObject.info['legend'] = sel['legend']
                                dataObject.info['targetwidgetid'] = targetwidgetid
                                source.addToPoller(dataObject)
                            else:
                                #this may happen on deletion??
                                return
                        ddict['dataobject'] = dataObject
                        selectionList.append(ddict)
                    else:
                        #this creates a weak reference to the source object
                        #the clients will be able to retrieve the data
                        #the problem is that 10 clients will requiere
                        #10 read outs
                        ddict["sourcereference"] = weakref.ref(source)
                        selectionList.append(ddict)
                    self.emit(qt.SIGNAL(event), selectionList)

    def _removeSelectionSlot(self, sel_list):
        if DEBUG:
            print("_removeSelectionSlot")
            print("sel_list = ",sel_list)
        for sel in sel_list:
            ddict = {}
            ddict.update(sel)
            ddict["event"] = "removeSelection"
            self.emit(qt.SIGNAL("removeSelection"), ddict)

    def _replaceSelectionSlot(self, sel_list):
        if DEBUG:
            print("_replaceSelectionSlot")
            print("sel_list = ",sel_list)

        if len(sel_list) == 1:
            self._addSelectionSlot([sel_list[0]], event = "replaceSelection")
        elif len(sel_list) > 1:
            self._addSelectionSlot([sel_list[0]], event = "replaceSelection")
            self._addSelectionSlot(sel_list[1:], event = "addSelection")

    def _otherSignalsSlot(self, ddict):
        self.emit(qt.SIGNAL("otherSignals"), ddict)

    def _sourceSelectorSlot(self, ddict):
        if DEBUG:
            print("_sourceSelectorSlot(self, ddict)")
            print("ddict = ",ddict)
        if ddict["event"] == "NewSourceSelected":
            source = QDataSource.QDataSource(ddict["sourcelist"])
            self.sourceList.append(source)
            sourceType = source.sourceType
            self.selectorWidget[sourceType].setDataSource(source)
            self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
            if sourceType == "SPS":
                source.sigUpdated.connect(self._selectionUpdatedSlot)

        elif (ddict["event"] == "SourceSelected") or \
             (ddict["event"] == "SourceReloaded"):
            found = 0
            for source in self.sourceList:
                if source.sourceName == ddict["sourcelist"]:
                    found = 1
                    break
            if not found:
                if DEBUG:
                    print("WARNING: source not found")
                return
            sourceType = source.sourceType
            if ddict["event"] == "SourceReloaded":
                source.refresh()
            self.selectorWidget[sourceType].setDataSource(source)
            self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
        elif ddict["event"] == "SourceClosed":
            found = 0
            for source in self.sourceList:
                if source.sourceName == ddict["sourcelist"]:
                    found = 1
                    break
            if not found:
                if DEBUG:
                    print("WARNING: source not found")
                return
            sourceType = source.sourceType
            del self.sourceList[self.sourceList.index(source)]
            for source in self.sourceList:
                if sourceType == source.sourceType:
                    self.selectorWidget[sourceType].setDataSource(source)
                    self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
                    return
            #there is no other selection of that type
            if len(self.sourceList):
                source = self.sourceList[0]
                sourceType = source.sourceType
                self.selectorWidget[sourceType].setDataSource(source)
            else:
                self.selectorWidget[sourceType].setDataSource(None)
            self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
        elif ddict["event"] == "SourceClosed":
            if DEBUG:
                print("not implemented yet")


    def _selectionUpdatedSlot(self, ddict):
        if DEBUG:
            print("_selectionUpdatedSlot(self, dict)",ddict)
        if 'selectionlist' in ddict:
            sel_list = ddict['selectionlist']
        else:
            sel_list = []
            for objectReference in ddict["id"]:
                targetwidgetid = ddict.get('targetwidgetid', None)
                if targetwidgetid not in [None, id(self)]:
                    continue
                sel = {}
                sel['SourceName'] = ddict['SourceName']
                sel['SourceType'] = ddict['SourceType']
                sel['Key']        = ddict['Key']
                if 0:
                    sel['selection']  = objectReference.info['selection']
                    sel['legend']     = objectReference.info['legend']
                    if 'scanselection' in objectReference.info.keys():
                        sel['scanselection']  = objectReference.info['scanselection']
                else:
                    sel['selection']  = ddict['selection']
                    sel['legend']     = ddict['legend']
                    sel['scanselection']  = ddict['scanselection']
                    sel['imageselection']  = ddict['imageselection']
                sel_list.append(sel)            
        self._addSelectionSlot(sel_list)

    def _tabChanged(self, value):
        if DEBUG:
            print("self._tabChanged(value), value =  ",value)
        text = str(self.tabWidget.tabText(value))
        ddict = {}
        ddict['SourceType'] = text
        if self.selectorWidget[text].data is not None:
            ddict['SourceType'] = self.selectorWidget[text].data.sourceType
            ddict['SourceName'] = self.selectorWidget[text].data.sourceName
        else:
            ddict['SourceName'] = None
        ddict['event'] = "SourceTypeChanged"
        self.emit(qt.SIGNAL("otherSignals"), ddict)

    def _pluginsClicked(self):
        ddict = {}
        value = self.tabWidget.currentIndex()
        text = str(self.tabWidget.tabText(value))
        ddict['SourceType'] = text
        if self.selectorWidget[text].data is not None:
            ddict['SourceType'] = self.selectorWidget[text].data.sourceType
            ddict['SourceName'] = self.selectorWidget[text].data.sourceName
        else:
            ddict['SourceName'] = None
        print(ddict)
        print("===========================")
        for source in self.sourceList:
            print(source)
            print(source.sourceType)
            sourceType = source.sourceType
            print(self.selectorWidget[sourceType].currentSelectionList())

        if self.pluginsCallback is not None:
            self.pluginsCallback(info)


def test():
    app = qt.QApplication([])
    w = QDispatcher()
    w.show()
    app.lastWindowClosed.connect(app.quit)
    app.exec_()
        
if __name__ == "__main__":
    test()
