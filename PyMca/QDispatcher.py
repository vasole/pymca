###########################################################################
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
#############################################################################
import sys
from QSourceSelector import qt
QTVERSION = qt.qVersion()
import QSourceSelector
import QDataSource
import os
#import weakref

DEBUG = 0

class QDispatcher(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.sourceList = []
        fileTypeList = ["Spec Files (*mca)",
                        "Spec Files (*dat)",
                        "Spec Files (*spec)",
                        "SPE Files (*SPE *spe)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "CSV Files (*csv)"]
        if QDataSource.NEXUS:
            fileTypeList.append("HDF5 Files (*.nxs *.hdf *.h5 *.hdf5)")
        fileTypeList.append("All Files (*)")
        
        self.sourceSelector = QSourceSelector.QSourceSelector(self, filetypelist=fileTypeList)
        self.selectorWidget = {}
        self.tabWidget = qt.QTabWidget(self)
        
        #for the time being just files
        for src_widget in QDataSource.source_widgets.keys():
            self.selectorWidget[src_widget] = QDataSource.source_widgets[src_widget]()
            self.tabWidget.addTab(self.selectorWidget[src_widget], src_widget)
            if QTVERSION < '4.0.0':
                self.connect(self.selectorWidget[src_widget],
                             qt.PYSIGNAL("addSelection"),
                             self._addSelectionSlot)
                self.connect(self.selectorWidget[src_widget],
                             qt.PYSIGNAL("removeSelection"),
                             self._removeSelectionSlot)
                self.connect(self.selectorWidget[src_widget],
                             qt.PYSIGNAL("replaceSelection"),
                             self._replaceSelectionSlot)
                if src_widget not in ['EdfFile']:
                    self.connect(self.selectorWidget[src_widget],
                             qt.PYSIGNAL("otherSignals"),
                             self._otherSignalsSlot)
            else:
                self.connect(self.selectorWidget[src_widget],
                             qt.SIGNAL("addSelection"),
                             self._addSelectionSlot)                                                 
                self.connect(self.selectorWidget[src_widget],
                             qt.SIGNAL("removeSelection"),
                             self._removeSelectionSlot)
                self.connect(self.selectorWidget[src_widget],
                             qt.SIGNAL("replaceSelection"),
                             self._replaceSelectionSlot)
                if src_widget not in ['EdfFile']:
                    self.connect(self.selectorWidget[src_widget],
                             qt.SIGNAL("otherSignals"),
                             self._otherSignalsSlot)
        
        self.mainLayout.addWidget(self.sourceSelector)
        self.mainLayout.addWidget(self.tabWidget)
        if QTVERSION < '4.0.0':
            self.connect(self.sourceSelector, 
                    qt.PYSIGNAL("SourceSelectorSignal"), 
                    self._sourceSelectorSlot)
        else:
            self.connect(self.sourceSelector, 
                    qt.SIGNAL("SourceSelectorSignal"), 
                    self._sourceSelectorSlot)
            self.connect(self.tabWidget,
                         qt.SIGNAL('currentChanged(int)'),
                         self._tabChanged)

    def _addSelectionSlot(self, sel_list, event=None):
        if DEBUG:
            print("_addSelectionSlot")
            print("sel_list = ",sel_list)

        if event is None:event = "addSelection"
        for sel in sel_list:
            #The dispatcher should be a singleton to work properly
            #implement a patch
            targetwidgetid = sel.get('targetwidgetid', None)
            if targetwidgetid not in [None, id(self)]:
                continue
            #find the source
            sourcelist = sel['SourceName']
            for source in self.sourceList:
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
                                    qt.QMessageBox.critical(self,"%s" % error[0], text)
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
                        if QTVERSION < '4.0.0':
                            self.emit(qt.PYSIGNAL(event), (ddict,))
                        else:
                            self.emit(qt.SIGNAL(event), ddict)
                    else:
                        #this creates a weak reference to the source object
                        #the clients will be able to retrieve the data
                        #the problem is that 10 clients will requiere
                        #10 read outs
                        ddict["sourcereference"] = weakref.ref(source)
                        if QTVERSION < '4.0.0':
                            self.emit(qt.PYSIGNAL(event), (ddict,))
                        else:
                            self.emit(qt.SIGNAL(event), ddict)

    def _removeSelectionSlot(self, sel_list):
        if DEBUG:
            print("_removeSelectionSlot")
            print("sel_list = ",sel_list)
        for sel in sel_list:
            ddict = {}
            ddict.update(sel)
            ddict["event"] = "removeSelection"
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("removeSelection"), (ddict,))
            else:
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
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("otherSignals"), (ddict,))
        else:
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
            if QTVERSION < '4.0.0':
                index = self.tabWidget.indexOf(self.selectorWidget[sourceType])
                self.tabWidget.setCurrentPage(index)  
            else:
                self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
            if sourceType == "SPS":
                if QTVERSION < '4.0.0':
                    self.connect(source, qt.PYSIGNAL("updated"),
                                        self._selectionUpdatedSlot)
                else:                                                 
                    self.connect(source, qt.SIGNAL("updated"),
                                        self._selectionUpdatedSlot)

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
            if QTVERSION < '4.0.0':
                index = self.tabWidget.indexOf(self.selectorWidget[sourceType])
                self.tabWidget.setCurrentPage(index)  
            else:
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
                    if QTVERSION < '4.0.0':
                        index = self.tabWidget.indexOf(self.selectorWidget[sourceType])
                        self.tabWidget.setCurrentPage(index)  
                    else:
                        self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
                    return
            #there is no other selection of that type
            if len(self.sourceList):
                source = self.sourceList[0]
                sourceType = source.sourceType
                self.selectorWidget[sourceType].setDataSource(source)
            else:
                self.selectorWidget[sourceType].setDataSource(None)
            if QTVERSION < '4.0.0':
                index = self.tabWidget.indexOf(self.selectorWidget[sourceType])
                self.tabWidget.setCurrentPage(index)  
            else:
                self.tabWidget.setCurrentWidget(self.selectorWidget[sourceType])
        elif ddict["event"] == "SourceClosed":
            if DEBUG:
                print("not implemented yet")


    def _selectionUpdatedSlot(self, ddict):
        if DEBUG:
            print("_selectionUpdatedSlot(self, dict)",ddict)
        if ddict.has_key('selectionlist'):
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
        if QTVERSION < '4.0.0':
            pass
        else:
            text = str(self.tabWidget.tabText(value))
        ddict = {}
        ddict['SourceType'] = text
        if self.selectorWidget[text].data is not None:
            ddict['SourceType'] = self.selectorWidget[text].data.sourceType
            ddict['SourceName'] = self.selectorWidget[text].data.sourceName
        else:
            ddict['SourceName'] = None
        ddict['event'] = "SourceTypeChanged"
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("otherSignals"), (ddict,))
        else:
            self.emit(qt.SIGNAL("otherSignals"), ddict)


def test():
    app = qt.QApplication([])
    w = QDispatcher()
    w.show()
    qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()
        
if __name__ == "__main__":
    test()
