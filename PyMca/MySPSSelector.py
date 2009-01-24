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
__revision__ = "$Revision: 1.4 $"
#import EventHandler
import spswrap as sps
#from qt import *
import qt
import PyMca_Icons as icons
DEBUG = 0
PYDVT = 0
SOURCE_TYPE = 'SPS'


class SPSMcaArrayWidget(qt.QWidget):
    def __init__(self, parent=None, name="SPS_MCA_DATA", fl=0, title="MCA", size=(0,8192)):
        qt.QWidget.__init__(self, parent, name, fl)

        layout= qt.QGridLayout(self, 5, 2)
        layout.setMargin(5)

        self.title= qt.QLabel(self)
        font= self.title.font()
        font.setBold(1)
        self.title.setFont(font)
        layout.addMultiCellWidget(self.title, 0, 0, 0, 1, qt.Qt.AlignCenter)
        layout.addRowSpacing(0, 40)
        if 0:
            self.limitCheck= qt.QCheckBox("Limit spectrum size to:", self)
            layout.addMultiCellWidget(self.limitCheck, 1, 1, 0, 1, qt.Qt.AlignLeft)

            text=qt.QLabel("First", self)
            layout.addWidget(text, 2, 0, qt.Qt.AlignRight)
            text=qt.QLabel("Last", self)
            layout.addWidget(text, 3, 0, qt.Qt.AlignRight)

            self.firstSpin= qt.QSpinBox(self)
            self.lastSpin= qt.QSpinBox(self)
            layout.addWidget(self.firstSpin, 2, 1, qt.Qt.AlignLeft)
            layout.addWidget(self.lastSpin, 3, 1,qt.Qt.AlignLeft)

            self.firstSpin.setMinValue(0)
            self.lastSpin.setMinValue(0)

            self.setSize(size)
        self.setTitle(title)
        
    def setInfo(self, info):
	self.setSize(info["rows"], info["cols"])
	self.setTitle(info["Key"])

    def setSize(self,rows,cols,selsize=None):
        self.rows= rows
        self.cols= cols
        if self.cols<=self.rows:
            self.idx='cols'
        else:
            self.idx='rows'

    """
    def setSize(self, maxsize, selsize=None):
        self.firstSpin.setMaxValue(maxsize[0])
        self.lastSpin.setMaxValue(maxsize[1])
        if selsize is not None:
            self.firstSpin.setValue(selsize[0])
            self.lastSpin.setValue(selsize[1])
        else:
            self.firstSpin.setValue(maxsize[0])
            self.lastSpin.setValue(maxsize[1])
    """
    def setTitle(self, title):
        self.title.setText("%s"%title)

    def getSelection(self):
        if 0:
            first= self.firstSpin.value()
            last= self.lastSpin.value()
            keys= {"plot":0, "x":0, "y":1}
            if self.limitCheck.isChecked():
                first= self.firstSpin.value()
                last= self.lastSpin.value()
                keys["limits"]= (first, last)
        else:
            keys = {"plot":self.idx,"x":0,"y":1}
        return [keys]

class SPSXiaArrayWidget(qt.QWidget):
    def __init__(self, parent=None, name="SPS_XIA_DATA", fl=0, title="XIA", size=(0,8192)):
        qt.QWidget.__init__(self, parent, name, fl)

        layout= qt.QGridLayout(self, 2, 1)
        layout.setMargin(5)

        self.title= qt.QLabel(self)
        font= self.title.font()
        font.setBold(1)
        self.title.setFont(font)
        self.title.setText(title)

	self.detList= qt.QListBox(self)
        self.detList.setSelectionMode(qt.QListBox.Multi)

        layout.addWidget(self.title, 0, 0, qt.Qt.AlignCenter)
        layout.addRowSpacing(0, 40)
        layout.addWidget(self.detList, 1, 0)

    def setTitle(self, title):
        self.title.setText("%s"%title)

    def setInfo(self, info):
	self.setSize(info["rows"], info["cols"], info.get("Detectors", None))
	self.setTitle(info["Key"])

    def setSize(self, rows, cols, dets=None):
        self.rows= rows
        self.cols= cols

        if dets is None or len(dets)!=rows:
            dets= range(self.rows)

        self.detList.clear()
        for idx in range(1, self.rows):
            self.detList.insertItem("Detector %d"%dets[idx])

    def getSelection(self):
        selection= []
        ylist= [ idx for idx in range(self.detList.count()) if self.detList.isSelected(idx) ]
        for y in ylist:
            selection.append({"plot":"rows", "x":0, "y":y+1})
        return selection
 
class SPS_StandardArray(qt.QWidget):
    def __init__(self, parent=None, name="SPS_StandardArray", fl=0, rows=0, cols=0):
        qt.QWidget.__init__(self, parent, name, fl)
        layout= qt.QGridLayout(self, 4, 2)
        layout.setMargin(5)

        plab= qt.QLabel("Plot", self)
        xlab= qt.QLabel("X :", self)
        ylab= qt.QLabel("Y :", self)

        layout.addWidget(plab, 0, 0, qt.Qt.AlignRight)
        layout.addWidget(xlab, 1, 0, qt.Qt.AlignRight)
        layout.addWidget(ylab, 2, 0, qt.Qt.AlignRight|qt.Qt.AlignTop)

        self.plotCombo= qt.QComboBox(self)
        self.plotCombo.setEditable(0)
        self.plotCombo.insertItem("Rows")
        self.plotCombo.insertItem("Columns")

        self.xCombo= qt.QComboBox(self)
        self.xCombo.setEditable(0)

        self.yList= qt.QListBox(self)
        self.yList.setSelectionMode(qt.QListBox.Multi)

        layout.addWidget(self.plotCombo, 0, 1)
        layout.addWidget(self.xCombo, 1, 1)
        layout.addWidget(self.yList, 2, 1)

        self.connect(self.plotCombo, qt.SIGNAL("activated(int)"), self.__plotChanged)

        self.setSize(rows, cols)

    def setSize(self, rows, cols):
        self.rows= rows
        self.cols= cols

        idx= self.cols<=self.rows
        self.plotCombo.setCurrentItem(idx)
        self.__plotChanged(idx)

    def __plotChanged(self, index):
        if index==1:
            txt= "Column"
            val= self.cols
        else:
            txt= "Row"
            val= self.rows
        self.xCombo.clear()
        self.xCombo.insertItem("Array Index")
        self.yList.clear()
        for x in range(val):
            self.xCombo.insertItem("%s %d"%(txt,x))
            self.yList.insertItem("%s %d"%(txt,x))
        if val==2:
            self.xCombo.setCurrentItem(0)
            self.__xChanged(0)

    def __xChanged(self, index):
        pass

    def getSelection(self):
        selection= []

        idx= self.plotCombo.currentItem()
        if idx==1: plot= "cols"
        else: plot= "rows"

        idx= self.xCombo.currentItem()
        if idx==0: x= None
        else: x= idx-1

        ylist= [ idx for idx in range(self.yList.count()) if self.yList.isSelected(idx) ]
        for y in ylist:
            selection.append({"plot":plot, "x":x, "y":y})
        return selection

class SPSSelector(qt.QWidget):
    HiddenArrays= ["SCAN_D", "MCA_DATA_PARAM", "XIA_STAT", "XIA_DET"]
    WidgetArrays= {"xia": SPSXiaArrayWidget, "mca": SPSMcaArrayWidget, "array": SPS_StandardArray, "empty": qt.QWidget}
    TypeArrays= {"MCA_DATA": "mca", "XIA_PLOT": "mca", "XIA_DATA": "xia", "XIA_BASELINE":"xia"}

    def __init__(self, parent=None, name="SPSSelector", fl=0):
        qt.QWidget.__init__(self, parent, name, fl)

        self.dataSource= None

        self.data= None
        self.currentSpec= None
        self.currentArray= None
        self.selection= None
        self.openFile = self.refreshSpecList

        self.selectPixmap= qt.QPixmap(icons.selected)
        self.unselectPixamp= qt.QPixmap(icons.unselected)

        mainLayout= qt.QVBoxLayout(self)

        # --- spec name selection
        specWidget= qt.QWidget(self)
        self.specCombo= qt.QComboBox(specWidget)
        self.specCombo.setEditable(0)
        self.reload= qt.QIconSet(qt.QPixmap(icons.reload))
        refreshButton= qt.QToolButton(specWidget)
        refreshButton.setIconSet(self.reload)
        refreshButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        self.closeIcon= qt.QIconSet(qt.QPixmap(icons.fileclose))
        closeButton= qt.QToolButton(specWidget)
        closeButton.setIconSet(self.closeIcon)
        closeButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        specLayout= qt.QHBoxLayout(specWidget)
        specLayout.addWidget(self.specCombo)
        specLayout.addWidget(refreshButton)
        specLayout.addWidget(closeButton)

        self.connect(refreshButton, qt.SIGNAL("clicked()"), self.refreshSpecList)
        self.connect(closeButton,qt.SIGNAL("clicked()"), self.closeCurrentSpec)
        self.connect(self.specCombo, qt.SIGNAL("activated(const QString &)"), self.refreshArrayList)

        # --- splitter
        self.splitter= qt.QSplitter(self)
        self.splitter.setOrientation(qt.QSplitter.Vertical)

        # --- shm array list
        self.arrayList= qt.QListView(self.splitter, "ShmArrayList")
        self.arrayList.addColumn("")
        self.arrayList.addColumn("Array Name")
        self.arrayList.addColumn("Rows")
        self.arrayList.addColumn("Cols")
        self.arrayList.setSorting(-1)
        self.arrayList.header().setClickEnabled(0,-1)
        self.arrayList.setAllColumnsShowFocus(1)
        self.arrayList.setSelectionMode(qt.QListView.Single)
        
        self.connect(self.arrayList, qt.SIGNAL("selectionChanged()"), self.__arraySelection)

        # --- array parameter
        self.paramIndex= {}
        self.paramWidget= qt.QWidgetStack(self.splitter)
        for type in self.WidgetArrays.keys():
            widclass= self.WidgetArrays[type]
            wid= widclass(self.paramWidget)
            self.paramWidget.addWidget(wid)
            self.paramIndex[type]= self.paramWidget.id(wid)

        # --- command buttons
        butWidget= qt.QWidget(self)
        butWidget.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum))
        addButton= qt.QPushButton("Add", butWidget)
        removeButton= qt.QPushButton("Remove", butWidget)
        replaceButton= qt.QPushButton("Replace", butWidget)

        butLayout= qt.QHBoxLayout(butWidget)
        butLayout.addWidget(removeButton)
        butLayout.addWidget(replaceButton)
        butLayout.addWidget(addButton)
        butLayout.setMargin(5)

        self.connect(addButton, qt.SIGNAL("clicked()"), self.__addClicked)
        self.connect(replaceButton, qt.SIGNAL("clicked()"), self.__replaceClicked)
        self.connect(removeButton, qt.SIGNAL("clicked()"), self.__removeClicked)

        # --- main layout
        mainLayout.setMargin(5)
        mainLayout.setSpacing(5)
        mainLayout.addWidget(specWidget)
        mainLayout.addWidget(self.splitter)
        mainLayout.addWidget(butWidget)

    def setData(self,data=None):
        if DEBUG:
            print "setData(self, data) called"
            print "spec data = ",data
        self.data= data
        self.refreshSpecList()
        self.refreshDataSelection()

    def refreshSpecList(self):
        speclist= sps.getspeclist()
        if self.specCombo.count():
            selected= str(self.specCombo.currentText())
        else:    selected= None
        self.specCombo.clear()
        if len(speclist):
            for spec in speclist:
                self.specCombo.insertItem(spec)
            self.selectSpec(selected or speclist[0])

    def selectSpec(self, specname=None):
        for idx in range(self.specCombo.count()):
            if str(self.specCombo.text(idx))==specname:
                self.specCombo.setCurrentItem(idx)

    def __getCurrentSpec(self):
        if self.specCombo.count():
            return str(self.specCombo.currentText())
        else:    return None

    def refreshDataSelection(self, source=None):
        spec= self.__getCurrentSpec()
        if spec is not None and self.dataSource is not None:
            arraylist= self.dataSource.GetDataList(spec)
            item= self.arrayList.firstChild()
            while item is not None:
                name= str(item.text(1))
                if name in arraylist:    item.setPixmap(0, self.selectPixmap)
                else:            item.setPixmap(0, self.unselectPixmap)
            item= item.nextSibling()

    def closeCurrentSpec(self):
        spec= self.__getCurrentSpec()
        if spec is not None and self.dataSource is not None:
            arraylist= self.DataSource.GetDataList(spec)
            if len(arraylist):
                msg= "%d spectrums are linked to that SPEC source.\n"%(len(arraylist))
                msg+= "Do you really want to delete all these spectrums ??"
                ans= qt.QMessageBox.information(self, "Remove Spec Shared %s"%spec, msg, \
                        qt.QMessageBox.No, qt.QMessageBox.Yes)
                if ans.qt.QMessageBox.Yes:
                    self.dataSource.RemoveData(spec)

    def refreshArrayList(self,qstring):
        self.arrayList.clear()
        #spec= self.__getCurrentSpec()
        self.currentSpec = str(qstring)
        spec = self.currentSpec
        if spec is not None:
            arraylist= {}
            for array in sps.getarraylist(spec):
                if array not in self.HiddenArrays:
                    info= sps.getarrayinfo(spec, array)
                    rows= info[0]
                    cols= info[1]
                    type= info[2]
                    flag= info[3]
                    if flag in (sps.IS_ARRAY, sps.IS_MCA, sps.IS_IMAGE) and type!=sps.STRING:
                        arraylist[array]= (rows, cols)
            if len(arraylist.keys()):
                arrayorder= arraylist.keys()
                arrayorder.sort()
                arrayorder.reverse()
                for name in arrayorder:
                    self.arrayList.insertItem(qt.QListViewItem(self.arrayList, 
                        "", name, str(arraylist[name][0]), str(arraylist[name][1])))
                self.refreshDataSelection()
            
        self.__getParamWidget("empty")

    def __arraySelection(self):
        item= self.arrayList.selectedItem()
        if item is not None:
            self.currentArray= str(item.text(1))
        else:
            #click on empty space
            return
        self.data.SetSource(self.currentSpec)
        self.data.LoadSource(self.currentArray)
        info= self.data.GetSourceInfo(self.currentArray)
        wid= None
        for (array, type) in self.TypeArrays.items():
            if self.currentArray[0:len(array)]==array:
                wid= self.__getParamWidget(type)
                wid.setInfo(info)
        if wid is None:
            wid= self.__getParamWidget("array")
            wid.setSize(info["rows"], info["cols"])

    def __getParamWidget(self, widtype):
        wid= self.paramWidget.visibleWidget()
        if self.paramWidget.id(wid)!=self.paramIndex[widtype]:
            self.paramWidget.raiseWidget(self.paramIndex[widtype])
            wid= self.paramWidget.visibleWidget()
        return wid

    def __replaceClicked(self):
        if DEBUG:
            print "replace clicked"
        selkeys= self.__getSelectedKeys()
        if len(selkeys):
            #self.eh.event(self.repEvent, selkeys)
            if DEBUG:
                print "Replace event"
            sel = {}
            sel['SourceType'] = SOURCE_TYPE            
            for selection in selkeys:
                if not sel.has_key('SourceName'):
                    sel['SourceName'] = selection['SourceName']
                arrayname = selection['Key']
                if not sel.has_key('Key'):
                    sel['Key'] = selection['Key']
                if not sel.has_key(arrayname):
                    sel[arrayname] = {'rows':[],'cols':[]}
                if selection['plot'] == 'cols':
                     sel[arrayname]['cols'].append({'x':selection['x'],'y':selection['y']})
                if selection['plot'] == 'rows':
                     sel[arrayname]['rows'].append({'x':selection['x'],'y':selection['y']})                              
                """
                if selection['plot'] == 0:
                     sel[arrayname]['mca'].append({'x':selection['x'],'y':selection['y']})
                """                              
            self.setSelected([sel],reset=1)
            self.emit(qt.PYSIGNAL("replaceSelection"), ([sel],))

    def __addClicked(self):
        if DEBUG:
            print "select clicked"
        selkeys= self.__getSelectedKeys()
        if DEBUG:
            print "selected keys = ",selkeys 
        if len(selkeys):
            #self.eh.event(self.addEvent, selkeys)
            if DEBUG:
                print "Select event"
            sel = {}
            sel['SourceType'] = SOURCE_TYPE            
            for selection in selkeys:
                if not sel.has_key('SourceName'):
                    sel['SourceName'] = selection['SourceName']
                arrayname = selection['Key']
                if not sel.has_key('Key'):
                    sel['Key'] = selection['Key']
                if not sel.has_key(arrayname):
                    sel[arrayname] = {'rows':[],'cols':[]}
                if selection['plot'] == 'cols':
                     sel[arrayname]['cols'].append({'x':selection['x'],'y':selection['y']})
                if selection['plot'] == 'rows':
                     sel[arrayname]['rows'].append({'x':selection['x'],'y':selection['y']})                              
            if self.selection is None: 
                self.setSelected([sel],reset=1)
            else:
                self.setSelected([sel],reset=0)
            self.emit(qt.PYSIGNAL("addSelection"), ([sel],))
            
    def __getSelectedKeys(self):
        selkeys= []
        parwid= self.paramWidget.visibleWidget()
        if self.currentArray is not None:
            for sel in parwid.getSelection():
                sel["SourceName"]= self.currentSpec
                sel['SourceType'] = SOURCE_TYPE            
                sel["Key"]= self.currentArray
                selkeys.append(sel)
        return selkeys

    def __removeClicked(self):
        if DEBUG:
            print "remove clicked"
        selkeys= self.__getSelectedKeys()
        if len(selkeys):
            #self.eh.event(self.delEvent, selkeys)
            if DEBUG:
                print "Remove Event"
                print "self.selection before = ",self.selection
            returnedselection=[]
            for selection in selkeys:
                sel = {}
                sel['SourceName'] = selection['SourceName']
                sel['SourceType'] = SOURCE_TYPE            
                sel['Key'] = selection['Key']
                arrayname = selection['Key']
                sel[arrayname] = {'rows':[],'cols':[]}
                if selection['plot'] == 'cols':
                     sel[arrayname]['cols'].append({'x':selection['x'],'y':selection['y']})
                if selection['plot'] == 'rows':
                     sel[arrayname]['rows'].append({'x':selection['x'],'y':selection['y']})
                returnedselection.append(sel)
                if self.selection is not None:
                    if DEBUG:
                        print "step 1"
                    if self.selection.has_key(sel['SourceName']):
                        if DEBUG:
                            print "step 2"
                        if self.selection[sel['SourceName']].has_key(arrayname):
                            if DEBUG:
                                print "step 3"
                            if self.selection[sel['SourceName']][arrayname].has_key('rows'):
                                if DEBUG:
                                    print "step 4"
                                for couple in  sel[arrayname]['rows']:
                                    if couple in  self.selection[sel['SourceName']][arrayname]['rows']:
                                        index= self.selection[sel['SourceName']][arrayname]['rows'].index(couple)
                                        del self.selection[sel['SourceName']][arrayname]['rows'][index]  
                                for couple in  sel[arrayname]['cols']:
                                    if couple in  self.selection[sel['SourceName']][arrayname]['cols']:
                                        index= self.selection[sel['SourceName']][arrayname]['cols'].index(couple)
                                        del self.selection[sel['SourceName']][arrayname]['cols'][index]
                                seln = {}
                                seln['SourceName'] = sel['SourceName'] 
                                seln['SourceType'] = SOURCE_TYPE            
                                seln['Key']        = sel['Key']
                                seln[seln['Key']]  = self.selection[seln['SourceName']][seln['Key']]
                                self.setSelected([seln],reset=0)
            self.emit(qt.PYSIGNAL("removeSelection"), (returnedselection,))
            
    def removeSelection(self,selection):
        if type(selection) != type([]):
            selection=[selection]
        for sel in selection:
                arrayname = sel['Key']                
                if self.selection is not None:
                    if DEBUG:
                        print "step 1"
                    if self.selection.has_key(sel['SourceName']):
                        if DEBUG:
                            print "step 2"
                        if self.selection[sel['SourceName']].has_key(arrayname):
                            if DEBUG:
                                print "step 3"
                            if self.selection[sel['SourceName']][arrayname].has_key('rows'):
                                if DEBUG:
                                    print "step 4"
                                for couple in  sel[arrayname]['rows']:
                                    if couple in  self.selection[sel['SourceName']][arrayname]['rows']:
                                        index= self.selection[sel['SourceName']][arrayname]['rows'].index(couple)
                                        del self.selection[sel['SourceName']][arrayname]['rows'][index]  
                                for couple in  sel[arrayname]['cols']:
                                    if couple in  self.selection[sel['SourceName']][arrayname]['cols']:
                                        index= self.selection[sel['SourceName']][arrayname]['cols'].index(couple)
                                        del self.selection[sel['SourceName']][arrayname]['cols'][index]
                                seln = {}
                                seln['SourceName'] = sel['SourceName'] 
                                seln['SourceType'] = SOURCE_TYPE            
                                seln['Key']        = sel['Key']
                                seln[seln['Key']]  = self.selection[seln['SourceName']][seln['Key']]
                                self.setSelected([seln],reset=0)
        self.emit(qt.PYSIGNAL("removeSelection"), (selection,))



                             
    def setSelected(self,sellist,reset=1):
        if DEBUG:
            print "setSelected(self,sellist,reset=1) called"
            print "sellist = ",sellist
            print "selection before = ",self.selection
            print "reset = ",reset
        if reset:
            self.selection = {}
        elif self.selection is None:
            self.selection = {}
        for sel in sellist:
            specname = sel['SourceName']
            #selkey is the array name what to do if multiple array names?
            if type(sel["Key"]) == type([]):
                selkey = sel["Key"][0]
            else:
                selkey = sel["Key"]
            if not self.selection.has_key(specname):
                self.selection[specname]= {}
            if not self.selection[specname].has_key(selkey):
                self.selection[specname][selkey] = {'rows':[],'cols':[]}
            if sel[selkey].has_key('rows'):
                for rowsel in sel[selkey]['rows']:
                    if rowsel not in self.selection[specname][selkey]['rows']:
                        self.selection[specname][selkey]['rows'].append(rowsel)   
            if sel[selkey].has_key('cols'):
                for rowsel in sel[selkey]['cols']:
                    if rowsel not in self.selection[specname][selkey]['cols']:
                        self.selection[specname][selkey]['cols'].append(rowsel)   
        if DEBUG:
            print "self.selection after = ",self.selection
        self.__refreshSelection()

    def getSelection(self):
        """
        Give the dicionary of dictionaries as an easy to understand list of
        individual selections
        """
        selection = []
        if self.selection is None: return selection
        for sourcekey in self.selection.keys():
            for arraykey in self.selection[sourcekey].keys():
                sel={}
                sel['SourceName']   = sourcekey
                sel['SourceType']   = 'SPS'
                sel['Key']          = arraykey
                sel[arraykey]        = self.selection[sourcekey][arraykey]
                selection.append(sel)
        return selection

        
    def __refreshSelection(self):
        return
        if DEBUG:
            print "__refreshSelection(self) called"
            print self.selection
        if self.selection is not None:
            sel = self.selection.get(self.data.SourceName, {})
            selkeys = []
            for key in sel.keys():
                if (sel[key]['mca'] != []) or (sel[key]['scan']['Ycnt'] !=  []):
                    selkeys.append(key)
            if DEBUG:
                print "selected scans =",selkeys,"but self.selection = ",self.selection
                print "and self.selection.get(self.data.SourceName, {}) =",sel
            self.scanList.markScanSelected(selkeys)
            scandict = sel.get(self.currentScan, {})
            if scandict.has_key('mca'):
                self.mcaTable.markMcaSelected(scandict['mca'])
            else:
                self.mcaTable.markMcaSelected([])
            if scandict.has_key('scan'):
                self.cntTable.markCntSelected(scandict['scan'])
            else:
                self.cntTable.markCntSelected({})
                
    def isSelectionUpdated(self,sellist):
        outsel = []
        if type(sellist) != type([]):
            sellist = [sellist]
        for dict in  sellist:
            #for dict in selection:
                if dict.has_key('SourceName'):
                    spec = dict['SourceName']
                    if dict.has_key('Key'):
                        shm  = dict['Key']
                        if dict.has_key(shm):
                            check = 0
                            rows = []
                            cols = []
                            if dict[shm].has_key('cols'):
                                cols = dict[shm]['cols']
                                if len(cols):
                                    check =  1
                            if dict[shm].has_key('rows'):
                                rows = dict[shm]['rows']
                                if len(rows):
                                    check =  1
                            if check and sps.specrunning(spec):
                                if sps.isupdated(spec,shm):
                                    outsel.append({'SourceName':spec,
                                                   'Key':shm,
                                                   shm:{'rows':rows,
                                                        'cols':cols},
                                                    'SourceType':'SPS'})
        return outsel                                    
                                        
                                        
        

def test():
    import sys
    if PYDVT:
        import SPSData
    else:
        import SPSLayer
    def repSelection(sel): print "repSelection", sel
    def addSelection(sel): print "addSelection", sel

    a= qt.QApplication(sys.argv)
    a.connect(a, qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))
    def repSelection(sel): print "replaceSelection", sel
    def removeSelection(sel): print "removeSelection", sel
    def addSelection(sel): print "addSelection", sel

    w= SPSSelector()
    qt.QObject.connect(w,qt.PYSIGNAL("addSelection"),addSelection)
    qt.QObject.connect(w,qt.PYSIGNAL("removeSelection"),removeSelection)
    qt.QObject.connect(w,qt.PYSIGNAL("replaceSelection"),repSelection)
    if PYDVT:
        d = SPSData.SPSData()
    else:
        d = SPSLayer.SPSLayer()
    w.setData(d)
    """
    w.eh.register("addSelection", addSelection)
    w.eh.register("repSelection", repSelection)
    """
    w.show()
    a.exec_loop()

if __name__=="__main__":
    test()
