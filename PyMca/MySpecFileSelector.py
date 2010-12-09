#/*##########################################################################
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
#############################################################################*/
__revision__ = "$Revision: 1.14 $"
import qt
if qt.qVersion() < '3.0.0':
    import Myqttable as qttable
else:
    import qttable
import types, os.path
import sys
import PyMca_Icons as icons
import string
import SpecFileDataInfo

DEBUG = 0
PYDVT = 0
SOURCE_TYPE = 'SpecFile'
class ScanList(qt.QWidget):

    allSelectionMode= {'single': qt.QListView.Single,
                        'multiple': qt.QListView.Multi,
                        'extended': qt.QListView.Extended
                        }

    allSelectionType= ["mca", "scan", "both"]

    def __init__(self, parent=None, name=None, fl=0, selection="single", type="both"):
        qt.QWidget.__init__(self, parent, name, fl)

        self.layout= qt.QVBoxLayout(self)
        self.list= qt.QListView(self, "ScanList")
        self.layout.addWidget(self.list)

        # --- font
        #self.setFont(qt.QFont("Application", 10))

        # --- list headers
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
        self.setSelectionMode(selection)
        self.disableMca    = (type=="scan")
        self.disableScan   = (type=="mca")

        # --- signal handling
        self.connect(self.list, qt.SIGNAL("selectionChanged()"), self.__selectionChanged)
        if qt.qVersion() > '3.0.0':
            self.connect(self.list, qt.SIGNAL("contextMenuRequested(QListViewItem *, const QPoint &, int)"), self.__contextMenu)
        else:
            self.connect(self.list, qt.SIGNAL("rightButtonPressed(QListViewItem *, const QPoint &, int)"), self.__contextMenu)
        self.connect(self.list, qt.SIGNAL("doubleClicked(QListViewItem *)"), self.__doubleClicked)


        # --- context menu
        self.menu= qt.QPopupMenu(self.list)
        id= self.menu.insertItem("Show scan header")
        self.menu.connectItem(id, self.__showScanInfo)

        self.data= None
        self.scans= []

    #
    # Widget options
    #
    def setSelectionMode(self, selection):
        if DEBUG:
            print("setSelectionMode(self, selection) called")
            print("selection = ",selection)
        if selection in self.allSelectionMode:
            self.list.setSelectionMode(self.allSelectionMode[selection])

    def setFont(self, qtfont):
        self.list.setFont(qtfont)

    # 
    # Data management
    #
    def setData(self, specfiledata):
        if DEBUG:
            print("setData(self, specfiledata) called")
            print("specfiledata = ",specfiledata)
        self.data= specfiledata
        self.refresh()

    def setDataSource(self, specfiledata):
        self.data= specfiledata
        self.data.SourceName = specfiledata.sourceName[0]
        self.data.GetSourceInfo = self.data.getSourceInfo
        self.refresh()

    def refresh(self):
        self.list.clear()
        if self.data is None or self.data.SourceName is None:    return
        info= self.data.GetSourceInfo()
        self.scans= []
        after= None
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

    def clear(self):
        self.list.clear()
        self.data= None
        self.scans= []

    def markScanSelected(self, scanlist):
        if qt.qVersion() > '3.0.0':
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
    #
    # signal/slot handling
    #
    def __selectionChanged(self):
        if qt.qVersion() > '3.0.0':
            sel= [sn for sn in self.scans if self.list.findItem(sn,1).isSelected()]
        else:
            sel = []
            item = self.list.firstChild()
            while item:
                if item.isSelected():
                    sel.append(str(item.text(1)))
                item=item.nextSibling()
        self.emit(qt.PYSIGNAL("scanSelection"), (sel,))

    def __doubleClicked(self, item):
        if item is not None:
            sn  = str(item.text(1))
            dict={}
            dict['Key']      = sn
            dict['Command']  = str(item.text(2))
            dict['NbPoints'] = int(str(item.text(3)))
            dict['NbMca']    = int(str(item.text(4)))
            self.emit(qt.PYSIGNAL("scanDoubleClicked"), (dict,))

        
    def __contextMenu(self, item, point, col=None):
        if item is not None:
            sn= str(item.text(1))
            self.menu.setItemParameter(self.menu.idAt(0), self.scans.index(sn))
            self.menu.popup(point)

    def __showScanInfo(self, idx):
        if DEBUG:
            print("Scan information:")
            print(self.data.GetSourceInfo(self.scans[idx]))
        info, data = self.data.LoadSource(self.scans[idx])
        self.dataInfoWidget= SpecFileDataInfo.SpecFileDataInfo(info)
        self.dataInfoWidget.show()



class McaTable(qt.QWidget):
    def __init__(self, parent=None, name=None, fl=0):
        qt.QWidget.__init__(self, parent, name, fl)

        self.mainLayout= qt.QVBoxLayout(self)

        self.table= qttable.QTable(self)
        self.table.setSelectionMode(qttable.QTable.Multi)
        if qt.qVersion() >= '3.0.0':
            self.table.setFocusStyle(qttable.QTable.FollowStyle)
            self.table.setReadOnly(1)
        else:
            if DEBUG:
                print("Methods to be implemented")
        self.table.verticalHeader().setResizeEnabled(0, -1)
        self.table.horizontalHeader().setResizeEnabled(0, -1)

        # --- selection in case of a single MCA by scan
        self.firstMca= qttable.QTableSelection()
        self.firstMca.init(0, 0)
        self.firstMca.expandTo(0, 0)

        self.mainLayout.addWidget(self.table)

        #self.connect(self.table, qt.SIGNAL("selectionChanged()"), self.__selectionChanged)
        # XXX: do not work correctly anymore !!! (after qt.Qt upgrade)
        self.connect(self.table, qt.SIGNAL("doubleClicked(int,int,int,const QPoint&)"), self.__doubleClicked)
        self.reset()

    #
    # data management
    #
    def setData(self, specfiledata):
        self.data= specfiledata
        self.reset()

    def setDataSource(self, specfiledata):
        self.data= specfiledata
        self.reset()

    def setScan(self, scankey):
        if type(scankey)==types.ListType: 
            if len(scankey): scankey= scankey[0]
            else: scankey=None
        if scankey is None:
            self.reset()
        else:
            self.info= self.data.GetSourceInfo(scankey)
            self.refresh()

    def markMcaSelected(self, mcalist):
        if DEBUG:
            print("markMcaSelected(self, mcalist) called")
            print("mcalist = ",mcalist)
        scankey = ""
        if mcalist != []:
            if len(mcalist):
                stringsplit=string.split(mcalist[0],".")
                if len(stringsplit) == 2:
                    scankey = ""
                else:
                    scankey = stringsplit[0]+"."+stringsplit[1]+"."
            for row in range(self.pts):
                for col in range(self.det):
                    if ("%s%d.%d" % (scankey,row+1, col+1)) in mcalist:
                        self.table.setText(row, col, "X")
                    else:
                        self.table.setText(row, col, "")
        else:
            for row in range(self.pts):
                for col in range(self.det):
                   self.table.setText(row, col, "") 
            
    def selectMcaList(self, mcalist):
        for mcano in mcalist:
            self.selectMcaNo(mcano)

    def selectMcaNo(self, mcano):
        (row, col)= self.__getRowCol(mcano)
        selection= qttable.QTableSelection()
        selection.init(row, col)
        selection.expandTo(row, col)
        self.table.addSelection(selection)
        #self.__selectionChanged()

    def selectFirstMca(self):
        self.table.addSelection(self.firstMca)
        #self.__selectionChanged()

    def refresh(self):
        mca= self.info["NbMca"] or 0
        pts= self.info["Lines"] or 0

        if mca==0: 
            self.reset()
        else:
            # --- try to compute number of detectors
            if pts>0 and mca%pts==0:
                self.det= mca/pts
                self.pts= pts
            else:
                self.det= mca
                self.pts= 1

            self.table.setNumCols(self.det)
            self.table.setNumRows(self.pts)
            Hheader= self.table.horizontalHeader()
            for idx in range(self.det):
                Hheader.setLabel(idx, "mca.%d"%(idx+1), -1)
                self.table.adjustColumn(idx)

            for row in range(self.pts):
                for col in range(self.det):
                    self.table.setText(row, col, "")

            #if self.info["Lines"]==pts:
            #    print("AddNorm", self.info["AllLabels"])

            if mca==1:
                self.selectFirstMca()

    def clear(self):
        self.table.clearSelection(1)

    def reset(self):
        self.table.clearSelection(1)
        self.table.setNumCols(1)
        self.table.setNumRows(0)
        self.table.horizontalHeader().setLabel(0, "NO MCA for the selected scan", -1)
        self.table.adjustColumn(0)
        self.det= 0
        self.pts= 0

    def getSelection(self):
        if self.det==0:    return []
        selection= []
        for row in range(self.pts):
            for col in range(self.det):
                if self.table.isSelected(row, col):
                    #selection.append("%s.%d"%(self.info["Key"], self.__getMcaNo(row,col)))
                    selection.append("%s.%d.%d"%(self.info["Key"], row+1,col+1))
                elif self.table.text(row, col) != "":
                    selection.append("%s.%d.%d"%(self.info["Key"], row+1,col+1))

        return selection
    #
    # signal / slot handling
    #
    def __selectionChanged(self):
        if self.det==0: return
        else:
            self.emit(qt.PYSIGNAL("mcaSelection"), (self.getSelection,))

    def __doubleClicked(self, *args):
        #self.emit(qt.PYSIGNAL("mcaDoubleClicked"), (["%s.%d"%(self.info["Key"], self.__getMcaNo(args[0], args[1]))],))
        self.emit(qt.PYSIGNAL("mcaDoubleClicked"), (["%s.%d.%d"%(self.info["Key"], args[0]+1, args[1]+1)],))

    #
    # (mcano) <--> (row,col) operations
    #
    def __getMcaNo(self, row, col):
        return (row*self.det+col+1)
        
    def __getRowCol(self, mcano):
        mcano= mcano-1
        row= int(mcano/self.det)
        col= mcano-(row*self.det)
        return (row, col)

class CntTable(qt.QWidget):
    def __init__(self, parent=None, name=None, fl=0):
        qt.QWidget.__init__(self, parent, name, fl)

        self.mainLayout= qt.QVBoxLayout(self)

        self.table= qttable.QTable(self)
        self.table.setSelectionMode(qttable.QTable.Multi)
        if qt.qVersion() > '3.0.0':
            self.table.setFocusStyle(qttable.QTable.FollowStyle)
            #self.table.setFocusStyle(qttable.QTable.SpreadSheet)
            #self.table.setReadOnly(1)
            self.table.setColumnReadOnly(0,1)
            self.table.setColumnReadOnly(2,1)
        self.table.verticalHeader().setResizeEnabled(0, -1)
        self.table.horizontalHeader().setResizeEnabled(0, -1)
        self.cnt=0
        self.cntlist=[]

        # --- selection in case of a single MCA by scan
        self.firstCnt= qttable.QTableSelection()
        self.firstCnt.init(0, 0)
        self.firstCnt.expandTo(0, 0)

        self.mainLayout.addWidget(self.table)

        #do not use double click
        #self.connect(self.table, qt.SIGNAL("doubleClicked(int,int,int,const qt.QPoint&)"), self.__doubleClicked)
        self.connect(self.table,qt.SIGNAL("clicked(int,int,int,const QPoint&)"),
                                self.__Clicked)
        self.connect(self.table,qt.SIGNAL("selectionChanged()"),
                                 self.__selectionChanged)
        self.reset()

    #
    # data management
    #
    def setData(self, specfiledata):
        self.data= specfiledata
        self.reset()

    def setDataSource(self, specfiledata):
        self.data= specfiledata
        self.reset()

    def setScan(self, scankey):
        if type(scankey)==types.ListType: 
            if len(scankey): scankey= scankey[0]
            else: scankey=None
        if scankey is None:
            self.reset()
        else:
            self.info= self.data.GetSourceInfo(scankey)
            self.refresh()

    def markCntSelected(self, cntdict):
        if DEBUG:
            print("markCntSelected(self, cntdict)")
            print("cntdict = ", cntdict)
            print("self.cntlist = ",self.cntlist)
        if (cntdict != {}):
            for cnt in self.cntlist:
                row =self.cntlist.index(cnt)
                if cnt in cntdict["Xcnt"]:
                    self.table.xcheck[row].setChecked(1)
                else:
                    self.table.xcheck[row].setChecked(0)
                if cnt in cntdict["Ycnt"]:
                    self.table.ycheck[row].setChecked(1)
                else:
                    self.table.ycheck[row].setChecked(0)
                if cnt in cntdict["Mcnt"]:
                    self.table.mcheck[row].setChecked(1)
                else:
                    self.table.mcheck[row].setChecked(0)
            
    def selectCntList(self, cntlist):
        if DEBUG:
            print("selectCntList(self, cntlist)")
            print("cntlist = ",cntlist)
        for cntname in cntlist:
            self.selectCntName(cntname)

    def selectCntName(self, cntname):
        (row, col)= self.__getRowCol(cntname)
        selection= qttable.QTableSelection()
        selection.init(row, col)
        selection.expandTo(row, col)
        self.table.addSelection(selection)
        #self.__selectionChanged()

    def selectFirstCnt(self):
        self.table.addSelection(self.firstCnt)
        #self.__selectionChanged()

    def refresh(self):
        #self.cntlist= self.info["AllLabels"] or []
        self.cntlist= self.info["LabelNames"] or []
        pts= self.info["Lines"] or 0

        if self.cntlist==[]: 
            self.reset()
        else:
            # --- try to compute number of counters
            self.cnt=len(self.cntlist)
            self.table.setNumCols(4)
            self.table.setNumRows(self.cnt)
            Hheader= self.table.horizontalHeader()
            Hheader.setLabel(0, "Label", -1)
            Hheader.setLabel(1, "X", -1)
            self.table.adjustColumn(1)
            Hheader.setLabel(2, "Y", -1)
            self.table.adjustColumn(2)
            Hheader.setLabel(3, "M", -1)
            self.table.adjustColumn(3)
            #for idx in range(4):
            #    Hheader.setLabel(idx, "mca.%d"%(idx+1), -1)
            self.table.xcheck=[]
            self.table.ycheck=[]
            self.table.mcheck=[]
            for row in range(self.cnt):
                for col in range(4):
                    if col==1:
                        #self.table.setText(row, col, "")
                        self.table.xcheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                        self.table.setItem(row,col,self.table.xcheck[row])
                    elif col == 2:
                        #self.table.setText(row, col, "")
                        self.table.ycheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                        self.table.setItem(row,col,self.table.ycheck[row])
                    elif col == 3:
                        self.table.mcheck.append(qttable.QCheckTableItem(self.table,qt.QString("")))
                        self.table.setItem(row,col,self.table.mcheck[row])
                    else:
                        self.table.setText(row, col, self.cntlist[row])
            self.table.adjustColumn(0)

            #if mca==1:
            #    self.selectFirstMca()

    def clear(self):
        self.table.clearSelection(1)

    def reset(self):
        self.table.clearSelection(1)
        self.table.setNumCols(1)
        self.table.setNumRows(0)
        self.table.horizontalHeader().setLabel(0, "NO Counters for the selected scan", -1)
        self.table.adjustColumn(0)
        self.cnt= 0
        self.pts= 0

    def getSelection(self):
        if self.cnt==0:    return []
        selection= []
        Xcnt=[]
        Ycnt=[]
        Mcnt=[]
        for row in range(self.cnt):
            #if self.table.isSelected(row, 1):
            if self.table.xcheck[row].isChecked():
                #Xnt
                Xcnt.append(self.cntlist[row])
            if self.table.ycheck[row].isChecked():
                #Ycnt
                Ycnt.append(self.cntlist[row])
            #if self.table.isSelected(row, 3):
            if self.table.mcheck[row].isChecked():
                #Mcnt
                Mcnt.append(self.cntlist[row])

        for cnt in Ycnt:
            selection.append({"Key":"%s"%(self.info["Key"]),
                              "scan":"%s"%(self.info["Key"]),
                              "Xcnt":Xcnt,
                              "Ycnt":cnt,
                              "Mcnt":Mcnt})
        return selection
    #
    # signal / slot handling
    #
    def __selectionChanged(self):
        if self.cnt==0: return
        else:
            for row in range(self.cnt):
                if self.table.isSelected(row, 2):
                    self.table.ycheck[row].setChecked(1)
                else:
                    self.table.ycheck[row].setChecked(0)
            self.emit(qt.PYSIGNAL("cntSelection"), (self.getSelection,))

    def __doubleClicked(self, *args):
        self.emit(qt.PYSIGNAL("cntDoubleClicked"), (["%s.%d"%(self.info["Key"], self.__getCntNo(args[0], args[1]))],))


    def __Clicked(self, *args):
        row=args[0]
        col=args[1]
        if col == 0:
            pass
        elif col==1:
            #xclick
            maxclicked=1
            for i in range(self.cnt):
                if i != row:
                    self.table.xcheck[i].setChecked(0)                
        elif col==2:
            #yclick
            pass
        elif col==3:
            #mclixk
            for i in range(self.cnt):
                if i != row:
                    self.table.mcheck[i].setChecked(0) 
    #
    # (mcano) <--> (row,col) operations
    #
    def __getCntNo(self, row, col):
        return (row*self.cnt+col+1)
        
    def __getRowCol(self, mcano):
        mcano= mcano-1
        row= int(mcano/self.cnt)
        col= mcano-(row*self.cnt)
        return (row, col)

class SpecFileSelector(qt.QWidget):
    def __init__(self, parent=None, name="SpecFileSelector", fl=0):
        qt.QWidget.__init__(self, parent, name, fl)

        self.data= None
        self.lastSelection= None
        self.currentScan  = None
        self.selection    = None
        self.lastInputDir = None
        self.lastInputFilter = "Specfiles (*.mca)\nSpecfiles (*.dat)\nAll files (*)"

        mainLayout= qt.QVBoxLayout(self)

        
        # --- file combo/open/close
        fileWidget= qt.QWidget(self)
        self.fileCombo= qt.QComboBox(fileWidget)
        self.fileCombo.setEditable(0)
        self.mapComboName= {}
        self.openIcon= qt.QIconSet(qt.QPixmap(icons.fileopen))
        openButton= qt.QToolButton(fileWidget)
        openButton.setIconSet(self.openIcon)
        openButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        self.closeIcon= qt.QIconSet(qt.QPixmap(icons.fileclose))
        closeButton= qt.QToolButton(fileWidget)
        closeButton.setIconSet(self.closeIcon)
        closeButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        fileLayout= qt.QHBoxLayout(fileWidget)
        fileLayout.addWidget(self.fileCombo)
        fileLayout.addWidget(openButton)
        fileLayout.addWidget(closeButton)

        self.connect(openButton, qt.SIGNAL("clicked()"), self.openFile)
        self.connect(closeButton, qt.SIGNAL("clicked()"), self.closeFile)
        self.connect(self.fileCombo, qt.SIGNAL("activated(const QString &)"), self.__fileSelection)

        # --- splitter with scan list/mca list
        splitter= qt.QSplitter(self)
        splitter.setOrientation(qt.QSplitter.Vertical)

        #self.scanList= ScanList(splitter, selection="single", type="both")
        self.scanList= ScanList(splitter, selection="extended", type="both")
        self.tabwidget=qt.QTabWidget(splitter)
        self.mcaTable= McaTable(self)
        self.cntTable= CntTable(self)
        self.tabwidget.addTab(self.cntTable,"Counters")
        self.tabwidget.addTab(self.mcaTable,"MCA")
        self.connect(self.scanList, qt.PYSIGNAL("scanSelection"),     self.__scanSelection)
        self.connect(self.scanList, qt.PYSIGNAL("scanDoubleClicked"), self.__scanDoubleClicked)
        self.connect(self.mcaTable, qt.PYSIGNAL("mcaDoubleClicked"), self.__mcaDoubleClicked)
        self.connect(self.cntTable, qt.PYSIGNAL("cntDoubleClicked"), self.__cntDoubleClicked)

        # --- select / remove buttons
        butWidget    = qt.QWidget(self)
        #self.directView    = qt.QCheckBox("direct view", butWidget)
        addButton    = qt.QPushButton("Add", butWidget)
        removeButton    = qt.QPushButton("Remove", butWidget)
        replaceButton    = qt.QPushButton("Replace", butWidget)

        butLayout= qt.QHBoxLayout(butWidget)
        #butLayout.addWidget(self.directView)
        butLayout.addWidget(addButton)
        butLayout.addWidget(removeButton)
        butLayout.addWidget(replaceButton)

        self.connect(addButton, qt.SIGNAL("clicked()"), self.__addClicked)
        self.connect(removeButton, qt.SIGNAL("clicked()"), self.__removeClicked)
        self.connect(replaceButton, qt.SIGNAL("clicked()"), self.__replaceClicked)

        mainLayout.addWidget(fileWidget)
        mainLayout.addWidget(splitter)
        mainLayout.addWidget(butWidget)

    def openFile(self, filename=None):
        if filename is None:
            if self.lastInputDir is not None:
                if not os.path.exists(self.lastInputDir):
                    self.lastInputDir = None   
            if sys.platform == "win32":
                windir = self.lastInputDir
                if windir is None:windir = ""
                filename= str(qt.QFileDialog.getOpenFileName(windir,
                                 self.lastInputFilter,
                                 self,
                                "openFile", "Open a new SpecFile"))
            else:
                try:
                    filename = qt.QFileDialog(self, "Open a new SpecFile", 1)
                    filename.setFilters(self.lastInputFilter)
                    if self.lastInputDir is not None:
                        filename.setDir(self.lastInputDir)
                    filename.setMode(qt.QFileDialog.ExistingFile)
                    if filename.exec_loop() == qt.QDialog.Accepted:
                        #selectedfilter = str(filename.selectedFilter())
                        filename= str(filename.selectedFile())
                        #print selectedfilter
                    else:
                        return
                except:
                    print("USING STATIC METHODS, PLEASE REPORT THIS ISSUE")
                    windir = self.lastInputDir
                    if windir is None:windir = ""
                    filename= str(qt.QFileDialog.getOpenFileName(windir,
                                     self.lastInputFilter,
                                     self,
                                    "openFile", "Open a new SpecFile"))
                    
            if not len(filename):    return
            else:
                self.lastInputDir    = os.path.dirname(filename)
                if len(filename) > 4:
                    if filename[-4:] == ".dat":
                        self.lastInputFilter = "Specfiles (*.dat)\nSpecfiles (*.mca)\nAll files (*)"
                    elif filename[-4:] == ".mca":
                        self.lastInputFilter = "Specfiles (*.mca)\nSpecfiles (*.dat)\nAll files (*)"
                    else:
                        self.lastInputFilter = "All files (*)\nSpecfiles (*.mca)\nSpecfiles (*.dat)"
    
        if filename in self.mapComboName.keys():
            self.selectFile(filename)
        else:
            if not self.data.SetSource(filename):
                qt.QMessageBox.critical(self, "ERROR opening Specfile",
                        "Cannot open following specfile:\n%s"%(filename))
            else:
                filename= self.data.SourceName
                self.mapComboName[filename]= os.path.basename(filename)
                self.fileCombo.insertItem(self.mapComboName[filename])
                self.selectFile(filename)

    def selectFile(self, filename=None):
        if filename is not None:
            if str(self.fileCombo.currentText())!=self.mapComboName[filename]:
              for idx in range(self.fileCombo.count()):
                if str(self.fileCombo.text(idx))==self.mapComboName[filename]:
                    self.fileCombo.setCurrentItem(idx)
                    break
            self.data.SetSource(filename)
        self.refresh()

    def closeFile(self, filename=None):
        if filename is None:
            file= str(self.fileCombo.currentText())
            for filename, comboname in self.mapComboName.items():
                if comboname==file: break

        if (self.selection is not None) and (filename in self.selection):
            mcakeys= []
            for scan in self.selection[filename].keys():
                mcakeys += [ "%s.%s"%(scan, mca) for mca in self.selection[filename][scan] ]
            if len(mcakeys):
                msg= "%d mca are linked to that Specfile source.\n"%len(mcakeys)
                msg+="Do you really want to delete all these graphs ??"
                ans= qt.QMessageBox.information(self, "Remove SpecFile %s"%filename, msg,
                        qt.QMessageBox.No, qt.QMessageBox.Yes)
                if ans==qt.QMessageBox.No: return
                self.emit(qt.PYSIGNAL("delSelection"), (self.data.SourceName, mcakeys))
            
        if (self.selection is not None) and (filename in self.selection):
            cntkeys= []
            for scan in self.selection[filename].keys():
                cntkeys += [ "%s.%s"%(scan, mca) for cnt in self.selection[filename][scan] ]
            if len(cntkeys):
                msg= "%d cnt are linked to that Specfile source.\n"%len(cntkeys)
                msg+="Do you really want to delete all these graphs ??"
                ans= qt.QMessageBox.information(self, "Remove SpecFile %s"%filename, msg,
                        qt.QMessageBox.No, qt.QMessageBox.Yes)
                if ans==qt.QMessageBox.No: return
                self.emit(qt.PYSIGNAL("delSelection"), (self.data.SourceName, cntkeys))
            
        for idx in range(self.fileCombo.count()):
            if str(self.fileCombo.text(idx))==self.mapComboName[filename]:
                #if idx==self.fileCombo.currentItem():
                self.fileCombo.removeItem(idx)
                del self.mapComboName[filename]
                break

        if not self.fileCombo.count():
            self.selectFile()
        else:
            self.selectFile(self.mapComboName.keys()[0])

    def __fileSelection(self, file):
        file= str(file)
        for filename, comboname in self.mapComboName.items():
            if comboname==file:
                self.selectFile(filename)
                break

    """    
    def setFileList(self, filelist, fileselected=None):
        self.fileCombo.clear()
        for file in filelist:
            self.fileCombo.insertItem(file)
        if fileselected is None: fileselected= filelist[0]
        self.selectFile(fileselected)
    """

    def setData(self, specfiledata):
        self.data= specfiledata
        self.scanList.setData(specfiledata)
        self.mcaTable.setData(specfiledata)
        self.cntTable.setData(specfiledata)

    def setDataSource(self, specfiledata):
        self.data= specfiledata
        self.data.SourceName = specfiledata.sourceName[0]
        self.scanList.setDataSource(specfiledata)
        self.mcaTable.setDataSource(specfiledata)
        self.cntTable.setDataSource(specfiledata)

    def setSelected(self, sellist,reset=1):
        if DEBUG:
            print("setSelected(self, sellist) called")
            print("sellist = ",sellist)
            print("self.selection before = ",self.selection)
            print("reset = ",reset)
        if reset:
            self.selection= {}
        for sel in sellist:
            if DEBUG:
                print sel.keys()
            filename= sel["SourceName"]
            if type(sel["Key"]) == type([]):
                selkey = sel["Key"][0]
            else:
                selkey = sel["Key"]
            stringsplit = string.split(selkey,".")
            scan = stringsplit[0]
            order= stringsplit[1]
            if len(stringsplit) < 3:
                #scan selection
                pass  
            elif len(stringsplit) < 4:
                mca = selkey
            else:
                mca = selkey
            scankey= "%s.%s"%(scan,order)
            if not (filename in self.selection):
                self.selection[filename]= {}
            if not (scankey in self.selection[filename]):
                self.selection[filename][scankey]= {}
                self.selection[filename][scankey]['mca'] = []
                self.selection[filename][scankey]['scan'] = {}
            if 'mca' in sel[selkey]:
                self.selection[filename][scankey]['mca'] = []
                for mca in sel[selkey]['mca']:               
                    self.selection[filename][scankey]['mca'].append(mca)
            if 'scan' in sel[selkey]:
                keys = list(sel[selkey]['scan'].keys())
                if ('Xcnt' in keys) and ('Ycnt' in keys) and ('Mcnt' in keys): 
                    for key in keys:
                        self.selection[filename][scankey]['scan'][key] = sel[selkey]['scan'][key]
                else:
                    self.selection[filename][scankey]['scan'] ['Xcnt'] = []
                    self.selection[filename][scankey]['scan'] ['Ycnt'] = []
                    self.selection[filename][scankey]['scan'] ['Mcnt'] = []
        if DEBUG:
            print("self.selection after = ",self.selection)
        self.__refreshSelection()

    def OLDsetSelection(self, seldict={}):
        self.selection= {}
        for filename in seldict.keys():
            self.selection[filename]= {}
            for key in seldict[filename]:
                scan, order, mca= key.split(".")
                scankey= "%s.%s"%(scan, order)
                if not ((scankey in self.selection[filename]):
                    self.selection[filename][scankey]= []
                self.selection[filename][scankey].append(mca)
        self.__refreshSelection()
    
    def __refreshSelection(self):
        if DEBUG:
            print("__refreshSelection(self) called")
            print(self.selection)
        if self.selection is not None:
            sel = self.selection.get(self.data.SourceName, {})
            selkeys = []
            for key in list(sel.keys()):
                if sel[key]['mca'] != []:
                    selkeys.append(key)
                elif 'Ycnt' in sel[key]['scan']:
                    if sel[key]['scan']['Ycnt'] !=  []:
                        selkeys.append(key)
                    
            if DEBUG:
                print("selected scans =", selkeys)
                print("but self.selection = ", self.selection)
                print("and self.selection.get(self.data.SourceName, {}) =", sel)
            self.scanList.markScanSelected(selkeys)
            scandict = sel.get(self.currentScan, {})
            if 'mca' in scandict:
                self.mcaTable.markMcaSelected(scandict['mca'])
            else:
                self.mcaTable.markMcaSelected([])
            if 'scan' in scandict:
                self.cntTable.markCntSelected(scandict['scan'])
            else:
                self.cntTable.markCntSelected({})
        
    def refresh(self):
        self.scanList.refresh()
        self.mcaTable.reset()
        self.cntTable.reset()
        self.__refreshSelection()
    
    def __scanSelection(self, scankey):
        if DEBUG:
            print("__scanSelection(self, scankey) called")
            print("scankey = %s" % scankey)
        self.mcaTable.setScan(scankey)
        self.cntTable.setScan(scankey)
        if len(scankey):
            self.currentScan     = scankey[0]
            self.currentScanList = scankey
            if (self.selection is not None) and\
                (self.data.SourceName in self.selection):
                scandict = self.selection[self.data.SourceName].get(self.currentScan, {})
                if 'mca' in scandict:
                    self.mcaTable.markMcaSelected(scandict['mca'])
                else:
                    self.mcaTable.markMcaSelected([])
                if 'scan' in scandict:
                    self.cntTable.markCntSelected(scandict['scan'])
                else:
                    self.cntTable.markCntSelected({})
                if DEBUG:
                    print("calling refresh selection")
                self.__refreshSelection()
        else:
            self.currentScan     = None
            self.currentScanList = None

    def __scanDoubleClicked(self, dict):
        if DEBUG:
            print("__scanDoubleClicked(self, dict) called")
            print("dict = ",dict)
        scankey = dict['Key']
        if not len(scankey):return
        if self.currentScan != scankey:
            self.__scanSelection([scankey])
        if self.currentScan is None: return
        key = self.currentScan
        if 'NbMca' in self.mcaTable.info:
            if self.mcaTable.info['NbMca'] == 1:
                if dict['NbPoints'] == 0:
                    mcakeys = ["%s.1.1" % key]
                elif dict['NbPoints'] == 1:
                    mcakeys = ["%s.1.1" % key]
                else:
                    return                    
                sel = {"SourceName":self.data.SourceName,
                       "Key":scankey,
                        scankey:{'mca':mcakeys}} 
                self.__toggleselection(sel)

    def __mcaSelection(self, mcakeys):
        self.lastSelection= mcakeys
        #if self.directView.isChecked():
        #    self.emit(qt.PYSIGNAL("tempSelection"), (mcakeys,))

    def __mcaDoubleClicked(self, mcakeys):
        if DEBUG:
            print("__mcaDoubleClicked(self, mcakeys) called")
            print("mcakeys = ",mcakeys)
        if self.currentScan is not None:
            key = self.currentScan
            if type(mcakeys) == type([]):
                sel= {"SourceName":self.data.SourceName, "Key":key,key:{'mca':mcakeys}}
            else:
                sel= {"SourceName":self.data.SourceName, "Key":key,key:{'mca':[mcakeys]}}
        self.__toggleselection(sel)

    def __toggleselection(self,sel):
        if DEBUG:
            print("toggleselection(self,sel) called")
            print("sel = ",sel)
        if self.selection is None:
            self.setSelected([sel])
            self.__addClicked()
        else:
            mca = None
            filename= sel["SourceName"]
            if type(sel["Key"]) == type([]):
                selkey = sel["Key"][0]
            else:
                selkey = sel["Key"]
            stringsplit = string.split(selkey,".") 
            scan = stringsplit[0]
            order= stringsplit[1]
            """
            if len(stringsplit) < 3:
                #scan selection
                pass  
            elif len(stringsplit) < 4:
                mca = selkey
            else:
                mca = selkey
            """
    
            scankey= "%s.%s"%(scan,order)
            if not (filename in self.selection):
                self.setSelected([sel],reset=0)
                self.__addClicked()
            elif not (scankey in self.selection[filename]):
                self.setSelected([sel],reset=0)
                self.__addClicked()
            elif not ('mca' in self.selection[filename][scankey]):
                self.setSelected([sel],reset=0)
                self.__addClicked()
            else:
                todelete = []
                toadd   = []
                for mca in sel[selkey]['mca']:
                    if mca in self.selection[filename][scankey]['mca']:
                        #i = self.selection[filename][scankey]['mca'].index(mca)
                        #del self.selection[filename][scankey]['mca'][i]
                        self.removeSelection([{'SourceType':SOURCE_TYPE,
                                               'SourceName':sel['SourceName'],
                                               'Key':selkey,
                                               selkey:{'mca':[mca],'scan':{}}}])
                        todelete.append(mca)
                    else:
                        self.selection[filename][scankey]['mca'].append(mca)
                        toadd.append(mca)
                self.__refreshSelection()
                if len(toadd):
                    self.__addClicked()
        
    def __cntSelection(self, cntkeys):
        self.lastSelection= cntkeys
        #if self.directView.isChecked():
        #    self.emit(PYqt.SIGNAL("tempSelection"), (mcakeys,))

    def __cntDoubleClicked(self, cntkeys):
        if DEBUG:
            print("__cntDoubleClicked(self, cntkeys) called")
        sel= {"SourceName":self.data.SourceName, "Key":cntkeys}
        #self.eh.event(self.addEvent, [sel])

    def __OLDmcaDoubleClicked(self, mcakeys):
        self.emit(qt.PYSIGNAL("newSelection"), (mcakeys,))

    def __addClicked(self):
        if DEBUG:
            print("__selectClicked called")
        if self.currentScan is not None:
            sellist = []
            scankeylist     =  self.currentScanList
            for scankey in scankeylist:
                sel   = {}
                sel['SourceName'] = self.data.SourceName
                sel['SourceType'] = SOURCE_TYPE
                sel['Key']  = scankey
                sel[scankey]= {}
                sel[scankey]['mca']  = []
                sel[scankey]['scan'] = {}
                for mca in self.mcaTable.getSelection():
                    if DEBUG:
                        print("scankey =",scankey,"adding mca = ",mca)
                    actualmcakey = mca.replace(scankeylist[0],scankey)
                    actualscan, actualorder,actualpoint,actualmca = actualmcakey.split('.')
                    if float(actualmca) <= self.data.GetSourceInfo(scankey)['NbMca']:
                        sel[scankey]['mca'].append(actualmcakey)
                    else:
                        if DEBUG:
                            print("mcakey ",actualmcakey," skipped")
                labellist = self.data.GetSourceInfo(scankey)['LabelNames']
                for item in self.cntTable.getSelection():
                    if DEBUG:
                        print(item)
                    if item["Xcnt"][0] in labellist:
                        if item["Ycnt"] in labellist:
                            if len(item["Mcnt"]):   
                                if item["Mcnt"][0] in labellist:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]
                            else:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]                                
                    else:
                        if DEBUG:
                            print("scankey ",scankey," skipped")
                sellist.append(sel)
                if self.selection is None: 
                    self.setSelected([sel],reset=1)
                else:
                    self.setSelected([sel],reset=0)
                self.emit(qt.PYSIGNAL("addSelection"), ([sel],))
        
    def __replaceClicked(self):
        if DEBUG:
            print("__selectClicked called")
        if self.currentScan is not None:
            scankeylist     =  self.currentScanList
            mcalist         =  self.mcaTable.getSelection()
            itemlist        =  self.cntTable.getSelection()
            reset = 1
            sellist = []
            for scankey in scankeylist:
                sel   = {}
                sel['SourceName'] = self.data.SourceName
                sel['SourceType'] = SOURCE_TYPE
                sel['Key']  = scankey
                sel[scankey]= {}
                sel[scankey]['mca']  = []
                sel[scankey]['scan'] = {}
                for mca in mcalist:
                    if DEBUG:
                        print("scankey =",scankey,"removing mca = ",mca)
                    actualmcakey = mca.replace(scankeylist[0],scankey)
                    actualscan, actualorder,actualpoint,actualmca = actualmcakey.split('.')
                    if float(actualmca) <= self.data.GetSourceInfo(scankey)['NbMca']:
                        sel[scankey]['mca'].append(actualmcakey)
                    else:
                        if DEBUG:
                            print("mcakey ",actualmcakey," skipped")
                labellist = self.data.GetSourceInfo(scankey)['LabelNames']
                for item in itemlist:
                    if DEBUG:
                        print("removing item ",item)
                    if item["Xcnt"][0] in labellist:
                        if item["Ycnt"] in labellist:
                            if len(item["Mcnt"]):   
                                if item["Mcnt"][0] in labellist:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]
                            else:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]                                
                    else:
                        if DEBUG:
                            print("scankey ",scankey," skipped")
                sellist.append(sel)
            if reset:
                self.setSelected(sellist,reset=1)
                reset = 0
            else:
                self.setSelected(sellist,reset=0)
            if DEBUG:
                print("replace sellist = ",sellist)
            self.emit(qt.PYSIGNAL("replaceSelection"), (sellist,))
            
    def __OLDselectClicked(self):
        selection= self.mcaTable.getSelection()
        if (len(selection)):
            self.emit(qt.PYSIGNAL("newSelection"), (self.data.SourceName, selection,))
        
    def __removeClicked(self):
        if self.currentScan is not None:
            scankeylist     =  self.currentScanList
            mcalist         =  self.mcaTable.getSelection()
            itemlist        =  self.cntTable.getSelection()
            sellist = []
            for scankey in scankeylist:
                sel   = {}
                sel['SourceName'] = self.data.SourceName
                sel['SourceType'] = SOURCE_TYPE
                sel['Key']  = scankey
                sel[scankey]= {}
                sel[scankey]['mca']  = []
                sel[scankey]['scan'] = {}
                for mca in mcalist:
                    if DEBUG:
                        print("scankey =",scankey,"removing mca = ",mca)
                    actualmcakey = mca.replace(scankeylist[0],scankey)
                    actualscan, actualorder,actualpoint,actualmca = actualmcakey.split('.')
                    if float(actualmca) <= self.data.GetSourceInfo(scankey)['NbMca']:
                        sel[scankey]['mca'].append(actualmcakey)
                    else:
                        if DEBUG:
                            print("mcakey ",actualmcakey," skipped")
                labellist = self.data.GetSourceInfo(scankey)['LabelNames']
                for item in itemlist:
                    if DEBUG:
                        print("removing item ",item)
                    if item["Xcnt"][0] in labellist:
                        if item["Ycnt"] in labellist:
                            if len(item["Mcnt"]):   
                                if item["Mcnt"][0] in labellist:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]
                            else:
                                    sel[scankey]['scan']['scan'] = item['scan']
                                    sel[scankey]['scan']['Xcnt'] = item["Xcnt"]
                                    if 'Ycnt' in sel[scankey]['scan']:
                                        sel[scankey]['scan']['Ycnt'].append(item["Ycnt"])
                                    else:
                                        sel[scankey]['scan']['Ycnt']=[item["Ycnt"]]
                                    sel[scankey]['scan']['Mcnt']=item["Mcnt"]                                
                    else:
                        if DEBUG:
                            print("scankey ",scankey," skipped")
                sellist.append(sel)
            if DEBUG:
                print("removeSelection list = ",sellist)
                
                
            self.removeSelection(sellist)
        if 0:
            if self.selection is not None:
                if self.data.SourceName in self.selection:
                    if scankey in self.selection[self.data.SourceName]:
                        if 'mca' in self.selection[self.data.SourceName][scankey]:
                            for mca in self.selection[self.data.SourceName][scankey]['mca']:
                                if mca in  sel[scankey]['mca']:
                                    index = self.selection[self.data.SourceName][scankey]['mca'].index(mca)
                                    del self.selection[self.data.SourceName][scankey]['mca'][index]
                        if 'scan' in self.selection[self.data.SourceName][scankey]:
                          if 'Ycnt' in self.selection[self.data.SourceName][scankey]['scan']:                                
                            for Ycnt in  self.selection[self.data.SourceName][scankey]['scan']['Ycnt']:
                                if Ycnt in  sel[scankey]['scan']['Ycnt']:
                                    index = self.selection[self.data.SourceName][scankey]['scan']['Ycnt'].index(Ycnt)
                                    del self.selection[self.data.SourceName][scankey]['scan']['Ycnt'][index]
                            if self.selection[self.data.SourceName][scankey]['scan']['Ycnt'] == []:
                               self.selection[self.data.SourceName][scankey]['scan']['Xcnt'] = []
                               self.selection[self.data.SourceName][scankey]['scan']['Mcnt'] = []
                        seln = {}
                        seln['SourceName']  = self.data.SourceName
                        seln['SourceType']  = SOURCE_TYPE
                        seln['Key']         = scankey
                        seln[scankey]       =   self.selection[self.data.SourceName][scankey]                                         
        """                self.setSelected([seln],reset=0)
        try:
            if (len(sel)):
                self.emit(qt.PYSIGNAL("removeSelection"), ([sel],))
        except:
            pass
        """

    def removeSelection(self,selection):
        if DEBUG:
            print("removeSelection(self,selection), selection = ",selection)
        if type(selection) != type([]):
            selection=[selection]
        for sel in selection:
            scankey = sel['Key']
            if self.selection is not None:
                if self.data.SourceName in self.selection:
                    if scankey in self.selection[self.data.SourceName]:
                        if 'mca' in self.selection[self.data.SourceName][scankey]:
                            for mca in self.selection[self.data.SourceName][scankey]['mca']:
                                if mca in  sel[scankey]['mca']:
                                    index = self.selection[self.data.SourceName][scankey]['mca'].index(mca)
                                    del self.selection[self.data.SourceName][scankey]['mca'][index]
                        if 'scan' in self.selection[self.data.SourceName][scankey]:
                          if 'Ycnt' in self.selection[self.data.SourceName][scankey]['scan']:                                
                            for Ycnt in  self.selection[self.data.SourceName][scankey]['scan']['Ycnt']:
                                if Ycnt in  sel[scankey]['scan']['Ycnt']:
                                    index = self.selection[self.data.SourceName][scankey]['scan']['Ycnt'].index(Ycnt)
                                    del self.selection[self.data.SourceName][scankey]['scan']['Ycnt'][index]
                            if self.selection[self.data.SourceName][scankey]['scan']['Ycnt'] == []:
                               self.selection[self.data.SourceName][scankey]['scan']['Xcnt'] = []
                               self.selection[self.data.SourceName][scankey]['scan']['Mcnt'] = []
                        seln = {}
                        seln['SourceName']  = self.data.SourceName
                        seln['SourceType']  = SOURCE_TYPE
                        seln['Key']         = scankey
                        seln[scankey]       =   self.selection[self.data.SourceName][scankey]                                         
                        self.setSelected([seln],reset=0)
        if (len(sel)):
                self.emit(qt.PYSIGNAL("removeSelection"), (selection,))
        

    def __OLDremoveClicked(self):
        selection= self.mcaTable.getSelection()
        if (len(selection)):
            self.emit(qt.PYSIGNAL("delSelection"), (self.data.SourceName, selection,))

    def getSelection(self):
        """
        Give the dicionary of dictionaries as an easy to understand list of
        individual selections
        """
        selection = []
        if self.selection is None: return selection
        for sourcekey in self.selection.keys():
            for scankey in self.selection[sourcekey].keys():
                sel={}
                sel['SourceName']   = sourcekey
                sel['SourceType']   = SOURCE_TYPE
                sel['Key']          = scankey
                sel[scankey]        = self.selection[sourcekey][scankey]
                selection.append(sel)
        return selection

def test():
    import sys
    if PYDVT:
        import SpecFileData
    else:
        import SpecFileLayer

    global CurrSelection, w
    CurrSelection= []

    def delSelection(sellist):
        global CurrSelection
        print("delSelection", sellist)
        for sel in sellist:
            if sel in CurrSelection:
                CurrSelection.remove(sel)
        w.setSelected(CurrSelection)

    def addSelection(sel):
        global CurrSelection, w
        print("addSelection", sel)
        CurrSelection+=sel
        w.setSelected(CurrSelection)
        
    def myprint(*var,**kw):
        global CurrSelection, w
        print(w.selection)
        print(w.getSelection())

    #if not len(sys.argv)>1:
    #    print "USAGE: %s <specfile>"%sys.argv[0]
    #    sys.exit(0)

    a = qt.QApplication(sys.argv)
    a.myprint = myprint
    w = SpecFileSelector()
    w.eh.register("delSelection", delSelection)
    w.eh.register("addSelection", addSelection)
    qt.QObject.connect(a, qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))
    a.connect(w,qt.PYSIGNAL("removeSelection"),myprint)
    a.connect(w,qt.PYSIGNAL("addSelection"),myprint)
    a.connect(w,qt.PYSIGNAL("replaceSelection"),myprint)
    a.setMainWidget(w)

    if PYDVT:
        d = SpecFileData.SpecFileData()
    else:
        d = SpecFileLayer.SpecFileLayer()
    w.setData(d)
    
    w.show()
    a.exec_loop()

if __name__=="__main__":
    test()
