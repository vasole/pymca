import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        if qt.qVersion() < '4.1.3':
            print "WARNING: Tested from Qt 4.1.3 on"
    except:
        import qt
else:
    import qt
QTVERSION = qt.qVersion()
if QTVERSION < '4.0.0':
    import MySpecFileSelector
else:
    import SpecFileCntTable
    import SpecFileMcaTable
import SpecFileDataInfo
import Icons as icons
import os

DEBUG = 0

class QSourceSelector(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout= qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)

        # --- file combo/open/close
        self.lastInputDir = None
        self.fileWidget= qt.QWidget(self)
        fileWidgetLayout= qt.QHBoxLayout(self.fileWidget)
        fileWidgetLayout.setMargin(0)
        fileWidgetLayout.setSpacing(0)
        self.fileCombo  = qt.QComboBox(self.fileWidget)
        self.fileCombo.setEditable(0)
        self.mapCombo= {}
        openButton= qt.QToolButton(self.fileWidget)
        if QTVERSION < '4.0.0':
            self.openIcon= qt.QIconSet(qt.QPixmap(icons.fileopen))
            self.closeIcon= qt.QIconSet(qt.QPixmap(icons.fileclose))
            openButton.setIconSet(self.openIcon)
            openButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
            closeButton= qt.QToolButton(self.fileWidget)
            closeButton.setIconSet(self.closeIcon)
        else:
            self.openIcon= qt.QIcon(qt.QPixmap(icons.fileopen))
            self.closeIcon= qt.QIcon(qt.QPixmap(icons.fileclose))
            openButton.setIcon(self.openIcon)
            openButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
            closeButton= qt.QToolButton(self.fileWidget)
            closeButton.setIcon(self.closeIcon)
        closeButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        self.connect(openButton, qt.SIGNAL("clicked()"), self._openFileSlot)
        self.connect(closeButton, qt.SIGNAL("clicked()"), self.closeFile)
        self.connect(self.fileCombo, qt.SIGNAL("activated(const QString &)"),
                                                     self._fileSelection)

        #specfile
        #self.specfileWidget = ScanList(self)

        fileWidgetLayout.addWidget(self.fileCombo)
        fileWidgetLayout.addWidget(openButton)
        fileWidgetLayout.addWidget(closeButton)
        self.mainLayout.addWidget(self.fileWidget)
        #self.mainLayout.addWidget(self.specfileWidget)

    def _openFileSlot(self):
        self.openFile(None, None)

    def openFile(self, filename=None,justloaded=None):
        if DEBUG:
            print "openfile = ",filename
        if justloaded is None: justloaded = True
        if filename is None:
            if self.lastInputDir is not None:
                if not os.path.exists(self.lastInputDir):
                    self.lastInputDir = None
            wdir = self.lastInputDir
            if wdir is None:wdir = os.getcwd()
            if QTVERSION < '4.0.0':
                filetypes = ""
                filetypes += "Spec Files (*mca)\n"
                filetypes += "Spec Files (*dat)\n"
                filetypes += "EDF Files (*edf)\n"
                filetypes += "EDF Files (*ccd)\n"
                filetypes += "All Files (*)\n"
                if sys.platform == 'win32':
                    filelist = qt.QFileDialog.getOpenFileNames(filetypes,
                                wdir,
                                self,"openFile", "Open a new EdfFile")
                else:
                    filedialog = qt.QFileDialog(self,"Open new EdfFile(s)",1)
                    if self.lastInputDir is not None:filedialog.setDir(self.lastInputDir)
                    filedialog.setMode(filedialog.ExistingFiles)
                    filedialog.setFilters(filetypes)           
                    if filedialog.exec_loop() == qt.QDialog.Accepted:
                        filelist= filedialog.selectedFiles()
                    else:
                        return              
            else:
                #if sys.platform == 'win32':
                if sys.platform != 'darwin':
                    filetypes = ""
                    filetypes += "Spec Files *mca\n"
                    filetypes += "Spec Files *dat\n"
                    filetypes += "EDF Files *edf\n"
                    filetypes += "EDF Files *ccd\n"
                    filetypes += "All Files *\n"
                    filelist = qt.QFileDialog.getOpenFileNames(self,
                                "Open a new source file",          wdir,
                                filetypes)
                else:
                    fdialog = qt.QFileDialog(self)
                    fdialog.setModal(True)
                    fdialog.setWindowTitle("Open a new EdfFile")
                    strlist = qt.QStringList()
                    strlist.append("Spec Files *mca")
                    strlist.append("Spec Files *dat")
                    strlist.append("EDF Files *edf")
                    strlist.append("EDF Files *ccd")
                    strlist.append("All Files *")
                    fdialog.setFilters(strlist)
                    fdialog.setFileMode(fdialog.ExistingFiles)
                    fdialog.setDirectory(wdir)
                    ret = fdialog.exec_()
                    if ret == qt.QDialog.Accepted:
                        filelist = fdialog.selectedFiles()
                        fdialog.close()
                        del fdialog                        
                    else:
                        fdialog.close()
                        del fdialog
                        return            
            filelist.sort()
            filename=[]
            for f in filelist:
                filename.append(str(f))
            if not len(filename):    return
            if len(filename):self.lastInputDir=os.path.dirname(filename[0])
            justloaded = True
        if justloaded:
            if type(filename) != type([]):
                filename = [filename]
        if not os.path.exists(filename[0]):
            raise "IOError",("File %s does not exist" % filename[0])

        #check if it is a stack
        if len(filename) > 1:
            key = "STACK from %s to %s" % (filename[0], filename[-1])
        else:
            key = os.path.basename(filename[0])

        ddict = {}
        ddict["event"] = "NewSourceSelected"
        if key in self.mapCombo.keys():
            print self.mapCombo
            if self.mapCombo[key] == filename:
                #Reloaded event
                ddict["event"] = "SourceReloaded"
            else:
                i = 0
                while key in self.mapCombo.keys():
                    key += "_%d" % i
        ddict["combokey"]   = key
        ddict["sourcelist"] = filename
        self.mapCombo[key] = filename
        if ddict["event"] =="NewSourceSelected":
            nitems = self.fileCombo.count()
            if QTVERSION < '4.0.0':
                self.fileCombo.insertItem(key)
                self.fileCombo.setCurrentItem(nitems)
            else:
                self.fileCombo.insertItem(nitems, key)
                self.fileCombo.setCurrentIndex(nitems)
        else:
            nitem = self.fileCombo.findText(qt.QString(key))
            self.fileCombo.setCurrentIndex(nitem)
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("SourceSelectorSignal"), (ddict,))        
        else:
            self.emit(qt.SIGNAL("SourceSelectorSignal"), ddict)

    def closeFile(self):
        if DEBUG:
            print "closeFile called"
        #get current combobox key
        qstring = self.fileCombo.currentText()
        if not len(qstring): return
        key = str(qstring)
        ddict = {}
        ddict["event"] = "SourceClosed"
        ddict["combokey"] = key
        ddict["sourcelist"] = self.mapCombo[key] * 1
        nitem = self.fileCombo.findText(qt.QString(key))
        self.fileCombo.removeItem(nitem)
        del self.mapCombo[key]
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("SourceSelectorSignal"), (ddict,))    
        else:
            self.emit(qt.SIGNAL("SourceSelectorSignal"), ddict)

    def _fileSelection(self, qstring):
        if DEBUG:
            print "file selected ", qstring
        key = str(qstring)
        ddict = {}
        ddict["event"] = "SourceSelected"
        ddict["combokey"] = key
        ddict["sourcelist"] = self.mapCombo[key]
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("SourceSelectorSignal"), (ddict,))    
        else:
            self.emit(qt.SIGNAL("SourceSelectorSignal"), ddict)

class ScanList(qt.QWidget):
    def __init__(self, parent=None, name=None, fl=0):
        qt.QWidget.__init__(self, parent)
        if name is not None:self.setWindowTitle(name)
        self.layout= qt.QVBoxLayout(self)
        self.list  = qt.QTreeWidget(self)
        self.mainTab = qt.QTabWidget(self)

        self.cntTable = SpecFileTable.CntTable()
        self.mcaTable = SpecFileMcaTable.McaTable()

        self.mainTab.addTab(self.cntTable,str("Counters"))
        self.mainTab.addTab(self.mcaTable,str("MCA"))
        self.layout.addWidget(self.list)
        self.layout.addWidget(self.mainTab)

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
        self.list.header().setResizeMode(0, qt.QHeaderView.Custom)
        self.list.header().setResizeMode(1, qt.QHeaderView.Custom)
        self.list.header().setResizeMode(2, qt.QHeaderView.Stretch)
        self.list.header().setResizeMode(3, qt.QHeaderView.Custom)
        self.list.header().setResizeMode(4, qt.QHeaderView.Custom)

        # --- signal handling
        self.connect(self.list, qt.SIGNAL("itemSelectionChanged()"), self.__selectionChanged)
        self.list.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.connect(self.list, qt.SIGNAL("customContextMenuRequested(const QPoint &)"), self.__contextMenu)
        self.connect(self.list, qt.SIGNAL("itemDoubleClicked(QTreeWidgetItem *, int)"), self.__doubleClicked)

        self.disableMca    = 0 #(type=="scan")
        self.disableScan   = 0 #(type=="mca")

        # --- context menu        
        self.data= None
        self.scans= []



    # 
    # Data management
    #
    #NEW data management
    def setDataSource(self, datasource):
        self.data = datasource
        self.refresh()
    
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
        if DEBUG:print "__selectionChanged"
        itemlist = self.list.selectedItems()
        sel = [str(item.text(1)) for item in itemlist]
        if DEBUG: print "selection = ",sel
        #try:
        info = self.data.getKeyInfo(sel[0])
        #except:
        #    info, data = self.data.LoadSource(sel[0])
        self.cntTable.build(info['LabelNames'])
        self.mcaTable.build(info)
        self.emit(qt.SIGNAL("scanSelection"), (sel))

    def __doubleClicked(self, item):
        if DEBUG:print "__doubleCliked"
        if item is not None:
            sn  = str(item.text(1))
            dict={}
            dict['Key']      = sn
            dict['Command']  = str(item.text(2))
            dict['NbPoints'] = int(str(item.text(3)))
            dict['NbMca']    = int(str(item.text(4)))
            self.emit(qt.SIGNAL("scanDoubleClicked"), dict)

    if qt.qVersion() < '4.0.0':        
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
            if qt.qVersion() > '4.0.0': 
                idx = self.menu_idx
        if DEBUG:
            print "Scan information:"
            print self.data.GetSourceInfo(self.scans[idx])
        info, data = self.data.LoadSource(self.scans[idx])
        self.dataInfoWidget= SpecFileDataInfo.SpecFileDataInfo(info)
        self.dataInfoWidget.show()

def test():
    a = qt.QApplication(sys.argv)
    if 0:
        w = ScanList()
        if len(sys.argv) > 1:
            d.SetSource(sys.argv[1])
        else:
            d.SetSource('03novs060sum.mca')
        w.setData(d)    
    elif 0:
        import SpecFileLayer
        d = SpecFileLayer.SpecFileLayer()
        w= FileSelector()
        def mySlot(ddict):
            print ddict
            if ddict["event"] == "NewSourceSelected":
                d.SetSource(ddict["sourcelist"][0])
                w.specfileWidget.setData(d)
        a.connect(w, qt.SIGNAL("SourceSelectorSignal"),
                           mySlot)
    else:
        #new access
        import DataSource
        w= QSourceSelector()
        def mySlot(ddict):
            print ddict
            if ddict["event"] == "NewSourceSelected":
                d = DataSource.DataSource(ddict["sourcelist"][0])
                w.specfileWidget.setDataSource(d)
                if QTVERSION < '4.0.0':
                    a.connect(w, qt.PYSIGNAL("SourceSelectorSignal"),
                          mySlot)
                else:
                    a.connect(w, qt.SIGNAL("SourceSelectorSignal"),
                           mySlot)

        
    qt.QObject.connect(a, qt.SIGNAL("lastWindowClosed()"),
              a, qt.SLOT("quit()"))

    if QTVERSION < '4.0.0':
        w.show()
        a.exec_loop()
    else:
        w.show()
        a.exec_()


if __name__=="__main__":
    test()
