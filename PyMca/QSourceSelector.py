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
import Icons as icons
import os

DEBUG = 0

class QSourceSelector(qt.QWidget):
    def __init__(self, parent=None, filetypelist=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout= qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        if filetypelist is None:
            self.fileTypeList = ["Spec Files (*mca)",
                                "Spec Files (*dat)",
                                "Spec Files (*spec)",
                                "EDF Files (*edf)",
                                "EDF Files (*ccd)",
                                "All Files (*)"]
        else:
            self.fileTypeList = filetypelist
        self.lastFileFilter = qt.QString(self.fileTypeList[0])

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

        fileWidgetLayout.addWidget(self.fileCombo)
        fileWidgetLayout.addWidget(openButton)
        fileWidgetLayout.addWidget(closeButton)
        self.mainLayout.addWidget(self.fileWidget)

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
                for filetype in self.fileTypeList:
                    filetypes += filetype+"\n"
                if sys.platform == 'win32':
                    filelist = qt.QFileDialog.getOpenFileNames(filetypes,
                                wdir,
                                self,"openFile", "Open a new EdfFile")
                else:
                    filedialog = qt.QFileDialog(self,"Open new EdfFile(s)",1)
                    if self.lastInputDir is not None:
                        filedialog.setDir(self.lastInputDir)
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
                    for filetype in self.fileTypeList:
                        filetypes += filetype+"\n"
                    filelist = qt.QFileDialog.getOpenFileNames(self,
                                "Open a new source file",          wdir,
                                filetypes,
                                self.lastFileFilter)
                else:
                    fdialog = qt.QFileDialog(self)
                    fdialog.setModal(True)
                    fdialog.setWindowTitle("Open a new EdfFile")
                    strlist = qt.QStringList()
                    for filetype in self.fileTypeList:
                        strlist.append(filetype)
                    fdialog.setFilters(strlist)
                    fdialog.selectFilter(self.lastFileFilter)
                    fdialog.setFileMode(fdialog.ExistingFiles)
                    fdialog.setDirectory(wdir)
                    ret = fdialog.exec_()
                    if ret == qt.QDialog.Accepted:
                        filelist = fdialog.selectedFiles()
                        self.lastFileFilter = str(fdialog.selectedFilter())
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
            if len(filename):
                self.lastInputDir=os.path.dirname(filename[0])
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

def test():
    a = qt.QApplication(sys.argv)
    #new access
    import QDataSource
    w= QSourceSelector()
    def mySlot(ddict):
        print ddict
        if ddict["event"] == "NewSourceSelected":
            d = QDataSource.QDataSource(ddict["sourcelist"][0])
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
