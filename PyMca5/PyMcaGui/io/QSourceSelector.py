#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()
from PyMca5.PyMcaGui import PyMca_Icons as icons
from PyMca5.PyMcaIO import spswrap as sps
from PyMca5 import PyMcaDirs

DEBUG = 0

class QSourceSelector(qt.QWidget):
    sigSourceSelectorSignal = qt.pyqtSignal(object)
    def __init__(self, parent=None, filetypelist=None, pluginsIcon=False):
        qt.QWidget.__init__(self, parent)
        self.mainLayout= qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        if filetypelist is None:
            self.fileTypeList = ["Spec Files (*mca)",
                                "Spec Files (*dat)",
                                "Spec Files (*spec)",
                                "SPE Files (*SPE)",
                                "EDF Files (*edf)",
                                "EDF Files (*ccd)",
                                "CSV Files (*csv)",
                                "All Files (*)"]
        else:
            self.fileTypeList = filetypelist
        self.lastFileFilter = self.fileTypeList[0]

        # --- file combo/open/close
        self.lastInputDir = PyMcaDirs.inputDir
        self.fileWidget= qt.QWidget(self)
        fileWidgetLayout= qt.QHBoxLayout(self.fileWidget)
        fileWidgetLayout.setContentsMargins(0, 0, 0, 0)
        fileWidgetLayout.setSpacing(0)
        self.fileCombo  = qt.QComboBox(self.fileWidget)
        self.fileCombo.setEditable(0)
        self.mapCombo= {}
        openButton= qt.QToolButton(self.fileWidget)

        self.openIcon   = qt.QIcon(qt.QPixmap(icons.fileopen))
        self.closeIcon  = qt.QIcon(qt.QPixmap(icons.fileclose))
        self.reloadIcon = qt.QIcon(qt.QPixmap(icons.reload_))
        self.specIcon   = qt.QIcon(qt.QPixmap(icons.spec))

        openButton.setIcon(self.openIcon)
        openButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        openButton.setToolTip("Open new file data source")

        closeButton= qt.QToolButton(self.fileWidget)
        closeButton.setIcon(self.closeIcon)
        closeButton.setToolTip("Close current data source")

        refreshButton= qt.QToolButton(self.fileWidget)
        refreshButton.setIcon(self.reloadIcon)
        refreshButton.setToolTip("Refresh data source")

        specButton= qt.QToolButton(self.fileWidget)
        specButton.setIcon(self.specIcon)
        specButton.setToolTip("Open new shared memory source")

        closeButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        specButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))
        refreshButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        openButton.clicked.connect(self._openFileSlot)
        closeButton.clicked.connect(self.closeFile)
        refreshButton.clicked.connect(self._reload)

        specButton.clicked.connect(self.openSpec)
        self.fileCombo.activated[str].connect(self._fileSelection)

        fileWidgetLayout.addWidget(self.fileCombo)
        fileWidgetLayout.addWidget(openButton)
        fileWidgetLayout.addWidget(closeButton)
        fileWidgetLayout.addWidget(specButton)
        if sys.platform == "win32":specButton.hide()
        fileWidgetLayout.addWidget(refreshButton)
        self.specButton = specButton
        if pluginsIcon:
            self.pluginsButton = qt.QToolButton(self.fileWidget)
            self.pluginsButton.setIcon(qt.QIcon(qt.QPixmap(icons.plugin)))
            self.pluginsButton.setToolTip("Plugin handling")
            fileWidgetLayout.addWidget(self.pluginsButton)
        self.mainLayout.addWidget(self.fileWidget)

    def _reload(self):
        if DEBUG:
            print("_reload called")
        qstring = self.fileCombo.currentText()
        if not len(qstring):
            return

        key = qt.safe_str(qstring)
        ddict = {}
        ddict["event"] = "SourceReloaded"
        ddict["combokey"] = key
        ddict["sourcelist"] = self.mapCombo[key] * 1
        self.sigSourceSelectorSignal.emit(ddict)

    def _openFileSlot(self):
        self.openFile(None, None)

    def openSource(self, sourcename, specsession=None):
        if specsession is None:
            if sourcename in sps.getspeclist():
                specsession=True
            else:
                specsession=False
        self.openFile(sourcename, specsession=specsession)

    def openFile(self, filename=None,justloaded=None, specsession = False):
        if DEBUG:
            print("openfile = ",filename)
        staticDialog = False
        if not specsession:
            if justloaded is None:
                justloaded = True
            if filename is None:
                if self.lastInputDir is not None:
                    if not os.path.exists(self.lastInputDir):
                        self.lastInputDir = None
                wdir = self.lastInputDir
                if wdir is None:
                    wdir = os.getcwd()
                if (sys.version < '3.0') and PyMcaDirs.nativeFileDialogs:
                    filetypes = self.fileTypeList[0]
                    for filetype in self.fileTypeList[1:]:
                        filetypes += ";;" + filetype
                    try:
                        # API 1
                        filelist = qt.QFileDialog.getOpenFileNames(self,
                                "Open a new source file",
                                wdir,
                                filetypes,
                                self.lastFileFilter)
                    except:
                        # API 2
                        filelist, self.lastFileFilter =\
                                qt.QFileDialog.getOpenFileNamesAndFilter(\
                                self,
                                "Open a new source file",
                                wdir,
                                filetypes,
                                self.lastFileFilter)
                    staticDialog = True
                else:
                    fdialog = qt.QFileDialog(self)
                    fdialog.setModal(True)
                    fdialog.setWindowTitle("Open a new source file")
                    if hasattr(qt, "QStringList"):
                        strlist = qt.QStringList()
                    else:
                        strlist = []
                    for filetype in self.fileTypeList:
                        strlist.append(filetype)
                    if QTVERSION < '5.0.0':
                        fdialog.setFilters(strlist)
                        fdialog.selectFilter(self.lastFileFilter)
                    else:
                        fdialog.setNameFilters(strlist)
                        fdialog.selectNameFilter(self.lastFileFilter)                        
                    fdialog.setFileMode(fdialog.ExistingFiles)
                    fdialog.setDirectory(wdir)
                    ret = fdialog.exec_()
                    if ret == qt.QDialog.Accepted:
                        filelist = fdialog.selectedFiles()
                        if QTVERSION < '5.0.0':
                            self.lastFileFilter = qt.safe_str(\
                                                    fdialog.selectedFilter())
                        else:
                            self.lastFileFilter = qt.safe_str(\
                                                    fdialog.selectedNameFilter())
                        fdialog.close()
                        del fdialog
                    else:
                        fdialog.close()
                        del fdialog
                        return
                #filelist.sort()
                filename=[]
                for f in filelist:
                    filename.append(qt.safe_str(f))
                if not len(filename):
                    return
                if len(filename):
                    self.lastInputDir  = os.path.dirname(filename[0])
                    PyMcaDirs.inputDir = os.path.dirname(filename[0])
                    if staticDialog:
                        if len(filename[0]) > 3:
                            #figure out the selected filter
                            extension = filename[0][-3:]
                            self.lastFileFilter = self.fileTypeList[-1]
                            for fileFilter in self.fileTypeList:
                                if extension == fileFilter[-4:-1]:
                                    self.lastFileFilter = fileFilter
                                    break
                justloaded = True
            if justloaded:
                if type(filename) != type([]):
                    filename = [filename]
            if not os.path.exists(filename[0]):
                if '%' not in filename[0]:
                    raise IOError("File %s does not exist" % filename[0])

            #check if it is a stack
            if len(filename) > 1:
                key = "STACK from %s to %s" % (filename[0], filename[-1])
            else:
                key = os.path.basename(filename[0])
        else:
            key = filename
            if key not in sps.getspeclist():
                qt.QMessageBox.critical(self,
                                    "SPS Error",
                                    "No shared memory source named %s" % key)
                return
        ddict = {}
        ddict["event"] = "NewSourceSelected"
        if key in self.mapCombo.keys():
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
            self.fileCombo.insertItem(nitems, key)
            self.fileCombo.setCurrentIndex(nitems)
        else:
            if hasattr(qt, "QString"):
                nitem = self.fileCombo.findText(qt.QString(key))
            else:
                nitem = self.fileCombo.findText(key)
            self.fileCombo.setCurrentIndex(nitem)
        self.sigSourceSelectorSignal.emit(ddict)

    def closeFile(self):
        if DEBUG:
            print("closeFile called")
        #get current combobox key
        qstring = self.fileCombo.currentText()
        if not len(qstring):
            return
        key = qt.safe_str(qstring)
        ddict = {}
        ddict["event"] = "SourceClosed"
        ddict["combokey"] = key
        ddict["sourcelist"] = self.mapCombo[key] * 1
        if hasattr(qt, "QString"):
            nitem = self.fileCombo.findText(qt.QString(key))
        else:
            nitem = self.fileCombo.findText(key)
        self.fileCombo.removeItem(nitem)
        del self.mapCombo[key]
        self.sigSourceSelectorSignal.emit(ddict)

    def openSpec(self):
        speclist = sps.getspeclist()
        if not len(speclist):
            qt.QMessageBox.information(self,
                    "No SPEC Shared Memory Found",
                    "No shared memory source available")
            return
        if QTVERSION < '4.0.0':
            print("should I keep Qt3 version?")
            return
        menu = qt.QMenu()
        for spec in speclist:
            if hasattr(qt, "QString"):
                menu.addAction(qt.QString(spec),
                        lambda i=spec:self.openFile(i, specsession=True))
            else:
                menu.addAction(spec,
                        lambda i=spec:self.openFile(i, specsession=True))
        menu.exec_(self.cursor().pos())

    def _fileSelection(self, qstring):
        if DEBUG:
            print("file selected ", qstring)
        key = str(qstring)
        ddict = {}
        ddict["event"] = "SourceSelected"
        ddict["combokey"] = key
        ddict["sourcelist"] = self.mapCombo[key]
        self.sigSourceSelectorSignal.emit(ddict)

def test():
    a = qt.QApplication(sys.argv)
    #new access
    from PyMca5.PyMcaGui.pymca import QDataSource
    w= QSourceSelector()
    def mySlot(ddict):
        print(ddict)
        if ddict["event"] == "NewSourceSelected":
            d = QDataSource.QDataSource(ddict["sourcelist"][0])
            w.specfileWidget.setDataSource(d)
            w.sigSourceSelectorSignal.connect(mySlot)

    a.lastWindowClosed.connect(a.quit)

    w.show()
    a.exec_()


if __name__=="__main__":
    test()
