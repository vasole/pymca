#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
import PyMca_Icons as icons
import os.path
import QtBlissGraph
from QtBlissGraph import qt
if qt.qVersion() > '4.0.0':
    qt.PYSIGNAL = qt.SIGNAL
    QT4 = True
else:
    QT4 = False
import numpy.oldnumeric as Numeric
import sys
import ColormapDialog
DEBUG = 0
SOURCE_TYPE = 'EdfFile'
__revision__ = "$Revision: 1.35 $"



class EdfFile_StandardArray(qt.QWidget):
    def __init__(self, parent=None, name="Edf_StandardArray", fl=0, images=None, rows=None, cols=None):
        if images is None:images = 1
        if rows is None:rows = 0
        if cols is None:cols = 0        
        qt.QWidget.__init__(self, parent)
        if qt.qVersion() < '4.0.0':
            layout = qt.QGridLayout(self, 4, 2)
            layout.setColStretch(0,0)
            layout.setColStretch(1,1)
        else:
            layout = qt.QGridLayout(self)
        layout.setMargin(5)

        ilab= qt.QLabel("Image:", self)
        self.plab= qt.QLabel("Plot", self)
        self.ylab= qt.QLabel("Columns :", self)

        layout.addWidget(ilab, 0, 0, qt.Qt.AlignRight)
        layout.addWidget(self.plab, 1, 0, qt.Qt.AlignRight)
        layout.addWidget(self.ylab, 2, 0, qt.Qt.AlignRight|qt.Qt.AlignTop)

        self.iCombo= qt.QComboBox(self)
        self.iCombo.setEditable(0)

        self.plotCombo= qt.QComboBox(self)
        self.plotCombo.setEditable(0)
        if qt.qVersion() < '4.0.0':
            self.plotCombo.insertItem("Rows")
            self.plotCombo.insertItem("Columns")
            self.yList= qt.QListBox(self)
            self.yList.setSelectionMode(qt.QListBox.Multi)
        else:
            self.plotCombo.insertItems(0, ["Rows", "Columns"])
            self.yList= qt.QListWidget(self)
            #self.yList.setSelectionMode(qt.QListBox.Multi)


        layout.addWidget(self.iCombo,   0, 1)
        layout.addWidget(self.plotCombo,1, 1)
        layout.addWidget(self.yList,    2, 1)

        self.connect(self.plotCombo, qt.SIGNAL("activated(int)"), self.__plotChanged)
        self.connect(self.iCombo, qt.SIGNAL("activated(int)"),    self.__iChanged)
        self.setImages(images)

        self.setSize(rows, cols)

    def setImages(self,images,info=None):
        self.iCombo.clear()
        if info is None: info = []
        for i in range(images):
            if qt.qVersion() < '4.0.0':
                if len(info) == images:
                    self.iCombo.insertItem("Image %d Key %s" % (i,info[i]))            
                else:
                    self.iCombo.insertItem("Image %d" % i)
            else:
                if len(info) == images:
                    self.iCombo.insertItem(i, "Image %d Key %s" % (i,info[i])) 
                else:
                    self.iCombo.insertItem(i, "Image %d" % i)

    def setCurrentImage(self,image):
        if image < self.iCombo.count():
            if QT4:self.iCombo.setCurrentIndex(image)
            else:  self.iCombo.setCurrentItem(image)
    
    def setSize(self, rows, cols):
        self.rows= rows
        self.cols= cols

        idx= self.cols<=self.rows
        if qt.qVersion() < '4.0.0':
            self.plotCombo.setCurrentItem(idx)
        else:
            self.plotCombo.setCurrentIndex(idx)
        self.__plotChanged(idx)

    def __plotChanged(self, index):
        if index==1:        
            self.ylab.setText('Columns')
            txt= "Column"
            val= self.cols
        else:
            self.ylab.setText('Rows')
            txt= "Row"
            val= self.rows
        if qt.qVersion() < '4.0.0': self.yList.clear()
        for x in range(val):
            if QT4:self.yList.addItem("%s %d"%(txt,x))
            else:  self.yList.insertItem("%s %d"%(txt,x))
        dict={}
        dict['event'] = "plotChanged"
        dict['plot']  =  txt+"s"
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("widgetSignal"),(dict,))
        else:
            self.emit(qt.SIGNAL("widgetSignal"),(dict))

    def __iChanged(self, index):
        dict={}
        dict['event'] = "imageChanged"
        dict['index'] =  index
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("widgetSignal"),(dict,))
        else:
            self.emit(qt.SIGNAL("widgetSignal"),(dict))

    def getSelection(self):
        selection= []

        idx= self.plotCombo.currentItem()
        if idx==1: plot= "cols"
        else: plot= "rows"

        if qt.qVersion() < '4.0.0':
            idx = self.iCombo.currentItem()
        else:
            idx = self.iCombo.currentIndex()
        if idx==0: image= None
        else: image= idx-1

        ylist= [ idx for idx in range(self.yList.count()) if self.yList.isSelected(idx) ]
        for y in ylist:
            selection.append({"plot":plot, "image": image,"x":None, "y":y})
        return selection

    def markImageSelected(self,imagelist=[]):
        if qt.qVersion() < '4.0.0':
            current = self.iCombo.currentItem()
        else:
            current = self.iCombo.currentIndex()
        images  = self.iCombo.count()
        #self.iCombo.clear()
        msg = " (selected)"
        for i in range(images):
            index = "%d" % i
            if qt.qVersion() < '4.0.0':
                text = str(self.iCombo.text(i)).split(msg)[0]
            else:
                text = str(self.iCombo.itemText(i)).split(msg)[0]
            key  = text.split()[-1]
            if qt.qVersion() < '4.0.0':
                if key in imagelist:
                    self.iCombo.changeItem("%s%s" % (text,msg),i)
                else:
                    self.iCombo.changeItem("%s" % (text),i)
            else:
                if key in imagelist:
                    self.iCombo.setItemText(i, "%s%s" % (text,msg))
                else:
                    self.iCombo.setItemText(i, "%s" % (text))
        if qt.qVersion() < '4.0.0':
            self.iCombo.setCurrentItem(current)
        else:
            self.iCombo.setCurrentIndex(current)
        
        
    def markRowSelected(self, rowlist=[]):
        if not str(self.plotCombo.currentText()) == "Rows":
            return
        current = self.yList.currentItem()
        n       = self.yList.count()
        self.yList.clear()
        for index in range(n):
            if qt.qVersion() < '4.0.0':
                if index in rowlist:
                    self.yList.insertItem(" Row %d (selected)" % index)
                else:
                    self.yList.insertItem(" Row %d" % index)
            else:
                if index in rowlist:
                    self.yList.addItem(" Row %d (selected)" % index)
                else:
                    self.yList.addItem(" Row %d" % index)
        #print "asking set"
        #self.yList.setCurrentItem(current)
        #print "DONE"
    
    def markColSelected(self, collist=[]):
        if not str(self.plotCombo.currentText()) == "Columns":
            return
        current = self.yList.currentItem()
        n       = self.yList.count()
        self.yList.clear()
        for index in range(n):
            if index in collist:
                self.yList.insertItem(" Column %d (selected)" % index)
            else:
                self.yList.insertItem(" Column %d" % index)            
        self.yList.setCurrentItem(current)
        

class EdfFileSelector(qt.QWidget):
    #ClassArrays= {"array": EdfFile_StandardArray}
    def __init__(self, parent=None, name="EdfSelector", fl=0, justviewer=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name, fl)
        else:
            qt.QWidget.__init__(self, parent)
        self.dataSource= None
        self.oldsource = ""
        self.oldcurrentArray = None
        self.data= None
        self.currentFile= None
        self.currentArray= 0
        self.selection= None
        self.__plotting = "Columns"
        self._edfstack = None
        self.lastInputDir = None
        self.colormapDialog = None
        self.colormap  = None
        self.selectPixmap= qt.QPixmap(icons.selected)
        self.unselectPixamp= qt.QPixmap(icons.unselected)

        mainLayout= qt.QVBoxLayout(self)

        # --- file combo/open/close
        fileWidget= qt.QWidget(self)
        self.fileCombo= qt.QComboBox(fileWidget)
        self.fileCombo.setEditable(0)
        self.mapComboName= {}
        openButton= qt.QToolButton(fileWidget)
        if QT4:
            self.openIcon= qt.QIcon(qt.QPixmap(icons.fileopen))
            self.closeIcon= qt.QIcon(qt.QPixmap(icons.fileclose))
            openButton.setIcon(self.openIcon)
        else:
            self.openIcon= qt.QIconSet(qt.QPixmap(icons.fileopen))
            self.closeIcon= qt.QIconSet(qt.QPixmap(icons.fileclose))
            openButton.setIconSet(self.openIcon)
        openButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        if qt.qVersion() < '3.0':
            self.colormapIcon= qt.QIconSet(qt.QPixmap(icons.colormap16))
        elif QT4:
            self.colormapIcon= qt.QIcon(qt.QPixmap(icons.colormap))        
        else:
            self.colormapIcon= qt.QIconSet(qt.QPixmap(icons.colormap))
        colormapButton= qt.QToolButton(fileWidget)
        if QT4:
            colormapButton.setIcon(self.colormapIcon)
            colormapButton.setToolTip("Shows Colormap Dialog")
        else:
            colormapButton.setIconSet(self.colormapIcon)
            qt.QToolTip.add(colormapButton, "Shows Colormap Dialog")
        colormapButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        closeButton= qt.QToolButton(fileWidget)
        if QT4:
            closeButton.setIcon(self.closeIcon)
        else:
            closeButton.setIconSet(self.closeIcon)
        closeButton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum))

        fileLayout= qt.QHBoxLayout(fileWidget)
        fileLayout.addWidget(self.fileCombo)
        fileLayout.addWidget(openButton)
        fileLayout.addWidget(colormapButton)
        fileLayout.addWidget(closeButton)

        self.connect(openButton, qt.SIGNAL("clicked()"), self.openFile)
        self.connect(colormapButton, qt.SIGNAL("clicked()"), self.selectColormap)
        self.connect(closeButton, qt.SIGNAL("clicked()"), self.closeFile)
        self.connect(self.fileCombo, qt.SIGNAL("activated(const QString &)"), self.__fileSelection)
        # --- splitter
        self.splitter= qt.QSplitter(self)
        if QT4:
            self.splitter.setOrientation(qt.Qt.Vertical)        
        else:
            self.splitter.setOrientation(qt.QSplitter.Vertical)


        # --- graph
        self.graph=QtBlissGraph.QtBlissGraph(self.splitter)
        #self.arrayList.plotimage()
        self.graph.setTitle('')
        self.graph.xlabel('Rows')
        self.graph.ylabel('Columns')
        self.connect(self.graph,qt.PYSIGNAL('QtBlissGraphSignal')  ,self.widgetSignal)
        self._x1Limit = self.graph.getx1axislimits()[-1]
        self._y1Limit = self.graph.gety1axislimits()[-1]
        #self.graph.hide()
        # --- array parameter
        if not justviewer:
            self.__dummyW = qt.QWidget(self.splitter)
            self.__dummyW.layout =qt.QVBoxLayout(self.__dummyW)
            if not QT4:
                self.applygroup = qt.QHButtonGroup(self.__dummyW,"")
                self.applytoone = qt.QCheckBox(self.applygroup)
                self.applytoone.setText("Apply to seen  image")
                self.applytoone.setChecked(1)
                self.applytoall = qt.QCheckBox(self.applygroup)
                self.applytoall.setText("Apply to all in file")
                self.applygroup.insert(self.applytoone,0)
                self.applygroup.insert(self.applytoall,1)
                self.applygroup.setExclusive(1)
                self.__dummyW.layout.addWidget(self.applygroup)            
                if qt.qVersion() > '3.0.0': self.applygroup.setFlat(1)
                self.applygroup.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Fixed))
            else:
                self.applygroup = qt.QButtonGroup(self.__dummyW)
            self.connect(self.applygroup,qt.SIGNAL("clicked(int)"),self.groupSignal)
        else:
            self.__dummyW = qt.QWidget(self.splitter)
            self.__dummyW.layout =qt.QVBoxLayout(self.__dummyW)
        self.paramWidget = EdfFile_StandardArray(self.__dummyW)
        self.__dummyW.layout.addWidget(self.paramWidget)
        self.connect(self.paramWidget,
                     qt.PYSIGNAL("widgetSignal"),
                     self.widgetSignal)
        if justviewer:
            self.paramWidget.plotCombo.hide()
            self.paramWidget.yList.hide()
            self.paramWidget.plab.hide()
            self.paramWidget.ylab.hide()
            self.getParamWidget = self.__getParamWidget
        self.allImages = 0
        # --- command buttons
        if not justviewer:
            butWidget= qt.QWidget(self)
            butLayout= qt.QHBoxLayout(butWidget)
            butWidget.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum))
            addButton= qt.QPushButton("Add", butWidget)
            removeButton= qt.QPushButton("Remove", butWidget)
            replaceButton= qt.QPushButton("Replace", butWidget)

            butLayout.addWidget(addButton)
            butLayout.addWidget(removeButton)
            butLayout.addWidget(replaceButton)
            butLayout.setMargin(5)

            self.connect(addButton,     qt.SIGNAL("clicked()"), self.__addClicked)
            self.connect(replaceButton, qt.SIGNAL("clicked()"), self.__replaceClicked)
            self.connect(removeButton,  qt.SIGNAL("clicked()"), self.__removeClicked)

        # --- main layout
        mainLayout.setMargin(5)
        mainLayout.setSpacing(5)

        mainLayout.addWidget(fileWidget)
        mainLayout.addWidget(self.splitter)
        if not justviewer:mainLayout.addWidget(butWidget)
        return

    def groupSignal(self,i):
        self.allImages = i

    def widgetSignal(self,dict={}):
        if dict.has_key('event'):
            if dict['event']    == 'plotChanged':
                self.__plotting = dict['plot']
                self.__refreshSelection()
            elif dict['event']    == 'MouseClick':
                row = min(int(round(dict['x'])), self._x1Limit - 1)
                col = min(int(round(dict['y'])), self._y1Limit - 1)
                if row < 0: row = 0
                if col < 0: col = 0
                if self.data.SourceName is None:return
                if self.selection is None:
                    self.selection = {}
                nsel = {}
                nsel['SourceType'] = SOURCE_TYPE
                nsel['SourceName'] = self.data.SourceName
                if self.currentArray == len(self.data.SourceInfo['KeyList']):
                    key = '0.0'
                else:
                    key = self.data.SourceInfo['KeyList'][self.currentArray] 
                if self.allImages:
                    arraynamelist = self.data.SourceInfo['KeyList']
                else:
                    arraynamelist = [key]
                for key in arraynamelist:
                    nsel['Key']        = key                    
                    if self.__plotting == 'Rows':
                        ptype = 'rows'
                        nsel[key] = {'rows':[{'y':row,'x':None}],'cols':[]}
                    else:
                        nsel[key] = {'rows':[],'cols':[{'y':col,'x':None}]}
                        ptype = 'cols'
                    if self.selection == {}:
                        self.setSelected([nsel],reset=0)
                        self.emit(qt.PYSIGNAL("addSelection"), ([nsel],))
                    elif not self.selection.has_key(nsel['SourceName']):
                        self.setSelected([nsel],reset=0)
                        self.emit(qt.PYSIGNAL("addSelection"), ([nsel],))
                    elif not self.selection[nsel['SourceName']].has_key(key):
                        self.setSelected([nsel],reset=0)
                        self.emit(qt.PYSIGNAL("addSelection"), ([nsel],))
                    elif len(self.selection[nsel['SourceName']][key][ptype]) == 0:
                        self.setSelected([nsel],reset=0)
                        self.emit(qt.PYSIGNAL("addSelection"), ([nsel],))
                    elif nsel[key][ptype][0] not in self.selection[nsel['SourceName']][key][ptype]:
                        self.setSelected([nsel],reset=0)
                        self.emit(qt.PYSIGNAL("addSelection"), ([nsel],))
                    else:
                        self.removeSelection([nsel])
            elif dict['event']  == 'imageChanged':
                if DEBUG:
                    print "Image changed"
                if dict['index'] != self.currentArray: 
                    self.currentArray = dict['index']
                    self.refresh()
                if DEBUG:
                    print "self.currentArray = ",self.currentArray


    def openFileOLD(self, filename=None):
        if DEBUG:
            print "openfile = ",filename
        if filename is None:
            filename= qt.QFileDialog(self,"Open a new EdfFile", 1)
            filename.setFilters("EdfFiles (*.edf)\nEdfFiles (*.mca)\nEdfFiles (*ccd)\nAll files (*)")
            if filename.exec_loop() == qt.QDialog.Accepted:
                filename= str(filename.selectedFile())
            else:
                return
            if not len(filename):    return
    
        if filename in self.mapComboName.keys():
            self.selectFile(filename)
        else:
            if not self.data.SetSource(filename):
                qt.QMessageBox.critical(self, "ERROR opening EdfFile",
                        "Cannot open following EdfFile:\n%s"%(filename))
            else:
                filename= self.data.SourceName
                self.mapComboName[filename]= os.path.basename(filename)
                self.fileCombo.insertItem(self.mapComboName[filename])
                self.selectFile(filename)

    def openFile(self, filename=None,justloaded=None):
        if DEBUG:
            print "openfile = ",filename
        if justloaded is None:justloaded = 0
        if filename is None:
            if self.lastInputDir is not None:
                if not os.path.exists(self.lastInputDir):
                    self.lastInputDir = None
            if QT4:
                fdialog = qt.QFileDialog(self)
                fdialog.setModal(True)
                fdialog.setWindowTitle("Open a new EdfFile")
                strlist = qt.QStringList()
                strlist.append("Config Files *edf")
                strlist.append("Config Files *ccd")
                strlist.append("All Files *")
                fdialog.setFilters(strlist)
                fdialog.setFileMode(fdialog.ExistingFiles)
                ret = fdialog.exec_()
                if ret == qt.QDialog.Accepted:
                    filelist = fdialog.selectedFiles()
                    fdialog.close()
                    del fdialog                        
                else:
                    fdialog.close()
                    del fdialog
                    return            
            elif sys.platform == 'win32':
                wdir = self.lastInputDir
                if wdir is None:wdir = ""
                filelist = qt.QFileDialog.getOpenFileNames("EdfFiles (*.edf)\nEdfFiles (*mca)\nEdfFiles (*ccd)\nAll files (*)",
                            wdir,
                            self,"openFile", "Open a new EdfFile")
            else:
                filedialog = qt.QFileDialog(self,"Open new EdfFile(s)",1)
                if self.lastInputDir is not None:filedialog.setDir(self.lastInputDir)
                filedialog.setMode(filedialog.ExistingFiles)
                filedialog.setFilters("EdfFiles (*.edf)\nEdfFiles (*.mca)\nEdfFiles (*ccd)\nAll files (*)")           
                if filedialog.exec_loop() == qt.QDialog.Accepted:
                    filelist= filedialog.selectedFiles()
                else:
                    return  
            filelist.sort()
            filename=[]
            for f in filelist:
                filename.append(str(f))
            if not len(filename):    return
            if len(filename):self.lastInputDir=os.path.dirname(filename[0])
            justloaded = 1
        if justloaded:
            if type(filename) != type([]):
                filename = [filename]
        if not os.path.exists(filename[0]):
            raise "IOError",("File %s does not exist" % filename[0])
        if (justloaded) and (filename in self.mapComboName.keys()):
            self.selectFile(filename,justloaded=justloaded)
        else:
            if not self.data.SetSource(filename):
                qt.QMessageBox.critical(self, "ERROR opening EdfFile",
                        "Cannot open following EdfFile:\n%s"%(filename))
            else:
                filename= self.data.SourceName.split("|")
                if len(filename) > 1:
                    combokey = 'EDF Stack'
                    self._edfstack = filename
                else:
                    combokey = os.path.basename(filename[0])
                if combokey not in self.mapComboName.keys():
                    self.mapComboName[combokey]= filename[0]
                    if QT4:
                        print "self.fileCombo =",self.fileCombo 
                        self.fileCombo.addItem(combokey)                    
                    else:
                        self.fileCombo.insertItem(combokey)
                self.selectFile(combokey,justloaded=justloaded)


    def selectFile(self, filename=None,justloaded=None):
        if justloaded is None:justloaded=0
        if filename is not None:
            #if str(self.fileCombo.currentText())!=self.mapComboName[filename]:
            if 1:
              for idx in range(self.fileCombo.count()):
                #if str(self.fileCombo.text(idx))==self.mapComboName[filename]:
                if QT4:itext = str(self.fileCombo.itemText(idx))
                else:  itext = str(self.fileCombo.text(idx)) 
                if itext ==filename:
                    if QT4:self.fileCombo.setCurrentIndex(idx)
                    else:self.fileCombo.setCurrentItem(idx)
                    break
            if filename == 'EDF Stack':filename= self._edfstack
            else: filename = self.mapComboName[filename]
            self.data.SetSource(filename)
            if justloaded and (filename==self._edfstack):
                self.currentArray=len(self.data.GetSourceInfo()['KeyList'])
            else:
                self.currentArray=0
        self.refresh()
    
    def selectColormap(self):
        if self.colormap is None: return
        if self.colormapDialog.isHidden():
            self.colormapDialog.show()
        if qt.qVersion() < '4.0.0':self.colormapDialog.raiseW()
        else:  self.colormapDialog.raise_()          
        self.colormapDialog.show()

    def updateColormap(self, *var):
        if len(var) > 5:
            self.colormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        else:
            self.colormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        self.graph.imagePlot(self.lastData, colormap = self.colormap)
        self.graph.replot()

    def closeFile(self, filename=None):
        if filename is None:
            file= str(self.fileCombo.currentText())
            #if file != "EDF Stack":
            #    filename = self.mapComboName[file]
            #else:
            #    filename ="EDF Stack"
            filename = file

        #print self.selection
        if self.selection is not None and self.selection.has_key(filename):
            nmca = 0
            for key in self.selection[filename].keys():
                nmca += len(self.selection[filename][key]['rows']) + len(self.selection[filename][key]['cols'])
            if nmca:
                msg= "%d mca are linked to that EdfFile source.\n"% nmca
                msg+="Do you really want to delete all these graphs ??"
                ans= qt.QMessageBox.information(self, "Remove SpecFile %s"%filename, msg,
                        qt.QMessageBox.No, qt.QMessageBox.Yes)
                if ans==qt.QMessageBox.No: return
                try:
                    self.emit(qt.PYSIGNAL("delSelection"), (self.data.SourceName, mcakeys))
                except:
                    print "This is to be implemented"
                            
        for idx in range(self.fileCombo.count()):
            if qt.qVersion() < '4.0.0':
                itext = self.fileCombo.text(idx)
            else:
                itext = self.fileCombo.itemText(idx)            
            if filename == "EDF Stack":
                if itext == filename:
                    self.fileCombo.removeItem(idx)
                    del self.mapComboName[filename]
                    break
            elif str(itext)==os.path.basename(self.mapComboName[filename]):
                self.fileCombo.removeItem(idx)
                del self.mapComboName[filename]
                break

        if not self.fileCombo.count():
            self.data.SourceName = None
            self.graph.plotImage = None
            self.oldsource = None
            self.graph.clearmarkers()
            self.graph.replot()
            wid = self.__getParamWidget('array')
            wid.setImages(1)
            wid.setSize(0,0)
            #self.selectFile()
        else:
            self.selectFile(self.mapComboName.keys()[0])

    def __fileSelection(self, file):
        file= str(file)
        for filename, comboname in self.mapComboName.items():
            if filename==file:
                self.selectFile(filename)
                break
 


    def setData(self,data=None):
        if DEBUG:
            print "setData(self, data) called"
            print "data = ",data
        self.data= data
        self.refresh()

    def refresh(self):
        if DEBUG:
            print "refresh method called"
        if self.data is None or self.data.SourceName is None:    return
        self.currentFile = self.data.SourceName
        #this gives the number of images in the file
        infoSource= self.data.GetSourceInfo()
        if DEBUG:
            print "info =",infoSource
        
        nimages=len(infoSource['KeyList'])
        #print self.data.SourceName,"nimages = ",nimages
        loadsum = 0
        if self.currentArray > nimages:
            self.currentArray =  0
        elif self.currentArray == nimages:
            loadsum=1            
        #print "SUM = ",loadsum, infoSource['KeyList']
        #print self.currentArray
        if (self.oldsource != self.currentFile) or (self.oldcurrentArray != self.currentArray):
            if DEBUG:
                print "I have to read again ... "
            if not loadsum:
                info,data = self.data.LoadSource(infoSource['KeyList'][self.currentArray])
                imageinfo = infoSource['KeyList']
            else:
                #print "Loading the sum"
                info,data = self.data.LoadSource('0.0')            
                imageinfo = infoSource['KeyList']
            wid= self.__getParamWidget("array")
            if nimages > 1:
                if info.has_key('Title'):
                    i = 0
                    for key in self.data.GetSourceInfo()['KeyList']:
                        source,image = key.split(".")
                        source = int(source)
                        image  = int(image)
                        header = self.data.Source[(source-1)].GetHeader(image-1)
                        if header.has_key('Title'):
                            imageinfo[i] +=  header['Title']
                        i+=1
                wid.setImages(nimages+1,info = imageinfo+["0.0 - SUM"])
            else:
                if info.has_key('Title'):imageinfo [self.currentArray] += info['Title']  
                wid.setImages(nimages,  info = imageinfo)                
            wid.setCurrentImage(self.currentArray)
            #P.B. -> pointer(a,d1,d2,i1,i2) = a+ (i1+i2 * d1) 
            wid.setSize(int(info["Dim_2"]), int(info["Dim_1"]))
            if DEBUG:
                print "Image size = ",info["Dim_1"],"x",info["Dim_2"]
                print "data  size = ",Numeric.shape(data) 


            if self.graph.isHidden():
                self.graph.show()
            self.graph.setx1axislimits(0, int(info["Dim_2"]))
            self.graph.sety1axislimits(0, int(info["Dim_1"]))
            self._x1Limit = int(info["Dim_2"])
            self._y1Limit = int(info["Dim_1"])
            self.graph.clear()
            minData = Numeric.minimum.reduce(Numeric.minimum.reduce(data))
            maxData = Numeric.maximum.reduce(Numeric.maximum.reduce(data))
            wasnone = 0
            self.lastData = data
            if self.colormapDialog is None:
                wasnone = 1
                self.colormapDialog = ColormapDialog.ColormapDialog()                
                self.colormapDialog.colormapIndex  = self.colormapDialog.colormapList.index("Temperature")
                self.colormapDialog.colormapString = "Temperature"
                self.connect(self.colormapDialog, qt.PYSIGNAL("ColormapChanged"),
                             self.updateColormap)
            self.colormapDialog.setDataMinMax(minData, maxData)
            if wasnone:
                self.colormapDialog.setAutoscale(1)
                self.colormapDialog.setColormap(self.colormapDialog.colormapIndex)
            self.colormap = (self.colormapDialog.colormapIndex, self.colormapDialog.autoscale,
                                 self.colormapDialog.minValue, 
                                 self.colormapDialog.maxValue,
                                 minData, maxData)
            self.graph.plotimage(data=data, colormap = self.colormap)
            self.colormapDialog._update()
        self.__refreshSelection()
        self.graph.replot()
        self.oldsource       = "%s" % self.data.SourceName
        self.oldcurrentArray = self.currentArray * 1
        
    def __getParamWidget(self, widtype):
        return self.paramWidget

    def __replaceClicked(self):
        if DEBUG:
            print "replace clicked"
        selkeys= self.__getSelectedKeys()
        if len(selkeys):
            #self.eh.event(self.repEvent, selkeys)
            if DEBUG:
                print "Replace event"
            if self.allImages:
                arraynamelist = self.data.SourceInfo['KeyList']
            else:
                arraynamelist = []
                for selection in selkeys:
                    arraynamelist.append(selection['Key'])
            sellist=[]
            for arrayname in arraynamelist:
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
                sellist.append(sel)
            self.setSelected(sellist,reset=1)
            self.emit(qt.PYSIGNAL("replaceSelection"), (sellist,))

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
            if self.allImages:
                arraynamelist = self.data.SourceInfo['KeyList']
            else:
                arraynamelist = []
                for selection in selkeys:
                    arraynamelist.append(selection['Key'])
            sellist=[]
            for arrayname in arraynamelist:
                sel = {}
                sel['SourceType'] = SOURCE_TYPE            
                for selection in selkeys:
                    if not sel.has_key('SourceName'):
                        sel['SourceName'] = selection['SourceName']
                    #arrayname = selection['Key']
                    if not sel.has_key('Key'):
                        sel['Key'] = arrayname
                    if not sel.has_key(arrayname):
                        sel[arrayname] = {'rows':[],'cols':[]}
                    if selection['plot'] == 'cols':
                         sel[arrayname]['cols'].append({'x':selection['x'],
                                                        'y':selection['y']})
                    if selection['plot'] == 'rows':
                         sel[arrayname]['rows'].append({'x':selection['x'],
                                                        'y':selection['y']})
                #print "sel = ",sel
                sellist.append(sel)
            if self.selection is None: 
                self.setSelected(sellist,reset=1)
            else:
                self.setSelected(sellist,reset=0)
            self.emit(qt.PYSIGNAL("addSelection"), (sellist,))
            
    def __getSelectedKeys(self):
        selkeys= []
        parwid= self.paramWidget
        #.visibleWidget()
        if self.currentArray is not None:
            for sel in parwid.getSelection():
                sel["SourceName"]= self.currentFile
                sel['SourceType'] = SOURCE_TYPE
                if 0:
                    sel["Key"]= "%d" % self.currentArray
                else:
                    if self.currentArray == len(self.data.SourceInfo['KeyList']):
                        sel["Key"]= "0.0"
                    else:
                        sel["Key"]= self.data.SourceInfo['KeyList'][self.currentArray]                
                selkeys.append(sel)
        return selkeys

    def __removeClicked(self):
        if DEBUG:
            print "remove clicked"
        selkeys= self.__getSelectedKeys()
        returnedselection=[]
        if len(selkeys):
            #self.eh.event(self.delEvent, selkeys)
            if DEBUG:
                print "Remove Event"
                print "self.selection before = ",self.selection
            if self.allImages:
                arraynamelist = self.data.SourceInfo['KeyList']
            else:
                arraynamelist = []
                for selection in selkeys:
                    arraynamelist.append(selection['Key'])
            for arrayname in arraynamelist:
                for selection in selkeys:
                    sel = {}
                    sel['SourceName'] = selection['SourceName']
                    sel['SourceType'] = SOURCE_TYPE            
                    #sel['Key'] = selection['Key']
                    #arrayname = "%s" % selection['Key']
                    sel['Key'] = arrayname
                    sel[arrayname] = {'rows':[],'cols':[]}
                    if selection['plot'] == 'cols':
                         sel[arrayname]['cols'].append({'x':selection['x'],'y':selection['y']})
                    if selection['plot'] == 'rows':
                         sel[arrayname]['rows'].append({'x':selection['x'],'y':selection['y']})
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
                    returnedselection.append(sel)
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
            if len(specname) == 1:specname=specname[0]
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
                sel['SourceType']   = 'EdfFile'
                sel['Key']          = arraykey
                sel[arraykey]        = self.selection[sourcekey][arraykey]
                selection.append(sel)
        return selection

        
    def __refreshSelection(self):
        if DEBUG:
            print "__refreshSelection(self) called"
            print self.selection
            print "self.data.SourceName = ",self.data.SourceName
        if self.selection is not None:
            if self.data.SourceName is None: return
            if "|" in self.data.SourceName:
                #print "here should be the multiple" 
                #sel = self.selection.get(self.data.SourceName[0], {})
                sel = self.selection.get(self.data.SourceName, {})
            else:
                sel = self.selection.get(self.data.SourceName, {})
            selkeys = []
            for key in sel.keys():
                if (sel[key]['rows'] != []) or (sel[key]['cols'] !=  []):
                    selkeys.append(key)
            if DEBUG:
                print "selected images =",selkeys,"but self.selection = ",self.selection
                print "and self.selection.get(self.data.SourceName, {}) =",sel
            
            wid = self.__getParamWidget("array")
            wid.markImageSelected(selkeys)
            #imagedict = sel.get("%d" % self.currentArray, {})
            if self.currentArray == len(self.data.SourceInfo['KeyList']):
                imagedict = sel.get("0.0",{})
            else:
                imagedict = sel.get(self.data.SourceInfo['KeyList'][self.currentArray],{})                
            if not imagedict.has_key('rows'):
                imagedict['rows'] = []
            if not imagedict.has_key('cols'):
                imagedict['cols'] = []
            rows = []
            for dict in imagedict['rows']:
                if dict.has_key('y'):
                    if dict['y'] not in rows:
                        rows.append(dict['y'])
            wid.markRowSelected(rows) 
            cols = []
            for dict in imagedict['cols']:
                if dict.has_key('y'):
                    if dict['y'] not in cols:
                        cols.append(dict['y'])            
            wid.markColSelected(cols)
            self.graph.clearmarkers()
            for i in rows:
                label = "R%d" % i
                marker=self.graph.insertx1marker(i,0.1,label=label)
                self.graph.setmarkercolor(marker,"white")
            for i in cols:
                label = "C%d" % i
                marker=self.graph.inserty1marker(0.1,i,label=label)
                self.graph.setmarkercolor(marker,"white")
            self.graph.replot()
            return

def test2():
    a= qt.QApplication(sys.argv)
    a.connect(a, qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))

    w = EdfFile_StandardArray()
    w.show()
    if qt.qVersion() < '4.0.0':
        a.exec_loop()
    else:
        a.exec_()
        

def test():
    import sys
    import EdfFileLayer
    def repSelection(sel):    print "replaceSelection", sel
    def removeSelection(sel): print "removeSelection", sel
    def addSelection(sel):    print "addSelection", sel

    a= qt.QApplication(sys.argv)
    a.connect(a, qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))

    w = EdfFileSelector(justviewer=0)
    #print w
    d = EdfFileLayer.EdfFileLayer()
    w.setData(d)
    qt.QObject.connect(w,qt.PYSIGNAL("addSelection"),addSelection)
    qt.QObject.connect(w,qt.PYSIGNAL("removeSelection"),removeSelection)
    qt.QObject.connect(w,qt.PYSIGNAL("replaceSelection"),repSelection)
    w.show()
    if qt.qVersion() < '4.0.0':
        a.exec_loop()
    else:
        a.exec_()

if __name__=="__main__":
    test()
 


