#!/usr/bin/env python
__revision__ = "$Revision: 1.41 $"
###########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
#############################################################################
try:
    import PyQt4.Qt as qt
    qt.Qt.WDestructiveClose = "TO BE DONE"
except:
    import qt
import os
import sys
import McaAdvancedFitBatch
import EdfFileLayer
import SpecFileLayer
import time
from PyMca_Icons import IconDict
import McaCustomEvent
import EdfFileSimpleViewer
import QtMcaAdvancedFitReport
import HtmlIndex
ROIWIDTH = 100.
DEBUG = 0

class McaBatchGUI(qt.QWidget):
    def __init__(self,parent=None,name="PyMca batch fitting",fl=qt.Qt.WDestructiveClose,
                filelist=None,config=None,outputdir=None, actions=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            self.setWindowTitle(name)
        self._layout = qt.QVBoxLayout(self)
        self.__build(actions)               
        if filelist is None: filelist = []
        self.outputDir  = None
        self.configFile = None
        self.setFileList(filelist)
        self.setConfigFile(config)
        self.setOutputDir(outputdir)
    
    def __build(self,actions):
        self.__grid= qt.QWidget(self)
        self._layout.addWidget(self.__grid)
        #self.__grid.setGeometry(qt.QRect(30,30,288,156))
        if qt.qVersion() < '4.0.0':
            grid       = qt.QGridLayout(self.__grid,3,3,11,6)
            grid.setColStretch(0,0)
            grid.setColStretch(1,1)
            grid.setColStretch(2,0)
        else:
            grid       = qt.QGridLayout(self.__grid)
            grid.setMargin(11)
            grid.setSpacing(6)
        #input list
        listrow  = 0
        listlabel   = qt.QLabel(self.__grid)
        listlabel.setText("Input File list:")
        if qt.qVersion() < '4.0.0':
            listlabel.setAlignment(qt.QLabel.WordBreak | qt.QLabel.AlignVCenter)
            self.__listView   = qt.QTextView(self.__grid)
            self.__listView.setMaximumHeight(30*listlabel.sizeHint().height())
        else:
            self.__listView   = qt.QTextEdit(self.__grid)
            self.__listView.setMaximumHeight(30*listlabel.sizeHint().height())
        self.__listButton = qt.QPushButton(self.__grid)
        self.__listButton.setText('Browse')
        self.connect(self.__listButton,qt.SIGNAL('clicked()'),self.browseList) 
        grid.addWidget(listlabel,        listrow, 0, qt.Qt.AlignTop|qt.Qt.AlignLeft)
        grid.addWidget(self.__listView,  listrow, 1)
        grid.addWidget(self.__listButton,listrow, 2, qt.Qt.AlignTop|qt.Qt.AlignRight)
        
        #config file
        configrow = 1
        configlabel = qt.QLabel(self.__grid)
        configlabel.setText("Fit Configuration File:")
        if qt.qVersion() < '4.0.0':
            configlabel.setAlignment(qt.QLabel.WordBreak | qt.QLabel.AlignVCenter)
        self.__configLine = qt.QLineEdit(self.__grid)
        self.__configLine.setReadOnly(True)
        self.__configButton = qt.QPushButton(self.__grid)
        self.__configButton.setText('Browse')
        self.connect(self.__configButton,qt.SIGNAL('clicked()'),self.browseConfig) 
        grid.addWidget(configlabel,         configrow, 0, qt.Qt.AlignLeft)
        grid.addWidget(self.__configLine,   configrow, 1)
        grid.addWidget(self.__configButton, configrow, 2, qt.Qt.AlignLeft)


        #output dir
        outrow    = 2
        outlabel   = qt.QLabel(self.__grid)
        outlabel.setText("Output dir:")
        if qt.qVersion() < '4.0.0':
            outlabel.setAlignment(qt.QLabel.WordBreak | qt.QLabel.AlignVCenter)
        self.__outLine = qt.QLineEdit(self.__grid)
        self.__outLine.setReadOnly(True)
        #self.__outLine.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed))
        self.__outButton = qt.QPushButton(self.__grid)
        self.__outButton.setText('Browse')
        self.connect(self.__outButton,qt.SIGNAL('clicked()'),self.browseOutputDir) 
        grid.addWidget(outlabel,         outrow, 0, qt.Qt.AlignLeft)
        grid.addWidget(self.__outLine,   outrow, 1)
        grid.addWidget(self.__outButton, outrow, 2, qt.Qt.AlignLeft)


        box  = qt.QWidget(self)
        box.l = qt.QHBoxLayout(box)
        vbox1 = qt.QWidget(box)
        vbox1.l = qt.QVBoxLayout(vbox1)
        box.l.addWidget(vbox1)
        vbox2 = qt.QWidget(box)
        vbox2.l = qt.QVBoxLayout(vbox2)
        box.l.addWidget(vbox2)
        vbox3 = qt.QWidget(box)
        vbox3.l = qt.QVBoxLayout(vbox3)
        box.l.addWidget(vbox3)
        self.__fitBox = qt.QCheckBox(vbox1)
        self.__fitBox.setText('Generate .fit Files')
        palette = self.__fitBox.palette()
        #if qt.qVersion() < '4.0.0':
        #    palette.setDisabled(palette.active())
        #else:
        #    print "palette set disabled"
        self.__fitBox.setChecked(False)
        self.__fitBox.setEnabled(True)
        vbox1.l.addWidget(self.__fitBox)

        self.__imgBox = qt.QCheckBox(vbox2)
        self.__imgBox.setText('Generate Peak Images')
        palette = self.__imgBox.palette()
        if qt.qVersion() < '4.0.0':
            palette.setDisabled(palette.active())
        else:
            print "palette set disabled"
        self.__imgBox.setChecked(True)
        self.__imgBox.setEnabled(False)
        vbox2.l.addWidget(self.__imgBox)
        """
        self.__specBox = qt.QCheckBox(box)
        self.__specBox.setText('Generate Peak Specfile')
        palette = self.__specBox.palette()
        palette.setDisabled(palette.active())
        self.__specBox.setChecked(False)
        self.__specBox.setEnabled(False)
        """
        self.__htmlBox = qt.QCheckBox(vbox3)
        self.__htmlBox.setText('Generate Report')
        #palette = self.__htmlBox.palette()
        #palette.setDisabled(palette.active())
        self.__htmlBox.setChecked(False)
        self.__htmlBox.setEnabled(True)
        vbox3.l.addWidget(self.__htmlBox)
        
        #report options
        #reportBox = qt.QHBox(self)
        self.__tableBox = qt.QCheckBox(vbox1)
        self.__tableBox.setText('Table in Report')
        palette = self.__tableBox.palette()
        #if qt.qVersion() < '4.0.0':
        #    palette.setDisabled(palette.active())
        #else:
        #    print "palette set disabled"
        self.__tableBox.setChecked(True)
        self.__tableBox.setEnabled(False)
        vbox1.l.addWidget(self.__tableBox)

        self.__extendedTable = qt.QCheckBox(vbox2)
        self.__extendedTable.setText('Extended Table')
        self.__extendedTable.setChecked(True)
        self.__extendedTable.setEnabled(False)
        vbox2.l.addWidget(self.__extendedTable)
        
        self.__concentrationsBox = qt.QCheckBox(vbox3)
        self.__concentrationsBox.setText('Concentrations (SLOW!)')
        self.__concentrationsBox.setChecked(False)
        #self.__concentrationsBox.setEnabled(True)
        self.__concentrationsBox.setEnabled(False)
        vbox3.l.addWidget(self.__concentrationsBox)
        self._layout.addWidget(box)
        
        # other stuff
        bigbox   = qt.QWidget(self)
        bigbox.l = qt.QHBoxLayout(bigbox)
        
        vBox = qt.QWidget(bigbox)
        vBox.l = qt.QVBoxLayout(vBox)
        bigbox.l.addWidget(vBox)
        self.__overwrite = qt.QCheckBox(vBox)
        self.__overwrite.setText('Overwrite Fit Files')
        self.__overwrite.setChecked(True)
        vBox.l.addWidget(self.__overwrite)
        
        self.__useExisting = qt.QCheckBox(vBox)
        self.__useExisting.setText('Use Existing Fit Files')
        self.__useExisting.setChecked(False)
        vBox.l.addWidget(self.__useExisting)
        
        #self.useGroup.setExclusive(1)
        #if qt.qVersion() > '3.0.0':self.useGroup.setFlat(1)
        self.connect(self.__overwrite,   qt.SIGNAL("clicked()"),
                                         self.__clickSignal0)
        self.connect(self.__useExisting, qt.SIGNAL("clicked()"),
                                         self.__clickSignal1)
        self.connect(self.__concentrationsBox, qt.SIGNAL("clicked()"),
                                               self.__clickSignal2)
        self.connect(self.__htmlBox, qt.SIGNAL("clicked()"),
                                               self.__clickSignal3)
        
        boxStep0   = qt.QWidget(bigbox)
        boxStep0.l = qt.QVBoxLayout(boxStep0)
        boxStep  = qt.QWidget(boxStep0)
        boxStep.l= qt.QHBoxLayout(boxStep)
        boxStep0.l.addWidget(boxStep)
        bigbox.l.addWidget(boxStep0)
        
        boxFStep   = qt.QWidget(boxStep)
        boxFStep.l = qt.QHBoxLayout(boxFStep)
        boxStep.l.addWidget(boxFStep)
        label= qt.QLabel(boxFStep)
        label.setText("File Step:")
        self.__fileSpin = qt.QSpinBox(boxFStep)
        if qt.qVersion() < '4.0.0':
            self.__fileSpin.setMinValue(1)
            self.__fileSpin.setMaxValue(10)
        else:
            self.__fileSpin.setMinimum(1)
            self.__fileSpin.setMaximum(10)
        self.__fileSpin.setValue(1)
        boxFStep.l.addWidget(label)
        boxFStep.l.addWidget(self.__fileSpin)


        boxMStep   = qt.QWidget(boxStep0)
        boxMStep.l = qt.QHBoxLayout(boxMStep)
        boxStep0.l.addWidget(boxMStep)
        
        label= qt.QLabel(boxMStep)
        label.setText("MCA Step:")
        self.__mcaSpin = qt.QSpinBox(boxMStep)
        if qt.qVersion() < '4.0.0':
            self.__mcaSpin.setMinValue(1)
            self.__mcaSpin.setMaxValue(10)
        else:
            self.__mcaSpin.setMinimum(1)
            self.__mcaSpin.setMaximum(10)
        self.__mcaSpin.setValue(1)

        boxMStep.l.addWidget(label)
        boxMStep.l.addWidget(self.__mcaSpin)
        

        
        #box2 = qt.QHBox(self)
        self.__roiBox = qt.QCheckBox(vBox)
        self.__roiBox.setText('ROI Fitting Mode')
        vBox.l.addWidget(self.__roiBox)
        #box3 = qt.QHBox(box2)
        box3 = qt.QWidget(boxStep0)
        box3.l = qt.QHBoxLayout(box3)
        boxStep0.l.addWidget(box3)

        
        label= qt.QLabel(box3)
        label.setText("ROI Width (eV):")
        self.__roiSpin = qt.QSpinBox(box3)
        if qt.qVersion() < '4.0.0':
            self.__roiSpin.setMinValue(10)
            self.__roiSpin.setMaxValue(1000)
        else:
            self.__roiSpin.setMinimum(10)
            self.__roiSpin.setMaximum(1000)
        self.__roiSpin.setValue(ROIWIDTH)
        box3.l.addWidget(label)
        box3.l.addWidget(self.__roiSpin)
        
        self._layout.addWidget(bigbox)

        if actions: self.__buildActions()


    def __clickSignal0(self):
        if self.__overwrite.isChecked():
            self.__useExisting.setChecked(0)
        else:
            self.__useExisting.setChecked(1)

    def __clickSignal1(self):
        if self.__useExisting.isChecked():
            self.__overwrite.setChecked(0)
        else:
            self.__overwrite.setChecked(1)

    def __clickSignal2(self):
        self.__tableBox.setEnabled(True)

    def __clickSignal3(self):
        if self.__htmlBox.isChecked():
            self.__tableBox.setEnabled(True)
            self.__concentrationsBox.setEnabled(True)
            self.__fitBox.setChecked(True)
            self.__fitBox.setEnabled(False)
        else:
            self.__tableBox.setEnabled(False)
            self.__concentrationsBox.setEnabled(False)
            self.__fitBox.setChecked(False)
            self.__fitBox.setEnabled(True)


    def __buildActions(self):
        box = qt.QWidget(self)
        box.l = qt.QHBoxLayout(box)
        box.l.addWidget(HorizontalSpacer(box))
        self.__dismissButton = qt.QPushButton(box)
        box.l.addWidget(self.__dismissButton)
        box.l.addWidget(HorizontalSpacer(box))
        self.__dismissButton.setText("Close")
        self.__startButton   = qt.QPushButton(box)
        box.l.addWidget(self.__startButton)
        box.l.addWidget(HorizontalSpacer(box))
        self.__startButton.setText("Start")
        self.connect(self.__dismissButton,qt.SIGNAL("clicked()"),self.close)
        self.connect(self.__startButton,qt.SIGNAL("clicked()"),self.start)
        self._layout.addWidget(box)



    def setFileList(self,filelist=None):
        if filelist is None:filelist = []
        if True or self.__goodFileList(filelist):
            text = ""
            oldtype = None
            filelist.sort()
            for file in filelist:
                filetype = self.__getFileType(file)
                if filetype is None:return
                if oldtype  is None: oldtype = filetype
                if oldtype != filetype:
                    qt.QMessageBox.critical(self, "ERROR",
                        "Type %s does not match type %s on\n%s"% (filetype,oldtype,file))
                    return           
                text += "%s\n" % file
            self.fileList = filelist
            if qt.qVersion() < '4.0.0':
                self.__listView.setText(text)
            else:
                self.__listView.clear()
                self.__listView.insertPlainText(text)
        
    def setConfigFile(self,configfile=None):
        if configfile is None:return
        if self.__goodConfigFile(configfile):
            self.configFile = configfile
            if type(configfile) == type([]):
                self.configFile.sort()
                self.__configLine.setText(self.configFile[0])
            else:
                self.__configLine.setText(configfile)
        else:
            qt.QMessageBox.critical(self, "ERROR",
                        "Cannot find fit configuration file:\n%s"% (configfile))           
        
    def setOutputDir(self,outputdir=None):
        if outputdir is None:return
        if self.__goodOutputDir(outputdir):
            self.outputDir = outputdir
            self.__outLine.setText(outputdir)
        else:
            qt.QMessageBox.critical(self, "ERROR",
            "Cannot use output directory:\n%s"% (outputdir))

    def __goodFileList(self,filelist):
        if not len(filelist):return True
        for file in filelist:
            if not os.path.exists(file):
                qt.QMessageBox.critical(self, "ERROR",
                                    'File %s\ndoes not exists' % file)
                self.raiseW()
                return False
        return True
        
    def __goodConfigFile(self,configfile0):
        if type(configfile0) != type([]):
            configfileList = [configfile0]
        else:
            configfileList = configfile0
        for configfile in configfileList:
            if not os.path.exists(configfile):
                qt.QMessageBox.critical(self,
                             "ERROR",'File %s\ndoes not exists' % configfile)
                self.raiseW()
                return False
        return True

    def __goodOutputDir(self,outputdir):
        if os.path.isdir(outputdir):return True
        else:return False

    def __getFileType(self,inputfile):
        try:
            file = None
            try:
                file   = EdfFileLayer.EdfFileLayer(fastedf=1)
                file.SetSource(inputfile)
                fileinfo = file.GetSourceInfo()
                if fileinfo['KeyList'] == []:file=None
                return "EdfFile"
            except:
                pass
            if (file is None):
                file   = SpecFileLayer.SpecFileLayer()
                file.SetSource(inputfile)
            del file
            return "Specfile" 
        except:
            qt.QMessageBox.critical(self, sys.exc_info()[0],'I do not know what to do with file\n %s' % file)
            self.raiseW()
            return None

    def browseList(self):
        if qt.qVersion() < '4.0.0':
            filedialog = qt.QFileDialog(self,"Open a set of files",1)
            filedialog.setMode(filedialog.ExistingFiles)
        else:
            filedialog = qt.QFileDialog(self)
            filedialog.setWindowTitle("Open a set of files")
            filedialog.setModal(1)
            filedialog.setFileMode(filedialog.ExistingFiles)
        if qt.qVersion() < '4.0.0' and (sys.platform == "win32"):
                filelist0= filedialog.getOpenFileNames(qt.QString("McaFiles (*.mca)\nEdfFiles (*.edf)\nSpecFiles (*.spec)\nAll files (*)"),
                            qt.QString.null,
                            self,"openFile", "Open a set of files")
        else:
            if qt.qVersion() < '4.0.0':
                filedialog.setFilters("McaFiles (*.mca)\nEdfFiles (*.edf)\nSpecFiles (*.spec)\nAll files (*)")
                ret = filedialog.exec_loop()
            else:
                filedialog.setFilters(["McaFiles (*.mca)","EdfFiles (*.edf)",
                                   "SpecFiles (*.spec)","All files (*)"])
                ret = filedialog.exec_()
            if  ret == qt.QDialog.Accepted:
                filelist0=filedialog.selectedFiles()
            else:
                if qt.qVersion() < '4.0.0':
                    self.raiseW()
                else:
                    self.raise_()
                return
        #filelist0.sort()
        filelist = []
        for f in filelist0:
            filelist.append(str(f)) 
        if len(filelist):self.setFileList(filelist)
        if qt.qVersion() < '4.0.0':
            self.raiseW()
        else:
            self.raise_()

    def browseConfig(self):
        if qt.qVersion() < '4.0.0':
            filename = qt.QFileDialog(self,"Open a new fit config file",1)
            filename.setMode(filename.ExistingFiles)
        else:
            filename = qt.QFileDialog(self)
            filename.setWindowTitle("Open a new fit config file")
            filename.setModal(1)
            filename.setFileMode(filename.ExistingFiles)
        if (qt.qVersion() < '4.0.0') and (sys.platform == "win32"):
            filenameList= filename.getOpenFileNames(qt.QString("Config Files (*.cfg)\nAll files (*)"),
                            qt.QString.null,
                            self,"openFile", "Open a new fit config file")
        else:
            if qt.qVersion() < '4.0.0':
                filename.setFilters("Config Files (*.cfg)\nAll files (*)")
                ret = filename.exec_loop() 
            else:
                filename.setFilters(["Config Files (*.cfg)", "All files (*)"])
                ret = filename.exec_()
                
            if  ret == qt.QDialog.Accepted:
                filenameList = filename.selectedFiles()
            else:
                if qt.qVersion() < '4.0.0':
                    self.raiseW()
                else:
                    self.raise_()
                return
        filename = []
        for f in filenameList:
            filename.append(str(f))

        if len(filename) == 1:self.setConfigFile(str(filename[0]))
        elif len(filenameList):self.setConfigFile(filename)
        if qt.qVersion() < '4.0.0':
            self.raiseW()
        else:
            self.raise_()

    def browseOutputDir(self):
        if qt.qVersion() < '4.0.0':
            outfile = qt.QFileDialog(self,"Output Directory Selection",1)
            outfile.setMode(outfile.DirectoryOnly)
            ret = outfile.exec_loop()
        else:
            outfile = qt.QFileDialog(self)
            outfile.setWindowTitle("Output Directory Selection")
            outfile.setModal(1)
            outfile.setFileMode(outfile.DirectoryOnly)
            ret = outfile.exec_()
        if ret:
            if qt.qVersion() < '4.0.0':
                outdir=str(outfile.selectedFile())
            else:
                outdir=str(outfile.selectedFiles()[0])
            outfile.close()
            del outfile
            self.setOutputDir(outdir)
        else:
            outfile.close()
            del outfile
        if qt.qVersion() < '4.0.0':
            self.raiseW()
        else:
            self.raise_()
            
    def start(self):
        if not len(self.fileList):
            qt.QMessageBox.critical(self, "ERROR",'Empty file list')
            if qt.qVersion() < '4.0.0':
                self.raiseW()
            else:
                self.raise_()
            return
        if (self.configFile is None) or (not self.__goodConfigFile(self.configFile)):
            qt.QMessageBox.critical(self, "ERROR",'Invalid fit configuration file')
            if qt.qVersion() < '4.0.0':
                self.raiseW()
            else:
                self.raise_()
            return
        if type(self.configFile) == type([]):
            if len(self.configFile) != len(self.fileList):
                qt.QMessageBox.critical(self, "ERROR",
      'Number of config files should be either one or equal to number of files')
                if qt.qVersion() < '4.0.0':
                    self.raiseW()
                else:
                    self.raise_()
                return    
        if (self.outputDir is None) or (not self.__goodOutputDir(self.outputDir)):
            qt.QMessageBox.critical(self, "ERROR",'Invalid output directory')
            if qt.qVersion() < '4.0.0':
                self.raiseW()
            else:
                self.raise_()
            return
        name = "Batch from %s to %s " % (os.path.basename(self.fileList[ 0]),
                                          os.path.basename(self.fileList[-1]))
        roifit  = self.__roiBox.isChecked()
        html    = self.__htmlBox.isChecked()
        if html:
            concentrations = self.__concentrationsBox.isChecked()
        else:
            concentrations = 0
        if self.__tableBox.isChecked():
            if self.__extendedTable.isChecked():
                table = 2
            else:
                table = 1
        else:   table =0
        
        #htmlindex = str(self.__htmlIndex.text())
        htmlindex = "index.html"
        if html:
            if  len(htmlindex)<5:
                htmlindex+=".html"
            if  len(htmlindex) == 5:
                htmlindex = "index.html" 
            if htmlindex[-5:] != "html":
                htmlindex+=".html"
        roiwidth = float(str(self.__roiSpin.text()))
        overwrite= self.__overwrite.isChecked()
        filestep = int(str(self.__fileSpin.text()))
        mcastep  = int(str(self.__mcaSpin.text()))
        fitfiles = self.__fitBox.isChecked()

        if roifit:
            window =  McaBatchWindow(name="ROI"+name,actions=1, outputdir=self.outputDir,
                                     html=html, htmlindex=htmlindex, table = 0)
            b = McaBatch(window,self.configFile,self.fileList,self.outputDir,roifit=roifit,
                         roiwidth=roiwidth,overwrite=overwrite,filestep=1,mcastep=1,
                         concentrations=0, fitfiles=fitfiles)
            def cleanup():
                b.pleasePause = 0
                b.pleaseBreak = 1
                b.wait()
                qt.qApp.processEvents()

            def pause():
                if b.pleasePause:
                    b.pleasePause=0
                    window.pauseButton.setText("Pause")
                else:
                    b.pleasePause=1
                    window.pauseButton.setText("Continue") 
            qt.QObject.connect(window.pauseButton,qt.SIGNAL("clicked()"),pause)
            qt.QObject.connect(window.abortButton,qt.SIGNAL("clicked()"),window.close)
            qt.QObject.connect(qt.qApp,qt.SIGNAL("aboutToQuit()"),cleanup)
            self.__window = window
            self.__b      = b
            window.show()
            b.start()
        elif sys.platform == 'darwin':
            #almost identical to batch    
            window =  McaBatchWindow(name="ROI"+name,actions=1,outputdir=self.outputDir,
                                     html=html,htmlindex=htmlindex, table = table)
            b = McaBatch(window,self.configFile,self.fileList,self.outputDir,roifit=roifit,
                         roiwidth=roiwidth,overwrite=overwrite,filestep=filestep,
                         mcastep=mcastep, concentrations=concentrations, fitfiles=fitfiles)
            def cleanup():
                b.pleasePause = 0
                b.pleaseBreak = 1
                b.wait()
                qt.qApp.processEvents()

            def pause():
                if b.pleasePause:
                    b.pleasePause=0
                    window.pauseButton.setText("Pause")
                else:
                    b.pleasePause=1
                    window.pauseButton.setText("Continue") 
            qt.QObject.connect(window.pauseButton,qt.SIGNAL("clicked()"),pause)
            qt.QObject.connect(window.abortButton,qt.SIGNAL("clicked()"),window.close)
            qt.QObject.connect(qt.qApp,qt.SIGNAL("aboutToQuit()"),cleanup)
            window._rootname = "%s"% b._rootname
            self.__window = window
            self.__b      = b
            window.show()
            b.start()
        elif sys.platform == 'win32':
            listfile = "tmpfile"
            self.genListFile(listfile, config=False)
            try:
                dirname = os.path.dirname(__file__)
            except:
                dirname = os.path.dirname(McaAdvancedFitBatch.__file__)
            if dirname[-3:] == "exe":
                dirname  = os.path.dirname(dirname)
                myself   = os.path.join(dirname, "PyMcaBatch.exe") 
            else:
                myself  = os.path.join(dirname, "PyMcaBatch.py")
            if type(self.configFile) == type([]):
                cfglistfile = "tmpfile.cfg"
                self.genListFile(cfglistfile, config=True)
                cmd = '"%s" --cfglistfile=%s --outdir=%s --overwrite=%d --filestep=%d --mcastep=%d --html=%d --htmlindex=%s --listfile=%s --concentrations=%d --table=%d --fitfiles=%d' % (myself,
                                                                    cfglistfile,
                                                                    self.outputDir, overwrite,
                                                                    filestep, mcastep,
                                                                    html,htmlindex,
                                                                    listfile,concentrations,
                                                                    table, fitfiles)
            else:
                cmd = '"%s" --cfg=%s --outdir=%s --overwrite=%d --filestep=%d --mcastep=%d --html=%d --htmlindex=%s --listfile=%s --concentrations=%d --table=%d --fitfiles=%d' % (myself,
                                                                    self.configFile,
                                                                    self.outputDir, overwrite,
                                                                    filestep, mcastep,
                                                                    html,htmlindex,
                                                                    listfile,concentrations,
                                                                    table, fitfiles)
            self.hide()
            qt.qApp.processEvents()
            if DEBUG:print "cmd = ", cmd
            os.system(cmd)
            self.show()
        else:
            listfile = "tmpfile"
            self.genListFile(listfile, config=False)
            try:
                dirname = os.path.dirname(__file__)
            except:
                dirname = os.path.dirname(McaAdvancedFitBatch.__file__)
            if dirname[-3:] == "exe":
                dirname  = os.path.dirname(dirname)
                myself   = os.path.join(dirname, "PyMcaBatch") 
            else:
                myself  = sys.executable+" "+ os.path.join(dirname, "PyMcaBatch.py")
            if type(self.configFile) == type([]):
                cfglistfile = "tmpfile.cfg"
                self.genListFile(cfglistfile, config=True)
                cmd = "%s --cfglistfile=%s --outdir=%s --overwrite=%d --filestep=%d --mcastep=%d --html=%d --htmlindex=%s --listfile=%s  --concentrations=%d --table=%d --fitfiles=%d &" % (myself,
                                                    cfglistfile,
                                                    self.outputDir, overwrite,
                                                    filestep, mcastep, html, htmlindex, listfile,
                                                    concentrations, table, fitfiles)
            else:
                cmd = "%s --cfg=%s --outdir=%s --overwrite=%d --filestep=%d --mcastep=%d --html=%d --htmlindex=%s --listfile=%s  --concentrations=%d --table=%d --fitfiles=%d &" % (myself, self.configFile,
                                                    self.outputDir, overwrite,
                                                    filestep, mcastep, html, htmlindex,
                                                    listfile, concentrations, table, fitfiles)
            if DEBUG:print "cmd = ", cmd
            os.system(cmd)
            
    def genListFile(self,listfile, config=None):
        try:
            os.remove(listfile)
        except:
            pass
        fd=open(listfile,'w')
        if config is None:lst = self.fileList
        elif config:      lst = self.configFile
        else:             lst = self.fileList
        for filename in lst:
            fd.write('%s\n'%filename)
        fd.close()
    
class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))
        
class McaBatch(qt.QThread,McaAdvancedFitBatch.McaAdvancedFitBatch):
    def __init__(self, parent, configfile, filelist=None, outputdir = None,
                     roifit = None, roiwidth=None, overwrite=1,
                     filestep=1, mcastep=1, concentrations=0, fitfiles=0):
        McaAdvancedFitBatch.McaAdvancedFitBatch.__init__(self, configfile, filelist, outputdir,
                                                         roifit=roifit, roiwidth=roiwidth,
                                                         overwrite=overwrite, filestep=filestep,
                                                         mcastep=mcastep,
                                                         concentrations=concentrations,
                                                         fitfiles=fitfiles) 
        qt.QThread.__init__(self)        
        self.parent = parent
        self.pleasePause = 0
        
    def run(self):
        self.processList()

    def onNewFile(self, file, filelist):
        self.__lastOnNewFile = file
        qt.QApplication.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'file':file,
                                                                   'filelist':filelist,
                                                                   'filestep':self.fileStep,
                                                                   'event':'onNewFile'}))
        if self.pleasePause:self.__pauseMethod()

    def onImage(self, key, keylist):
        qt.QApplication.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'key':key,
                                                                   'keylist':keylist,
                                                                   'event':'onImage'}))

    def onMca(self, mca, nmca, filename=None, key=None, info=None):
        if DEBUG:print "onMca", "key = ",key
        qt.QApplication.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'mca':mca,
                                                                   'nmca':nmca,
                                                                   'mcastep':self.mcaStep,
                                                                   'filename':filename,
                                                                   'key':key,
                                                                   'info':info,
                                                                   'outputdir':self._outputdir,
                                                   'useExistingFiles':self.useExistingFiles,
                                                                   'roifit':self.roiFit,
                                                                   'event':'onMca'}))
        if self.pleasePause:self.__pauseMethod()
                                                                   
    def onEnd(self):
        if DEBUG: print "onEnd"
        qt.QApplication.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'onEnd',
                                                                   'filestep':self.fileStep,
                                                                   'mcastep':self.mcaStep,
                                                                   'savedimages':self.savedImages}))
        if self.pleasePause:self.__pauseMethod()
        
    def __pauseMethod(self):
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'batchPaused'}))
        while(self.pleasePause):
            time.sleep(1)
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'batchResumed'}))
            

class McaBatchWindow(qt.QWidget):
    def __init__(self,parent=None, name="BatchWindow", fl=0, actions = 0, outputdir=None, html=0,
                                            htmlindex = None, table=2):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name, fl)
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            self.setWindowTitle(name)
        self.l = qt.QVBoxLayout(self)
        #self.l.setAutoAdd(1)
        self.bars =qt.QWidget(self)
        self.l.addWidget(self.bars)
        if qt.qVersion() < '4.0.0':
            self.barsLayout = qt.QGridLayout(self.bars,2,3)
        else:
            self.barsLayout = qt.QGridLayout(self.bars)
            self.barsLayout.setMargin(2)
            self.barsLayout.setSpacing(3)
        self.progressBar   = qt.QProgressBar(self.bars)
        self.progressLabel = qt.QLabel(self.bars)
        self.progressLabel.setText('File Progress:')
        self.imageBar   = qt.QProgressBar(self.bars)
        self.imageLabel = qt.QLabel(self.bars)
        self.imageLabel.setText('Image in File:')
        self.mcaBar   = qt.QProgressBar(self.bars)
        self.mcaLabel = qt.QLabel(self.bars)
        self.mcaLabel.setText('MCA in Image:')
        
        self.barsLayout.addWidget(self.progressLabel,0,0)        
        self.barsLayout.addWidget(self.progressBar,0,1)
        self.barsLayout.addWidget(self.imageLabel,1,0)        
        self.barsLayout.addWidget(self.imageBar,1,1)
        self.barsLayout.addWidget(self.mcaLabel,2,0)        
        self.barsLayout.addWidget(self.mcaBar,2,1)
        self.status      = qt.QLabel(self)
        self.status.setText(" ")
        self.timeLeft      = qt.QLabel(self)
        self.l.addWidget(self.status)
        self.l.addWidget(self.timeLeft)
        
        self.timeLeft.setText("Estimated time left = ???? min")
        self.time0 = None
        self.html  = html
        if htmlindex is None:htmlindex="index.html"
        self.htmlindex = htmlindex
        self.outputdir = outputdir
        self.table = table
        self.__ended         = False
        self.__writingReport = False

        if actions: self.addButtons()
        self.show()
        if qt.qVersion() < '4.0.0':
            self.raiseW()
        else:
            self.raise_()


    def addButtons(self):
        self.actions = 1
        self.buttonsBox = qt.QWidget(self)
        l = qt.QHBoxLayout(self.buttonsBox)
        l.addWidget(HorizontalSpacer(self.buttonsBox))        
        self.pauseButton = qt.QPushButton(self.buttonsBox)
        l.addWidget(self.pauseButton)
        l.addWidget(HorizontalSpacer(self.buttonsBox))
        self.pauseButton.setText("Pause")
        self.abortButton   = qt.QPushButton(self.buttonsBox)
        l.addWidget(self.abortButton)
        l.addWidget(HorizontalSpacer(self.buttonsBox))
        self.abortButton.setText("Abort")
        self.l.addWidget(self.buttonsBox)
        self.update()

    def customEvent(self,event):
        if   event.dict['event'] == 'onNewFile':self.onNewFile(event.dict['file'],
                                                               event.dict['filelist'],
                                                               event.dict['filestep'])
        elif event.dict['event'] == 'onImage':  self.onImage  (event.dict['key'],
                                                               event.dict['keylist'])
        elif event.dict['event'] == 'onMca':    self.onMca    (event.dict)
                                                               #event.dict['mca'],
                                                               #event.dict['nmca'],
                                                               #event.dict['mcastep'],
                                                               #event.dict['filename'],
                                                               #event.dict['key'])
        elif event.dict['event'] == 'onEnd':    self.onEnd(event.dict)

        elif event.dict['event'] == 'batchPaused': self.onPause()
        
        elif event.dict['event'] == 'batchResumed':self.onResume()

        elif event.dict['event'] == 'reportWritten':self.onReportWritten()

        else:
            print "Unhandled event",event 
                                                

    def onNewFile(self, file, filelist,filestep):
        if DEBUG:print "onNewFile",file
        indexlist = range(0,len(filelist),filestep)
        index  = indexlist.index(filelist.index(file))
        if index == 0:
            self.report= None
            if self.html:
                self.htmlindex = os.path.join(self.outputdir, 'HTML')
                htmlindex = os.path.join(os.path.basename(file)+"_HTMLDIR", 
                            "index.html")
                self.htmlindex = os.path.join(self.htmlindex,htmlindex)
                if os.path.exists(self.htmlindex):
                    try:
                        os.remove(self.htmlindex)
                    except:
                        print "cannot delete file %s" % self.htmlindex
        nfiles = len(indexlist)
        self.status.setText("Processing file %s" % file)
        e = time.time()
        if qt.qVersion() < '4.0.0':
            self.progressBar.setTotalSteps(nfiles)
            self.progressBar.setProgress(index)
        else:
            self.progressBar.setMaximum(nfiles)
            self.progressBar.setValue(index)
        if self.time0 is not None:
            t = (e - self.time0) * (nfiles - index)
            self.time0 =e
            if t < 120:
                self.timeLeft.setText("Estimated time left = %d sec" % (t))
            else:
                self.timeLeft.setText("Estimated time left = %d min" % (int(t / 60.)))
        else:
            self.time0 = e
        if sys.platform == 'darwin':
            qt.qApp.processEvents()

    def onImage(self,key,keylist):
        if DEBUG:print "onImage ",key
        i = keylist.index(key) + 1
        n = len(keylist)
        if qt.qVersion() < '4.0.0':
            self.imageBar.setTotalSteps(n)
            self.imageBar.setProgress(i)
            self.mcaBar.setTotalSteps(1)
            self.mcaBar.setProgress(0)
        else:
            self.imageBar.setMaximum(n)
            self.imageBar.setValue(i)
            self.mcaBar.setMaximum(1)
            self.mcaBar.setValue(0)
            

    #def onMca(self, mca, nmca, mcastep):
    def onMca(self, dict):
        if DEBUG:print "onMca ",dict['mca']
        mca  = dict['mca']
        nmca = dict['nmca']
        mcastep  = dict['mcastep']
        filename = dict['filename']
        key = dict['key']
        info = dict['info']
        outputdir = dict['outputdir']
        useExistingFiles = dict['useExistingFiles']
        self.roiFit = dict['roifit']
        if self.html:
            try:
                if not self.roiFit:
                    if mca == 0:
                        self.__htmlReport(filename, key, outputdir, useExistingFiles, info, firstmca = True)
                    else:
                        self.__htmlReport(filename, key, outputdir, useExistingFiles, info, firstmca = False)
            except Exception, err:
                print "ERROR on REPORT",sys.exc_info(),err
        if qt.qVersion() < '4.0.0':
            self.mcaBar.setTotalSteps(nmca)
            self.mcaBar.setProgress(mca)
        else:
            self.mcaBar.setMaximum(nmca)
            self.mcaBar.setValue(mca)
        if sys.platform == 'darwin':
            qt.qApp.processEvents()

    def __htmlReport(self, filename, key, outputdir, useExistingFiles, info=None, firstmca = True): 
        """
        file=self.file
        fileinfo = file.GetSourceInfo()
        nimages = nscans = len(fileinfo['KeyList'])

        filename = os.path.basename(info['SourceName'])
        """
        fitdir = os.path.join(outputdir,"HTML")
        if not os.path.exists(fitdir):
            try:
                os.mkdir(fitdir)
            except:
                print "I could not create directory %s" % fitdir
                return
        fitdir = os.path.join(fitdir,filename+"_HTMLDIR")
        if not os.path.exists(fitdir):
            try:
                os.mkdir(fitdir)
            except:
                print "I could not create directory %s" % fitdir
                return
        localindex = os.path.join(fitdir, "index.html")
        if not os.path.isdir(fitdir):
            print "%s does not seem to be a valid directory" % fitdir
        else:
            outfile = filename +"_"+key+".html" 
            outfile = os.path.join(fitdir,  outfile)
        useExistingResult = useExistingFiles
        if os.path.exists(outfile):
            if not useExistingFiles:
                try:
                    os.remove(outfile)
                except:
                    print "cannot delete file %s" % outfile
                useExistingResult = 0
        else:
            useExistingResult = 0    
        outdir = fitdir
        fitdir = os.path.join(outputdir,"FIT")
        fitdir = os.path.join(fitdir,filename+"_FITDIR")
        fitfile= os.path.join(fitdir,  filename +"_"+key+".fit")
        if not os.path.exists(fitfile):
            print "fit file %s does not exists!" % fitfile
            return
        if self.report is None:
            #first file
            self.forcereport = 0
            self._concentrationsFile = os.path.join(outputdir,
                                self._rootname + "_concentrations.txt")
            if os.path.exists(self._concentrationsFile):
                try:
                    os.remove(self._concentrationsFile)
                except:
                    pass
            else:
                #this is to generate the concentrations file
                #from an already existing set of fitfiles
                self.forcereport = 1                        
        if self.forcereport or (not useExistingResult):
            self.report = QtMcaAdvancedFitReport.QtMcaAdvancedFitReport(fitfile = fitfile,
                        outfile = outfile, table = self.table)
            self.__writingReport = True
            a=self.report.writeReport()
            if len(self.report._concentrationsTextASCII) > 1:
                text  = ""
                text += "SOURCE: "+ filename +"\n"
                text += "KEY: "+key+"\n"
                text += self.report._concentrationsTextASCII + "\n"
                f=open(self._concentrationsFile,"a")
                f.write(text)
                f.close()
            self.__writingReport = False
            qt.QApplication.postEvent(self, McaCustomEvent.McaCustomEvent({'event':'reportWritten'}))
            
    def onEnd(self,dict):
        self.__ended = True
        if qt.qVersion() < '4.0.0':
            n = self.progressBar.progress()
            self.progressBar.setProgress(n+dict['filestep'])
            n = self.mcaBar.progress()
            self.mcaBar.setProgress(n+dict['mcastep'])
        else:
            n = self.progressBar.value()
            self.progressBar.setValue(n+dict['filestep'])
            n = self.mcaBar.value()
            self.mcaBar.setValue(n+dict['mcastep'])
        self.status.setText  ("Batch Finished")
        self.timeLeft.setText("Estimated time left = 0 sec")
        if self.actions:
            self.pauseButton.hide()
            self.abortButton.setText("OK")
        if dict.has_key('savedimages'):self.plotImages(dict['savedimages'])
        if self.html:
            if not self.__writingReport:
                directory = os.path.join(self.outputdir,"HTML")
                a = HtmlIndex.HtmlIndex(directory)
                a.buildRecursiveIndex()
        #self.__ended = True
    
    def onReportWritten(self):
        if self.__ended:
            directory = os.path.join(self.outputdir,"HTML")
            a = HtmlIndex.HtmlIndex(directory)
            a.buildRecursiveIndex()
        


    def onPause(self):    
        pass

    def onResume(self):    
        pass
        
        
    def plotImages(self,imagelist):
        if (sys.platform == 'win32') or (sys.platform == 'darwin'):
            filelist = " "
            for file in imagelist:
                filelist+=" %s" % file
            try:
                dirname = os.path.dirname(__file__)
            except:
                dirname = os.path.dirname(McaAdvancedFitBatch.__file__)
            if dirname[-3:] == "exe":
                myself  = os.path.dirname(dirname) 
                myself  = os.path.join(myself, "EdfFileSimpleViewer.exe")
            else:
                myself  = os.path.join(dirname, "EdfFileSimpleViewer.py")
            cmd = '"%s" %s ' % (myself, filelist)
            if 0:
                self.hide()
                qt.qApp.processEvents()
                os.system(cmd)
                self.show()                
            else:
                self.__viewer = EdfFileSimpleViewer.EdfFileSimpleViewer()
                d = EdfFileLayer.EdfFileLayer()
                self.__viewer.setData(d)
                self.__viewer.show()
                self.__viewer.setFileList(imagelist)
        else:
            filelist = " "
            for file in imagelist:
                filelist+=" %s" % file
            try:
                dirname = os.path.dirname(__file__)
            except:
                dirname = os.path.dirname(McaAdvancedFitBatch.__file__)
            if DEBUG:print "final dirname = ",dirname
            if dirname[-3:] == "exe":
                myself  = os.path.dirname(dirname) 
                myself  = os.path.join(myself, "EdfFileSimpleViewer")
            else:
                myself  = sys.executable+" "+os.path.join(dirname, "EdfFileSimpleViewer.py")
            cmd = "%s %s &" % (myself, filelist)
            if DEBUG:print "cmd = ",cmd
            os.system(cmd)
                              
def main():
    import getopt
    options     = 'f'
    longoptions = ['cfg=','outdir=','roifit=','roi=','roiwidth=',
                   'overwrite=', 'filestep=', 'mcastep=', 'html=','htmlindex=',
                   'listfile=','cfglistfile=', 'concentrations=', 'table=', 'fitfiles=']
    filelist = None
    outdir   = None
    cfg      = None
    listfile = None
    cfglistfile = None    
    roifit   = 0
    roiwidth = ROIWIDTH
    overwrite= 1
    filestep = 1
    html = 0
    htmlindex= None
    mcastep  = 1
    table    = 2
    fitfiles = 1
    concentrations = 0
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)
    for opt,arg in opts:
        if opt in ('--cfg'):
            cfg = arg
        elif opt in ('--outdir'):
            outdir = arg
        elif opt in ('--roi','--roifit'):
            roifit   = int(arg)
        elif opt in ('--roiwidth'):
            roiwidth = float(arg)
        elif opt in ('--overwrite'):
            overwrite= int(arg)
        elif opt in ('--filestep'):
            filestep = int(arg)
        elif opt in ('--mcastep'):
            mcastep  = int(arg)
        elif opt in ('--html'):
            html  = int(arg)
        elif opt in ('--htmlindex'):
            htmlindex  = arg
        elif opt in ('--listfile'):
            listfile  = arg
        elif opt in ('--cfglistfile'):
            cfglistfile  = arg
        elif opt in ('--concentrations'):
            concentrations  = int(arg)
        elif opt in ('--table'):
            table  = int(arg)
        elif opt in ('--fitfiles'):
            fitfiles  = int(arg)
    if listfile is None: 
        filelist=[]
        for item in args:
            filelist.append(item)
    else:
        fd = open(listfile)
        filelist = fd.readlines()
        fd.close()
        for i in range(len(filelist)):
            filelist[i]=filelist[i].replace('\n','')
    if cfglistfile is not None:
        fd = open(cfglistfile)
        cfg = fd.readlines()
        fd.close()
        for i in range(len(cfg)):
            cfg[i]=cfg[i].replace('\n','')
    app=qt.QApplication(sys.argv) 
    winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
    app.setPalette(winpalette)       
    if len(filelist) == 0:
        qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
        w = McaBatchGUI(actions=1)
        if qt.qVersion() < '4.0.0':
            app.setMainWidget(w)
            w.show()
            app.exec_loop()
        else:
            w.show()
            app.exec_()
    else:
        qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
        text = "Batch from %s to %s" % (os.path.basename(filelist[0]), os.path.basename(filelist[-1]))
        window =  McaBatchWindow(name=text,actions=1,
                                outputdir=outdir,html=html, htmlindex=htmlindex, table=table)
                                
        if html or concentrations:fitfiles=1
        try:
            b = McaBatch(window,cfg,filelist,outdir,roifit=roifit,roiwidth=roiwidth,
                     overwrite = overwrite, filestep=filestep, mcastep=mcastep,
                      concentrations=concentrations, fitfiles=fitfiles)
        except:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("%s" % sys.exc_info()[1])
            if qt.qVersion() < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return

            
        def cleanup():
            b.pleasePause = 0
            b.pleaseBreak = 1
            b.wait()
            qt.qApp.processEvents()

        def pause():
            if b.pleasePause:
                b.pleasePause=0
                window.pauseButton.setText("Pause")
            else:
                b.pleasePause=1
                window.pauseButton.setText("Continue") 
        qt.QObject.connect(window.pauseButton,qt.SIGNAL("clicked()"),pause)
        qt.QObject.connect(window.abortButton,qt.SIGNAL("clicked()"),window.close)
        qt.QObject.connect(app,qt.SIGNAL("aboutToQuit()"),cleanup)        
        window._rootname = "%s"% b._rootname
        window.show()
        b.start()
        if qt.qVersion() < '4.0.0':
            app.setMainWidget(window)
            app.exec_loop()
        else:
            app.exec_()
 
if __name__ == "__main__":
    main()
 
 
# PyMcaBatch.py --cfg=/mntdirect/_bliss/users/sole/COTTE/WithLead.cfg --outdir=/tmp/   /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0007.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0008.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0009.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0010.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0011.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0012.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0013.edf &
# PyMcaBatch.exe --cfg=E:/COTTE/WithLead.cfg --outdir=C:/tmp/   E:/COTTE/ch09/ch09__mca_0003_0000_0007.edf E:/COTTE/ch09/ch09__mca_0003_0000_0008.edf
