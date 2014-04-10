#!/usr/bin/env python
__revision__ = "$Revision: 1.5 $"
#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
import os
import sys
import time
import McaCustomEvent
import ConfigDict
ROIWIDTH = 250.

from PyMca5 import PyMcaQt as qt

class Fit2SpecGUI(qt.QWidget):
    def __init__(self,parent=None,name="Fit to Spec Conversion",
                filelist=None,outputdir=None, actions=0):
        qt.QWidget.__init__(self,parent,name)
        layout = qt.QVBoxLayout(self)
        layout.setAutoAdd(1)
        self.setCaption(name)
        self.__build(actions)               
        if filelist is None: filelist = []
        self.outputDir  = None
        self.setFileList(filelist)
        self.setOutputDir(outputdir)
    
    def __build(self,actions):
        self.__grid= qt.QWidget(self)
        #self.__grid.setGeometry(qt.QRect(30,30,288,156))
        grid       = qt.QGridLayout(self.__grid,2,3,11,6)
        grid.setColStretch(0,0)
        grid.setColStretch(1,1)
        grid.setColStretch(2,0)
        #input list
        listrow  = 0
        listlabel   = qt.QLabel(self.__grid)
        listlabel.setText("Input File list:")
        listlabel.setAlignment(qt.QLabel.WordBreak | qt.QLabel.AlignVCenter)
        self.__listView   = qt.QTextView(self.__grid)
        self.__listView.setMaximumHeight(30*listlabel.sizeHint().height())
        self.__listButton = qt.QPushButton(self.__grid)
        self.__listButton.setText('Browse')
        self.connect(self.__listButton,qt.SIGNAL('clicked()'),self.browseList) 
        grid.addWidget(listlabel,        listrow, 0, qt.Qt.AlignTop|qt.Qt.AlignLeft)
        grid.addWidget(self.__listView,  listrow, 1)
        grid.addWidget(self.__listButton,listrow, 2, qt.Qt.AlignTop|qt.Qt.AlignRight)
        
        #output dir
        outrow    = 1
        outlabel   = qt.QLabel(self.__grid)
        outlabel.setText("Output dir:")
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


        if actions: self.__buildActions()

    def __buildActions(self):
        box = qt.QHBox(self)
        qt.HorizontalSpacer(box)
        self.__dismissButton = qt.QPushButton(box)
        qt.HorizontalSpacer(box)
        self.__dismissButton.setText("Close")
        self.__startButton   = qt.QPushButton(box)
        qt.HorizontalSpacer(box)
        self.__startButton.setText("Start")
        self.connect(self.__dismissButton,qt.SIGNAL("clicked()"),self.close)
        self.connect(self.__startButton,qt.SIGNAL("clicked()"),self.start)

    def setFileList(self,filelist=None):
        if filelist is None:
            filelist = []
        if True or self.__goodFileList(filelist):
            text = ""
            filelist.sort()
            for ffile in filelist:
                text += "%s\n" % ffile
            self.fileList = filelist
            self.__listView.setText(text)

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
                qt.QMessageBox.critical(self, "ERROR",'File %s\ndoes not exists' % file)
                self.raiseW()
                return False
        return True
        
    def __goodOutputDir(self,outputdir):
        if os.path.isdir(outputdir):return True
        else:return False

    def browseList(self):
        filedialog = qt.QFileDialog(self,"Open a set of files",1)
        filedialog.setMode(filedialog.ExistingFiles)
        filedialog.setFilters("Fit Files (*.fit)\n")
        if filedialog.exec_loop() == qt.QDialog.Accepted:
            filelist0= filedialog.selectedFiles()
        else:
            self.raiseW()
            return    
        filelist = []
        for f in filelist0:
            filelist.append(qt.safe_str(f)) 
        if len(filelist):self.setFileList(filelist)
        self.raiseW()

    def browseConfig(self):
        filename= qt.QFileDialog(self,"Open a new fit config file",1)
        filename.setMode(filename.ExistingFiles)
        filename.setFilters("Config Files (*.cfg)\nAll files (*)")
        if filename.exec_loop() == qt.QDialog.Accepted:
            filename = filename.selectedFile()
        else:
            self.raiseW()
            return                
        filename = qt.safe_str(filename)
        if len(filename):
            self.setConfigFile(filename)
        self.raiseW()

    def browseOutputDir(self):
        outfile = qt.QFileDialog(self,"Output Directory Selection",1)
        outfile.setMode(outfile.DirectoryOnly)
        ret = outfile.exec_loop()
        if ret:
            outdir = qt.safe_str(outfile.selectedFile())
            outfile.close()
            del outfile
            self.setOutputDir(outdir)
        else:
            outfile.close()
            del outfile
        self.raiseW()
            
    def start(self):
        if not len(self.fileList):
            qt.QMessageBox.critical(self, "ERROR",'Empty file list')
            self.raiseW()
            return
        if (self.outputDir is None) or (not self.__goodOutputDir(self.outputDir)):
            qt.QMessageBox.critical(self, "ERROR",'Invalid output directory')
            self.raiseW()
            return
        name = "Batch from %s to %s " % (os.path.basename(self.fileList[ 0]),
                                          os.path.basename(self.fileList[-1]))

        window =  Fit2SpecWindow(name="Fit 2 Spec "+name,actions=1)
        b = Fit2SpecBatch(window,self.fileList,self.outputDir)
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

        
class Fit2SpecBatch(qt.QThread):
    def __init__(self, parent, filelist=None, outputdir = None):
        self._filelist  = filelist
        self.outputdir = outputdir
        qt.QThread.__init__(self)
        self.parent = parent
        self.pleasePause = 0

    def processList(self):
        for fitfile in self._filelist:
            self.onNewFile(fitfile, self._filelist)
            d = ConfigDict.ConfigDict()
            d.read(fitfile)
            f = open(os.path.join(self.outputdir,os.path.basename(fitfile)+".dat"),'w+')
            npoints = len(d['result']['xdata'])
            f.write("\n")
            f.write("#S 1 %s\n" % fitfile)
            i=0
            for parameter in d['result']['parameters']:
                f.write("#U%d %s %.6g +/- %.3g\n" % (i, parameter,
                                                     d['result']['fittedpar'][i],
                                                     d['result']['sigmapar'][i]))
                i+=1
            f.write("#N 6\n")
            f.write("#L Energy  channel  counts  fit  continuum  pileup\n")
            for i in range(npoints):
                f.write("%.6g  %.6g   %.6g  %.6g  %.6g  %.6g\n" % (d['result']['energy'][i],
                                   d['result']['xdata'][i],
                                   d['result']['ydata'][i],
                                   d['result']['yfit'][i],
                                   d['result']['continuum'][i],
                                   d['result']['pileup'][i]))
            f.close()
        self.onEnd()

    def run(self):
        self.processList()

    def onNewFile(self, file, filelist):
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'file':file,
                                                                   'filelist':filelist,
                                                                   'event':'onNewFile'}))
        if self.pleasePause:self.__pauseMethod()

    def onEnd(self):
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'onEnd'}))
        if self.pleasePause:self.__pauseMethod()
        

    def __pauseMethod(self):
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'batchPaused'}))
        while(self.pleasePause):
            time.sleep(1)
        self.postEvent(self.parent, McaCustomEvent.McaCustomEvent({'event':'batchResumed'}))
            

class Fit2SpecWindow(qt.QWidget):
    def __init__(self,parent=None, name="BatchWindow", fl=0, actions = 0):
        qt.QWidget.__init__(self, parent, name, fl)
        self.setCaption(name)
        self.l = qt.QVBoxLayout(self)
        self.l.setAutoAdd(1)
        self.bars =qt.QWidget(self)
        self.barsLayout = qt.QGridLayout(self.bars,2,3)
        self.progressBar   = qt.QProgressBar(self.bars)
        self.progressLabel = qt.QLabel(self.bars)
        self.progressLabel.setText('File Progress:')
        
        self.barsLayout.addWidget(self.progressLabel,0,0)        
        self.barsLayout.addWidget(self.progressBar,0,1)
        self.status      = qt.QLabel(self)
        self.status.setText(" ")
        self.timeLeft      = qt.QLabel(self)
        self.timeLeft.setText("Estimated time left = ???? min")
        self.time0 = None
        if actions: self.addButtons()
        self.show()
        self.raiseW()


    def addButtons(self):
        self.actions = 1
        self.buttonsBox = qt.QWidget(self)
        l = qt.QHBoxLayout(self.buttonsBox)
        l.setAutoAdd(1)
        qt.HorizontalSpacer(self.buttonsBox)
        self.pauseButton = qt.QPushButton(self.buttonsBox)
        qt.HorizontalSpacer(self.buttonsBox)
        self.pauseButton.setText("Pause")
        self.abortButton   = qt.QPushButton(self.buttonsBox)
        qt.HorizontalSpacer(self.buttonsBox)
        self.abortButton.setText("Abort")
        self.update()

    def customEvent(self,event):
        if   event.dict['event'] == 'onNewFile':self.onNewFile(event.dict['file'],
                                                               event.dict['filelist'])
        elif event.dict['event'] == 'onEnd':    self.onEnd(event.dict)

        elif event.dict['event'] == 'batchPaused': self.onPause()
        
        elif event.dict['event'] == 'batchResumed':self.onResume()

        else:
            print("Unhandled event",event)
                                                

    def onNewFile(self, file, filelist):
        indexlist = range(0,len(filelist))
        index  = indexlist.index(filelist.index(file))
        nfiles = len(indexlist)
        self.status.setText("Processing file %s" % file)
        e = time.time()
        self.progressBar.setTotalSteps(nfiles)
        self.progressBar.setProgress(index)
        if self.time0 is not None:
            t = (e - self.time0) * (nfiles - index)
            self.time0 =e
            if t < 120:
                self.timeLeft.setText("Estimated time left = %d sec" % (t))
            else:
                self.timeLeft.setText("Estimated time left = %d min" % (int(t / 60.)))
        else:
            self.time0 = e

    def onEnd(self,dict):
        n = self.progressBar.progress()
        self.progressBar.setProgress(n+1)
        self.status.setText  ("Batch Finished")
        self.timeLeft.setText("Estimated time left = 0 sec")
        if self.actions:
            self.pauseButton.hide()
            self.abortButton.setText("OK")

    def onPause(self):    
        pass

    def onResume(self):    
        pass
                
if __name__ == "__main__":
    import getopt
    options     = 'f'
    longoptions = ['outdir=', 'listfile=']
    filelist = None
    outdir   = None
    listfile = None    
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)
    for opt,arg in opts:
        if opt in ('--outdir'):
            outdir = arg
        elif opt in  ('--listfile'):
            listfile  = arg
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
             
    app=qt.QApplication(sys.argv) 
    winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
    app.setPalette(winpalette)       
    if len(filelist) == 0:
        qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
        w = Fit2SpecGUI(actions=1)
        app.setMainWidget(w)
        w.show()
        app.exec_loop()
    else:
        qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),app, qt.SLOT("quit()"))
        text = "Batch from %s to %s" % (os.path.basename(filelist[0]), os.path.basename(filelist[-1]))
        window =  Fit2SpecWindow(name=text,actions=1)
        b = Fit2SpecBatch(window,filelist,outdir)
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
        window.show()
        b.start()
        app.setMainWidget(window)
        app.exec_loop()
 
 
 
# PyMcaBatch.py --cfg=/mntdirect/_bliss/users/sole/COTTE/WithLead.cfg --outdir=/tmp/   /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0007.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0008.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0009.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0010.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0011.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0012.edf /mntdirect/_bliss/users/sole/COTTE/ch09/ch09__mca_0003_0000_0013.edf &
# PyMcaBatch.exe --cfg=E:/COTTE/WithLead.cfg --outdir=C:/tmp/   E:/COTTE/ch09/ch09__mca_0003_0000_0007.edf E:/COTTE/ch09/ch09__mca_0003_0000_0008.edf
