#!/usr/bin/env python
__revision__ = "$Revision: 1.59 $"
#/*##########################################################################
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
#############################################################################*/
import sys, getopt, string
import qt
import PyMcaMdi
from PyMca_Icons import IconDict
from PyMca_help import HelpDict
import os
__version__ = "3.9.1"
if (sys.platform == 'darwin') or (qt.qVersion() < '3.0.0'):
    class SplashScreen(qt.QWidget):
        def __init__(self,parent=None,name="SplashScreen",
                        fl=qt.Qt.WStyle_Customize  | qt.Qt.WDestructiveClose,
                        pixmap = None):
            qt.QWidget.__init__(self,parent,name,fl)
            self.setCaption("PyMCA %s" % __version__)
            layout = qt.QVBoxLayout(self)
            layout.setAutoAdd(1)
            label = qt.QLabel(self)
            if pixmap is not None:label.setPixmap(pixmap)
            else:label.setText("Hello") 
            self.bottomText = qt.QLabel(self)

        def message(self, text):
            font = self.bottomText.font()
            font.setBold(True)
            self.bottomText.setFont(font)
            self.bottomText.setText(text)
            self.bottomText.show()
            self.show()
            self.raiseW()

if __name__ == "__main__":
    app = qt.QApplication(sys.argv)
    if qt.qVersion >= '3.0.0':
        strlist = qt.QStyleFactory.keys()
        if sys.platform == "win32":
            for item in strlist:
                text = str(item)
                if text == "WindowsXP":
                    style = qt.QStyleFactory.create(item)
                    app.setStyle(style)
                    break
    if 1:
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)

    mpath = os.path.dirname(PyMcaMdi.__file__)
    if mpath[-3:] == "exe":
        mpath = os.path.dirname(mpath)
    qt.QMimeSourceFactory.defaultFactory().addFilePath(mpath)
    if (sys.platform == 'darwin') or (qt.qVersion() < '3.0.0'):
        pixmap = qt.QPixmap('PyMcaSplashImage.png')    
        splash  = SplashScreen(pixmap=pixmap)
        splash.message( 'PyMCA version %s\n' % __version__)     
    else:
        pixmap = qt.QPixmap.fromMimeSource('PyMcaSplashImage.png')
        splash  = qt.QSplashScreen(pixmap)
        splash.show()
        font = splash.font()
        font.setBold(True)
        splash.setFont(font)
        splash.message( 'PyMCA %s' % __version__, 
                qt.Qt.AlignLeft|qt.Qt.AlignBottom, 
                qt.Qt.white)


import McaWindow
import EdfFileLayer
import SpecFileLayer
import SPSLayer
import MySpecFileSelector as SpecFileSelector
import MyEdfFileSelector  as EdfFileSelector
import ElementsInfo
import PeakIdentifier
import PyMcaBatch
import Fit2Spec
import ConfigDict
DEBUG = 0

SOURCES = {"SpecFile":{'widget':SpecFileSelector.SpecFileSelector,'data':SpecFileLayer.SpecFileLayer},
           "EdfFile":{'widget':EdfFileSelector.EdfFileSelector,'data':EdfFileLayer.EdfFileLayer}}

SOURCESLIST = ["SpecFile","EdfFile"]

if (sys.platform != 'win32') and (sys.platform != 'darwin'):
    import MySPSSelector as SPSSelector
    SOURCES["SPS"] = {'widget':SPSSelector.SPSSelector,'data':SPSLayer.SPSLayer}
    SOURCESLIST.append("SPS")

class PyMca(PyMcaMdi.PyMca):
    def __init__(self, parent=None, name="PyMca", fl=qt.Qt.WDestructiveClose,**kw):
            PyMcaMdi.PyMca.__init__(self, parent, name, fl)
            
            self.setCaption(name)
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))            
            self.menuBar().setIcon(qt.QPixmap(IconDict['gioconda16']))            
            self.initSourceBrowser()
            self.initSource()
            self.openMenu = qt.QPopupMenu()
            self.elementsInfo= None
            self.identifier  = None
            self.__batch     = None
            self.__fit2Spec  = None
            self.openMenu.insertItem("PyMca Configuration",0)
            i=1
            for source in SOURCESLIST:
                self.openMenu.insertItem(source,i)
                i+=1
            self.connect(self.openMenu,qt.SIGNAL('activated(int)'),self.openSource)
            

            if (sys.platform != 'win32') and (sys.platform != 'darwin'):
                self.mcawindow = McaWindow.McaWidget(parent=self.mdi,
                                                     SpecFileWidget= self.sourceWidget['SpecFile'],
                                                     EdfFileWidget = self.sourceWidget['EdfFile'],                
                                                     SPSWidget     = self.sourceWidget['SPS'],
                                                     fl=0,**kw)
            else:
                self.mcawindow = McaWindow.McaWidget(parent=self.mdi,
                                                     SpecFileWidget= self.sourceWidget['SpecFile'],
                                                     EdfFileWidget = self.sourceWidget['EdfFile'],
                                                     fl=0,**kw)
            self.connect(self.mcawindow,qt.PYSIGNAL('McaWindowSignal'),self.__McaWindowSignal)
            if kw.has_key('shm'):
                if len(kw['shm']) >= 8:
                    if kw['shm'][0:8] in ['MCA_DATA', 'XIA_DATA']:
                        self.mcawindow.showMaximized()
                        self.toggleSource()
            else:
                self.mcawindow.showMaximized()
            currentConfigDict = ConfigDict.ConfigDict()
            defaultFileName = self.__getDefaultSettingsFile()
            self.configDir  = os.path.dirname(defaultFileName)
            if not kw.has_key('fresh'):
                if os.path.exists(defaultFileName):
                    currentConfigDict.read(defaultFileName)
                    self.setConfig(currentConfigDict)

                
    def setConfig(self, configDict):
        if configDict.has_key('PyMca'):    self.__configurePyMca(configDict['PyMca'])
        if configDict.has_key('ROI'):      self.__configureRoi(configDict['ROI'])
        if configDict.has_key('Elements'): self.__configureElements(configDict['Elements'])
        if configDict.has_key('Fit'):      self.__configureFit(configDict['Fit'])
        if configDict.has_key('SimpleFit'):self.__configureSimpleFit(configDict['SimpleFit'])
        
    def getConfig(self):
        d = {}
        d['PyMca']    = {}
        d['PyMca']['VERSION']   = __version__
        d['PyMca']['ConfigDir'] = self.configDir
        
        #geometry
        d['PyMca']['Geometry']  ={}
        r = self.geometry()
        d['PyMca']['Geometry']['MainWindow'] = [r.x(), r.y(), r.width(), r.height()]
        r = self.splitter.sizes()
        d['PyMca']['Geometry']['Splitter'] = r
        r = self.mcawindow.geometry()
        d['PyMca']['Geometry']['McaWindow'] = [r.x(), r.y(), r.width(), r.height()]
        r = self.mcawindow.graph.geometry()
        d['PyMca']['Geometry']['McaGraph'] = [r.x(), r.y(), r.width(), r.height()]
        #sources
        d['PyMca']['Sources'] = {}
        for source in SOURCESLIST:
            d['PyMca'][source] = {}
            if 'lastInputDir' in self.sourceWidget[source].__dict__.keys():
                if self.sourceWidget[source].lastInputDir is not None:
                    d['PyMca'][source]['lastInputDir'] = self.sourceWidget[source].lastInputDir
                else:
                    d['PyMca'][source]['lastInputDir'] = "None"
            if source == "SpecFile":
                d['PyMca'][source]['SourceName'] = []
                for key in self.sourceWidget[source].mapComboName.keys():
                    d['PyMca'][source]['SourceName'].append(key)
            elif source == "EdfFile":
                d['PyMca'][source]['SourceName'] = []
                for key in self.sourceWidget[source].mapComboName.keys():
                    if key == "EDF Stack":
                        d['PyMca'][source]['SourceName'].append(self.sourceWidget[source]._edfstack)
                    else:
                        d['PyMca'][source]['SourceName'].append(self.sourceWidget[source].mapComboName[key])
            
            #d['PyMca'][source]['Selection'] = self.sourceWidget[source].getSelection()
        #ROIs
        d['ROI']={}
        roilist, roidict = self.mcawindow.roiwidget.getroilistanddict()
        d['ROI']['roilist'] = roilist
        d['ROI']['roidict'] = {}
        d['ROI']['roidict'].update(roidict)

        #fit related        
        d['Elements'] = {}
        d['Elements']['Material'] = {}
        d['Elements']['Material'].update(ElementsInfo.Elements.Material)
        d['Fit'] = {}
        if self.mcawindow.advancedfit.configDir is not None:
            d['Fit'] ['ConfigDir'] = self.mcawindow.advancedfit.configDir * 1
        d['Fit'] ['Configuration'] = {}
        d['Fit'] ['Configuration'].update(self.mcawindow.advancedfit.mcafit.configure())
        d['Fit'] ['Information'] = {}
        d['Fit'] ['Information'].update(self.mcawindow.advancedfit.info)
        d['Fit'] ['LastFit'] = {}
        d['Fit'] ['LastFit']['hidden'] = self.mcawindow.advancedfit.isHidden()
        d['Fit'] ['LastFit']['xdata0'] = self.mcawindow.advancedfit.mcafit.xdata0
        d['Fit'] ['LastFit']['ydata0'] = self.mcawindow.advancedfit.mcafit.ydata0
        d['Fit'] ['LastFit']['sigmay0']= self.mcawindow.advancedfit.mcafit.sigmay0
        d['Fit'] ['LastFit']['fitdone']= self.mcawindow.advancedfit._fitdone()
        #d['Fit'] ['LastFit']['fitdone']= 1
        #d['Fit'] ['LastFit']['xmin'] = self.mcawindow.advancedfit.mcafit.sigma0
        #d['Fit'] ['LastFit']['xmax'] = self.mcawindow.advancedfit.mcafit.sigma0
        return d
        
    def saveConfig(self, config, filename = None):
        d = ConfigDict.ConfigDict()
        d.update(config)
        if filename is None:filename = self.__getDefaultSettingsFile()
        d.write(filename)

    def __configurePyMca(self, dict):
        if dict.has_key('ConfigDir'):self.configDir = dict['ConfigDir']

        if dict.has_key('Geometry'):
            r = qt.QRect(*dict['Geometry']['MainWindow'])
            self.setGeometry(r)
            key = 'Splitter'
            if key in dict['Geometry'].keys():
                self.splitter.setSizes(dict['Geometry'][key])
            key = 'McaWindow'
            if key in dict['Geometry'].keys():
                r = qt.QRect(*dict['Geometry']['McaWindow'])
                self.mcawindow.setGeometry(r)
            key = 'McaGraph'
            if key in dict['Geometry'].keys():
                r = qt.QRect(*dict['Geometry']['McaGraph'])
                self.mcawindow.graph.setGeometry(r)
            self.show()
            qt.qApp.processEvents()
            qt.qApp.postEvent(self, qt.QResizeEvent(qt.QSize(dict['Geometry']['MainWindow'][2]+1,
                                                          dict['Geometry']['MainWindow'][3]+1),
                                                 qt.QSize(dict['Geometry']['MainWindow'][2],
                                                          dict['Geometry']['MainWindow'][3])))
            self.mcawindow.showMaximized()
        for source in SOURCESLIST:
            if dict.has_key(source):
                if dict[source].has_key('lastInputDir'):
                    if dict[source] ['lastInputDir'] != "None":
                        self.sourceWidget[source].lastInputDir =  dict[source] ['lastInputDir']
                    else:
                        self.sourceWidget[source].lastInputDir =  None
                if dict[source].has_key('SourceName'):
                    if type(dict[source]['SourceName']) != type([]):
                        dict[source]['SourceName'] = [dict[source]['SourceName'] * 1]
                    for SourceName in dict[source]['SourceName']:
                        try:
                            if source == "EdfFile":
                                self.sourceWidget[source].openFile(SourceName, justloaded=1)
                            else:
                                self.sourceWidget[source].openFile(SourceName)
                        except:
                            msg = qt.QMessageBox(self)
                            msg.setIcon(qt.QMessageBox.Critical)
                            msg.setText("Error: %s\n opening file %s" % (sys.exc_info()[1],SourceName ))
                            msg.exec_loop()
                """
                if dict[source].has_key('Selection'):
                    if type(dict[source]['Selection']) != type([]):
                        dict[source]['Selection'] = [dict[source]['Selection']]
                    if source == "EdfFile":
                        self.sourceWidget[source].setSelected(dict[source]['Selection'])
                """    
        if self.sourceWidget["SpecFile"].lastInputDir  is None:
            self.sourceWidget["SpecFile"].lastInputDir = self.sourceWidget["EdfFile"].lastInputDir
        if self.sourceWidget["EdfFile"].lastInputDir is None:
            self.sourceWidget["EdfFile"].lastInputDir  = self.sourceWidget["SpecFile"].lastInputDir


    def __configureRoi(self, dict):
        if dict.has_key('roidict'):
            if dict.has_key('roilist'):
                roilist = dict['roilist']
                if type(roilist) != type([]):
                    roilist=[roilist]                
                roidict = dict['roidict']
                self.mcawindow.roiwidget.fillfromroidict(roilist=roilist,
                                                         roidict=roidict)
            

    def __configureElements(self, dict):
        if dict.has_key('Material'):
            ElementsInfo.Elements.Material.update(dict['Material'])    

    def __configureFit(self, d):
        if d.has_key('Configuration'):
            self.mcawindow.advancedfit.mcafit.configure(d['Configuration'])
            if not self.mcawindow.advancedfit.isHidden():
                self.mcawindow.advancedfit._updateTop()
        if d.has_key('ConfigDir'):
            self.mcawindow.advancedfit.configDir = d['ConfigDir'] * 1
        if False and d.has_key('LastFit'):
            if (d['LastFit']['ydata0'] != None) and \
               (d['LastFit']['ydata0'] != 'None'):               
                self.mcawindow.advancedfit.setdata(x=d['LastFit']['xdata0'],
                                                   y=d['LastFit']['ydata0'],
                                              sigmay=d['LastFit']['sigmay0'],
                                              **d['Information'])
                if d['LastFit']['hidden'] == 'False':
                    self.mcawindow.advancedfit.show()
                    self.mcawindow.advancedfit.raiseW()
                    if d['LastFit']['fitdone']:
                        try:
                            self.mcawindow.advancedfit.fit()
                        except:
                            pass  
                else:
                    print "hidden"     
            
    def __configureSimpleFit(self, dict):
        pass
                
    def initMenuBar(self):
        if self.options["MenuFile"]:
            self.menuFile= qt.QPopupMenu(self.menuBar())
            idx= self.menuFile.insertItem(self.Icons["fileopen"], qt.QString("&Open"), self.onOpen, qt.Qt.CTRL+qt.Qt.Key_O)
            self.menuFile.setWhatsThis(idx, HelpDict["fileopen"])
            idx= self.menuFile.insertItem(self.Icons["filesave"], "&Save as", self.onSaveAs, qt.Qt.CTRL+qt.Qt.Key_S)
            self.menuFile.setWhatsThis(idx, HelpDict["filesave"])
            self.menuFile.insertItem("Save &Defaults", self.onSave)
            self.menuFile.insertSeparator()
            idx= self.menuFile.insertItem(self.Icons["fileprint"], "&Print", self.onPrint, qt.Qt.CTRL+qt.Qt.Key_P)
            self.menuFile.setWhatsThis(idx, HelpDict["fileprint"])
            self.menuFile.insertSeparator()
            self.menuFile.insertItem("&Quit", qt.qApp, qt.SLOT("closeAllWindows()"), qt.Qt.CTRL+qt.Qt.Key_Q)
            self.menuBar().insertItem('&File',self.menuFile)

        self.onInitMenuBar(self.menuBar())

        if self.options["MenuTools"]:
            self.menuTools= qt.QPopupMenu()
            self.menuTools.setCheckable(1)
            self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"), self.menuToolsAboutToShow)
            self.menuBar().insertItem("&Tools", self.menuTools)

        if self.options["MenuWindow"]:
            self.menuWindow= qt.QPopupMenu()
            self.menuWindow.setCheckable(1)
            self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
            self.menuBar().insertItem("&Window", self.menuWindow)

        if self.options["MenuHelp"]:
            self.menuHelp= qt.QPopupMenu(self)
            self.menuHelp.insertItem("&Menu", self.onMenuHelp)
            self.menuHelp.insertItem("&Data Display HOWTOs", self.onDisplayHowto)
            self.menuHelp.insertItem("MCA &HOWTOs",self.onMcaHowto)
            self.menuHelp.insertSeparator()
            self.menuHelp.insertItem("&About", self.onAbout)
            self.menuHelp.insertItem("About &Qt",self.onAboutQt)
            self.menuBar().insertSeparator()
            self.menuBar().insertItem("&Help", self.menuHelp)
            self.menuBrowser    = None
            self.displayBrowser = None
            self.mcaBrowser     = None

    def initSourceBrowser(self):
        self.sourceFrame     =qt.QWidget(self.splitter)
        self.splitter.moveToFirst(self.sourceFrame)
        #self.splitter.setResizeMode(self.sourceFrame,qt.QSplitter.KeepSize)
        layout = qt.QVBoxLayout(self.sourceFrame)
        layout.setAutoAdd(1)
        
        sourceToolbar = qt.QWidget(self.sourceFrame)
        layout1       = qt.QHBoxLayout(sourceToolbar)
        #self.line1 = qt.QFrame(sourceToolbar,"line1")
        self.line1 = Line(sourceToolbar,"line1")
        self.line1.setFrameShape(qt.QFrame.HLine)
        self.line1.setFrameShadow(qt.QFrame.Sunken)
        self.line1.setFrameShape(qt.QFrame.HLine)
        layout1.addWidget(self.line1)
        #self.closelabel = qt.QLabel(sourceToolbar)
        self.closelabel = PixmapLabel(sourceToolbar)
        self.closelabel.setPixmap(qt.QPixmap(IconDict['close']))
        layout1.addWidget(self.closelabel)
        self.closelabel.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.sourceBrowserTab=qt.QTabWidget(self.sourceFrame)
        
        #connections
        self.connect(self.line1,qt.PYSIGNAL("LineDoubleClickEvent"),self.sourceReparent)
        self.connect(self.closelabel,qt.PYSIGNAL("PixmapLabelMousePressEvent"),self.toggleSource)
        
        #tips
        qt.QToolTip.add(self.line1,"DoubleClick toggles floating window mode")
        qt.QToolTip.add(self.closelabel,"Hides Source Area")
        
    def sourceReparent(self,dict):
        if self.sourceFrame.parent() is not None:
            self.sourceFrame.reparent(None,self.cursor().pos(),1)
        else:
            self.sourceFrame.reparent(self.splitter,qt.QPoint(),1)
            self.splitter.moveToFirst(self.sourceFrame)
        
    def initSource(self):
        self.sourceWidget = {}
        i = 0
        for source in SOURCESLIST:
            data = SOURCES[source]['data']()
            wid = SOURCES[source]['widget'](self.sourceBrowserTab)
            wid.setData(data)
            self.sourceBrowserTab.insertTab(wid, source,i)
            self.sourceWidget[source]= wid
            #self.sourceBrowserTab.removePage(wid)
            i += 1

    def menuToolsAboutToShow(self):
        if DEBUG:
            print "menu ToolsAboutToShow"
        self.menuTools.clear()
        if self.sourceFrame.isHidden():
            self.menuTools.insertItem("Show Source",self.toggleSource)
        else:
            self.menuTools.insertItem("Hide Source",self.toggleSource)
        #self.menuTools.insertItem("Choose Font",self.fontdialog)
        self.menuTools.insertItem("Elements   Info",self.__elementsInfo)
        self.menuTools.insertItem("Identify  Peaks",self.__peakIdentifier)
        self.menuTools.insertItem("Batch   Fitting",self.__batchFitting)
        self.menuTools.insertItem("Fit to Specfile",self.__fit2SpecConversion)
        
    def fontdialog(self):
        fontd = qt.QFontDialog.getFont(self)
        if fontd[1]:
            qt.qApp.setFont(fontd[0],1)
           

    def toggleSource(self,**kw):
        if DEBUG:
            print "toggleSource called"
        if self.sourceFrame.isHidden():
            self.sourceFrame.show()
            self.sourceFrame.raiseW()
        else:
            self.sourceFrame.hide()
            
    def __elementsInfo(self):
        if self.elementsInfo is None:self.elementsInfo=ElementsInfo.ElementsInfo(None,"Elements Info")
        if self.elementsInfo.isHidden():
           self.elementsInfo.show()
        self.elementsInfo.raiseW()

    def __peakIdentifier(self):
        if self.identifier is None:
            self.identifier=PeakIdentifier.PeakIdentifier(energy=5.9,
                                useviewer=1)
            self.identifier.myslot()
        if self.identifier.isHidden():
            self.identifier.show()
        self.identifier.raiseW()
        
    def __batchFitting(self):
        if self.__batch is None:self.__batch = PyMcaBatch.McaBatchGUI(fl=0,actions=1)
        if self.__batch.isHidden():self.__batch.show()
        self.__batch.raiseW()

    def __fit2SpecConversion(self):
        if self.__fit2Spec is None:self.__fit2Spec = Fit2Spec.Fit2SpecGUI(fl=0,actions=1)
        if self.__fit2Spec.isHidden():self.__fit2Spec.show()
        self.__fit2Spec.raiseW()

    def onOpen(self):
        self.openMenu.exec_loop(self.cursor().pos())

    def onSave(self):
        self._saveAs()

    def onSaveAs(self):
        cwd = os.getcwd()
        outfile = qt.QFileDialog(self,"Output File Selection",1)
        if os.path.exists(self.configDir):
            outfile.setDir(self.configDir)        
        outfile.setFilters('PyMca  *.ini')
        outfile.setMode(outfile.AnyFile)
        ret = outfile.exec_loop()
        if ret:
            filterused = str(outfile.selectedFilter()).split()
            extension = ".ini"
            outdir=str(outfile.selectedFile())
            try:
                outputDir  = os.path.dirname(outdir)
            except:
                outputDir  = "."
            try:
                outputFile = os.path.basename(outdir)
            except:
                outputFile  = "PyMca.ini"
            outfile.close()
            del outfile
        else:
            outfile.close()
            del outfile
            return
        #always overwrite for the time being
        if len(outputFile) < len(extension[:]):
            outputFile += extension[:]
        elif outputFile[-4:] != extension[:]:
            outputFile += extension[:]
        filename = os.path.join(outputDir, outputFile)
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_loop()
                return
        try:
            self._saveAs(filename)
            self.configDir = outputDir
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error Saving Configuration: %s" % (sys.exc_info()[1]))
            msg.exec_loop()
            return

    def _saveAs(self, filename=None):
        if filename is None:filename = self.__getDefaultSettingsFile()
        self.saveConfig(self.getConfig(), filename)
        
    def __getDefaultSettingsFile(self):
        filename = "PyMca.ini"
        if sys.platform == 'win32':
            home = os.getenv('USERPROFILE')
            try:
                l = len(home)
                directory = os.path.join(home,"My Documents")
            except:
                home = '\\'
                directory = '\\'
            #print home
            #print directory
            if os.path.isdir('%s' % directory):
                directory = os.path.join(directory,"PyMca")
            else:
                #print "My Documents is not there"
                directory = os.path.join(home,"PyMca")
            if not os.path.exists('%s' % directory):
                #print "PyMca directory not present"
                os.mkdir('%s' % directory)
            #print filename
            finalfile = os.path.join(directory, filename)
            #print finalfile
        else:
            home = os.getenv('HOME')
            directory = os.path.join(home,"PyMca")
            if not os.path.exists('%s' % directory):
                os.mkdir('%s' % directory)
            finalfile =  os.path.join(directory, filename)
        return finalfile

    def openSource(self,index):
        if DEBUG:
            print "index = ",index
            print "Source to open = ",SOURCESLIST[index]
        if index <= 0:
            outfile = qt.QFileDialog(self,"Select PyMca Configuration File",1)
            if os.path.exists(self.configDir):
                outfile.setDir(self.configDir)        
            outfile.setFilters('PyMca  *.ini')
            outfile.setMode(outfile.ExistingFile)
            ret = outfile.exec_loop()
            if ret:
                filename=str(outfile.selectedFile())
                outfile.close()
                del outfile
            else:
                outfile.close()
                del outfile
                return
            currentConfigDict = ConfigDict.ConfigDict()
            self.configDir  = os.path.dirname(filename)
            currentConfigDict.read(filename)
            self.setConfig(currentConfigDict)
            return
        else:
            index -= 1
        source = SOURCESLIST[index]
        if self.sourceFrame.isHidden():
            self.sourceFrame.show()
        self.sourceFrame.raiseW()
        self.sourceBrowserTab.showPage(self.sourceWidget[source])
        qt.qApp.processEvents()
        self.sourceWidget[source].openFile()
        
    def onMenuHelp(self):
        if self.menuBrowser is None:
            self.menuBrowser= qt.QTextBrowser()
            self.menuBrowser.setCaption(qt.QString("Main Menu Help"))
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(dir+"/HTML"+"/Menu.html"):
                dir = os.path.dirname(dir)
            self.menuBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
            self.menuBrowser.setSource(qt.QString("Menu.html"))
            self.menuBrowser.show()
        if self.menuBrowser.isHidden():self.menuBrowser.show()
        self.menuBrowser.raiseW()

    def onDisplayHowto(self):
        if self.displayBrowser is None:
            self.displayBrowser= qt.QTextBrowser()
            self.displayBrowser.setCaption(qt.QString("Data Display HOWTO"))
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(dir+"/HTML"+"/Display-HOWTO.html"):
                dir = os.path.dirname(dir)
            self.displayBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
            self.displayBrowser.setSource(qt.QString("Display-HOWTO.html"))
            self.displayBrowser.show()
        if self.displayBrowser.isHidden():self.displayBrowser.show()
        self.displayBrowser.raiseW()
    
    def onMcaHowto(self):
        if self.mcaBrowser is None:
            self.mcaBrowser= MyQTextBrowser()
            self.mcaBrowser.setCaption(qt.QString("MCA HOWTO"))
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(dir+"/HTML"+"/MCA-HOWTO.html"):
                dir = os.path.dirname(dir)
            self.mcaBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
            self.mcaBrowser.setSource(qt.QString("MCA-HOWTO.html"))
            self.mcaBrowser.show()
        if self.mcaBrowser.isHidden():self.mcaBrowser.show()
        self.mcaBrowser.raiseW()

    def onAbout(self):
            qt.QMessageBox.about(self, "PyMca",
                    "PyMca Application\nVersion: "+__version__)
            #self.onDebug()
    
    
    def onDebug(self):
        print "Module name = ","PyMca",__revision__.replace("$","")
        for module in sys.modules.values():
            try:
                if 'Revision' in module.__revision__:
                    if module.__name__ != "__main__":
                        print "Module name = ",module.__name__,module.__revision__.replace("$","")
            except:
                pass 
    
    def onPrint(self):
        if DEBUG:
            print "onPrint called"
        if self.mcawindow.scanwindow.hasFocus():
            self.mcawindow.scanwindow.graph.printps() 
        else:
            self.mcawindow.show()
            self.mcawindow.raiseW()
            self.mcawindow.graph.printps()    

    def __McaWindowSignal(self,dict):
        if dict['event'] == 'NewScanCurve':
            if self.mcawindow.scanwindow.isHidden():
                self.mcawindow.scanwindow.show()
            self.mcawindow.scanwindow.setFocus()   

if 0:            
            
    #
    # GraphWindow operations
    #
    def openGraph(self,name="MCA Graph"):
            """
            Creates a new GraphWindow on the MDI
            """
            self.setFollowActiveWindow(0)

            #name= self.__getNewGraphName()
            if name == "MCA Graph":
                graph= McaWindow.McaWindow(self.mdi, name=name)
            self.connect(graph, qt.SIGNAL("windowClosed()"), self.closeGraph)
            graph.show()

            if len(self.mdi.windowList())==1:
                    graph.showMaximized()
            else:   self.windowTile()

            self.setFollowActiveWindow(1)

    def getActiveGraph(self):
            """
            Return active GraphWindow instance or a new one
            """
            graph= self.mdi.activeWindow()
            if graph is None:
                    graph= self.openGraph()
            return graph

    def getGraph(self, name):
            """
            Return GraphWindow instance indexed by name
            Or None if not found
            """
            for graph in self.mdi.windowList():
                    if str(graph.caption())==name:
                            return graph
            return None

    def closeGraph(self, name):
            """
            Called after a graph is closed
            """
            print "closeGraph", name

    def __getGraphNames(self):
            return [ str(window.caption()) for window in self.mdi.windowList() ]


    def __getNewGraphName(self):
            names= self.__getGraphNames()
            idx= 0
            while "Graph %d"%idx in names:
                    idx += 1
            return "Graph %d"%(idx)


class MyQTextBrowser(qt.QTextBrowser):
    def  setSource(self,name):
        if name == qt.QString("./PyMCA.html"):
            if sys.platform == 'win32':
                dir=os.path.dirname(PyMcaMdi.__file__)
                if not os.path.exists(dir+"/HTML"+"/PyMCA.html"):
                    dir = os.path.dirname(dir)
                cmd = dir+"/HTML/PyMCA.pdf"
                os.system('"%s"' % cmd)
                return
            try:
                self.report.show()
            except:
                self.report = qt.QTextBrowser()
                self.report.setCaption(qt.QString("PyMca Report"))
                self.report.mimeSourceFactory().addFilePath(qt.QString(os.path.dirname(PyMcaMdi.__file__)+"/HTML"))
                self.report.mimeSourceFactory().addFilePath(qt.QString(os.path.dirname(PyMcaMdi.__file__)+"/HTML/PyMCA_files"))
                self.report.setSource(name)
            if self.report.isHidden():self.report.show()
            self.report.raiseW()
        else:
            qt.QTextBrowser.setSource(self,name)                           
            
class Line(qt.QFrame):
    def mouseDoubleClickEvent(self,event):
        if DEBUG:
            print "Double Click Event"
        dict={}
        dict['event']="DoubleClick"
        dict['data'] = event
        self.emit(qt.PYSIGNAL("LineDoubleClickEvent"),(dict,))

class PixmapLabel(qt.QLabel):
    def mousePressEvent(self,event):
        if DEBUG:
            print "Mouse Press Event"
        dict={}
        dict['event']="MousePress"
        dict['data'] = event
        self.emit(qt.PYSIGNAL("PixmapLabelMousePressEvent"),(dict,))

if __name__ == '__main__':
    options     = '-f'
    longoptions = ['spec=','shm=','debug=']
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except getopt.error,msg:
        print msg
        splash.close()
        sys.exit(1)
    
    kw={}
    debugreport = 0
    for opt, arg in opts:
        if  opt in ('--spec'):
            kw['spec'] = arg
        elif opt in ('--shm'):
            kw['shm']  = arg
        elif opt in ('--debug'):
            debugreport = 1
        elif opt in ('-f'):
            kw['fresh'] = 1
    #demo = McaWindow.McaWidget(**kw)
    demo = PyMca(**kw)
    if debugreport:demo.onDebug()
    app.setMainWidget(demo)
    demo.show()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
    # --- close waiting widget
    splash.close()
    app.exec_loop()

