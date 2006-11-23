#!/usr/bin/env python
__revision__ = "$Revision: 1.64 $"
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
import PyMcaMdi
from PyMcaMdi import qt

QTVERSION = qt.qVersion()

from PyMca_Icons import IconDict
from PyMca_help import HelpDict
import os
__version__ = "3.9.4 Qt3 & Qt4 RC1"
if (QTVERSION < '4.0.0') and ((sys.platform == 'darwin') or (qt.qVersion() < '3.0.0')):
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
    if 1 or QTVERSION < '4.0.0':
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)
    else:
        palette = app.palette()
        role = qt.QPalette.Window           #this is the background
        palette.setColor(role, qt.QColor(238,234,238))
        app.setPalette(palette)

    mpath = os.path.dirname(PyMcaMdi.__file__)
    if mpath[-3:] == "exe":
        mpath = os.path.dirname(mpath)
    if QTVERSION < '4.0.0':
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
    else:
        pixmap = qt.QPixmap(qt.QString(os.path.join(mpath,'PyMcaSplashImage.png')))
        splash  = qt.QSplashScreen(pixmap)
        splash.show()
        font = splash.font()
        font.setBold(1)
        splash.setFont(font)
        splash.showMessage( 'PyMCA %s' % __version__, 
                qt.Qt.AlignLeft|qt.Qt.AlignBottom, 
                qt.Qt.white)

import McaWindow
import ScanWindow
import QDispatcher
import ElementsInfo
import PeakIdentifier
import PyMcaBatch
###########import Fit2Spec
if QTVERSION > '4.0.0':
    import PyMcaPostBatch
import ConfigDict

DEBUG = 0
SOURCESLIST = QDispatcher.QDataSource.source_types.keys()

"""

SOURCES = {"SpecFile":{'widget':SpecFileSelector.SpecFileSelector,'data':SpecFileLayer.SpecFileLayer},
           "EdfFile":{'widget':EdfFileSelector.EdfFileSelector,'data':EdfFileLayer.EdfFileLayer}}

SOURCESLIST = ["SpecFile","EdfFile"]

if (sys.platform != 'win32') and (sys.platform != 'darwin'):
    import MySPSSelector as SPSSelector
    SOURCES["SPS"] = {'widget':SPSSelector.SPSSelector,'data':SPSLayer.SPSLayer}
    SOURCESLIST.append("SPS")
"""

class PyMca(PyMcaMdi.PyMca):
    def __init__(self, parent=None, name="PyMca", fl=None,**kw):
            if QTVERSION < '4.0.0':
                if fl is None:qt.Qt.WDestructiveClose
                PyMcaMdi.PyMca.__init__(self, parent, name, fl)
                self.setCaption(name)
                self.setIcon(qt.QPixmap(IconDict['gioconda16']))
                self.menuBar().setIcon(qt.QPixmap(IconDict['gioconda16']))            
            else:
                if fl is None: fl = qt.Qt.WA_DeleteOnClose
                PyMcaMdi.PyMca.__init__(self, parent, name, fl)
                self.setWindowTitle(name)
                self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
            self.initSourceBrowser()
            self.initSource()
            self.elementsInfo= None
            self.identifier  = None
            self.__batch     = None
            self.__fit2Spec  = None
            self.__correlator  = None
            if QTVERSION < '4.0.0':
                self.openMenu = qt.QPopupMenu()
                self.openMenu.insertItem("PyMca Configuration",0)
                self.openMenu.insertItem("Data Source",1)
                self.connect(self.openMenu,qt.SIGNAL('activated(int)'),self.openSource)
            else:
                self.openMenu = qt.QMenu()
                self.openMenu.addAction("PyMca Configuration", self.openSource)
                self.openMenu.addAction("Data Source",
                             self.sourceWidget.sourceSelector._openFileSlot)
                self.connect(self.openMenu,qt.SIGNAL('activated(int)'),self.openSource)

            self.mcawindow = McaWindow.McaWidget(self.mdi)
            self.scanwindow = ScanWindow.ScanWindow(self.mdi)
            self.scanwindow.setDispatcher(self.sourceWidget)
            self.connectDispatcher(self.mcawindow, self.sourceWidget)
            self.connectDispatcher(self.scanwindow, self.sourceWidget)
                                   
            if QTVERSION < '4.0.0':
                pass
            else:
                self.mdi.addWindow(self.mcawindow)
                self.mdi.addWindow(self.scanwindow)
                #self.scanwindow.showMaximized()
                #self.mcawindow.showMaximized()

            if 0:
                if QTVERSION < '4.0.0':
                    self.connect(self.mcawindow,qt.PYSIGNAL('McaWindowSignal'),
                                 self.__McaWindowSignal)
                else:
                    self.connect(self.mcawindow,qt.SIGNAL('McaWindowSignal'),
                                 self.__McaWindowSignal)
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

    def connectDispatcher(self, viewer, dispatcher = None):
        #I could connect sourceWidget to myself and then
        #pass the selections to the active window!!
        #That will be made in a next iteration I guess
        if dispatcher is None: dispatcher = self.sourceWidget
        if QTVERSION < '4.0.0':
            self.connect(dispatcher, qt.PYSIGNAL("addSelection"),
                             viewer._addSelection)
            self.connect(dispatcher, qt.PYSIGNAL("removeSelection"),
                             viewer._removeSelection)
            self.connect(dispatcher, qt.PYSIGNAL("replaceSelection"),
                             viewer._replaceSelection)
        else:
            self.connect(dispatcher, qt.SIGNAL("addSelection"),
                             viewer._addSelection)
            self.connect(dispatcher, qt.SIGNAL("removeSelection"),
                             viewer._removeSelection)
            self.connect(dispatcher, qt.SIGNAL("replaceSelection"),
                             viewer._replaceSelection)
                
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
            if self.sourceWidget.sourceSelector.lastInputDir is not None:
                d['PyMca'][source]['lastInputDir'] = self.sourceWidget.sourceSelector.lastInputDir
            else:
                d['PyMca'][source]['lastInputDir'] = "None"
            if source == "SpecFile":
                d['PyMca'][source]['SourceName'] = []
                for key in self.sourceWidget.sourceList:
                    if key.sourceType == source:
                        d['PyMca'][source]['SourceName'].append(key.sourceName)
            elif source == "EdfFile":
                d['PyMca'][source]['SourceName'] = []
                for key in self.sourceWidget.sourceList:
                    if key.sourceType == source:
                        d['PyMca'][source]['SourceName'].append(key.sourceName)
                    #if key == "EDF Stack":
                    #    d['PyMca'][source]['SourceName'].append(self.sourceWidget.selectorWidget[source]._edfstack)
                    #else:
                    #    d['PyMca'][source]['SourceName'].append(self.sourceWidget.selectorWidget[source].mapComboName[key])
            
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
                        self.sourceWidget.sourceSelector.lastInputDir =  dict[source] ['lastInputDir']
                    #else:
                    #    self.sourceWidget.selectorWidget[source].lastInputDir =  None
                if dict[source].has_key('SourceName'):
                    if type(dict[source]['SourceName']) != type([]):
                        dict[source]['SourceName'] = [dict[source]['SourceName'] * 1]
                    for SourceName in dict[source]['SourceName']:
                        if len(SourceName):
                            try:
                                ndict = {}
                                ndict["event"] = "NewSourceSelected"
                                ndict["sourcelist"] = [SourceName]
                                self.sourceWidget._sourceSelectorSlot(ndict)
                                continue
                                if source == "EdfFile":
                                    self.sourceWidget.selectorWidget[source].openFile(SourceName, justloaded=1)
                                else:
                                    self.sourceWidget.selectorWidget[source].openFile(SourceName)
                            except:
                                msg = qt.QMessageBox(self)
                                msg.setIcon(qt.QMessageBox.Critical)
                                msg.setText("Error: %s\n opening file %s" % (sys.exc_info()[1],SourceName ))
                                if QTVERSION < '4.0.0':
                                    msg.exec_loop()
                                else:
                                    msg.exec_()
                """
                if dict[source].has_key('Selection'):
                    if type(dict[source]['Selection']) != type([]):
                        dict[source]['Selection'] = [dict[source]['Selection']]
                    if source == "EdfFile":
                        self.sourceWidget[source].setSelected(dict[source]['Selection'])
                """    


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
            if QTVERSION < '4.0.0':
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

            else:
                #build the actions
                #fileopen
                self.actionOpen = qt.QAction(self)
                self.actionOpen.setText(qt.QString("&Open"))
                self.actionOpen.setIcon(self.Icons["fileopen"])
                self.actionOpen.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
                self.connect(self.actionOpen, qt.SIGNAL("triggered(bool)"),
                             self.onOpen)
                #filesaveas
                self.actionSaveAs = qt.QAction(self)
                self.actionSaveAs.setText(qt.QString("&Save"))
                self.actionSaveAs.setIcon(self.Icons["filesave"])
                self.actionSaveAs.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSaveAs, qt.SIGNAL("triggered(bool)"),
                             self.onSaveAs)

                #filesave
                self.actionSave = qt.QAction(self)
                self.actionSave.setText(qt.QString("Save &Defaults"))
                #self.actionSave.setIcon(self.Icons["filesave"])
                #self.actionSave.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSave, qt.SIGNAL("triggered(bool)"),
                             self.onSave)
                #fileprint
                self.actionPrint = qt.QAction(self)
                self.actionPrint.setText(qt.QString("&Print"))
                self.actionPrint.setIcon(self.Icons["fileprint"])
                self.actionPrint.setShortcut(qt.Qt.CTRL+qt.Qt.Key_P)
                self.connect(self.actionPrint, qt.SIGNAL("triggered(bool)"),
                             self.onPrint)

                #filequit
                self.actionQuit = qt.QAction(self)
                self.actionQuit.setText(qt.QString("&Quit"))
                #self.actionQuit.setIcon(self.Icons["fileprint"])
                self.actionQuit.setShortcut(qt.Qt.CTRL+qt.Qt.Key_Q)
                qt.QObject.connect(self.actionQuit,
                                   qt.SIGNAL("triggered(bool)"),
                                   qt.qApp,
                                   qt.SLOT("closeAllWindows()"))

                #self.menubar = qt.QMenuBar(self)
                self.menuFile= qt.QMenu(self.menuBar())
                self.menuFile.addAction(self.actionOpen)
                self.menuFile.addAction(self.actionSaveAs)
                self.menuFile.addAction(self.actionSave)
                self.menuFile.addSeparator()
                self.menuFile.addAction(self.actionPrint)
                self.menuFile.addSeparator()
                self.menuFile.addAction(self.actionQuit)
                self.menuBar().addMenu(self.menuFile)
                self.menuFile.setTitle("&File")
                self.onInitMenuBar(self.menuBar())

        if self.options["MenuTools"]:
            if QTVERSION < '4.0.0':
                self.menuTools= qt.QPopupMenu()
                self.menuTools.setCheckable(1)
                self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"), self.menuToolsAboutToShow)
                self.menuBar().insertItem("&Tools", self.menuTools)
            else:
                self.menuTools= qt.QMenu()
                #self.menuTools.setCheckable(1)
                self.connect(self.menuTools, qt.SIGNAL("aboutToShow()"),
                             self.menuToolsAboutToShow)
                self.menuTools.setTitle("&Tools")
                self.menuBar().addMenu(self.menuTools)

        if self.options["MenuWindow"]:
            if QTVERSION < '4.0.0':
                self.menuWindow= qt.QPopupMenu()
                self.menuWindow.setCheckable(1)
                self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
                self.menuBar().insertItem("&Window", self.menuWindow)
            else:
                self.menuWindow= qt.QMenu()
                #self.menuWindow.setCheckable(1)
                self.connect(self.menuWindow, qt.SIGNAL("aboutToShow()"), self.menuWindowAboutToShow)
                self.menuWindow.setTitle("&Window")
                self.menuBar().addMenu(self.menuWindow)

        if self.options["MenuHelp"]:
            if QTVERSION < '4.0.0':
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
            else:
                self.menuHelp= qt.QMenu(self)
                self.menuHelp.addAction("&Menu", self.onMenuHelp)
                self.menuHelp.addAction("&Data Display HOWTOs", self.onDisplayHowto)
                self.menuHelp.addAction("MCA &HOWTOs",self.onMcaHowto)
                self.menuHelp.addSeparator()
                self.menuHelp.addAction("&About", self.onAbout)
                self.menuHelp.addAction("About &Qt",self.onAboutQt)
                self.menuBar().addSeparator()
                self.menuHelp.setTitle("&Help")
                self.menuBar().addMenu(self.menuHelp)
                self.menuBrowser    = None
                self.displayBrowser = None
                self.mcaBrowser     = None


    def initSourceBrowser(self):
        self.sourceFrame     = qt.QWidget(self.splitter)
        if QTVERSION < '4.0.0':
            self.splitter.moveToFirst(self.sourceFrame)
        else:
            self.splitter.insertWidget(0, self.sourceFrame)
        #self.splitter.setResizeMode(self.sourceFrame,qt.QSplitter.KeepSize)
        self.sourceFrameLayout = qt.QVBoxLayout(self.sourceFrame)
        self.sourceFrameLayout.setMargin(0)
        self.sourceFrameLayout.setSpacing(0)
        #layout.setAutoAdd(1)
        
        sourceToolbar = qt.QWidget(self.sourceFrame)
        layout1       = qt.QHBoxLayout(sourceToolbar)
        #self.line1 = qt.QFrame(sourceToolbar,"line1")
        self.line1 = Line(sourceToolbar)
        self.line1.setFrameShape(qt.QFrame.HLine)
        self.line1.setFrameShadow(qt.QFrame.Sunken)
        self.line1.setFrameShape(qt.QFrame.HLine)
        layout1.addWidget(self.line1)
        #self.closelabel = qt.QLabel(sourceToolbar)
        self.closelabel = PixmapLabel(sourceToolbar)
        self.closelabel.setPixmap(qt.QPixmap(IconDict['close']))
        layout1.addWidget(self.closelabel)
        self.closelabel.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        #self.sourceBrowserTab=qt.QTabWidget(self.sourceFrame)

        self.sourceFrameLayout.addWidget(sourceToolbar)
        
        if QTVERSION < '4.0.0':
            #connections
            self.connect(self.line1,qt.PYSIGNAL("LineDoubleClickEvent"),self.sourceReparent)
            self.connect(self.closelabel,qt.PYSIGNAL("PixmapLabelMousePressEvent"),self.toggleSource)
        
            #tips
            qt.QToolTip.add(self.line1,"DoubleClick toggles floating window mode")
            qt.QToolTip.add(self.closelabel,"Hides Source Area")
        else:
            #connections
            self.connect(self.line1,qt.SIGNAL("LineDoubleClickEvent"),self.sourceReparent)
            self.connect(self.closelabel,qt.SIGNAL("PixmapLabelMousePressEvent"),self.toggleSource)
        
            #tips
            self.line1.setToolTip("DoubleClick toggles floating window mode")
            self.closelabel.setToolTip("Hides Source Area")
            
    def sourceReparent(self,dict):
        if self.sourceFrame.parent() is not None:
            if QTVERSION < '4.0.0':
                self.sourceFrame.reparent(None,self.cursor().pos(),1)
                self.splitter.moveToFirst(self.sourceFrame)
            else:
                self.sourceFrame.setParent(None)
                self.sourceFrame.show()
                #,self.cursor().pos(),1)
        else:
            if QTVERSION < '4.0.0':
                self.sourceFrame.reparent(self.splitter,qt.QPoint(),1)
                self.splitter.moveToFirst(self.sourceFrame)
            else:
                self.sourceFrame.setParent(self.splitter)
        
    def initSource(self):
        self.sourceWidget = QDispatcher.QDispatcher(self.sourceFrame)
        self.sourceFrameLayout.addWidget(self.sourceWidget)

    def menuToolsAboutToShow(self):
        if DEBUG:
            print "menu ToolsAboutToShow"
        self.menuTools.clear()
        if QTVERSION < '4.0.0':
            if self.sourceFrame.isHidden():
                self.menuTools.insertItem("Show Source",self.toggleSource)
            else:
                self.menuTools.insertItem("Hide Source",self.toggleSource)
            #self.menuTools.insertItem("Choose Font",self.fontdialog)
            self.menuTools.insertItem("Elements   Info",self.__elementsInfo)
            self.menuTools.insertItem("Identify  Peaks",self.__peakIdentifier)
            self.menuTools.insertItem("Batch   Fitting",self.__batchFitting)
            #self.menuTools.insertItem("Fit to Specfile",self.__fit2SpecConversion)
        else:
            if self.sourceFrame.isHidden():
                self.menuTools.addAction("Show Source",self.toggleSource)
            else:
                self.menuTools.addAction("Hide Source",self.toggleSource)
            self.menuTools.addAction("Elements   Info",self.__elementsInfo)
            self.menuTools.addAction("Identify  Peaks",self.__peakIdentifier)
            self.menuTools.addAction("Batch   Fitting",self.__batchFitting)
            #self.menuTools.addAction("Fit to Specfile",self.__fit2SpecConversion)
            self.menuTools.addAction("RGB Correlator",self.__rgbCorrelator)
        if DEBUG:"print Fit to Specfile missing"
    def fontdialog(self):
        fontd = qt.QFontDialog.getFont(self)
        if fontd[1]:
            qt.qApp.setFont(fontd[0],1)
           

    def toggleSource(self,**kw):
        if DEBUG:
            print "toggleSource called"
        if self.sourceFrame.isHidden():
            self.sourceFrame.show()
            if QTVERSION < '4.0.0': self.sourceFrame.raiseW()
            else:self.sourceFrame.raise_()
        else:
            self.sourceFrame.hide()
            
    def __elementsInfo(self):
        if self.elementsInfo is None:self.elementsInfo=ElementsInfo.ElementsInfo(None,"Elements Info")
        if self.elementsInfo.isHidden():
           self.elementsInfo.show()
        if QTVERSION < '4.0.0': self.elementsInfo.raiseW()
        else:self.elementsInfo.raise_()

    def __peakIdentifier(self):
        if self.identifier is None:
            self.identifier=PeakIdentifier.PeakIdentifier(energy=5.9,
                                useviewer=1)
            self.identifier.myslot()
        if self.identifier.isHidden():
            self.identifier.show()
        if QTVERSION < '4.0.0': self.identifier.raiseW()
        else:self.identifier.raise_()
            
    def __batchFitting(self):
        if self.__batch is None:self.__batch = PyMcaBatch.McaBatchGUI(fl=0,actions=1)
        if self.__batch.isHidden():self.__batch.show()
        if QTVERSION < '4.0.0': self.__batch.raiseW()
        else: self.__batch.raise_()

    def __fit2SpecConversion(self):
        if self.__fit2Spec is None:self.__fit2Spec = Fit2Spec.Fit2SpecGUI(fl=0,actions=1)
        if self.__fit2Spec.isHidden():self.__fit2Spec.show()
        if QTVERSION < '4.0.0': self.__fit2Spec.raiseW()
        else:self.__fit2Spec.raise_()

    def __rgbCorrelator(self):
        if self.__correlator is None:self.__correlator = []
        wdir = os.getcwd()
        if self.sourceWidget.sourceSelector.lastInputDir is not None:
            if os.path.exists(self.sourceWidget.sourceSelector.lastInputDir):
                wdir =  self.sourceWidget.sourceSelector.lastInputDir
        fileTypeList = ["Batch Result Files (*dat)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "All Files (*)"]

        if sys.platform != 'darwin':
            filetypes = ""
            for filetype in fileTypeList:
                filetypes += filetype+"\n"
            filelist = qt.QFileDialog.getOpenFileNames(self,
                        "Open a Batch result file or several EDF files",
                        wdir,
                        filetypes)
            if not len(filelist):return
        else:
            fdialog = qt.QFileDialog(self)
            fdialog.setModal(True)
            fdialog.setWindowTitle("Open ONE Batch result file or SEVERAL EDF files")
            strlist = qt.QStringList()
            for filetype in fileTypeList:
                strlist.append(filetype.replace("(","").replace(")",""))
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
        filelist = map(str, filelist)
        self.sourceWidget.sourceSelector.lastInputDir = os.path.dirname(filelist[0])
        self.__correlator.append(PyMcaPostBatch.PyMcaPostBatch())
        for correlator in self.__correlator:
            if correlator.isHidden():
                correlator.show()
            correlator.raise_()
        self.connect(self.__correlator[-1],
                     qt.SIGNAL("RGBCorrelatorSignal"),
                     self._deleteCorrelator)
        if len(filelist) == 1:
            correlator.addBatchDatFile(filelist[0])
        else:
            correlator.addFileList(filelist)

    def _deleteCorrelator(self, ddict):        
        n = len(self.__correlator)
        if ddict['event'] == "RGBCorrelatorClosed":
            for i in range(n):
                if id(self.__correlator[i]) == ddict["id"]:
                    self.__correlator[i].deleteLater()
                    del self.__correlator[i]
                    break
    
    def onOpen(self):
        if QTVERSION < '4.0.0':
            self.openMenu.exec_loop(self.cursor().pos())
        else:
            self.openMenu.exec_(self.cursor().pos())

    def onSave(self):
        self._saveAs()

    def onSaveAs(self):
        cwd = os.getcwd()
        if QTVERSION < '4.0.0':
            outfile = qt.QFileDialog(self,"Output File Selection",1)
            outfile.setFilters('PyMca  *.ini')
            outfile.setMode(outfile.AnyFile)
        else:
            outfile = qt.QFileDialog(self)
            outfile.setFilter('PyMca  *.ini')
            outfile.setFileMode(outfile.AnyFile)

        if os.path.exists(self.configDir):cwd =self.configDir 
        if QTVERSION < '4.0.0': outfile.setDir(cwd)
        else: outfile.setDirectory(cwd)
        if QTVERSION < '4.0.0':ret = outfile.exec_loop()
        else:ret = outfile.exec_()
        if ret:
            filterused = str(outfile.selectedFilter()).split()
            extension = ".ini"
            if QTVERSION < '4.0.0':
                outdir=str(outfile.selectedFile())
            else:
                outdir=str(outfile.selectedFiles()[0])
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
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
        try:
            self._saveAs(filename)
            self.configDir = outputDir
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error Saving Configuration: %s" % (sys.exc_info()[1]))
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
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

    def openSource(self,index=0):
        if DEBUG:
            print "index = ",index
        if index <= 0:
            if QTVERSION < '4.0.0':
                outfile = qt.QFileDialog(self,"Select PyMca Configuration File",1)
                if os.path.exists(self.configDir):
                    outfile.setDir(self.configDir)        
                outfile.setFilters('PyMca  *.ini')
                outfile.setMode(outfile.ExistingFile)
                ret = outfile.exec_loop()
            else:
                outfile = qt.QFileDialog(self)
                outfile.setWindowTitle("Select PyMca Configuration File")
                if os.path.exists(self.configDir):
                    outfile.setDirectory(self.configDir)        
                outfile.setFilters(['PyMca  *.ini'])
                outfile.setFileMode(outfile.ExistingFile)
                ret = outfile.exec_()
            if ret:
                if QTVERSION < '4.0.0':filename=str(outfile.selectedFile())
                else:filename=str(outfile.selectedFiles()[0])
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
        if QTVERSION < '4.0.0':self.sourceFrame.raiseW()
        else:self.sourceFrame.raise_()
        self.sourceBrowserTab.showPage(self.sourceWidget[source])
        qt.qApp.processEvents()
        self.sourceWidget[source].openFile()
        
    def onMenuHelp(self):
        if self.menuBrowser is None:
            self.menuBrowser= qt.QTextBrowser()
            if QTVERSION < '4.0.0':
                self.menuBrowser.setCaption(qt.QString("Main Menu Help"))
            else:
                self.menuBrowser.setWindowTitle(qt.QString("Main Menu Help"))
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(os.path.join(dir,"HTML","Menu.html")):
                dir = os.path.dirname(dir)
            if QTVERSION < '4.0.0':
                self.menuBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
                self.menuBrowser.setSource(qt.QString("Menu.html"))
            else:
                self.menuBrowser.setSearchPaths([os.path.join(dir,"HTML")])
                self.menuBrowser.setSource(qt.QUrl(qt.QString("Menu.html")))
            self.menuBrowser.show()
        if self.menuBrowser.isHidden():self.menuBrowser.show()
        if QTVERSION < '4.0.0': self.menuBrowser.raiseW()
        else: self.menuBrowser.raise_()

    def onDisplayHowto(self):
        if self.displayBrowser is None:
            self.displayBrowser= qt.QTextBrowser()
            if QTVERSION < '4.0.0':
                self.displayBrowser.setCaption(qt.QString("Data Display HOWTO"))
            else:
                self.displayBrowser.setWindowTitle(qt.QString("Data Display HOWTO"))                
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(os.path.join(dir,"HTML","Display-HOWTO.html")):
                dir = os.path.dirname(dir)
            if QTVERSION < '4.0.0':
                self.displayBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
                self.displayBrowser.setSource(qt.QString("Display-HOWTO.html"))
            else:
                self.displayBrowser.setSearchPaths([os.path.join(dir,"HTML")])
                self.displayBrowser.setSource(qt.QUrl(qt.QString("Display-HOWTO.html")))
            self.displayBrowser.show()
        if self.displayBrowser.isHidden():self.displayBrowser.show()
        if QTVERSION < '4.0.0':self.displayBrowser.raiseW()
        else:self.displayBrowser.raise_()
    
    def onMcaHowto(self):
        if self.mcaBrowser is None:
            self.mcaBrowser= MyQTextBrowser()
            if QTVERSION < '4.0.0':
                self.mcaBrowser.setCaption(qt.QString("MCA HOWTO"))
            else:
                self.mcaBrowser.setWindowTitle(qt.QString("MCA HOWTO"))
            dir=os.path.dirname(PyMcaMdi.__file__)
            if not os.path.exists(dir+"/HTML"+"/MCA-HOWTO.html"):
                dir = os.path.dirname(dir)
            if QTVERSION < '4.0.0':
                self.mcaBrowser.mimeSourceFactory().addFilePath(qt.QString(dir+"/HTML"))
                self.mcaBrowser.setSource(qt.QString("MCA-HOWTO.html"))
            else:
                self.mcaBrowser.setSearchPaths([os.path.join(dir,"HTML")])
                self.mcaBrowser.setSource(qt.QUrl(qt.QString("MCA-HOWTO.html")))
                #f = open(os.path.join(dir,"HTML","MCA-HOWTO.html"))
                #self.mcaBrowser.setHtml(f.read())
                #f.close()
            self.mcaBrowser.show()
        if self.mcaBrowser.isHidden():self.mcaBrowser.show()
        if QTVERSION < '4.0.0':self.mcaBrowser.raiseW()
        else:self.mcaBrowser.raise_()
            
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
        if self.scanwindow.hasFocus():
            self.scanwindow.graph.printps() 
        else:
            self.mcawindow.show()
            if QTVERSION < '4.0.0':self.mcawindow.raiseW()
            else:self.mcawindow.raise_()
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
        ddict={}
        ddict['event']="DoubleClick"
        ddict['data'] = event
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("LineDoubleClickEvent"),(ddict,))
        else:
            self.emit(qt.SIGNAL("LineDoubleClickEvent"),(ddict))

class PixmapLabel(qt.QLabel):
    def mousePressEvent(self,event):
        if DEBUG:
            print "Mouse Press Event"
        ddict={}
        ddict['event']="MousePress"
        ddict['data'] = event
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("PixmapLabelMousePressEvent"),(ddict,))
        else:
            self.emit(qt.SIGNAL("PixmapLabelMousePressEvent"),(ddict))

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
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        # --- close waiting widget
        splash.close()
        app.exec_loop()
    else:
        demo.show()
        # --- close waiting widget
        splash.close()
        app.exec_()

