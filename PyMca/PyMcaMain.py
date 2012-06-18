#!/usr/bin/env python
__revision__ = "$Revision: 2.02 $"
#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
import sys, getopt
import traceback
nativeFileDialogs = None
DEBUG = 0
if __name__ == '__main__':
    options     = '-f'
    longoptions = ['spec=',
                   'shm=',
                   'debug=',
                   'qt=',
                   'nativefiledialogs=']
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except getopt.error:
        print(sys.exc_info()[1])
        sys.exit(1)
    
    keywords={}
    debugreport = 0
    qtversion = '4'
    for opt, arg in opts:
        if  opt in ('--spec'):
            keywords['spec'] = arg
        elif opt in ('--shm'):
            keywords['shm']  = arg
        elif opt in ('--debug'):
            debugreport = 1
            DEBUG = 1
        elif opt in ('-f'):
            keywords['fresh'] = 1
        elif opt in ('--qt'):
            qtversion = arg
        elif opt in ('--nativefiledialogs'):
            if int(arg):
                nativeFileDialogs = True
            else:
                nativeFileDialogs = False
    if qtversion == '3':
        import qt

from PyMca import PyMcaMdi
from PyMca import PyMcaQt as qt

if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str

QTVERSION = qt.qVersion()

from PyMca.PyMca_Icons import IconDict
from PyMca.PyMca_help import HelpDict
from PyMca import PyMcaDataDir
import os
__version__ = "4.6.0"
if (QTVERSION < '4.0.0') and (sys.platform == 'darwin'):
    class SplashScreen(qt.QWidget):
        def __init__(self,parent=None,name="SplashScreen",
                        fl=qt.Qt.WStyle_Customize  | qt.Qt.WDestructiveClose,
                        pixmap = None):
            qt.QWidget.__init__(self,parent,name,fl)
            self.setCaption("PyMca %s" % __version__)
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
    strlist = qt.QStyleFactory.keys()
    if sys.platform == "win32":
        for item in strlist:
            text = str(item)
            if text == "WindowsXP":
                style = qt.QStyleFactory.create(item)
                app.setStyle(style)
                break
    # TODO why this strange test
    if 1 or QTVERSION < '4.0.0':
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)
    else:
        palette = app.palette()
        role = qt.QPalette.Window           #this is the background
        palette.setColor(role, qt.QColor(238,234,238))
        app.setPalette(palette)

    mpath = PyMcaDataDir.PYMCA_DATA_DIR
    if mpath[-3:] == "exe":
        mpath = os.path.dirname(mpath)
    if QTVERSION < '4.0.0':
        qt.QMimeSourceFactory.defaultFactory().addFilePath(mpath)
        if (sys.platform == 'darwin'):
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
        fname = os.path.join(mpath,'PyMcaSplashImage.png')
        if not os.path.exists(fname):
           while len(mpath) > 3:
             fname = os.path.join(mpath,'PyMcaSplashImage.png')
             if not os.path.exists(fname):
                 mpath = os.path.dirname(mpath)
             else:
                 break
        if os.path.exists(fname):
            pixmap = qt.QPixmap(QString(fname))
            splash  = qt.QSplashScreen(pixmap)
        else:
            splash = qt.QSplashScreen()
        splash.show()
        from PyMca import ChangeLog
        font = splash.font()
        font.setBold(1)
        splash.setFont(font)
        splash.showMessage( 'PyMca %s' % __version__, 
                qt.Qt.AlignLeft|qt.Qt.AlignBottom, 
                qt.Qt.white)
        if sys.platform == "darwin":
            qt.qApp.processEvents()

from PyMca import McaWindow
from PyMca import ScanWindow
OBJECT3D = False
if QTVERSION > '4.0.0':
    from PyMca import PyMcaImageWindow
    from PyMca import PyMcaHKLImageWindow
    try:
        #This is to make sure it is properly frozen
        #and that Object3D is fully supported
        import OpenGL.GL
        import PyQt4.QtOpenGL
        #import Object3D.SceneGLWindow as SceneGLWindow
        import PyMca.PyMcaGLWindow as SceneGLWindow
        OBJECT3D = True
    except:
        OBJECT3D = False
from PyMca import QDispatcher
from PyMca import ElementsInfo
from PyMca import PeakIdentifier
from PyMca import PyMcaBatch
###########import Fit2Spec
from PyMca import Mca2Edf
STACK = False
if QTVERSION > '4.0.0':
    try:
        from PyMca import QStackWidget
        from PyMca import StackSelector
        STACK = True
    except:
        STACK = False
    from PyMca import PyMcaPostBatch
    from PyMca import RGBCorrelator
    from PyMca import MaterialEditor

from PyMca import ConfigDict
from PyMca import PyMcaDirs

XIA_CORRECT = False
if QTVERSION > '4.3.0':
    try:
        from PyMca import XiaCorrect
        XIA_CORRECT = True
    except:
        pass

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

class PyMcaMain(PyMcaMdi.PyMcaMdi):
    def __init__(self, parent=None, name="PyMca", fl=None,**kw):
            if QTVERSION < '4.0.0':
                if fl is None:qt.Qt.WDestructiveClose
                PyMcaMdi.PyMcaMdi.__init__(self, parent, name, fl)
                self.setCaption(name)
                self.setIcon(qt.QPixmap(IconDict['gioconda16']))
                self.menuBar().setIcon(qt.QPixmap(IconDict['gioconda16']))            
            else:
                if fl is None: fl = qt.Qt.WA_DeleteOnClose
                PyMcaMdi.PyMcaMdi.__init__(self, parent, name, fl)
                maxheight = qt.QDesktopWidget().height()
                if maxheight < 799:
                    self.setMinimumHeight(int(0.8*maxheight))
                    self.setMaximumHeight(int(0.9*maxheight))
                self.setWindowTitle(name)
                self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
                self.changeLog = None

            self._widgetDict = {}
            self.initSourceBrowser()
            self.initSource()

            self.elementsInfo= None
            self.attenuationTool =  None
            self.identifier  = None
            self.__batch     = None
            self.__mca2Edf   = None
            self.__fit2Spec  = None
            self.__correlator  = None
            self.__imagingTool = None
            if QTVERSION < '4.0.0':
                self.openMenu = qt.QPopupMenu()
                self.openMenu.insertItem("PyMca Configuration",0)
                self.openMenu.insertItem("Data Source",1)
                self.connect(self.openMenu,qt.SIGNAL('activated(int)'),
                             self.openSource)
            else:
                self.openMenu = qt.QMenu()
                self.openMenu.addAction("PyMca Configuration", self.openSource)
                self.openMenu.addAction("Data Source",
                             self.sourceWidget.sourceSelector._openFileSlot)
                self.openMenu.addAction("Load Training Data",
                                            self.loadTrainingData)

                #self.connect(self.openMenu,qt.SIGNAL('activated(int)'),self.openSource)

            if QTVERSION > '4.0.0':
                self.__useTabWidget = True
            else:
                self.__useTabWidget = False

            if not self.__useTabWidget:
                self.mcawindow = McaWindow.McaWidget(self.mdi)
                self.scanwindow = ScanWindow.ScanWindow(self.mdi)
                self.imageWindowDict = None
                self.connectDispatcher(self.mcawindow, self.sourceWidget)
                self.connectDispatcher(self.scanwindow, self.sourceWidget)
                if QTVERSION < '4.0.0':
                    pass
                else:
                    self.mdi.addWindow(self.mcawindow)
                    self.mdi.addWindow(self.scanwindow)
                    #self.scanwindow.showMaximized()
                    #self.mcawindow.showMaximized()
            else:
                if QTVERSION < '4.0.0':
                    self.mainTabWidget = qt.QTabWidget(self.mdi)
                    self.mainTabWidget.setCaption("Main Window")
                    self.mcawindow = McaWindow.McaWidget()
                    self.scanwindow = ScanWindow.ScanWindow()
                    self.mainTabWidget.addTab(self.mcawindow, "MCA")
                    self.mainTabWidget.addTab(self.scanwindow, "SCAN")
                    #self.mdi.addWindow(self.mainTabWidget)
                    self.mainTabWidget.showMaximized()
                    if False:
                        self.connectDispatcher(self.mcawindow, self.sourceWidget)
                        self.connectDispatcher(self.scanwindow, self.sourceWidget)
                    else:
                        self.imageWindowDict = {}
                        self.imageWindowCorrelator = None
                        self.connect(self.sourceWidget,
                                 qt.PYSIGNAL("addSelection"),
                                 self.dispatcherAddSelectionSlot)
                        self.connect(self.sourceWidget,
                                 qt.PYSIGNAL("removeSelection"),
                                 self.dispatcherRemoveSelectionSlot)
                        self.connect(self.sourceWidget,
                                 qt.PYSIGNAL("replaceSelection"),
                                 self.dispatcherReplaceSelectionSlot)
                        self.connect(self.mainTabWidget,
                                     qt.SIGNAL("currentChanged(QWidget*)"),
                                     self.currentTabIndexChanged)
                else:
                    self.mainTabWidget = qt.QTabWidget(self.mdi)
                    self.mainTabWidget.setWindowTitle("Main Window")
                    self.mcawindow = McaWindow.McaWidget()
                    self.scanwindow = ScanWindow.ScanWindow()
                    if OBJECT3D:
                        self.glWindow = SceneGLWindow.SceneGLWindow()
                    self.mainTabWidget.addTab(self.mcawindow, "MCA")
                    self.mainTabWidget.addTab(self.scanwindow, "SCAN")
                    if OBJECT3D:
                        self.mainTabWidget.addTab(self.glWindow, "OpenGL")
                    self.mdi.addWindow(self.mainTabWidget)
                    #print "Markus patch"
                    #self.mainTabWidget.show()
                    #print "end Markus patch"
                    self.mainTabWidget.showMaximized()
                    if False:
                        self.connectDispatcher(self.mcawindow, self.sourceWidget)
                        self.connectDispatcher(self.scanwindow, self.sourceWidget)
                    else:
                        self.imageWindowDict = {}
                        self.imageWindowCorrelator = None
                        self.connect(self.sourceWidget,
                                 qt.SIGNAL("addSelection"),
                                 self.dispatcherAddSelectionSlot)
                        self.connect(self.sourceWidget,
                                 qt.SIGNAL("removeSelection"),
                                 self.dispatcherRemoveSelectionSlot)
                        self.connect(self.sourceWidget,
                                 qt.SIGNAL("replaceSelection"),
                                 self.dispatcherReplaceSelectionSlot)
                        self.connect(self.mainTabWidget,
                                     qt.SIGNAL("currentChanged(int)"),
                                     self.currentTabIndexChanged)


            if QTVERSION < '4.0.0':
                self.connect(self.sourceWidget,
                             qt.PYSIGNAL("otherSignals"),
                             self.dispatcherOtherSignalsSlot)
            else:
                self.connect(self.sourceWidget,
                             qt.SIGNAL("otherSignals"),
                             self.dispatcherOtherSignalsSlot)
            if 0:
                if QTVERSION < '4.0.0':
                    self.connect(self.mcawindow,qt.PYSIGNAL('McaWindowSignal'),
                                 self.__McaWindowSignal)
                else:
                    self.connect(self.mcawindow,qt.SIGNAL('McaWindowSignal'),
                                 self.__McaWindowSignal)
                if 'shm' in kw:
                    if len(kw['shm']) >= 8:
                        if kw['shm'][0:8] in ['MCA_DATA', 'XIA_DATA']:
                            self.mcawindow.showMaximized()
                            self.toggleSource()
                else:
                    self.mcawindow.showMaximized()
            currentConfigDict = ConfigDict.ConfigDict()
            defaultFileName = self.__getDefaultSettingsFile()
            self.configDir  = os.path.dirname(defaultFileName)
            if not ('fresh' in kw):
                if os.path.exists(defaultFileName):
                    currentConfigDict.read(defaultFileName)
                    self.setConfig(currentConfigDict)
            if ('spec' in kw) and ('shm' in kw):
                if len(kw['shm']) >= 8:
                    #if kw['shm'][0:8] in ['MCA_DATA', 'XIA_DATA']:
                    if kw['shm'][0:8] in ['MCA_DATA']:
                        #self.mcawindow.showMaximized()
                        self.toggleSource()
                        self._startupSelection(source=kw['spec'], 
                                                selection=kw['shm'])
                    else:
                        self._startupSelection(source=kw['spec'], 
                                                selection=None)
                else:
                     self._startupSelection(source=kw['spec'], 
                                                selection=None)

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
            
    def currentTabIndexChanged(self, index):
        if QTVERSION < '4.0.0':
            #is not an index but a widget
            index = self.mainTabWidget.indexOf(index)
            legend = "%s" % self.mainTabWidget.label(index)
        else:
            legend = "%s" % self.mainTabWidget.tabText(index)
            for key in self.imageWindowDict.keys():
                if key == legend:value = True
                else: value = False
                self.imageWindowDict[key].setPlotEnabled(value)

    def _is2DSelection(self, ddict):
        if 'imageselection' in ddict:
            if ddict['imageselection']:
                return True
        if 'selection' in ddict:
            if ddict['selection'] is None:
                return False
            if 'selectiontype' in ddict['selection']:
                if ddict['selection']['selectiontype'] == '2D':
                    return True
        return False

    def _is3DSelection(self, ddict):
        if self._is2DSelection(ddict):
            return False

        if 'selection' in ddict:
            if ddict['selection'] is None:
                return False
            if 'selectiontype' in ddict['selection']:
                if ddict['selection']['selectiontype'] == '3D':
                    return True
            
            if 'x' in ddict['selection']:
                if ddict['selection']['x'] is not None:
                    if len(ddict['selection']['x']) > 1:
                        return True           
        return False

    def _isStackSelection(self, ddict):
        if self._is2DSelection(ddict):
            return False
        if self._is3DSelection(ddict):
            return False
        if 'selection' in ddict:
            if ddict['selection'] is None:
                return False
            if 'selectiontype' in ddict['selection']:
                if ddict['selection']['selectiontype'] == 'STACK':
                    return True
        return False


    def dispatcherAddSelectionSlot(self, ddict):
        if self.__useTabWidget:
            if self.mainTabWidget.isHidden():
                #make sure it is visible in case of being closed
                self.mainTabWidget.show()
        if DEBUG:
            return self._dispatcherAddSelectionSlot(ddict)
        try:
            return self._dispatcherAddSelectionSlot(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error: %s" % sys.exc_info()[1])
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.setInformativeText(str(sys.exc_info()[1]))
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()

    def _dispatcherAddSelectionSlot(self, ddict):
        if DEBUG:
            print("self.dispatcherAddSelectionSlot(ddict), ddict = ",ddict)

        toadd = False
        if self._is2DSelection(ddict):
            if QTVERSION < '4.0.0':
                if DEBUG:
                    print("For the time being, no Qt3 support")
                return
            if self.imageWindowCorrelator is None:
                self.imageWindowCorrelator = RGBCorrelator.RGBCorrelator()
                #toadd = True
            title  = "ImageWindow RGB Correlator"
            self.imageWindowCorrelator.setWindowTitle(title)
            legend = ddict['legend']
            if legend not in self.imageWindowDict.keys():
                hkl = False
                try:
                    motor_mne = ddict['dataobject'].info['motor_mne'].split()
                    if ('phi' in motor_mne) and ('chi' in motor_mne):
                        if ('mu'  in motor_mne) and ('gam' in motor_mne) :
                           if 'del' in motor_mne:
                               #SIXC
                               hkl = True
                        if ('tth' in motor_mne) and ('th' in motor_mne):
                            #FOURC
                            hkl = True
                except:
                    pass
                if hkl:
                    imageWindow = PyMcaHKLImageWindow.PyMcaHKLImageWindow(name = legend,
                                correlator = self.imageWindowCorrelator,
                                scanwindow=self.scanwindow)
                else:
                    imageWindow = PyMcaImageWindow.PyMcaImageWindow(name = legend,
                                correlator = self.imageWindowCorrelator,
                                scanwindow=self.scanwindow)
                self.imageWindowDict[legend] = imageWindow
                if QTVERSION > '4.0.0':
                    self.connect(imageWindow, qt.SIGNAL("addImageClicked"),
                         self.imageWindowCorrelator.addImageSlot)
                    self.connect(imageWindow, qt.SIGNAL("removeImageClicked"),
                         self.imageWindowCorrelator.removeImageSlot)
                    self.connect(imageWindow, qt.SIGNAL("replaceImageClicked"),
                         self.imageWindowCorrelator.replaceImageSlot)
                self.mainTabWidget.addTab(imageWindow, legend)
                if toadd:
                    self.mainTabWidget.addTab(self.imageWindowCorrelator,
                        "RGB Correlator")
                self.imageWindowDict[legend].setPlotEnabled(False)
                self.imageWindowDict[legend]._addSelection(ddict)
                if QTVERSION < '4.0.0':
                    self.mainTabWidget.setCurrentPage(self.mainTab.indexOf(imageWindow))
                else:
                    self.mainTabWidget.setCurrentWidget(imageWindow)
                #self.imageWindowDict[legend].setPlotEnabled(True)
                return
            if self.mainTabWidget.indexOf(self.imageWindowDict[legend]) < 0:
                self.mainTabWidget.addTab(self.imageWindowDict[legend],
                                          legend)
                self.imageWindowDict[legend].setPlotEnabled(False)
                self.imageWindowDict[legend]._addSelection(ddict)
                if QTVERSION < '4.0.0':
                    self.mainTabWidget.setCurrentPage(self.mainTab.indexOf\
                                             (self.imageWindowDict[legend]))
                else:
                    self.mainTabWidget.setCurrentWidget(self.imageWindowDict\
                                                        [legend])
            else:
                self.imageWindowDict[legend]._addSelection(ddict)
        elif self._isStackSelection(ddict):
            legend = ddict['legend']
            widget = QStackWidget.QStackWidget()
            widget.notifyCloseEventToWidget(self)            
            widget.setStack(ddict['dataobject'])
            widget.setWindowTitle(legend)
            widget.show()
            self._widgetDict[id(widget)] = widget        
        else:
            if OBJECT3D:
                if ddict['dataobject'].info['selectiontype'] == "1D":
                    self.mcawindow._addSelection(ddict)
                    self.scanwindow._addSelection(ddict)
                else:
                    self.mainTabWidget.setCurrentWidget(self.glWindow)
                    self.glWindow._addSelection(ddict)            
            else:            
                self.mcawindow._addSelection(ddict)
                self.scanwindow._addSelection(ddict)

    def dispatcherRemoveSelectionSlot(self, ddict):
        try:
            return self._dispatcherRemoveSelectionSlot(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error: %s" % sys.exc_info()[1])
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()


    def _dispatcherRemoveSelectionSlot(self, ddict):
        if DEBUG:
            print("self.dispatcherRemoveSelectionSlot(ddict), ddict = ",ddict)

        if self._is2DSelection(ddict):
            if QTVERSION < '4.0.0':
                if DEBUG:
                    print("For the time being, no Qt3 support")
                return
            legend = ddict['legend'] 
            if legend in self.imageWindowDict.keys():
                index = self.mainTabWidget.indexOf(self.imageWindowDict[legend])
                if index >0:
                    self.imageWindowDict[legend].close()
                    self.imageWindowDict[legend].setParent(None)
                    self.mainTabWidget.removeTab(index)
                    self.imageWindowDict[legend]._removeSelection(ddict)
                    del self.imageWindowDict[legend]
        elif self._is3DSelection(ddict):
            self.glWindow._removeSelection(ddict)             
        else:
            self.mcawindow._removeSelection(ddict)
            self.scanwindow._removeSelection(ddict)

    def dispatcherReplaceSelectionSlot(self, ddict):
        try:
            return self._dispatcherReplaceSelectionSlot(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error: %s" % sys.exc_info()[1])
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()

    def _dispatcherReplaceSelectionSlot(self, ddict):
        if DEBUG:
            print("self.dispatcherReplaceSelectionSlot(ddict), ddict = ",ddict)
        if self._is2DSelection(ddict):
            if QTVERSION < '4.0.0':
                if DEBUG:
                    print("For the time being, no Qt3 support")
                return
            legend = ddict['legend']
            for key in list(self.imageWindowDict.keys()):
                index = self.mainTabWidget.indexOf(self.imageWindowDict[key])
                if key == legend:continue
                if index >= 0:
                    self.imageWindowDict[key].close()
                    self.imageWindowDict[key].setParent(None)
                    self.mainTabWidget.removeTab(index)
                    self.imageWindowDict[key]._removeSelection(ddict)
                    del self.imageWindowDict[key]
            if legend in self.imageWindowDict.keys():
                self.imageWindowDict[legend].setPlotEnabled(False)
            self.dispatcherAddSelectionSlot(ddict)
            index = self.mainTabWidget.indexOf(self.imageWindowDict[legend])
            if index != self.mainTabWidget.currentIndex():
                if QTVERSION < '4.0.0':
                    self.mainTabWidget.setCurrentPage(index)
                else:
                    self.mainTabWidget.setCurrentWidget(self.imageWindowDict[legend])
        elif self._is3DSelection(ddict):
            self.glWindow._replaceSelection(ddict)             
        else:
            self.mcawindow._replaceSelection(ddict)
            self.scanwindow._replaceSelection(ddict)

    def dispatcherOtherSignalsSlot(self, ddict):
        if DEBUG:
            print("self.dispatcherOtherSignalsSlot(ddict), ddict = ",ddict)
        if not self.__useTabWidget:return
        if ddict['event'] == "SelectionTypeChanged":
            if QTVERSION < '4.0.0':
                if ddict['SelectionType'].upper() == "COUNTERS":
                    index = self.mainTabWidget.indexOf(self.scanwindow)
                    self.mainTabWidget.setCurrentPage(index)
                    return
                for i in range(self.mainTabWidget.count()):
                    if str(self.mainTabWidget.label(i)) == \
                                       ddict['SelectionType']:                        
                        self.mainTabWidget.setCurrentPage(i)
            else:
                if ddict['SelectionType'].upper() == "COUNTERS":
                    self.mainTabWidget.setCurrentWidget(self.scanwindow)
                    return
                for i in range(self.mainTabWidget.count()):
                    if str(self.mainTabWidget.tabText(i)) == \
                                       ddict['SelectionType']:                        
                        self.mainTabWidget.setCurrentIndex(i)
            return
        if ddict['event'] == "SourceTypeChanged":
            pass
            return
        if DEBUG:
            print("Unhandled dict")
        
    def setConfig(self, configDict):
        if 'PyMca' in configDict:
            self.__configurePyMca(configDict['PyMca'])
        if 'ROI' in configDict:
            self.__configureRoi(configDict['ROI'])
        if 'Elements' in configDict:
            self.__configureElements(configDict['Elements'])
        if 'Fit' in configDict:
            self.__configureFit(configDict['Fit'])
        if 'ScanSimpleFit' in configDict:
            self.__configureScanSimpleFit(configDict['ScanSimpleFit'])
        if 'ScanCustomFit' in configDict:
            self.__configureScanCustomFit(configDict['ScanCustomFit'])
        
    def getConfig(self):
        d = {}
        d['PyMca']    = {}
        d['PyMca']['VERSION']   = __version__
        d['PyMca']['ConfigDir'] = self.configDir
        d['PyMca']['nativeFileDialogs'] = PyMcaDirs.nativeFileDialogs
        
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
        d['PyMca']['Sources']['lastFileFilter'] = self.sourceWidget.sourceSelector.lastFileFilter
        for source in SOURCESLIST:
            d['PyMca'][source] = {}
            if self.sourceWidget.sourceSelector.lastInputDir is not None:
                d['PyMca'][source]['lastInputDir'] = self.sourceWidget.sourceSelector.lastInputDir
                try:
                    PyMcaDirs.inputDir = self.sourceWidget.sourceSelector.lastInputDir
                except ValueError:
                    pass
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
            elif source == "HDF5":
                d['PyMca'][source]['SourceName'] = []
                for key in self.sourceWidget.sourceList:
                    if key.sourceType == source:
                        d['PyMca'][source]['SourceName'].append(key.sourceName)
            selectorWidget = self.sourceWidget.selectorWidget[source]
            if hasattr(selectorWidget,'setWidgetConfiguration'):
                d['PyMca'][source]['WidgetConfiguration'] = selectorWidget.getWidgetConfiguration()                        

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

        #ScanFit related
        d['ScanSimpleFit'] = {}
        d['ScanSimpleFit']['Configuration'] = {}
        if DEBUG:
                  d['ScanSimpleFit']['Configuration'].update(\
                      self.scanwindow.scanFit.getConfiguration())
        else:
            try:
                  d['ScanSimpleFit']['Configuration'].update(\
                      self.scanwindow.scanFit.getConfiguration())
            except:
                print("Error getting ScanFint configuration")
        return d
        
    def saveConfig(self, config, filename = None):
        d = ConfigDict.ConfigDict()
        d.update(config)
        if filename is None:
            filename = self.__getDefaultSettingsFile()
        d.write(filename)

    def __configurePyMca(self, dict):
        if 'ConfigDir' in dict:
            self.configDir = dict['ConfigDir']

        if 'Geometry' in dict:
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
            
        PyMcaDirs.nativeFileDialogs = dict.get('nativeFileDialogs', True)

        if 'Sources' in dict:
            if 'lastFileFilter' in dict['Sources']:
                self.sourceWidget.sourceSelector.lastFileFilter = dict['Sources']['lastFileFilter']
        for source in SOURCESLIST:
            if source in dict:
                if 'lastInputDir' in dict[source]:
                    if dict[source] ['lastInputDir'] != "None":
                        self.sourceWidget.sourceSelector.lastInputDir =  dict[source] ['lastInputDir']
                        try:
                            PyMcaDirs.inputDir = dict[source] ['lastInputDir']
                        except ValueError:
                            pass
                if 'SourceName' in dict[source]:
                    if type(dict[source]['SourceName']) != type([]):
                        dict[source]['SourceName'] = [dict[source]['SourceName'] * 1]
                    for SourceName in dict[source]['SourceName']:
                        if len(SourceName):
                            try:
                                if not os.path.exists(SourceName): continue
                                self.sourceWidget.sourceSelector.openFile(SourceName, justloaded =1)
                                continue
                                #This event is not needed
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

                if 'WidgetConfiguration' in dict[source]:
                    selectorWidget = self.sourceWidget.selectorWidget[source]
                    if hasattr(selectorWidget,'setWidgetConfiguration'):
                        try:
                            selectorWidget.setWidgetConfiguration(dict[source]['WidgetConfiguration'])
                        except:
                            msg = qt.QMessageBox(self)
                            msg.setIcon(qt.QMessageBox.Critical)
                            msg.setText("Error: %s\n configuring %s widget" % (sys.exc_info()[1], source ))
                            if QTVERSION < '4.0.0':
                                msg.exec_loop()
                            else:
                                msg.exec_()
                """
                if 'Selection' in dict[source]:
                    if type(dict[source]['Selection']) != type([]):
                        dict[source]['Selection'] = [dict[source]['Selection']]
                    if source == "EdfFile":
                        self.sourceWidget[source].setSelected(dict[source]['Selection'])
                """    


    def __configureRoi(self, ddict):
        if 'roidict' in ddict:
            if 'roilist' in ddict:
                roilist = ddict['roilist']
                if type(roilist) != type([]):
                    roilist=[roilist]                
                roidict = ddict['roidict']
                self.mcawindow.roiwidget.fillfromroidict(roilist=roilist,
                                                         roidict=roidict)
            

    def __configureElements(self, ddict):
        if 'Material' in ddict:
            ElementsInfo.Elements.Material.update(ddict['Material'])

    def __configureFit(self, d):
        if 'Configuration' in d:
            self.mcawindow.advancedfit.mcafit.configure(d['Configuration'])
            if not self.mcawindow.advancedfit.isHidden():
                self.mcawindow.advancedfit._updateTop()
        if 'ConfigDir' in d:
            self.mcawindow.advancedfit.configDir = d['ConfigDir'] * 1
        if False and ('LastFit' in d):
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
                    print("hidden")
            
    def __configureFit(self, d):
        if 'Configuration' in d:
            self.mcawindow.advancedfit.mcafit.configure(d['Configuration'])
            if not self.mcawindow.advancedfit.isHidden():
                self.mcawindow.advancedfit._updateTop()
        if 'ConfigDir' in d:
            self.mcawindow.advancedfit.configDir = d['ConfigDir'] * 1
        if False and ('LastFit' in d):
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
                    print("hidden")

    def __configureScanCustomFit(self, ddict):
        pass

    def __configureScanSimpleFit(self, ddict):
        if 'Configuration' in ddict:
            self.scanwindow.scanFit.setConfiguration(ddict['Configuration'])
                
    def initMenuBar(self):
        if self.options["MenuFile"]:
            if QTVERSION < '4.0.0':
                self.menuFile= qt.QPopupMenu(self.menuBar())
                idx= self.menuFile.insertItem(self.Icons["fileopen"], QString("&Open"), self.onOpen, qt.Qt.CTRL+qt.Qt.Key_O)
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
                self.actionOpen.setText(QString("&Open"))
                self.actionOpen.setIcon(self.Icons["fileopen"])
                self.actionOpen.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
                self.connect(self.actionOpen, qt.SIGNAL("triggered(bool)"),
                             self.onOpen)
                #filesaveas
                self.actionSaveAs = qt.QAction(self)
                self.actionSaveAs.setText(QString("&Save"))
                self.actionSaveAs.setIcon(self.Icons["filesave"])
                self.actionSaveAs.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSaveAs, qt.SIGNAL("triggered(bool)"),
                             self.onSaveAs)

                #filesave
                self.actionSave = qt.QAction(self)
                self.actionSave.setText(QString("Save &Default Settings"))
                #self.actionSave.setIcon(self.Icons["filesave"])
                #self.actionSave.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
                self.connect(self.actionSave, qt.SIGNAL("triggered(bool)"),
                             self.onSave)
                #fileprint
                self.actionPrint = qt.QAction(self)
                self.actionPrint.setText(QString("&Print"))
                self.actionPrint.setIcon(self.Icons["fileprint"])
                self.actionPrint.setShortcut(qt.Qt.CTRL+qt.Qt.Key_P)
                self.connect(self.actionPrint, qt.SIGNAL("triggered(bool)"),
                             self.onPrint)

                #filequit
                self.actionQuit = qt.QAction(self)
                self.actionQuit.setText(QString("&Quit"))
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
                self.menuHelp.addAction("Changes", self.onChanges)
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
            self.sourceFrame.setWindowTitle("Source Selector")
            self.sourceFrame.setWindowIcon(self.windowIcon())
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
            
    def sourceReparent(self,ddict = None):
        if self.sourceFrame.parent() is not None:
            if QTVERSION < '4.0.0':
                self.sourceFrame.reparent(None,self.cursor().pos(),1)
                self.splitter.moveToFirst(self.sourceFrame)
            else:
                self.sourceFrame.setParent(None)
                self.sourceFrame.move(self.cursor().pos())
                self.sourceFrame.show()
        else:
            if QTVERSION < '4.0.0':
                self.sourceFrame.reparent(self.splitter,qt.QPoint(),1)
                self.splitter.moveToFirst(self.sourceFrame)
            else:
                try:
                    self.splitter.insertWidget(0, self.sourceFrame)
                except:
                    self.sourceFrame.setParent(self.splitter)
            
    def initSource(self):
        self.sourceWidget = QDispatcher.QDispatcher(self.sourceFrame)
        self.sourceFrameLayout.addWidget(self.sourceWidget)

    def _startupSelection(self, source, selection):
        self.sourceWidget.sourceSelector.openSource(source)
        if selection is None:return
        
        if len(selection) >= 8:
            if selection[0:8] == "MCA_DATA":
                ddict= {}
                ddict['event'] = "addSelection"
                ddict['SourceName'] = source
                ddict['Key'] = selection
                ddict["selection"] = {'cols': {'y': [1], 'x': [0]}}
                ddict["legend"] = ddict['SourceName'] + ' %s.c.1' %selection
                ddict["SourceType"] =  'SPS'
                self.sourceWidget._addSelectionSlot([ddict])
                if QTVERSION < '4.0.0':
                    self.mcawindow.control.calbox.setCurrentItem(2)
                else:
                    self.mcawindow.control.calbox.setCurrentIndex(2)
                self.mcawindow.calibration = self.mcawindow.calboxoptions[2]
                self.mcawindow.control._calboxactivated("Internal")
        else:
            return
        """ 
        elif selection == "XIA_DATA":
            ddict= {}
            ddict['event'] = "addSelection"
            ddict['SourceName'] = "armando5"
            ddict['Key'] = selection
            ddict["selection"] = {'rows': {'y': [1], 'x': [0]}}
            ddict["legend"] = ddict['SourceName'] + ' XIA_DATA.c.1'
            ddict["SourceType"] =  'SPS'
            self.sourceWidget._addSelectionSlot([ddict])
        """

    def menuToolsAboutToShow(self):
        if DEBUG:
            print("menu ToolsAboutToShow")
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
            self.menuTools.insertItem("Convert Mca to Edf",self.__mca2EdfConversion)
            #self.menuTools.insertItem("Fit to Specfile",self.__fit2SpecConversion)
            if STACK:
                self.menuTools.insertItem("ROI Imaging",self.__roiImaging)
        else:
            if self.sourceFrame.isHidden():
                self.menuTools.addAction("Show Source",self.toggleSource)
            else:
                self.menuTools.addAction("Hide Source",self.toggleSource)
            self.menuTools.addAction("Elements   Info",self.__elementsInfo)
            self.menuTools.addAction("Material Transmission",self.__attTool)
            self.menuTools.addAction("Identify  Peaks",self.__peakIdentifier)
            self.menuTools.addAction("Batch   Fitting",self.__batchFitting)
            self.menuTools.addAction("Convert Mca to Edf",self.__mca2EdfConversion)
            #self.menuTools.addAction("Fit to Specfile",self.__fit2SpecConversion)
            self.menuTools.addAction("RGB Correlator",self.__rgbCorrelator)
            if STACK:
                self.menuTools.addAction("ROI Imaging",self.__roiImaging)
            if XIA_CORRECT:
                self.menuTools.addAction("XIA Correct",
                                         self.__xiaCorrect)
        if DEBUG:
            print("Fit to Specfile missing")
            
    def fontdialog(self):
        fontd = qt.QFontDialog.getFont(self)
        if fontd[1]:
            qt.qApp.setFont(fontd[0],1)
           

    def toggleSource(self,**kw):
        if DEBUG:
            print("toggleSource called")
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

    def __attTool(self):
        if self.attenuationTool is None:
            self.attenuationTool = MaterialEditor.MaterialEditor(toolmode=True)
        if self.attenuationTool.isHidden():
            self.attenuationTool.show()
        self.attenuationTool.raise_()

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
        if self.__batch is None:
            self.__batch = PyMcaBatch.McaBatchGUI(fl=0,actions=1)
        if self.__batch.isHidden():
            self.__batch.show()
        if QTVERSION < '4.0.0':
            self.__batch.raiseW()
        else:
            self.__batch.raise_()

    def __mca2EdfConversion(self):
        if self.__mca2Edf is None:
            self.__mca2Edf = Mca2Edf.Mca2EdfGUI(fl=0,actions=1)
        if self.__mca2Edf.isHidden():
            self.__mca2Edf.show()
        if QTVERSION < '4.0.0':
            self.__mca2Edf.raiseW()
        else:
            self.__mca2Edf.raise_()

    def __fit2SpecConversion(self):
        if self.__fit2Spec is None:
            self.__fit2Spec = Fit2Spec.Fit2SpecGUI(fl=0,actions=1)
        if self.__fit2Spec.isHidden():
            self.__fit2Spec.show()
        if QTVERSION < '4.0.0':
            self.__fit2Spec.raiseW()
        else:
            self.__fit2Spec.raise_()

    def __rgbCorrelator(self):
        if self.__correlator is None:
            self.__correlator = []
        fileTypeList = ["Batch Result Files (*dat)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "All Files (*)"]
        message = "Open ONE Batch result .dat file or SEVERAL EDF files"
        filelist = self.__getStackOfFiles(fileTypeList, message) 
        if not(len(filelist)):
            return
        filelist.sort()
        self.sourceWidget.sourceSelector.lastInputDir = os.path.dirname(filelist[0])
        PyMcaDirs.inputDir = os.path.dirname(filelist[0])
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

    def __getStackOfFiles(self, typelist, message="", getfilter=False):
        wdir = PyMcaDirs.inputDir
        fileTypeList = typelist
        filterused = None
        if QTVERSION < '4.0.0':
            filetypes = ""
            for filetype in fileTypeList:
                filetypes += filetype+"\n"
            filelist = qt.QFileDialog.getOpenFileNames(filetypes,
                        wdir,
                        self,
                        message,
                        message)
            if not len(filelist):
                if getfilter:
                    return [], filterused
                else:
                    return []
        else:
            if False and PyMcaDirs.nativeFileDialogs:
                #windows cannot handle thousands of files in a file dialog
                filetypes = ""
                for filetype in fileTypeList:
                    filetypes += filetype+"\n"
                filelist = qt.QFileDialog.getOpenFileNames(self,
                            message,
                            wdir,
                            filetypes)
                if not len(filelist):
                    if getfilter:
                        return [], filterused
                    else:
                        return []
                else:
                    sample  = qt.safe_str(filelist[0])
                    for filetype in fileTypeList:
                        ftype = filetype.replace("(", "").replace(")","")
                        extensions = ftype.split()[2:]
                        for extension in extensions:
                            if sample.endswith(extension[-3:]):
                                filterused = filetype
                                break                    
            else:
                fdialog = qt.QFileDialog(self)
                fdialog.setModal(True)
                fdialog.setWindowTitle(message)
                if hasattr(qt, "QStringList"):
                    strlist = qt.QStringList()
                else:
                    strlist = []
                for filetype in fileTypeList:
                    strlist.append(filetype.replace("(","").replace(")",""))
                fdialog.setFilters(strlist)
                fdialog.setFileMode(fdialog.ExistingFiles)
                fdialog.setDirectory(wdir)
                if QTVERSION > '4.3.0':
                    history = fdialog.history()
                    if len(history) > 6:
                        fdialog.setHistory(history[-6:])
                ret = fdialog.exec_()
                if ret == qt.QDialog.Accepted:
                    filelist = fdialog.selectedFiles()
                    if getfilter:
                        filterused = qt.safe_str(fdialog.selectedFilter())
                    fdialog.close()
                    del fdialog                        
                else:
                    fdialog.close()
                    del fdialog
                    if getfilter:
                        return [], filterused
                    else:
                        return []
        filelist = [qt.safe_str(x) for x in filelist]
        if getfilter:
            return filelist, filterused
        else:
            return filelist


    def __roiImaging(self):
        if self.__imagingTool is None:
            rgbWidget = None
            try:
                widget = QStackWidget.QStackWidget(mcawidget=self.mcawindow,
                                                   rgbwidget=rgbWidget,
                                                   master=True)
                widget.notifyCloseEventToWidget(self)
                self.__imagingTool = id(widget)                
                self._widgetDict[self.__imagingTool] = widget
                #w = StackSelector.StackSelector(self)
                #stack = w.getStack()
                widget.loadStack()
                widget.show()
            except IOError:
                widget = None
                del self._widgetDict[self.__imagingTool]
                self.__imagingTool = None
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
            except:
                widget = None
                del self._widgetDict[self.__imagingTool]
                self.__imagingTool = None
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                print("Error info = ",sys.exc_info())
                msg.setText("Unexpected Error: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
        else:
            widget = self._widgetDict[self.__imagingTool]
            if widget.isHidden():
                widget.show()
            widget.raise_()


    def customEvent(self, event):
        if hasattr(event, 'dict'):
            ddict = event.dict
            if 'event' in ddict:
                if ddict['event'] == "closeEventSignal":
                    if ddict['id'] in self._widgetDict:
                        if ddict['id'] == self.__imagingTool:
                            self.__imagingTool = None
                        del self._widgetDict[ddict['id']]

    def __xiaCorrect(self):
        XiaCorrect.mainGUI(qt.qApp)
    
    def onOpen(self):
        if QTVERSION < '4.0.0':
            self.openMenu.exec_loop(self.cursor().pos())
        else:
            self.openMenu.exec_(self.cursor().pos())

    def onSave(self):
        self._saveAs()

    def onSaveAs(self):
        if QTVERSION < '4.0.0':
            return self._onSaveAs()
        index = self.mainTabWidget.currentIndex()
        text  = str(self.mainTabWidget.tabText(index))
        self.saveMenu = qt.QMenu()
        self.saveMenu.addAction("PyMca Configuration", self._onSaveAs)
        if text.upper() == 'MCA':
            self.saveMenu.addAction("Active Mca",
                             self.mcawindow._saveIconSignal)
        elif text.upper() == 'SCAN':
            self.saveMenu.addAction("Active Scan",
                             self.scanwindow._saveIconSignal)
        elif text in self.imageWindowDict.keys():
            self.saveMenu.addAction("Active Image",
                  self.imageWindowDict[text].graphWidget._saveIconSignal)
        self.saveMenu.exec_(self.cursor().pos())

    def _onSaveAs(self):
        cwd = os.getcwd()
        if QTVERSION < '4.0.0':
            outfile = qt.QFileDialog(self,"Output File Selection",1)
            outfile.setFilters('PyMca  *.ini')
            outfile.setMode(outfile.AnyFile)
        else:
            outfile = qt.QFileDialog(self)
            outfile.setFilter('PyMca  *.ini')
            outfile.setFileMode(outfile.AnyFile)
            outfile.setAcceptMode(qt.QFileDialog.AcceptSave)

        if os.path.exists(self.configDir):cwd =self.configDir 
        if QTVERSION < '4.0.0': outfile.setDir(cwd)
        else: outfile.setDirectory(cwd)
        if QTVERSION < '4.0.0':ret = outfile.exec_loop()
        else:ret = outfile.exec_()
        if ret:
            filterused = qt.safe_str(outfile.selectedFilter()).split()
            extension = ".ini"
            if QTVERSION < '4.0.0':
                outdir=qt.safe_str(outfile.selectedFile())
            else:
                outdir=qt.safe_str(outfile.selectedFiles()[0])
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
                directory = os.path.join(home, "My Documents")
            except:
                home = '\\'
                directory = '\\'
            #print home
            #print directory
            if os.path.isdir('%s' % directory):
                directory = os.path.join(directory, "PyMca")
            else:
                #print "My Documents is not there"
                directory = os.path.join(home, "PyMca")
            if not os.path.exists('%s' % directory):
                #print "PyMca directory not present"
                os.mkdir('%s' % directory)
            #print filename
            finalfile = os.path.join(directory, filename)
            #print finalfile
        else:
            home = os.getenv('HOME')
            directory = os.path.join(home, "PyMca")
            if not os.path.exists('%s' % directory):
                os.mkdir('%s' % directory)
            finalfile =  os.path.join(directory, filename)
        return finalfile

    def loadTrainingData(self):
        try:
            source = os.path.join(PyMcaDataDir.PYMCA_DATA_DIR,
                                    'XRFSpectrum.mca')
            self.sourceWidget.sourceSelector.openSource(source)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error opening data source")
            msg.setText("Cannot open data source %s" % source)
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()

    def openSource(self,index=0):
        if DEBUG:
            print("index = %d " % index)
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
                if QTVERSION < '4.0.0':
                    filename = qt.safe_str(outfile.selectedFile())
                else:
                    filename = qt.safe_str(outfile.selectedFiles()[0])
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
                self.menuBrowser.setCaption(QString("Main Menu Help"))
            else:
                self.menuBrowser.setWindowTitle(QString("Main Menu Help"))
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(os.path.join(ddir,"HTML","Menu.html")):
                ddir = os.path.dirname(ddir)
            if QTVERSION < '4.0.0':
                self.menuBrowser.mimeSourceFactory().addFilePath(QString(ddir+"/HTML"))
                self.menuBrowser.setSource(QString("Menu.html"))
            else:
                self.menuBrowser.setSearchPaths([os.path.join(ddir,"HTML")])
                self.menuBrowser.setSource(qt.QUrl(QString("Menu.html")))
            self.menuBrowser.show()
        if self.menuBrowser.isHidden():
            self.menuBrowser.show()
        if QTVERSION < '4.0.0':
            self.menuBrowser.raiseW()
        else:
            self.menuBrowser.raise_()

    def onDisplayHowto(self):
        if self.displayBrowser is None:
            self.displayBrowser= qt.QTextBrowser()
            if QTVERSION < '4.0.0':
                self.displayBrowser.setCaption(QString("Data Display HOWTO"))
            else:
                self.displayBrowser.setWindowTitle(QString("Data Display HOWTO"))                
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(os.path.join(ddir,"HTML","Display-HOWTO.html")):
                ddir = os.path.dirname(ddir)
            if QTVERSION < '4.0.0':
                self.displayBrowser.mimeSourceFactory().addFilePath(QString(ddir+"/HTML"))
                self.displayBrowser.setSource(QString("Display-HOWTO.html"))
            else:
                self.displayBrowser.setSearchPaths([os.path.join(ddir,"HTML")])
                self.displayBrowser.setSource(qt.QUrl(QString("Display-HOWTO.html")))
            self.displayBrowser.show()
        if self.displayBrowser.isHidden():
            self.displayBrowser.show()
        if QTVERSION < '4.0.0':
            self.displayBrowser.raiseW()
        else:
            self.displayBrowser.raise_()
    
    def onMcaHowto(self):
        if self.mcaBrowser is None:
            self.mcaBrowser= MyQTextBrowser()
            if QTVERSION < '4.0.0':
                self.mcaBrowser.setCaption(QString("MCA HOWTO"))
            else:
                self.mcaBrowser.setWindowTitle(QString("MCA HOWTO"))
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(ddir+"/HTML"+"/MCA-HOWTO.html"):
                ddir = os.path.dirname(ddir)
            if QTVERSION < '4.0.0':
                self.mcaBrowser.mimeSourceFactory().addFilePath(QString(ddir+"/HTML"))
                self.mcaBrowser.setSource(QString("MCA-HOWTO.html"))
            else:
                self.mcaBrowser.setSearchPaths([os.path.join(ddir,"HTML")])
                self.mcaBrowser.setSource(qt.QUrl(QString("MCA-HOWTO.html")))
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

    def onChanges(self):
        if self.changeLog is None:
            self.changeLog = qt.QTextEdit()
            self.changeLog.setCursor(self.cursor())
            self.changeLog.setWindowTitle("PyMCA %s Changes" % __version__)
            self.changeLog.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
            # Does this belong to the data dir or the doc dir?
            mpath = PyMcaDataDir.PYMCA_DATA_DIR
            fname = os.path.join(mpath,'changelog.txt')
            if not os.path.exists(fname):
               while len(mpath) > 3:
                 fname = os.path.join(mpath,'changelog.txt')
                 #print "looking for ", fname
                 if not os.path.exists(fname):
                     mpath = os.path.dirname(mpath)
                 else:
                     break
            if os.path.exists(fname):
                self.log = ChangeLog.ChangeLog(textfile='changelog.txt')
                self.changeLog.setDocument(self.log)
                self.changeLog.setMinimumWidth(500)
            else:
                self.log = ChangeLog.ChangeLog()
                self.log.setPlainText('Cannot find file changelog.txt')
        self.changeLog.show()
    
    def onDebug(self):
        print("Module name PyMca ",__revision__.replace("$",""))
        print("Module name PyQt  ",qt.PYQT_VERSION_STR)
        for module in sys.modules.values():
            try:
                if 'Revision' in module.__revision__:
                    if module.__name__ != "__main__":
                        print("Module name = ",module.__name__,module.__revision__.replace("$",""))
            except:
                pass 
    
    def onPrint(self):
        if DEBUG:
            print("onPrint called")
        if self.scanwindow.hasFocus():
            self.scanwindow.graph.printps() 
        else:
            self.mcawindow.show()
            if QTVERSION < '4.0.0':self.mcawindow.raiseW()
            else:self.mcawindow.raise_()
            self.mcawindow.graph.printps()    

    def __McaWindowSignal(self, ddict):
        if ddict['event'] == 'NewScanCurve':
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
                if qt.safe_str(graph.caption())== name:
                    return graph
            return None

    def closeGraph(self, name):
            """
            Called after a graph is closed
            """
            print("closeGraph", name)

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
        if name == QString("./PyMCA.html"):
            if sys.platform == 'win32':
                ddir=PyMcaDataDir.PYMCA_DOC_DIR
                if not os.path.exists(ddir+"/HTML"+"/PyMCA.html"):
                    ddir = os.path.dirname(ddir)
                cmd = ddir+"/HTML/PyMCA.pdf"
                os.system('"%s"' % cmd)
                return
            try:
                self.report.show()
            except:
                self.report = qt.QTextBrowser()
                self.report.setCaption(QString("PyMca Report"))
                ddir=PyMcaDataDir.PYMCA_DOC_DIR
                self.report.mimeSourceFactory().addFilePath(QString(ddir+"/HTML"))
                self.report.mimeSourceFactory().addFilePath(QString(ddir+"/HTML/PyMCA_files"))
                self.report.setSource(name)
            if self.report.isHidden():self.report.show()
            self.report.raiseW()
        else:
            qt.QTextBrowser.setSource(self,name)                           
            
class Line(qt.QFrame):
    def mouseDoubleClickEvent(self,event):
        if DEBUG:
            print("Double Click Event")
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
            print("Mouse Press Event")
        ddict={}
        ddict['event']="MousePress"
        ddict['data'] = event
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("PixmapLabelMousePressEvent"),(ddict,))
        else:
            self.emit(qt.SIGNAL("PixmapLabelMousePressEvent"),(ddict))

if __name__ == '__main__':
    PROFILING = 0
    if PROFILING:
        import profile
        import pstats
    PyMcaMainWidgetInstance = PyMcaMain(**keywords)
    if nativeFileDialogs is not None:
        PyMcaDirs.nativeFileDialogs = nativeFileDialogs
    if debugreport:PyMcaMainWidgetInstance.onDebug()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
            
    if QTVERSION < '4.0.0':
        app.setMainWidget(PyMcaMainWidgetInstance)
        PyMcaMainWidgetInstance.show()
        # --- close waiting widget
        splash.close()
        if PROFILING:
            profile.run('sys.exit(app.exec_loop())',"test")
            p=pstats.Stats("test")
            p.strip_dirs().sort_stats(-1).print_stats()
        else:
            app.exec_loop()
    else:
        splash.finish(PyMcaMainWidgetInstance)
        PyMcaMainWidgetInstance.show()
        #try to interpret rest of command line arguments as data sources
        try:
            for source in args:
                PyMcaMainWidgetInstance.sourceWidget.sourceSelector.openSource(source)
        except:
            msg = qt.QMessageBox(PyMcaMainWidgetInstance)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Error opening data source")
            msg.setText("Cannot open data source %s" % source)
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
                    
        if PROFILING:
            profile.run('sys.exit(app.exec_())',"test")
            p=pstats.Stats("test")
            p.strip_dirs().sort_stats(-1).print_stats()
        else:
            sys.exit(app.exec_())
        
