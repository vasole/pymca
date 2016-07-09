#!/usr/bin/env python
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
import sys, getopt
import traceback
if sys.platform == 'win32':
    import ctypes
    from ctypes.wintypes import MAX_PATH
nativeFileDialogs = None
DEBUG = 0
backend=None
if __name__ == '__main__':
    options     = '-f'
    longoptions = ['spec=',
                   'shm=',
                   'debug=',
                   'qt=',
                   'backend=',
                   'nativefiledialogs=',
                   'PySide=']
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
        elif opt in ('--backend'):
            backend = arg
        elif opt in ('--nativefiledialogs'):
            if int(arg):
                nativeFileDialogs = True
            else:
                nativeFileDialogs = False
        elif opt in ('--PySide'):
            import PySide
    if qtversion == '3':
        raise NotImplementedError("Qt3 is not longer supported")

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.pymca import PyMcaMdi
IconDict = PyMcaMdi.IconDict
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str

QTVERSION = qt.qVersion()

try:
    from PyMca5.PyMcaGui.physics.xrf import XRFMCPyMca
    XRFMC_FLAG = True
except:
    XRFMC_FLAG = False

try:
    from PyMca5.PyMcaGui.pymca import SumRulesTool
    SUMRULES_FLAG = True
except:
    SUMRULES_FLAG = False

import PyMca5
from PyMca5.PyMcaGui.pymca.PyMca_help import HelpDict
from PyMca5 import PyMcaDataDir
import os
__version__ = PyMca5.version()

if __name__ == "__main__":
    sys.excepthook = qt.exceptionHandler

    app = qt.QApplication(sys.argv)
    strlist = qt.QStyleFactory.keys()
    if sys.platform == "win32":
        for item in strlist:
            text = str(item)
            if text == "WindowsXP":
                style = qt.QStyleFactory.create(item)
                app.setStyle(style)
                break

    if sys.platform not in ["win32", "darwin"]:
        # some themes of Ubuntu 16.04 give black tool tips on black background
        app.setStyleSheet("QToolTip { color: #000000; background-color: #fff0cd; border: 1px solid black; }")

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
    splash.raise_()
    from PyMca5.PyMcaGui.pymca import ChangeLog
    font = splash.font()
    font.setBold(1)
    splash.setFont(font)
    splash.showMessage( 'PyMca %s' % __version__,
            qt.Qt.AlignLeft|qt.Qt.AlignBottom,
            qt.Qt.white)
    if sys.platform == "darwin":
        qApp = qt.QApplication.instance()
        qApp.processEvents()

from PyMca5.PyMcaGraph.Plot import Plot
from PyMca5.PyMcaGui.pymca import ScanWindow
from PyMca5.PyMcaGui.pymca import McaWindow

from PyMca5.PyMcaGui.pymca import PyMcaImageWindow
from PyMca5.PyMcaGui.pymca import PyMcaHKLImageWindow
try:
    #This is to make sure it is properly frozen
    #and that Object3D is fully supported
    import OpenGL.GL
    #import Object3D.SceneGLWindow as SceneGLWindow
    import PyMca5.PyMcaGui.pymca.PyMcaGLWindow as SceneGLWindow
    OBJECT3D = False
    if ("PyQt4.QtOpenGL" in sys.modules) or \
       ("PySide.QtOpenGL") in sys.modules or \
       ("PyQt5.QtOpenGL") in sys.modules:
        OBJECT3D = True
except:
    OBJECT3D = False
from PyMca5.PyMcaGui.pymca import QDispatcher
from PyMca5.PyMcaGui import ElementsInfo
from PyMca5.PyMcaGui import PeakIdentifier
from PyMca5.PyMcaGui.pymca import PyMcaBatch
###########import Fit2Spec
from PyMca5.PyMcaGui.pymca import Mca2Edf
try:
    from PyMca5.PyMcaGui.pymca import QStackWidget
    from PyMca5.PyMcaGui.pymca import StackSelector
    STACK = True
except:
    STACK = False
from PyMca5.PyMcaGui.pymca import PyMcaPostBatch
from PyMca5.PyMcaGui import RGBCorrelator
from PyMca5.PyMcaGui import MaterialEditor

from PyMca5.PyMcaIO import ConfigDict
from PyMca5 import PyMcaDirs

XIA_CORRECT = False
if QTVERSION > '4.3.0':
    try:
        from PyMca5.PyMcaCore import XiaCorrect
        XIA_CORRECT = True
    except:
        pass

SOURCESLIST = QDispatcher.QDataSource.source_types.keys()

class PyMcaMain(PyMcaMdi.PyMcaMdi):
    def __init__(self, parent=None, name="PyMca", fl=None,**kw):
            if fl is None:
                fl = qt.Qt.WA_DeleteOnClose
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
            self._xrfmcTool = None
            self._sumRulesTool = None
            self.openMenu = qt.QMenu()
            self.openMenu.addAction("PyMca Configuration", self.openSource)
            self.openMenu.addAction("Data Source",
                         self.sourceWidget.sourceSelector._openFileSlot)
            self.openMenu.addAction("Load Training Data",
                                        self.loadTrainingData)


            self.__useTabWidget = True

            if not self.__useTabWidget:
                self.mcaWindow = McaWindow.McaWidget(self.mdi)
                self.scanWindow = ScanWindow.ScanWindow(self.mdi)
                self.imageWindowDict = None
                self.connectDispatcher(self.mcaWindow, self.sourceWidget)
                self.connectDispatcher(self.scanWindow, self.sourceWidget)
                self.mdi.addWindow(self.mcaWindow)
                self.mdi.addWindow(self.scanWindow)
            else:
                if backend is not None:
                    Plot.defaultBackend = backend
                self.mainTabWidget = qt.QTabWidget(self.mdi)
                self.mainTabWidget.setWindowTitle("Main Window")
                self.mcaWindow = McaWindow.McaWindow(backend=backend)
                self.scanWindow = ScanWindow.ScanWindow(info=True,
                                                        backend=backend)
                self.scanWindow._togglePointsSignal()
                if OBJECT3D:
                    self.glWindow = SceneGLWindow.SceneGLWindow()
                self.mainTabWidget.addTab(self.mcaWindow, "MCA")
                self.mainTabWidget.addTab(self.scanWindow, "SCAN")
                if OBJECT3D:
                    self.mainTabWidget.addTab(self.glWindow, "OpenGL")
                if QTVERSION < '5.0.0':
                    self.mdi.addWindow(self.mainTabWidget)
                else:
                    self.mdi.addSubWindow(self.mainTabWidget)
                #print "Markus patch"
                #self.mainTabWidget.show()
                #print "end Markus patch"
                self.mainTabWidget.showMaximized()
                if False:
                    self.connectDispatcher(self.mcaWindow, self.sourceWidget)
                    self.connectDispatcher(self.scanWindow, self.sourceWidget)
                else:
                    self.imageWindowDict = {}
                    self.imageWindowCorrelator = None
                    self.sourceWidget.sigAddSelection.connect( \
                             self.dispatcherAddSelectionSlot)
                    self.sourceWidget.sigRemoveSelection.connect( \
                             self.dispatcherRemoveSelectionSlot)
                    self.sourceWidget.sigReplaceSelection.connect( \
                             self.dispatcherReplaceSelectionSlot)
                    self.mainTabWidget.currentChanged[int].connect( \
                        self.currentTabIndexChanged)

            self.sourceWidget.sigOtherSignals.connect( \
                         self.dispatcherOtherSignalsSlot)
            if 0:
                if 'shm' in kw:
                    if len(kw['shm']) >= 8:
                        if kw['shm'][0:8] in ['MCA_DATA', 'XIA_DATA']:
                            self.mcaWindow.showMaximized()
                            self.toggleSource()
                else:
                    self.mcaWindow.showMaximized()
            currentConfigDict = ConfigDict.ConfigDict()
            try:
                defaultFileName = self.__getDefaultSettingsFile()
                self.configDir  = os.path.dirname(defaultFileName)
            except:
                if not ('fresh' in kw):
                    raise
            if not ('fresh' in kw):
                if os.path.exists(defaultFileName):
                    currentConfigDict.read(defaultFileName)
                    self.setConfig(currentConfigDict)
            if ('spec' in kw) and ('shm' in kw):
                if len(kw['shm']) >= 8:
                    #if kw['shm'][0:8] in ['MCA_DATA', 'XIA_DATA']:
                    if kw['shm'][0:8] in ['MCA_DATA']:
                        #self.mcaWindow.showMaximized()
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
        if dispatcher is None:
            dispatcher = self.sourceWidget
        dispatcher.sigAddSelection.connect(viewer._addSelection)
        dispatcher.sigRemoveSelection.connect(viewer._removeSelection)
        dispatcher.sigReplaceSelection.connect(viewer._replaceSelection)

    def currentTabIndexChanged(self, index):
        legend = "%s" % self.mainTabWidget.tabText(index)
        for key in self.imageWindowDict.keys():
            if key == legend:
                value = True
            else:
                value = False
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
            self._dispatcherAddSelectionSlot(ddict)
        else:
            try:
                self._dispatcherAddSelectionSlot(ddict)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error: %s" % sys.exc_info()[1])
                msg.setInformativeText(str(sys.exc_info()[1]))
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()

    def _dispatcherAddSelectionSlot(self, dictOrList):
        if DEBUG:
            print("self._dispatcherAddSelectionSlot(ddict), ddict = ", dictOrList)
        if type(dictOrList) == type([]):
            ddict = dictOrList[0]
        else:
            ddict = dictOrList

        toadd = False
        if self._is2DSelection(ddict):
            if DEBUG:
                print("2D selection")
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
                                scanwindow=self.scanWindow)
                else:
                    imageWindow = PyMcaImageWindow.PyMcaImageWindow(name = legend,
                                correlator = self.imageWindowCorrelator,
                                scanwindow=self.scanWindow)
                self.imageWindowDict[legend] = imageWindow

                imageWindow.sigAddImageClicked.connect( \
                     self.imageWindowCorrelator.addImageSlot)
                imageWindow.sigRemoveImageClicked.connect( \
                     self.imageWindowCorrelator.removeImageSlot)
                imageWindow.sigReplaceImageClicked.connect( \
                     self.imageWindowCorrelator.replaceImageSlot)
                self.mainTabWidget.addTab(imageWindow, legend)
                if toadd:
                    self.mainTabWidget.addTab(self.imageWindowCorrelator,
                        "RGB Correlator")
                self.imageWindowDict[legend].setPlotEnabled(False)
                self.imageWindowDict[legend]._addSelection(ddict)
                self.mainTabWidget.setCurrentWidget(imageWindow)
                #self.imageWindowDict[legend].setPlotEnabled(True)
                return
            if self.mainTabWidget.indexOf(self.imageWindowDict[legend]) < 0:
                self.mainTabWidget.addTab(self.imageWindowDict[legend],
                                          legend)
                self.imageWindowDict[legend].setPlotEnabled(False)
                self.imageWindowDict[legend]._addSelection(ddict)
                self.mainTabWidget.setCurrentWidget(self.imageWindowDict\
                                                        [legend])
            else:
                self.imageWindowDict[legend]._addSelection(ddict)
        elif self._isStackSelection(ddict):
            if DEBUG:
                print("Stack selection")
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
                    if DEBUG:
                        print("1D selection")
                    self.mcaWindow._addSelection(dictOrList)
                    self.scanWindow._addSelection(dictOrList)
                else:
                    if DEBUG:
                        print("3D selection")
                    self.mainTabWidget.setCurrentWidget(self.glWindow)
                    self.glWindow._addSelection(dictOrList)
            else:
                if DEBUG:
                    print("1D selection")
                self.mcaWindow._addSelection(dictOrList)
                self.scanWindow._addSelection(dictOrList)

    def dispatcherRemoveSelectionSlot(self, ddict):
        try:
            return self._dispatcherRemoveSelectionSlot(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error: %s" % sys.exc_info()[1])
            msg.exec_()


    def _dispatcherRemoveSelectionSlot(self, dictOrList):
        if DEBUG:
            print("self.dispatcherRemoveSelectionSlot(ddict), ddict = ",ddict)
        if type(dictOrList) == type([]):
            ddict = dictOrList[0]
        else:
            ddict = dictOrList
        if self._is2DSelection(ddict):
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
            self.glWindow._removeSelection(dictOrList)
        else:
            self.mcaWindow._removeSelection(dictOrList)
            self.scanWindow._removeSelection(dictOrList)

    def dispatcherReplaceSelectionSlot(self, ddict):
        try:
            return self._dispatcherReplaceSelectionSlot(ddict)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error: %s" % sys.exc_info()[1])
            msg.exec_()

    def _dispatcherReplaceSelectionSlot(self, dictOrList):
        if DEBUG:
            print("self.dispatcherReplaceSelectionSlot(ddict), ddict = ", dictOrList)
        if type(dictOrList) == type([]):
            ddict = dictOrList[0]
        else:
            ddict = dictOrList
        if self._is2DSelection(ddict):
            legend = ddict['legend']
            for key in list(self.imageWindowDict.keys()):
                index = self.mainTabWidget.indexOf(self.imageWindowDict[key])
                if key == legend:
                    continue
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
                self.mainTabWidget.setCurrentWidget(self.imageWindowDict[legend])
            if legend in self.imageWindowDict.keys():
                # force an update
                self.imageWindowDict[legend].setPlotEnabled(True)
                self.imageWindowDict[legend].setPlotEnabled(False)
        elif self._is3DSelection(ddict):
            self.glWindow._replaceSelection(dictOrList)
        else:
            self.mcaWindow._replaceSelection(dictOrList)
            self.scanWindow._replaceSelection(dictOrList)

    def dispatcherOtherSignalsSlot(self, dictOrList):
        if DEBUG:
            print("self.dispatcherOtherSignalsSlot(ddict), ddict = ",dictOrList)
        if type(dictOrList) == type([]):
            ddict = dictOrList[0]
        else:
            ddict = dictOrList
        if not self.__useTabWidget:
            return
        if ddict['event'] == "SelectionTypeChanged":
            if ddict['SelectionType'].upper() == "COUNTERS":
                self.mainTabWidget.setCurrentWidget(self.scanWindow)
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
        r = self.mcaWindow.geometry()
        d['PyMca']['Geometry']['McaWindow'] = [r.x(), r.y(), r.width(), r.height()]

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

        # McaWindow calibrations
        d["PyMca"]["McaWindow"] = {}
        d["PyMca"]["McaWindow"]["calibrations"] = self.mcaWindow.getCalibrations()

        #ROIs
        d['ROI']={}
        if self.mcaWindow.roiWidget is None:
            roilist = []
            roidict = {}
        else:
            roilist, roidict = self.mcaWindow.roiWidget.getROIListAndDict()
        d['ROI']['roilist'] = roilist
        d['ROI']['roidict'] = {}
        d['ROI']['roidict'].update(roidict)

        #fit related
        d['Elements'] = {}
        d['Elements']['Material'] = {}
        d['Elements']['Material'].update(ElementsInfo.Elements.Material)
        d['Fit'] = {}
        if self.mcaWindow.advancedfit.configDir is not None:
            d['Fit'] ['ConfigDir'] = self.mcaWindow.advancedfit.configDir * 1
        d['Fit'] ['Configuration'] = {}
        d['Fit'] ['Configuration'].update(self.mcaWindow.advancedfit.mcafit.configure())
        d['Fit'] ['Information'] = {}
        d['Fit'] ['Information'].update(self.mcaWindow.advancedfit.info)
        d['Fit'] ['LastFit'] = {}
        d['Fit'] ['LastFit']['hidden'] = self.mcaWindow.advancedfit.isHidden()
        d['Fit'] ['LastFit']['xdata0'] = self.mcaWindow.advancedfit.mcafit.xdata0
        d['Fit'] ['LastFit']['ydata0'] = self.mcaWindow.advancedfit.mcafit.ydata0
        d['Fit'] ['LastFit']['sigmay0']= self.mcaWindow.advancedfit.mcafit.sigmay0
        d['Fit'] ['LastFit']['fitdone']= self.mcaWindow.advancedfit._fitdone()
        #d['Fit'] ['LastFit']['fitdone']= 1
        #d['Fit'] ['LastFit']['xmin'] = self.mcaWindow.advancedfit.mcafit.sigma0
        #d['Fit'] ['LastFit']['xmax'] = self.mcaWindow.advancedfit.mcafit.sigma0

        #ScanFit related
        d['ScanSimpleFit'] = {}
        d['ScanSimpleFit']['Configuration'] = {}
        if DEBUG:
                  d['ScanSimpleFit']['Configuration'].update(\
                      self.scanWindow.scanFit.getConfiguration())
        else:
            try:
                  d['ScanSimpleFit']['Configuration'].update(\
                      self.scanWindow.scanFit.getConfiguration())
            except:
                print("Error getting ScanFint configuration")
        return d

    def saveConfig(self, config, filename = None):
        d = ConfigDict.ConfigDict()
        d.update(config)
        if filename is None:
            filename = self.__getDefaultSettingsFile()
        d.write(filename)

    def __configurePyMca(self, ddict):
        savedVersion = ddict.get("VERSION", '4.7.2')
        if 'ConfigDir' in ddict:
            self.configDir = ddict['ConfigDir']

        if 'Geometry' in ddict:
            r = qt.QRect(*ddict['Geometry']['MainWindow'])
            self.setGeometry(r)
            key = 'Splitter'
            if key in ddict['Geometry'].keys():
                self.splitter.setSizes(ddict['Geometry'][key])
            if hasattr(self.mcaWindow, "graph"):
                # this was the way of working of 4.x.x versions
                key = 'McaWindow'
                if key in ddict['Geometry'].keys():
                    r = qt.QRect(*ddict['Geometry']['McaWindow'])
                    self.mcaWindow.setGeometry(r)
                key = 'McaGraph'
                if key in ddict['Geometry'].keys():
                    r = qt.QRect(*ddict['Geometry']['McaGraph'])
                    self.mcaWindow.graph.setGeometry(r)
                self.show()
            qApp = qt.QApplication.instance()
            qApp.processEvents()
            qApp.postEvent(self, qt.QResizeEvent(qt.QSize(ddict['Geometry']['MainWindow'][2]+1,
                                                          ddict['Geometry']['MainWindow'][3]+1),
                                                 qt.QSize(ddict['Geometry']['MainWindow'][2],
                                                          ddict['Geometry']['MainWindow'][3])))
            self.mcaWindow.showMaximized()

        native = ddict.get('nativeFileDialogs', True)
        if native in ["False", "0", 0]:
            native = False
        PyMcaDirs.nativeFileDialogs = native

        if 'Sources' in ddict:
            if 'lastFileFilter' in ddict['Sources']:
                self.sourceWidget.sourceSelector.lastFileFilter = ddict['Sources']['lastFileFilter']
        for source in SOURCESLIST:
            if source in ddict:
                if 'lastInputDir' in ddict[source]:
                    if ddict[source] ['lastInputDir'] != "None":
                        self.sourceWidget.sourceSelector.lastInputDir =  ddict[source] ['lastInputDir']
                        try:
                            PyMcaDirs.inputDir = ddict[source] ['lastInputDir']
                        except ValueError:
                            pass
                if 'SourceName' in ddict[source]:
                    if type(ddict[source]['SourceName']) != type([]):
                        ddict[source]['SourceName'] = [ddict[source]['SourceName'] * 1]
                    for SourceName0 in ddict[source]['SourceName']:
                        if type(SourceName0) == type([]):
                            SourceName = SourceName0[0]
                        else:
                            SourceName = SourceName0
                        if len(SourceName):
                            try:
                                if not os.path.exists(SourceName):
                                    continue
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
                                txt = "Error: %s\n opening file %s" % (sys.exc_info()[1],SourceName )
                                msg.setInformativeText(txt)
                                msg.setDetailedText(traceback.format_exc())
                                msg.exec_()

                if 'WidgetConfiguration' in ddict[source]:
                    selectorWidget = self.sourceWidget.selectorWidget[source]
                    if hasattr(selectorWidget,'setWidgetConfiguration'):
                        try:
                            selectorWidget.setWidgetConfiguration(ddict[source]['WidgetConfiguration'])
                        except:
                            msg = qt.QMessageBox(self)
                            msg.setIcon(qt.QMessageBox.Critical)
                            txt = "Error: %s\n configuring %s widget" % (sys.exc_info()[1], source )
                            msg.setInformativeText(txt)
                            msg.setDetailedText(traceback.format_exc())
                            msg.exec_()
        if "McaWindow" in ddict:
            self.mcaWindow.setCalibrations(ddict["McaWindow"]["calibrations"])

    def __configureRoi(self, ddict):
        if 'roidict' in ddict:
            if 'roilist' in ddict:
                roilist = ddict['roilist']
                if type(roilist) != type([]):
                    roilist=[roilist]
                roidict = ddict['roidict']
                if self.mcaWindow.roiWidget is None:
                    self.mcaWindow.showRoiWidget(qt.Qt.BottomDockWidgetArea)
                self.mcaWindow.roiWidget.fillFromROIDict(roilist=roilist,
                                                         roidict=roidict)

    def __configureElements(self, ddict):
        if 'Material' in ddict:
            ElementsInfo.Elements.Material.update(ddict['Material'])

    def __configureFit(self, d):
        if 'Configuration' in d:
            self.mcaWindow.advancedfit.configure(d['Configuration'])
            if not self.mcaWindow.advancedfit.isHidden():
                self.mcaWindow.advancedfit._updateTop()
        if 'ConfigDir' in d:
            self.mcaWindow.advancedfit.configDir = d['ConfigDir'] * 1
        if False and ('LastFit' in d):
            if (d['LastFit']['ydata0'] != None) and \
               (d['LastFit']['ydata0'] != 'None'):
                self.mcaWindow.advancedfit.setdata(x=d['LastFit']['xdata0'],
                                                   y=d['LastFit']['ydata0'],
                                              sigmay=d['LastFit']['sigmay0'],
                                              **d['Information'])
                if d['LastFit']['hidden'] == 'False':
                    self.mcaWindow.advancedfit.show()
                    self.mcaWindow.advancedfit.raiseW()
                    if d['LastFit']['fitdone']:
                        try:
                            self.mcaWindow.advancedfit.fit()
                        except:
                            pass
                else:
                    print("hidden")

    def __configureFit(self, d):
        if 'Configuration' in d:
            self.mcaWindow.advancedfit.mcafit.configure(d['Configuration'])
            if not self.mcaWindow.advancedfit.isHidden():
                self.mcaWindow.advancedfit._updateTop()
        if 'ConfigDir' in d:
            self.mcaWindow.advancedfit.configDir = d['ConfigDir'] * 1
        if False and ('LastFit' in d):
            if (d['LastFit']['ydata0'] != None) and \
               (d['LastFit']['ydata0'] != 'None'):
                self.mcaWindow.advancedfit.setdata(x=d['LastFit']['xdata0'],
                                                   y=d['LastFit']['ydata0'],
                                              sigmay=d['LastFit']['sigmay0'],
                                              **d['Information'])
                if d['LastFit']['hidden'] == 'False':
                    self.mcaWindow.advancedfit.show()
                    self.mcaWindow.advancedfit.raiseW()
                    if d['LastFit']['fitdone']:
                        try:
                            self.mcaWindow.advancedfit.fit()
                        except:
                            pass
                else:
                    print("hidden")

    def __configureScanCustomFit(self, ddict):
        pass

    def __configureScanSimpleFit(self, ddict):
        if 'Configuration' in ddict:
            self.scanWindow.scanFit.setConfiguration(ddict['Configuration'])

    def initMenuBar(self):
        if self.options["MenuFile"]:
            #build the actions
            #fileopen
            self.actionOpen = qt.QAction(self)
            self.actionOpen.setText(QString("&Open"))
            self.actionOpen.setIcon(self.Icons["fileopen"])
            self.actionOpen.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
            self.actionOpen.triggered[bool].connect(self.onOpen)
            #filesaveas
            self.actionSaveAs = qt.QAction(self)
            self.actionSaveAs.setText(QString("&Save"))
            self.actionSaveAs.setIcon(self.Icons["filesave"])
            self.actionSaveAs.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
            self.actionSaveAs.triggered[bool].connect(self.onSaveAs)

            #filesave
            self.actionSave = qt.QAction(self)
            self.actionSave.setText(QString("Save &Default Settings"))
            #self.actionSave.setIcon(self.Icons["filesave"])
            #self.actionSave.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
            self.actionSave.triggered[bool].connect(self.onSave)

            #fileprint
            self.actionPrint = qt.QAction(self)
            self.actionPrint.setText(QString("&Print"))
            self.actionPrint.setIcon(self.Icons["fileprint"])
            self.actionPrint.setShortcut(qt.Qt.CTRL+qt.Qt.Key_P)
            self.actionPrint.triggered[bool].connect(self.onPrint)

            #filequit
            self.actionQuit = qt.QAction(self)
            self.actionQuit.setText(QString("&Quit"))
            #self.actionQuit.setIcon(self.Icons["fileprint"])
            self.actionQuit.setShortcut(qt.Qt.CTRL+qt.Qt.Key_Q)
            qApp = qt.QApplication.instance()
            self.actionQuit.triggered.connect(qApp.closeAllWindows)

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
            self.menuTools= qt.QMenu()
            #self.menuTools.setCheckable(1)
            self.menuTools.aboutToShow[()].connect(self.menuToolsAboutToShow)
            self.menuTools.setTitle("&Tools")
            self.menuBar().addMenu(self.menuTools)

        if self.options["MenuWindow"]:
            self.menuWindow= qt.QMenu()
            #self.menuWindow.setCheckable(1)
            self.menuWindow.aboutToShow[()].connect(self.menuWindowAboutToShow)
            self.menuWindow.setTitle("&Window")
            self.menuBar().addMenu(self.menuWindow)

        if self.options["MenuHelp"]:
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
        self.splitter.insertWidget(0, self.sourceFrame)
        self.sourceFrame.setWindowTitle("Source Selector")
        self.sourceFrame.setWindowIcon(self.windowIcon())
        #self.splitter.setResizeMode(self.sourceFrame,qt.QSplitter.KeepSize)
        self.sourceFrameLayout = qt.QVBoxLayout(self.sourceFrame)
        self.sourceFrameLayout.setContentsMargins(0, 0, 0, 0)
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

        #connections
        self.line1.sigLineDoubleClickEvent.connect(self.sourceReparent)
        self.closelabel.sigPixmapLabelMousePressEvent.connect(self.toggleSource)

        #tips
        self.line1.setToolTip("DoubleClick toggles floating window mode")
        self.closelabel.setToolTip("Hides Source Area")

    def sourceReparent(self,ddict = None):
        if self.sourceFrame.parent() is not None:
            self.sourceFrame.setParent(None)
            self.sourceFrame.move(self.cursor().pos())
            self.sourceFrame.show()
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
        if selection is None:
            return

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
                self.mcaWindow.controlWidget.calbox.setCurrentIndex(2)
                self.mcaWindow.calibration = self.mcaWindow.calboxoptions[2]
                self.mcaWindow.controlWidget._calboxactivated("Internal")
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
        if XRFMC_FLAG:
            self.menuTools.addAction("XMI-MSIM PyMca",
                                     self._xrfmcPyMca)
        if SUMRULES_FLAG:
            self.menuTools.addAction("Sum Rules Tool", self._sumRules)
        if DEBUG:
            print("Fit to Specfile missing")

    def fontdialog(self):
        fontd = qt.QFontDialog.getFont(self)
        if fontd[1]:
            qApp = qt.QApplication.instance()
            qApp.setFont(fontd[0],1)


    def toggleSource(self,**kw):
        if DEBUG:
            print("toggleSource called")
        if self.sourceFrame.isHidden():
            self.sourceFrame.show()
            self.sourceFrame.raise_()
        else:
            self.sourceFrame.hide()

    def __elementsInfo(self):
        if self.elementsInfo is None:
            self.elementsInfo=ElementsInfo.ElementsInfo(None,"Elements Info")
        if self.elementsInfo.isHidden():
           self.elementsInfo.show()
        self.elementsInfo.raise_()

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
        self.identifier.raise_()

    def __batchFitting(self):
        if self.__batch is None:
            self.__batch = PyMcaBatch.McaBatchGUI(fl=0,actions=1)
        if self.__batch.isHidden():
            self.__batch.show()
        self.__batch.raise_()

    def __mca2EdfConversion(self):
        if self.__mca2Edf is None:
            self.__mca2Edf = Mca2Edf.Mca2EdfGUI(fl=0,actions=1)
        if self.__mca2Edf.isHidden():
            self.__mca2Edf.show()
        self.__mca2Edf.raise_()

    def __fit2SpecConversion(self):
        if self.__fit2Spec is None:
            self.__fit2Spec = Fit2Spec.Fit2SpecGUI(fl=0,actions=1)
        if self.__fit2Spec.isHidden():
            self.__fit2Spec.show()
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
        self.__correlator[-1].sigRGBCorrelatorSignal.connect( \
                self._deleteCorrelator)

        if len(filelist) == 1:
            correlator.addBatchDatFile(filelist[0])
        else:
            correlator.addFileList(filelist)

    def __sumRules(self):
        if self.__correlator is None:
            self.__correlator = []


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
            if hasattr(fdialog, "setFilters"):
                fdialog.setFilters(strlist)
            else:
                fdialog.setNameFilters(strlist)
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
                widget = QStackWidget.QStackWidget(mcawidget=self.mcaWindow,
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
                txt = "Input Output Error: %s" % (sys.exc_info()[1])
                msg.setInformativeText(txt)
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()
                return
            except:
                widget = None
                del self._widgetDict[self.__imagingTool]
                self.__imagingTool = None
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                txt = "Error info = %s" % (sys.exc_info()[1])
                msg.setInformativeText(txt)
                msg.setDetailedText(traceback.format_exc())
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
        qApp = qt.QApplication.instance()
        XiaCorrect.mainGUI(qApp)

    def _xrfmcPyMca(self):
        if self._xrfmcTool is None:
            self._xrfmcTool = XRFMCPyMca.XRFMCPyMca()
        self._xrfmcTool.show()
        self._xrfmcTool.raise_()

    def _sumRules(self):
        if self._sumRulesTool is None:
            self._sumRulesTool = SumRulesTool.SumRulesWindow()
        self._sumRulesTool.show()
        self._sumRulesTool.raise_()

    def onOpen(self):
        self.openMenu.exec_(self.cursor().pos())

    def onSave(self):
        self._saveAs()

    def onSaveAs(self):
        index = self.mainTabWidget.currentIndex()
        text  = str(self.mainTabWidget.tabText(index))
        self.saveMenu = qt.QMenu()
        self.saveMenu.addAction("PyMca Configuration", self._onSaveAs)
        if text.upper() == 'MCA':
            self.saveMenu.addAction("Active Mca",
                             self.mcaWindow._saveIconSignal)
        elif text.upper() == 'SCAN':
            self.saveMenu.addAction("Active Scan",
                             self.scanWindow._saveIconSignal)
        elif text in self.imageWindowDict.keys():
            self.saveMenu.addAction("Active Image",
                  self.imageWindowDict[text].graphWidget._saveIconSignal)
        self.saveMenu.exec_(self.cursor().pos())

    def _onSaveAs(self):
        cwd = os.getcwd()
        outfile = qt.QFileDialog(self)
        if hasattr(outfile, "setFilters"):
            outfile.setFilters(['PyMca  *.ini'])
        else:
            outfile.setNameFilters(['PyMca  *.ini'])
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(qt.QFileDialog.AcceptSave)

        if os.path.exists(self.configDir):
            cwd =self.configDir
        outfile.setDirectory(cwd)
        ret = outfile.exec_()
        if ret:
            if hasattr(outfile, "selectedFilter"):
                filterused = qt.safe_str(outfile.selectedFilter()).split()
            else:
                filterused = qt.safe_str(outfile.selectedNameFilter()).split()
            extension = ".ini"
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
                txt = "Input Output Error: %s" % (sys.exc_info()[1])
                msg.setInformativeText(txt)
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()
                return
        try:
            self._saveAs(filename)
            self.configDir = outputDir
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error Saving Configuration: %s" % (sys.exc_info()[1]))
            msg.exec_()
            return

    def _saveAs(self, filename=None):
        if filename is None:
            filename = self.__getDefaultSettingsFile()
        self.saveConfig(self.getConfig(), filename)

    def __getDefaultSettingsFile(self):
        filename = "PyMca.ini"
        if sys.platform == 'win32':
            # recipe based on: http://bugs.python.org/issue1763#msg62242
            dll = ctypes.windll.shell32
            buf = ctypes.create_unicode_buffer(MAX_PATH + 1)
            if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
                directory = buf.value
            else:
                # the above should have worked
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
            outfile = qt.QFileDialog(self)
            outfile.setWindowTitle("Select PyMca Configuration File")
            if os.path.exists(self.configDir):
                outfile.setDirectory(self.configDir)
            if hasattr(outfile, "setFilters"):
                outfile.setFilters(['PyMca  *.ini'])
            else:
                outfile.setNameFilters(['PyMca  *.ini'])
            outfile.setFileMode(outfile.ExistingFile)
            ret = outfile.exec_()
            if ret:
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
        self.sourceFrame.raise_()
        self.sourceBrowserTab.showPage(self.sourceWidget[source])
        qApp = qt.QApplication.instance()
        qApp.processEvents()
        self.sourceWidget[source].openFile()

    def onMenuHelp(self):
        if self.menuBrowser is None:
            self.menuBrowser= qt.QTextBrowser()
            self.menuBrowser.setWindowTitle(QString("Main Menu Help"))
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(os.path.join(ddir,"HTML","Menu.html")):
                ddir = os.path.dirname(ddir)
            self.menuBrowser.setSearchPaths([os.path.join(ddir,"HTML")])
            self.menuBrowser.setSource(qt.QUrl(QString("Menu.html")))
            self.menuBrowser.show()
        if self.menuBrowser.isHidden():
            self.menuBrowser.show()
        self.menuBrowser.raise_()

    def onDisplayHowto(self):
        if self.displayBrowser is None:
            self.displayBrowser= qt.QTextBrowser()
            self.displayBrowser.setWindowTitle(QString("Data Display HOWTO"))
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(os.path.join(ddir,"HTML","Display-HOWTO.html")):
                ddir = os.path.dirname(ddir)
            self.displayBrowser.setSearchPaths([os.path.join(ddir,"HTML")])
            self.displayBrowser.setSource(qt.QUrl(QString("Display-HOWTO.html")))
            self.displayBrowser.show()
        if self.displayBrowser.isHidden():
            self.displayBrowser.show()
        self.displayBrowser.raise_()

    def onMcaHowto(self):
        if self.mcaBrowser is None:
            self.mcaBrowser= MyQTextBrowser()
            self.mcaBrowser.setWindowTitle(QString("MCA HOWTO"))
            ddir=PyMcaDataDir.PYMCA_DOC_DIR
            if not os.path.exists(ddir+"/HTML"+"/MCA-HOWTO.html"):
                ddir = os.path.dirname(ddir)
            self.mcaBrowser.setSearchPaths([os.path.join(ddir,"HTML"),
                                            os.path.join(ddir,"HTML", "PyMCA_files"),
                                            os.path.join(ddir,"HTML", "images")])
            self.mcaBrowser.setSource(qt.QUrl(QString("MCA-HOWTO.html")))
            #f = open(os.path.join(dir,"HTML","MCA-HOWTO.html"))
            #self.mcaBrowser.setHtml(f.read())
            #f.close()
            self.mcaBrowser.show()
        if self.mcaBrowser.isHidden():
            self.mcaBrowser.show()
        self.mcaBrowser.raise_()

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
        if not self.scanWindow.isHidden():
            self.scanWindow.printGraph()
            return

        if not self.__useTabWidget:
            self.mcaWindow.show()
            self.mcaWindow.raise_()
        else:
            self.mainTabWidget.setCurrentWidget(self.mcaWindow)
        self.mcaWindow.printGraph()

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
            graph.windowClosed[()].connect(self.closeGraph)
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
    def setSource(self, name):
        if name == QString("./PyMCA.html") or ("PyMCA.html" in ("%s" % name)):
            if sys.platform == 'win32':
                ddir=PyMcaDataDir.PYMCA_DOC_DIR
                if not os.path.exists(os.path.join(ddir, "HTML", "PyMCA.html")):
                    ddir = os.path.dirname(ddir)
                cmd = os.path.join(ddir,"HTML", "PyMCA.pdf")
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
            qt.QTextBrowser.setSource(self, name)

class Line(qt.QFrame):
    sigLineDoubleClickEvent = qt.pyqtSignal(object)
    def mouseDoubleClickEvent(self,event):
        if DEBUG:
            print("Double Click Event")
        ddict={}
        ddict['event']="DoubleClick"
        ddict['data'] = event
        self.sigLineDoubleClickEvent.emit(ddict)

class PixmapLabel(qt.QLabel):
    sigPixmapLabelMousePressEvent = qt.pyqtSignal(object)
    def mousePressEvent(self,event):
        if DEBUG:
            print("Mouse Press Event")
        ddict={}
        ddict['event']="MousePress"
        ddict['data'] = event
        self.sigPixmapLabelMousePressEvent.emit(ddict)

if __name__ == '__main__':
    PROFILING = 0
    if PROFILING:
        import profile
        import pstats
    PyMcaMainWidgetInstance = PyMcaMain(**keywords)
    if nativeFileDialogs is not None:
        PyMcaDirs.nativeFileDialogs = nativeFileDialogs
    if debugreport:
        PyMcaMainWidgetInstance.onDebug()
    app.lastWindowClosed.connect(app.quit)

    splash.finish(PyMcaMainWidgetInstance)
    PyMcaMainWidgetInstance.show()
    PyMcaMainWidgetInstance.raise_()
    PyMcaMainWidgetInstance.mcaWindow.replot()

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

