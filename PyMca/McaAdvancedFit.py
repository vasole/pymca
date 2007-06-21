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
__revision__ = "$Revision: 1.58 $"
__author__="V.A. Sole - ESRF BLISS Group"
import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        if qt.qVersion() < '4.0.0':
            print "WARNING: Using Qt %s version" % qt.qVersion()
    except:
        import qt
else:
    import qt

QTVERSION = qt.qVersion()

try:
    import PyMcaMatplotlibSave
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
    
import ClassMcaTheory
import FitParam
import McaAdvancedTable
import QtMcaAdvancedFitReport
import ConcentrationsWidget
import ConcentrationsTool
import types
import copy
import os
import QtBlissGraph
from PyMca_Icons import IconDict
import time
import McaCalWidget
import PeakIdentifier
import ElementsInfo
Elements = ElementsInfo.Elements
#import McaROIWidget
import numpy.oldnumeric as Numeric
import PyMcaPrintPreview
import PyMcaDirs

DEBUG = 0
if DEBUG:
    print "############################################"
    print "#    McaAdvancedFit is in DEBUG mode %s     #" % DEBUG 
    print "############################################"
class McaAdvancedFit(qt.QWidget):
    def __init__(self, parent=None, name="PyMca - McaAdvancedFit",fl=0,sections=None): 
                #fl=qt.Qt.WDestructiveClose):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self, parent, name,fl)
            self.setCaption(name)
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
        else:
            qt.QWidget.__init__(self, parent)
            self.setWindowTitle(name)
            self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.lastInputDir = None
        self.configDialog = None
        self.layout = qt.QVBoxLayout(self)
        if sections is None:sections=["TABLE"]
        self.headerLabel = qt.QLabel(self)
        self.layout.addWidget(self.headerLabel)

        self.headerLabel.setAlignment(qt.Qt.AlignHCenter) 
        font = self.font()
        font.setBold(True)
        self.headerLabel.setFont(font)     
        self.setheader('Fit of XXXXXXXXXX from Channel XXXXX to XXXX')
        self.top = Top(self)
        self.layout.addWidget(self.top)
        self.sthread = None
        self.elementsInfo = None
        self.identifier   = None
        if False and len(sections) == 1:
            w = self
            self.mcatable  = McaAdvancedTable.McaTable(w)
            self.concentrationsWidget = None
        else:
            self.mainTab = qt.QTabWidget(self)
            self.layout.addWidget(self.mainTab)
            if  QTVERSION < '3.0.0':
                self.mainTab.label = self.__mainTabPatch
                self.mainTab.tabText = self.mainTab.label
                self.mainTab.currentIndex = self.mainTab.currentPageIndex
                self.mainTab.setCurrentIndex = self.mainTab.setCurrentPage
                self.mainTabLabels = ["GRAPH", "TABLE", "CONCENTRATIONS", "DIAGNOSTICS"]
            #graph
            if QTVERSION < '4.0.0':
                self.mainTab.tabText = self.mainTab.label
                self.mainTab.currentIndex = self.mainTab.currentPageIndex
                self.mainTab.setCurrentIndex = self.mainTab.setCurrentPage
                self.tabGraph  = qt.QWidget(self.mainTab,"tabGraph")
                tabGraphLayout = qt.QVBoxLayout(self.tabGraph,11,6,"tabGraphLayout")
            else:
                self.tabGraph  = qt.QWidget()
                tabGraphLayout = qt.QVBoxLayout(self.tabGraph)
                tabGraphLayout.setMargin(11)
                tabGraphLayout.setSpacing(6)
            #self.graphToolbar  = qt.QHBox(self.tabGraph)
            self.graphWindow = McaGraphWindow(self.tabGraph)
            tabGraphLayout.addWidget(self.graphWindow)
            self.graph = self.graphWindow.graph
            self.graph.xlabel('Channel')
            self.graph.ylabel('Counts')
            self.graph.canvas().setMouseTracking(1)
            self.graph.setCanvasBackground(qt.Qt.white)
            if QTVERSION < '4.0.0':
                self.mainTab.insertTab(self.tabGraph,"GRAPH")
                qt.QObject.connect(self.graphWindow,    
                                   qt.PYSIGNAL('McaGraphSignal'),
                                   self._mcaGraphSignalSlot)
            else:
                self.mainTab.addTab(self.tabGraph,"GRAPH")
                qt.QObject.connect(self.graphWindow,    
                                   qt.SIGNAL('McaGraphSignal'),
                                   self._mcaGraphSignalSlot)
            #table
            if QTVERSION < '4.0.0':
                self.tabMca  = qt.QWidget(self.mainTab)
            else:
                self.tabMca  = qt.QWidget()
            tabMcaLayout = qt.QVBoxLayout(self.tabMca)
            tabMcaLayout.setMargin(11)
            tabMcaLayout.setSpacing(6)
            w = self.tabMca
            line = Line(w, info="TABLE")
            tabMcaLayout.addWidget(line)

            if QTVERSION < '4.0.0':
                qt.QToolTip.add(line,"DoubleClick toggles floating window mode")
            else:
                line.setToolTip("DoubleClick toggles floating window mode")

            if QTVERSION < '4.0.0':
                self.mcatable  = McaAdvancedTable.McaTable(w)
                tabMcaLayout.addWidget(self.mcatable)
                self.mainTab.insertTab(w,"TABLE")
                self.connect(line,qt.PYSIGNAL("LineDoubleClickEvent"),
                                self._tabReparent) 
                self.connect(self.mcatable,qt.PYSIGNAL("closed"),
                                self._mcatableClose)
            else:
                self.mcatable  = McaAdvancedTable.McaTable(w)
                tabMcaLayout.addWidget(self.mcatable)
                self.mainTab.addTab(w,"TABLE")
                self.connect(line,qt.SIGNAL("LineDoubleClickEvent"),
                                self._tabReparent) 
                self.connect(self.mcatable,qt.SIGNAL("closed"),
                                self._mcatableClose)             
            
            #concentrations
            if QTVERSION < '4.0.0':
                self.tabConcentrations  = qt.QWidget(self.mainTab,"tabConcentrations")
            else:
                self.tabConcentrations  = qt.QWidget()
            tabConcentrationsLayout = qt.QVBoxLayout(self.tabConcentrations)
            tabConcentrationsLayout.setMargin(11)
            tabConcentrationsLayout.setSpacing(6)
            line2 = Line(self.tabConcentrations, info="CONCENTRATIONS")
            self.concentrationsWidget = ConcentrationsWidget.Concentrations(self.tabConcentrations)
            tabConcentrationsLayout.addWidget(line2)
            tabConcentrationsLayout.addWidget(self.concentrationsWidget)
            

            if QTVERSION < '4.0.0':
                self.mainTab.insertTab(self.tabConcentrations,"CONCENTRATIONS")
                qt.QToolTip.add(line2,"DoubleClick toggles floating window mode")
                self.connect(self.concentrationsWidget,
                            qt.PYSIGNAL("ConcentrationsSignal"),
                            self.__configureFromConcentrations)
                self.connect(line2,qt.PYSIGNAL("LineDoubleClickEvent"),
                                self._tabReparent)
                self.connect(self.concentrationsWidget,qt.PYSIGNAL("closed"),
                                self._concentrationsWidgetClose)
            else:
                self.mainTab.addTab(self.tabConcentrations,"CONCENTRATIONS")
                line2.setToolTip("DoubleClick toggles floating window mode")
                self.connect(self.concentrationsWidget,
                            qt.SIGNAL("ConcentrationsSignal"),
                            self.__configureFromConcentrations)
                self.connect(line2,qt.SIGNAL("LineDoubleClickEvent"),
                                self._tabReparent)
                self.connect(self.concentrationsWidget,qt.SIGNAL("closed"),
                                self._concentrationsWidgetClose)

            #diagnostics
            if QTVERSION < '4.0.0':
                self.tabDiagnostics  = qt.QWidget(self.mainTab)
            else:
                self.tabDiagnostics  = qt.QWidget()
            tabDiagnosticsLayout = qt.QVBoxLayout(self.tabDiagnostics)
            tabDiagnosticsLayout.setMargin(11)
            tabDiagnosticsLayout.setSpacing(6)
            w = self.tabDiagnostics
            if  QTVERSION < '3.0.0':
                self.diagnosticsWidget = qt.QTextView(w)
            else:
                self.diagnosticsWidget = qt.QTextEdit(w)
                self.diagnosticsWidget.setReadOnly(1)
                
            tabDiagnosticsLayout.addWidget(self.diagnosticsWidget)
            if QTVERSION < '4.0.0':
                self.mainTab.insertTab(w,"DIAGNOSTICS")
                self.connect(self.mainTab, qt.SIGNAL('currentChanged(QWidget *)'),
                             self._tabChanged)
            else:
                self.mainTab.addTab(w,"DIAGNOSTICS")
                self.connect(self.mainTab, qt.SIGNAL('currentChanged(int)'),
                             self._tabChanged)


        self._logY       = False
        self._energyAxis = False
        if QTVERSION < '4.0.0' :
            self.__printmenu = qt.QPopupMenu()
            self.__printmenu.insertItem(qt.QString("Calibrate"),     self._calibrate)        
            self.__printmenu.insertItem(qt.QString("Identify Peaks"),self.__peakIdentifier)
            self.__printmenu.insertItem(qt.QString("Elements Info"), self.__elementsInfo)
            #self.__printmenu.insertItem(qt.QString("Concentrations"),self.concentrations)
            #self.__printmenu.insertItem(qt.QString("Spectrum from Matrix"),self.matrixSpectrum)
            #self.__printmenu.insertItem(qt.QString("Print Table"),self.printps)
            #self.__printmenu.insertItem(qt.QString("HTML Report"),self.htmlReport)
        else:
            self.__printmenu = qt.QMenu()
            self.__printmenu.addAction(qt.QString("Calibrate"),     self._calibrate)        
            self.__printmenu.addAction(qt.QString("Identify Peaks"),self.__peakIdentifier)
            self.__printmenu.addAction(qt.QString("Elements Info"), self.__elementsInfo)
        self.outdir      = None
        self.configDir   = None
        self.__lastreport= None
        self.browser     = None
        self.info        = {}
        self.__fitdone   = 0
        self._concentrationsDict = None
        #self.graph.hide()
        #self.guiconfig = FitParam.Fitparam()      
        """
        self.specfitGUI.guiconfig.MCACheckBox.setEnabled(0)
        palette = self.specfitGUI.guiconfig.MCACheckBox.palette()
        palette.setDisabled(palette.active())
        """
        ##############
        hbox=qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        self.fitButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.fitButton)
        #font = self.fitButton.font()
        #font.setBold(True)
        #self.fitButton.setFont(font)
        self.fitButton.setText("Fit Again!")
        self.printButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.printButton)
        self.printButton.setText("Print")
        self.htmlReportButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.htmlReportButton)
        self.htmlReportButton.setText("HTML Report")
        self.matrixSpectrumButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.matrixSpectrumButton)
        self.matrixSpectrumButton.setText("Matrix Spectrum")
        self.peaksSpectrumButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.peaksSpectrumButton)
        self.peaksSpectrumButton.setText("Peaks Spectrum")
        if QTVERSION < '4.0.0':
            self.matrixSpectrumButton.setToggleButton(1)
            self.peaksSpectrumButton.setToggleButton(1)
            self.matrixSpectrumButton.isChecked = self.matrixSpectrumButton.isOn
            self.peaksSpectrumButton.isChecked = self.peaksSpectrumButton.isOn
        else:
            self.matrixSpectrumButton.setCheckable(1)
            self.peaksSpectrumButton.setCheckable(1)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        self.dismissButton = qt.QPushButton(hbox)
        hboxLayout.addWidget(self.dismissButton)
        self.dismissButton.setText("Dismiss")
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        
        self.layout.addWidget(hbox)
        if QTVERSION < '4.0.0':
            qt.QToolTip.add(self.printButton,'Print Active Tab')
            qt.QToolTip.add(self.htmlReportButton,'Generate Browser Compatible Output\nin Chosen Directory')
            qt.QToolTip.add(self.matrixSpectrumButton,'Toggle Matrix Spectrum Calculation On/Off')
            qt.QToolTip.add(self.peaksSpectrumButton,'Toggle Individual Peaks Spectrum Calculation On/Off')
        else:
            self.printButton.setToolTip('Print Active Tab')
            self.htmlReportButton.setToolTip('Generate Browser Compatible Output\nin Chosen Directory')
            self.matrixSpectrumButton.setToolTip('Toggle Matrix Spectrum Calculation On/Off')
            self.peaksSpectrumButton.setToolTip('Toggle Individual Peaks Spectrum Calculation On/Off')
                
        self.mcafit   = ClassMcaTheory.McaTheory()

        self.connect(self.fitButton,                qt.SIGNAL("clicked()"),self.fit)
        self.connect(self.printButton,              qt.SIGNAL("clicked()"),self.printActiveTab)
        self.connect(self.htmlReportButton,         qt.SIGNAL("clicked()"),self.htmlReport)
        self.connect(self.matrixSpectrumButton,     qt.SIGNAL("clicked()"),self.__toggleMatrixSpectrum)
        self.connect(self.peaksSpectrumButton,      qt.SIGNAL("clicked()"),self.__togglePeaksSpectrum)
        self.connect(self.dismissButton,            qt.SIGNAL("clicked()"),self.dismiss)
        self.connect(self.top.configureButton,qt.SIGNAL("clicked()") ,   self.__configure)
        self.connect(self.top.printButton,qt.SIGNAL("clicked()") ,self.__printps)
        if QTVERSION < '4.0.0':
            self.connect(self.top,qt.PYSIGNAL("TopSignal"), self.__updatefromtop)
        else:
            self.connect(self.top,qt.SIGNAL("TopSignal"), self.__updatefromtop)
        self._updateTop()
    
    def __mainTabPatch(self, index):
        return self.mainTabLabels[index]

    def _fitdone(self):
        if self.__fitdone:
            return True
        else:
            return False

    def _submitThread(self, function, parameters, message="Please wait",
                    expandparametersdict=None):
        if expandparametersdict is None: expandparametersdict= False          
        sthread = SimpleThread()
        sthread._expandParametersDict=expandparametersdict
        sthread._function = function
        sthread._kw       = parameters
        #try:
        sthread.start()
        #except:
        #    raise "ThreadError",sys.exc_info()
        if QTVERSION < '3.0.0':
            msg = qt.QDialog(self, "Please Wait", False,qt.Qt.WStyle_NoBorder)            
        elif QTVERSION < '4.0.0':
            msg = qt.QDialog(self, "Please Wait",
                             1,
                             qt.Qt.WStyle_NoBorder)
        else:
            msg = qt.QDialog(self, qt.Qt.FramelessWindowHint)
            msg.setModal(0)
            msg.setWindowTitle("Please Wait")                
        layout = qt.QHBoxLayout(msg)
        l1 = qt.QLabel(msg)
        layout.addWidget(l1)
        l1.setFixedWidth(l1.fontMetrics().width('##'))
        l2 = qt.QLabel(msg)
        layout.addWidget(l2)
        l2.setText("%s" % message)
        l3 = qt.QLabel(msg)
        layout.addWidget(l3)
        l3.setFixedWidth(l3.fontMetrics().width('##'))
        msg.show()
        qt.qApp.processEvents()
        t0 = time.time()
        i = 0
        ticks = ['-','\\', "|", "/","-","\\",'|','/']
        if QTVERSION < '4.0.0':
            while (sthread.running()):
                i = (i+1) % 8
                l1.setText(ticks[i])
                l3.setText(" "+ticks[i])
                qt.qApp.processEvents()
                time.sleep(1)
            msg.close(True)
        else:
            while (sthread.isRunning()):
                i = (i+1) % 8
                l1.setText(ticks[i])
                l3.setText(" "+ticks[i])
                qt.qApp.processEvents()
                time.sleep(1)
            msg.close()
        result = sthread._result
        del sthread
        if QTVERSION < '4.0.0':self.raiseW()
        else:self.raise_()
        return result

    def refreshWidgets(self):
        """
        This method just forces the graphical widgets to get updated.
        It should be called if somehow you have modified the fit and/
        or concentrations parameters by other means than the graphical
        interface.
        """ 
        self.__configure(justupdate=True)

        
    def __configure(self, justupdate=False):
        config = {}
        config.update(self.mcafit.config)
        #config['fit']['use_limit'] = 1
        if not justupdate:
            if self.configDialog is None:
                if self.__fitdone:
                    dialog = FitParam.FitParamDialog(modal=1,
                                                     fl=0,
                                                     initdir=self.configDir,
                                                     fitresult=self.dict['result'])  
                else:
                    dialog = FitParam.FitParamDialog(modal=1,
                                                     fl=0,
                                                     initdir=self.configDir,
                                                     fitresult=None)
                if QTVERSION < '4.0.0':
                    self.connect(dialog.fitparam.peakTable,
                                 qt.PYSIGNAL("FitPeakSelect"),
                                 self.__elementclicked)
                else:
                    self.connect(dialog.fitparam.peakTable,
                                 qt.SIGNAL("FitPeakSelect"),
                                 self.__elementclicked)
                self.configDialog = dialog
            else:
                dialog = self.configDialog
                if self.__fitdone: dialog.setFitResult(self.dict['result'])
                else:dialog.setFitResult(None)
            dialog.setParameters(config)
            #dialog.fitparam.regionCheck.setDisabled(True)
            #dialog.fitparam.minSpin.setDisabled(True)
            #dialog.fitparam.maxSpin.setDisabled(True)
            if QTVERSION < '4.0.0':
                ret = dialog.exec_loop()
            else:
                ret = dialog.exec_()
            if dialog.initDir is not None: self.configDir = 1 * dialog.initDir
            else: self.configDir = None
            if ret != qt.QDialog.Accepted:
                dialog.close()
                #del dialog
                return
            try:
                #this may crash in qt 2.3.0
                npar = dialog.getParameters()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("%s" % sys.exc_info()[1])
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
            config.update(npar)
            dialog.close()
            #del dialog

        self.graph.removeMarkers()
        self.graph.replot()
        self.__fitdone = False
        self._concentrationsDict = None
        if self.concentrationsWidget is not None:
            self.concentrationsWidget.concentrationsTable.setRowCount(0)
        if self.mcatable is not None:
            self.mcatable.setRowCount(0)
            
        if DEBUG:
            self.mcafit.configure(config)
        elif (QTVERSION < '3.0.0'):
            try:
                self.mcafit.configure(config)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("%s" % sys.exc_info()[1]) 
                msg.exec_loop()
        else:
            try:
                threadResult=self._submitThread(self.mcafit.configure, config,
                                 "Configuring, please wait")
                if type(threadResult) == type((1,)):
                    if len(threadResult):
                        if threadResult[0] == "Exception":
                            raise threadResult[1],threadResult[2]
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("%s" % sys.exc_info()[1])
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return

        #update graph
        delcurves = []
        for key in self.graph.curves.keys():
            if key not in ["Data"]:
                delcurves.append(key)
        for key in delcurves:
            self.graph.delcurve(key)
        
        if not justupdate:
            self.plot()

        self._updateTop()
        if self.concentrationsWidget is not None:
            try:
                qt.qApp.processEvents()
                self.concentrationsWidget.setParameters(config['concentrations'], signal=False)
            except:
                if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "CONCENTRATIONS":
                    self.mainTab.setCurrentIndex(0)   

    def __configureFromConcentrations(self,dummy):
        config = self.concentrationsWidget.getParameters()
        self.mcafit.config['concentrations'].update(config)

    def __elementclicked(self,ddict):
        ddict['event'] = 'McaAdvancedFitElementClicked'
        self.__showElementMarker(ddict)
        self.__anasignal(ddict)

    def __showElementMarker(self, dict):
        self.graph.removeMarkers()
        ele = dict['current']
        items = []
        if not dict.has_key(ele):
            self.graph.replot()
            return
        for rays in dict[ele]:
            for transition in Elements.Element[ele][rays +" xrays"]:
                items.append([transition,
                              Elements.Element[ele][transition]['energy'],
                              Elements.Element[ele][transition]['rate']])

        config = self.mcafit.configure()
        xdata  = self.mcafit.xdata * 1.0
        xmin = xdata[0]
        xmax = xdata[-1]
        ymin,ymax = self.graph.getY1AxisLimits()
        calib = [config['detector'] ['zero'], config['detector'] ['gain']]
        for transition,energy,rate in items:
            marker = ""
            x = (energy - calib[0])/calib[1]
            if (x < xmin) or (x > xmax):continue
            if not self._energyAxis:
                if abs(calib[1]) > 0.0000001:
                    marker=self.graph.insertX1Marker(x,
                                                     ymax*rate,
                                                     label=transition)
            else: 
                marker=self.graph.insertX1Marker(energy,
                                                 ymax*rate,
                                                 label=transition)
            if marker is not "":
                self.graph.setmarkercolor(marker,'orange')
        self.graph.replot()

    def _updateTop(self):
        config = {}
        if 0:
            config.update(self.mcafit.config['fit'])
        else:
            config['stripflag']    = self.mcafit.config['fit'].get('stripflag',0)
            config['hypermetflag'] = self.mcafit.config['fit'].get('hypermetflag',1)
            config['sumflag']      = self.mcafit.config['fit'].get('sumflag',0)
            config['escapeflag']   = self.mcafit.config['fit'].get('escapeflag',0)
            config['continuum']    = self.mcafit.config['fit'].get('continuum',0)
        self.top.setParameters(config)

    def __updatefromtop(self,ndict):
        config = self.mcafit.configure()
        for key in ndict.keys():
            if DEBUG:
                if key not in ['stripflag','hypermetflag','sumflag','escapeflag','continuum']:
                    print "UNKNOWN key ",key
            config['fit'][key] = ndict[key]

        self.__fitdone = False
        #erase table
        if self.mcatable is not None:
            self.mcatable.setRowCount(0)
        #erase concentrations
        if self.concentrationsWidget is not None:
            self.concentrationsWidget.concentrationsTable.setRowCount(0)
        #update graph
        delcurves = []
        for key in self.graph.curves.keys():
            if key not in ["Data"]:
                delcurves.append(key)
        for key in delcurves:
            self.graph.delcurve(key)
        self.plot()

        if DEBUG or (QTVERSION < '3.0.0'):
            self.mcafit.configure(config)
        else:
            threadResult=self._submitThread(self.mcafit.configure, config,
                             "Configuring, please wait")
            if type(threadResult) == type((1,)):
                if len(threadResult):
                    if threadResult[0] == "Exception":
                        raise threadResult[1],threadResult[2]

    def _tabChanged(self, value):
        if DEBUG:print "_tabChanged(self, value) called"
        if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "CONCENTRATIONS":
            self.printButton.setEnabled(False)
            w = self.concentrationsWidget
            if w.parent() is None:
                if w.isHidden():
                    w.show()
                if QTVERSION < '4.0.0': w.raiseW()
                else:w.raise_()
                self.printButton.setEnabled(True)
                #do not calculate again. It should be already updated
                return
            if DEBUG:
                self.concentrations()
                self.printButton.setEnabled(True)
            else:
                try:
                    self.concentrations()
                    self.printButton.setEnabled(True)
                except:
                    #print "try to set"
                    self.printButton.setEnabled(False)
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Concentrations error: %s" % sys.exc_info()[1])
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    self.mainTab.setCurrentIndex(0)
        elif str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "TABLE":
            self.printButton.setEnabled(True)
            w = self.mcatable
            if w.parent() is None:
                if w.isHidden():
                    w.show()
                if QTVERSION < '4.0.0':
                    w.raiseW()
                else:
                    w.raise_()
        elif str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "DIAGNOSTICS":
            self.printButton.setEnabled(False)
            self.diagnostics()
        else:
            self.printButton.setEnabled(True)

    def _concentrationsWidgetClose(self, dict):
        dict['info'] = "CONCENTRATIONS"
        self._tabReparent(dict)

    def _mcatableClose(self, dict):
        dict['info'] = "TABLE"
        self._tabReparent(dict)

    def _tabReparent(self, dict):
        if dict['info'] == "CONCENTRATIONS":
            w = self.concentrationsWidget
            parent = self.tabConcentrations
        elif dict['info'] == "TABLE":
            w = self.mcatable
            parent = self.tabMca
        if w.parent() is not None:
            if QTVERSION < '4.0.0':
                w.reparent(None,self.cursor().pos(),1)
            else:
                parent.layout().removeWidget(w)
                w.setParent(None)
                w.show()
        else: 
            if QTVERSION < '4.0.0':
                w.reparent(parent,qt.QPoint(),1)
            else:
                w.setParent(parent)
                parent.layout().addWidget(w)

    def _calibrate(self):
        config = self.mcafit.configure()
        x = self.mcafit.xdata0[:]
        y = self.mcafit.ydata0[:]
        legend = "Calibration for " +str(self.headerLabel.text())
        ndict={}
        ndict[legend] = {'A':config['detector']['zero'],
               'B':config['detector']['gain'],
               'C': 0.0}
        caldialog = McaCalWidget.McaCalWidget(legend=legend,
                                                     x=x,
                                                     y=y,
                                                     modal=1,
                                                     caldict=ndict,
                                                     fl=0)
        caldialog.calpar.orderbox.setEnabled(0)
        caldialog.calpar.CText.setEnabled(0)
        caldialog.calpar.savebox.setEnabled(0)
        if QTVERSION < '4.0.0':
            ret = caldialog.exec_loop()
        else:
            ret = caldialog.exec_()
        if ret == qt.QDialog.Accepted:
            dict = caldialog.getdict()
            config['detector']['zero'] = dict[legend]['A']
            config['detector']['gain'] = dict[legend]['B']
            #self.mcafit.configure(config)
            self.mcafit.config['detector']['zero'] = 1. * dict[legend]['A']
            self.mcafit.config['detector']['gain'] = 1. * dict[legend]['B']
            self.__fitdone = 0
            self.plot()
        del caldialog

    def __elementsInfo(self):
        if self.elementsInfo is None:self.elementsInfo=ElementsInfo.ElementsInfo(None,"Elements Info")
        if self.elementsInfo.isHidden():
           self.elementsInfo.show()
        if QTVERSION < '4.0.0':self.elementsInfo.raiseW()
        else:self.elementsInfo.raise_()

    def __peakIdentifier(self, energy = None):
        if energy is None:energy = 5.9
        if self.identifier is None:
            self.identifier=PeakIdentifier.PeakIdentifier(energy=energy,
                                                          threshold=0.040,
                                                          useviewer=1)
            self.identifier.myslot()
        self.identifier.setEnergy(energy)
        if self.identifier.isHidden():
            self.identifier.show()
        if QTVERSION < '4.0.0':
            self.identifier.raiseW()
        else:
            self.identifier.raise_()
        
    def printActiveTab(self):
        if   str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "GRAPH":
            self.graph.printps()
        elif str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "TABLE":
            self.printps(True)
        elif str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "CONCENTRATIONS":
            self.printConcentrations(True)
        elif str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "DIAGNOSTICS":
            pass
        else:
            pass

    def diagnostics(self):
        if not self.__fitdone:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "Sorry. You need to perform a fit first.\n"
            msg.setText(text)
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "DIAGNOSTICS":
                self.mainTab.setCurrentIndex(0)
            return
        fitresult = self.dict
        x = fitresult['result']['xdata']
        energy = fitresult['result']['energy']
        y = fitresult['result']['ydata']
        yfit = fitresult['result']['yfit']
        param = fitresult['result']['fittedpar']
        i    = fitresult['result']['parameters'].index('Noise')
        noise= param[i] * param[i]
        i    = fitresult['result']['parameters'].index('Fano')
        fano = param[i] * 2.3548*2.3548*0.00385
        meanfwhm = Numeric.sqrt(noise + 0.5 * (energy[0] + energy[-1]) * fano)
        i = fitresult['result']['parameters'].index('Gain')
        gain = fitresult['result']['fittedpar'][i]
        meanfwhm = int(meanfwhm/gain) + 1
        missed =  self.mcafit.detectMissingPeaks(y, yfit, meanfwhm)
        hcolor = 'white'
        finalcolor = 'white'
        text=""
        if len(missed):
            text+="<br><b><font color=blue size=4>Possibly Missing or Underestimated Peaks</font></b>"
            text+="<nobr><table><tr>"
            text+='<td align="right" bgcolor="%s"><b>' % hcolor
            text+='Channel'
            text+="</b></td>"
            text+='<td align="right" bgcolor="%s"><b>' % hcolor
            text+='Energy'
            text+="</b></td>"
            text+="</tr>"
            text+="<tr>"
            for peak in missed:
                text+="<tr>"
                text+='<td align="right" bgcolor="%s">' % finalcolor
                text+="<b><font size=3>%d </font></b>"  % x[int(peak)]
                text+="</td>"
                text+='<td align="right" bgcolor="%s">' % finalcolor
                text+="<b><font size=3>%.3f </font></b>"  % energy[int(peak)]
                text+="</td>"        
            text+="</tr>"
            text+="</table>"
        missed =  self.mcafit.detectMissingPeaks(yfit, y, meanfwhm)
        if len(missed):
            text+="<br><b><font color=blue size=4>Possibly Overestimated Peaks</font></b>"
            text+="<nobr><table><tr>"
            text+='<td align="right" bgcolor="%s"><b>' % hcolor
            text+='Channel'
            text+="</b></td>"
            text+='<td align="right" bgcolor="%s"><b>' % hcolor
            text+='Energy'
            text+="</b></td>"
            text+="</tr>"
            text+="<tr>"
            for peak in missed:
                text+="<tr>"
                text+='<td align="right" bgcolor="%s">' % finalcolor
                text+="<b><font size=3>%d </font></b>"  % x[int(peak)]
                text+="</td>"
                text+='<td align="right" bgcolor="%s">' % finalcolor
                text+="<b><font size=3>%.3f </font></b>"  % energy[int(peak)]
                text+="</td>"        
            text+="</tr>"
            text+="</table>"
        if QTVERSION < '4.0.0':
            self.diagnosticsWidget.setText(text)
        else:
            self.diagnosticsWidget.clear()
            self.diagnosticsWidget.insertHtml(text)
        
    def concentrations(self):
        self._concentrationsDict = None
        if not self.__fitdone:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "Sorry, You need to perform a fit first.\n"
            msg.setText(text)
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == "CONCENTRATIONS":
                self.mainTab.setCurrentIndex(0)
            return
        fitresult = self.dict
        if False:
            #from the fit
            config = fitresult['result']['config']
        else:
            #from current
            config = self.mcafit.configure()
        #tool = ConcentrationsWidget.Concentrations(fl=qt.Qt.WDestructiveClose)
        if self.concentrationsWidget is None:
           self.concentrationsWidget = ConcentrationsWidget.Concentrations()
           if QTVERSION < '4.0.0':
               self.connect(self.concentrationsWidget,qt.PYSIGNAL("ConcentrationsSignal"),
                                    self.__configureFromConcentrations)
           else:
               self.connect(self.concentrationsWidget,qt.SIGNAL("ConcentrationsSignal"),
                                    self.__configureFromConcentrations)
        tool = self.concentrationsWidget 
        toolconfig = tool.getParameters()
        dict = {}
        dict.update(config['concentrations'])
        tool.setParameters(dict, signal=False)
        if DEBUG:
            dict = tool.processFitResult(config=dict,fitresult=fitresult,
                    elementsfrommatrix=False,
                    fluorates = self.mcafit._fluoRates)
        else:
            try:
                dict = tool.processFitResult(config=dict,fitresult=fitresult,
                    elementsfrommatrix=False,
                    fluorates = self.mcafit._fluoRates)        
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error processing fit result: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == 'CONCENTRATIONS':
                    self.mainTab.setCurrentIndex(0)
                return
        self._concentrationsDict = dict
        tool.show()
        tool.setFocus()
        if QTVERSION < '4.0.0':
            tool.raiseW()
        else:
            tool.raise_()

    def __toggleMatrixSpectrum(self):
        if self.matrixSpectrumButton.isChecked():
            self.matrixSpectrum()
            self.plot()
        else:
            if "Matrix" in self.graph.curves.keys():
                self.graph.delcurve("Matrix")
                self.plot()

    def __togglePeaksSpectrum(self):
        if self.peaksSpectrumButton.isChecked():
            self.peaksSpectrum()
            self.plot()
        else:
            self.__clearPeaksSpectrum()
            self.plot()

    def __clearPeaksSpectrum(self):
        delcurves = []
        for key in self.graph.curves.keys():
            if key not in ["Data", "Fit", "Continuum", "Pile-up", "Matrix"]:
                delcurves.append(key)
        for key in delcurves:
            self.graph.delcurve(key)

    def matrixSpectrum(self):
        if not self.__fitdone:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "Sorry, for the time being you need to perform a fit first\n"
            text+= "in order to calculate the spectrum derived from the matrix.\n"
            text+= "Background and detector parameters are taken from last fit"
            msg.setText(text)
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        #fitresult = self.dict['result']
        fitresult = self.dict
        config = self.mcafit.configure()
        tool   = ConcentrationsTool.ConcentrationsTool()
        toolconfig = tool.configure()
        dict = {}
        dict.update(config['concentrations'])
        tool.configure(dict)
        if DEBUG:
            dict = tool.processFitResult(fitresult=fitresult,
                                         elementsfrommatrix=True)
        elif (QTVERSION < '3.0.0'):
            try:
                dict = tool.processFitResult(fitresult=fitresult,
                                         elementsfrommatrix=True)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error: %s" % (sys.exc_info()[1]))
                msg.exec_loop()
                return
        else:
            try:
                threadResult = self._submitThread(tool.processFitResult,
                                   {'fitresult':fitresult,'elementsfrommatrix':True},
                                   "Calculating Matrix Spectrum",
                                   True)
                if type(threadResult) == type((1,)):
                    if len(threadResult):
                        if threadResult[0] == "Exception":
                            raise threadResult[1],threadResult[2]
                dict = threadResult
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
        groupsList = fitresult['result']['groups']

        if type(groupsList) != types.ListType:
            groupsList = [groupsList]
        areas = []
        for group in groupsList:
            item = group.split()
            element = item[0]
            if len(element) >2:
                areas.append(0.0)
            else:
                transitions = item[1] + " xrays"
                areas.append(dict['area'][group])

        nglobal    = len(fitresult['result']['parameters']) - len(groupsList) 
        parameters = []
        for i in range(len(fitresult['result']['parameters'])):
            if i < nglobal:
                parameters.append(fitresult['result']['fittedpar'][i])
            else:
                parameters.append(areas[i-nglobal])
        
        xmatrix = fitresult['result']['xdata']
        ymatrix = self.mcafit.mcatheory(parameters,xmatrix)
        ymatrix.shape =  [len(ymatrix),1]
        dict=copy.deepcopy(self.dict)
        dict['event'] = "McaAdvancedFitMatrixFinished"
        if self.mcafit.STRIP:
            dict['result']['ymatrix']  = ymatrix + self.mcafit.zz
        else:
            dict['result']['ymatrix']  = ymatrix
        dict['result']['ymatrix'].shape  = (len(dict['result']['ymatrix']),)
        dict['result']['continuum'].shape  = (len(dict['result']['ymatrix']),)
        if self.matrixSpectrumButton.isChecked():self.dict['result']['ymatrix']= dict['result']['ymatrix'] * 1.0
        """
        if self.graph is not None:
            if self._logY:
                logfilter = 1
            else:
                logfilter = 0
            if self._energyAxis:
                xdata = dict['result']['energy'][:]
            else:
                xdata = dict['result']['xdata'][:]
            self.graph.newCurve("Matrix",xdata,dict['result']['ymatrix'],logfilter=logfilter)
        """
        try:
            self.__anasignal(dict)
        except:
            print "Error generating matrix output. "
            print "Try to perform your fit again.  "
            print  sys.exc_info()
            print "If error persists, please report this error."
            print "ymatrix shape = ", dict['result']['ymatrix'].shape
            print "xmatrix shape = ", xmatrix.shape
            print "continuum shape = ", dict['result']['continuum'].shape
            print "zz      shape = ", self.mcafit.zz.shape

    def peaksSpectrum(self):
        if not self.__fitdone:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            text = "You need to perform a fit first\n"
            msg.setText(text)
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        #fitresult = self.dict['result']
        fitresult = self.dict
        config = self.mcafit.configure()
        groupsList = fitresult['result']['groups']
        if type(groupsList) != types.ListType:
            groupsList = [groupsList]

        nglobal    = len(fitresult['result']['parameters']) - len(groupsList) 
        dict=copy.deepcopy(self.dict)
        dict['event'] = "McaAdvancedFitPeaksFinished"
        newparameters = fitresult['result']['fittedpar'] * 1
        for i in range(nglobal,len(fitresult['result']['parameters'])):
            newparameters[i] = 0.0
        for i in range(nglobal,len(fitresult['result']['parameters'])):
            group = fitresult['result']['parameters'][i]
            parameters    = newparameters * 1
            parameters[i] = fitresult['result']['fittedpar'][i] 
            xmatrix = fitresult['result']['xdata']
            ymatrix = self.mcafit.mcatheory(parameters,xmatrix)
            ymatrix.shape =  [len(ymatrix),1]
            label = 'y'+group
            if self.mcafit.STRIP:
                dict['result'][label]  = ymatrix + self.mcafit.zz
            else:
                dict['result'][label]  = ymatrix
            dict['result'][label].shape  = (len(dict['result'][label]),)
            if self.peaksSpectrumButton.isChecked():self.dict['result'][label]= dict['result'][label] * 1.0
        try:
            self.__anasignal(dict)
        except:
            print "Error generating peaks output. "
            print "Try to perform your fit again.  "
            print "If error persists, please report this error."
            print "ymatrix shape = ", dict['result'][group].shape
            print "xmatrix shape = ", xmatrix.shape
            print "zz      shape = ", self.mcafit.zz.shape
    
    def __printps(self):
        if QTVERSION < '4.0.0':
            self.__printmenu.exec_loop(self.cursor().pos())
        else:
            self.__printmenu.exec_(self.cursor().pos())    
    
    def htmlReport(self,index=None):
        if not self.__fitdone:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("You should perform a fit \nfirst,\n shouldn't you?")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        oldoutdir = self.outdir
        if self.outdir is None:
            cwd = PyMcaDirs.outputDir
            if QTVERSION < '4.0.0':
                outfile = qt.QFileDialog(self,"Output Directory Selection",1) 
                outfile.setMode(outfile.DirectoryOnly)
                outfile.setDir(cwd)
                ret = outfile.exec_loop()
            else:
                outfile = qt.QFileDialog(self)
                outfile.setWindowTitle("Output Directory Selection")
                outfile.setModal(1) 
                outfile.setFileMode(outfile.DirectoryOnly)
                outfile.setDirectory(cwd)
                ret = outfile.exec_()
            if ret:
                if QTVERSION < '4.0.0':
                    self.outdir=str(outfile.selectedFile())
                else:
                    self.outdir=str(outfile.selectedFiles()[0])
                outfile.close()
                del outfile
            else:
                outfile.close()
                del outfile
                return
            if self.outdir[-1]=="/":self.outdir=self.outdir[0:-1]
        try:
            report = self.__htmlReport()
        except IOError:
            self.outdir = None
            if oldoutdir is not None:
                if os.path.exists(oldoutdir):
                    self.outdir = oldoutdir
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            
                    
    def __htmlReport(self,outfile=None):
        report = QtMcaAdvancedFitReport.QtMcaAdvancedFitReport(None,
                    outfile,self.outdir,self.info['sourcename'],
                    self.info['legend'],self.dict,
                    concentrations=self._concentrationsDict)
        if 0:
            #this forces to open and read the file
            self.__lastreport = report.writeReport()
        else:
            text = report.getText()
            self.__lastreport = report.writeReport(text=text)
        if self.browser is None:
            self.browser= qt.QWidget()
            if QTVERSION < '4.0.0':
                self.browser.setCaption(qt.QString(self.__lastreport))
            else:
                self.browser.setWindowTitle(qt.QString(self.__lastreport))
            self.browser.layout = qt.QVBoxLayout(self.browser)
            self.browser.layout.setMargin(0)
            self.browser.layout.setSpacing(0)
            if QTVERSION < '4.0.0':
                self.__printmenu.insertSeparator()
                self.__printmenu.insertItem(qt.QString("Last Report"),self.showLastReport)
            else:
                self.__printmenu.addSeparator()
                self.__printmenu.addAction(qt.QString("Last Report"),self.showLastReport)
                
            if QTVERSION < '4.0.0':
                self.browsertext= qt.QTextView(self.browser)
            else:
                self.browsertext = qt.QTextBrowser(self.browser)
                self.browsertext.setReadOnly(1)
                screenWidth = qt.QDesktopWidget().width()
                if screenWidth > 0:
                    self.browsertext.setMinimumWidth(min(self.width(), int(0.5 * screenWidth)))
                else:
                    self.browsertext.setMinimumWidth(self.width())
                screenHeight = qt.QDesktopWidget().height()
                if screenHeight > 0:
                    self.browsertext.setMinimumHeight(min(self.height(), int(0.5 * screenHeight)))
                else:
                    self.browsertext.setMinimumHeight(self.height())

            self.browser.layout.addWidget(self.browsertext)
        else:
            if QTVERSION < '4.0.0':
                self.browser.setCaption(qt.QString(self.__lastreport))
            else:
                self.browser.setWindowTitle(qt.QString(self.__lastreport))

        if QTVERSION < '4.0.0':
            self.browsertext.mimeSourceFactory().addFilePath(qt.QString(os.path.dirname(self.__lastreport)))
            self.browsertext.setText(text)
        else:
            dirname  = os.path.dirname(self.__lastreport)
            basename = os.path.basename(self.__lastreport)
            #self.browsertext.setMimeSourceFactory(qt.QMimeFactory.defaultFactory())
            #self.browsertext.mimeSourceFactory().addFilePath(qt.QString(dirname))
            self.browsertext.setSearchPaths([qt.QString(dirname)])
            #self.browsertext.setSource(qt.QUrl(qt.QString(basename)))
            self.browsertext.clear()
            if QTVERSION < '4.2.0':
                self.browsertext.insertHtml(text)
            else:
                self.browsertext.setText(text)
        self.browsertext.show()
        self.showLastReport()

    def showLastReport(self):
        if self.browser is not None:
            self.browser.show()
            if QTVERSION < '4.0.0':
                self.browser.raiseW()
            else:
                self.browser.raise_()

    def printConcentrations(self,doit=0):
        text = "<CENTER>"+self.concentrationsWidget.concentrationsTable.getHtmlText()+"</CENTER>"
        if (__name__ == "__main__") or (doit):
            self.__print(text)
            #print h+text
        else:
            dict={}
            dict['event'] = "McaAdvancedFitPrint"
            dict['text' ] = h+text
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaAdvancedFitSignal'),(dict,))
            else:
                self.emit(qt.SIGNAL('McaAdvancedFitSignal'),(dict))
    
    def printps(self,doit=0):
        h = self.__htmlheader()
        text = "<CENTER>"+self.mcatable.gettext()+"</CENTER>"
        #text = self.mcatable.gettext()
        if (__name__ == "__main__") or (doit):
            self.__print(h+text)
            #print h+text
        else:
            dict={}
            dict['event'] = "McaAdvancedFitPrint"
            dict['text' ] = h+text
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaAdvancedFitSignal'),(dict,))
            else:
                self.emit(qt.SIGNAL('McaAdvancedFitSignal'),(dict))

    def __htmlheader(self):
        header = "%s" % str(self.headerLabel.text())
        if header[0] == "<":
            header = header[3:-3]
        if self.mcafit.config['fit']['sumflag']:
            sumflag = "Y"
        else:
            sumflag = "N"
                 
        if self.mcafit.config['fit']['escapeflag']:
            escapeflag =  "Y"
        else:
            escapeflag = "N"
        
        if self.mcafit.config['fit']['stripflag']:
            stripflag = "Y"
        else:
            stripflag = "N"
        #bkg = self.mcafit.config['fit']['continuum']
        #theory = "Hypermet"
        bkg    = "%s" % str(self.top.BkgComBox.currentText())
        theory = "%s" % str(self.top.FunComBox.currentText())
        hypermetflag=self.mcafit.config['fit']['hypermetflag']
        g_term    = hypermetflag  & 1
        st_term   = (hypermetflag >>1) & 1
        lt_term   = (hypermetflag >>2) & 1
        step_term = (hypermetflag >>3) & 1
        if st_term:
            st_term = "Y"
        else:
            st_term = "N"
        if st_term:
            lt_term = "Y"
        else:
            lt_term = "N"
        if step_term:
            step_term = "Y"
        else:
            step_term = "N"

        
        h=""
        h+="    <CENTER>"
        h+="<B>%s</B>" % header
        h+="    </CENTER>"
        h+="    <SPACER TYPE=BLOCK HEIGHT=10>"
        #h+="<BR></BR>"
        h+="    <CENTER>"
        h+="<TABLE>"
        h+="<TR>"
        h+="    <TD ALIGN=LEFT><B>Function</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD NOWRAP ALIGN=LEFT>%s</TD>" % theory
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=15></TD>"
        
        h+="    <TD ALIGN=LEFT><B>ShortTail</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD>%s</TD>" % st_term
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=5></B></TD>"
        
        h+="    <TD ALIGN=LEFT><B>LongTail</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD>%s</TD>" % lt_term
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=5></B></TD>"
        
        h+="    <TD ALIGN=LEFT><B>StepTail</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD>%s</TD>" % step_term
        h+="</TR>"

        h+="<TR>"
        #h+="    <TD ALIGN=LEFT><B>Background</B></TH>"
        h+="    <TD ALIGN=LEFT><B>Backg.</B></TH>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD NOWRAP ALIGN=LEFT>%s</TD>" % bkg
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=15></B></TD>"

        h+="    <TD ALIGN=LEFT><B>Escape</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD>%s</TD>" % escapeflag
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=5></B></TD>"

        h+="    <TD ALIGN=LEFT><B>Pile-up</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD ALIGN=LEFT>%s</TD>" % sumflag
        h+="    <TD><SPACER TYPE=BLOCK WIDTH=5></B></TD>"

        h+="    <TD ALIGN=LEFT><B>Strip</B></TD>"
        h+="    <TD><B>:</B></TD>"
        h+="    <TD>%s</TD>" % stripflag

        h+="</TR>"
        h+="</TABLE>"
        h+="</CENTER>"
        return h


    if QTVERSION < '4.0.0':
        def __print(self,text):
            printer = qt.QPrinter()
            if printer.setup(self):
                painter = qt.QPainter()
                if not(painter.begin(printer)):
                    return 0
                metrics = qt.QPaintDeviceMetrics(printer)
                dpiy    = metrics.logicalDpiY()
                margin  = int((2/2.54) * dpiy) #2cm margin
                body = qt.QRect(0.5*margin, margin, metrics.width()- 1 * margin, metrics.height() - 2 * margin)
                #html output -> print text
                richtext = qt.QSimpleRichText(text, qt.QFont(),
                                                    qt.QString(""),
                                                    #0,
                                                    qt.QStyleSheet.defaultSheet(),
                                                    qt.QMimeSourceFactory.defaultFactory(),
                                                    body.height())
                view = qt.QRect(body)
                richtext.setWidth(painter,view.width())
                page = 1                
                while(1):
                    if QTVERSION < '3.0.0':
                        richtext.draw(painter,body.left(),body.top(),
                                    qt.QRegion(0.5*margin, margin, metrics.width()- 1 * margin, metrics.height() - 2 * margin),
                                    qt.QColorGroup())
                        #richtext.draw(painter,body.left(),body.top(),
                        #            qt.QRegion(view),
                        #            qt.QColorGroup())
                    else:
                        richtext.draw(painter,body.left(),body.top(),
                                    view,qt.QColorGroup())
                    view.moveBy(0, body.height())
                    painter.translate(0, -body.height())
                    painter.drawText(view.right()  - painter.fontMetrics().width(qt.QString.number(page)),
                                     view.bottom() - painter.fontMetrics().ascent() + 5,qt.QString.number(page))
                    if view.top() >= richtext.height():
                        break
                    printer.newPage()
                    page += 1

                #painter.flush()
                painter.end()
    else:
        def __print(self,text):
            printer = qt.QPrinter()
            printDialog = qt.QPrintDialog(printer, self)
            if printDialog.exec_():
                if 0:
                    #this was crashing in Qt 4.2.2
                    #with the PyQt snapshot of 20061203
                    editor = qt.QTextEdit()
                    cursor = editor.textCursor()
                    cursor.movePosition(qt.QTextCursor.Start)
                    editor.insertHtml(text)
                    document = editor.document()
                else:
                    document = qt.QTextDocument()
                    document.setHtml(text)        
                document.print_(printer)

    def setdata(self, *var, **kw):
        if DEBUG:print "McaAdvancedFit.setdata deprecated, use setData instead:"
        return self.setData( *var, **kw)

    def setData(self,*var,**kw):
        self.__fitdone = 0
        self.info ={}
        if kw.has_key('legend'):
            self.info['legend'] = kw['legend']
        else:
            self.info['legend'] = 'Unknown Origin'
        if kw.has_key('xlabel'):
            self.info['xlabel'] = kw['xlabel']
        else:
            self.info['xlabel'] = 'X'
        if kw.has_key('xmin'):
            self.info['xmin'] = "%.3f" % kw['xmin']
        else:
            self.info['xmin'] = "????"  

        if kw.has_key('xmax'):
            self.info['xmax'] = "%.3f" % kw['xmax']
        else:
            self.info['xmax'] = "????"  
        if kw.has_key('sourcename'):
            self.info['sourcename'] = "%s" % kw['sourcename']
        else:
            self.info['sourcename'] = "Unknown Source"  
        self.__var = var
        self.__kw  = kw
        self.mcafit.setdata(*var,**kw)

        if kw.has_key('calibration'):
            if kw['calibration'] is not None:
                if kw['calibration'] != [0.0,1.0,0.0]:
                    if 0:
                        dict=self.mcafit.configure()
                        if kw['calibration'][1] > 0.1:
                          dict['detector']['zero']=kw['calibration'][0]/1000.
                          dict['detector']['gain']=kw['calibration'][1]/1000.
                        else:
                          dict['detector']['zero']=kw['calibration'][0]
                          dict['detector']['gain']=kw['calibration'][1]                        
                        self.mcafit.configure(dict)
                    else:
                        #this is faster
                        if kw['calibration'][1] > 0.1:
                          self.mcafit.config['detector']['zero']=kw['calibration'][0]/1000.
                          self.mcafit.config['detector']['gain']=kw['calibration'][1]/1000.
                        else:
                          self.mcafit.config['detector']['zero']=kw['calibration'][0] * 1
                          self.mcafit.config['detector']['gain']=kw['calibration'][1] * 1
                    
        self.setheader(text="Fit of %s from %s %s to %s" % (self.info['legend'],
                                                            self.info['xlabel'],
                                                            self.info['xmin'],
                                                            self.info['xmax']))
        self._updateTop()
        self.plot()

    def setheader(self, *var, **kw):
        if DEBUG:print "McaAdvancedFit.setheader deprecated, use setHeader instead:"
        return self.setHeader( *var, **kw)

    def setHeader(self,*var,**kw):
        if len(var):
            text = var[0]
        elif kw.has_key('text'):
            text = kw['text']
        elif kw.has_key('header'):
            text = kw['header']
        else:
            text = ""
        self.headerLabel.setText("%s" % text)

    def fit(self):
        self.__fitdone = 0
        self.mcatable.setRowCount(0)
        if self.concentrationsWidget is not None:
            self.concentrationsWidget.concentrationsTable.setRowCount(0)            
        fitconfig = {}
        fitconfig.update(self.mcafit.configure())
        if fitconfig['peaks'] == {}:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("No peaks defined.\nPlease configure peaks")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        if DEBUG or (QTVERSION < '3.0.0'):
            if DEBUG: print "calling estimate"
            self.mcafit.estimate()
            if DEBUG: print "calling startfit"
            fitresult,result = self.mcafit.startfit(digest=1)
            if DEBUG: print "filling table"
            self.mcatable.fillfrommca(result)
            if DEBUG: print "finished"
        else:
            try:
                self.mcafit.estimate()
                threadResult = self._submitThread(self.mcafit.startfit,
                                            {'digest':1},
                                            "Calculating Fit",
                                            True)
                if type(threadResult) == type((1,)):
                    if len(threadResult):
                        if threadResult[0] == "Exception":
                            raise threadResult[1],threadResult[2]
                fitresult = threadResult[0]
                result    = threadResult[1]
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on fit: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
            try:
                #self.mcatable.fillfrommca(self.mcafit.result)
                self.mcatable.fillfrommca(result)
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error filling Table: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
        dict={}
        dict['event']     = "McaAdvancedFitFinished"
        dict['fitresult'] = fitresult
        dict['result']    = result
        #I should make a copy but ...
        self.dict = {}
        self.dict['info'] = {}
        self.dict['info'] = self.info.copy()
        self.dict['result'] = dict['result']
        self.__fitdone      = 1
        # add the matrix spectrum
        if self.matrixSpectrumButton.isChecked():
            self.matrixSpectrum()
        else:
            if "Matrix" in self.graph.curves.keys():
                self.graph.delcurve("Matrix")
        # add the peaks spectrum
        if self.peaksSpectrumButton.isChecked():
            self.peaksSpectrum()
        else:
            self.__clearPeaksSpectrum()
        self.plot()

        if self.concentrationsWidget is not None:
            if (str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == 'CONCENTRATIONS') or \
                (self.concentrationsWidget.parent() is None):
                if not self.concentrationsWidget.isHidden():
                    if DEBUG:
                        self.concentrations()
                    else:
                        try:
                            self.concentrations()
                        except:
                            msg = qt.QMessageBox(self)
                            msg.setIcon(qt.QMessageBox.Critical)
                            msg.setText("Concentrations Error: %s" % (sys.exc_info()[1]))
                            if QTVERSION < '4.0.0':
                                msg.exec_loop()
                            else:
                                msg.exec_()
                            return
        if str(self.mainTab.tabText(self.mainTab.currentIndex())).upper() == 'DIAGNOSTICS':
            try:
                self.diagnostics()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Diagnostics Error: %s" % (sys.exc_info()[1]))
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
                
        self.__anasignal(dict)
        
    
    def __anasignal(self,dict):
        if type(dict) != type({}):
            return
        if dict.has_key('event'):
            dict['info'] = {}
            dict['info'].update(self.info)
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaAdvancedFitSignal'),(dict,))
            else:
                self.emit(qt.SIGNAL('McaAdvancedFitSignal'),(dict))
                
    def dismiss(self):
        self.close()

    def closeEvent(self, event):
        if self.identifier is not None:
            self.identifier.close()
        qt.QWidget.closeEvent(self, event)
    
    def _mcaGraphSignalSlot(self, ddict):
        if ddict['event'] == "FitClicked":
            self.fit()
        elif ddict['event'] == "LogClicked":
            self.toggleLogY(ddict)
        elif ddict['event'] == "EnergyClicked":
            self.toggleEnergyAxis()
        elif ddict['event'] == "SaveClicked":
            self._saveGraph()
        elif ddict['event'] == 'MouseClick':
            if self._energyAxis:
                self.__peakIdentifier(ddict['x'])
        else:
            pass
        return

    def toggleEnergyAxis(self, dict=None):
        if self._energyAxis:
            self._energyAxis = False
            self.graph.xlabel('Channel')
        else:
            self._energyAxis = True
            self.graph.xlabel('Energy')
        self.plot()
        
    def toggleLogY(self, dict=None):
        if self._logY:
            self.graph.ToggleLogY()
            self._logY = False
            self.plot()
        else:
            self._logY = True
            self.plot()
            self.graph.ToggleLogY()

    def plot(self, dict=None):
        if self._logY:
            logfilter = 1
        else:
            logfilter = 0    
        config = self.mcafit.configure()
        if dict is None: 
            if not self.__fitdone:
                for key in self.graph.curves.keys():
                    if key not in ["Data"]:
                        self.graph.delcurve(key)
                #just the data
                xdata  = self.mcafit.xdata * 1.0
                if self._energyAxis:
                    xdata = config['detector'] ['zero'] + config['detector'] ['gain'] * xdata
                if self.mcafit.STRIP:
                    ydata  = self.mcafit.ydata + self.mcafit.zz
                else:
                    ydata  = self.mcafit.ydata * 1.0
                xdata.shape= [len(xdata),]
                ydata.shape= [len(ydata),]
                self.graph.newCurve("Data",xdata,
                                           ydata,logfilter=logfilter)
                self.graph.replot()
                return
            else:
                dict = self.dict
        if self._energyAxis:
            xdata = dict['result']['energy'][:]
        else:
            xdata = dict['result']['xdata'][:]
        self.graph.newCurve("Data",xdata,
                                   dict['result']['ydata'],logfilter=logfilter)
        self.graph.newCurve("Fit", xdata,
                                   dict['result']['yfit'],logfilter=logfilter)
        self.graph.newCurve("Continuum",xdata,
                                   dict['result']['continuum'],logfilter=logfilter)
                                   
        if config['fit']['sumflag']:
            self.graph.newCurve("Pile-up", xdata,
                                       dict['result']['pileup']+dict['result']['continuum'],
                                       logfilter=logfilter)
        elif "Pile-up" in self.graph.curves.keys():
            self.graph.delcurve("Pile-up")
        
        if self.matrixSpectrumButton.isChecked():
            if dict['result'].has_key('ymatrix'):
                self.graph.newCurve("Matrix",xdata, dict['result']['ymatrix'],
                                    logfilter=logfilter)
            else:
                if "Matrix" in self.graph.curves.keys():
                    self.graph.delcurve("Matrix")
        else:
            if "Matrix" in self.graph.curves.keys():
                self.graph.delcurve("Matrix")

        if self.peaksSpectrumButton.isChecked():
            keep = ['Data','Fit','Continuum','Matrix','Pile-up']
            for group in dict['result']['groups']:
                keep += ['y'+group]
            for key in self.graph.curves.keys():
                if key not in keep:
                    self.graph.delcurve(key)
            for group in dict['result']['groups']:
                label = 'y'+group
                if dict['result'].has_key(label):
                    self.graph.newCurve(label,xdata, dict['result'][label],
                                    logfilter=logfilter)
                else:
                    if group in self.graph.curves.keys():
                        self.graph.delcurve(label)
        else:
            self.__clearPeaksSpectrum()
               
        self.graph.replot()
       
    def _saveGraph(self, dict=None):
        curves = self.graph.curves.keys()
        if not len(curves):return
        if not self.__fitdone:
            if False:
                #just the data
                xdata  = self.mcafit.xdata
                config = self.mcafit.configure()
                energy = config['detector'] ['zero'] + config['detector'] ['gain'] * xdata
                if self.mcafit.STRIP:
                    ydata  = self.mcafit.ydata + self.mcafit.zz
                else:
                    ydata  = self.mcafit.ydata
                savelist = ['xdata', 'energy','ydata']
            else:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                text = "Sorry, You need to perform a fit first.\n"
                msg.setText(text)
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
        if dict is None:
            #everything
            fitresult = self.dict
        else:
            fitresult = dict            
        xdata     = fitresult['result']['xdata']
        energy    = fitresult['result']['energy']
        ydata     = fitresult['result']['ydata']
        yfit      = fitresult['result']['yfit']
        continuum = fitresult['result']['continuum']
        pileup    = fitresult['result']['pileup']
        savelist  = ['xdata', 'energy','ydata','yfit','continuum','pileup']
        parNames  = fitresult['result']['parameters']
        parFit    = fitresult['result']['fittedpar']
        parSigma  = fitresult['result']['sigmapar']
        #still to add the matrix spectrum

        #get outputfile
        outfile = qt.QFileDialog(self)
        outfile.setModal(1)
        if self.lastInputDir is None:
            self.lastInputDir = PyMcaDirs.outputDir
        if QTVERSION < '4.0.0':
            outfile.setCaption("Output File Selection")
            filterlist = 'Specfile MCA  *.mca\nSpecfile Scan *.dat\nRaw ASCII  *.txt'
            if MATPLOTLIB:
                filterlist += '\nGraphics EPS *.eps\nGraphics PNG *.png'
                if not self.peaksSpectrumButton.isChecked():
                    filterlist += '\nB/WGraphics EPS *.eps\nB/WGraphics PNG *.png'
            outfile.setFilters(filterlist)
            outfile.setMode(outfile.AnyFile)
            outfile.setDir(self.lastInputDir)
            ret = outfile.exec_loop()
        else:
            outfile.setWindowTitle("Output File Selection")
            strlist = qt.QStringList()
            format_list = ['Specfile MCA  *.mca','Specfile Scan *.dat','Raw ASCII  *.txt']
            if MATPLOTLIB:
                format_list.append('Graphics PNG *.png')
                format_list.append('Graphics EPS *.eps')
                format_list.append('Graphics SVG *.svg')
                if not self.peaksSpectrumButton.isChecked():
                    format_list.append('B/WGraphics PNG *.png')
                    format_list.append('B/WGraphics EPS *.eps')
                    format_list.append('B/WGraphics SVG *.svg')
            for f in format_list:
                strlist.append(f)
            outfile.setFilters(strlist)

            outfile.setFileMode(outfile.AnyFile)
            outfile.setAcceptMode(qt.QFileDialog.AcceptSave)
            outfile.setDirectory(self.lastInputDir)
            ret = outfile.exec_()

        if ret:
            filterused = str(outfile.selectedFilter()).split()
            filedescription = filterused[0]
            filetype  = filterused[1]
            extension = filterused[2]
            if QTVERSION < '4.0.0':
                outstr=str(outfile.selectedFile())
            else:
                outstr=str(outfile.selectedFiles()[0])
            try:            
                outputDir  = os.path.dirname(outstr)
                self.lastInputDir   = outputDir
                PyMcaDirs.outputDir = outputDir
            except:
                outputDir  = "."
            #self.outdir = outputDir
            try:            
                outputFile = os.path.basename(outstr)
            except:
                outputFile  = outstr
            outfile.close()
            del outfile
        else:
            outfile.close()
            del outfile
            return
        #always overwrite for the time being
        if len(outputFile) < len(extension[1:]):
            outputFile += extension[1:]
        elif outputFile[-4:] != extension[1:]:
            outputFile += extension[1:]
        specFile = os.path.join(outputDir, outputFile)
        try:
            os.remove(specFile)
        except:
            pass
        systemline = os.linesep
        os.linesep = '\n'
        try:
            if MATPLOTLIB:
                if filetype in ['EPS', 'PNG', 'SVG']:
                    size = (6, 3) #in inches
                    logy = self._logY
                    if filedescription == "B/WGraphics":
                        bw = True
                    else:
                        bw = False
                    if self.peaksSpectrumButton.isChecked():  legends = True
                    elif 'ymatrix' in fitresult['result'].keys(): legends = False
                    else: legends = False
                    mtplt = PyMcaMatplotlibSave.PyMcaMatplotlibSave(size=size,
                                                                    logy=logy,
                                                                    legends=legends,
                                                                    bw = bw)
                                                                    
                    if self._energyAxis:
                        x = fitresult['result']['energy']
                    else:
                        x = fitresult['result']['xdata']
                    xmin, xmax = self.graph.getx1axislimits()
                    ymin, ymax = self.graph.gety1axislimits()
                    mtplt.setLimits(xmin, xmax, ymin, ymax)
                    index = Numeric.nonzero((xmin <= x) & (x <= xmax))
                    x = Numeric.take(x, index)
                    if bw:
                        mtplt.addDataToPlot( x,
                                Numeric.take(fitresult['result']['ydata'],index),
                                legend='data',
                                color='k',linestyle=':', linewidth=1.5, markersize=3)
                    else:
                        mtplt.addDataToPlot( x,
                                Numeric.take(fitresult['result']['ydata'],index),
                                legend='data',
                                linewidth=1)

                    mtplt.addDataToPlot( x,
                                Numeric.take(fitresult['result']['yfit'],index),
                                legend='fit',
                                linewidth=1.5)
                    if not self.peaksSpectrumButton.isChecked():
                        mtplt.addDataToPlot( x,
                                    Numeric.take(fitresult['result']['continuum'],index),
                                    legend='bck', linewidth=1.5)
                    if self.top.sumbox.isChecked():
                        mtplt.addDataToPlot( x,
                                Numeric.take(fitresult['result']['pileup']+\
                                             fitresult['result']['continuum'],index),
                                             legend="pile up",
                                             linewidth=1.5)
                    if 'ymatrix' in fitresult['result'].keys():
                        mtplt.addDataToPlot( x,
                                Numeric.take(fitresult['result']['ymatrix'],index),
                                legend='matrix',
                                linewidth=1.5)
                    if self.peaksSpectrumButton.isChecked():
                        for group in fitresult['result']['groups']:
                            label = 'y'+group
                            if label in fitresult['result'].keys():
                                mtplt.addDataToPlot( x,
                                    Numeric.take(fitresult['result'][label],index),
                                                legend=group,
                                                linewidth=1.5)
                    mtplt.plotLegends()
                    if self._energyAxis:
                        mtplt.setXLabel('Energy (keV)')
                    else:
                        mtplt.setXLabel('Channel')
                    mtplt.setYLabel('Counts')
                    mtplt.saveFile(specFile)
                    return
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Matplotlib or Input Output Error: %s" % (sys.exc_info()[1]))
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        try:
            file=open(specFile,'wb')
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
            if filetype == 'ASCII':
                keys = fitresult['result'].keys()
                for i in range(len(fitresult['result']['ydata'])):
                    file.write("%.7g  %.7g   %.7g  %.7g  %.7g  %.7g" % (fitresult['result']['xdata'][i],
                                       fitresult['result']['energy'][i],
                                       fitresult['result']['ydata'][i],
                                       fitresult['result']['yfit'][i],
                                       fitresult['result']['continuum'][i],
                                       fitresult['result']['pileup'][i]))
                    if 'ymatrix' in fitresult['result'].keys():
                        file.write("  %.7g" %  fitresult['result']['ymatrix'][i])
                    for group in fitresult['result']['groups']:
                        label = 'y'+group
                        if label in keys:
                            file.write("  %.7g" %  fitresult['result'][label][i])
                    file.write("\n")            
                file.close()
                return
            #header is almost common to specfile and mca
            file.write("#F %s\n" % specFile)
            file.write("#D %s\n"%(time.ctime(time.time())))
            file.write("\n")
            legend = str(self.headerLabel.text()).strip('<b>')
            legend.strip('<\b>')
            file.write("#S 1 %s\n" % legend)
            file.write("#D %s\n"%(time.ctime(time.time())))
            i = 0
            for parameter in fitresult['result']['parameters']:
                file.write("#U%d %s %.7g +/- %.3g\n" % (i, parameter,
                                                     fitresult['result']['fittedpar'][i],
                                                     fitresult['result']['sigmapar'][i]))
                i+=1
            if filetype == 'Scan':
                keys = fitresult['result'].keys()
                labelline= "#L channel  Energy  counts  fit  continuum  pileup"
                if 'ymatrix' in keys:
                    nlabels = 7
                    labelline += "ymatrix"
                else:
                    nlabels = 6

                for group in fitresult['result']['groups']:
                    label = 'y'+group
                    if label in keys:
                        nlabels += 1
                        labelline += '  '+group
                file.write("#N %d\n" % nlabels)
                file.write(labelline)
                file.write("\n")
                for i in range(len(fitresult['result']['ydata'])):
                    file.write("%.7g  %.7g  %.7g  %.7g  %.7g  %.7g" % (fitresult['result']['xdata'][i],
                                       fitresult['result']['energy'][i],
                                       fitresult['result']['ydata'][i],
                                       fitresult['result']['yfit'][i],
                                       fitresult['result']['continuum'][i],
                                       fitresult['result']['pileup'][i]))
                    if 'ymatrix' in keys:
                        file.write("  %.7g" % fitresult['result']['ymatrix'][i])
                    for group in fitresult['result']['groups']:
                        label = 'y'+group
                        if label in keys:
                            file.write("  %.7g" %  fitresult['result'][label][i])
                    file.write("\n")
            else:
                file.write("#@MCA %16C\n")
                file.write("#@CHANN %d %d %d 1\n" %  (len(fitresult['result']['ydata']),
                                                        fitresult['result']['xdata'][0],
                                                        fitresult['result']['xdata'][-1]))
                zeroindex = fitresult['result']['parameters'].index('Zero')
                gainindex = fitresult['result']['parameters'].index('Gain')
                file.write("#@CALIB %.7g %.7g 0.0\n" % (fitresult['result']['fittedpar'][zeroindex],
                                                        fitresult['result']['fittedpar'][gainindex]))
                file.write(self.array2SpecMca(fitresult['result']['ydata']))
                file.write(self.array2SpecMca(fitresult['result']['yfit']))
                file.write(self.array2SpecMca(fitresult['result']['continuum']))
                file.write(self.array2SpecMca(fitresult['result']['pileup']))
                keys = fitresult['result'].keys()
                if 'ymatrix' in keys:
                    file.write(self.array2SpecMca(fitresult['result']['ymatrix']))
                for group in fitresult['result']['groups']:
                    label = 'y'+group
                    if label in keys:
                        file.write(self.array2SpecMca(fitresult['result'][label]))    
            file.write("\n")
            file.close()
        except:
            os.linesep = systemline
            raise
        return

    def array2SpecMca(self, data):
        """ Write a python array into a Spec array.
            Return the string containing the Spec array
        """
        tmpstr = "@A "
        length = len(data)
        for idx in range(0, length, 16):
            if idx+15 < length:
                for i in range(0,16):
                    tmpstr += "%.4f " % data[idx+i]
                if idx+16 != length:
                    tmpstr += "\\"
            else:
                for i in range(idx, length):
                    tmpstr += "%.4f " % data[i]
            tmpstr += "\n"
        return tmpstr


class Top(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        qt.QWidget.__init__(self,parent)
        self.layout= qt.QHBoxLayout(self)
        self.build()
        
    def build(self):
        self.__w=qt.QWidget(self)
        w = self.__w
        self.layout.addWidget(w)
        wlayout = qt.QGridLayout(w)
        wlayout.setSpacing(5)
        #function
        FunLabel = qt.QLabel(w)
        FunLabel.setText(str("Function"))
        if QTVERSION < '4.0.0':
            self.FunComBox = qt.QComboBox(0,w,"FunComBox")
            self.FunComBox.insertStrList(["Mca Hypermet"])
        else:
            self.FunComBox = qt.QComboBox(w)
            self.FunComBox.insertItem(0, "Mca Hypermet")
        wlayout.addWidget(FunLabel,0,0)
        wlayout.addWidget(self.FunComBox,0,1)
        #background
        BkgLabel = qt.QLabel(w)
        BkgLabel.setText(str("Background"))
        if QTVERSION < '4.0.0':
            self.BkgComBox = qt.QComboBox(0,w,"BkgComBox")
            self.BkgComBox.insertStrList(['No Background',
                                          'Constant',
                                          'Linear',
                                          'Parabolic',
                                          'Linear Polynomial',
                                          'Exp. Polynomial'])
        else:
            self.BkgComBox = qt.QComboBox(w)
            options = ['No Background',
                       'Constant',
                       'Linear',
                       'Parabolic',
                       'Linear Polynomial',
                       'Exp. Polynomial']
            for item in options:
                self.BkgComBox.insertItem(options.index(item), item)            

        self.connect(self.BkgComBox,
                     qt.SIGNAL("activated(int)"),self.mysignal)
        #                        qt.SIGNAL("activated(const QString &)"),self.bkgevent)
        wlayout.addWidget(BkgLabel,1,0)
        wlayout.addWidget(self.BkgComBox,1,1)
        dummy = qt.QWidget(self)
        dummy.setMinimumSize(20,0)
        self.layout.addWidget(dummy)
        self.layout.addWidget(HorizontalSpacer(self))
        
        #the checkboxes
        if 0:
             w1 = qt.QVBox(self)
             self.WeightCheckBox = qt.QCheckBox(w1)
             self.WeightCheckBox.setText(str("Weight"))
             self.McaModeCheckBox = qt.QCheckBox(w1)
             self.McaModeCheckBox.setText(str("Mca Mode"))

        # Flags
        f       = qt.QWidget(self)
        self.layout.addWidget(f)
        f.layout= qt.QGridLayout(f)
        f.layout.setSpacing(5)
        flagsoffset = -1
        coffset     = 0
        #hyplabel = qt.QLabel(f)
        #hyplabel.setText(str("<b>%s</b>" % 'FLAGS'))
        self.stbox = qt.QCheckBox(f)
        self.stbox.setText('Short Tail')
        self.ltbox = qt.QCheckBox(f)
        self.ltbox.setText('Long Tail')        
        self.stepbox = qt.QCheckBox(f)
        self.stepbox.setText('Step Tail')        
        self.escapebox = qt.QCheckBox(f)
        self.escapebox.setText('Escape')        
        self.sumbox = qt.QCheckBox(f)
        self.sumbox.setText('Pile-up')        
        self.stripbox = qt.QCheckBox(f)
        self.stripbox.setText('Strip Back.')
        #checkbox connections
        self.connect(self.stbox,qt.SIGNAL("clicked()"),     self.mysignal)
        self.connect(self.ltbox,qt.SIGNAL("clicked()"),     self.mysignal)
        self.connect(self.stepbox,qt.SIGNAL("clicked()"),   self.mysignal)
        self.connect(self.escapebox,qt.SIGNAL("clicked()"), self.mysignal)
        self.connect(self.sumbox,qt.SIGNAL("clicked()"),    self.mysignal)
        self.connect(self.stripbox,qt.SIGNAL("clicked()"),  self.mysignal)
        #f.layout.addWidget(hyplabel,flagsoffset,coffset +1) 
        f.layout.addWidget(self.stbox,flagsoffset+1,coffset +0)              
        f.layout.addWidget(self.ltbox,flagsoffset+1,coffset +1)              
        f.layout.addWidget(self.stepbox,flagsoffset+1,coffset +2)        
        f.layout.addWidget(self.escapebox,flagsoffset+2,coffset +0)              
        f.layout.addWidget(self.sumbox,flagsoffset+2,coffset +1)              
        f.layout.addWidget(self.stripbox,flagsoffset+2,coffset +2)
        self.layout.addWidget(HorizontalSpacer(self))
        
        #buttons
        g = qt.QWidget(self)
        self.layout.addWidget(g)
        glayout = qt.QGridLayout(g)
        glayout.setSpacing(5)
        self.configureButton = qt.QPushButton(g)
        self.configureButton.setText(str("Configure"))
        self.printButton = qt.QPushButton(g)
        self.printButton.setText(str("Tools"))
        glayout.addWidget(self.configureButton,0,0)
        glayout.addWidget(self.printButton,1,0)

    def setParameters(self,ddict=None):
        if ddict == None: ddict = {}
        if ddict.has_key('hypermetflag'):
            hypermetflag = ddict['hypermetflag']
            g_term    =  hypermetflag & 1
            st_term   = (hypermetflag >>1) & 1
            lt_term   = (hypermetflag >>2) & 1
            step_term = (hypermetflag >>3) & 1
            if st_term:
                self.stbox.setChecked(1)
            else:
                self.stbox.setChecked(0)
            if lt_term:
                self.ltbox.setChecked(1)
            else:
                self.ltbox.setChecked(0)
            if step_term:
                self.stepbox.setChecked(1)
            else:
                self.stepbox.setChecked(0)
        
        if ddict.has_key('sumflag'):
            if ddict['sumflag'] == 1:
                self.sumbox.setChecked(1)
            else:
                self.sumbox.setChecked(0)

        if ddict.has_key('stripflag'):
            if ddict['stripflag'] == 1:
                self.stripbox.setChecked(1)
            else:
                self.stripbox.setChecked(0)

        if ddict.has_key('escapeflag'):
            if ddict['escapeflag'] == 1:
                self.escapebox.setChecked(1)
            else:
                self.escapebox.setChecked(0)
        
        if ddict.has_key('continuum'):
            if QTVERSION < '4.0.0':
                self.BkgComBox.setCurrentItem(ddict['continuum'])
            else:
                self.BkgComBox.setCurrentIndex(ddict['continuum'])
         
    def getParameters(self):
        ddict={}
        ddict['hypermetflag'] = 1
        if self.stbox.isChecked():
            ddict['hypermetflag'] += 2 
        if self.ltbox.isChecked():
            ddict['hypermetflag'] += 4 
        
        if self.stepbox.isChecked():
            ddict['hypermetflag'] += 8
            
        if self.sumbox.isChecked():
            ddict['sumflag'] = 1
        else:
            ddict['sumflag'] = 0

        if self.stripbox.isChecked():
            ddict['stripflag'] = 1
        else:
            ddict['stripflag'] = 0

        if self.escapebox.isChecked():
            ddict['escapeflag'] = 1
        else:
            ddict['escapeflag'] = 0
        if QTVERSION < '4.0.0':
            ddict['continuum'] = self.BkgComBox.currentItem()
        else:
            ddict['continuum'] = self.BkgComBox.currentIndex()
        return ddict

    def mysignal(self,*var):
        ddict = self.getParameters()
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('TopSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('TopSignal'),(ddict))
    
class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

class Line(qt.QFrame):
    def __init__(self, parent=None, name="Line", fl=0, info=None):
        qt.QFrame.__init__(self, parent)
        self.info = info        
        self.setFrameShape(qt.QFrame.HLine)
        self.setFrameShadow(qt.QFrame.Sunken)
        self.setFrameShape(qt.QFrame.HLine)

        
    def mouseDoubleClickEvent(self,event):
        if DEBUG:
            print "Double Click Event"
        dict={}
        dict['event']="DoubleClick"
        dict['data'] = event
        dict['info'] = self.info
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("LineDoubleClickEvent"),(dict,))
        else:
            self.emit(qt.SIGNAL("LineDoubleClickEvent"), dict)
        
class SimpleThread(qt.QThread):
    def __init__(self, function = None, kw = None):
        if kw is None:kw={}
        qt.QThread.__init__(self)
        self._function = function        
        self._kw       = kw
        self._result   = None
        self._expandParametersDict = False
    
    def run(self):
        try:
            if self._expandParametersDict:
                self._result = self._function(**self._kw)
            else:
                self._result = self._function(self._kw)
        except:
            self._result = ("Exception",) + sys.exc_info()
        
      
class McaGraphWindow(qt.QWidget):
    def __init__(self, parent=None, name="Toolbar",fl=0):
        qt.QWidget.__init__(self,parent)
        layout = qt.QVBoxLayout(self)
        
        self.toolbar = qt.QWidget(self)
        self.toolbar.layout = qt.QHBoxLayout(self.toolbar)
        self.graph   = QtBlissGraph.QtBlissGraph(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.graph)
        self.initToolBar()
        if QTVERSION < '4.0.0':
            self.connect(self.graph,
                         qt.PYSIGNAL('QtBlissGraphSignal'),
                         self.__graphSignal)
        else:
            self.connect(self.graph,
                         qt.SIGNAL('QtBlissGraphSignal'),
                         self.__graphSignal)

        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)
        if DEBUG: print "printPreview id = ", id(self.printPreview)
        

                     
    def __roiSlot(self,dict=None):
        if dict is None:
            self.roimarkers = [-1, -1]
            self.roilist    = ['Integral']
            self.roidict    = {}
            self.roidict['Integral'] = {'type':'Default',
                                        'from':0,'to':-1}
            self.currentroi = None
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                       roidict=self.roidict)
        elif dict['event'] == 'AddROI':
            xmin,xmax = self.graph.getx1axislimits()
            fromdata = xmin+ 0.25 * (xmax - xmin)
            todata   = xmin+ 0.75 * (xmax - xmin)
            if self.roimarkers[0] < 0:
                self.roimarkers[0] = self.graph.insertx1marker(fromdata,1.1,
                                        label = 'ROI min')
            if self.roimarkers[1] < 0:
                self.roimarkers[1] = self.graph.insertx1marker(todata,1.1,
                                        label = 'ROI max')
            self.graph.setmarkercolor(self.roimarkers[0],'blue')
            self.graph.setmarkercolor(self.roimarkers[1],'blue')
            self.graph.replot()
            #if self.roilist is None:
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            nrois = len(self.roilist)
            for i in range(nrois):
                i += 1
                newroi = "newroi %d" % i
                if newroi not in self.roilist:
                    break  
            self.roilist.append(newroi)
            self.roidict[newroi] = {}
            self.roidict[newroi]['type']    = str(self.graph.xlabel())
            self.roidict[newroi]['from']    = fromdata
            self.roidict[newroi]['to']      = todata
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=newroi)
            self.currentroi = newroi

        elif dict['event'] == 'DelROI':
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            self.currentroi = self.roidict.keys()[0]
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=self.currentroi)
        
        elif dict['event'] == 'ResetROI':
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            self.currentroi = self.roidict.keys()[0]
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=self.currentroi)
            self.graph.clearmarkers()
        elif dict['event'] == 'ActiveROI':
            print "ActiveROI event"
            pass
        ndict = {}
        ndict['event']  = "SetActiveCurveEvent"
        self.__graphSignal(ndict)

    def initIcons(self):
		if QTVERSION > '4.0.0':qt.QIconSet = qt.QIcon
		self.normalIcon	= qt.QIconSet(qt.QPixmap(IconDict["normal"]))
		self.zoomIcon	= qt.QIconSet(qt.QPixmap(IconDict["zoom"]))
		self.roiIcon	= qt.QIconSet(qt.QPixmap(IconDict["roi"]))
		self.peakIcon	= qt.QIconSet(qt.QPixmap(IconDict["peak"]))

		self.zoomResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["zoomreset"]))
		self.roiResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["roireset"]))
		self.peakResetIcon	= qt.QIconSet(qt.QPixmap(IconDict["peakreset"]))
		self.refreshIcon	= qt.QIconSet(qt.QPixmap(IconDict["reload"]))

		self.logxIcon	= qt.QIconSet(qt.QPixmap(IconDict["logx"]))
		self.logyIcon	= qt.QIconSet(qt.QPixmap(IconDict["logy"]))
		self.energyIcon	= qt.QIconSet(qt.QPixmap(IconDict["energy"]))
		self.xAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["xauto"]))
		self.yAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["yauto"]))
		self.fitIcon	= qt.QIconSet(qt.QPixmap(IconDict["fit"]))
		self.deriveIcon	= qt.QIconSet(qt.QPixmap(IconDict["derive"]))
		self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))
		self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
		self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))

        
    def initToolBar(self):
        toolbar = self.toolbar
        self.initIcons()
        # AutoScale
        self._addToolButton(self.zoomResetIcon,
                            self.graph.ResetZoom,
                            'Auto-Scale the Graph')

        #y Autoscale
        self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle=1,
                            state=1)
        
        #x Autoscale
        self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle=True,
                            state=1)
        # Logarithmic
        self._addToolButton(self.logyIcon,
                            self._logIconSignal,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle=True)

        # Energy
        self._addToolButton(self.energyIcon,
                            self._energyIconSignal,
                            'Toggle Energy Axis (On/Off)',
                            toggle=True)
        # Fit
        self._addToolButton(self.fitIcon,
                            self._fitIconSignal,
                            'Fit Again!')
        
        # ROI
        self.roiwidget = None
        """
        self._initRoi()
        self.roiwidget.hide()
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.roiIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self._roiIconSignal)
        qt.QToolTip.add(tb,'Show ROI Widget')
        """
        
        #save
        self._addToolButton(self.saveIcon,
                            self._saveIconSignal,
                            'Save Curves')
         
        # Search
        """
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.searchIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.peaksearch)
        qt.QToolTip.add(tb,'Clear Peak Table and Search Peaks') 
        # Marker
        self.markerButton      = qt.QToolButton(toolbar)
        self.markerButton.setIconSet(self.normalIcon)
        self.markerButton.setToggleButton(1)
        self.connect(self.markerButton,qt.SIGNAL('clicked()'),self.__peakmarkermode)
        qt.QToolTip.add(self.markerButton,'Allow Right-Click Peak Selection from Graph') 
        """
        self.toolbar.layout.addWidget(HorizontalSpacer(toolbar))
        label=qt.QLabel(toolbar)
        #label.setText('<b>Channel:</b>')
        label.setText('<b>X:</b>')
        self.xpos = qt.QLineEdit(toolbar)
        self.xpos.setText('------')
        self.xpos.setReadOnly(1)
        self.xpos.setFixedWidth(self.xpos.fontMetrics().width('#########'))
        self.toolbar.layout.addWidget(label)
        self.toolbar.layout.addWidget(self.xpos)
        
        label=qt.QLabel(toolbar)
        label.setText('<b>Y:</b>')
        self.ypos = qt.QLineEdit(toolbar)
        self.ypos.setText('------')
        self.ypos.setReadOnly(1)
        self.ypos.setFixedWidth(self.ypos.fontMetrics().width('############'))
        self.toolbar.layout.addWidget(label)
        self.toolbar.layout.addWidget(self.ypos)
        
        """
        label=qt.QLabel(toolbar)
        label.setText('<b>Energy:</b>')
        self.epos = qt.QLineEdit(toolbar)
        self.epos.setText('------')
        self.epos.setReadOnly(1)
        self.epos.setFixedWidth(self.epos.fontMetrics().width('########'))
        """
        self.toolbar.layout.addWidget(HorizontalSpacer(toolbar))
        # ---print
        if 0:
            self._addToolButton(self.printIcon,
                            self.graph.printps,
                            'Prints the Graph')
        else:
            tb = self._addToolButton(self.printIcon,
                    self.printGraph,
                    'Print the graph')
            toolbar.layout.addWidget(tb)
            
    def printGraph(self):
        pixmap = qt.QPixmap.grabWidget(self.graph)
        self.printPreview.addPixmap(pixmap)
        if self.printPreview.isHidden():
            self.printPreview.show()
        if QTVERSION < '4.0.0':
            self.printPreview.raiseW()
        else:
            self.printPreview.raise_()

    def _addToolButton(self, icon, action, tip, toggle=None, state=None):
            toolbar = self.toolbar
            tb      = qt.QToolButton(toolbar)            
            if QTVERSION < '4.0.0':
                tb.setIconSet(icon)
                qt.QToolTip.add(tb,tip) 
                if toggle is not None:
                    if toggle:
                        tb.setToggleButton(1)
                        if state is not None:
                            if state:
                                tb.setState(qt.QButton.On)

            else:
                tb.setIcon(icon)
                tb.setToolTip(tip)
                if toggle is not None:
                    if toggle:
                        tb.setCheckable(1)
                        if state is not None:
                            if state:
                                tb.setChecked(state)
                        else:
                            tb.setChecked(False)
            self.toolbar.layout.addWidget(tb)
            self.connect(tb,qt.SIGNAL('clicked()'), action)
            return tb

    def _yAutoScaleToggle(self):
        if self.graph.yAutoScale:
            self.graph.yAutoScale = False
        else:
            self.graph.yAutoScale = True
            
    def _xAutoScaleToggle(self):
        if self.graph.xAutoScale:
            self.graph.xAutoScale = False
        else:
            self.graph.xAutoScale = True

    def _energyIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        dict={}
        dict['event']  = 'EnergyClicked'
        dict['active'] = legend
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaGraphSignal'), (dict))

    def _logIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        dict={}
        dict['event']  = 'LogClicked'
        dict['active'] = legend
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaGraphSignal'), (dict))

    def _fitIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        dict={}
        dict['event']  = 'FitClicked'
        dict['active'] = legend
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaGraphSignal'), (dict))

    def _initRoi(self):
        self.roiwidget = McaROIWidget.McaROIWidget(None)
        if QTVERSION < '4.0.0':
            self.connect(self.roiwidget,qt.PYSIGNAL("McaROIWidgetSignal"),
                     self.__roiSlot)
        else:
            self.connect(self.roiwidget,qt.SIGNAL("McaROIWidgetSignal"),
                     self.__roiSlot)
        self.__roiSlot()
    
    def _roiIconSignal(self):
        if self.roiwidget is None:self._initRoi()
        if self.roiwidget.isHidden():self.roiwidget.show()
        if QTVERSION < '4.0.0':
            self.roiwidget.raiseW()
        else:
            self.roiwidget.raise_()
        legend = self.graph.getactivecurve(justlegend=1)
        dict={}
        dict['event']  = 'RoiClicked'
        dict['active'] = legend
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaGraphSignal'), (dict))

    def _saveIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        dict={}
        dict['event']  = 'SaveClicked'
        dict['active'] = legend
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaGraphSignal'), (dict))

    def __graphSignal(self, dict):
        if dict['event'] == 'MouseAt':
            self.xpos.setText('%.4g' % dict['x'])
            self.ypos.setText('%.5g' % dict['y'])
        elif dict['event'] == 'MouseClick':
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('McaGraphSignal'),(dict,))
            else:
                self.emit(qt.SIGNAL('McaGraphSignal'), (dict))
        elif dict['event'] == "SetActiveCurveEvent":
            legend = None
            if dict.has_key('legend'):
                legend = dict['legend']
            if (legend is not None) and (self.roiwidget is not None):
                legend,x,y = self.graph.getactivecurve()
                #if self.roidict is None:
                self.roilist,self.roidict = self.roiwidget.getroilistanddict()
                if not len(x):
                    #only zeros ...
                    for i in range(len(self.roilist)):
                        key = self.roilist[i]
                        self.roidict[key]['rawcounts'] = 0.0
                        self.roidict[key]['netcounts'] = 0.0
                        #self.roidict[key]['from'  ] = 0.0
                        #self.roidict[key]['to'    ] = 0.0
                else:    
                    for i in range(len(self.roilist)):
                        key = self.roilist[i]
                        if key == self.roilist[0]:
                            #take care of LLD
                            #if len(x) > 3:
                            #    fromdata = x[3]
                            #else:
                            fromdata = x[0]
                            todata   = x[-1]
                        else:
                            fromdata = self.roidict[key]['from']
                            todata   = self.roidict[key]['to']
                        i1 = Numeric.nonzero(x>=fromdata)
                        xw = Numeric.take(x,i1)
                        yw = Numeric.take(y,i1)
                        i1 = Numeric.nonzero(xw<=todata)
                        xw = Numeric.take(xw,i1)
                        yw = Numeric.take(yw,i1)
                        counts = Numeric.sum(yw)
                        self.roidict[key]['rawcounts'] = counts
                        if len(yw):
                            self.roidict[key]['netcounts'] = counts - \
                                                      len(yw) *  0.5 * (yw[0] + yw[-1])
                        else:
                            self.roidict[key]['netcounts'] = 0
                        self.roidict[key]['from'  ] = fromdata
                        self.roidict[key]['to'    ] = todata
                xlabel = self.graph.xlabel()
                #self.roiwidget.setheader(text="%s ROIs of %s" % (xlabel,legend))
                self.roiwidget.setheader(text="ROIs of %s" % (legend))
                self.roiwidget.fillfromroidict(roilist=self.roilist,
                                                roidict=self.roidict)

        
    
    
def test(file='03novs060sum.mca'):
    import specfilewrapper as specfile
    app = qt.QApplication([])
    sf=specfile.Specfile(file)
    scan=sf[0]
    mcadata=scan.mca(1)
    y0= Numeric.array(mcadata)
    x = Numeric.arange(len(y0))*1.0
    demo = McaAdvancedFit()
    xmin = demo.mcafit.config['fit']['xmin']
    xmax = demo.mcafit.config['fit']['xmax']
    demo.setdata(x,y0,xmin=xmin,xmax=xmax,sourcename=file)
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        app.exec_loop()
    else:
        demo.show()
        app.exec_()
        
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) >1:
        file = sys.argv[1]
    else:
        file = '03novs060sum.mca'
    test(file)
