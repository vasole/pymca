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
__revision__ = "$Revision: 1.44 $"
import sys
import os
import time
import QtBlissGraph
qt = QtBlissGraph.qt
qwt = QtBlissGraph.qwt
from Icons import IconDict
import McaControlGUI
import Numeric
import McaAdvancedFit
import DataObject
import copy
import McaCalWidget
import Elements
import McaSimpleFit
import Specfit


DEBUG = 0
QTVERSION = qt.qVersion()

class McaWindow(qt.QMainWindow):
    def __init__(self, parent=None, name="MCA Window", specfit=None,
                         fl=None, **kw):
        if qt.qVersion() < '4.0.0':
            if fl is None: fl = qt.Qt.WDestructiveClose
            qt.QMainWindow.__init__(self, parent, name, fl)
            self.parent = parent
        else:
            qt.QMainWindow.__init__(self, parent)
            if fl is None: fl = qt.Qt.WA_DeleteOnClose
        self.mcawidget = McaWidget(self,**kw)
        self.mcawidget.show()
        self.setCentralWidget(self.mcawidget)

class McaWidget(qt.QWidget):
    def __init__(self, parent=None, name="Mca Window", specfit=None,fl=None,
                     vertical=True, **kw):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent)
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
            self.setWindowTitle(name)
        self.outputdir = None
        """ 
        class McaWidget(qt.QSplitter):
            def __init__(self, parent=None, specfit=None,fl=None,**kw):
                if qt.qVersion() < '4.0.0':
                    if fl is None: fl = qt.Qt.WDestructiveClose
                    self.parent = parent
                else:
                    if fl is None: fl = qt.Qt.WA_DeleteOnClose

                if qt.qVersion() < '4.0.0':
                    qt.QSplitter.__init__(self, parent)
                    self.setOrientation(qt.Qt.Vertical)
                else:
                    qt.QSplitter.__init__(self, parent)
                    self.setOrientation(qt.Qt.Vertical)
        """
        self.calibration = 'None'
        self.calboxoptions = ['None','Original (from Source)','Internal (from Source OR PyMca)']
        self.caldict={}
        self.calwidget   =  None
        self.roilist = None
        self.roidict = None
        self.currentroi = None
        self.roimarkers     = [-1,-1]
        self.elementmarkers = []
        self.peakmarker     = None
        self.dataObjectsDict = {}
        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            self.specfit = specfit
        self.simplefit   = McaSimpleFit.McaSimpleFit(specfit=self.specfit)
        self.advancedfit = McaAdvancedFit.McaAdvancedFit()

        self.build(vertical)
        self.initIcons()
        self.initToolbar()
        self.connections()


    def build(self, vertical=True):
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(1)
        self.mainLayout.setSpacing(1)
        self.splitter = qt.QSplitter(self)
        if vertical:
            self.splitter.setOrientation(qt.Qt.Vertical)
        else:
            self.splitter.setOrientation(qt.Qt.Horizontal)
        
        #the box to contain the graphics
        self.graphBox = qt.QWidget(self.splitter)
        self.graphBoxlayout = qt.QVBoxLayout(self.graphBox)
        self.graphBoxlayout.setMargin(0)
        self.graphBoxlayout.setSpacing(0)
        #self.layout.addWidget(self.graphBox)
        
        self.toolbar  = qt.QWidget(self.graphBox)
        self.toolbar.layout  = qt.QHBoxLayout(self.toolbar)
        self.toolbar.layout.setMargin(0)
        self.toolbar.layout.setSpacing(0)
        self.graphBoxlayout.addWidget(self.toolbar)
        
        self.graph    = QtBlissGraph.QtBlissGraph(self.graphBox,uselegendmenu=1)
        self.graph.xlabel('Channel')
        self.graph.ylabel('Counts')
        self.graph.canvas().setMouseTracking(1)
        self.graph.setCanvasBackground(qt.Qt.white)
        self.graphBoxlayout.addWidget(self.graph)
            
        #the box to contain the control widget(s)
        self.controlbox = qt.QWidget(self.splitter) 
        self.controlboxlayout = qt.QVBoxLayout(self.controlbox)
        self.controlboxlayout.setMargin(0)
        self.controlboxlayout.setSpacing(0)
        self.control    = McaControlGUI.McaControlGUI(self.controlbox)
        self.controlboxlayout.addWidget(self.control)

        self.roiwidget  = self.control.roiwidget
        if not vertical:
            table = self.roiwidget.mcaROITable
            rheight = table.horizontalHeader().sizeHint().height()
            table.setMinimumHeight(12 * rheight)

        if qt.qVersion() < '4.0.0':
            self.fitmenu = qt.QPopupMenu()
            self.fitmenu.insertItem(qt.QString("Simple"),    self.mcasimplefitsignal)
            self.fitmenu.insertItem(qt.QString("Advanced") , self.mcaadvancedfitsignal)
        else:
            self.fitmenu = qt.QMenu()
            self.fitmenu.addAction(qt.QString("Simple"),    self.mcasimplefitsignal)
            self.fitmenu.addAction(qt.QString("Advanced") , self.mcaadvancedfitsignal)


        if QTVERSION < '4.0.0':
            self.splitter.moveToLast(self.graphBox)
            self.splitter.moveToLast(self.controlbox)
        else:
            self.splitter.insertWidget(0, self.graphBox)
            self.splitter.insertWidget(1, self.controlbox)
            
        self.mainLayout.addWidget(self.splitter)

    def connections(self):
        if QTVERSION < '4.0.0':
            self.connect(self.control,    qt.PYSIGNAL('McaControlGUISignal') ,self.__anasignal)
            #self.connect(self.scanfit,    qt.PYSIGNAL('ScanFitSignal') , self.__anasignal)
            self.connect(self.simplefit,  qt.PYSIGNAL('McaSimpleFitSignal') , self.__anasignal)
            self.connect(self.advancedfit,qt.PYSIGNAL('McaAdvancedFitSignal') , self.__anasignal)
            #self.connect(self.scanwindow, qt.PYSIGNAL('ScanWindowSignal') ,   self.__anasignal)
            #self.connect(self.scanwindow, qt.PYSIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)
            self.connect(self.graph,      qt.PYSIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)
        else:
            self.connect(self.control,    qt.SIGNAL('McaControlGUISignal') ,self.__anasignal)
            #self.connect(self.scanfit,    qt.SIGNAL('ScanFitSignal') , self.__anasignal)
            self.connect(self.simplefit,  qt.SIGNAL('McaSimpleFitSignal') , self.__anasignal)
            self.connect(self.advancedfit,qt.SIGNAL('McaAdvancedFitSignal') , self.__anasignal)
            #self.connect(self.scanwindow, qt.SIGNAL('ScanWindowSignal') ,   self.__anasignal)
            #self.connect(self.scanwindow, qt.SIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)
            self.connect(self.graph,      qt.SIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)

    def initIcons(self):
        if qt.qVersion() > '4.0.0':qt.QIconSet = qt.QIcon
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
        self.xAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["xauto"]))
        self.yAutoIcon	= qt.QIconSet(qt.QPixmap(IconDict["yauto"]))
        self.fitIcon	= qt.QIconSet(qt.QPixmap(IconDict["fit"]))
        self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))
        self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
        self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))

    def initToolbar(self):
        toolbar = self.toolbar
        # AutoScale
        self._addToolButton(self.zoomResetIcon,
                            self.graph.ResetZoom,
                            'Auto-Scale the Graph')
        #y Autoscale
        tb = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True)
        if qt.qVersion() < '4.0.0':
            tb.setState(qt.QButton.On)
        else:
            tb.setChecked(True)
            tb.setDown(True)
        self.ytb = tb
        #x Autoscale
        tb = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True)
        self.xtb = tb
        if qt.qVersion() < '4.0.0':
            tb.setState(qt.QButton.On)
        else:
            tb.setChecked(True)
            tb.setDown(True)

        # Logarithmic
        tb = self._addToolButton(self.logyIcon,
                            self.graph.ToggleLogY,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle = True)       
        # Fit
        tb = self._addToolButton(self.fitIcon,
                                 self.__fitsignal,
                                 'Fit Active Curve')
        
        #save
        tb = self._addToolButton(self.saveIcon,
                                 self.__saveIconSignal,
                                 'Save Active Curve')
         
        toolbar.layout.addWidget(HorizontalSpacer(toolbar))
        label=qt.QLabel(toolbar)
        #label.setText('<b>Channel:</b>')
        label.setText('<b>X:</b>')
        toolbar.layout.addWidget(label)

        self.xpos = qt.QLineEdit(toolbar)
        self.xpos.setText('------')
        self.xpos.setReadOnly(1)
        self.xpos.setFixedWidth(self.xpos.fontMetrics().width('########'))
        toolbar.layout.addWidget(self.xpos)


        label=qt.QLabel(toolbar)
        label.setText('<b>Y:</b>')
        toolbar.layout.addWidget(label)

        self.ypos = qt.QLineEdit(toolbar)
        self.ypos.setText('------')
        self.ypos.setReadOnly(1)
        self.ypos.setFixedWidth(self.ypos.fontMetrics().width('#########'))
        toolbar.layout.addWidget(self.ypos)
        """
        label=qt.QLabel(toolbar)
        label.setText('<b>Energy:</b>')
        self.epos = qt.QLineEdit(toolbar)
        self.epos.setText('------')
        self.epos.setReadOnly(1)
        self.epos.setFixedWidth(self.epos.fontMetrics().width('########'))
        """
        toolbar.layout.addWidget(HorizontalSpacer(toolbar))

        # ---print
        if 0:
            tb      = qt.QToolButton(toolbar)
            tb.setIconSet(self.printIcon)
            self.connect(tb,qt.SIGNAL('clicked()'),self.graph.printps)
            qt.QToolTip.add(tb,'Prints the Graph')
            toolbar.layout.addWidget(tb)
            
    def _addToolButton(self, icon, action, tip, toggle=None):
            toolbar = self.toolbar
            tb      = qt.QToolButton(toolbar)            
            if qt.qVersion() < '4.0.0':
                tb.setIconSet(icon)
                qt.QToolTip.add(tb,tip) 
                if toggle is not None:
                    if toggle:
                        tb.setToggleButton(1)
            else:
                tb.setIcon(icon)
                tb.setToolTip(tip)
                if toggle is not None:
                    if toggle:
                        tb.setCheckable(1)
            self.toolbar.layout.addWidget(tb)
            self.connect(tb,qt.SIGNAL('clicked()'), action)
            return tb

    def _yAutoScaleToggle(self):
        if self.graph.yAutoScale:
            self.graph.yAutoScale = False
            self.ytb.setDown(False)
            if QTVERSION < '4.0.0':
                self.ytb.setState(qt.QButton.Off)
            else:
                self.ytb.setChecked(False)
        else:
            self.graph.yAutoScale = True
            self.ytb.setDown(True)
            if QTVERSION < '4.0.0':
                self.ytb.setState(qt.QButton.On)
            else:
                self.ytb.setChecked(True)
            
    def _xAutoScaleToggle(self):
        if self.graph.xAutoScale:
            self.graph.xAutoScale = False
            self.xtb.setDown(False)
            if QTVERSION < '4.0.0':
                self.xtb.setState(qt.QButton.Off)
            else:
                self.xtb.setChecked(False)
        else:
            self.graph.xAutoScale = True
            self.xtb.setDown(True)
            if QTVERSION < '4.0.0':
                self.xtb.setState(qt.QButton.On)
            else:
                self.xtb.setChecked(True)


    def setDispatcher(self, w):
        """
        OBSOLETE: Prefer to make the connections at the parent level 
        even better replace the connections by direct calls from the parent.
        """
        if QTVERSION < '4.0.0':
            self.connect(w, qt.PYSIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.PYSIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.PYSIGNAL("replaceSelection"),
                             self._replaceSelection)
        else:
            self.connect(w, qt.SIGNAL("addSelection"),
                             self._addSelection)
            self.connect(w, qt.SIGNAL("removeSelection"),
                             self._removeSelection)
            self.connect(w, qt.SIGNAL("replaceSelection"),
                             self._replaceSelection)

    def peaksearch(self):
        if DEBUG:
            print "Peak search called"
        #get current plot limits
        xmin,xmax=self.graph.getx1axislimits()
        #set the data into specfit
        self.specfit.setdata(x=self.dict['x'],y=self.dict['y'],xmin=xmin,xmax=xmax)
        pars = self.specfit.configure()
        if pars["AutoFwhm"]:
            fwhm = self.specfit.guess_fwhm()
        else:
            fwhm = pars["FwhmPoints"]
        if pars["AutoYscaling"]:
            yscaling = self.specfit.guess_yscaling()
        else:
            yscaling = pars["Yscaling"]
        sensitivity  = pars["Sensitivity"]
        ysearch = self.specfit.ydata*yscaling
        peaksidx=SpecfitFuns.seek(ysearch,1,len(ysearch),
                                    fwhm,
                                    sensitivity)
        self.foundpeaks = []
        self.graph.clearmarkers()
        self.__destroylinewidgets()
        """
        self.peaktable.setNumRows(0)
        """
        i = 0
        for idx in peaksidx:
            self.foundpeaks.append(self.specfit.xdata[int(idx)])            
            self.graph.insertx1marker(self.specfit.xdata[int(idx)],self.specfit.ydata[int(idx)])
            i += 1
        self.graph.replot()

    def __peakmarkermode(self):
        if self.markermode:
            #enable zoom back
            self.graph.enablezoomback()
            #disable marking
            qt.QToolTip.add(self.markerButton,'Allow Right-click Peak Selection from Graph') 
            self.graph.disablemarkermode()
            self.graph.canvas().setCursor(qt.QCursor(qt.QCursor.CrossCursor))
            #save the cursor
            self.markermode = 0
        else:
            #disable zoomback
            self.graph.disablezoomback()
            #enable marking
            self.graph.enablemarkermode()
            qt.QToolTip.add(self.markerButton,'Disable Right-click Peak Selection from Graph') 
            self.markermode = 1
            self.nomarkercursor = self.graph.canvas().cursor().shape()
            self.graph.canvas().setCursor(qt.QCursor(qt.QCursor.PointingHandCursor))
            
        self.markerButton.setOn(self.markermode == 1)
    
    def __fitsignal(self):
        if QTVERSION < '4.0.0':
            self.fitmenu.exec_loop(self.cursor().pos())
        else:
            self.fitmenu.exec_(self.cursor().pos())

    def __saveIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            if qt.qVersion() < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            return
        #get outputfile
        if self.outputdir is None:
            self.outputdir = os.getcwd()
            wdir = os.getcwd()
        elif os.path.exists(self.outputdir): wdir = self.outputdir
        else:
            self.outputdir = os.getcwd()
            wdir = self.outputdir
            
        if QTVERSION < '4.0.0':
            outfile = qt.QFileDialog(self,"Output File Selection",1)
            outfile.setFilters('Specfile MCA  *.mca\nSpecfile Scan *.dat\nRaw ASCII  *.txt')
            outfile.setMode(outfile.AnyFile)
            outfile.setDir(wdir)
            ret = outfile.exec_loop()
        else:
            outfile = qt.QFileDialog(self)
            outfile.setWindowTitle("Output File Selection")
            outfile.setModal(1)
            outfile.setFilters(['Specfile MCA  *.mca',
                                'Specfile Scan *.dat',
                                'Raw ASCII  *.txt'])
            outfile.setFileMode(outfile.AnyFile)
            outfile.setDirectory(wdir)
            ret = outfile.exec_()
        if ret:
            filterused = str(outfile.selectedFilter()).split()
            filetype  = filterused[1]
            extension = filterused[2]
            if QTVERSION < '4.0.0':
                self.outdir=str(outfile.selectedFile())
            else:
                self.outdir=str(outfile.selectedFiles()[0])
            try:            
                outputDir  = os.path.dirname(self.outdir)
            except:
                outputDir  = "."
            try:            
                outputFile = os.path.basename(self.outdir)
            except:
                outputFile  = self.outdir
            outfile.close()
            del outfile
        else:
            outfile.close()
            del outfile
            return

        #get active curve
        info,x, y = self.getinfodatafromlegend(legend)
        if info is None: return
        ndict = {}
        ndict[legend] = {'order':1,'A':0.0,'B':1.0,'C':0.0}
        if self.caldict.has_key(legend):
            ndict[legend].update(self.caldict[legend])
            if abs(ndict[legend]['C']) > 0.0:
                ndict[legend]['order']  = 2    
        elif info.has_key('McaCalib'):
            if type(info['McaCalib'][0]) == type([]):
                calib = info['McaCalib'][0]
            else:
                calib = info['McaCalib']
            if len(calib) > 1:
                ndict[legend]['A'] = calib[0]
                ndict[legend]['B'] = calib[1]
                if len(calib) >2:
                    ndict[legend]['order']  = 2
                    ndict[legend]['C']      = calib[2]
        #I should have x, y, caldict
        """ 
        caldialog = McaCalWidget.McaCalWidget(legend=legend,
                                                 x=x,
                                                 y=y,
                                                 modal=1,
                                                 caldict=ndict,
                                                 fl=0)
        """
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
            file=open(specFile,'wb')
        except IOError:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
            msg.exec_loop()
            return
        systemline = os.linesep
        os.linesep = '\n'
        #This was giving problems on legends with a leading b
        #legend = legend.strip('<b>')
        #legend = legend.strip('<\b>')
        try:
            if filetype == 'Scan':
                file.write("#F %s\n" % specFile)
                file.write("#D %s\n"%(time.ctime(time.time())))
                file.write("\n")
                file.write("#S 1 %s\n" % legend)
                file.write("#D %s\n"%(time.ctime(time.time())))
                file.write("#N 3\n")
                file.write("#L channel  counts  energy\n")
                energy = ndict[legend]['A'] + ndict[legend]['B'] * x + ndict[legend]['C'] * x * x
                for i in range(len(y)):
                    file.write("%.7g  %.7g  %.7g\n" % (x[i], y[i], energy[i]))
                file.write("\n")
            elif filetype == 'ASCII':
                energy = ndict[legend]['A'] + ndict[legend]['B'] * x + ndict[legend]['C'] * x * x
                for i in range(len(y)):
                    file.write("%.7g  %.7g  %.7g\n" % (x[i], y[i], energy[i]))
            else:
                file.write("#F %s\n" % specFile)
                file.write("#D %s\n"%(time.ctime(time.time())))
                file.write("\n")
                file.write("#S 1 %s\n" % legend)
                file.write("#D %s\n"%(time.ctime(time.time())))
                file.write("#@MCA %16C\n")
                file.write("#@CHANN %d %d %d 1\n" %  (len(y), x[0], x[-1]))
                file.write("#@CALIB %.7g %.7g %.7g\n" % (ndict[legend]['A'],
                                                         ndict[legend]['B'],
                                                         ndict[legend]['C']))
                file.write(self.array2SpecMca(y))
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
        


    def mcasimplefitsignal(self):
        legend = self.graph.getactivecurve(justlegend = 1)
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.exec_loop()
           return
        info,x,y = self.getinfodatafromlegend(legend)
        self.advancedfit.hide()
        self.simplefit.show()
        self.simplefit.setFocus()
        if QTVERSION < '4.0.0':
            self.simplefit.raiseW()
        else:
            self.simplefit.raise_()
        if info is not None:
            xmin,xmax=self.graph.getx1axislimits()
            self.__simplefitcalmode = self.calibration
            self.simplefit.setdata(x=x,y=y,
                                    xmin=xmin,
                                    xmax=xmax,
                                    legend=legend)
            self.specfit.fitconfig['McaMode'] = 1
            """
            if self.specfit.fitconfig['McaMode']:
                self.specfitGUI.guiparameters.fillfromfit(self.specfit.paramlist,
                                    current='Region 1')
                self.specfitGUI.guiparameters.removeallviews(keep='Region 1')
            else:
                self.specfitGUI.guiparameters.fillfromfit(self.specfit.paramlist,
                                        current='Fit')
                self.specfitGUI.guiparameters.removeallviews(keep='Fit')
            """
            self.simplefit.fit()
        else:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error. Trying to fit fitted data?")
                msg.exec_loop() 
                    
    def mcaadvancedfitsignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            msg.exec_loop()
            return
        else:
            info,x,y = self.getinfodatafromlegend(legend)
            curveinfo = self.graph.getcurveinfo(legend)
            if self.calibration == 'None':
                xmin,xmax =self.graph.getx1axislimits()
                if curveinfo.has_key('McaCalibSource'):
                    calib = curveinfo['McaCalibSource']
                else:
                    calib = [0.0,1.0,0.0]
            else:
                calib = curveinfo['McaCalib']
                xmin,xmax = self.graph.getx1axislimits()
                energy = calib[0] + calib[1] * x + calib[2] * x * x
                i1 = min(Numeric.nonzero(energy >= xmin))
                i2 = max(Numeric.nonzero(energy <= xmax))
                xmin = x[i1] * 1.0
                xmax = x[i2] * 1.0

        if self.simplefit is not None: self.simplefit.hide()
        self.advancedfit.show()
        self.advancedfit.setFocus()
        if QTVERSION < '4.0.0':
            self.advancedfit.raiseW()
        else:
            self.advancedfit.raise_()
        if info is not None:
            xlabel = 'Channel'
            self.advancedfit.setdata(x=x,y=y,
                                     xmin=xmin,
                                     xmax=xmax,
                                     legend=legend,
                                     xlabel=xlabel,
                                     calibration=calib,
                                     sourcename=info['SourceName'])
            self.advancedfit.fit()
        else:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error. Trying to fit fitted data?")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()

    def __anasignal(self,dict):
        if DEBUG:
            print "__anasignal called dict = ",dict
            
        if dict['event'] == 'clicked':
            # A button has been cicked
            if   dict['button'] == 'Source':
                pass
            elif dict['button'] == 'Calibration':
                #legend,x,y = self.graph.getactivecurve()
                legend = self.graph.getactivecurve(justlegend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    msg.exec_loop()
                    return
                else:
                    info,x, y = self.getinfodatafromlegend(legend)
                    if info is None: return
                    ndict = {}
                    ndict[legend] = {'order':1,'A':0.0,'B':1.0,'C':0.0}
                    if self.caldict.has_key(legend):
                        ndict[legend].update(self.caldict[legend])
                        if abs(ndict[legend]['C']) > 0.0:
                            ndict[legend]['order']  = 2    
                    elif info.has_key('McaCalib'):
                        if type(info['McaCalib'][0]) == type([]):
                            calib = info['McaCalib'][0]
                        else:
                            calib = info['McaCalib']
                        if len(calib) > 1:
                            ndict[legend]['A'] = calib[0]
                            ndict[legend]['B'] = calib[1]
                            if len(calib) >2:
                                ndict[legend]['order']  = 2
                                ndict[legend]['C']      = calib[2]
                    caldialog = McaCalWidget.McaCalWidget(legend=legend,
                                                             x=x,
                                                             y=y,
                                                             modal=1,
                                                             caldict=ndict,
                                                             fl=0)
                    #info,x,y = self.getinfodatafromlegend(legend)
                    #caldialog.graph.newcurve("fromlegend",x=x,y=y)
                    if QTVERSION < '4.0.0':
                        ret = caldialog.exec_loop()
                    else:
                        ret = caldialog.exec_()

                    if ret == qt.QDialog.Accepted:
                        self.caldict.update(caldialog.getdict())
                        item, text = self.control.calbox.getcurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.control.calbox.setoptions(options)
                        except:
                            pass
                        if QTVERSION < '4.0.0':
                            self.control.calbox.setCurrentItem(item)
                        else:
                            self.control.calbox.setCurrentIndex(item)
                        self.refresh()
                    del caldialog
            elif dict['button'] == 'CalibrationCopy':
                #legend,x,y = self.graph.getactivecurve()
                legend = self.graph.getactivecurve(justlegend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    msg.exec_loop()
                    return
                else:
                    info,x, y = self.getinfodatafromlegend(legend)
                    if info is None: return
                    ndict=copy.deepcopy(self.caldict)
                    if info.has_key('McaCalib'):
                        if type(info['McaCalib'][0]) == type([]):
                            sourcecal = info['McaCalib'][0]
                        else:
                            sourcecal = info['McaCalib']
                    else:
                        sourcecal = [0.0,1.0,0.0]
                    for curve in self.graph.curveslist:
                        curveinfo = self.graph.getcurveinfo(curve)
                        if curveinfo.has_key('McaCalibSource'):
                            key = "%s (Source)" % curve
                            if key not in ndict:
                                if curveinfo['McaCalibSource'] != [0.0,1.0,0.0]:
                                    ndict[key] = {'A':curveinfo['McaCalibSource'][0],
                                                  'B':curveinfo['McaCalibSource'][1],
                                                  'C':curveinfo['McaCalibSource'][2]} 
                                    if curveinfo['McaCalibSource'][2] != 0.0:
                                        ndict[key]['order'] = 2
                                    else:
                                        ndict[key]['order'] = 1
                            if curve not in self.caldict.keys():
                                if curveinfo['McaCalib'] != [0.0,1.0,0.0]:
                                    if curveinfo['McaCalib'] != curveinfo['McaCalibSource']:
                                        key = "%s (PyMca)" % curve    
                                        ndict[key] = {'A':curveinfo['McaCalib'][0],
                                                      'B':curveinfo['McaCalib'][1],
                                                      'C':curveinfo['McaCalib'][2]} 
                                        if curveinfo['McaCalib'][2] != 0.0:
                                            ndict[key]['order'] = 2
                                        else:
                                            ndict[key]['order'] = 1
                        else:
                            if curve not in self.caldict.keys():
                                if curveinfo['McaCalib'] != [0.0,1.0,0.0]:
                                        key = "%s (PyMca)" % curve    
                                        ndict[key] = {'A':curveinfo['McaCalib'][0],
                                                      'B':curveinfo['McaCalib'][1],
                                                      'C':curveinfo['McaCalib'][2]} 
                                        if curveinfo['McaCalib'][2] != 0.0:
                                            ndict[key]['order'] = 2
                                        else:
                                            ndict[key]['order'] = 1                                         
                    
                    if not self.caldict.has_key(legend):
                        ndict[legend]={}
                        ndict[legend]['A'] = sourcecal[0] 
                        ndict[legend]['B'] = sourcecal[1] 
                        ndict[legend]['C'] = sourcecal[2]
                        if sourcecal[2] != 0.0:
                            ndict[legend]['order'] = 2
                        else: 
                            ndict[legend]['order'] = 1
                    caldialog = McaCalWidget.McaCalCopy(legend=legend,modal=1,
                                                        caldict=ndict,
                                                        sourcecal=sourcecal,
                                                        fl=0)
                    #info,x,y = self.getinfodatafromlegend(legend)
                    #caldialog.graph.newcurve("fromlegend",x=x,y=y)
                    ret = caldialog.exec_loop()
                    if ret == qt.QDialog.Accepted:
                        self.caldict.update(caldialog.getdict())
                        item, text = self.control.calbox.getcurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.control.calbox.setoptions(options)
                        except:
                            pass 
                        self.control.calbox.setCurrentItem(item)
                        self.refresh()
                    del caldialog
            elif dict['button'] == 'CalibrationLoad':
                item     = dict['box'][0]
                itemtext = dict['box'][1]
                filename = dict['line_edit']
                if not os.path.exists(filename):
                    text = "Error. Calibration file %s not found " % filename
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return
                cald = ConfigDict.ConfigDict()
                try:
                    cald.read(filename)
                except:
                    text = "Error. Cannot read calibration file %s" % filename
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return
                self.caldict.update(cald)
                options = []
                for option in self.calboxoptions:
                    options.append(option)
                for key in self.caldict.keys():
                    if key not in options:
                        options.append(key)
                try:
                    self.control.calbox.setoptions(options)
                    if QTVERSION < '4.0.0':
                        self.control.calbox.setCurrentItem(options.index(itemtext))
                    else:
                        self.control.calbox.setCurrentIndex(options.index(itemtext))                        
                    self.calibration = itemtext * 1
                    self.control._calboxactivated(itemtext)
                except:
                    text = "Error. Problem updating combobox"
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return
            elif dict['button'] == 'CalibrationSave':
                filename = dict['line_edit']
                cald = ConfigDict.ConfigDict()
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except:
                        text = "Error. Problem deleting existing file %s" % filename
                        msg = qt.QMessageBox(self)
                        msg.setIcon(qt.QMessageBox.Critical)
                        msg.setText(text)
                        if QTVERSION < '4.0.0':
                            msg.exec_loop()
                        else:
                            msg.exec_()
                        return
                cald.update(self.caldict)
                cald.write(filename)
            elif dict['button'] == 'Detector':
                pass
            elif dict['button'] == 'Search':
                pass
            elif dict['button'] == 'Fit':
                if dict['box'][1]   == 'Simple':
                    self.mcasimplefitsignal()
                elif dict['box'][1]   == 'Advanced':
                    self.mcaadvancedfitsignal()
                else:
                    print "Unknown Fit Event"
        elif dict['event'] == 'activated':
            # A comboBox has been selected
            if   dict['boxname'] == 'Source':
                pass
            elif dict['boxname'] == 'Calibration':
                self.calibration = dict['box'][1]
                self.graph.clearMarkers()
                self.roimarkers = [-1,-1]
                self.refresh()
                self.graph.ResetZoom()
                
            elif dict['boxname'] == 'Detector':
                pass
            elif dict['boxname'] == 'Search':
                pass
            elif dict['boxname'] == 'ROI':
                if dict['combotext'] == 'Add':
                    pass
                elif dict['combotext'] == 'Del':
                    pass
                else:
                    pass
            elif dict['boxname'] == 'Fit':
                """
                if dict['box'][1] == 'Simple':
                    self.anacontainer.hide()
                else:
                    self.anacontainer.show()
                """
                pass
            else:
                if DEBUG:
                    print "Unknown combobox",dict['boxname']

        elif (dict['event'] == 'EstimateFinished'):
            pass
        elif (dict['event'] == 'McaAdvancedFitFinished') or \
             (dict['event'] == 'McaAdvancedFitMatrixFinished') :
            x      = dict['result']['xdata']
            yb     = dict['result']['continuum']
            legend0= dict['info']['legend']
            if dict['event'] == 'McaAdvancedFitMatrixFinished':
                legend = dict['info']['legend'] + " Fit"
                legend3 = dict['info']['legend'] + " Matrix"
                ymatrix   = dict['result']['ymatrix'] * 1.0
                #copy the original info from the curve
                newDataObject = DataObject.DataObject()
                newDataObject.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
                newDataObject.info['SourceType']= 'AdvancedFit'
                newDataObject.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
                newDataObject.info['legend']    = legend3
                newDataObject.info['Key']       = legend3
                newDataObject.x = [x]
                newDataObject.y = [ymatrix]
                newDataObject.m = None
                self.dataObjectsDict[legend3] = newDataObject
                self.graph.newcurve(legend3,x=x,y=ymatrix,logfilter=1)
            else:
                legend = dict['info']['legend'] + " Fit"
                yfit   = dict['result']['yfit'] * 1.0

                #copy the original info from the curve
                newDataObject = DataObject.DataObject()
                newDataObject.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
                newDataObject.info['SourceType']= 'AdvancedFit'
                newDataObject.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
                newDataObject.info['legend'] = legend
                newDataObject.info['Key']  = legend
                newDataObject.data = Numeric.reshape(Numeric.concatenate((x,yfit,yb),0),(3,len(x)))
                newDataObject.x = [x]
                newDataObject.y = [yfit]
                newDataObject.m = None
                self.dataObjectsDict[legend] = newDataObject
                self.graph.newcurve(legend,x=x,y=yfit,logfilter=1)

                #the same for the background
                legend2 = dict['info']['legend'] + " Bkg"
                newDataObject2 = DataObject.DataObject()
                newDataObject2.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
                newDataObject2.info['SourceType']= 'AdvancedFit'
                newDataObject2.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
                newDataObject2.info['legend'] = legend2
                newDataObject2.info['Key']  = legend2
                newDataObject2.data = None
                newDataObject2.x = [x]
                newDataObject2.y = [yb]
                newDataObject2.m = None
                self.dataObjectsDict[legend2] = newDataObject2
                self.graph.newcurve(legend2,x=x,y=yb,logfilter=1)

            if not self.caldict.has_key(legend):
                self.caldict[legend] = {}
            self.caldict[legend] ['order'] = 1
            self.caldict[legend] ['A']     = dict['result']['fittedpar'][0]
            self.caldict[legend] ['B']     = dict['result']['fittedpar'][1]
            self.caldict[legend] ['C']     = 0.0 
            options = []
            for option in self.calboxoptions:
                options.append(option)
            for key in self.caldict.keys():
                if key not in options:
                    options.append(key)
            try:
                self.control.calbox.setoptions(options)
                #I only reset the graph scale after a fit, not on a matrix spectrum
                if dict['event'] == 'McaAdvancedFitFinished':
                    #get current limits
                    if self.calibration == 'None':
                        xmin,xmax =self.graph.getx1axislimits()
                        emin    = dict['result']['fittedpar'][0] + \
                                  dict['result']['fittedpar'][1] * xmin
                        emax    = dict['result']['fittedpar'][0] + \
                                  dict['result']['fittedpar'][1] * xmax
                    else:
                        emin,emax = self.graph.getx1axislimits()
                    ymin,ymax =self.graph.gety1axislimits()
                    self.control.calbox.setCurrentItem(options.index(legend))
                    self.calibration = legend
                    self.control._calboxactivated(legend)
                    self.graph.sety1axislimits(ymin, ymax, False)
                    if emin < emax:
                        self.graph.setx1axislimits(emin, emax, True)
                    else:
                        self.graph.setx1axislimits(emax, emin, True)
            except:
                pass
            self.graph.replot()

        elif dict['event'] == 'McaFitFinished':
            mcaresult = dict['data']
            legend = dict['info']['legend'] + " "
            i = 0
            xfinal = []
            yfinal = []
            ybfinal= []
            regions = []
            for result in mcaresult:
                i += 1
                if result['chisq'] is not None:
                     idx=Numeric.nonzero((self.specfit.xdata0>=result['xbegin']) & \
                                         (self.specfit.xdata0<=result['xend']))
                     x=Numeric.take(self.specfit.xdata0,idx)
                     y=self.specfit.gendata(x=x,parameters=result['paramlist'])
                     nparb= len(self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1])
                     yb   = self.specfit.gendata(x=x,parameters=result['paramlist'][0:nparb])
                     regions.append([result['xbegin'],result['xend']])
                     xfinal = xfinal + x.tolist()
                     yfinal = yfinal + y.tolist()
                     ybfinal= ybfinal + yb.tolist()
                    #self.graph.newcurve(legend + 'Region %d' % i,x=x,y=yfit,logfilter=1)
            legend0= dict['info']['legend']
            legend = legend0 + " SFit"            
            #copy the original info from the curve
            newDataObject = DataObject.DataObject()
            newDataObject.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
            newDataObject.info['SourceType']= 'SimpleFit'
            newDataObject.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
            newDataObject.info['legend']    = legend
            newDataObject.info['Key']       = legend
            newDataObject.info['Regions']   = regions
            newDataObject.info['CalMode']   = self.__simplefitcalmode
            x    = Numeric.array(xfinal)
            yfit = Numeric.array(yfinal)
            yb = Numeric.array(ybfinal)
            newDataObject.x = [x]
            newDataObject.y = [yfit]
            newDataObject.m = None
            newDataObject.baseline = [yb]
            self.dataObjectsDict[legend] = newDataObject
            if 1 or QTVERSION < '4.0.0':
                self.graph.newCurve(legend,x=x,
                                    y=yfit,
                                    logfilter=1,
                                    regions=regions,
                                    baseline=yb)
            else:
                self.graph.newCurve(legend,x=x,
                                    y=yfit,
                                    logfilter=1)
            #self.graph.setxofy(legend)
            self.graph.replot()
        elif dict['event'] == 'McaTableFilled':
            if self.peakmarker is not None:
                self.graph.removeMarker(self.peakmarker)
            self.peakmarker = None
        
        elif dict['event'] == 'McaTableRowHeaderClicked':
            #I have to mark the peaks
            if dict['row'] >= 0:
                pos = dict['Position']
                label = 'PEAK %d' % (dict['row']+1)
                if self.peakmarker is None:
                    self.peakmarker = self.graph.insertx1marker(pos,1.1,
                                        label = label)
                self.graph.setx1markerpos(self.peakmarker,pos)
                self.graph.setmarkercolor(self.peakmarker,'pink',
                                        label=label)
                self.graph.replot()
            else:
                if self.peakmarker is not None:
                    self.graph.removeMarker(self.peakmarker)
                self.peakmarker = None
                
        elif dict['event'] == 'McaTableClicked':
            if self.peakmarker is not None:
                self.graph.removeMarker(self.peakmarker)
            self.peakmarker = None
            self.graph.replot()    

        elif (dict['event'] == 'McaAdvancedFitElementClicked') or (dict['event'] == 'ElementClicked'):
            for marker in self.elementmarkers:
                self.graph.removeMarker(marker)
            self.elementmarkers = []
            if dict.has_key('current'):
                legend = self.graph.getactivecurve(justlegend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    msg.exec_loop()
                    return
                ele = dict['current']
                items = []
                if dict.has_key(ele):
                    for rays in dict[ele]:
                        for transition in  Elements.Element[ele][rays +" xrays"]:
                            items.append([transition,Elements.Element[ele][transition]['energy'],
                                                     Elements.Element[ele][transition]['rate']])

                if self.calibration == 'None':
                    xmin,xmax =self.graph.getx1axislimits()
                    curveinfo = self.graph.getcurveinfo(legend)
                    if 'McaCalibSource' in curveinfo.keys():
                        calib = curveinfo['McaCalibSource']
                    else:
                        #it makes no sense to draw anything
                        return
                else:
                    calib = self.graph.getcurveinfo(legend)['McaCalib']
                    xmin,xmax = self.graph.getx1axislimits()
                factor = 1 # Kev
                if calib[1] > 0.1:factor = 1000. # ev 
                #clear existing markers
                xmin,xmax = self.graph.getx1axislimits()
                ymin,ymax = self.graph.gety1axislimits()
                for transition,energy,rate in items:
                    marker = ""
                    if self.calibration == 'None':
                        if abs(calib[1]) > 0.0000001:
                            x = (energy * factor - calib[0])/calib[1]
                            marker=self.graph.insertx1marker(x,ymax*rate,label=transition)
                    else: 
                            marker=self.graph.insertx1marker(energy*factor,ymax*rate,label=transition)
                    if marker is not "":
                        self.elementmarkers.append(marker)
                        self.graph.setmarkercolor(marker,'orange')
            self.graph.replot()                  
                
        elif dict['event'] == 'McaAdvancedFitPrint':
            #self.advancedfit.printps(doit=1)
            self.printHtml(dict['text'])

        elif dict['event'] == 'McaSimpleFitPrint':
            self.printHtml(dict['text'])

        elif dict['event'] == 'McaSimpleFitClosed':
            if self.peakmarker is not None:
                self.graph.removeMarker(self.peakmarker)
            self.peakmarker = None
            self.graph.replot()
        elif dict['event'] == 'ScanFitPrint':
            self.printHtml(dict['text'])

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
            if QTVERSION < '4.0.0':
                self.roidict[newroi]['type']    = str(self.graph.xlabel())
            else:
                self.roidict[newroi]['type']    = str(self.graph.xlabel())
            self.roidict[newroi]['from']    = fromdata
            self.roidict[newroi]['to']      = todata
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=newroi)
            self.currentroi = newroi
            ndict = {}
            ndict['event'] = "SetActiveCurveEvent"
            self.__graphsignal(ndict)

        elif dict['event'] == 'DelROI':
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            self.currentroi = self.roidict.keys()[0]
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=self.currentroi)
            ndict = {}
            ndict['event'] = "SetActiveCurveEvent"
            self.__graphsignal(ndict)
            
        elif dict['event'] == 'ResetROI':
            self.graph.clearMarkers()
            self.roimarkers = [-1, -1]
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            self.currentroi = self.roidict.keys()[0]
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict,
                                           currentroi=self.currentroi)
            ndict = {}
            ndict['event'] = "SetActiveCurveEvent"
            self.__graphsignal(ndict)
            
        elif dict['event'] == 'ActiveROI':
            print "ActiveROI event"
            pass
        elif dict['event'] == 'selectionChanged':
            if DEBUG:
                print "Selection changed"
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            fromdata = dict['roi']['from']
            todata   = dict['roi']['to']
            if self.roimarkers[0] < 0:
                self.roimarkers[0] = self.graph.insertx1marker(fromdata,1.1,
                                        label = 'ROI min')
            if self.roimarkers[1] < 0:
                self.roimarkers[1] = self.graph.insertx1marker(todata,1.1,
                                        label = 'ROI max')
            self.graph.setx1markerpos(self.roimarkers[0],fromdata)
            self.graph.setx1markerpos(self.roimarkers[1],todata )
            self.currentroi = dict['key']
            if dict['key'] == 'ICR':
                #select the colors
                self.graph.setmarkercolor(self.roimarkers[1],'black' )
                self.graph.setmarkercolor(self.roimarkers[0],'black' )
                #set the follow mouse propierty
                self.graph.setmarkerfollowmouse(self.roimarkers[1],0)
                self.graph.setmarkerfollowmouse(self.roimarkers[0],0)
                self.graph.disablemarkermode()
            else:
                #select the colors
                self.graph.setmarkercolor(self.roimarkers[0],'blue' )
                self.graph.setmarkercolor(self.roimarkers[1],'blue' )
                #set the follow mouse propierty
                self.graph.setmarkerfollowmouse(self.roimarkers[0],1)
                self.graph.setmarkerfollowmouse(self.roimarkers[1],1)
                self.graph.enablemarkermode()
            self.emitCurrentROISignal()
            if dict['colheader'] == 'From':
                #I should put the table in RW mode
                pass
            elif dict['colheader'] == 'To':    
                #I should put the table in RW mode
                pass
            elif dict['colheader'] == 'Raw Counts':    
                #I should put the table in RW mode
                pass
            elif dict['colheader'] == 'Net Counts':    
                #I should put the table in RW mode
                pass
            else:
                pass
            self.graph.replot()
        else:
            if DEBUG:
                print "Unknown or ignored event",dict['event']


    def refresh(self):
        activecurve = self.graph.getactivecurve(justlegend=1)
        if 0:
            self.graph.clearcurves()
            for key in self.dataObjectsDict.keys():
                self.graph.newcurve(key,x=self.dataObjectsDict[key].x[0],
                                    y=self.dataObjectsDict[key].y[0],
                                    logfilter = 1)
        else:
            sellist = []
            for key in self.dataObjectsDict.keys():
                sel ={}
                sel['SourceName'] = self.dataObjectsDict[key].info['SourceName']
                sel['dataobject'] = self.dataObjectsDict[key]
                sel['legend'] = key
                sel['Key'] = self.dataObjectsDict[key].info['Key']
                sellist.append(sel)
            self.graph.clearcurves()
            self._addSelection(sellist)
        self.graph.setactivecurve(activecurve)
        self.graph.show()

        
    def __graphsignal(self,dict):
        if DEBUG:
            print "__graphsignal called dict = ",dict
        if dict['event'] == 'markerSelected':
            pass
        elif dict['event'] == 'markerMoved':
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            if self.currentroi is None:
                print "self.currentroi unset :(  "
                return
            if self.currentroi not in self.roidict.keys():
                print "self.currentroi wrongly set"
                return            
            if dict['marker'] == self.roimarkers[0]:
                self.roidict[self.currentroi]['from'] = dict['x']
            elif dict['marker'] == self.roimarkers[1]:
                self.roidict[self.currentroi]['to'] = dict['x']            
            else:
                pass
            self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict)
            dict ={}
            dict['event']  = "SetActiveCurveEvent"
            dict['legend'] = self.graph.getactivecurve(justlegend=1)
            self.__graphsignal(dict)
        elif dict['event'] == 'MouseAt':
            if self.calibration == self.calboxoptions[0]:
                self.xpos.setText('%.2f' % dict['x'])
                self.ypos.setText('%.2f' % dict['y'])
            else:
                self.xpos.setText('%.4f' % dict['x'])
                self.ypos.setText('%.2f' % dict['y'])
        elif dict['event'] == "SetActiveCurveEvent":
            legend = None
            if dict.has_key('legend'):
                legend = dict['legend']
            if legend is not None:
                legend = self.graph.getactivecurve(justlegend=1)
                if legend in self.dataObjectsDict.keys():
                    x0 = self.dataObjectsDict[legend].x[0]
                    y = self.dataObjectsDict[legend].y[0]
                    #those are the actual data
                    if str(self.graph.xlabel()).upper() != "CHANNEL":
                        #I have to get the energy
                        A = self.control.calinfo.caldict['']['A']
                        B = self.control.calinfo.caldict['']['B']
                        C = self.control.calinfo.caldict['']['C']
                        order = self.control.calinfo.caldict['']['order']
                    else:
                        A = 0.0
                        B = 1.0
                        C = 0.0
                        order = 1
                    calib = [A,B,C]
                    x = calib[0]+ \
                        calib[1]* x0 + \
                        calib[2]* x0 * x0
                else:
                    print "Should not be here"
                    return
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
                        if key == 'ICR':
                            #take care of LLD
                            #if len(x) > 3:
                            #    fromdata = x[3]
                            #else:
                            fromdata = x0[0]
                            todata   = x0[-1]
                            #I profit to update ICR
                            self.roidict[key]['from'] = x0[0]
                            self.roidict[key]['to'] = x0[-1]
                        else:
                            fromdata = self.roidict[key]['from']
                            todata   = self.roidict[key]['to']
                        if self.roidict[key]['type'].upper() != "CHANNEL":
                            i1 = Numeric.nonzero(x>=fromdata)
                            xw = Numeric.take(x,i1)
                        else:
                            i1 = Numeric.nonzero(x0>=fromdata)
                            xw = Numeric.take(x0,i1)
                        yw = Numeric.take(y, i1)
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
                    self.emitCurrentROISignal()
                xlabel = self.graph.xlabel()
                #self.roiwidget.setheader(text="%s ROIs of %s" % (xlabel,legend))
                self.roiwidget.setheader(text="ROIs of %s" % (legend))
                self.roiwidget.fillfromroidict(roilist=self.roilist,
                                                roidict=self.roidict)
                try:
                    calib = self.graph.getcurveinfo(legend)['McaCalib']
                    self.control.calinfo.setParameters({'A':calib[0],'B':calib[1],'C':calib[2]})
                except:
                    self.control.calinfo.AText.setText("?????")
                    self.control.calinfo.BText.setText("?????")
                    self.control.calinfo.CText.setText("?????")
            else:
                    self.control.calinfo.AText.setText("?????")
                    self.control.calinfo.BText.setText("?????")
                    self.control.calinfo.CText.setText("?????")                
                
        elif dict['event'] == "RemoveCurveEvent":
            #WARNING this is to be called just from the graph!"
            #get selection from legend
            legend = dict['legend']
            self.graph.delcurve(legend)
            if self.dataObjectsDict.has_key(legend):
                del self.dataObjectsDict[legend]
            self.graph.replot()
            #I should generate an event to allow the controller
            #to eventually inform other widgets
        else:
            if DEBUG:
                print "Unhandled event ",   dict['event']   

    def emitCurrentROISignal(self):
        if self.currentroi is None: return
        #I have to get the current calibration
        if str(self.graph.xlabel()).upper() != "CHANNEL":
            #I have to get the energy
            A = self.control.calinfo.caldict['']['A']
            B = self.control.calinfo.caldict['']['B']
            C = self.control.calinfo.caldict['']['C']
            order = self.control.calinfo.caldict['']['order']
        else:
            A = 0.0
            B = 1.0
            C = 0.0
            order = 1
        key = self.currentroi
        fromdata = self.roidict[key]['from' ]
        todata   = self.roidict[key]['to']
        ddict = {}
        ddict['event']      = "ROISignal"
        ddict['name'] = key
        ddict['from'] = fromdata
        ddict['to']   = todata
        ddict['type'] = self.roidict[self.currentroi]["type"]
        ddict['calibration']= [A, B, C, order]
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL("McaWindowSignal"),
                    (ddict,))
        else:
            self.emit(qt.SIGNAL("McaWindowSignal"),
                    ddict)

    def _addSelection(self,selection):
        if DEBUG:
            print "__add, selection = ",selection

        if type(selection) == type([]):
            sellist = selection
        else:
            sellist = [selection]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            if sel.has_key("scanselection"):
                if sel["scanselection"]:continue
            mcakeys    = [key]
            for mca in mcakeys:
                curveinfo={}
                curvesel={}
                legend = sel['legend']
                dataObject = sel['dataobject']
                if dataObject.info.has_key("selectiontype"):
                    if dataObject.info["selectiontype"] != "1D": continue
                if dataObject.x is None:
                    xhelp = None
                else:
                    xhelp = dataObject.x[0]
                data  = dataObject.y[0]
                info = dataObject.info
                self.dataObjectsDict[legend] = dataObject
                try:
                    calib = [0.0,1.0,0.0]
                    curveinfo['McaCalib'] = calib
                    if info.has_key('McaCalib'):
                        if type(info['McaCalib'][0]) == type([]):
                            calib0 = info['McaCalib'][info['McaDet']-1]
                        else:
                            calib0 = info['McaCalib']
                        curveinfo['McaCalibSource'] = calib0
                    if xhelp is None:
                        xhelp =info['Channel0'] + Numeric.arange(len(data)).astype(Numeric.Float)
                    if self.calibration == self.calboxoptions[1]:
                        if info.has_key('McaCalib'):
                            if type(info['McaCalib'][0]) == type([]):
                                calib = info['McaCalib'][info['McaDet']-1]
                            else:
                                calib = info['McaCalib']
                        if len(calib) > 1:
                            xdata=calib[0]+ \
                                  calib[1]* xhelp
                            if len(calib) == 3:
                                  xdata = xdata + calib[2]* xhelp * xhelp
                            curveinfo['McaCalib'] = calib
                            self.graph.newCurve(legend,
                                                x=xdata,y=data,logfilter=1, curveinfo=curveinfo)
                            self.graph.xlabel('Energy')
                    elif self.calibration == self.calboxoptions[2]:
                        if self.caldict.has_key(legend):
                            A = self.caldict[legend]['A']
                            B = self.caldict[legend]['B']
                            C = self.caldict[legend]['C']
                            calib = [A,B,C]
                        elif info.has_key('McaCalib'):
                            if type(info['McaCalib'][0]) == type([]):
                                calib = info['McaCalib'][info['McaDet']-1]
                            else:
                                calib = info['McaCalib']
                        if len(calib) > 1:
                            xdata=calib[0]+ \
                                  calib[1]* xhelp
                            if len(calib) == 3:
                                  xdata = xdata + calib[2]* xhelp * xhelp
                            curveinfo['McaCalib'] = calib
                            self.graph.newcurve(legend,
                                                x=xdata,y=data,logfilter=1, curveinfo=curveinfo)
                            self.graph.xlabel('Energy')
                    elif self.calibration == 'Fit':
                        print "Not yet implemented"
                        continue
                    elif self.calibration in  self.caldict.keys():
                            A = self.caldict[self.calibration]['A']
                            B = self.caldict[self.calibration]['B']
                            C = self.caldict[self.calibration]['C']
                            calib = [A,B,C]
                            xdata=calib[0]+ \
                                  calib[1]* xhelp + \
                                  calib[2]* xhelp * xhelp
                            curveinfo['McaCalib'] = calib
                            self.graph.newCurve(legend,
                                                x=xdata,y=data,logfilter=1, curveinfo=curveinfo)
                            self.graph.xlabel('Energy')
                    else:
                        self.graph.newCurve(legend,x=xhelp,y=data,
                                            logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Channel')
                except:
                    del self.dataObjectsDict[legend]
                    raise
        self.graph.replot()
    
    def _removeSelection(self,selection):
        if DEBUG: print "McaWindow._removeSelection, selection =  ",selection
        if type(selection) == type([]):
            sellist = selection
        else:
            sellist = [selection]

        legendlist = []
        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            if sel.has_key("scanselection"):
                if sel["scanselection"]:continue
            mcakeys    = [key]
            for mca in mcakeys:
                legend = sel['legend']
                legendlist.append(legend)

        for legend in legendlist:
            self.graph.delcurve(legend)
            if self.dataObjectsDict.has_key(legend):
                del self.dataObjectsDict[legend]
        if len(legendlist):self.graph.replot()
    
    def _replaceSelection(self,selection):
        if DEBUG: print "McaWindow._replaceSelection, selection =  ",selection
        if type(selection) == type([]):
            sellist = selection
        else:
            sellist = [selection]
        legendlist = []
        doit = False
        for sel in sellist:
            if sel.has_key("scanselection"):
                if sel["scanselection"]:continue
            doit = True
            break
            
        if not doit:return
        self.graph.clearcurves()
        for key in self.dataObjectsDict.keys():
            del self.dataObjectsDict[key]
        self.roimarkers=[-1,-1]
        self._addSelection(selection)

    if QTVERSION < '4.0.0':
        def printHtml(self,text):
            printer = qt.QPrinter()
            if printer.setup(self):
                painter = qt.QPainter()
                if not(painter.begin(printer)):
                    return 0
                try:
                    metrics = qt.QPaintDeviceMetrics(printer)
                    dpiy    = metrics.logicalDpiY()
                    margin  = int((2/2.54) * dpiy) #2cm margin
                    body = qt.QRect(0.5*margin, margin, metrics.width()- 1 * margin, metrics.height() - 2 * margin)
                    #text = self.mcatable.gettext()
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
                        if qt.qVersion() < '3.0.0':
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
                except:
                    #painter.flush()
                    painter.end()
                    msg =  qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("%s" % sys.exc_info()[1])
                    msg.exec_loop()
    else:
        def printHtml(self,text):
            printer = qt.QPrinter()
            printDialog = qt.QPrintDialog(printer, self)
            if printDialog.exec_():
                document = qt.QTextDocument()
                document.setHtml(text)
                document.print_(printer)

    def getselfromlegend(self,legend):
        if DEBUG:
            print "OLD getselfromlegend = ",self.OLDgetselfromlegend(legend)
            print "complete             = ",self.graph.getcurveinfo(legend)
            print "NEW getselfromlegend = ",self.graph.getcurveinfo(legend)['sel']
        return self.graph.getcurveinfo(legend)

    
    def getinfodatafromlegend(self,legend,full=0):
        info = None
        xdata    = None
        ydata    = None
        if legend in self.dataObjectsDict.keys():
            info  = self.dataObjectsDict[legend].info
            xdata = self.dataObjectsDict[legend].x[0]
            ydata = self.dataObjectsDict[legend].y[0]
        else:
            info = None
            xdata    = None
            ydata    = None
        if full:
            return info,None
        else:
            return info,xdata,ydata 
        
    def getinfo(self,info):
        sourcetype = info['SourceType']
        if  sourcetype == 'SpecFile':
            nbmca      = info['NbMcaDet']
            mcacurrent = info['McaDet']
            header     = info['Header']
            calib      = info['McaCalib']
            

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))
      
"""
def finish():
    print "finish"
    while qt.qApp.hasPendingEvents():
        qt.qApp.processEvents()
    qt.qApp.quit()    
"""    
def main(args):
    import QDispatcher
    app = qt.QApplication(args)
    #demo = make()
    if sys.platform == 'win32':
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)
    
    #demo = McaWindow()
    widget = qt.QWidget()
    layout = qt.QHBoxLayout(widget)
    #layout.setMargin(1)
    #layout.setSpacing(1)
    #layout.setSizeConstraint(layout.SetNoConstraint)
    if QTVERSION > '4.0.0':layout.setSizeConstraint(layout.SetMinimumSize)
    dispatcher = QDispatcher.QDispatcher(widget)
    demo = McaWidget(widget)
    demo.setDispatcher(dispatcher)
    layout.addWidget(dispatcher)
    layout.addWidget(demo)
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(widget)
        widget.show()
        #app.thread = SPSthread(demo)
        #app.thread.start()    
        #demo.container.graphwindow.setMinimumWidth(2* demo.tabset.width())
        qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
        app.exec_loop()
    else:
        widget.show()
        qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
        app.exec_()

if __name__ == '__main__':
    main(sys.argv)

