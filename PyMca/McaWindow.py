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
import qt
import qwt
import QtBlissGraph
import os
import Numeric
import McaCalWidget
import McaControlGUI
import McaFitSetupGUI
PYDVT = 0
if PYDVT:
    import SpecFileData
    import SPSData
else:
    import EdfFileLayer
    import SpecFileLayer
    import SPSLayer
import MySpecFileSelector as SpecFileSelector
import MyEdfFileSelector  as EdfFileSelector
if (sys.platform != 'win32') and (sys.platform != 'darwin'):
    import MySPSSelector as SPSSelector
from Icons import IconDict
#import SpecfitFuns
import McaSimpleFit
import McaAdvancedFit
import ScanFit
import Specfit
import getopt
import threading
import time
import McaCustomEvent
import string
import Elements
import copy
import SimpleMath
import ConfigDict
#import SpecfitGUI
DEBUG = 0


class McaWindow(qt.QMainWindow):
    def __init__(self, parent=None, name="McaWindow", specfit=None, fl=qt.Qt.WDestructiveClose,**kw):
        qt.QMainWindow.__init__(self, parent, name, fl)
        self.mcawidget = McaWidget(self,**kw)
        self.mcawidget.show()
        self.setCentralWidget(self.mcawidget)

class McaWidget(qt.QWidget):
    def __init__(self, parent=None, name="Mca Window", specfit=None,fl=qt.Qt.WDestructiveClose,**kw):
        qt.QWidget.__init__(self, parent, name,fl)
        self.parent = parent
        self.inputdict = {}
        self.inputdict['SpecFile']   ={'widget':None,'sel':[]}
        self.inputdict['EdfFile']    ={'widget':None,'sel':[]}
        self.inputdict['SPS']        ={'widget':None,'sel':[]}
        self.inputdict['AdvancedFit']={'widget':McaAdvancedFitSelector(),'sel':[]}
        self.inputdict['SimpleFit']  ={'widget':McaSimpleFitSelector(),'sel':[]}
        self.inputdict['ScanFit']    ={'widget':ScanFitSelector(),'sel':[]}

        for key in self.inputdict.keys():
            if kw.has_key(key+"Widget"):
                self.inputdict[key]['widget'] = kw[key+"Widget"]
                self.connect(self.inputdict[key]['widget'],
                             qt.PYSIGNAL("removeSelection"),
                             self.__remove)
                self.connect(self.inputdict[key]['widget'],
                             qt.PYSIGNAL("addSelection"),
                             self.__add)
                self.connect(self.inputdict[key]['widget'],
                             qt.PYSIGNAL("replaceSelection"),
                             self.__replace)
                             
        for key in ['AdvancedFit','SimpleFit','ScanFit']:
            self.connect(self.inputdict[key]['widget'],
                                 qt.PYSIGNAL("removeSelection"),
                                 self.__remove)
            self.connect(self.inputdict[key]['widget'],
                                 qt.PYSIGNAL("addSelection"),
                                 self.__add)
            self.connect(self.inputdict[key]['widget'],
                                 qt.PYSIGNAL("replaceSelection"),
                                 self.__replace)
        self.calibration = 'None'
        self.calboxoptions = ['None','Original (from Source)','Internal (from Source or PyMca)']
        self.mcadata = {}
        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            self.specfit = specfit
        self.simplefit   = McaSimpleFit.McaSimpleFit(specfit=self.specfit)
        self.scanfit     = ScanFit.ScanFit(specfit=self.specfit)
        self.advancedfit = McaAdvancedFit.McaAdvancedFit()
        self.specfitGUI = self.simplefit.specfitGUI
        self.simplefit.hide()
        self.advancedfit.hide()
        self.caldict={}
        self.calwidget   =  None
        self.roilist = None
        self.roidict = None
        self.currentroi = None
        self.roimarkers     = [-1,-1]
        self.elementmarkers = []
        self.peakmarker     = None
        self.build()
        self.control.calbox.setoptions(self.calboxoptions)
        self.initIcons()
        self.initToolBar()
        self.connections()
        self.setCaption(name)
        #start shared memory survey
        if (sys.platform != 'win32') and (sys.platform != 'darwin'):
            self.thread = SPSthread(self)
            self.thread.start()    

        if kw.has_key('spec'):
            spec = kw['spec']
            #Open sps
            dict={}
            dict['event'] = 'clicked'
            dict['box']   = [0,'SPS']
            dict['button'] = 'Source'
            self.__anasignal(dict)
            #refresh speclist
            self.inputdict['SPS']['widget'].refreshSpecList()
            
            #get the list of opened spec sessions
            n=self.inputdict['SPS']['widget'].specCombo.count()
            speclist = []
            for i in range(n):
                speclist.append(str(self.inputdict['SPS']['widget'].specCombo.text(i)))
            if spec not in speclist:
                qt.QMessageBox.critical(self, "ERROR", "Session %s not open" % spec)
                return
            if 1:
                i = speclist.index(spec)
                self.inputdict['SPS']['widget'].specCombo.setCurrentItem(i)
            else:
                self.inputdict['SPS']['widget'].selectSpec(spec)
            self.inputdict['SPS']['widget'].refreshArrayList(spec)
            if kw.has_key('shm'):
                shm = kw['shm']
            
                n = self.inputdict['SPS']['widget'].arrayList.childCount()


                item = self.inputdict['SPS']['widget'].arrayList.firstChild()
                myitem= None
                while item is not None:
                    name   = str(item.text(1))
                    if name == shm:
                        myitem = item
                        break 
                    item = item.nextSibling()
                if myitem is None:
                    qt.QMessageBox.critical(self, "ERROR", "Array %s not in %s" % (shm,spec))
                    return
                self.inputdict['SPS']['widget'].arrayList.setSelected(myitem,1)
                rows = int(str(myitem.text(2)))
                cols = int(str(myitem.text(3)))
                sel={}
                sel['SourceType'] = "SPS"
                sel['SourceName'] = spec
                sel['Key']        = shm
                sel[shm]    = {'rows':[],'cols':[]}
                if rows > cols:
                    sel[shm]['cols'].append({'x':None,'y':cols-1}) 
                else:
                    sel[shm]['rows'].append({'x':None,'y':rows-1})                
                self.inputdict['SPS']['widget'].setSelected([sel],reset=1)
                try:
                    if shm[0:8] == 'MCA_DATA':
                        self.control.calbox.setCurrentItem(2)
                        self.calibration = self.calboxoptions[2]                        
                except:
                    pass
                self.__spsreplace([sel])
                self.inputdict['SPS']['widget'].hide()
        
    def customEvent(self,event):
        dict = event.dict
        if dict.has_key('source'):
            if dict['source'] == 'SPS':
                if dict['event'] == 'addSelection':
                    self.__spsadd(dict['selection'])
        
    def build(self):
        if self.parent is None:
            self.layout = qt.QHBoxLayout(self)
            #the horizontal separator
            self.vsplit = qt.QSplitter(self)
            self.vsplit.setOrientation(qt.Qt.Vertical)
            #self.vsplit.layout = qt.QVBoxLayout(self.vsplit)
            self.layout.addWidget(self.vsplit)
            #the box to contain the graphics
            self.graphbox = qt.QVBox(self.vsplit)
            self.toolbar  = qt.QHBox(self.graphbox)
            self.graph    = QtBlissGraph.QtBlissGraph(self.graphbox,uselegendmenu=1)
            self.graph.xlabel('Channel')
            self.graph.ylabel('Counts')
            self.graph.canvas().setMouseTracking(1) 
            self.graph.setCanvasBackground(qt.Qt.white)
            if 0:
                self.scanwindow = ScanWindow(self)
                self.layout.addWidget(self.scanwindow)
            else:
                self.scanwindow = ScanWindow(self.parent)
                self.scanwindow.show()

            #the box to contain the control widget(s)
            self.controlbox = qt.QVBox(self.vsplit)
            self.control    = McaControlGUI.McaControlGUI(self.controlbox)
            self.roiwidget  = self.control.roiwidget        
            #self.controlbox.setSizePolicy(qt.QSizePolicy(1,0))
            #the analysis widget is now obsolete
            self.ana = None
            #self.setMinimumWidth(2.5*self.ana.sizeHint().width())
        else:
            self.layout = qt.QHBoxLayout(self)
            #the horizontal separator
            self.vsplit = qt.QSplitter(self)
            self.vsplit.setOrientation(qt.Qt.Vertical)
            #self.vsplit.layout = qt.QVBoxLayout(self.vsplit)
            self.layout.addWidget(self.vsplit)
            #the box to contain the graphics
            self.graphbox = qt.QVBox(self.vsplit)
            self.toolbar  = qt.QHBox(self.graphbox)
            self.graph    = QtBlissGraph.QtBlissGraph(self.graphbox,uselegendmenu=1)
            self.graph.xlabel('Channel')
            self.graph.ylabel('Counts')
            self.graph.canvas().setMouseTracking(1) 
            self.graph.setCanvasBackground(qt.Qt.white)
            if 0:
                self.scanwindow = ScanWindow(self)
                self.layout.addWidget(self.scanwindow)
            else:
                self.scanwindow = ScanWindow(self.parent)
                self.scanwindow.show()

            #the box to contain the control widget(s)
            self.control    = McaControlGUI.McaControlGUI(self.vsplit)
            self.roiwidget  = self.control.roiwidget        
            #self.controlbox.setSizePolicy(qt.QSizePolicy(1,0))
            #the analysis widget is now obsolete
            self.ana = None
            #self.setMinimumWidth(2.5*self.ana.sizeHint().width())
            self.fitmenu = qt.QPopupMenu()
            self.fitmenu.insertItem(qt.QString("Simple"),    self.mcasimplefitsignal)
            self.fitmenu.insertItem(qt.QString("Advanced") , self.mcaadvancedfitsignal)


    def connections(self):
        if self.ana is not None:
            self.connect(self.ana,    qt.PYSIGNAL('McaFitSetupGUISignal'),self.__anasignal)
        self.connect(self.control,    qt.PYSIGNAL('McaControlGUISignal') ,self.__anasignal)
        self.connect(self.scanfit,    qt.PYSIGNAL('ScanFitSignal') , self.__anasignal)
        self.connect(self.simplefit,  qt.PYSIGNAL('McaSimpleFitSignal') , self.__anasignal)
        self.connect(self.advancedfit,qt.PYSIGNAL('McaAdvancedFitSignal') , self.__anasignal)
        self.connect(self.scanwindow, qt.PYSIGNAL('ScanWindowSignal') ,   self.__anasignal)
        self.connect(self.scanwindow, qt.PYSIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)
        self.connect(self.graph,      qt.PYSIGNAL('QtBlissGraphSignal')  ,self.__graphsignal)


    def initIcons(self):
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

    def initToolBar(self):
        toolbar = self.toolbar
        # AutoScale
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.zoomResetIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.graph.ResetZoom)
        qt.QToolTip.add(tb,'Auto-Scale the Graph')

        #y Autoscale
        tb = qt.QToolButton(toolbar)
        tb.setIconSet(self.yAutoIcon)
        #tb.setText("Y Auto")
        tb.setToggleButton(1)
        tb.setState(qt.QButton.On)        
        qt.QToolTip.add(tb,'Toggle Autoscale Y Axis (On/Off)') 
        self.connect(tb, qt.SIGNAL('clicked()'),self._yAutoScaleToggle)

        #x Autoscale
        tb = qt.QToolButton(toolbar)
        #tb.setText("X Auto")
        tb.setIconSet(self.xAutoIcon)
        tb.setToggleButton(1)
        tb.setState(qt.QButton.On)
        qt.QToolTip.add(tb,'Toggle Autoscale X Axis (On/Off)') 
        self.connect(tb, qt.SIGNAL('clicked()'),self._xAutoScaleToggle)

        # Logarithmic
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.logyIcon)
        tb.setToggleButton(1)
        self.connect(tb,qt.SIGNAL('clicked()'),self.graph.ToggleLogY)
        qt.QToolTip.add(tb,'Toggle Logarithmic Y Axis (On/Off)') 
        # Fit
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.fitIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.__fitsignal)
        qt.QToolTip.add(tb,'Fit Active Curve')
        
        #save
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.saveIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.__saveIconSignal)
        qt.QToolTip.add(tb,'Save Active Curve')
         
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
        HorizontalSpacer(toolbar)
        label=qt.QLabel(toolbar)
        #label.setText('<b>Channel:</b>')
        label.setText('<b>X:</b>')
        self.xpos = qt.QLineEdit(toolbar)
        self.xpos.setText('------')
        self.xpos.setReadOnly(1)
        self.xpos.setFixedWidth(self.xpos.fontMetrics().width('########'))
        label=qt.QLabel(toolbar)
        label.setText('<b>Y:</b>')
        self.ypos = qt.QLineEdit(toolbar)
        self.ypos.setText('------')
        self.ypos.setReadOnly(1)
        self.ypos.setFixedWidth(self.ypos.fontMetrics().width('#########'))
        """
        label=qt.QLabel(toolbar)
        label.setText('<b>Energy:</b>')
        self.epos = qt.QLineEdit(toolbar)
        self.epos.setText('------')
        self.epos.setReadOnly(1)
        self.epos.setFixedWidth(self.epos.fontMetrics().width('########'))
        """
        HorizontalSpacer(toolbar)
        # ---print
        if 0:
            tb      = qt.QToolButton(toolbar)
            tb.setIconSet(self.printIcon)
            self.connect(tb,qt.SIGNAL('clicked()'),self.graph.printps)
            qt.QToolTip.add(tb,'Prints the Graph') 


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
        self.fitmenu.exec_loop(self.cursor().pos())

    def __saveIconSignal(self):
        legend = self.graph.getactivecurve(justlegend=1)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            msg.exec_loop()
            return
        #get outputfile
        outfile = qt.QFileDialog(self,"Output File Selection",1)
        #outfile.addFilter('Specfile MCA  *.mca')
        #outfile.addFilter('Specfile Scan *.dat')
        #outfile.addFilter('Raw ASCII  *.txt')
        outfile.setFilters('Specfile MCA  *.mca\nSpecfile Scan *.dat\nRaw ASCII  *.txt')
         
        outfile.setMode(outfile.AnyFile)
        ret = outfile.exec_loop()
        if ret:
            filterused = str(outfile.selectedFilter()).split()
            filetype  = filterused[1]
            extension = filterused[2]
            self.outdir=str(outfile.selectedFile())
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
        legend,x,y = self.graph.getactivecurve()
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.exec_loop()
           return
        if self.calibration == 'None':
            info,x,y = self.getinfodatafromlegend(legend)
        else:
            info,xdummy,ydummy = self.getinfodatafromlegend(legend)
        self.advancedfit.hide()
        self.simplefit.show()
        self.simplefit.setFocus()
        self.simplefit.raiseW()
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
                
        self.simplefit.hide()
        self.advancedfit.show()
        self.advancedfit.setFocus()
        self.advancedfit.raiseW()
        if info is not None:
            xlabel = 'Channel'
            self.advancedfit.setdata(x=x,y=y,
                                     xmin=xmin,
                                     xmax=xmax,
                                     legend=legend,
                                     xlabel=xlabel,
                                     calibration=calib,
                                     sourcename=curveinfo['sel']['SourceName'])
            self.advancedfit.fit()
        else:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error. Trying to fit fitted data?")
            msg.exec_loop()

    def __anasignal(self,dict):
        if DEBUG:
            print "__anasignal called dict = ",dict
            
        if dict['event'] == 'clicked':
            # A button has been cicked
            if   dict['button'] == 'Source':
                if dict['box'][1]   == 'SpecFile':
                    if self.inputdict['SpecFile']['widget'] is None:
                        if PYDVT:
                            d = SpecFileData.SpecFileData()
                        else:
                            d = SpecFileLayer.SpecFileLayer()    
                        self.inputdict['SpecFile']['widget'] = SpecFileSelector.SpecFileSelector()
                        self.inputdict['SpecFile']['widget'].setData(d)
                        self.inputdict['SpecFile']['widget'].show()
                        self.inputdict['SpecFile']['widget'].raiseW()
                        self.connect(self.inputdict['SpecFile']['widget'],
                                     qt.PYSIGNAL("removeSelection"),
                                     self.__specfileremove)
                        self.connect(self.inputdict['SpecFile']['widget'],
                                     qt.PYSIGNAL("addSelection"),
                                     self.__specfileadd)
                        self.connect(self.inputdict['SpecFile']['widget'],
                                     qt.PYSIGNAL("replaceSelection"),
                                     self.__specfilereplace)
                    else:
                        self.inputdict['SpecFile']['widget'].show()
                        self.inputdict['SpecFile']['widget'].setFocus()
                        self.inputdict['SpecFile']['widget'].raiseW()
                elif dict['box'][1]   == 'EDF File':
                    if self.inputdict['EdfFile']['widget'] is None:
                        if PYDVT:
                            d = EdfFileData.EdfFileData()
                        else:
                            d = EdfFileLayer.EdfFileLayer()    
                        self.inputdict['EdfFile']['widget'] = EdfFileSelector.EdfFileSelector()
                        self.inputdict['EdfFile']['widget'].setData(d)
                        self.inputdict['EdfFile']['widget'].show()
                        self.inputdict['EdfFile']['widget'].raiseW()
                        self.connect(self.inputdict['EdfFile']['widget'],
                                     qt.PYSIGNAL("removeSelection"),
                                     self.__edffileremove)
                        self.connect(self.inputdict['EdfFile']['widget'],
                                     qt.PYSIGNAL("addSelection"),
                                     self.__edffileadd)
                        self.connect(self.inputdict['EdfFile']['widget'],
                                     qt.PYSIGNAL("replaceSelection"),
                                     self.__edffilereplace)
                    else:
                        self.inputdict['EdfFile']['widget'].show()
                        self.inputdict['EdfFile']['widget'].setFocus()
                        self.inputdict['EdfFile']['widget'].raiseW()
                elif (dict['box'][1] == 'SPS') or (dict['box'][1] == 'SPEC'):
                    if self.inputdict['SPS']['widget'] is None:
                        if PYDVT:
                            d = SPSData.SPSData()
                        else:
                            d = SPSLayer.SPSLayer()    
                        #self.inputdict['SPS']['widget'] = SPSSelector.SPSSelector(fl=qt.Qt.WType_TopLevel)
                        #self.inputdict['SPS']['widget'] = SPSSelector.SPSSelector(fl=qt.Qt.WStyle_StaysOnTop)
                        self.inputdict['SPS']['widget'] = SPSSelector.SPSSelector()
                        self.inputdict['SPS']['widget'].setData(d)
                        self.inputdict['SPS']['widget'].show()
                        self.inputdict['SPS']['widget'].setFocus()
                        self.inputdict['SPS']['widget'].raiseW()
                        self.connect(self.inputdict['SPS']['widget'],
                                     qt.PYSIGNAL("removeSelection"),
                                     self.__spsremove)
                        self.connect(self.inputdict['SPS']['widget'],
                                     qt.PYSIGNAL("addSelection"),
                                     self.__spsadd)
                        self.connect(self.inputdict['SPS']['widget'],
                                     qt.PYSIGNAL("replaceSelection"),
                                     self.__spsreplace)
                    else:
                        self.inputdict['SPS']['widget'].show()
                        self.inputdict['SPS']['widget'].setFocus()
                        self.inputdict['SPS']['widget'].raiseW()
                else:
                    print dict['box'][1],' option not implemented' 
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
                            self.ana.calbox.setoptions(options)
                        except:
                            pass
                        try:
                            self.control.calbox.setoptions(options)
                        except:
                            pass 
                        self.control.calbox.setCurrentItem(item)
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
                            self.ana.calbox.setoptions(options)
                        except:
                            pass
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
                    msg.exec_loop() 
                    return
                cald = ConfigDict.ConfigDict()
                try:
                    cald.read(filename)
                except:
                    text = "Error. Cannot read calibration file %s" % filename
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
                    msg.exec_loop() 
                    return
                self.caldict.update(cald)
                options = []
                for option in self.calboxoptions:
                    options.append(option)
                for key in self.caldict.keys():
                    if key not in options:
                        options.append(key)
                try:
                    self.ana.calbox.setoptions(options)
                except:
                    pass
                try:
                    self.control.calbox.setoptions(options)
                    self.control.calbox.setCurrentItem(options.index(itemtext))
                    self.calibration = itemtext * 1
                    self.control._calboxactivated(itemtext)
                except:
                    text = "Error. Problem updating combobox"
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
                    msg.exec_loop() 
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
                        msg.exec_loop() 
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
        elif dict['event'] == 'ScanFit':
                x    = dict['x']
                y    = dict['y']
                info = dict['sel']
                xmin = dict['xmin']
                xmax = dict['xmax']
                legend = dict['legend']
                self.advancedfit.hide()
                self.simplefit.hide()
                self.scanfit.show()
                self.scanfit.setFocus()
                self.scanfit.raiseW()
                if info is not None:
                    xmin,xmax=self.scanwindow.graph.getx1axislimits()
                    self.scanfit.setdata(x=x,y=y,
                                            xmin=xmin,
                                            xmax=xmax,
                                            legend=legend)
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
                    self.specfit.fitconfig['McaMode'] = 0
                    self.scanfit.estimate()
                    self.scanfit.fit()
                else:
                        msg = qt.QMessageBox(self)
                        msg.setIcon(qt.QMessageBox.Critical)
                        msg.setText("Error. Trying to fit fitted data?")
                        msg.exec_loop() 
        elif dict['event'] == 'ScanSave':
                x    = dict['x']
                y    = dict['y']
                info = dict['sel']
                legend0 = dict['legend']
                legend = legend0 + " '"            
                curveinfo=self.scanwindow.graph.getcurveinfo(legend0)
                curveinfo['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
                        #get outputfile
                outfile = qt.QFileDialog(self,"Output File Selection",1)
                outfile.setFilters('Raw ASCII  *.txt\nSpecfile Scan *.dat\nSpecfile MCA  *.mca')
                outfile.setMode(outfile.AnyFile)
                ret = outfile.exec_loop()
                if ret:
                    filterused = str(outfile.selectedFilter()).split()
                    filetype  = filterused[1]
                    extension = filterused[2]
                    self.outdir=str(outfile.selectedFile())
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
                xlabel = self.scanwindow.graph.xlabel()
                ylabel = self.scanwindow.graph.ylabel()
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
                if filetype == 'Scan':
                    file=open(specFile,'w+')
                    file.write("#F %s\n" % specFile)
                    file.write("#D %s\n"%(time.ctime(time.time())))
                    file.write("\n")
                    file.write("#S 1 %s\n" % legend)
                    file.write("#D %s\n"%(time.ctime(time.time())))
                    file.write("#N 2\n")
                    file.write("#L %s  %s\n" % (xlabel, ylabel))
                    for i in range(len(y)):
                        file.write("%.7g  %.7g\n" % (x[i], y[i]))
                    file.write("\n")
                elif filetype == 'ASCII':
                    file=open(specFile,'w+')
                    for i in range(len(y)):
                        file.write("%.7g  %.7g\n" % (x[i], y[i]))
                else:
                    file=open(specFile,'w+')
                    file.write("#F %s\n" % specFile)
                    file.write("#D %s\n"%(time.ctime(time.time())))
                    file.write("\n")
                    file.write("#S 1 %s\n" % legend)
                    file.write("#D %s\n"%(time.ctime(time.time())))
                    file.write("#@MCA %16C\n")
                    file.write("#@CHANN %d %d %d 1\n" %  (len(y), x[0], x[-1]))
                    file.write("#@CALIB 0.0 1.0 0.0\n")
                    file.write(self.array2SpecMca(y))
                    file.write("\n")
                file.close()
        elif dict['event'] == 'Derive':
                x    = dict['x']
                y    = dict['y']
                info = dict['sel']
                xmin = dict['xmin']
                xmax = dict['xmax']
                legend0 = dict['legend']
                legend = legend0 + " '"            
                curveinfo=self.scanwindow.graph.getcurveinfo(legend0)
                curveinfo['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
                sourcetype = 'ScanFit'
                curveinfo['sel']['SourceType']= sourcetype
                curveinfo['sel']['Key']       = legend
                self.inputdict[sourcetype]['widget'].setData(sourcename=curveinfo['sel']['SourceName'],                                                         
                                                         info = curveinfo,
                                                         data = Numeric.reshape(Numeric.concatenate((x,y),0),(2,len(x))),
                                                         key  = legend)

                self.inputdict[sourcetype]['widget'].setSelected([curveinfo['sel']],reset=0)

            
        elif dict['event'] == 'activated':
            # A comboBox has been selected
            if   dict['boxname'] == 'Source':
                pass
            elif dict['boxname'] == 'Calibration':
                self.calibration = dict['box'][1]
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
                legend3 = dict['info']['legend'] + "Matrix"
                ymatrix   = dict['result']['ymatrix'] * 1.0
                #copy the original info from the curve
                curveinfo3=self.graph.getcurveinfo(legend0)
                curveinfo3['sel']['SourceName']= curveinfo3['sel']['SourceType']+":"+curveinfo3['sel']['SourceName']
                curveinfo3['sel']['SourceType']= 'AdvancedFit'
                curveinfo3['sel']['Key']       = dict['info']['legend']+"Matrix"
                legend2 = dict['info']['legend']+"B"
                self.inputdict['AdvancedFit']['widget'].setData(sourcename=curveinfo3['sel']['SourceName'],
                                                        info = curveinfo3,
                                                        data = Numeric.reshape(Numeric.concatenate((x,ymatrix,yb),0),(3,len(x))),
                                                        key  = legend3)                                                             
                self.inputdict['AdvancedFit']['widget'].setSelected([curveinfo3['sel']],reset=0)
                legend = dict['info']['legend'] + "A"
            else:
                legend = dict['info']['legend'] + "A"
                yfit   = dict['result']['yfit'] * 1.0
                #copy the original info from the curve
                curveinfo=self.graph.getcurveinfo(legend0)
                curveinfo['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
                curveinfo['sel']['SourceType']= 'AdvancedFit'
                curveinfo['sel']['Key']       = legend

                self.inputdict['AdvancedFit']['widget'].setData(sourcename=curveinfo['sel']['SourceName'],                                                         
                                                                info = curveinfo,
                                                                data = Numeric.reshape(Numeric.concatenate((x,yfit,yb),0),(3,len(x))),
                                                                key  = legend)
                self.inputdict['AdvancedFit']['widget'].setSelected([curveinfo['sel']],reset=0)

                curveinfo2=self.graph.getcurveinfo(legend0)
                curveinfo2['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
                curveinfo2['sel']['SourceType']= 'AdvancedFit'
                curveinfo2['sel']['Key']       = dict['info']['legend']+"B"
                legend2 = dict['info']['legend']+"B"
                self.inputdict['AdvancedFit']['widget'].setData(sourcename=curveinfo2['sel']['SourceName'],
                                                        info = curveinfo2,
                                                        data = Numeric.reshape(Numeric.concatenate((x,yb,yb),0),(3,len(x))),
                                                                key  = legend2)                                                             
                self.inputdict['AdvancedFit']['widget'].setSelected([curveinfo2['sel']],reset=0)
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
                self.ana.calbox.setoptions(options)
            except:
                pass
            try:
                self.control.calbox.setoptions(options)
                #if dict['event'] == 'McaAdvancedFitFinished':
                if 1:
                    self.control.calbox.setCurrentItem(options.index(legend))
                    self.calibration = legend
                    self.control._calboxactivated(legend)
            except:
                pass                        

        elif (dict['event'] == 'FitFinished'):
            #fit events
            x    = self.specfit.xdata * 1.0
            yfit = self.specfit.gendata(parameters=dict['data'])
            nparb= len(self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1])
            yb   = self.specfit.gendata(x=x,parameters=dict['data'][0:nparb])
            legend0= dict['info']['legend']
            legend = legend0 + " Fit"            
            curveinfo=self.scanwindow.graph.getcurveinfo(legend0)
            curveinfo['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
            sourcetype = 'ScanFit'
            curveinfo['sel']['SourceType']= sourcetype
            curveinfo['sel']['Key']       = legend
            self.inputdict[sourcetype]['widget'].setData(sourcename=curveinfo['sel']['SourceName'],                                                         
                                                         info = curveinfo,
                                                         data = Numeric.reshape(Numeric.concatenate((x,yfit,yb),0),(3,len(x))),
                                                         key  = legend)

            self.inputdict[sourcetype]['widget'].setSelected([curveinfo['sel']],reset=0)
            
            #self.graph.newcurve(legend,x=x,y=yfit,logfilter=1)
            #self.scanwindow.graph.newcurve(legend,x=x,y=yfit,logfilter=0,curveinfo=curveinfo)
            #self.scanwindow.graph.replot()
                        
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
            legend = legend0 + " Fit"            
            curveinfo=self.graph.getcurveinfo(legend0)
            curveinfo['sel']['SourceName']= curveinfo['sel']['SourceType']+":"+curveinfo['sel']['SourceName']
            curveinfo['sel']['SourceType']= 'SimpleFit'
            curveinfo['sel']['Key']       = legend
            curveinfo['Regions']          = regions
            curveinfo['CalMode']          = self.__simplefitcalmode
            x    = Numeric.array(xfinal)
            yfit = Numeric.array(yfinal)
            yb   = Numeric.array(ybfinal)
            self.inputdict['SimpleFit']['widget'].setData(sourcename=curveinfo['sel']['SourceName'],                                                         
                                                            info = curveinfo,
                                                            data = Numeric.reshape(Numeric.concatenate((x,yfit,yb),0),(3,len(x))),
                                                            key  = legend)
            self.inputdict['SimpleFit']['widget'].setSelected([curveinfo['sel']],reset=0)

            if 0:
                self.graph.newcurve(legend,x=Numeric.array(xfinal),
                                    y=Numeric.array(yfinal),
                                    logfilter=1,
                                    regions=regions,
                                    baseline=Numeric.array(ybfinal),curveinfo=curveinfo)
                #self.graph.setxofy(legend)
                self.graph.replot()
        elif dict['event'] == 'McaTableFilled':
            if self.peakmarker is not None:
                self.graph.removemarker(self.peakmarker)
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
                    self.graph.removemarker(self.peakmarker)
                self.peakmarker = None
                
        elif dict['event'] == 'McaTableClicked':
            if self.peakmarker is not None:
                self.graph.removemarker(self.peakmarker)
            self.peakmarker = None
            self.graph.replot()    

        elif (dict['event'] == 'McaAdvancedFitElementClicked') or (dict['event'] == 'ElementClicked'):
            for marker in self.elementmarkers:
                self.graph.removemarker(marker)
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
            self.printhtml(dict['text'])

        elif dict['event'] == 'McaSimpleFitPrint':
            self.printhtml(dict['text'])

        elif dict['event'] == 'ScanFitPrint':
            self.printhtml(dict['text'])

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
        self.graph.clearcurves()
        for key in self.inputdict.keys():
            if self.inputdict[key]['widget'] is not None:
                if len(self.inputdict[key]['sel']):
                    self.__add(self.inputdict[key]['sel'])
        self.graph.setactivecurve(activecurve)

        
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
                        if key == 'ICR':
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
            if dict.has_key('sel'):
                sel = dict['sel']
            else:
                sel = self.getselfromlegend(legend)
            if sel == {}:
                print "Empty selection?????"
                return
            if DEBUG:
                print "calling to remove sel = ",sel
            self.inputdict[sel['SourceType']]['widget'].removeSelection([sel])
            self.graph.replot()
            self.scanwindow.graph.replot()
        else:
            if DEBUG:
                print "Unhandled event ",   dict['event']   


    def __add(self,selection):
        if DEBUG:
            print "__add, selection = ",selection
        if len(selection):
            if selection[0].has_key('SourceType'):
                if selection[0]['SourceType'] == 'SPS':
                    self.__spsadd(selection)
                elif selection[0]['SourceType'] == 'SpecFile':
                    self.__specfileadd(selection)            
                elif selection[0]['SourceType'] == 'EdfFile':
                    self.__edffileadd(selection)
                elif selection[0]['SourceType'] == 'AdvancedFit':
                    self.__advancedfitadd(selection)
                elif selection[0]['SourceType'] == 'SimpleFit':
                    self.__advancedfitadd(selection)
                elif (selection[0]['SourceType'] == 'ScanFit') or (selection[0]['SourceType'] == 'Derive'):
                    self.__scanfitadd(selection)
                else:
                    print "Unknown selection source ",selection[0]['SourceType']
    
    def __remove(self,selection):
        if len(selection):
            if selection[0].has_key('SourceType'):
                if DEBUG:
                    print "__remove ",selection
                if selection[0]['SourceType'] == 'SPS':
                    self.__spsremove(selection)
                elif selection[0]['SourceType'] == 'SpecFile':
                    self.__specfileremove(selection)            
                elif selection[0]['SourceType'] == 'EdfFile':
                    self.__edffileremove(selection)
                elif selection[0]['SourceType'] == 'AdvancedFit':
                    self.__advancedfitremove(selection)
                elif selection[0]['SourceType'] == 'SimpleFit':
                    self.__advancedfitremove(selection)
                elif (selection[0]['SourceType'] == 'ScanFit') or (selection[0]['SourceType'] == 'Derive'):
                    self.__scanfitremove(selection)
                else:
                    print "Unknown selection source ",selection[0]['SourceType']
    
    def __replace(self,selection):
        self.__advancedfitremove(self.inputdict['AdvancedFit']['sel'], replot=False)
        for sel in self.inputdict['AdvancedFit']['sel']:
              self.inputdict['AdvancedFit']['widget'].removeSelection([sel])                
        if len(selection):
            if selection[0].has_key('SourceType'):
                if selection[0]['SourceType'] == 'SPS':
                    self.__spsreplace(selection)
                elif selection[0]['SourceType'] == 'SpecFile':
                    self.__specfilereplace(selection)            
                elif selection[0]['SourceType'] == 'EdfFile':
                    self.__edffilereplace(selection)
                elif selection[0]['SourceType'] == 'AdvancedFit':
                    self.__advancedfitreplace(selection)
                elif selection[0]['SourceType'] == 'SimpleFit':
                    self.__advancedfitreplace(selection)
                elif (selection[0]['SourceType'] == 'ScanFit') or (selection[0]['SourceType'] == 'Derive'):
                    self.__scanfitreplace(selection)
                else:
                    print "Unknown selection source ",selection[0]['SourceType']


            
    def __specfileadd(self,selection0):
        if DEBUG:
            print "__specfileadd",selection0
        #self.inputdict['SpecFile']={'widget':None,'sel':[]}
        selection = copy.deepcopy(selection0)
        sourcetype = 'SpecFile'
        for sel in selection:
            source = sel['SourceName']
            key    = sel['Key']
            for mca in sel[key]['mca']:
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['mca'] = [mca]
                curvesel[key]['scan'] = {}
                curveinfo['sel']      = curvesel
                #select the source
                self.inputdict[sourcetype]['widget'].data.SetSource(source)
                legend = os.path.basename(source)+" "+mca
                #get the data
                if PYDVT:
                    self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = mca)
                    data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                    info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
                else:
                    info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = mca)
                calib = [0.0,1.0,0.0]
                curveinfo['McaCalib'] = calib
                if info.has_key('McaCalib'):
                    if type(info['McaCalib'][0]) == type([]):
                        calib0 = info['McaCalib'][info['McaDet']-1]
                    else:
                        calib0 = info['McaCalib']
                    curveinfo['McaCalibSource'] = calib0
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
                        self.graph.newcurve(legend,
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
                    pass
                elif self.calibration in  self.caldict.keys():
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=data,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    self.graph.newcurve(legend,x=xhelp,y=data,
                                        logfilter=1, curveinfo=curveinfo)
                    self.graph.xlabel('Channel')

            if sel[key].has_key('scan'):
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['mca'] = []
                curveinfo['sel']      = curvesel
                if sel[key]['scan'].has_key('Ycnt'):
                    if len(sel[key]['scan']['Ycnt']):
                        self.inputdict[sourcetype]['widget'].data.SetSource(source)                        
                        info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)                    
                    if len(sel[key]['scan']['Xcnt']):
                        xcnt   = sel[key]['scan']['Xcnt'][0]
                        xindex = info['LabelNames'].index(xcnt)
                        xdata  = data[:,xindex]
                    else:
                        xcnt = "-"
                        
                    if len(sel[key]['scan']['Mcnt']):
                        mcnt   = sel[key]['scan']['Mcnt'][0]
                        mindex = info['LabelNames'].index(mcnt)
                        mdata  = data[:,mindex]
                    else:
                        mcnt = "-"
                    
                    for ycnt in sel[key]['scan']['Ycnt']:
                        yindex = info['LabelNames'].index(ycnt)
                        ydata  = data[:,yindex]
                        legend = os.path.basename(source)+" s"+key+":"+xcnt+":"+ycnt+":"+mcnt
                        if len(ydata):
                            if mcnt is not "-":
                                try:
                                    ydata = ydata / mdata
                                except:
                                    msg = qt.QMessageBox(self)
                                    msg.setIcon(qt.QMessageBox.Critical)
                                    msg.setText("Monitor Error: %s" % (sys.exc_info()[1]))
                                    msg.exec_loop()
                                    break
                        try:
                            if legend not in self.scanwindow.graph.curveslist:
                                isnewscancurve=1
                            else:
                                isnewscancurve=0
                            if xcnt is "-":
                                self.scanwindow.graph.newcurve(legend,y=ydata,curveinfo=curveinfo)
                            else:
                                self.scanwindow.graph.newcurve(legend,x=xdata,y=ydata,curveinfo=curveinfo)
                            if isnewscancurve:
                                self.emit(qt.PYSIGNAL('McaWindowSignal'),({'event':'NewScanCurve',
                                                                           'legend':legend},))
                        except:
                             msg = qt.QMessageBox(self)
                             msg.setIcon(qt.QMessageBox.Critical)
                             msg.setText("Scan Plot Error: %s" % (sys.exc_info()[1]))
                             msg.exec_loop()
                             break
                            
        if self.inputdict[sourcetype]['widget'] is not None:
            self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        self.graph.replot()
        self.scanwindow.graph.replot()
        
    def printhtml(self,text):
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


    def __specfileremove(self,selection):
        if DEBUG:
            print "__specfileremove",selection
        for sel in selection:
            sourcetype=sel['SourceType']
            source = sel['SourceName']
            key    = sel['Key']
            for mca in sel[key]['mca']:
                self.graph.delcurve(os.path.basename(source)+" "+mca)
                
            #scan
            if sel[key].has_key('scan'):
                if sel[key]['scan'].has_key('Ycnt'):
                    if len(sel[key]['scan']['Ycnt']):
                        self.inputdict[sourcetype]['widget'].data.SetSource(source)                        
                        info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)                    
                    if len(sel[key]['scan']['Xcnt']):
                        xcnt   = sel[key]['scan']['Xcnt'][0]
                        xindex = info['LabelNames'].index(xcnt)
                        xdata  = data[:,xindex]
                    else:
                        xcnt = "-"
                    if len(sel[key]['scan']['Mcnt']):
                        mcnt   = sel[key]['scan']['Mcnt'][0]
                        mindex = info['LabelNames'].index(mcnt)
                        mdata  = data[:,mindex]
                    else:
                        mcnt = "-"
                    for ycnt in sel[key]['scan']['Ycnt']:
                        yindex = info['LabelNames'].index(ycnt)
                        ydata  = data[:,yindex]
                        legend = os.path.basename(source)+" s"+key+":"+xcnt+":"+ycnt+":"+mcnt
                        self.scanwindow.graph.delcurve(legend)
                        self.scanwindow.graph.replot()

        self.inputdict["SpecFile"]['sel']=self.inputdict["SpecFile"]['widget'].getSelection()

    def __specfilereplace(self,selection):
        if DEBUG:
            print "__specfilereplace",selection
        self.graph.clearcurves()
        self.scanwindow.graph.clearcurves()
        self.roimarkers=[-1,-1]
        for sel in selection:
            self.__specfileadd([sel])

    def __advancedfitadd(self,selection):
        if DEBUG:
            print "__advancedfitadd",selection
        for sel in selection:
            sourcetype= sel['SourceType']
            source    = sel['SourceName']
            key       = sel['Key']
            curveinfo={}
            curvesel={}
            curvesel.update(sel)
            curveinfo['sel']      = curvesel
            #select the source
            self.inputdict[sourcetype]['widget'].data.SetSource(source)
            legend = key
            info,data= self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
            xhelp    = data[0,:]
            ydata    = data[1,:]
            yb       = data[2,:]
            calib = [0.0,1.0,0.0]
            curveinfo['McaCalib'] = calib
            if info.has_key('McaCalib'):
                if type(info['McaCalib'][0]) == type([]):
                    calib0 = info['McaCalib'][info['McaDet']-1]
                else:
                    calib0 = info['McaCalib']
                curveinfo['McaCalibSource'] = calib0
            if info.has_key('Regions'):
                regions = info['Regions']
            else:
                regions = []
            if info.has_key('CalMode'):
                if info['CalMode'] != self.calibration:
                    if DEBUG:
                        print "calmode = ",info['CalMode']," current = ",self.calibration
                else:
                    if len(regions):
                        self.graph.newcurve(legend,
                                            x=xhelp,y=ydata,logfilter=1,
                                            curveinfo=curveinfo,baseline=yb,regions=regions)
                    else:
                        self.graph.newcurve(legend,
                                            x=xhelp,y=ydata,logfilter=1,
                                            curveinfo=curveinfo)                
            else:    
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
                        if len(regions):
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo,regions=regions)                    
                        else:
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo)
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
                        if len(regions):
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo,regions=regions)                    
                        else:
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                elif self.calibration == 'Fit':
                    print "Not yet implemented"
                    pass
                elif self.calibration in  self.caldict.keys():
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        if len(regions):
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo,regions=regions)                    
                        else:
                            self.graph.newcurve(legend,
                                                x=xdata,y=ydata,logfilter=1,
                                                curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    if len(regions):
                        self.graph.newcurve(legend,
                                            x=xhelp,y=ydata,logfilter=1,
                                            curveinfo=curveinfo,regions=regions)
                    else:
                        self.graph.newcurve(legend,
                                            x=xhelp,y=ydata,logfilter=1,
                                            curveinfo=curveinfo)
                    self.graph.xlabel('Channel')
        self.graph.replot()
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()


    def __advancedfitremove(self,selection, replot=None):
        if replot is None:replot=True
        if DEBUG:
            print "__advancedfitremove",selection
        for sel in selection:
            sourcetype= sel['SourceType']
            source    = sel['SourceName']
            key       = sel['Key']
            self.graph.delcurve(key)
        sourcetype = 'AdvancedFit'
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        if replot:self.graph.replot()


    def __advancedfitreplace(self,selection):
        if DEBUG:
            print "__specfilereplace",selection
        self.graph.clearcurves()
        self.roimarkers=[-1,-1]
        for sel in selection:
            self.__advancedfitadd([sel])


    def __scanfitadd(self,selection):
        if DEBUG:
            print "__scanfitadd",selection
        for sel in selection:
            sourcetype= sel['SourceType']
            source    = sel['SourceName']
            key       = sel['Key']
            curveinfo={}
            curvesel={}
            curvesel.update(sel)
            curveinfo['sel']      = curvesel
            #select the source
            self.inputdict[sourcetype]['widget'].data.SetSource(source)
            legend = key
            info,data= self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
            xhelp    = data[0,:]
            ydata    = data[1,:]
            if data.shape[0] > 2:
                yb       = data[2,:]
            if info.has_key('Regions'):
                regions = info['Regions']
            else:
                regions = []
            if len(regions):
                self.scanwindow.graph.newcurve(legend,
                                    x=xhelp,y=ydata,logfilter=0,
                                    curveinfo=curveinfo,regions=regions)
            else:
                self.scanwindow.graph.newcurve(legend,
                                    x=xhelp,y=ydata,logfilter=0,
                                    curveinfo=curveinfo)
            #if sourcetype == 'Derive':
            #    self.scanwindow.graph.maptoy2(legend)                   
        self.scanwindow.graph.replot()
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()


    def __scanfitremove(self,selection):
        if DEBUG:
            print "__scanfitremove",selection
        for sel in selection:
            sourcetype= sel['SourceType']
            source    = sel['SourceName']
            key       = sel['Key']
            self.scanwindow.graph.delcurve(key)
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        self.scanwindow.graph.replot()


    def __scanfitreplace(self,selection):
        if DEBUG:
            print "__scanfitreplace",selection
        self.scanwindow.graph.clearcurves()
        for sel in selection:
            self.__scanfitadd([sel])


    def __spsadd(self,selection0):
        if DEBUG:
            print "__spsadd",selection0
        sourcetype = 'SPS'
        selection = copy.deepcopy(selection0)
        
        for sel in selection:
            source = sel['SourceName']
            key    = sel['Key']
            for pair in sel[key]['rows']:
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['rows'] = [pair]
                curvesel[key]['cols'] = {}
                curveinfo['sel']      = curvesel

                #select the source
                self.inputdict[sourcetype]['widget'].data.SetSource(source)
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("r[-, %d]" % (pair['y']))
                #get the data
                if PYDVT:
                    self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                    data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                    info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
                else:
                    info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)                
                ydata = data[pair['y'],:]
                if pair['x'] is not None:
                    xdata = data[pair['x'],:]
                else:
                    if info.has_key('Channel0'):
                        xdata = info['Channel0']  + \
                            Numeric.arange(len(ydata)).astype(Numeric.Float)
                    else:
                        xdata =  Numeric.arange(len(ydata)).astype(Numeric.Float)  
                calib = [0.0,1.0,0.0]
                if info.has_key('McaCalib'):
                    if type(info['McaCalib'][0]) == type([]):
                        calib0 = info['McaCalib'][info['McaDet']-1]
                    else:
                        calib0 = info['McaCalib']
                    curveinfo['McaCalibSource'] = calib0                    
                if self.calibration == self.calboxoptions[1]:
                    if info.has_key('McaCalib'):
                        if type(info['McaCalib'][0]) == type([]):
                            calib = info['McaCalib'][info['McaDet']-1]
                        else:
                            calib = info['McaCalib']
                    if len(calib) > 1:
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        xdata=calib[0]+ \
                              calib[1]* xhelp
                        if len(calib) == 3:
                              xdata = xdata + calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
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
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        xdata=calib[0]+ \
                              calib[1]* xhelp
                        if len(calib) == 3:
                              xdata = xdata + calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                elif self.calibration == 'Fit':
                    print "Not yet implemented"
                    pass
                elif self.calibration in  self.caldict.keys():
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    curveinfo['McaCalib'] = calib
                    self.graph.newcurve(legend,x=xdata,y=ydata,
                                        logfilter=1, curveinfo=curveinfo)
                    self.graph.xlabel('Channel')
            for pair in sel[key]['cols']:
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['cols'] = [pair]
                curvesel[key]['rows'] = {}
                curveinfo['sel']      = curvesel
                #select the source
                self.inputdict[sourcetype]['widget'].data.SetSource(source)
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                #get the data
                if PYDVT:
                    self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                    data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                    info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
                else:
                    info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                ydata = data[:,pair['y']]
                if pair['x'] is not None:
                    xdata = data[:,pair['x']]
                else:
                    xdata=Numeric.arange(len(ydata)).astype(Numeric.Float)
                calib = [0.0,1.0,0.0]
                if info.has_key('McaCalib'):
                    if type(info['McaCalib'][0]) == type([]):
                        calib0 = info['McaCalib'][info['McaDet']-1]
                    else:
                        calib0 = info['McaCalib']
                    curveinfo['McaCalibSource'] = calib0
                if self.calibration == self.calboxoptions[1]:
                    if info.has_key('McaCalib'):
                        if type(info['McaCalib'][0]) == type([]):
                            calib = info['McaCalib'][info['McaDet']-1]
                        else:
                            calib = info['McaCalib']
                    if len(calib) > 1:
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        xdata=calib[0]+ \
                              calib[1]* xhelp
                        if len(calib) == 3:
                              xdata = xdata + calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
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
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        xdata=calib[0]+ \
                              calib[1]* xhelp
                        if len(calib) == 3:
                              xdata = xdata + calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                elif self.calibration == 'Fit':
                    print "Not yet implemented"
                    pass
                elif self.calibration in  self.caldict.keys():
                        #xhelp = Numeric.arange(len(ydata)).astype(Numeric.Float)
                        xhelp = xdata * 1.0
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    curveinfo['McaCalib'] = calib
                    self.graph.newcurve(legend,x=xdata,y=ydata,
                                        logfilter=1, curveinfo=curveinfo)
                    self.graph.xlabel('Channel')
        if self.inputdict[sourcetype]['widget'] is not None:
            self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        if qt.qVersion() < '3.0.0':
            self.graph.replot()
        else:
            self.graph.replot()

    def __spsremove(self,selection):
        if DEBUG:
            print "__spsremove",selection
        sourcetype = 'SPS'
        for sel in selection:
            source = sel['SourceName']
            key    = sel['Key']
            for pair in sel[key]['cols']:
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                self.graph.delcurve(legend)

            for pair in sel[key]['rows']:
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("r[-,%d]" % (pair['y']))
                self.graph.delcurve(legend)
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()

    def __spsreplace(self,selection):
        if DEBUG:
            print "__spsreplace",selection
        self.graph.clearcurves()
        #self.roimarkers=[-1,-1]
        for sel in selection:
            self.__spsadd([sel])

    def __edffileadd(self,selection0):
        if DEBUG:
            print "__edfadd",selection0
        sourcetype = 'EdfFile'
        selection=copy.deepcopy(selection0)
        for sel in selection:
            source = sel['SourceName']
            key    = sel['Key']
            for pair in sel[key]['rows']:
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['rows'] = [pair]
                curvesel[key]['cols'] = []
                curveinfo['sel']      = curvesel
                #select the source
                self.inputdict[sourcetype]['widget'].data.SetSource(source)
                """
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("r[-, %d]" % (pair['y']))
                """
                legend = os.path.basename(source)+" Image "+key+" "+("Row %d" %  pair['y'])
                #get the data
                if PYDVT:
                    self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                    data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                    info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
                else:
                    info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                ydata = data[pair['y'],:]
                #if pair['x'] is not None:
                #    xdata = data[pair['x'],:]
                xhelp = info['Channel0'] + Numeric.arange(len(ydata)).astype(Numeric.Float)
                calib = [0.0,1.0,0.0]
                if info.has_key('McaCalib'):
                    if type(info['McaCalib'][0]) == type([]):
                        calib0 = info['McaCalib'][info['McaDet']-1]
                    else:
                        calib0 = info['McaCalib']
                    curveinfo['McaCalibSource'] = calib0
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
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
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
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                elif self.calibration == 'Fit':
                    print "Not yet implemented"
                    pass
                elif self.calibration in  self.caldict.keys():
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    curveinfo['McaCalib'] = calib
                    self.graph.newcurve(legend,x=xhelp,y=ydata,
                                        logfilter=1, curveinfo=curveinfo)
                    self.graph.xlabel('Channel')
            for pair in sel[key]['cols']:
                curveinfo={}
                curvesel={}
                curvesel.update(sel)
                curvesel[key]['cols'] = [pair]
                curvesel[key]['rows'] = []
                curveinfo['sel']      = curvesel
                #select the source
                self.inputdict[sourcetype]['widget'].data.SetSource(source)
                """
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                """
                legend = os.path.basename(source)+" Image "+key+" "+("Col %d" %  pair['y'])                #get the data
                if PYDVT:
                    self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                    data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                    info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
                else:
                    info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                ydata = data[:,pair['y']]
                if pair['x'] is not None:
                    xdata = data[:,pair['x']]
                xhelp = info['Channel0'] + Numeric.arange(len(ydata)).astype(Numeric.Float)
                calib = [0.0,1.0,0.0]
                if info.has_key('McaCalib'):
                    if type(info['McaCalib'][0]) == type([]):
                        calib0 = info['McaCalib'][info['McaDet']-1]
                    else:
                        calib0 = info['McaCalib']
                    curveinfo['McaCalibSource'] = calib0
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
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
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
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                elif self.calibration == 'Fit':
                    print "Not yet implemented"
                    pass
                elif self.calibration in  self.caldict.keys():
                        A = self.caldict[self.calibration]['A']
                        B = self.caldict[self.calibration]['B']
                        C = self.caldict[self.calibration]['C']
                        calib = [A,B,C]
                        xdata=calib[0]+ \
                              calib[1]* xhelp + \
                              calib[2]* xhelp * xhelp
                        curveinfo['McaCalib'] = calib
                        self.graph.newcurve(legend,
                                            x=xdata,y=ydata,logfilter=1, curveinfo=curveinfo)
                        self.graph.xlabel('Energy')
                else:
                    curveinfo['McaCalib'] = calib
                    self.graph.newcurve(legend,x=xhelp,y=ydata,
                                        logfilter=1, curveinfo=curveinfo)
                    self.graph.xlabel('Channel')

        if self.inputdict[sourcetype]['widget'] is not None:
            self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        if qt.qVersion() < '3.0.0':
            self.graph.replot()
        else:
            self.graph.replot()

    def __edffileremove(self,selection):
        if DEBUG:
            print "__spsremove",selection
        sourcetype = 'EdfFile'
        for sel in selection:
            source = sel['SourceName']
            key    = sel['Key']
            for pair in sel[key]['cols']:
                """
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                """
                legend = os.path.basename(source)+" Image "+key+" "+("Col %d" %  pair['y'])
                self.graph.delcurve(legend)
                
            for pair in sel[key]['rows']:
                """
                if pair['x'] is not None:
                    legend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                else:
                    legend = source+" "+key+" "+("r[-,%d]" % (pair['y']))
                """
                legend = os.path.basename(source)+" Image "+key+" "+("Row %d" %  pair['y'])
                self.graph.delcurve(legend)
        self.inputdict[sourcetype]['sel']=self.inputdict[sourcetype]['widget'].getSelection()
        self.graph.replot()

    def __edffilereplace(self,selection):
        if DEBUG:
            print "__spsreplace",selection
        self.graph.clearcurves()
        self.roimarkers=[-1,-1]
        for sel in selection:
            self.__edffileadd([sel])


    def getselfromlegend(self,legend):
        if DEBUG:
            print "OLD getselfromlegend = ",self.OLDgetselfromlegend(legend)
            print "complete             = ",self.graph.getcurveinfo(legend)
            print "NEW getselfromlegend = ",self.graph.getcurveinfo(legend)['sel']
        return self.graph.getcurveinfo(legend)['sel']

    def OLDgetselfromlegend(self,legend):
        for sourcetype in self.inputdict:
            if self.inputdict[sourcetype].has_key('sel'):
                for sel in self.inputdict[sourcetype]['sel']:
                    if sourcetype == 'SpecFile':
                        source  = sel['SourceName']
                        key     = sel['Key']
                        for mca in sel[key]['mca']:
                            newlegend = os.path.basename(source)+" "+mca
                            if legend in newlegend:
                                if newlegend.index(legend) == 0:
                                    return {'SourceName':source,
                                            'SourceType':sourcetype,
                                            'Key':       key,
                                            key:{'mca':[mca],'scan':{}}}
                    elif sourcetype == 'SPS':
                        source  = sel['SourceName']
                        key     = sel['Key']
                        for pair in sel[key]['cols']:
                            if pair['x'] is not None:
                                newlegend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                            else:
                                newlegend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                            if legend in newlegend:
                                if newlegend.index(legend) == 0:
                                    return {'SourceName':source,
                                            'SourceType':sourcetype,
                                            'Key':       key,
                                            key:{'cols':[pair],'rows':[]}}
                        for pair in sel[key]['rows']:
                            if pair['x'] is not None:
                                newlegend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                            else:
                                newlegend = source+" "+key+" "+("r[-,%d]" % (pair['y']))
                            if legend in newlegend:
                                if newlegend.index(legend) == 0:
                                    return {'SourceName':source,
                                            'SourceType':sourcetype,
                                            'Key':       key,
                                            key:{'rows':[pair],'cols':[]}} 
                    elif sourcetype == 'EdfFile':
                        source  = sel['SourceName']
                        key     = sel['Key']
                        for pair in sel[key]['cols']:
                            #if pair['x'] is not None:
                            #    newlegend = source+" "+key+" "+("c[%d,%d]" % (pair['x'],pair['y']))
                            #else:
                            #    newlegend = source+" "+key+" "+("c[-,%d]" % (pair['y']))
                            newlegend = os.path.basename(source)+" Image "+key+" "+("Col %d" %  pair['y'])
                            if legend in newlegend:
                                if newlegend.index(legend) == 0:
                                    return {'SourceName':source,
                                            'SourceType':sourcetype,
                                            'Key':       key,
                                            key:{'cols':[pair],'rows':[]}}
                        for pair in sel[key]['rows']:
                            #if pair['x'] is not None:
                            #    newlegend = source+" "+key+" "+("r[%d,%d]" % (pair['x'],pair['y']))
                            #else:
                            #    newlegend = source+" "+key+" "+("r[-,%d]" % (pair['y']))
                            newlegend = os.path.basename(source)+" Image "+key+" "+("Row %d" %  pair['y'])
                            if legend in newlegend:
                                if newlegend.index(legend) == 0:
                                    return {'SourceName':source,
                                            'SourceType':sourcetype,
                                            'Key':       key,
                                            key:{'rows':[pair],'cols':[]}} 
        return {}
    
    
    def getinfodatafromlegend(self,legend,full=0):
        sel = self.getselfromlegend(legend)
        if sel == {}:
            info = None
            xdata    = None
            ydata    = None
            if full:
                return info,None
            else:
                return info,xdata,ydata 
        source     = sel['SourceName']
        sourcetype = sel['SourceType']
        key        = sel['Key']
        self.inputdict[sourcetype]['widget'].data.SetSource(source)
        if full:
            if PYDVT:
                self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
            else:
                info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
            return info,data

        if (sourcetype == 'AdvancedFit') or (sourcetype == 'SimpleFit') or (sourcetype == 'ScanFit'):
            info,data = self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
            xdata = data[0,:]
            ydata = data[1,:]
            return info,xdata,ydata
            
        if sourcetype == 'SpecFile':
            mca = sel[key]['mca'][0]            
            if PYDVT:
                self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = mca)
                data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
            else:
                info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = mca)
            if info.has_key('Channel0'):                
                xdata=info['Channel0'] + \
                      Numeric.arange(len(data)).astype(Numeric.Float)
            else:
                xdata = Numeric.arange(len(data)).astype(Numeric.Float)
            return info,xdata,data
        
        if sourcetype == 'SPS':
            if PYDVT:
                self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
            else:
                info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)        
            for pair in sel[key]['cols']:
                ydata = data[:,pair['y']]
                if pair['x'] is not None:
                    xdata = data[:,pair['x']]
                else:
                    if info.has_key('Channel0'):                
                        xdata=info['Channel0'] + \
                              Numeric.arange(len(ydata)).astype(Numeric.Float)
                    else:
                        xdata = Numeric.arange(len(ydata)).astype(Numeric.Float)
                return info,xdata,ydata
            for pair in sel[key]['rows']:
                ydata = data[pair['y'],:]
                if pair['x'] is not None:
                    xdata = data[pair['x'],:]
                else:
                    if info.has_key('Channel0'):                
                        xdata=info['Channel0'] + \
                              Numeric.arange(len(ydata)).astype(Numeric.Float)
                    else:
                        xdata = Numeric.arange(len(ydata)).astype(Numeric.Float)
                return info,xdata,ydata

        if sourcetype == 'EdfFile':
            if PYDVT:
                self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)
                data=self.inputdict[sourcetype]['widget'].data.GetPageArray(0)
                info=self.inputdict[sourcetype]['widget'].data.GetPageInfo(0)
            else:
                info,data=self.inputdict[sourcetype]['widget'].data.LoadSource(key_list = key)        
            for pair in sel[key]['cols']:
                ydata = data[:,pair['y']]
                #if pair['x'] is not None:
                #    xdata = data[:,pair['x']]
                #else:
                xdata = info['Channel0']+Numeric.arange(len(ydata)).astype(Numeric.Float)
                return info,xdata,ydata
            for pair in sel[key]['rows']:
                ydata = data[pair['y'],:]
                #if pair['x'] is not None:
                #    xdata = data[pair['x'],:]
                #else:
                xdata = info['Channel0']+Numeric.arange(len(ydata)).astype(Numeric.Float)
                return info,xdata,ydata


        
    def getinfo(self,info):
        sourcetype = info['SourceType']
        if  sourcetype == 'SpecFile':
            nbmca      = info['NbMcaDet']
            mcacurrent = info['McaDet']
            header     = info['Header']
            calib      = info['McaCalib']
            
    def __spssurvey__(self,period):
        period = period * 0.001
        while not self.closeThread.isSet():
            try:
                self.closeThread.wait(period)
                insel = self.inputdict['SPS']['sel']
                if self.inputdict['SPS']['widget'] is not None:
                    outsel= self.inputdict['SPS']['widget'].isSelectionUpdated(insel)
                    if len(outsel):
                        mcaevent=McaCustomEvent.McaCustomEvent()
                        #mcaevent.dict={}
                        mcaevent.dict['event']     = 'addSelection'
                        mcaevent.dict['source']    = 'SPS'
                        mcaevent.dict['selection'] = outsel
                        #qt.qApp.processEvents()
                        qt.QApplication.postEvent(self, mcaevent)
                
            except:
                print "Error in thread"
                print sys.exc_info()[1]
        print "Thread Stopped"

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))
      

class SPSthread(qt.QThread):
    def __init__(self,parent,time=1000.):
        self.parent = parent
        self.time   = time * 0.001
        qt.QThread.__init__(self)
        
    def run(self):
        import McaCustomEvent
        while(1):
            try:
                time.sleep(self.time)
                insel = self.parent.inputdict['SPS']['sel']
                if self.parent.inputdict['SPS']['widget'] is not None:
                    outsel= self.parent.inputdict['SPS']['widget'].isSelectionUpdated(insel)
                    if len(outsel):
                        if 0:
                            qt.QObject.emit(self.parent.inputdict['SPS']['widget'],
                                    qt.PYSIGNAL("addSelection"),(outsel,))
                        mcaevent=McaCustomEvent.McaCustomEvent()
                        #mcaevent.dict={}
                        mcaevent.dict['event']     = 'addSelection'
                        mcaevent.dict['source']    = 'SPS'
                        mcaevent.dict['selection'] = outsel
                        qt.QApplication.postEvent(self.parent, mcaevent)
                
            except:
                if DEBUG:
                    try:
                        print "SPS Thread: Error in thread", sys.exc_info()[1]
                    except:
                        pass



class ScanWindow(qt.QWidget):
    def __init__(self, parent=None, name="Scan Window", specfit=None, fl=0,**kw):
        qt.QWidget.__init__(self, parent, name,fl)
        self.build()
        self.initIcons()
        self.initToolBar()
        self.setCaption(qt.QString(name))
        self.mathTools = SimpleMath.SimpleMath()


    def build(self):
        self.layout   = qt.QVBoxLayout(self)
        self.layout.setAutoAdd(1)
        self.toolbar  = qt.QHBox(self)
        self.graph    = QtBlissGraph.QtBlissGraph(self,uselegendmenu=1)
        self.graph.xlabel('X Counter')
        self.graph.ylabel('Y Counter')
        self.graph.canvas().setMouseTracking(1) 
        self.graph.setCanvasBackground(qt.Qt.white)
        self.connect(self.graph,
                     qt.PYSIGNAL('QtBlissGraphSignal'),
                     self.__graphsignal)

    def initIcons(self):
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
		self.fitIcon	= qt.QIconSet(qt.QPixmap(IconDict["fit"]))
		self.deriveIcon	= qt.QIconSet(qt.QPixmap(IconDict["derive"]))
		self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))
		self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
		self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))

    def initToolBar(self):
        toolbar = self.toolbar
        # AutoScale
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.zoomResetIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.graph.ResetZoom)
        qt.QToolTip.add(tb,'Auto-Scale the Graph') 
        # Logarithmic
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.logyIcon)
        tb.setToggleButton(1)
        self.connect(tb,qt.SIGNAL('clicked()'),self.graph.ToggleLogY)
        qt.QToolTip.add(tb,'Toggle Logarithmic Y Axis (On/Off)') 
        # Fit
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.fitIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.fitsignal)
        qt.QToolTip.add(tb,'Fit active curve') 
        # Derive
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.deriveIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.derivesignal)
        qt.QToolTip.add(tb,'Derivate Active Curve') 
        #save
        tb      = qt.QToolButton(toolbar)
        tb.setIconSet(self.saveIcon)
        self.connect(tb,qt.SIGNAL('clicked()'),self.__saveIconSignal)
        qt.QToolTip.add(tb,'Save Active Curve')

        # X and Y info
        HorizontalSpacer(toolbar)
        label=qt.QLabel(toolbar)
        #label.setText('<b>Channel:</b>')
        label.setText('<b>X:</b>')
        self.xpos = qt.QLineEdit(toolbar)
        self.xpos.setText('------')
        self.xpos.setReadOnly(1)
        self.xpos.setFixedWidth(self.xpos.fontMetrics().width('########'))
        label=qt.QLabel(toolbar)
        label.setText('<b>Y:</b>')
        self.ypos = qt.QLineEdit(toolbar)
        self.ypos.setText('------')
        self.ypos.setReadOnly(1)
        self.ypos.setFixedWidth(self.ypos.fontMetrics().width('#########'))
        HorizontalSpacer(toolbar)
        # ---print
        if 0:
            tb      = qt.QToolButton(toolbar)
            tb.setIconSet(self.printIcon)
            self.connect(tb,qt.SIGNAL('clicked()'),self.graph.printps)
            qt.QToolTip.add(tb,'Prints the Graph') 

    def fitsignal(self):
        #this are pure graphical operations
        #I should deal with them here
        legend,x,y = self.graph.getactivecurve()
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.exec_loop()
           return
        else:
            dict={}
            dict['event']  = "ScanFit"
            info = self.graph.getcurveinfo(legend)
            dict['x'] = x
            dict['y'] = y 
            xmin,xmax=self.graph.getx1axislimits()
            dict['xmin'] = xmin
            dict['xmax'] = xmax
            dict['sel'] = info['sel']
            dict['info'] = info
            dict['legend'] = legend
            self.emit(qt.PYSIGNAL("ScanWindowSignal"),(dict,))

    def derivesignal(self):
        #this are pure graphical operations
        #I should deal with them here
        legend,x,y = self.graph.getactivecurve()
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.exec_loop()
           return
        else:
            dict={}
            dict['event']  = "Derive"
            info = self.graph.getcurveinfo(legend)
            dict['x'], dict['y'] = self.mathTools.derivate(x,y)
            #dict['y'] = y 
            xmin,xmax=self.graph.getx1axislimits()
            dict['xmin'] = xmin
            dict['xmax'] = xmax
            dict['sel'] = info['sel']
            dict['info'] = info
            dict['legend'] = legend
            self.emit(qt.PYSIGNAL("ScanWindowSignal"),(dict,))

    def __saveIconSignal(self):
        #this are pure graphical operations
        #I should deal with them here
        legend,x,y = self.graph.getactivecurve()
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.exec_loop()
           return
        else:
            dict={}
            dict['event']  = "ScanSave"
            info = self.graph.getcurveinfo(legend)
            dict['x'] = x 
            dict['y'] = y 
            dict['sel'] = info['sel']
            dict['info'] = info
            dict['legend'] = legend
            self.emit(qt.PYSIGNAL("ScanWindowSignal"),(dict,))


    def __graphsignal(self,dict):
        if DEBUG:
            print "__graphsignal called dict = ",dict
        if dict['event'] == 'markerSelected':
            pass
        elif dict['event'] == 'markerMoved':
            pass
        elif dict['event'] == 'MouseAt':            
            self.xpos.setText('%.3f' % dict['x'])
            self.ypos.setText('%.2f' % dict['y'])
        elif dict['event'] == "SetActiveCurveEvent":
            legend,x,y = self.graph.getactivecurve()
            if legend is not None:
                counters = string.split(legend,":")
                xcnt     = counters[1]
                if xcnt == "-":
                    xcnt = "Index"
                ycnt     = counters[2]
                mcnt     = counters[3]
                if mcnt != "-":
                    ycnt = ycnt +"/"+mcnt
                self.graph.xlabel(xcnt)
                self.graph.ylabel(ycnt)
        elif dict['event'] == "RemoveCurveEvent":
            legend = dict['legend']
            curveinfo = self.graph.getcurveinfo(legend)
            if curveinfo.has_key('sel'):
                dict['sel'] = curveinfo['sel']
                self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(dict,))
            else:
                print "dict does not have sel ",dict
        else:
            if DEBUG:
                print "Unhandled event ",   dict['event']
            #follow the signal
            self.emit(qt.PYSIGNAL("ScanWindowSignal"),(dict,))


class McaAdvancedFitSelector(qt.QObject):
    def __init__(self,parent=None):
        """
        pseudo selector widget
        """
        qt.QObject.__init__(self,parent)
        self.dict = {}
        self.data = self
        self.sourcetype = "AdvancedFit"
        self.sourcename = None
        self.dict['SourceType'] = "AdvancedFit"
        self.selection = {}

    def SetSource(self,sourcename=None):
        if sourcename is None:return
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}
        self.sourcename=sourcename  
    
    def setData(self,sourcename=None,info=None,data=None,key=None):
        if sourcename is None:
            print 0
            return
        else: self.sourcename = sourcename
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}        
        if key is None:key=sourcename
        self.dict[sourcename][key]   = {'info':{},
                                       'data':Numeric.array([])}
        if info is not None:self.dict[sourcename][key]['info'] = info
        if data is not None:self.dict[sourcename][key]['data'] = data

    def LoadSource(self,key_list=None):
        if DEBUG:
            print "self.dict = ",self.dict
            print "LoadSource self.sourcename = ",self.sourcename,"keylist = ",key_list
        if key_list is not None:
            if type(key_list) is not type([]):
                key_list = [key_list]
        output = []
        for key in key_list:
            if self.sourcename is None:output.append([{},Numeric.array([])])
            else:
                if key in self.dict[self.sourcename].keys():
                    info = self.dict[self.sourcename][key]['info']
                    data = self.dict[self.sourcename][key]['data']
                    output.append([info,data])
                else:
                    output.append([{},Numeric.array([])])
        if len(output) == 1:
            return output[0]
        else:
            return output
    
    def removeSelection(self,selection):
        for sel in selection:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if self.selection.has_key(sourcename):
                if key in self.selection[sourcename]:
                    index = self.selection[sourcename].index(key)
                    del self.selection[sourcename][index]
                    if len(self.selection[sourcename]) == 0:
                        #nothing is selected so I can get rid of the fit data
                        if sourcename in self.dict.keys():
                            del self.dict[sourcename]
                        del self.selection[sourcename]
        self.emit(qt.PYSIGNAL("removeSelection"),(selection,))    
    
    def setSelected(self,sellist,reset=1):
        if DEBUG:
            print "selection before = ",self.selection
        if reset:
            self.selection={}
        for sel in sellist:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if not self.selection.has_key(sourcename):
                self.selection[sourcename]= []
            if key not in self.selection[sourcename]:
                self.selection[sourcename].append(key)
        if DEBUG:
            print "selection after = ",self.selection
        self.emit(qt.PYSIGNAL("addSelection"), (self.getSelection(),))
            
    def getSelection(self):
        selection=[]
        if self.selection is None: return selection
        for sourcekey in self.selection.keys():
            for key in self.selection[sourcekey]:
                sel={}
                sel['SourceName']   = sourcekey
                sel['SourceType']   = 'AdvancedFit'
                sel['Key']          = key
                selection.append(sel)
        return selection
    
    
    def isHidden(self):
        return 0
        
    def show(self):
        pass
        
    def raiseW(self):
        pass

class McaSimpleFitSelector(qt.QObject):
    def __init__(self,parent=None):
        """
        pseudo selector widget
        """
        qt.QObject.__init__(self,parent)
        self.dict = {}
        self.data = self
        self.sourcetype = "SimpleFit"
        self.sourcename = None
        self.dict['SourceType'] = "SimpleFit"
        self.selection = {}

    def SetSource(self,sourcename=None):
        if sourcename is None:return
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}
        self.sourcename=sourcename  
    
    def setData(self,sourcename=None,info=None,data=None,key=None):
        if sourcename is None:
            print 0
            return
        else: self.sourcename = sourcename
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}        
        if key is None:key=sourcename
        self.dict[sourcename][key]   = {'info':{},
                                       'data':Numeric.array([])}
        if info is not None:self.dict[sourcename][key]['info'] = info
        if data is not None:self.dict[sourcename][key]['data'] = data

    def LoadSource(self,key_list=None):
        if DEBUG:
            print "self.dict = ",self.dict
            print "LoadSource self.sourcename = ",self.sourcename,"keylist = ",key_list
        if key_list is not None:
            if type(key_list) is not type([]):
                key_list = [key_list]
        output = []
        for key in key_list:
            if self.sourcename is None:output.append([{},Numeric.array([])])
            else:
                if key in self.dict[self.sourcename].keys():
                    info = self.dict[self.sourcename][key]['info']
                    data = self.dict[self.sourcename][key]['data']
                    output.append([info,data])
                else:
                    output.append([{},Numeric.array([])])
        if len(output) == 1:
            return output[0]
        else:
            return output
    
    def removeSelection(self,selection):
        for sel in selection:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if self.selection.has_key(sourcename):
                if key in self.selection[sourcename]:
                    index = self.selection[sourcename].index(key)
                    del self.selection[sourcename][index]
                    if len(self.selection[sourcename]) == 0:
                        #nothing is selected so I can get rid of the fit data
                        if sourcename in self.dict.keys():
                            del self.dict[sourcename]
                        del self.selection[sourcename]
        self.emit(qt.PYSIGNAL("removeSelection"),(selection,))    
    
    def setSelected(self,sellist,reset=1):
        if DEBUG:
            print "selection before = ",self.selection
        if reset:
            self.selection={}
        for sel in sellist:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if not self.selection.has_key(sourcename):
                self.selection[sourcename]= []
            if key not in self.selection[sourcename]:
                self.selection[sourcename].append(key)
        if DEBUG:
            print "selection after = ",self.selection
        self.emit(qt.PYSIGNAL("addSelection"), (self.getSelection(),))
            
    def getSelection(self):
        selection=[]
        if self.selection is None: return selection
        for sourcekey in self.selection.keys():
            for key in self.selection[sourcekey]:
                sel={}
                sel['SourceName']   = sourcekey
                sel['SourceType']   = 'SimpleFit'
                sel['Key']          = key
                selection.append(sel)
        return selection
    
    
    def isHidden(self):
        return 0
        
    def show(self):
        pass
        
    def raiseW(self):
        pass
    

class ScanFitSelector(qt.QObject):
    def __init__(self,parent=None):
        """
        pseudo selector widget
        """
        qt.QObject.__init__(self,parent)
        self.dict = {}
        self.data = self
        self.sourcetype = "ScanFit"
        self.sourcename = None
        self.dict['SourceType'] = "ScanFit"
        self.selection = {}

    def SetSource(self,sourcename=None):
        if sourcename is None:return
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}
        self.sourcename=sourcename  
    
    def setData(self,sourcename=None,info=None,data=None,key=None):
        if sourcename is None:
            print 0
            return
        else: self.sourcename = sourcename
        if not self.dict.has_key(sourcename):
            self.dict[sourcename] = {}        
        if key is None:key=sourcename
        self.dict[sourcename][key]   = {'info':{},
                                       'data':Numeric.array([])}
        if info is not None:self.dict[sourcename][key]['info'] = info
        if data is not None:self.dict[sourcename][key]['data'] = data

    def LoadSource(self,key_list=None):
        if DEBUG:
            print "self.dict = ",self.dict
            print "LoadSource self.sourcename = ",self.sourcename,"keylist = ",key_list
        if key_list is not None:
            if type(key_list) is not type([]):
                key_list = [key_list]
        output = []
        for key in key_list:
            if self.sourcename is None:output.append([{},Numeric.array([])])
            else:
                if key in self.dict[self.sourcename].keys():
                    info = self.dict[self.sourcename][key]['info']
                    data = self.dict[self.sourcename][key]['data']
                    output.append([info,data])
                else:
                    output.append([{},Numeric.array([])])
        if len(output) == 1:
            return output[0]
        else:
            return output
    
    def removeSelection(self,selection):
        for sel in selection:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if self.selection.has_key(sourcename):
                if key in self.selection[sourcename]:
                    index = self.selection[sourcename].index(key)
                    del self.selection[sourcename][index]
                    if len(self.selection[sourcename]) == 0:
                        #nothing is selected so I can get rid of the fit data
                        if sourcename in self.dict.keys():
                            del self.dict[sourcename]
                        del self.selection[sourcename]
        self.emit(qt.PYSIGNAL("removeSelection"),(selection,))    
    
    def setSelected(self,sellist,reset=1):
        if DEBUG:
            print "selection before = ",self.selection
        if reset:
            self.selection={}
        for sel in sellist:
            sourcename = sel['SourceName']
            key        = sel['Key']
            if not self.selection.has_key(sourcename):
                self.selection[sourcename]= []
            if key not in self.selection[sourcename]:
                self.selection[sourcename].append(key)
        if DEBUG:
            print "selection after = ",self.selection
        nsel=self.getSelection()
        if len(nsel):
            self.emit(qt.PYSIGNAL("addSelection"), (nsel,))
            
    def getSelection(self):
        selection=[]
        if self.selection is None: return selection
        for sourcekey in self.selection.keys():
            for key in self.selection[sourcekey]:
                sel={}
                sel['SourceName']   = sourcekey
                sel['SourceType']   = 'ScanFit'
                sel['Key']          = key
                selection.append(sel)
        return selection
    
    
    def isHidden(self):
        return 0
        
    def show(self):
        pass
        
    def raiseW(self):
        pass


"""
def finish():
    print "finish"
    while qt.qApp.hasPendingEvents():
        qt.qApp.processEvents()
    qt.qApp.quit()    
"""    
def main(args):
    app = qt.QApplication(args)
    #demo = make()
    if sys.platform == 'win32':
        winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
        app.setPalette(winpalette)




    options     = ''
    longoptions = ['spec=','shm=']
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except getopt.error,msg:
        print msg
        sys.exit(1)
    kw={}
    for opt, arg in opts:
        if  opt in ('--spec'):
            kw['spec'] = arg
        elif opt in ('--shm'):
            kw['shm']  = arg
    
    #demo = McaWindow()
    demo = McaWidget(**kw)
    app.setMainWidget(demo)
    demo.show()
    #app.thread = SPSthread(demo)
    #app.thread.start()    
    #demo.container.graphwindow.setMinimumWidth(2* demo.tabset.width())
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                            app,qt.SLOT("quit()"))
    app.exec_loop()

if __name__ == '__main__':
    main(sys.argv)

