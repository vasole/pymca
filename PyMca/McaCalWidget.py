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
__revision__ = "$Revision: 1.13 $"
__author__="V.A. Sole - ESRF BLISS Group"

import sys
from QtBlissGraph import qt
import QtBlissGraph
import os
import Numeric
from LinearAlgebra import inverse
import Specfit
import SpecfitFuns
from Icons import IconDict
import PeakTableWidget
import copy
DEBUG = 0

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))
      
#class McaCalWidget(qt.QWidget):
class McaCalWidget(qt.QDialog):
    def __init__(self, parent=None, name="McaCalWidget", 
                x = None,y=None,current=None,peaks=None,caldict=None,
                specfit=None,legend="",modal=0,fl=0):
                #fl=qt.Qt.WDestructiveClose):
        self.name= name
        if qt.qVersion() < '4.0.0':
            qt.QDialog.__init__(self, parent, name, modal,fl)
            self.setCaption(self.name)
        else:
            qt.QDialog.__init__(self, parent)
            self.setModal(modal)            
            self.setWindowTitle(self.name)
        #qt.QMainWindow.__init__(self, parent, name,qt.Qt.WType_Popup)

        self.__manualsearch=0
        self.foundpeaks    =[]
        if caldict is None:
            caldict = {}
        self.dict = {}
        if x is None:
            if len(y):
                x = Numeric.arange(len(y)).astype(Numeric.Float)
        self.dict ['x']        = x
        self.dict ['y']        = y
        self.dict ['legend']   = legend
        self.current = legend
        self.caldict           = caldict
        if legend not in self.caldict.keys():
            self.caldict[legend] = {}
            self.caldict[legend]['order'] = 1  
            self.caldict[legend]['A'] = 0.0  
            self.caldict[legend]['B'] = 1.0  
            self.caldict[legend]['C'] = 0.0
        if not self.caldict[legend].has_key('order'):
                if abs(self.caldict[legend]['C']) > 0.0:
                    self.caldict[legend]['order'] = 2
                else:
                    self.caldict[legend]['order'] = 1   
        self.callist           = self.caldict.keys()
        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            self.specfit = specfit
        self.build()
        self.initIcons()
        self.initToolBar()
        self.connections()
        if self.dict ['y'] is not None:
            self.plot(x,y,legend)
        self.markermode = 0
        self.linewidgets=[]
        self.graph.ToggleLogY()
        self.graph.setCanvasBackground(qt.Qt.white)
        self.__peakmarkermode()
       
        
    def build(self):
        self.layout = qt.QVBoxLayout(self)
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.toolbar   = qt.QWidget(self)
        self.toolbar.layout = qt.QHBoxLayout(self.toolbar)
        self.toolbar.layout.setMargin(0)
        self.toolbar.layout.setSpacing(0)
        self.layout.addWidget(self.toolbar)
        self.container = qt.QWidget(self)
        self.container.layout = qt.QVBoxLayout(self.container)

        self.layout.addWidget(self.container)
        #The graph
        self.graph= QtBlissGraph.QtBlissGraph(self.container)
        self.graph.xlabel('Channel')
        self.graph.ylabel('Counts')
        self.graph.canvas().setMouseTracking(1)
        
        #self.setCentralWidget(self.container)
        #self.initIcons()
        #self.initToolBar()
        #The calibration Widget
        self.bottomPanel = qt.QWidget(self.container)
        self.bottomPanel.layout = qt.QHBoxLayout(self.bottomPanel)
        self.bottomPanel.layout.setSpacing(0)
        self.bottomPanel.layout.setMargin(10)
        self.peakpar        = PeakSearchParameters(self.bottomPanel)
        self.bottomPanel.layout.addWidget(self.peakpar)
        """
        self.calpar         = CalibrationParameters(self.bottomPanel)
        self.calpar. setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
        """
        if qt.qVersion() < '4.0.0':
            self.bottomPanel.layout.addWidget(HorizontalSpacer(self.bottomPanel))
        #self.cal.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.MinimumExpanding)
        self.peakpar.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                                                  qt.QSizePolicy.Fixed))       
        self.peaktable      = PeakTableWidget.PeakTableWidget(self.bottomPanel)
        self.bottomPanel.layout.addWidget(self.peaktable)
        self.peaktable.verticalHeader().hide()
        if qt.qVersion() < '4.0.0':
            self.peaktable.setLeftMargin(0)
        self.container.layout.addWidget(self.graph)
        self.container.layout.addWidget(self.bottomPanel)
            
        #self.peaktable.setRowReadOnly(0,1)


    def initIcons(self):
        if qt.qVersion() < '4.0.0': qt.QIcon = qt.QIconSet
        self.normalIcon	= qt.QIcon(qt.QPixmap(IconDict["normal"]))
        self.zoomIcon	= qt.QIcon(qt.QPixmap(IconDict["zoom"]))
        self.roiIcon	= qt.QIcon(qt.QPixmap(IconDict["roi"]))
        self.peakIcon	= qt.QIcon(qt.QPixmap(IconDict["peak"]))

        self.zoomResetIcon	= qt.QIcon(qt.QPixmap(IconDict["zoomreset"]))
        self.roiResetIcon	= qt.QIcon(qt.QPixmap(IconDict["roireset"]))
        self.peakResetIcon	= qt.QIcon(qt.QPixmap(IconDict["peakreset"]))
        self.refreshIcon	= qt.QIcon(qt.QPixmap(IconDict["reload"]))

        self.logxIcon	= qt.QIcon(qt.QPixmap(IconDict["logx"]))
        self.logyIcon	= qt.QIcon(qt.QPixmap(IconDict["logy"]))
        self.fitIcon	= qt.QIcon(qt.QPixmap(IconDict["fit"]))
        self.searchIcon	= qt.QIcon(qt.QPixmap(IconDict["peaksearch"]))

    def initToolBar(self):
        toolbar = self.toolbar
        #Zoom Reset
        self._addToolButton(self.zoomResetIcon,
                            self.graph.ResetZoom,
                            'Auto-Scale the Graph')
        # Logarithmic
        self._addToolButton(self.logyIcon,
                            self.graph.ToggleLogY,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle=True)
        # Search
        self._addToolButton(self.searchIcon,
                            self.peaksearch,
                            'Clear Peak Table and Search Peaks') 
        # Clear peaks
        self._addToolButton(self.peakResetIcon,
                            self.clearpeaks,
                            'Clear Peak Table') 
        # Manual Search
        self.__msb = self._addToolButton(self.peakIcon,
                            self.manualsearch,
                            'Add a peak to the graph',
                            toggle=True)
        self.toolbar.layout.addWidget(HorizontalSpacer(toolbar))
        label=qt.QLabel(toolbar)
        label.setText('<b>Channel:</b>')
        self.toolbar.layout.addWidget(label)
        self.xpos = qt.QLineEdit(toolbar)
        self.xpos.setText('------')
        self.xpos.setReadOnly(1)
        self.xpos.setFixedWidth(self.xpos.fontMetrics().width('########'))
        self.toolbar.layout.addWidget(self.xpos)
        label=qt.QLabel(toolbar)
        label.setText('<b>Counts:</b>')
        self.toolbar.layout.addWidget(label)
        self.ypos = qt.QLineEdit(toolbar)
        self.ypos.setText('------')
        self.ypos.setReadOnly(1)
        self.ypos.setFixedWidth(self.ypos.fontMetrics().width('#########'))
        self.toolbar.layout.addWidget(self.ypos)
        label=qt.QLabel(toolbar)
        label.setText('<b>Energy:</b>')
        self.toolbar.layout.addWidget(label)
        self.epos = qt.QLineEdit(toolbar)
        self.epos.setText('------')
        self.epos.setReadOnly(1)
        self.epos.setFixedWidth(self.epos.fontMetrics().width('#########'))
        self.toolbar.layout.addWidget(self.epos)


        #rest
        toolbar2 = qt.QWidget(self)
        self.layout.addWidget(toolbar2)
        toolbar2.layout = qt.QHBoxLayout(toolbar2)
        toolbar2.layout.setMargin(0)
        toolbar2.layout.setSpacing(0)
        self.calpar         = CalibrationParameters(toolbar2,
                                calname=self.current,caldict=self.caldict)
        self.calpar. setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.okButton       = qt.QPushButton(toolbar2)
        self.okButton.setText('OK')
        self.cancelButton       = qt.QPushButton(toolbar2)
        self.cancelButton.setText('Cancel')
        self.okButton. setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.cancelButton. setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        toolbar2.layout.addWidget(self.calpar)
        toolbar2.layout.addWidget(self.okButton)
        toolbar2.layout.addWidget(self.cancelButton)


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
        
    def connections(self):
        self.connect(self.peakpar.searchbut,qt.SIGNAL('clicked()')  ,self.peaksearch)
        if qt.qVersion() < '4.0.0':
            self.connect(self.graph, qt.PYSIGNAL('QtBlissGraphSignal')  ,
                         self.__graphsignal) 
            self.connect(self.peaktable, qt.PYSIGNAL('PeakTableWidgetSignal') , 
                         self.__peaktablesignal)
            self.connect(self.calpar, qt.PYSIGNAL('CalibrationParametersSignal'),
                         self.__calparsignal)
            self.connect(self.okButton,qt.SIGNAL('clicked()'),self.accept)
            self.connect(self.cancelButton,qt.SIGNAL('clicked()'),self.reject)
        else:
            self.connect(self.graph, qt.SIGNAL('QtBlissGraphSignal')  ,
                         self.__graphsignal) 
            self.connect(self.peaktable, qt.SIGNAL('PeakTableWidgetSignal') , 
                         self.__peaktablesignal)
            self.connect(self.calpar, qt.SIGNAL('CalibrationParametersSignal'),
                         self.__calparsignal)
            self.connect(self.okButton,qt.SIGNAL('clicked()'),self.accept)
            self.connect(self.cancelButton,qt.SIGNAL('clicked()'),self.reject)
    
    def plot(self,x,y,legend):
        #clear graph
        self.graph.clearcurves()
        self.graph.newcurve(legend,x=x,y=y,logfilter=1)
        self.dict['x']      = x
        self.dict['y']      = y
        self.dict['legend'] = legend
        #reset the zoom
        self.graph.ResetZoom()
        
    def peaksearch(self):
        if DEBUG:
            print "Peak search called"
        if self.__manualsearch:
            self.__manualsearch = 0
            self.__msb.setState(qt.QButton.Off)
        #get current plot limits
        xmin,xmax=self.graph.getx1axislimits()
        #set the data into specfit
        self.specfit.setdata(x=self.dict['x'],y=self.dict['y'],xmin=xmin,xmax=xmax)
        #get the search parameters from the interface
        pars = self.peakpar.getParameters()
        if pars["AutoFwhm"]:
            fwhm = self.specfit.guess_fwhm()
        else:
            fwhm = pars["FwhmPoints"]
        if pars["AutoYscaling"]:
            yscaling = self.specfit.guess_yscaling()
        else:
            yscaling = pars["Yscaling"]
        sensitivity  = pars["Sensitivity"]
        self.peakpar.FwhmText.setText("%d" % fwhm)
        self.peakpar.YscalingText.setText("%f" % yscaling)
        ysearch = self.specfit.ydata*yscaling
        peaksidx=SpecfitFuns.seek(ysearch,1,len(ysearch),
                                    fwhm,
                                    sensitivity)
        self.foundpeaks = []
        self.graph.clearmarkers()
        self.__destroylinewidgets()
        self.peaktable.setRowCount(0)
        i = 0
        for idx in peaksidx:
            self.foundpeaks.append(self.specfit.xdata[int(idx)])            
            #self.graph.insertx1marker(self.specfit.xdata[int(idx)],self.specfit.ydata[int(idx)])
            self.graph.insertx1marker(self.specfit.xdata[int(idx)],1.1)
            i += 1
        self.graph.replot()
        #make sure marker mode is on
        self.markermode = 0
        self.__peakmarkermode()


    def clearpeaks(self):
        self.foundpeaks = []
        self.graph.clearmarkers()
        self.__destroylinewidgets()
        self.peaktable.setRowCount(0)
        self.graph.replot()

    def manualsearch(self):
        #disable peak selection
        self.markermode     = 1
        self.__peakmarkermode()
        self.__manualsearch = 1
        #self.__msb.setDown(1)

    def __destroylinewidgets(self):
        for widget in self.linewidgets:
            widget.close(1)
        self.linewidgets=[]
        
    def __peakmarkermode(self):
        self.__manualsearch = 0
        if self.markermode:
            #enable zoom back
            #self.graph.enablezoomback()
            #disable marking
            """
            qt.QToolTip.add(self.markerButton,'Allow Peak Selection from Graph') 
            """
            self.graph.disablemarkermode()
            if qt.qVersion() < '4.0.0':
                self.graph.canvas().setCursor(qt.QCursor(qt.QCursor.CrossCursor))
            else:
                self.graph.canvas().setCursor(qt.QCursor(qt.Qt.CrossCursor))                
            #save the cursor
            self.markermode = 0
        else:
            #disable zoomback
            #self.graph.disablezoomback()
            #enable marking
            self.graph.enablemarkermode()
            """
            qt.QToolTip.add(self.markerButton,'Disable Peak Selection from Graph') 
            """
            self.markermode = 1
            self.nomarkercursor = self.graph.canvas().cursor().shape()
            if qt.qVersion() < '4.0.0':
                self.graph.canvas().setCursor(qt.QCursor(qt.QCursor.PointingHandCursor))
            else:
                self.graph.canvas().setCursor(qt.QCursor(qt.Qt.PointingHandCursor))
        #self.markerButton.setOn(self.markermode == 1)

    def __calparsignal(self,dict):
        if DEBUG:
            print "__calparsignal called dict = ",dict
        if dict['event'] == 'coeff':
            current = dict['calname' ]
            self.current  = current
            self.caldict[current]['order'] =dict['caldict'][dict['calname']]['order']
            self.caldict[current]['A'] =dict['caldict'][dict['calname']]['A']
            self.caldict[current]['B'] =dict['caldict'][dict['calname']]['B']
            self.caldict[current]['C'] =dict['caldict'][dict['calname']]['C']
            peakdict = self.peaktable.getdict()
            for peak in peakdict.keys():
                channel = peakdict[peak]['channel']
                calenergy  = self.caldict[current]['A'] + \
                                 self.caldict[current]['B'] * channel +\
                                 self.caldict[current]['C'] * channel * channel  
                self.peaktable.configure(name=peak,use=0,
                                         calenergy=calenergy)
        elif dict['event'] == 'order':
            current = dict['calname' ]
            self.current  = current
            self.caldict[current]['order'] = dict['caldict'][current]['order']
            self.caldict[current]['A']     = dict['caldict'][current]['A']
            self.caldict[current]['B']     = dict['caldict'][current]['B']
            self.caldict[current]['C']     = dict['caldict'][current]['C']
            """
            peakdict = self.peaktable.getdict()
            usedpeaks = []
            for peak in peakdict.keys():
                if peakdict[peak]['use']:
                    channel     = peakdict[peak]['channel']
                    setenergy   = peakdict[peak]['setenergy']
                    usedpeaks.append([channel,setenergy])
            if len(usedpeaks):
                newcal = self.calculate(usedpeaks,order=self.dict['current']['order'])
                self.caldict[current]['A'] = newcal[0]
                self.caldict[current]['B'] = newcal[1]
                self.caldict[current]['C'] = newcal[2]
                self.calcalpar.setPars(self.caldict[current])
                for peak in peakdict.keys():
                    channel = peakdict[peak]['channel']
                    calenergy  = self.caldict[current]['A'] + \
                                 self.caldict[current]['B'] * channel +\
                                 self.caldict[current]['C'] * channel * channel  
                    self.peaktable.configure(name=peak,
                                            calenergy=calenergy)
            """
            self.__peaktablesignal({'event':'use'})
        elif dict['event'] == 'savebox':
            current = dict['calname' ]
            if current not in self.caldict.keys():
                self.caldict[current] = {}    
            self.current  = current
            self.caldict[current]['order'] = dict['caldict'][current]['order']
            self.caldict[current]['A']     = dict['caldict'][current]['A']
            self.caldict[current]['B']     = dict['caldict'][current]['B']
            self.caldict[current]['C']     = dict['caldict'][current]['C']

        elif dict['event'] == 'activated':
            # A comboBox has been selected
            if   dict['boxname'] == 'Source':
                pass
            elif dict['boxname'] == 'Calibration':
                pass
            else:
                if DEBUG:
                    print "Unknown combobox",dict['boxname']
        else:
            print "Unknown signal ",dict

    def __graphsignal(self, ddict):
        if DEBUG:
            print "__graphsignal called with dict = ", ddict
        if ddict['event'] == 'markerSelected':
            if DEBUG:
                print "Setting marker color"
            marker = int(ddict['marker'])
            name   = "Peak %d" % marker
            number = marker
            #channel= dict['x']
            if qt.qVersion() < '4.0.0':
                channel=self.foundpeaks[marker-1]
            else:
                channel=ddict['x']                
            self.graph.setmarkercolor(marker,'red')
            self.graph.replot()
            current = self.current
            calenergy = self.caldict[current]['A']+ \
                        self.caldict[current]['B'] * channel+ \
                        self.caldict[current]['C'] * channel * channel
            linewidget = InputLine(self,name="Enter Selected Peak Parameters",
                                    peakpars={'name':name,
                                    'number':number,
                                    'channel':channel,
                                    'use':1,
                                    'calenergy':calenergy})
            if qt.qVersion() < '4.0.0':
                ret = linewidget.exec_loop()
            else:
                ret = linewidget.exec_()
            if ret == qt.QDialog.Accepted:
                ddict=linewidget.getdict()
                if DEBUG:
                    print "dict got from dialog = ",ddict
                if ddict != {}:
                    if name in self.peaktable.peaks.keys():
                        self.peaktable.configure(*ddict)
                    else:
                        nlines=self.peaktable.rowCount()
                        ddict['name'] = name
                        self.peaktable.newpeakline(name,nlines+1)
                        self.peaktable.configure(**ddict)
                    peakdict = self.peaktable.getdict()
                    usedpeaks = []
                    for peak in peakdict.keys():
                        if peakdict[peak]['use'] == 1:
                            usedpeaks.append([peakdict[peak]['channel'],
                                              peakdict[peak]['setenergy']])
                    if len(usedpeaks):
                        newcal = self.calculate(usedpeaks,order=self.caldict[current]['order'])
                        self.caldict[current]['A'] = newcal[0]
                        self.caldict[current]['B'] = newcal[1]
                        self.caldict[current]['C'] = newcal[2]
                    self.__peaktablesignal({'event':'use'})
            else:
                if DEBUG:
                    print "Dialog cancelled or closed "
                self.graph.setmarkercolor(marker,'black')
                self.graph.replot()
            del linewidget
        elif ddict['event'] == 'MouseAt':            
            self.xpos.setText('%.1f' % ddict['x'])
            self.ypos.setText('%.1f' % ddict['y'])
            current = self.current
            calenergy = self.caldict[current]['A']+ \
                        self.caldict[current]['B'] * ddict['x']+ \
                        self.caldict[current]['C'] * ddict['x'] * ddict['x']
            self.epos.setText('%.3f' % calenergy)
        elif ddict['event'] == 'MouseClick':
            if self.__manualsearch:
                x = ddict['x']
                y = ddict['y']
                if (y <= 1.0): y=1.1
                # insert the marker
                self.foundpeaks.append(x)            
                #self.graph.insertx1marker(self.specfit.xdata[int(idx)],self.specfit.ydata[int(idx)])
                self.graph.insertx1marker(x,y)
                self.graph.replot()
                self.markermode = 0
                self.__peakmarkermode()
            if qt.qVersion() < '4.0.0':
                self.__msb.setState(qt.QButton.Off)
            else:
                self.__msb.setChecked(0)
        else:
            if DEBUG:
                print "Unhandled event ",   ddict['event']

    def __peaktablesignal(self, ddict):
        if DEBUG:
            print "__peaktablesignal called dict = ",ddict
        if (ddict['event'] == 'use') or (ddict['event'] == 'setenergy'):
            #get table dictionary
            peakdict = self.peaktable.getdict()
            usedpeaks = []
            for peak in peakdict.keys():
                if peakdict[peak]['use'] == 1:
                    usedpeaks.append([peakdict[peak]['channel'],
                                      peakdict[peak]['setenergy']])
            if len(usedpeaks):
              if usedpeaks != [[0.0,0.0]]:
                current = self.current
                newcal = self.calculate(usedpeaks,order=self.caldict[current]['order'])                
                self.caldict[current]['A'] = newcal[0]
                self.caldict[current]['B'] = newcal[1]
                self.caldict[current]['C'] = newcal[2]
                self.calpar.setParameters(self.caldict[current])
                for peak in peakdict.keys():
                    channel = peakdict[peak]['channel']
                    calenergy  = self.caldict[current]['A'] + \
                                 self.caldict[current]['B'] * channel +\
                                 self.caldict[current]['C'] * channel * channel  
                    self.peaktable.configure(name=peak,
                                            calenergy=calenergy)

    def calculate(self,usedpeaks,order=1):
        """
        used peaks has the form [[x0,e0],[x1,e1],...]
        """
        if len(usedpeaks) == 1:
            if (usedpeaks[0][0] - 0.0) > 1.0E-20:
                return [0.0,usedpeaks[0][1]/usedpeaks[0][0],0.0]
            else:
                if DEBUG:
                    print "Division by zero"
                current = self.current
                self.caldict[current]['A'] = newcal[0]
                self.caldict[current]['B'] = newcal[1]
                self.caldict[current]['C'] = newcal[2]
                return [self.caldict[current]['A'],
                        self.caldict[current]['B'],
                        self.caldict[current]['C']]
        if (order > 1) and (len(usedpeaks) == 2):
            usedpeaks.append([0.0,0.0])            
        usedarray = Numeric.array(usedpeaks).astype(Numeric.Float)
        energy = usedarray[:,1]
        channel= usedarray[:,0]
        
        if order < 2:
            X = Numeric.array([Numeric.ones(len(channel)), channel])        
        else:
            X= Numeric.array([Numeric.ones(len(channel)), channel, channel*channel])
        TX = Numeric.transpose(X)
        XTX= Numeric.matrixmultiply(X, TX)
        INV= inverse(XTX)
        PC = Numeric.matrixmultiply(energy, TX)
        C  = Numeric.matrixmultiply(PC, INV)

        if order==1:
                result= tuple(C.tolist())+(0.,)
        else:   result= tuple(C.tolist())
        return result


    def getdict(self):
        dict = {}
        dict.update(self.caldict)
        return dict
            
class PeakSearchParameters(qt.QWidget):
    def __init__(self, parent=None, name="", specfit=None, config=None,
                searchbutton=1,fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name, fl)    
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)    
            self.setWindowTitle(name)
            
        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            specfit      = specfit
        if config is None: 
            config=self.specfit.fitconfig
        if config.has_key("AutoYscaling"):
            autoscaling = config["AutoYscaling"]
        else:
            autoscaling = 0
        self.searchbutton = searchbutton
        parameters= { "FwhmPoints":  config["FwhmPoints"],
                      "Sensitivity": config["Sensitivity"],
                      "Yscaling":    config["Yscaling"],
                      "AutoYscaling": autoscaling,
                      "AutoFwhm": 0
                     }
        self.build()
        self.setParameters(parameters)
        
    def build(self):
        if 1:
            if qt.qVersion() < '4.0.0':
                layout= qt.QVBoxLayout(self)

            # --- parameters
                parf= qt.QHGroupBox(self)
                parf.setTitle('Search Parameters')
                parf.setAlignment(qt.Qt.AlignHCenter)
                parw= qt.QWidget(parf)
            else:
                layout= qt.QVBoxLayout(self)

            # --- parameters
                parf= qt.QGroupBox(self)
                parf.layout = qt.QVBoxLayout(parf)                
                parf.setTitle('Search Parameters')
                parf.setAlignment(qt.Qt.AlignHCenter)
                parw= qt.QWidget(parf)
                parf.layout.addWidget(parw)
        else:
            parw = self
        if qt.qVersion() < '4.0.0':
            if self.searchbutton:
                grid= qt.QGridLayout(parw, 4, 3)
            else:
                grid= qt.QGridLayout(parw, 3, 3)
        else:
            grid= qt.QGridLayout(parw)
        lab= qt.QLabel("Sensitivity", parw)
        grid.addWidget(lab, 0, 0, qt.Qt.AlignLeft)
        lab= qt.QLabel("Fwhm (pts)", parw)
        grid.addWidget(lab, 1, 0, qt.Qt.AlignLeft)
        lab= qt.QLabel("Yscaling", parw)
        grid.addWidget(lab, 2, 0, qt.Qt.AlignLeft)
        self.SensitivityText= qt.QLineEdit(parw)
        grid.addWidget(self.SensitivityText, 0, 1)
        self.FwhmText= qt.QLineEdit(parw)
        grid.addWidget(self.FwhmText, 1, 1)
        self.YscalingText= qt.QLineEdit(parw)
        grid.addWidget(self.YscalingText, 2, 1)
        self.FwhmAuto= qt.QCheckBox("Auto", parw)
        self.connect(self.FwhmAuto, qt.SIGNAL("toggled(bool)"), self.__fwhmToggled)
        grid.addWidget(self.FwhmAuto, 1, 2, qt.Qt.AlignLeft)
        self.YscalingAuto= qt.QCheckBox("Auto", parw)
        self.connect(self.YscalingAuto, qt.SIGNAL("toggled(bool)"), self.__yscalingToggled)
        grid.addWidget(self.YscalingAuto, 2, 2, qt.Qt.AlignLeft)
        if self.searchbutton:
            self.searchbut = qt.QPushButton(parw)   
            self.searchbut.setText('Search')
            grid.addWidget(self.searchbut,3,1)
        layout.addWidget(parf)

    def setParameters(self, pars):
        self.SensitivityText.setText(str(pars["Sensitivity"]))
        self.FwhmText.setText(str(pars["FwhmPoints"]))
        self.YscalingText.setText(str(pars["Yscaling"]))
        self.FwhmAuto.setChecked(pars["AutoFwhm"])
        self.YscalingAuto.setChecked(pars["AutoYscaling"])
        #self.specfit.configure(pars)

    def getParameters(self):
        pars= {}
        pars["Sensitivity"]= float(str(self.SensitivityText.text()))
        pars["FwhmPoints"]= float(str(self.FwhmText.text()))
        pars["Yscaling"]= float(str(self.YscalingText.text()))
        pars["AutoFwhm"]= self.FwhmAuto.isChecked()
        pars["AutoYscaling"]= self.YscalingAuto.isChecked()
        self.specfit.configure(**pars)
        return pars

    def __fwhmToggled(self, on):
        if on: self.FwhmText.setReadOnly(1)
        else: self.FwhmText.setReadOnly(0)

    def __yscalingToggled(self, on):
        if on: self.YscalingText.setReadOnly(1)
        else: self.YscalingText.setReadOnly(0)


class CalibrationParameters(qt.QWidget):
    def __init__(self, parent=None, name="", calname="", 
                 caldict = {},fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name, fl)    
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
        self.caldict=caldict
        if calname not in self.caldict.keys():
            self.caldict[calname] = {}
            self.caldict[calname]['order'] = 1
            self.caldict[calname]['A'] = 0.0   
            self.caldict[calname]['B'] = 1.0   
            self.caldict[calname]['C'] = 0.0
        self.currentcal = calname
        self.build()
        self.setParameters(self.caldict[calname])
        self.connections()
        
    def build(self):
        layout= qt.QHBoxLayout(self)
        parw = self
        
        lab= qt.QLabel("Order:", parw)

        self.orderbox = SimpleComboBox(parw,
                                       options=['1st','2nd'])
        layout.addWidget(lab)
        layout.addWidget(self.orderbox)
        lab= qt.QLabel("A:", parw)
        #self.AText= qt.QLineEdit(parw)
        self.AText= MyQLineEdit(parw)
        layout.addWidget(lab)
        layout.addWidget(self.AText)
        lab= qt.QLabel("B:", parw)
        self.BText= MyQLineEdit(parw)
        layout.addWidget(lab)
        layout.addWidget(self.BText)
        lab= qt.QLabel("C:", parw)
        layout.addWidget(lab)
        self.CText= MyQLineEdit(parw)
        layout.addWidget(self.CText)

        if 0:
            self.savebut= qt.QPushButton(parw)
            self.savebut.setText("Add as")
        else:
            lab = qt.QLabel("Add as", parw)
            layout.addWidget(lab)
        self.savebox = SimpleComboBox(parw,
                                       options=self.caldict.keys())
        layout.addWidget(self.savebox)

        self.savebox.setEditable(1)
        self.savebox.setDuplicatesEnabled(0)
    
    def connections(self):
        self.connect(self.AText,qt.SIGNAL('returnPressed()'),self._Aslot)
        self.connect(self.BText,qt.SIGNAL('returnPressed()'),self._Bslot)
        self.connect(self.CText,qt.SIGNAL('returnPressed()'),self._Cslot)
        self.connect(self.orderbox,qt.SIGNAL('activated(const QString &)'),self.__orderbox)
        #self.connect(self.savebut,qt.SIGNAL('clicked()')    ,self.myslot)
        self.connect(self.savebox,qt.SIGNAL('activated(const QString &)'),self.__savebox)
        
    def setParameters(self, pars):
        self.AText.setText("%.4g" % pars["A"])
        self.BText.setText("%.4g" % pars["B"])
        self.CText.setText("%.4g" % pars["C"])
        if pars['order'] > 1:
            if qt.qVersion() < '4.0.0':
                self.orderbox.setCurrentItem(1)
            else:
                self.orderbox.setCurrentIndex(1)
            self.CText.setReadOnly(0)
        else:
            if qt.qVersion() < '4.0.0':
                self.orderbox.setCurrentItem(0)
            else:
                self.orderbox.setCurrentIndex(0)
            self.CText.setReadOnly(1)
        self.caldict[self.currentcal]["A"] = pars["A"]
        self.caldict[self.currentcal]["B"] = pars["B"]
        self.caldict[self.currentcal]["C"] = pars["C"]
        self.caldict[self.currentcal]["order"] = pars["order"]

    
    def getcurrentdict(self):
        return self.caldict[self.currentcal]
    
    def getcurrentcal(self):
        return self.current
    
      
    def getdict(self):
        return self.caldict
    
    def __orderbox(self,qstring):
        qstring = str(qstring)
        if qstring == "1st":
            self.caldict[self.currentcal]['order'] = 1
            self.CText.setText("0.0")
            self.CText.setReadOnly(1)
            self.caldict[self.currentcal]['C'] = 0.0        
        else:
            self.caldict[self.currentcal]['order'] = 2
            self.CText.setReadOnly(0)
        self.myslot(event='order')

    def __savebox(self,qstring):
        key = str(qstring)
        if key not in self.caldict.keys():
            self.caldict[key] = {}
        self.caldict[key]['order'] = self.orderbox.currentItem()+1
        self.caldict[key]['A']     = float(str(self.AText.text()))
        self.caldict[key]['B']     = float(str(self.BText.text()))
        self.caldict[key]['C']     = float(str(self.CText.text()))
        self.currentcal = key
        self.myslot(event='savebox')

    def _Aslot(self):
        qstring = self.AText.text()
        try:
            value = float(str(qstring))
            self.caldict[self.currentcal]['A'] = value
            self.myslot(event='coeff')
        except:
            msg=qt.QMessageBox(self.AText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.AText.setFocus()
        
    def _Bslot(self):
        qstring = self.BText.text()
        try:
            value = float(str(qstring))
            self.caldict[self.currentcal]['B'] = value
            self.myslot(event='coeff')
        except:
            msg=qt.QMessageBox(self.BText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.BText.setFocus()

    def _Cslot(self):
        qstring = self.CText.text()
        try:
            value = float(str(qstring))
            self.caldict[self.currentcal]['C'] = value
            self.myslot(event='coeff')
        except:
            msg=qt.QMessageBox(self.CText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.CText.setFocus()

    def myslot(self,*var,**kw):
        if DEBUG:
            print "Cal Parameters Slot ",var,kw
            print self.caldict[self.currentcal]
        if kw.has_key('event'):
            dict={}
            if (kw['event'] == 'order'):
                dict={}
                dict['event']         = "order"
                dict['calname' ]      = self.currentcal
                dict['caldict']       = self.caldict          
            if (kw['event'] == 'coeff'):
                dict={}
                dict['event']         = "coeff"
                dict['calname' ]      = self.currentcal
                dict['caldict']       = self.caldict          
            if (kw['event'] == 'savebox'):
                dict={}
                dict['event']         = "savebox"
                dict['calname' ]      = self.currentcal
                dict['caldict']       = self.caldict
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('CalibrationParametersSignal'),(dict,))
            else:
                self.emit(qt.PYSIGNAL('CalibrationParametersSignal'), dict)

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=None):
        qt.QLineEdit.__init__(self,parent)
        
    def focusInEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass        
        else:
            if qt.qVersion() < '4.0.0':
                self.backgroundcolor = self.paletteBackgroundColor()
                self.setPaletteBackgroundColor(qt.QColor('yellow'))
    
    def focusOutEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass        
        else:
            if qt.qVersion() < '4.0.0':
                self.setPaletteBackgroundColor(qt.QColor('white'))
        self.emit(qt.SIGNAL("returnPressed()"),())


#class Popup(qt.QDialog):
class Popup(qt.QWidget):
    def __init__(self, parent=None, name=None, specfit=None, config=None,fl=0):
        qt.QWidget.__init__(self, parent, name,qt.Qt.WType_Popup)  
        #qt.QDialog.__init__(self, parent, name,qt.Qt.WType_Popup)  
        label = qt.QLabel(self.parent)
        label.setText("Hello")

class InputLine(qt.QDialog):
    def __init__(self,parent ,name = "Peak Parameters",modal=1,peakpars={},fl=0):
        #fl=qt.Qt.WDestructiveClose):
        if qt.qVersion() < '4.0.0':
            qt.QDialog.__init__(self, parent, name, modal, fl)
            self.setCaption(name)
        else:
            qt.QDialog.__init__(self, parent)
            self.setModal(modal)
            self.setWindowTitle(name)
            self.resize(600,200)
        layout = qt.QVBoxLayout(self)
        self.table = PeakTableWidget.PeakTableWidget(self)
        layout.addWidget(self.table)
        self.bottom = qt.QWidget(self)
        self.bottom.layout = qt.QHBoxLayout(self.bottom)
        layout.addWidget(self.bottom)
        self.bottom.layout.addWidget(HorizontalSpacer(self.bottom))
        okbutton       = qt.QPushButton(self.bottom)
        self.bottom.layout.addWidget(okbutton)
        okbutton.setText('OK')
        cancelbutton   = qt.QPushButton(self.bottom)
        cancelbutton.setText('Cancel')
        self.bottom.layout.addWidget(cancelbutton)
        okbutton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        cancelbutton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.bottom.layout.addWidget(HorizontalSpacer(self.bottom))
        self.connect(cancelbutton, qt.SIGNAL("clicked()"), self.reject)
        self.connect(okbutton, qt.SIGNAL("clicked()"), self.accept)
        if peakpars.has_key('name'):
            peakname = peakpars['name']
        else:
            peakname = 'PEAK 1'
        if peakpars.has_key('number'):
            number = peakpars['number']
        else:
            number = 1
        if peakpars.has_key('channel'):
            channel = peakpars['channel']
        else:
            channel = 0
        if peakpars.has_key('element'):
            element = peakpars['element']
        else:
            element = '-'
        if peakpars.has_key('elementline'):
            elementline = peakpars['elementline']
        else:
            elementline = '-'
        if elementline == '-':
            if peakpars.has_key('setenergy'):
                setenergy = peakpars['setenergy']
            else:
                setenergy = '0.0'
        if peakpars.has_key('use'):
            use = peakpars['use']
        else:
            use = 1
        if peakpars.has_key('calenergy'):
            calenergy = peakpars['calenergy']
        else:
            calenergy = ""
        self.table.newpeakline(peakname,1)
        self.peakname = peakname 
        self.table.configure(name   =peakname,
                             number =number,
                             channel=channel,
                             element=element,
                             elementline=elementline,
                             setenergy=setenergy,
                             use=use,
                             calenergy=calenergy)

    def getdict(self):
        dict=self.table.getdict(self.peakname)
        return dict

class McaCalCopy(qt.QDialog):
    def __init__(self,parent=None ,name = None,modal=1,fl=0,
                        legend=None,sourcecal=None,currentcal=None,caldict=None):
        #fl=qt.Qt.WDestructiveClose):
        if legend is None:
            legend= 'Active Curve'
        name = "Enter Calibration for %s" % legend
        qt.QDialog.__init__(self, parent, name, modal, fl)
        layout0 = qt.QVBoxLayout(self)
        layout0.setAutoAdd(1)
        
        currentcal = legend
        if sourcecal is None:
            sourcecal  = [0.0,1.0,0.0]
        if caldict is None:
            caldict    = {}
        self.caldict    = caldict
        self.currentcal = currentcal
        if currentcal in caldict.keys():
            currentval = [caldict[currentcal]['A'],
                                  caldict[currentcal]['B'],
                                  caldict[currentcal]['C']] 
        else:
            currentval = [0.0,1.0,0.0]

        # --- source ---
        sgroup = qt.QHGroupBox(self)
        sgroup.setTitle('Calibration from Source (Read Only)')
        sgroup.setAlignment(qt.Qt.AlignHCenter)
        w      = qt.QWidget(sgroup)
        wlayout= qt.QVBoxLayout(w)
        wlayout.setAutoAdd(1)
        
        """
        l           = qt.QHBox(w)
        HorizontalSpacer(l)
        sourcelabel = qt.QLabel(l)
        HorizontalSpacer(l)
        f = sourcelabel.font()
        f.setBold(1)
        sourcelabel.setText('Calibration from Source')
        """
        
        lines  = qt.QHBox(w)
        asl=qt.QLabel(lines)
        asl.setText('A:')
        asl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        as=qt.QLineEdit(lines)
        as.setReadOnly(1)
        as.setText("%.4g" % sourcecal[0])

        bsl=qt.QLabel(lines)
        bsl.setText('B:')
        bsl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        bs=qt.QLineEdit(lines)
        bs.setReadOnly(1)
        bs.setText("%.4g" % sourcecal[1])

        csl=qt.QLabel(lines)
        csl.setText('C:')
        csl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        cs=qt.QLineEdit(lines)
        cs.setReadOnly(1)
        cs.setText("%.4g" % sourcecal[2])
        
        # --- PyMca/Current ---
        cgroup = qt.QHGroupBox(self)
        fontc = cgroup.font()
        fontc.setBold(1)
        cgroup.setFont(fontc)
        cgroup.setTitle('Enter New Calibration (PyMca)')
        cgroup.setAlignment(qt.Qt.AlignHCenter)
        wc     = qt.QVBox(cgroup)
        wc.layout().setSpacing(3)
        linec  = qt.QHBox(wc)
        acl=qt.QLabel(linec)
        #acl.setFont(font)
        acl.setText('A:')
        acl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.AText=MyQLineEdit(linec)
        self.AText.setReadOnly(0)
        self.AText.setText("%.4g" % currentval[0])

        bcl=qt.QLabel(linec)
        #bcl.setFont(font)
        bcl.setText('B:')
        bcl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.BText=MyQLineEdit(linec)
        self.BText.setReadOnly(0)
        self.BText.setText("%.4g" % currentval[1])


        ccl=qt.QLabel(linec)
        #ccl.setFont(font)
        ccl.setText('C:')
        ccl.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        self.CText=MyQLineEdit(linec)
        self.CText.setReadOnly(0)
        self.CText.setText("%.4g" % currentval[2])

        self.connect(self.AText,qt.SIGNAL('returnPressed()'),self._Aslot)
        self.connect(self.BText,qt.SIGNAL('returnPressed()'),self._Bslot)
        self.connect(self.CText,qt.SIGNAL('returnPressed()'),self._Cslot)
        
        # --- available for copy ---
        if len(caldict.keys()):
            wid = qt.QWidget(wc)
            wfont = wid.font()
            wfont.setBold(0)
            wid.setFont(wfont)
            layout2=qt.QHBoxLayout(wid)
            layout2.setAutoAdd(1)
            layout2.setSpacing(3)
            
            copybut = qt.QPushButton(wid)
            copybut.setText('Copy From')
            copybut.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,qt.QSizePolicy.Fixed))
            self.connect(copybut,qt.SIGNAL("clicked()"),self.__copybuttonclicked)
            
            self.combo = SimpleComboBox(wid,options=caldict.keys())
            self.combo.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,qt.QSizePolicy.Fixed))
            

        # --- dialog buttons ---
        bottom = qt.QHBox(self)
        self.setCaption(name)
        HorizontalSpacer(bottom)
        okbutton       = qt.QPushButton(bottom)
        okbutton.setText('OK')
        cancelbutton   = qt.QPushButton(bottom)
        cancelbutton.setText('Cancel')
        okbutton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        cancelbutton.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed))
        HorizontalSpacer(bottom)
        self.connect(cancelbutton, qt.SIGNAL("clicked()"), self.reject)
        self.connect(okbutton,     qt.SIGNAL("clicked()"), self.accept)

        self.AText.setFocus()

    def _Aslot(self):
        qstring = self.AText.text()
        try:
            value = float(str(qstring))
        except:
            msg=qt.QMessageBox(self.AText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.AText.setFocus()
        
    def _Bslot(self):
        qstring = self.BText.text()
        try:
            value = float(str(qstring))
        except:
            msg=qt.QMessageBox(self.BText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.BText.setFocus()

    def _Cslot(self):
        qstring = self.CText.text()
        try:
            value = float(str(qstring))
        except:
            msg=qt.QMessageBox(self.CText)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            msg.exec_loop()
            self.CText.setFocus()
        
    def __copybuttonclicked(self):
        item, text = self.combo.getcurrent()
        self.AText.setText("%.4g" % self.caldict[text]['A'])
        self.BText.setText("%.4g" % self.caldict[text]['B'])
        self.CText.setText("%.4g" % self.caldict[text]['C'])
            

    def getdict(self):
        dict = {}
        dict[self.currentcal] = {}
        dict[self.currentcal]['A'] = float(str(self.AText.text()))
        dict[self.currentcal]['B'] = float(str(self.BText.text()))
        dict[self.currentcal]['C'] = float(str(self.CText.text()))
        if dict[self.currentcal]['C'] != 0.0:
            dict[self.currentcal]['order'] = 2
        else:
            dict[self.currentcal]['order'] = 1
        self.caldict.update(dict)        
        return copy.deepcopy(self.caldict)
    
                                 
class SimpleComboBox(qt.QComboBox):
    def __init__(self,parent = None,name = None,fl = 0,options=['1','2','3']):
        qt.QComboBox.__init__(self,parent)
        self.setoptions(options) 

    def setoptions(self,options=['1','2','3']):
        self.clear()
        if qt.qVersion() < '4.0.0':
            self.insertStrList(options)
        else:
            for item in options:
                self.addItem(qt.QString(item))

    def getcurrent(self):
        return   self.currentItem(),str(self.currentText())
             
def test(x,y,legend):
    app = qt.QApplication(args)
    if qt.qVersion() < '4.0.0':
        qt.QObject.connect(app,qt.SIGNAL("lastWindowClosed()"),
                           app, qt.SLOT("quit()"))
    demo = McaCalWidget(x=x,y=y,modal=1,legend=legend)
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(demo)
        ret=demo.exec_loop()
    else:
        ret=demo.exec_()
    if ret == qt.QDialog.Accepted:
        dict=demo.getdict()
    else:
        dict={}
    print " output = ",dict
    demo.close()
    del demo
    #app.exec_loop()

if __name__ == '__main__':
    import getopt
    import specfile
    options     = 'f:s:o'
    longoptions = ['file=','scan=','pkm=',
                    'output=','linear=','strip=',
                    'maxiter=','sumflag=','plotflag=']
    opts, args = getopt.getopt(
        sys.argv[1:],
        options,
        longoptions)
    inputfile = None
    scan      = None
    pkm       = None
    scankey   = None
    plotflag  = 0
    strip = 1
    linear    = 0
    for opt,arg in opts:
        if opt in ('-f','--file'):
            inputfile = arg
        if opt in ('-s','--scan'):
            scan = arg
        if opt in ('--pkm'):
            pkm = arg
        if opt in ('--linear'):
            linear = int(float(arg))
        if opt in ('--strip'):
            strip = int(float(arg))
        if opt in ('--maxiter'):
            maxiter = int(float(arg))
        if opt in ('--sum'):
            sumflag = int(float(arg))
        if opt in ('--plotflag'):
            plotflag = int(float(arg))
    if inputfile is None:
        inputfile = '03novs060sum.mca'
    sf=specfile.Specfile(inputfile)
    if scankey is None:
        scan=sf[0]
    else:
        scan=sf.select(scankey)
    nbmca=scan.nbmca()
    mcadata=scan.mca(1)
    y=Numeric.array(mcadata).astype(Numeric.Float)
    x=Numeric.arange(len(y)).astype(Numeric.Float)
    test(x,y,inputfile)

