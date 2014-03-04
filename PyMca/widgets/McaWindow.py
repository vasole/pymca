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
import sys
import os
import numpy
#from numpy import argsort, nonzero, take
import time
import traceback
from PyMca import PyMcaQt as qt
if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str
if __name__ == "__main__":
    app = qt.QApplication([])

import copy

from PyMca.widgets import ScanWindow
from PyMca.PyMca_Icons import IconDict
from PyMca.widgets import McaControlGUI
from PyMca import ConfigDict
from PyMca import McaAdvancedFit
from PyMca import DataObject
from PyMca import McaCalWidget
from PyMca import McaSimpleFit
from PyMca import Specfit
from PyMca import SpecfitFuns
from PyMca import PyMcaPrintPreview
from PyMca import PyMcaDirs
#implement the plugins interface
try:
    from PyMca import QPyMcaMatplotlibSave1D
    MATPLOTLIB = True
    #force understanding of utf-8 encoding
    #otherways it cannot generate svg output
    try:
        import encodings.utf_8
    except:
        #not a big problem
        pass
except:
    MATPLOTLIB = False

from PyMca import SimpleFitGUI

DEBUG = 0
class McaWindow(ScanWindow.ScanWindow):
    def __init__(self, parent=None, name="Mca Window", specfit=None, backend=None,
                 plugins=True, newplot=False, roi=True, fit=True, **kw):

        ScanWindow.ScanWindow.__init__(self, parent,
                                         name=name,
                                         newplot=newplot,
                                         plugins=plugins,
                                         backend=backend,
                                         roi=roi,
                                         fit=fit,
                                         **kw)
        self.setWindowType("MCA")
        # this two objects are the same
        self.dataObjectsList = self._curveList
        # but this is tricky
        self.dataObjectsDict = {}
        
        #self.setWindowTitle(name)
        self.outputDir = None
        self.outputFilter = None
        self.matplotlibDialog = None


        self.calibration = 'None'
        self.calboxoptions = ['None','Original (from Source)','Internal (from Source OR PyMca)']
        self.caldict={}
        self.calwidget   =  None
        self.currentROI = None
        self.roimarkers     = [-1,-1]
        self._middleRoiMarker = -1
        self._middleRoiMarkerFlag = False
        self.elementmarkers = []
        self.peakmarker     = None
        self.dataObjectsDict = {}
        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            self.specfit = specfit
        self.simplefit   = McaSimpleFit.McaSimpleFit(specfit=self.specfit)
        self.specfit.fitconfig['McaMode'] = 1

        self.advancedfit = McaAdvancedFit.McaAdvancedFit()

        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)
        if DEBUG:
            print("printPreview id = %d" % id(self.printPreview))

        self._buildControlWidget()
        #self._toggleROI()
        self._toggleCounter = 2
        self._togglePointsSignal()
        self.changeGridLevel()
        self.connections()
        self.setGraphYLabel('Counts')

        if 1:
            self.fitButtonMenu = qt.QMenu()
            self.fitButtonMenu.addAction(QString("Simple"),    self.mcaSimpleFitSignal)
            self.fitButtonMenu.addAction(QString("Advanced") , self.mcaAdvancedFitSignal)
            #self.fitButtonMenu.addAction(QString("Simple Fit"),
            #                       self._simpleFitSignal)
            #self.fitButtonMenu.addAction(QString("Customized Fit") ,
            #                       self._customFitSignal)

    def _buildControlWidget(self):
        self.controlWidget = McaControlGUI.McaControlGUI()
        self.roiWidget  = self.controlWidget
        self.roiDockWidget = None
        self.controlWidget.sigMcaControlGUISignal.connect(self.__anasignal)
        self.controlWidget.sigMcaROIWidgetSignal.connect(self._roiSignal)
        
    def _toggleROI(self):
        if self.roiDockWidget is None:
            self.roiDockWidget = qt.QDockWidget(self)
            self.roiDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.roiDockWidget.setWidget(self.controlWidget)
            w = self.centralWidget().width()
            h = self.centralWidget().height()
            if w > 1.25*h:
                self.addDockWidget(qt.Qt.RightDockWidgetArea,
                                   self.roiDockWidget)
            else:
                self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                                   self.roiDockWidget)
            if hasattr(self, "legendDockWidget"):
                self.tabifyDockWidget(self.roiDockWidget,
                                      self.legendDockWidget)
            self.roiDockWidget.setWindowTitle(self.windowTitle()+(" ROI"))
        if self.roiDockWidget.isHidden():
            self.roiDockWidget.show()
        else:
            self.roiDockWidget.hide()

    def _roiSignal(self, ddict):
        return super(McaWindow, self)._roiSignal(ddict)

    def connections(self):
        #self.connect(self.scanfit,    qt.SIGNAL('ScanFitSignal') , self.__anasignal)
        self.connect(self.simplefit,  qt.SIGNAL('McaSimpleFitSignal') , self.__anasignal)
        self.connect(self.advancedfit,qt.SIGNAL('McaAdvancedFitSignal') , self.__anasignal)
        #self.connect(self.scanwindow, qt.SIGNAL('ScanWindowSignal') ,   self.__anasignal)

    def mcaSimpleFitSignal(self, ddict):
        print(ddict)

    def getActiveCurve(self, just_legend=False):
        legend = super(McaWindow, self).getActiveCurve(just_legend)
        if just_legend:
            return legend
        activeCurve = legend
        if not len(activeCurve):
            return None

        legend = activeCurve[2]
        curveinfo = activeCurve[3]

        if legend in self.dataObjectsDict.keys():
            info  = self.dataObjectsDict[legend].info
            x = self.dataObjectsDict[legend].x[0]
            y = self.dataObjectsDict[legend].y[0]
        else:
            info = None

        if info is not None:
            if self.calibration == 'None':
                calib = [0.0,1.0,0.0]
            else:
                if 'McaCalib' in curveinfo:
                    calib = curveinfo['McaCalib']
                else:
                    calib = [0.0, 1.0, 0.0]
            calibrationOrder = curveinfo.get('McaCalibOrder',2)
            if calibrationOrder == 'TOF':
                x = calib[2] + calib[0] / pow(x - calib[1],2)
            else:
                x = calib[0] + calib[1] * x + calib[2] * x * x
        else:
            info = {}
        info['xlabel'] = self.getGraphXLabel()
        info['ylabel'] = self.getGraphYLabel()
        return x, y, legend, info

    def mcaAdvancedFitSignal(self):
        legend = self.getActiveCurve(just_legend=True)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            msg.setWindowTitle('MCA Window')
            msg.exec_()
            return
        x, y, legend, info = self.getActiveCurve()
        if self.calibration == 'None':
            xmin, xmax =self.getGraphXLimits()
            calib = info.get('McaCalibSource', [0.0,1.0,0.0])
        else:
            calib = info['McaCalib']
            xmin, xmax =self.getGraphXLimits()
            energy = calib[0] + calib[1] * x + calib[2] * x * x
            i1 = min(numpy.nonzero(energy >= xmin)[0])
            i2 = max(numpy.nonzero(energy <= xmax)[0])
            xmin = x[i1] * 1.0
            xmax = x[i2] * 1.0

        if self.simplefit is not None:
            self.simplefit.hide()
        self.advancedfit.show()
        self.advancedfit.setFocus()
        self.advancedfit.raise_()

        if info is not None:
            xlabel = 'Channel'
            self.advancedfit.setData(x=x,y=y,
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
            msg.exec_()

        
    def __anasignal(self,dict):
        if DEBUG:
            print("__anasignal called dict = ",dict)
            
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
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return
                else:
                    info,x, y = self.getinfodatafromlegend(legend)
                    if info is None: return
                    ndict = {}
                    ndict[legend] = {'order':1,'A':0.0,'B':1.0,'C':0.0}
                    if legend in self.caldict:
                        ndict[legend].update(self.caldict[legend])
                        if abs(ndict[legend]['C']) > 0.0:
                            ndict[legend]['order']  = self.caldict[legend].get('order', 2)
                    elif 'McaCalib' in info:
                        if type(info['McaCalib'][0]) == type([]):
                            calib = info['McaCalib'][0]
                        else:
                            calib = info['McaCalib']
                        calibrationOrder = info.get('McaCalibOrder', 2)
                        if len(calib) > 1:
                            ndict[legend]['A'] = calib[0]
                            ndict[legend]['B'] = calib[1]
                            if len(calib) >2:
                                ndict[legend]['order']  = calibrationOrder
                                ndict[legend]['C']      = calib[2]
                    caldialog = McaCalWidget.McaCalWidget(legend=legend,
                                                             x=x,
                                                             y=y,
                                                             modal=1,
                                                             caldict=ndict,
                                                             fl=0)
                    #info,x,y = self.getinfodatafromlegend(legend)
                    #caldialog.graph.newCurve("fromlegend",x=x,y=y)
                    if QTVERSION < '4.0.0':
                        ret = caldialog.exec_loop()
                    else:
                        ret = caldialog.exec_()

                    if ret == qt.QDialog.Accepted:
                        self.caldict.update(caldialog.getDict())
                        item, text = self.control.calbox.getcurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.controlWidget.calbox.setoptions(options)
                        except:
                            pass
                        if QTVERSION < '4.0.0':
                            self.controlWidget.calbox.setCurrentItem(item)
                        else:
                            self.controlWidget.calbox.setCurrentIndex(item)
                        self.refresh()
                    del caldialog
            elif dict['button'] == 'CalibrationCopy':
                #legend,x,y = self.graph.getactivecurve()
                legend = self.graph.getactivecurve(justlegend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.exec_()
                    return
                else:
                    info,x, y = self.getinfodatafromlegend(legend)
                    if info is None: return
                    ndict=copy.deepcopy(self.caldict)
                    if 'McaCalib' in info:
                        if type(info['McaCalib'][0]) == type([]):
                            sourcecal = info['McaCalib'][0]
                        else:
                            sourcecal = info['McaCalib']
                    else:
                        sourcecal = [0.0,1.0,0.0]
                    for curve in self.graph.curveslist:
                        curveinfo = self.graph.getcurveinfo(curve)
                        if 'McaCalibSource' in curveinfo:
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
                    
                    if not (legend in self.caldict):
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
                    #caldialog.graph.newCurve("fromlegend",x=x,y=y)
                    ret = caldialog.exec_()
                    if ret == qt.QDialog.Accepted:
                        self.caldict.update(caldialog.getDict())
                        item, text = self.controlWidget.calbox.getcurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.controlWidget.calbox.setoptions(options)
                        except:
                            pass
                        self.controlWidget.calbox.setCurrentIndex(item)
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
                    self.controlWidget.calbox.setoptions(options)
                    self.controlWidget.calbox.setCurrentIndex(options.index(itemtext))                        
                    self.calibration = itemtext * 1
                    self.controlWidget._calboxactivated(itemtext)
                except:
                    text = "Error. Problem updating combobox"
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText(text)
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
                    print("Unknown Fit Event")
        elif dict['event'] == 'activated':
            # A comboBox has been selected
            if   dict['boxname'] == 'Source':
                pass
            elif dict['boxname'] == 'Calibration':
                self.calibration = dict['box'][1]
                self.clearMarkers()
                self.refresh()
                self.resetZoom()
                
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
                    print("Unknown combobox",dict['boxname'])

        elif (dict['event'] == 'EstimateFinished'):
            pass
        elif (dict['event'] == 'McaAdvancedFitFinished') or \
             (dict['event'] == 'McaAdvancedFitMatrixFinished') :
            x      = dict['result']['xdata']
            yb     = dict['result']['continuum']
            legend0= dict['info']['legend']
            fitcalibration = [dict['result']['fittedpar'][0],
                              dict['result']['fittedpar'][1],
                              0.0] 
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
                newDataObject.info['McaCalib']  = fitcalibration * 1
                newDataObject.x = [x]
                newDataObject.y = [ymatrix]
                newDataObject.m = None
                self.dataObjectsDict[legend3] = newDataObject
                #self.graph.newCurve(legend3,x=x,y=ymatrix,logfilter=1)
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
                newDataObject.info['McaCalib']  = fitcalibration * 1
                newDataObject.data = numpy.reshape(numpy.concatenate((x,yfit,yb),0),(3,len(x)))
                newDataObject.x = [x]
                newDataObject.y = [yfit]
                newDataObject.m = None

                self.dataObjectsDict[legend] = newDataObject
                #self.graph.newCurve(legend,x=x,y=yfit,logfilter=1)

                #the same for the background
                legend2 = dict['info']['legend'] + " Bkg"
                newDataObject2 = DataObject.DataObject()
                newDataObject2.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
                newDataObject2.info['SourceType']= 'AdvancedFit'
                newDataObject2.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
                newDataObject2.info['legend'] = legend2
                newDataObject2.info['Key']  = legend2
                newDataObject2.info['McaCalib']  = fitcalibration * 1
                newDataObject2.data = None
                newDataObject2.x = [x]
                newDataObject2.y = [yb]
                newDataObject2.m = None
                self.dataObjectsDict[legend2] = newDataObject2
                #self.graph.newCurve(legend2,x=x,y=yb,logfilter=1)

            if not (legend in self.caldict):
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
                self.controlWidget.calbox.setoptions(options)
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
                    if QTVERSION < '4.0.0':
                        self.controlWidget.calbox.setCurrentItem(options.index(legend))
                    else:
                        self.controlWidget.calbox.setCurrentIndex(options.index(legend))
                    self.calibration = legend
                    self.controlWidget._calboxactivated(legend)
                    self.graph.sety1axislimits(ymin, ymax, False)
                    if emin < emax:
                        self.graph.setx1axislimits(emin, emax, True)
                    else:
                        self.graph.setx1axislimits(emax, emin, True)
            except:
                self.refresh()
                #self.graph.replot()

        elif dict['event'] == 'McaFitFinished':
            mcaresult = dict['data']
            legend = dict['info']['legend'] + " "
            i = 0
            xfinal = []
            yfinal = []
            ybfinal= []
            regions = []
            legend0= dict['info']['legend']
            mcamode = True
            for result in mcaresult:
                i += 1
                if result['chisq'] is not None:
                     mcamode = result['fitconfig']['McaMode']                         
                     idx=numpy.nonzero((self.specfit.xdata0>=result['xbegin']) & \
                                    (self.specfit.xdata0<=result['xend']))[0]
                     x=numpy.take(self.specfit.xdata0,idx)
                     y=self.specfit.gendata(x=x,parameters=result['paramlist'])
                     nparb= len(self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1])
                     yb   = self.specfit.gendata(x=x,parameters=result['paramlist'][0:nparb])
                     xtoadd = numpy.take(self.dataObjectsDict[legend0].x[0],idx).tolist()
                     if not len(xtoadd): continue
                     xfinal = xfinal + xtoadd
                     regions.append([xtoadd[0],xtoadd[-1]])
                     yfinal = yfinal + y.tolist()
                     ybfinal= ybfinal + yb.tolist()
                    #self.graph.newCurve(legend + 'Region %d' % i,x=x,y=yfit,logfilter=1)
            legend = legend0 + " SFit"            
            if legend in self.dataObjectsDict.keys():
                if legend in self.graph.curves.keys():                    
                    if mcamode:
                        if not ('baseline' in self.dataObjectsDict[legend].info):
                            self.graph.delcurve(legend)
                    else:
                        if 'baseline' in self.dataObjectsDict[legend].info:
                            self.graph.delcurve(legend)
            #copy the original info from the curve
            newDataObject = DataObject.DataObject()
            newDataObject.info = copy.deepcopy(self.dataObjectsDict[legend0].info)
            newDataObject.info['SourceType']= 'SimpleFit'
            newDataObject.info['SourceName'] = 1 * self.dataObjectsDict[legend0].info['SourceName']
            newDataObject.info['legend']    = legend
            newDataObject.info['Key']       = legend
            newDataObject.info['CalMode']   = self.__simplefitcalmode
            newDataObject.info['McaCalib']  = self.__simplefitcalibration
            x    = numpy.array(xfinal)
            yfit = numpy.array(yfinal)
            yb = numpy.array(ybfinal)
            newDataObject.x = [x]
            newDataObject.y = [yfit]
            newDataObject.m = [numpy.ones(len(yfit)).astype(numpy.float)]
            if mcamode:
                newDataObject.info['regions']   = regions
                newDataObject.info['baseline'] = yb
            self.dataObjectsDict[legend] = newDataObject
            self.refresh()
            return
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

        elif (dict['event'] == 'McaAdvancedFitElementClicked') or \
             (dict['event'] == 'ElementClicked'):
            #this has been moved to the fit window
            pass
                
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
            return super(McaWindow, self)._roiSignal(dict)
        elif dict['event'] == 'DelROI':
            return super(McaWindow, self)._roiSignal(dict)
            
        elif dict['event'] == 'ResetROI':
            return super(McaWindow, self)._roiSignal(dict)
            
        elif dict['event'] == 'ActiveROI':
            print("ActiveROI event")
            pass
        elif dict['event'] == 'selectionChanged':
            if DEBUG:
                print("Selection changed")
            ##############
            self.roilist,self.roidict = self.roiwidget.getroilistanddict()
            fromdata = dict['roi']['from']
            todata   = dict['roi']['to']
            if self.roimarkers[0] == -1:
                self.roimarkers[0] = self.graph.insertx1marker(fromdata,1.1,
                                        label = 'ROI min')
            if self.roimarkers[1] == -1:
                self.roimarkers[1] = self.graph.insertx1marker(todata,1.1,
                                        label = 'ROI max')
            self.graph.setx1markerpos(self.roimarkers[0],fromdata)
            self.graph.setx1markerpos(self.roimarkers[1],todata )
            self.currentROI = dict['key']
            if dict['key'] == 'ICR':
                #select the colors
                self.graph.setmarkercolor(self.roimarkers[1],'black' )
                self.graph.setmarkercolor(self.roimarkers[0],'black' )
                #set the follow mouse propierty
                self.graph.setmarkerfollowmouse(self.roimarkers[1],0)
                self.graph.setmarkerfollowmouse(self.roimarkers[0],0)
                #deal with the middle marker
                self.graph.removeMarker(self._middleRoiMarker)
                self._middleRoiMarker = -1
                #disable marker mode
                self.graph.disablemarkermode()
            else:
                if self._middleRoiMarkerFlag:
                    pos = 0.5 * (fromdata + todata)                        
                    if self._middleRoiMarker == -1:
                        self._middleRoiMarker = self.graph.insertx1marker(pos,\
                                                            1.1,
                                                            label = ' ')
                    else:
                        self.graph.setx1markerpos(self._middleRoiMarker,
                                                  pos)
                else:
                    if self._middleRoiMarker != -1:
                        self.graph.removeMarker(self._middleRoiMarker)
                #select the colors
                self.graph.setmarkercolor(self.roimarkers[0],'blue' )
                self.graph.setmarkercolor(self.roimarkers[1],'blue' )
                #set the follow mouse propierty
                self.graph.setmarkerfollowmouse(self.roimarkers[0],1)
                self.graph.setmarkerfollowmouse(self.roimarkers[1],1)
                #middle marker
                if self._middleRoiMarker != -1:
                    self.graph.setmarkercolor(self._middleRoiMarker,'yellow' )
                    self.graph.setmarkerfollowmouse(self._middleRoiMarker, 1)
                self.graph.enablemarkermode()
            if dict['colheader'] in ['From', 'To']:
                dict ={}
                dict['event']  = "SetActiveCurveEvent"
                dict['legend'] = self.graph.getactivecurve(justlegend=1)
                self.__graphsignal(dict)
            elif dict['colheader'] == 'Raw Counts':    
                pass
            elif dict['colheader'] == 'Net Counts':    
                pass
            else:
                self.emitCurrentROISignal()
            self.graph.replot()
        else:
            if DEBUG:
                print("Unknown or ignored event",dict['event'])


    def emitCurrentROISignal(self):
        if self.currentROI is None:
            return
        #I have to get the current calibration
        if self.getGraphXLabel().upper() != "CHANNEL":
            #I have to get the energy
            A = self.control.calinfo.caldict['']['A']
            B = self.control.calinfo.caldict['']['B']
            C = self.control.calinfo.caldict['']['C']
            order = self.control.calinfo.caldict['']['order']
        else:
            A = 0.0
            try:
                legend = self.graph.getActiveCurve(just_legend=True)
                if legend in self.dataObjectsDict.keys():
                    A = self.dataObjectsDict[legend].x[0][0]
            except:
                if DEBUG:
                    print("X axis offset not found")
            B = 1.0
            C = 0.0
            order = 1
        key = self.currentROI        
        roiList, roiDict = self.roiWidget.getROIListAndDict()
        fromdata = roiDict[key]['from' ]
        todata   = roiDict[key]['to']
        ddict = {}
        ddict['event'] = "ROISignal"
        ddict['name'] = key
        ddict['from'] = fromdata
        ddict['to']   = todata
        ddict['type'] = roiDict[self.currentROI]["type"]
        ddict['calibration']= [A, B, C, order]
        self.sigROISignal.emit(ddict)

    def setDispatcher(self, w):
        self.connect(w, qt.SIGNAL("addSelection"),
                         self._addSelection)
        self.connect(w, qt.SIGNAL("removeSelection"),
                         self._removeSelection)
        self.connect(w, qt.SIGNAL("replaceSelection"),
                         self._replaceSelection)

    def _addSelection(self, selection, replot=True):
        if DEBUG:
            print("__add, selection = ",selection)

        if type(selection) == type([]):
            sellist = selection
        else:
            sellist = [selection]

        for sel in sellist:
            # force the selections to include their source for completeness?
            # source = sel['SourceName']
            key    = sel['Key']
            if "scanselection" in sel:
                if sel["scanselection"] not in [False, "MCA"]:
                    continue
            mcakeys    = [key]
            for mca in mcakeys:
                curveinfo={}
                legend = sel['legend']
                dataObject = sel['dataobject']
                info = dataObject.info
                data  = dataObject.y[0]
                if "selectiontype" in dataObject.info:
                    if dataObject.info["selectiontype"] != "1D": continue
                if dataObject.x is None:
                    xhelp = None
                else:
                    xhelp = dataObject.x[0]

                if xhelp is None:
                    xhelp =info['Channel0'] + numpy.arange(len(data)).astype(numpy.float)
                    dataObject.x = [xhelp]

                ylen = len(data)
                if ylen == 1:
                    if len(xhelp) > 1:
                        data = data[0] * numpy.ones(len(xhelp)).astype(numpy.float)
                        dataObject.y = [data]
                elif len(xhelp) == 1:
                    xhelp = xhelp[0] * numpy.ones(ylen).astype(numpy.float)
                    dataObject.x = [xhelp]

                if not hasattr(dataObject, 'm'):
                    dataObject.m = None

                if dataObject.m is not None:
                    if len(dataObject.m[0]) > 0:
                        mdata = dataObject.m[0]
                        if len(mdata) == len(data):
                            mdata[data == 0] += 0.00000001
                            index = numpy.nonzero(mdata)[0]
                            if not len(index):
                                continue
                            xhelp = numpy.take(xhelp, index)
                            data = numpy.take(data, index)
                            mdata = numpy.take(mdata, index)
                            data = data/mdata
                            dataObject.x = [xhelp * 1]
                            dataObject.m = [numpy.ones(len(data)).astype(numpy.float)]
                        elif (len(mdata) == 1) or (ylen == 1):
                            if mdata[0] == 0.0:
                                continue
                            data = data/mdata
                        else:
                            raise ValueError("Cannot normalize data")
                        dataObject.y = [data]
                self.dataObjectsDict[legend] = dataObject
                if ('baseline' in info) and ('regions' in info):
                    simplefitplot = True
                else:
                    simplefitplot = False
                try:
                    calib = [0.0,1.0,0.0]
                    for inputkey in ['baseline', 'regions']: 
                        if inputkey in info:
                            curveinfo[inputkey] = info[inputkey]
                    curveinfo['McaCalib'] = calib
                    if 'McaCalib' in info:
                        if type(info['McaCalib'][0]) == type([]):
                            calib0 = info['McaCalib'][info['McaDet']-1]
                        else:
                            calib0 = info['McaCalib']
                        curveinfo['McaCalibSource'] = calib0
                    if self.calibration == self.calboxoptions[1]:
                        if 'McaCalib' in info:
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
                            if simplefitplot:
                                inforegions = []
                                for region in info['regions']:
                                    inforegions.append([calib[0] + \
                                                        calib[1] * region[0] +\
                                                        calib[2] * region[0] * region[0],
                                                        calib[0] + \
                                                        calib[1] * region[1] +\
                                                        calib[2] * region[1] * region[1]])
                                self.addCurve(xdata, data, legend=legend,
                                                info=curveinfo)
                            else:
                                self.addCurve(xdata, data, legend=legend,
                                                info=curveinfo)
                            self.setGraphXLabel('Energy')
                    elif self.calibration == self.calboxoptions[2]:
                        calibrationOrder = None
                        if legend in self.caldict:
                            A = self.caldict[legend]['A']
                            B = self.caldict[legend]['B']
                            C = self.caldict[legend]['C']
                            calibrationOrder = self.caldict[legend]['order']  
                            calib = [A,B,C]
                        elif 'McaCalib' in info:
                            if type(info['McaCalib'][0]) == type([]):
                                calib = info['McaCalib'][info['McaDet']-1]
                            else:
                                calib = info['McaCalib']
                        if len(calib) > 1:
                            xdata=calib[0]+ \
                                  calib[1]* xhelp
                            if len(calib) == 3:
                                if calibrationOrder == 'TOF':
                                    xdata = calib[2] + calib[0] / pow(xhelp-calib[1],2)
                                else:
                                    xdata = xdata + calib[2]* xhelp * xhelp
                            curveinfo['McaCalib'] = calib
                            curveinfo['McaCalibOrder'] = calibrationOrder
                            if simplefitplot:
                                inforegions = []
                                for region in info['regions']:
                                    if calibrationOrder == 'TOF':
                                        inforegions.append([calib[2] + calib[0] / pow(region[0]-calib[1],2),
                                                            calib[2] + calib[0] / pow(region[1]-calib[1],2)])
                                    else:
                                        inforegions.append([calib[0] + \
                                                        calib[1] * region[0] +\
                                                        calib[2] * region[0] * region[0],
                                                        calib[0] + \
                                                        calib[1] * region[1] +\
                                                        calib[2] * region[1] * region[1]])
                                self.addCurve(xdata, data,
                                                legend=legend,
                                                info=curveinfo)
                            else:
                                self.addCurve(xdata, data,
                                                legend=legend,
                                                info=curveinfo)
                            if calibrationOrder == 'ID18':
                                self.setGraphXLabel('Time')
                            else:
                                self.setGraphXLabel('Energy')
                    elif self.calibration == 'Fit':
                        print("Not yet implemented")
                        continue
                    elif self.calibration in  self.caldict.keys():
                            A = self.caldict[self.calibration]['A']
                            B = self.caldict[self.calibration]['B']
                            C = self.caldict[self.calibration]['C']
                            calibrationOrder = self.caldict[self.calibration]['order'] 
                            calib = [A,B,C]
                            if calibrationOrder == 'TOF':
                                xdata =  C + (A / ((xhelp - B) * (xhelp - B)))
                            else:
                                xdata=calib[0]+ \
                                  calib[1]* xhelp + \
                                  calib[2]* xhelp * xhelp
                            curveinfo['McaCalib'] = calib
                            curveinfo['McaCalibOrder'] = calibrationOrder
                            if simplefitplot:
                                inforegions = []
                                for region in info['regions']:
                                    if calibrationOrder == 'TOF':
                                        inforegions.append([calib[2] + calib[0] / pow(region[0]-calib[1],2),
                                                            calib[2] + calib[0] / pow(region[1]-calib[1],2)])
                                    else:
                                        inforegions.append([calib[0] + \
                                                        calib[1] * region[0] +\
                                                        calib[2] * region[0] * region[0],
                                                        calib[0] + \
                                                        calib[1] * region[1] +\
                                                        calib[2] * region[1] * region[1]])
                                self.addCurve(xdata, data,
                                              legend=legend,
                                              info=curveinfo)
                                                #baseline = info['baseline'],
                                                #regions = inforegions)
                            else:
                                self.addCurve(xdata, data,
                                              legend=legend,
                                              info=curveinfo)
                            if calibrationOrder == 'ID18':
                                self.setGraphXLabel('Time')
                            else:
                                self.setGraphXLabel('Energy')
                    else:
                        if simplefitplot:
                            self.addCurve(xhelp, data,
                                          legend=legend,
                                          info=curveinfo)
                                          #baseline = info['baseline'],
                                          #regions = info['regions'])
                        else:
                            self.addCurve(xhelp, data,
                                          legend=legend,
                                          info=curveinfo)
                        self.setGraphXLabel('Channel')
                except:
                    del self.dataObjectsDict[legend]
                    raise
        if replot:
            self.replot()
            self.resetZoom()

    def _removeSelection(self, selectionlist):
        if DEBUG:
            print("_removeSelection(self, selectionlist)",selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        legendlist = []
        for sel in sellist:
            key    = sel['Key']
            if "scanselection" in sel:
                if sel['scanselection'] not in [False, "MCA"]:
                    continue

            mcakeys    = [key]
            for mca in mcakeys:
                legend = sel['legend']
                legendlist.append(legend)

        self.removeCurves(legendlist, replot=True)

    def removeCurves(self, removelist, replot=True):
        for legend in removelist:
            self.removeCurve(legend, replot=False)
            if legend in self.dataObjectsDict.keys():
                del self.dataObjectsDict[legend]
        self.dataObjectsList = self._curveList
        if replot:
            self.replot()

    def _replaceSelection(self, selectionlist):
        if DEBUG:
            print("_replaceSelection(self, selectionlist)",selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        doit = False
        for sel in sellist:
            if "scanselection" in sel:
                if sel['scanselection'] not in [False, "MCA"]:
                    continue
            doit = True
            break
        if not doit:
            return
        self.clearCurves()
        self.dataObjectsDict={}
        self.dataObjectsList=self._curveList
        self._addSelection(selectionlist)

    def _handleMarkerEvent(self, ddict):
        if ddict['event'] == 'markerMoved':
            label = ddict['label'] 
            if label.startswith('ROI'):
                return self._handleROIMarkerEvent(ddict)
            else:
                print("Unhandled marker %s" % label)
                return

    def _graphSignalReceived(self, ddict):
        if DEBUG:
            print("_graphSignalReceived", ddict)
        if ddict['event'] in ['markerMoved', 'markerSelected']:
            return self._handleMarkerEvent(ddict)
        elif ddict['event'] == 'MouseAt':
            if self.calibration == self.calboxoptions[0]:
                self.xpos.setText('%.2f' % ddict['x'])
                self.ypos.setText('%.2f' % ddict['y'])
            else:
                self.xpos.setText('%.4f' % ddict['x'])
                self.ypos.setText('%.2f' % ddict['y'])
        elif ddict['event'] in ["curveClicked", "legendClicked", "SetActiveCurveEvent"]:
            legend = ddict.get('label', None)
            if legend is not None:
                legend = self.getActiveCurve(just_legend=True)
                if legend in self.dataObjectsDict.keys():
                    x0 = self.dataObjectsDict[legend].x[0]
                    y = self.dataObjectsDict[legend].y[0]
                    #those are the actual data
                    if str(self.getGraphXLabel()).upper() != "CHANNEL":
                        #I have to get the energy
                        A = self.controlWidget.calinfo.caldict['']['A']
                        B = self.controlWidget.calinfo.caldict['']['B']
                        C = self.controlWidget.calinfo.caldict['']['C']
                        order = self.controlWidget.calinfo.caldict['']['order']
                    else:
                        A = 0.0
                        B = 1.0
                        C = 0.0
                        order = 1
                    calib = [A,B,C]
                    if order == "TOF":
                        x = calib[2] + calib[0] / pow(x0-calib[1],2)
                    else:
                        x = calib[0]+ \
                            calib[1]* x0 + \
                            calib[2]* x0 * x0
                else:
                    print("Should not be here, the legend is not in the dict")
                    return
                #if self.roidict is None:
                roiList, roiDict = self.roiWidget.getROIListAndDict()
                if not len(x):
                    #only zeros ...
                    for i in range(len(roilist)):
                        key = roiList[i]
                        self.roidict[key]['rawcounts'] = 0.0
                        self.roidict[key]['netcounts'] = 0.0
                        #self.roidict[key]['from'  ] = 0.0
                        #self.roidict[key]['to'    ] = 0.0
                else:    
                    for i in range(len(roiList)):
                        key = roiList[i]
                        if key == 'ICR':
                            #take care of LLD
                            #if len(x) > 3:
                            #    fromdata = x[3]
                            #else:
                            fromdata = x[0]
                            todata   = x[-1]
                            #I profit to update ICR
                            roiDict[key]['from'] = x[0]
                            roiDict[key]['to'] = x[-1]
                        else:
                            fromdata = roiDict[key]['from']
                            todata   = roiDict[key]['to']
                        if roiDict[key]['type'].upper() != "CHANNEL":
                            i1 = numpy.nonzero(x>=fromdata)[0]
                            xw = numpy.take(x,i1)
                        else:
                            i1 = numpy.nonzero(x0>=fromdata)[0]
                            xw = numpy.take(x0,i1)
                        yw = numpy.take(y, i1)
                        i1 = numpy.nonzero(xw<=todata)[0]
                        xw = numpy.take(xw,i1)
                        yw = numpy.take(yw,i1)
                        counts = yw.sum()
                        roiDict[key]['rawcounts'] = counts
                        if len(yw):
                            roiDict[key]['netcounts'] = counts - \
                                                      len(yw) *  0.5 * (yw[0] + yw[-1])
                        else:
                            roiDict[key]['netcounts'] = 0
                    self.emitCurrentROISignal()
                self.roiWidget.setHeader(text="ROIs of %s" % (legend))
                self.roiWidget.fillFromROIDict(roilist=roiList,
                                               roidict=roiDict)
                try:
                    calib = self.getActiveCurve()[3]['McaCalib']
                    self.controlWidget.calinfo.setParameters({'A':calib[0],'B':calib[1],'C':calib[2]})
                except:
                    self.controlWidget.calinfo.AText.setText("?????")
                    self.controlWidget.calinfo.BText.setText("?????")
                    self.controlWidget.calinfo.CText.setText("?????")
            else:
                    self.controlWidget.calinfo.AText.setText("?????")
                    self.controlWidget.calinfo.BText.setText("?????")
                    self.controlWidget.calinfo.CText.setText("?????")                
                
        elif ddict['event'] == "RemoveCurveEvent":
            #WARNING this is to be called just from the graph!"
            legend = ddict['legend']
            self.graph.delcurve(legend)
            if legend in self.dataObjectsDict:
                del self.dataObjectsDict[legend]
            self.graph.replot()
            #I should generate an event to allow the controller
            #to eventually inform other widgets
        elif ddict['event'] == "RenameCurveEvent":
            legend = ddict['legend']
            newlegend = ddict['newlegend']
            if legend in self.dataObjectsDict:
                self.dataObjectsDict[newlegend]= copy.deepcopy(self.dataObjectsDict[legend])
                self.dataObjectsDict[newlegend].info['legend'] = newlegend
                self.graph.delcurve(legend)
                self.graph.newCurve(self.dataObjectsDict[newlegend].info['legend'],
                                    self.dataObjectsDict[newlegend].x[0],
                                    self.dataObjectsDict[newlegend].y[0])
                if legend in self.caldict:
                    self.caldict[newlegend] = copy.deepcopy(self.caldict[legend])
                del self.dataObjectsDict[legend]
            self.graph.replot()
        elif ddict['event'] == "MouseClick":
            #check if we are in automatic ROI mode
            if self.currentROI not in ["ICR", None, "None"]:
                self.roilist,self.roidict = self.roiwidget.getroilistanddict()
                fromdata = self.roidict[self.currentROI]['from']
                todata = self.roidict[self.currentROI]['to']
                pos = 0.5 * (fromdata + todata)
                delta = ddict['x'] - pos
                self.roidict[self.currentROI]['to'] += delta
                self.roidict[self.currentROI]['from'] += delta
                self.graph.setx1markerpos(self.roimarkers[0],fromdata)
                self.graph.setx1markerpos(self.roimarkers[1],todata )
                self.roiwidget.fillfromroidict(roilist=self.roilist,
                                           roidict=self.roidict)
                key = self.currentROI
                ddict = {}
                ddict['event'] = 'selectionChanged'
                ddict['key'] = key
                ddict['roi'] = {}
                ddict['roi']['from'] = self.roidict[key]['from' ]
                ddict['roi']['to'] = self.roidict[key]['to' ]
                ddict['colheader'] = 'Raw Counts'
                self.__anasignal(ddict)
                ddict ={}
                ddict['event']  = "SetActiveCurveEvent"
                ddict['legend'] = self.graph.getactivecurve(justlegend=1)
                self.__graphsignal(ddict)
        else:
            if DEBUG:
                print("Unhandled event %s" % dict['event'])

    def _customFitSignalReceived(self, ddict):
        if ddict['event'] == "FitFinished":
            newDataObject = self.__customFitDataObject

            xplot = ddict['x']
            yplot = ddict['yfit']
            newDataObject.x = [xplot]
            newDataObject.y = [yplot]
            newDataObject.m = [numpy.ones(len(yplot)).astype(numpy.float)]            

            #here I should check the log or linear status
            self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
            self.addCurve(xplot,
                          yplot,
                          legend=newDataObject.info['legend'])        

    def _scanFitSignalReceived(self, ddict):
        if DEBUG:
            print("_graphSignalReceived", ddict)
        if ddict['event'] == "EstimateFinished":
            return
        if ddict['event'] == "FitFinished":
            newDataObject = self.__fitDataObject

            xplot = self.scanFit.specfit.xdata * 1.0
            yplot = self.scanFit.specfit.gendata(parameters=ddict['data'])
            newDataObject.x = [xplot]
            newDataObject.y = [yplot]
            newDataObject.m = [numpy.ones(len(yplot)).astype(numpy.float)]            

            self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
            self.addCurve(x=xplot, y=yplot, legend=newDataObject.info['legend'])
            
    def _fitIconSignal(self):
        if DEBUG:
            print("_fitIconSignal")
        self.fitButtonMenu.exec_(self.cursor().pos())

    def _simpleFitSignal(self):
        if DEBUG:
            print("_simpleFitSignal")
        self.__QSimpleOperation("fit")

    def _customFitSignal(self):
        if DEBUG:
            print("_customFitSignal")
        self.__QSimpleOperation("custom_fit")

    def _saveIconSignal(self):
        if DEBUG:
            print("_saveIconSignal")
        self.__QSimpleOperation("save")
        
    def _averageIconSignal(self):
        if DEBUG:
            print("_averageIconSignal")
        self.__QSimpleOperation("average")
        
    def _smoothIconSignal(self):
        if DEBUG:
            print("_smoothIconSignal")
        self.__QSimpleOperation("smooth")
        
    def _getOutputFileName(self):
        #get outputfile
        self.outputDir = PyMcaDirs.outputDir
        if self.outputDir is None:
            self.outputDir = os.getcwd()
            wdir = os.getcwd()
        elif os.path.exists(self.outputDir):
            wdir = self.outputDir
        else:
            self.outputDir = os.getcwd()
            wdir = self.outputDir
            
        outfile = qt.QFileDialog(self)
        outfile.setWindowTitle("Output File Selection")
        outfile.setModal(1)
        filterlist = ['Specfile MCA  *.mca',
                      'Specfile Scan *.dat',
                      'Specfile MultiScan *.dat',
                      'Raw ASCII *.txt',
                      '","-separated CSV *.csv',
                      '";"-separated CSV *.csv',
                      '"tab"-separated CSV *.csv',
                      'OMNIC CSV *.csv',
                      'Widget PNG *.png',
                      'Widget JPG *.jpg']
        if MATPLOTLIB:
            filterlist.append('Graphics PNG *.png')
            filterlist.append('Graphics EPS *.eps')
            filterlist.append('Graphics SVG *.svg')

        if self.outputFilter is None:
            self.outputFilter = filterlist[0]
        outfile.setFilters(filterlist)
        outfile.selectFilter(self.outputFilter)
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(outfile.AcceptSave)
        outfile.setDirectory(wdir)
        ret = outfile.exec_()
        if not ret:
            return None
        self.outputFilter = qt.safe_str(outfile.selectedFilter())
        filterused = self.outputFilter.split()
        filetype  = filterused[1]
        extension = filterused[2]
        outdir = qt.safe_str(outfile.selectedFiles()[0])
        try:            
            self.outputDir  = os.path.dirname(outdir)
            PyMcaDirs.outputDir = os.path.dirname(outdir)
        except:
            print("setting output directory to default")
            self.outputDir  = os.getcwd()
        try:     
            outputFile = os.path.basename(outdir)
        except:
            outputFile = outdir
        outfile.close()
        del outfile
        if len(outputFile) < 5:
            outputFile = outputFile + extension[-4:]
        elif outputFile[-4:] != extension[-4:]:
            outputFile = outputFile + extension[-4:]
        return os.path.join(self.outputDir, outputFile), filetype, filterused

    def array2SpecMca(self, data):
        """ Write a python array into a Spec array.
            Return the string containing the Spec array
        """
        tmpstr = "@A "
        length = len(data)
        for idx in range(0, length, 16):
            if idx+15 < length:
                for i in range(0,16):
                    tmpstr += "%.7g " % data[idx+i]
                if idx+16 != length:
                    tmpstr += "\\"
            else:
                for i in range(idx, length):
                    tmpstr += "%.7g " % data[i]
            tmpstr += "\n"
        return tmpstr
        
    def __QSimpleOperation(self, operation):
        try:
            self.__simpleOperation(operation)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("%s" % sys.exc_info()[1])
            msg.exec_()
    
    def __simpleOperation(self, operation):
        if operation == 'subtract':
            self._subtractOperation()
            return
        if operation != "average":
            #get active curve
            legend = self.getActiveCurveLegend()
            if legend is None:return

            found = False
            for key in self.dataObjectsList:
                if key == legend:
                    found = True
                    break

            if found:
                dataObject = self.dataObjectsDict[legend]
            else:
                print("I should not be here")
                print("active curve =",legend)
                print("but legend list = ",self.dataObjectsList)
                return
            y = dataObject.y[0]
            if dataObject.x is not None:
                x = dataObject.x[0]
            else:
                x = numpy.arange(len(y)).astype(numpy.float)
            ilabel = dataObject.info['selection']['y'][0]
            ylabel = dataObject.info['LabelNames'][ilabel]
            if len(dataObject.info['selection']['x']):
                ilabel = dataObject.info['selection']['x'][0]
                xlabel = dataObject.info['LabelNames'][ilabel]
            else:
                xlabel = "Point Number"
        else:
            x = []
            y = []
            legend = ""
            i = 0
            ndata = 0
            for key in self._curveList:
                if DEBUG:
                    print("key -> ", key)
                if key in self.dataObjectsDict.keys():
                    x.append(self.dataObjectsDict[key].x[0]) #only the first X
                    if len(self.dataObjectsDict[key].y) == 1:
                        y.append(self.dataObjectsDict[key].y[0])
                    else:
                        sel_legend = self.dataObjectsDict[key].info['legend']
                        ilabel = 0
                        #I have to get the proper y associated to the legend
                        if sel_legend in key:
                            if key.index(sel_legend) == 0:
                                label = key[len(sel_legend):]
                                while (label.startswith(' ')):
                                    label = label[1:]
                                    if not len(label):
                                        break
                                if label in self.dataObjectsDict[key].info['LabelNames']:
                                    ilabel = self.dataObjectsDict[key].info['LabelNames'].index(label)
                                if DEBUG:
                                    print("LABEL = ", label)
                                    print("ilabel = ", ilabel)
                        y.append(self.dataObjectsDict[key].y[ilabel])
                    if i == 0:
                        legend = key
                        firstcurve = key
                        i += 1
                    else:
                        legend += " + " + key
                    ndata += 1
            if ndata == 0: return #nothing to average
            dataObject = self.dataObjectsDict[firstcurve]

        if operation == "save":
            #getOutputFileName
            filename = self._getOutputFileName()
            if filename is None:return
            filterused = filename[2]
            filetype = filename[1]
            filename = filename[0]
            if os.path.exists(filename):
                os.remove(filename)
            if filterused[0].upper() == "WIDGET":
                fformat = filename[-3:].upper()
                pixmap = qt.QPixmap.grabWidget(self)
                if not pixmap.save(filename, fformat):
                    qt.QMessageBox.critical(self,
                                        "Save Error",
                                        "%s" % sys.exc_info()[1])
                return
            if MATPLOTLIB:
                try:
                    if filename[-3:].upper() in ['EPS', 'PNG', 'SVG']:
                        self.graphicsSave(filename)
                        return
                except:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Graphics Saving Error: %s" % (sys.exc_info()[1]))
                    msg.exec_()
                    return
            systemline = os.linesep
            os.linesep = '\n'
            try:
                ffile=open(filename,'wb')
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_loop()
                return
            try:
                if filetype in ['Scan', 'MultiScan']:
                    ffile.write("#F %s\n" % filename)
                    savingDate = "#D %s\n"%(time.ctime(time.time()))                    
                    ffile.write(savingDate)
                    ffile.write("\n")
                    ffile.write("#S 1 %s\n" % legend)
                    ffile.write(savingDate)
                    ffile.write("#N 2\n")
                    ffile.write("#L %s  %s\n" % (xlabel, ylabel) )
                    for i in range(len(y)):
                        ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
                    ffile.write("\n")
                    if filetype == 'MultiScan':
                        scan_n  = 1
                        keylist = list(self.dataObjectsList)
                        for key in self._curveList:
                            if key not in keylist:
                                keylist.append(key)        
                        for key in keylist:
                            if key not in self.dataObjectsDict.keys():
                                continue
                            if key == legend: continue
                            dataObject = self.dataObjectsDict[key]
                            y = dataObject.y[0]
                            if dataObject.x is not None:
                                x = dataObject.x[0]
                            else:
                                x = numpy.arange(len(y)).astype(numpy.float)
                            ilabel = dataObject.info['selection']['y'][0]
                            ylabel = dataObject.info['LabelNames'][ilabel]
                            if len(dataObject.info['selection']['x']):
                                ilabel = dataObject.info['selection']['x'][0]
                                xlabel = dataObject.info['LabelNames'][ilabel]
                            else:
                                xlabel = "Point Number"
                            scan_n += 1
                            ffile.write("#S %d %s\n" % (scan_n, key))
                            ffile.write(savingDate)
                            ffile.write("#N 2\n")
                            ffile.write("#L %s  %s\n" % (xlabel, ylabel) )
                            for i in range(len(y)):
                                ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
                            ffile.write("\n")
                elif filetype == 'ASCII':
                    for i in range(len(y)):
                        ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
                elif filetype == 'CSV':
                    if "," in filterused[0]:
                        csvseparator = ","
                    elif ";" in filterused[0]:
                        csvseparator = ";"
                    elif "OMNIC" in filterused[0]:
                        csvseparator = ","
                    else:
                        csvseparator = "\t"
                    if "OMNIC" not in filterused[0]:
                        ffile.write('"%s"%s"%s"\n' % (xlabel,csvseparator,ylabel)) 
                    for i in range(len(y)):
                        ffile.write("%.7E%s%.7E\n" % (x[i], csvseparator,y[i]))
                else:
                    ffile.write("#F %s\n" % filename)
                    ffile.write("#D %s\n"%(time.ctime(time.time())))
                    ffile.write("\n")
                    ffile.write("#S 1 %s\n" % legend)
                    ffile.write("#D %s\n"%(time.ctime(time.time())))
                    ffile.write("#@MCA %16C\n")
                    ffile.write("#@CHANN %d %d %d 1\n" %  (len(y), x[0], x[-1]))
                    ffile.write("#@CALIB %.7g %.7g %.7g\n" % (0, 1, 0))
                    ffile.write(self.array2SpecMca(y))
                    ffile.write("\n")
                ffile.close()
                os.linesep = systemline
            except:
                os.linesep = systemline
                raise
            return

        #create the output data object
        newDataObject = DataObject.DataObject()
        newDataObject.data = None
        newDataObject.info = copy.deepcopy(dataObject.info)
        if 'selectionlegend' in newDataObject.info:
            del newDataObject.info['selectionlegend']
        if not ('operations' in newDataObject.info):
            newDataObject.info['operations'] = []
        newDataObject.info['operations'].append(operation)

        sel = {}
        sel['SourceType'] = "Operation"
        #get new x and new y
        if operation == "derivate":
            #xmin and xmax
            xlimits=self.getGraphXLimits()
            xplot, yplot = self.simpleMath.derivate(x, y, xlimits=xlimits)
            ilabel = dataObject.info['selection']['y'][0]
            ylabel = dataObject.info['LabelNames'][ilabel]
            newDataObject.info['LabelNames'][ilabel] = ylabel+"'"
            sel['SourceName'] = legend
            sel['Key']    = "'"
            sel['legend'] = legend + sel['Key']
            outputlegend  = legend + sel['Key']
        elif operation == "average":
            xplot, yplot = self.simpleMath.average(x, y)
            sel['SourceName'] = legend
            sel['Key']    = ""
            sel['legend'] = "(%s)/%d" % (legend, ndata)
            outputlegend  = "(%s)/%d" % (legend, ndata)
        elif operation == "swapsign":
            xplot =  x * 1
            yplot = -y
            sel['SourceName'] = legend
            sel['Key']    = ""
            sel['legend'] = "-(%s)" % legend
            outputlegend  = "-(%s)" % legend
        elif operation == "smooth":
            xplot =  x * 1
            yplot = self.simpleMath.smooth(y)
            sel['SourceName'] = legend
            sel['Key']    = ""
            sel['legend'] = "%s Smooth" % legend
            outputlegend  = "%s Smooth" % legend
            if 'operations' in dataObject.info:
                if len(dataObject.info['operations']):
                    if dataObject.info['operations'][-1] == "smooth":
                        sel['legend'] = legend
                        outputlegend  = legend
        elif operation == "forceymintozero":
            xplot =  x * 1
            yplot =  y - min(y)
            sel['SourceName'] = legend
            sel['Key']    = ""
            sel['legend'] = "(%s) - ymin" % legend
            outputlegend  = "(%s) - ymin" % legend
        elif operation == "fit":
            #remove a existing fit if present
            xmin,xmax=self.getGraphXLimits()
            outputlegend = legend + " Fit"
            for key in self._curveList:
                if key == outputlegend:
                    self.removeCurves([outputlegend], replot=False)
                    break
            self.scanFit.setData(x = x,
                                 y = y,
                                 xmin = xmin,
                                 xmax = xmax,
                                 legend = legend)
            if self.scanFit.isHidden():
                self.scanFit.show()
            self.scanFit.raise_()
        elif operation == "custom_fit":
            #remove a existing fit if present
            xmin, xmax=self.getGraphXLimits()
            outputlegend = legend + "Custom Fit"
            keyList = list(self._curveList)
            for key in keyList:
                if key == outputlegend:
                    self.removeCurves([outputlegend], replot=False)
                    break
            self.customFit.setData(x = x,
                                   y = y,
                                   xmin = xmin,
                                   xmax = xmax,
                                   legend = legend)
            if self.customFit.isHidden():
                self.customFit.show()
            self.customFit.raise_()
        else:
            raise ValueError("Unknown operation %s" % operation)
        if operation not in ["fit", "custom_fit"]:
            newDataObject.x = [xplot]
            newDataObject.y = [yplot]
            newDataObject.m = [numpy.ones(len(yplot)).astype(numpy.float)]

        #and add it to the plot
        if True and (operation not in ['fit', 'custom_fit']):
            sel['dataobject'] = newDataObject
            sel['scanselection'] = True
            sel['selection'] = copy.deepcopy(dataObject.info['selection'])
            sel['selectiontype'] = "1D"
            if operation == 'average':
                self._replaceSelection([sel])
            elif operation != 'fit':
                self._addSelection([sel])
            else:
                self.__fitDataObject = newDataObject
                return
        else:
            newDataObject.info['legend'] = outputlegend
            if operation == 'fit':
                self.__fitDataObject = newDataObject
                return
            if operation == 'custom_fit':
                self.__customFitDataObject = newDataObject
                return

            self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
            #here I should check the log or linear status
            self.addCurve(x=xplot, y=yplot, legend=newDataObject.info['legend'], replot=False)
        self.replot()

    def graphicsSave(self, filename):
        #use the plugin interface
        x, y, legend, info = self.getActiveCurve()
        size = (6, 3) #in inches
        bw = False
        if len(self.graph.curves.keys()) > 1:
            legends = True
        else:
            legends = False
        if self.matplotlibDialog is None:
            self.matplotlibDialog = QPyMcaMatplotlibSave1D.\
                                    QPyMcaMatplotlibSaveDialog(size=size,
                                                        logx=self._logX,
                                                        logy=self._logY,
                                                        legends=legends,
                                                        bw = bw)
        mtplt = self.matplotlibDialog.plot
        mtplt.setParameters({'logy':self._logY,
                             'logx':self._logX,
                             'legends':legends,
                             'bw':bw})
        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits()
        mtplt.setLimits(xmin, xmax, ymin, ymax)

        legend0 = legend
        xdata = x
        ydata = y
        dataCounter = 1
        alias = "%c" % (96+dataCounter)
        mtplt.addDataToPlot( xdata, ydata, legend=legend0, alias=alias )
        curveList = self.getAllCurves()
        for curve in curveList:
            xdata, ydata, legend, info = curve
            if legend == legend0:
                continue
            dataCounter += 1
            alias = "%c" % (96+dataCounter)
            mtplt.addDataToPlot( xdata, ydata, legend=legend, alias=alias )

        if sys.version < '3.0':
            self.matplotlibDialog.setXLabel(qt.safe_str(self.graph.x1Label()))
            self.matplotlibDialog.setYLabel(qt.safe_str(self.graph.y1Label()))
        else:
            self.matplotlibDialog.setXLabel(self.graph.x1Label())
            self.matplotlibDialog.setYLabel(self.graph.y1Label())
        if legends:
            mtplt.plotLegends()
        ret = self.matplotlibDialog.exec_()
        if ret == qt.QDialog.Accepted:
            mtplt.saveFile(filename)
        return

    def _deriveIconSignal(self):
        if DEBUG:
            print("_deriveIconSignal")
        self.__QSimpleOperation('derivate')

    def _swapSignIconSignal(self):
        if DEBUG:
            print("_swapSignIconSignal")
        self.__QSimpleOperation('swapsign')

    def _yMinToZeroIconSignal(self):
        if DEBUG:
            print("_yMinToZeroIconSignal")
        self.__QSimpleOperation('forceymintozero')

    def _subtractIconSignal(self):
        if DEBUG:
            print("_subtractIconSignal")
        self.__QSimpleOperation('subtract')

    def _subtractOperation(self):
        #identical to twice the average with the negative active curve
        #get active curve
        legend = self.getActiveCurveLegend()
        if legend is None:
            return

        found = False
        for key in self.dataObjectsList:
            if key == legend:
                found = True
                break

        if found:
            dataObject = self.dataObjectsDict[legend]
        else:
            print("I should not be here")
            print("active curve =",legend)
            print("but legend list = ",self.dataObjectsList)
            return
        x = dataObject.x[0]
        y = dataObject.y[0]
        ilabel = dataObject.info['selection']['y'][0]
        ylabel = dataObject.info['LabelNames'][ilabel]
        if len(dataObject.info['selection']['x']):
            ilabel = dataObject.info['selection']['x'][0]
            xlabel = dataObject.info['LabelNames'][ilabel]
        else:
            xlabel = "Point Number"

        xActive = x
        yActive = y
        yActiveLegend = legend
        yActiveLabel  = ylabel
        xActiveLabel  = xlabel

        operation = "subtract"    
        sel_list = []
        i = 0
        ndata = 0
        keyList = list(self._curveList)
        for key in keyList:
            legend = ""
            x = [xActive]
            y = [-yActive]
            if DEBUG:
                print("key -> ", key)
            if key in self.dataObjectsDict.keys():
                x.append(self.dataObjectsDict[key].x[0]) #only the first X
                if len(self.dataObjectsDict[key].y) == 1:
                    y.append(self.dataObjectsDict[key].y[0])
                    ilabel = self.dataObjectsDict[key].info['selection']['y'][0]
                else:
                    sel_legend = self.dataObjectsDict[key].info['legend']
                    ilabel = self.dataObjectsDict[key].info['selection']['y'][0]
                    #I have to get the proper y associated to the legend
                    if sel_legend in key:
                        if key.index(sel_legend) == 0:
                            label = key[len(sel_legend):]
                            while (label.startswith(' ')):
                                label = label[1:]
                                if not len(label):
                                    break
                            if label in self.dataObjectsDict[key].info['LabelNames']:
                                ilabel = self.dataObjectsDict[key].info['LabelNames'].index(label)
                            if DEBUG:
                                print("LABEL = ", label)
                                print("ilabel = ", ilabel)
                    y.append(self.dataObjectsDict[key].y[ilabel])
                outputlegend = "(%s - %s)" %  (key, yActiveLegend)
                ndata += 1
                xplot, yplot = self.simpleMath.average(x, y)
                yplot *= 2
                #create the output data object
                newDataObject = DataObject.DataObject()
                newDataObject.data = None
                newDataObject.info.update(self.dataObjectsDict[key].info)
                if not ('operations' in newDataObject.info):
                    newDataObject.info['operations'] = []
                newDataObject.info['operations'].append(operation)
                newDataObject.info['LabelNames'][ilabel] = "(%s - %s)" %  \
                                        (newDataObject.info['LabelNames'][ilabel], yActiveLabel)
                newDataObject.x = [xplot]
                newDataObject.y = [yplot]
                newDataObject.m = None
                sel = {}
                sel['SourceType'] = "Operation"
                sel['SourceName'] = key
                sel['Key']    = ""
                sel['legend'] = outputlegend
                sel['dataobject'] = newDataObject
                sel['scanselection'] = True
                sel['selection'] = copy.deepcopy(dataObject.info['selection'])
                #sel['selection']['y'] = [ilabel]
                sel['selectiontype'] = "1D"
                sel_list.append(sel)
        if True:
            #The legend menu was not working with the next line
            #but if works if I add the list
            self._replaceSelection(sel_list)
        else:
            oldlist = list(self.dataObjectsDict.keys())
            self._addSelection(sel_list)
            self.removeCurves(oldlist)

    #The plugins interface
    def getGraphYLimits(self):
        #if the active curve is mapped to second axis
        #I should give the second axis limits
        return super(McaWindow, self).getGraphYLimits()

    #end of plugins interface
    def addCurve(self, x, y, legend=None, info=None, **kw):
        #administrate the colors properly
        if legend in self._curveList:
            if info is None:
                info = {}
            oldStuff = self.getCurve(legend)
            if len(oldStuff):
                oldX, oldY, oldLegend, oldInfo = oldStuff
            else:
                oldInfo = {}
            color = info.get("plot_color", oldInfo.get("plot_color", None))
            symbol =  info.get("plot_symbol",oldInfo.get("plot_symbol", None))
            line_style =  info.get("plot_line_style",oldInfo.get("plot_line_style", None))
            info['plot_color'] = color
            info['plot_symbol'] = symbol
            info['plot_line_style'] = line_style
        if legend in self.dataObjectsDict:
            # the info is changing
            super(McaWindow, self).addCurve(x, y, legend=legend, info=info, **kw)
        else:
            # create the data object (Is this necessary????)
            self.newCurve(x, y, legend=legend, info=info, **kw)
    
    def newCurve(self, x, y, legend=None, xlabel=None, ylabel=None,
                 replace=False, replot=True, info=None, **kw):
        print("DATA OBJECT CREATION TO BE IMPLEMENTED FOR MCAs")
        return
        if legend is None:
            legend = "Unnamed curve 1.1"
        if xlabel is None:
            xlabel = "X"
        if ylabel is None:
            ylabel = "Y"
        if info is None:
            info = {}
        newDataObject = DataObject.DataObject()
        newDataObject.x = [x]
        newDataObject.y = [y]
        newDataObject.m = None
        newDataObject.info = copy.deepcopy(info)
        newDataObject.info['legend'] = legend
        newDataObject.info['SourceName'] = legend
        newDataObject.info['Key'] = ""
        newDataObject.info['selectiontype'] = "1D"
        newDataObject.info['LabelNames'] = [xlabel, ylabel]
        newDataObject.info['selection'] = {'x':[0], 'y':[1]}
        sel_list = []
        sel = {}
        sel['SourceType'] = "Operation"
        sel['SourceName'] = legend
        sel['Key']    = ""
        sel['legend'] = legend
        sel['dataobject'] = newDataObject
        sel['scanselection'] = False
        sel['selection'] = {'x':[0], 'y':[1], 'm':[], 'cntlist':[xlabel, ylabel]}
        #sel['selection']['y'] = [ilabel]
        sel['selectiontype'] = "1D"
        sel_list.append(sel)
        if replace:
            self._replaceSelection(sel_list)
        else:
            self._addSelection(sel_list, replot=replot)

    def printGraph(self):
        #temporary print
        pixmap = qt.QPixmap.grabWidget(self.centralWidget())

        if self.scanWindowInfoWidget is not None:
            info = self.scanWindowInfoWidget.getInfo()
            title = info['scan'].get('source', None)
            comment = info['scan'].get('scan', None)+"\n"
            h, k, l = info['scan'].get('hkl')
            if h != "----":
                comment += "H = %s  K = %s  L = %s\n" % (h, k, l)
            peak   = info['graph']['peak']
            peakAt = info['graph']['peakat']
            fwhm   = info['graph']['fwhm']
            fwhmAt = info['graph']['fwhmat']
            com    = info['graph']['com']
            mean   = info['graph']['mean']
            std    = info['graph']['std']
            minimum = info['graph']['min']
            maximum = info['graph']['max']
            delta   = info['graph']['delta']
            xLabel = self.graph.x1Label()
            comment += "Peak %s at %s = %s\n" % (peak, xLabel, peakAt)
            comment += "FWHM %s at %s = %s\n" % (fwhm, xLabel, fwhmAt)
            comment += "COM = %s  Mean = %s  STD = %s\n" % (com, mean, std)
            comment += "Min = %s  Max = %s  Delta = %s\n" % (minimum,
                                                            maximum,
                                                            delta)           
        else:
            title = None
            comment = None
        if not self.scanFit.isHidden():
            if comment is None:
                comment = ""
            comment += "\n"
            comment += self.scanFit.getText()
            
        self.printPreview.addPixmap(pixmap,
                                    title=title,
                                    comment=comment,
                                    commentposition="LEFT")
        if self.printPreview.isHidden():
            self.printPreview.show()        
        self.printPreview.raise_()            

    def refresh(self):
        activeCurve = self.getActiveCurve(just_legend=True)
        sellist = []
        for key in self.dataObjectsDict.keys():
            sel ={}
            sel['SourceName'] = self.dataObjectsDict[key].info['SourceName']
            sel['dataobject'] = self.dataObjectsDict[key]
            sel['legend'] = key
            sel['Key'] = self.dataObjectsDict[key].info['Key']
            sellist.append(sel)
        self.clearCurves()
        self._addSelection(sellist)
        if activeCurve is not None:
            self.setActiveCurve(activeCurve)
        self.replot()
        
def test():
    w = McaWindow()
    x = numpy.arange(1000.)
    y =  10 * x + 10000. * numpy.exp(-0.5*(x-500)*(x-500)/400)
    w.addCurve(x, y, legend="dummy", replot=True, replace=True)
    w.resetZoom()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
