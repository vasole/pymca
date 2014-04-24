#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
import os
import numpy
#from numpy import argsort, nonzero, take
import time
import traceback
from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str
if __name__ == "__main__":
    app = qt.QApplication([])

import copy

from PyMca5.PyMcaGui import IconDict
from . import ScanWindow
from . import McaCalibrationControlGUI
from PyMca5.PyMcaIO import ConfigDict
from PyMca5.PyMcaGui import McaAdvancedFit
from PyMca5.PyMcaCore import DataObject
from PyMca5.PyMcaGui.physics.xrf import McaCalWidget
from . import McaSimpleFit
from PyMca5.PyMcaMath.fitting import Specfit
from PyMca5.PyMcaMath.fitting import SpecfitFuns
from PyMca5.PyMcaGui import PyMcaPrintPreview
from PyMca5 import PyMcaDirs

from PyMca5.PyMcaGui import QPyMcaMatplotlibSave1D
MATPLOTLIB = True

#force understanding of utf-8 encoding
#otherways it cannot generate svg output
try:
    import encodings.utf_8
except:
    #not a big problem
    pass
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

        self._buildCalibrationControlWidget()
        self._toggleCounter = 2
        self._togglePointsSignal()
        self.changeGridLevel()
        self.connections()
        self.setGraphYLabel('Counts')

        if 1:
            self.fitButtonMenu = qt.QMenu()
            self.fitButtonMenu.addAction(QString("Simple"),
                                         self.mcaSimpleFitSignal)
            self.fitButtonMenu.addAction(QString("Advanced") ,
                                         self.mcaAdvancedFitSignal)
            #self.fitButtonMenu.addAction(QString("Simple Fit"),
            #                       self._simpleFitSignal)
            #self.fitButtonMenu.addAction(QString("Customized Fit") ,
            #                       self._customFitSignal)

    def _buildCalibrationControlWidget(self):
        widget = self.centralWidget()
        self.controlWidget = McaCalibrationControlGUI.McaCalibrationControlGUI(\
                                        widget)
        widget.layout().addWidget(self.controlWidget)
        self.controlWidget.sigMcaCalibrationControlGUISignal.connect(\
                            self.__anasignal)

    def connections(self):
        self.simplefit.sigMcaSimpleFitSignal.connect(self.__anasignal)
        self.advancedfit.sigMcaAdvancedFitSignal.connect(self.__anasignal)

    def mcaSimpleFitSignal(self):
        legend = self.getActiveCurve(just_legend=True)
        if legend is None:
           msg = qt.QMessageBox(self)
           msg.setIcon(qt.QMessageBox.Critical)
           msg.setText("Please Select an active curve")
           msg.setWindowTitle('MCA Window Simple Fit')
           msg.exec_()
           return
        x, y, info = self.getDataAndInfoFromLegend(legend)
        self.advancedfit.hide()
        self.simplefit.show()
        self.simplefit.setFocus()
        self.simplefit.raise_()
        if info is not None:
            xmin, xmax = self.getGraphXLimits()
            self.__simplefitcalmode = self.calibration
            curveinfo = info
            if self.calibration == 'None':
                calib = [0.0,1.0,0.0]
            else:
                if 'McaCalib' in curveinfo:
                    calib = curveinfo['McaCalib']
                else:
                    calib = [0.0, 1.0, 0.0]
            self.__simplefitcalibration = calib
            calibrationOrder = curveinfo.get('McaCalibOrder',2)
            if calibrationOrder == 'TOF':
                x = calib[2] + calib[0] / pow(x-calib[1],2)
            else:
                x = calib[0] + calib[1] * x + calib[2] * x * x
            self.simplefit.setdata(x=x,y=y,
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
            if self.specfit.fitconfig['McaMode']:
                self.simplefit.fit()
        else:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error. Trying to fit fitted data?")
                msg.setWindowTitle('MCA Window Simple Fit')
                msg.exec_()

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

    def getDataAndInfoFromLegend(self, legend):
        xdata    = None
        ydata    = None
        info = None
        if legend in self.dataObjectsDict.keys():
            info  = self.dataObjectsDict[legend].info
            xdata = self.dataObjectsDict[legend].x[0]
            ydata = self.dataObjectsDict[legend].y[0]
        else:
            info = None
            xdata    = None
            ydata    = None
        return xdata, ydata, info



    def mcaAdvancedFitSignal(self):
        legend = self.getActiveCurve(just_legend=True)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            msg.setWindowTitle('MCA Window')
            msg.exec_()
            return

        x, y, info = self.getDataAndInfoFromLegend(legend)
        curveinfo = self.getCurve(legend)[3]
        xmin,xmax = self.getGraphXLimits()
        if self.calibration == 'None':
            if 'McaCalibSource' in curveinfo:
                calib = curveinfo['McaCalibSource']
            else:
                calib = [0.0,1.0,0.0]
        else:
            calib = curveinfo['McaCalib']
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
        return
        
    def __anasignal(self,dict):
        if DEBUG:
            print("__anasignal called dict = ",dict)
            
        if dict['event'] == 'clicked':
            # A button has been cicked
            if   dict['button'] == 'Source':
                pass
            elif dict['button'] == 'Calibration':
                #legend,x,y = self.graph.getactivecurve()
                legend = self.getActiveCurve(just_legend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    msg.exec_()
                    return
                else:
                    x, y, info = self.getDataAndInfoFromLegend(legend)
                    if info is None:
                        return
                    ndict = {}
                    ndict[legend] = {'order':1,
                                     'A':0.0,
                                     'B':1.0,
                                     'C':0.0}
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
                                                             caldict=ndict)
                    #info,x,y = self.getinfodatafromlegend(legend)
                    #caldialog.graph.newCurve("fromlegend",x=x,y=y)
                    ret = caldialog.exec_()

                    if ret == qt.QDialog.Accepted:
                        self.caldict.update(caldialog.getDict())
                        item, text = self.controlWidget.calbox.getCurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.controlWidget.calbox.setOptions(options)
                        except:
                            pass
                        self.controlWidget.calbox.setCurrentIndex(item)
                        self.refresh()
                    del caldialog
            elif dict['button'] == 'CalibrationCopy':
                #legend,x,y = self.graph.getactivecurve()
                legend = self.getActiveCurve(just_legend=1)
                if legend is None:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please Select an active curve")
                    msg.exec_()
                    return
                else:
                    x, y, info = self.getDataAndInfoFromLegend(legend)
                    if info is None:
                        return
                    ndict=copy.deepcopy(self.caldict)
                    if 'McaCalib' in info:
                        if type(info['McaCalib'][0]) == type([]):
                            sourcecal = info['McaCalib'][0]
                        else:
                            sourcecal = info['McaCalib']
                    else:
                        sourcecal = [0.0,1.0,0.0]
                    for curve in self.getAllCurves(just_legend=True):
                        curveinfo = self.getCurve(curve)[3]
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
                        item, text = self.controlWidget.calbox.getCurrent()
                        options = []
                        for option in self.calboxoptions:
                            options.append(option)
                        for key in self.caldict.keys():
                            if key not in options:
                                options.append(key)
                        try:
                            self.controlWidget.calbox.setOptions(options)
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
                    self.controlWidget.calbox.setOptions(options)
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
                self.controlWidget.calbox.setOptions(options)
                #I only reset the graph scale after a fit, not on a matrix spectrum
                if dict['event'] == 'McaAdvancedFitFinished':
                    #get current limits
                    if self.calibration == 'None':
                        xmin, xmax =self.getGraphXLimits()
                        emin    = dict['result']['fittedpar'][0] + \
                                  dict['result']['fittedpar'][1] * xmin
                        emax    = dict['result']['fittedpar'][0] + \
                                  dict['result']['fittedpar'][1] * xmax
                    else:
                        emin,emax = self.getGraphXLimits()
                    ymin, ymax =self.getGraphYLimits()
                    self.controlWidget.calbox.setCurrentIndex(options.index(legend))
                    self.calibration = legend
                    self.controlWidget._calboxactivated(legend)
                    self.setGraphYLimits(ymin, ymax, replot=False)
                    if emin < emax:
                        self.setGraphXLimits(emin, emax, replot=True)
                    else:
                        self.setGraphXLimits(emax, emin, replot=True)
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
                if legend in self.getAllCurves(just_legend=True):
                    if mcamode:
                        if not ('baseline' in self.dataObjectsDict[legend].info):
                            self.removeCurve(legend)
                    else:
                        if 'baseline' in self.dataObjectsDict[legend].info:
                            self.removeCurve(legend)
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
                if self.peakmarker is not None:
                    self.removeMarker(self.peakmarker)
                self.insertXMarker(pos,
                                   label,
                                   label=label,
                                   color='pink',
                                   draggable=False)
                self.peakmarker = label
            else:
                if self.peakmarker is not None:
                    self.removeMarker(self.peakmarker)
                self.peakmarker = None
        elif dict['event'] == 'McaTableClicked':
            if self.peakmarker is not None:
                self.removeMarker(self.peakmarker)
            self.peakmarker = None
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
                self.removeMarker(self.peakmarker)
            self.peakmarker = None
            self.replot()
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
            print("Selection changed event not implemented any more")
        else:
            if DEBUG:
                print("Unknown or ignored event",dict['event'])


    def emitCurrentROISignal(self):
        if self.currentROI is None:
            return
        #I have to get the current calibration
        if self.getGraphXLabel().upper() != "CHANNEL":
            #I have to get the energy
            A = self.controlWidget.calinfo.caldict['']['A']
            B = self.controlWidget.calinfo.caldict['']['B']
            C = self.controlWidget.calinfo.caldict['']['C']
            order = self.controlWidget.calinfo.caldict['']['order']
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
        w.sigAddSelection.connect(self._addSelection)
        w.sigRemoveSelection.connect(self._removeSelection)
        w.sigReplaceSelection.connect(self._replaceSelection)

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
            #self.replot()
            self.resetZoom()
        self.updateLegends()

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

    def graphCallback(self, ddict):
        if DEBUG:
            print("McaWindow._graphCallback", ddict)
        if ddict['event'] in ['markerMoved', 'markerSelected']:
            return self._handleMarkerEvent(ddict)
        elif ddict['event'] in ["mouseMoved", "MouseAt"]:
            if self.calibration == self.calboxoptions[0]:
                self._xPos.setText('%.2f' % ddict['x'])
                self._yPos.setText('%.2f' % ddict['y'])
            else:
                self._xPos.setText('%.4f' % ddict['x'])
                self._yPos.setText('%.2f' % ddict['y'])
        elif ddict['event'] in ["curveClicked", "legendClicked"]:
            legend = ddict.get('legend', None)
            legend = ddict.get('label', legend)
            if legend is None:
                if len(self.dataObjectsList):
                    legend = self.dataObjectsList[0]
                else:
                    return
            self.setActiveCurve(legend)
        elif ddict['event'] == "renameCurveEvent":
            legend = ddict['legend']
            newlegend = ddict['newlegend']
            if legend in self.dataObjectsDict:
                self.dataObjectsDict[newlegend]= copy.deepcopy(\
                                    self.dataObjectsDict[legend])
                self.dataObjectsDict[newlegend].info['legend'] = newlegend
                self.removeCurve(legend)
                self.addCurve(self.dataObjectsDict[newlegend].x[0],
                              self.dataObjectsDict[newlegend].y[0],
                              legend=newlegend,
                              info=self.dataObjectsDict[newlegend].info['legend'])
                if legend in self.caldict:
                    self.caldict[newlegend] = copy.deepcopy(self.caldict[legend])
                del self.dataObjectsDict[legend]
            self.replot()
        else:
            super(McaWindow, self).graphCallback(ddict)
            return
        self.sigPlotSignal.emit(ddict)

    def setActiveCurve(self, legend=None):
        if legend is None:
            legend = self.getActiveCurve(just_legend=True)
        if legend is None:
            self.controlWidget.calinfo.AText.setText("?????")
            self.controlWidget.calinfo.BText.setText("?????")
            self.controlWidget.calinfo.CText.setText("?????")
            return
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
            print("Should not be here")
            return
        try:
            info = self.getCurve(legend)[3]
            calib = info['McaCalib']
            self.controlWidget.calinfo.setParameters({'A':calib[0],
                                                'B':calib[1],
                                                'C':calib[2]})
        except KeyError:
            self.controlWidget.calinfo.AText.setText("?????")
            self.controlWidget.calinfo.BText.setText("?????")
            self.controlWidget.calinfo.CText.setText("?????")
        super(McaWindow, self).setActiveCurve(legend)

        
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
        legend = self.getActiveCurve(just_legend=True)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            msg.setWindowTitle('MCA window')
            msg.exec_()
            return
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
        format_list = ['Specfile MCA  *.mca',
                       'Specfile Scan *.dat',
                       'Raw ASCII  *.txt',
                       '";"-separated CSV *.csv',
                       '","-separated CSV *.csv',
                       '"tab"-separated CSV *.csv',
                       'OMNIC CSV *.csv',
                       'Widget PNG *.png',
                       'Widget JPG *.jpg']
        if self.outputFilter is None:
            self.outputFilter = format_list[0]
        if MATPLOTLIB:
            format_list.append('Graphics PNG *.png')
            format_list.append('Graphics EPS *.eps')
            format_list.append('Graphics SVG *.svg')
            
        outfile.setFilters(format_list)
        outfile.selectFilter(self.outputFilter)
        outfile.setFileMode(outfile.AnyFile)
        outfile.setAcceptMode(outfile.AcceptSave)
        outfile.setDirectory(wdir)
        ret = outfile.exec_()
        if ret:
            self.outputFilter = qt.safe_str(outfile.selectedFilter())
            filterused = self.outputFilter.split()
            filetype  = filterused[1]
            extension = filterused[2]
            outdir=qt.safe_str(outfile.selectedFiles()[0])
            try:            
                self.outputDir  = os.path.dirname(outdir)
                PyMcaDirs.outputDir = os.path.dirname(outdir) 
            except:
                self.outputDir  = "."
            try:            
                outputFile = os.path.basename(outdir)
            except:
                outputFile  = outdir
            outfile.close()
            del outfile
        else:
            # pyflakes bug http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=666494
            outfile.close()
            del outfile
            return

        #get active curve
        x, y, info = self.getDataAndInfoFromLegend(legend)
        if info is None:
            return

        ndict = {}
        ndict[legend] = {'order':1,'A':0.0,'B':1.0,'C':0.0}
        if self.getGraphXLabel().upper() == "CHANNEL":
            if legend in self.caldict:
                ndict[legend].update(self.caldict[legend])
                if abs(ndict[legend]['C']) > 0.0:
                    ndict[legend]['order']  = 2    
            elif 'McaCalib' in info:
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
        else:
            #I have to get current plot energy
            A = self.controlWidget.calinfo.caldict['']['A']
            B = self.controlWidget.calinfo.caldict['']['B']
            C = self.controlWidget.calinfo.caldict['']['C']
            order = self.controlWidget.calinfo.caldict['']['order']
            ndict[legend] = {'order':order,'A':A,'B':B,'C':C}

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
        if not outputFile.endswith(extension[1:]):
            outputFile += extension[1:]
        specFile = os.path.join(self.outputDir, outputFile)
        try:
            os.remove(specFile)
        except:
            pass
        systemline = os.linesep
        os.linesep = '\n'
        if filterused[0].upper() == "WIDGET":
            fformat = specFile[-3:].upper()
            pixmap = qt.QPixmap.grabWidget(self.getWidgetHandle())
            if not pixmap.save(specFile, fformat):
                qt.QMessageBox.critical(self,
                        "Save Error",
                        "%s" % "I could not save the file\nwith the desired format")
            return

        if MATPLOTLIB:
            try:
                if specFile[-3:].upper() in ['EPS', 'PNG', 'SVG']:
                    self.graphicsSave(specFile)
                    return
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Graphics Saving Error: %s" % (sys.exc_info()[1]))
                msg.exec_()
                return

        try:
            ffile = open(specFile,'wb')
        except IOError:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Input Output Error: %s" % (sys.exc_info()[1]))
            msg.exec_()
            return
        systemline = os.linesep
        os.linesep = '\n'
        #This was giving problems on legends with a leading b
        #legend = legend.strip('<b>')
        #legend = legend.strip('<\b>')
        try:
            if filetype == 'Scan':
                ffile.write("#F %s\n" % specFile)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("\n")
                ffile.write("#S 1 %s\n" % legend)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("#N 3\n")
                ffile.write("#L channel  counts  energy\n")
                energy = ndict[legend]['A'] + ndict[legend]['B'] * x + ndict[legend]['C'] * x * x
                for i in range(len(y)):
                    ffile.write("%.7g  %.7g  %.7g\n" % (x[i], y[i], energy[i]))
                ffile.write("\n")
            elif filetype == 'ASCII':
                energy = ndict[legend]['A'] + \
                         ndict[legend]['B'] * x + \
                         ndict[legend]['C'] * x * x
                for i in range(len(y)):
                   ffile.write("%.7g  %.7g  %.7g\n" % (x[i], y[i], energy[i]))
            elif filetype == 'CSV':
                if "," in filterused[0]:
                    csv = ","
                elif ";" in filterused[0]:
                    csv = ";"
                elif "OMNIC" in filterused[0]:
                    csv = ","
                else:
                    csv = "\t"                
                energy = ndict[legend]['A'] + \
                         ndict[legend]['B'] * x + \
                         ndict[legend]['C'] * x * x
                if "OMNIC" in filterused[0]:
                    for i in range(len(y)):
                        ffile.write("%.7E%s%.7E\n" % \
                               (energy[i], csv, y[i]))
                else:
                    ffile.write('"channel"%s"counts"%s"energy"\n' % (csv, csv))
                    for i in range(len(y)):
                        ffile.write("%.7E%s%.7E%s%.7E\n" % \
                               (x[i], csv, y[i], csv, energy[i]))
            else:
                ffile.write("#F %s\n" % specFile)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("\n")
                ffile.write("#S 1 %s\n" % legend)
                ffile.write("#D %s\n"%(time.ctime(time.time())))
                ffile.write("#@MCA %16C\n")
                ffile.write("#@CHANN %d %d %d 1\n" %  (len(y), x[0], x[-1]))
                ffile.write("#@CALIB %.7g %.7g %.7g\n" % (ndict[legend]['A'],
                                                         ndict[legend]['B'],
                                                         ndict[legend]['C']))
                ffile.write(self.array2SpecMca(y))
                ffile.write("\n")
            ffile.close()
        except:
            os.linesep = systemline
            raise
        return
        
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
    app.lastWindowClosed.connect(app.quit)
    w.show()
    app.exec_()


if __name__ == "__main__":
    test()
