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
import time
import traceback
from PyMca import PyMcaQt as qt
if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str
if __name__ == "__main__":
    app = qt.QApplication([])

from PyMca.widgets import PlotWindow
from PyMca import ScanFit
from PyMca import SimpleMath
from PyMca import DataObject
import copy
from PyMca import PyMcaPrintPreview
from PyMca import PyMcaDirs
from PyMca import ScanWindowInfoWidget
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
from PyMca import PyMcaPlugins

DEBUG = 0
class ScanWindow(PlotWindow.PlotWindow):
    def __init__(self, parent=None, name="Scan Window", specfit=None):
        super(ScanWindow, self).__init__(parent,
                                         newplot=True,
                                         plugins=True)
        # this two objects are the same
        self.dataObjectsList = self._curveList
        # but this is tricky
        self.dataObjectsDict = {}

        
        self.setWindowTitle(name)
        self.matplotlibDialog = None
        pluginDir = [os.path.dirname(os.path.abspath(PyMcaPlugins.__file__))]
        #self.setPluginDirectoryList(pluginDir)
        self.getPlugins(method="getPlugin1DInstance",
                        directoryList=pluginDir)
        """
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContenstsMargin(0)
        self.mainLayout.setSpacing(0)
        self.scanWindowInfoWidget = ScanWindowInfoWidget.\
                                        ScanWindowInfoWidget(self)
        self.mainLayout.addWidget(self.scanWindowInfoWidget)
        """
        self.scanWindowInfoWidget = None
        #self.fig = None
        self.scanFit = ScanFit.ScanFit(specfit=specfit)
        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)
        self.simpleMath = SimpleMath.SimpleMath()
        #self.graph.canvas().setMouseTracking(1)
        #self.graph.setCanvasBackground(qt.Qt.white)
        self.outputDir = None
        self.outputFilter = None

        #signals
        self.setCallback(self._graphSignalReceived)
        if 1:
            self.customFit = SimpleFitGUI.SimpleFitGUI()
            #self.connect(self.graph,
            #             qt.SIGNAL("QtBlissGraphSignal"),
            #             self._graphSignalReceived)
            self.connect(self.scanFit,
                         qt.SIGNAL('ScanFitSignal') ,
                         self._scanFitSignalReceived)
            self.connect(self.customFit,
                         qt.SIGNAL('SimpleFitSignal') ,
                         self._customFitSignalReceived)
        self.dataObjectsDict = {}
        self.dataObjectsList = []

        if 1:
            #fit icon
            self.fitButton = self._addToolButton(self.fitIcon,
                                 self._fitIconMenu,
                                 'Simple Fit of Active Curve')
            self.fitButtonMenu = qt.QMenu()
            self.fitButtonMenu.addAction(QString("Simple Fit"),
                                   self._fitIconSignal)
            self.fitButtonMenu.addAction(QString("Customized Fit") ,
                                   self._customFitSignal)

    def _pluginClicked(self):
        actionList = []
        menu = qt.QMenu(self)
        text = QString("Reload Plugins")
        menu.addAction(text)
        actionList.append(text)
        text = QString("Set User Plugin Directory")
        menu.addAction(text)
        actionList.append(text)
        global DEBUG
        if DEBUG:
            text = QString("Toggle DEBUG mode OFF")
        else:
            text = QString("Toggle DEBUG mode ON")
        menu.addAction(text)
        actionList.append(text)
        menu.addSeparator()
        callableKeys = ["Dummy0", "Dummy1", "Dummy2"]
        for m in self.pluginList:
            if m == "PyMcaPlugins.Plugin1DBase":
                continue
            module = sys.modules[m]
            if hasattr(module, 'MENU_TEXT'):
                text = QString(module.MENU_TEXT)
            else:
                text = os.path.basename(module.__file__)
                if text.endswith('.pyc'):
                    text = text[:-4]
                elif text.endswith('.py'):
                    text = text[:-3]
                text = QString(text)
            methods = self.pluginInstanceDict[m].getMethods(plottype="SCAN") 
            if not len(methods):
                continue
            menu.addAction(text)
            actionList.append(text)
            callableKeys.append(m)
        a = menu.exec_(qt.QCursor.pos())
        if a is None:
            return None
        idx = actionList.index(a.text())
        if idx == 0:
            n = self.getPlugins()
            if n < 1:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                msg.setText("Problem loading plugins")
                msg.exec_()
            return
        if idx == 1:
            dirName = qt.safe_str(qt.QFileDialog.getExistingDirectory(self,
                                "Enter user plugins directory",
                                os.getcwd()))
            if len(dirName):
                pluginsDir = self.getPluginDirectoryList()
                pluginsDirList = [pluginsDir[0], dirName]
                self.setPluginDirectoryList(pluginsDirList)
            return
        if idx == 2:
            if DEBUG:
                DEBUG = 0
            else:
                DEBUG = 1
            Plot1DBase.DEBUG = DEBUG
            return
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods(plottype="SCAN")
        if len(methods) == 1:
            idx = 0
        else:
            actionList = []
            # allow the plugin designer to specify the order
            #methods.sort()
            menu = qt.QMenu(self)
            for method in methods:
                text = QString(method)
                pixmap = self.pluginInstanceDict[key].getMethodPixmap(method)
                tip = QString(self.pluginInstanceDict[key].getMethodToolTip(method))
                if pixmap is not None:
                    action = qt.QAction(qt.QIcon(qt.QPixmap(pixmap)), text, self)
                else:
                    action = qt.QAction(text, self)
                if tip is not None:
                    action.setToolTip(tip)
                menu.addAction(action)
                actionList.append((text, pixmap, tip, action))
            qt.QObject.connect(menu, qt.SIGNAL("hovered(QAction *)"), self._actionHovered)
            a = menu.exec_(qt.QCursor.pos())
            if a is None:
                return None
            idx = -1
            for action in actionList:
                if a.text() == action[0]:
                    idx = actionList.index(action)
        try:
            self.pluginInstanceDict[key].applyMethod(methods[idx])
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Plugin error")
            msg.setText("An error has occured while executing the plugin:")
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()

    def _actionHovered(self, action):
        tip = action.toolTip()
        if qt.safe_str(tip) != qt.safe_str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)

    def _buildGraph(self):
        self.graph = QtBlissGraph.QtBlissGraph(self, uselegendmenu=True,
                                               legendrename=True,
                                               usecrosscursor=True)
        self.graph.setPanningMode(True)
        self.mainLayout.addWidget(self.graph)

        self.graphBottom = qt.QWidget(self)
        self.graphBottomLayout = qt.QHBoxLayout(self.graphBottom)
        self.graphBottomLayout.setMargin(0)
        self.graphBottomLayout.setSpacing(0)
        
        label=qt.QLabel(self.graphBottom)
        label.setText('<b>X:</b>')
        self.graphBottomLayout.addWidget(label)

        self._xPos = qt.QLineEdit(self.graphBottom)
        self._xPos.setText('------')
        self._xPos.setReadOnly(1)
        self._xPos.setFixedWidth(self._xPos.fontMetrics().width('##############'))
        self.graphBottomLayout.addWidget(self._xPos)


        label=qt.QLabel(self.graphBottom)
        label.setText('<b>Y:</b>')
        self.graphBottomLayout.addWidget(label)

        self._yPos = qt.QLineEdit(self.graphBottom)
        self._yPos.setText('------')
        self._yPos.setReadOnly(1)
        self._yPos.setFixedWidth(self._yPos.fontMetrics().width('##############'))
        self.graphBottomLayout.addWidget(self._yPos)
        self.graphBottomLayout.addWidget(qt.HorizontalSpacer(self.graphBottom))
        self.mainLayout.addWidget(self.graphBottom)

    def setDispatcher(self, w):
        self.connect(w, qt.SIGNAL("addSelection"),
                         self._addSelection)
        self.connect(w, qt.SIGNAL("removeSelection"),
                         self._removeSelection)
        self.connect(w, qt.SIGNAL("replaceSelection"),
                         self._replaceSelection)
        
    def _addSelection(self, selectionlist, replot=True):
        if DEBUG:
            print("_addSelection(self, selectionlist)",selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            if not ("scanselection" in sel): continue
            if sel['scanselection'] == "MCA":
                continue
            if not sel["scanselection"]:continue
            if len(key.split(".")) > 2: continue
            dataObject = sel['dataobject']
            #only one-dimensional selections considered
            if dataObject.info["selectiontype"] != "1D": continue
            
            #there must be something to plot
            if not hasattr(dataObject, 'y'): continue                
            if not hasattr(dataObject, 'x'):
                ylen = len(dataObject.y[0]) 
                if ylen:
                    xdata = numpy.arange(ylen).astype(numpy.float)
                else:
                    #nothing to be plot
                    continue
            if dataObject.x is None:
                ylen = len(dataObject.y[0]) 
                if ylen:
                    xdata = numpy.arange(ylen).astype(numpy.float)
                else:
                    #nothing to be plot
                    continue                    
            elif len(dataObject.x) > 1:
                if DEBUG:
                    print("Mesh plots")
                continue
            else:
                xdata = dataObject.x[0]
            sps_source = False
            if 'SourceType' in sel:
                if sel['SourceType'] == 'SPS':
                    sps_source = True

            if sps_source:
                ycounter = -1
                dataObject.info['selection'] = copy.deepcopy(sel['selection'])
                for ydata in dataObject.y:
                    ycounter += 1
                    if dataObject.m is None:
                        mdata = [numpy.ones(len(ydata)).astype(numpy.float)]
                    elif len(dataObject.m[0]) > 0:
                        if len(dataObject.m[0]) == len(ydata):
                            index = numpy.nonzero(dataObject.m[0])[0]
                            if not len(index):
                                continue
                            xdata = numpy.take(xdata, index)
                            ydata = numpy.take(ydata, index)
                            mdata = numpy.take(dataObject.m[0], index)
                            #A priori the graph only knows about plots
                            ydata = ydata/mdata
                        else:
                            raise ValueError("Monitor data length different than counter data")
                    else:
                        mdata = [numpy.ones(len(ydata)).astype(numpy.float)]
                    ylegend = 'y%d' % ycounter
                    if sel['selection'] is not None:
                        if type(sel['selection']) == type({}):
                            if 'x' in sel['selection']:
                                #proper scan selection
                                ilabel = dataObject.info['selection']['y'][ycounter]
                                ylegend = dataObject.info['LabelNames'][ilabel]
                    newLegend = legend + " " + ylegend
                    self.dataObjectsDict[newLegend] = dataObject
                    self.addCurve(xdata, ydata, legend=newLegend, info=dataObject.info, replot=False)
                    if self.scanWindowInfoWidget is not None:
                        activeLegend = self.getActiveCurve(just_legend=True)
                        if activeLegend is not None:
                            if activeLegend == newLegend:
                                self.scanWindowInfoWidget.updateFromDataObject\
                                                            (dataObject)
                        else:
                            dummyDataObject = DataObject.DataObject()
                            dummyDataObject.y=[numpy.array([])]
                            dummyDataObject.x=[numpy.array([])]
                            self.scanWindowInfoWidget.updateFromDataObject(dummyDataObject)                            
            else:
                #we have to loop for all y values
                ycounter = -1
                for ydata in dataObject.y:
                    ylen = len(ydata)
                    if ylen == 1:
                        if len(xdata) > 1:
                            ydata = ydata[0] * numpy.ones(len(xdata)).astype(numpy.float)
                    elif len(xdata) == 1:
                        xdata = xdata[0] * numpy.ones(ylen).astype(numpy.float)
                    ycounter += 1
                    newDataObject   = DataObject.DataObject()
                    newDataObject.info = copy.deepcopy(dataObject.info)
                    if dataObject.m is None:
                        mdata = numpy.ones(len(ydata)).astype(numpy.float)
                    elif len(dataObject.m[0]) > 0:
                        if len(dataObject.m[0]) == len(ydata):
                            index = numpy.nonzero(dataObject.m[0])[0]
                            if not len(index):
                                continue
                            xdata = numpy.take(xdata, index)
                            ydata = numpy.take(ydata, index)
                            mdata = numpy.take(dataObject.m[0], index)
                            #A priori the graph only knows about plots
                            ydata = ydata/mdata
                        elif len(dataObject.m[0]) == 1:
                            mdata = numpy.ones(len(ydata)).astype(numpy.float)
                            mdata *= dataObject.m[0][0]
                            index = numpy.nonzero(dataObject.m[0])[0]
                            if not len(index):
                                continue
                            xdata = numpy.take(xdata, index)
                            ydata = numpy.take(ydata, index)
                            mdata = numpy.take(dataObject.m[0], index)
                            #A priori the graph only knows about plots
                            ydata = ydata/mdata
                        else:
                            raise ValueError("Monitor data length different than counter data")
                    else:
                        mdata = numpy.ones(len(ydata)).astype(numpy.float)
                    newDataObject.x = [xdata]
                    newDataObject.y = [ydata]
                    newDataObject.m = [mdata]
                    newDataObject.info['selection'] = copy.deepcopy(sel['selection'])
                    ylegend = 'y%d' % ycounter
                    if sel['selection'] is not None:
                        if type(sel['selection']) == type({}):
                            if 'x' in sel['selection']:
                                #proper scan selection
                                newDataObject.info['selection']['x'] = sel['selection']['x'] 
                                newDataObject.info['selection']['y'] = [sel['selection']['y'][ycounter]]
                                newDataObject.info['selection']['m'] = sel['selection']['m']
                                ilabel = newDataObject.info['selection']['y'][0]
                                ylegend = newDataObject.info['LabelNames'][ilabel]
                    if ('operations' in dataObject.info) and len(dataObject.y) == 1:
                        newDataObject.info['legend'] = legend
                        symbol = 'x'
                    else:
                        symbol=None
                        newDataObject.info['legend'] = legend + " " + ylegend
                        newDataObject.info['selectionlegend'] = legend
                    maptoy2 = False
                    if 'operations' in dataObject.info:
                        if dataObject.info['operations'][-1] == 'derivate':
                            maptoy2 = True
                        
                    self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
                    self.addCurve(xdata, ydata, legend=newDataObject.info['legend'],
                                    symbol=symbol,maptoy2=maptoy2, replot=False)
        self.dataObjectsList = self._curveList
        if replot:
            self.replot()

            
    def _removeSelection(self, selectionlist):
        if DEBUG:
            print("_removeSelection(self, selectionlist)",selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        removelist = []
        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            if not ("scanselection" in sel): continue
            if sel['scanselection'] == "MCA":
                continue
            if not sel["scanselection"]:continue
            if len(key.split(".")) > 2: continue

            legend = sel['legend'] #expected form sourcename + scan key
            if type(sel['selection']) == type({}):
                if 'y' in sel['selection']:
                    for lName in ['cntlist', 'LabelNames']:
                        if lName in sel['selection']:
                            for index in sel['selection']['y']:
                                removelist.append(legend +" "+\
                                                  sel['selection'][lName][index])

        if len(removelist):
            self.removeCurves(removelist)

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

        doit = 0
        for sel in sellist:
            if not ("scanselection" in sel): continue
            if sel['scanselection'] == "MCA":
                continue
            if not sel["scanselection"]:continue
            if len(sel["Key"].split(".")) > 2: continue
            dataObject = sel['dataobject']
            if dataObject.info["selectiontype"] == "1D":
                if hasattr(dataObject, 'y'):
                    doit = 1
                    break
        if not doit:
            return
        self.clearCurves()
        self.dataObjectsDict={}
        self.dataObjectsList=self._curveList
        self._addSelection(selectionlist)

    def _graphSignalReceived(self, ddict):
        if DEBUG:
            print("_graphSignalReceived", ddict)
        if ddict['event'] == "MouseAt":
            if ddict['xcurve'] is not None:
                if self.__toggleCounter == 0:
                    self._xPos.setText('%.7g' % ddict['x'])
                    self._yPos.setText('%.7g' % ddict['y'])
                elif ddict['distance'] < 20:
                    #print ddict['point'], ddict['distance'] 
                    self._xPos.setText('%.7g' % ddict['xcurve'])
                    self._yPos.setText('%.7g' % ddict['ycurve'])
                else:
                    self._xPos.setText('----')
                    self._yPos.setText('----')
            else:
                self._xPos.setText('%.7g' % ddict['x'])
                self._yPos.setText('%.7g' % ddict['y'])
            return
        if ddict['event'] == "SetActiveCurveEvent":
            legend = ddict["legend"]
            if legend is None:
                if len(self.dataObjectsList):
                    legend = self.dataObjectsList[0]
                else:
                    return
            if legend not in self.dataObjectsList:
                if DEBUG:
                    print("unknown legend %s" % legend)
                return
            
            #force the current x label to the appropriate value
            dataObject = self.dataObjectsDict[legend]
            ilabel = dataObject.info['selection']['y'][0]
            ylabel = dataObject.info['LabelNames'][ilabel]
            if len(dataObject.info['selection']['x']):
                ilabel = dataObject.info['selection']['x'][0]
                xlabel = dataObject.info['LabelNames'][ilabel]
            else:
                xlabel = "Point Number"
            if len(dataObject.info['selection']['m']):
                ilabel = dataObject.info['selection']['m'][0]
                ylabel += "/" + dataObject.info['LabelNames'][ilabel]
            self.setGraphYLabel(ylabel)
            self.setGraphXLabel(xlabel)
            if self.scanWindowInfoWidget is not None:
                self.scanWindowInfoWidget.updateFromDataObject\
                                                            (dataObject)
            return

        if ddict['event'] == "RemoveCurveEvent":
            legend = ddict['legend']
            self.removeCurves([legend])
            return
        
        if ddict['event'] == "RenameCurveEvent":
            legend = ddict['legend']
            newlegend = ddict['newlegend']
            if legend in self.dataObjectsDict:
                self.dataObjectsDict[newlegend]= copy.deepcopy(self.dataObjectsDict[legend])
                self.dataObjectsDict[newlegend].info['legend'] = newlegend
                self.dataObjectsList.append(newlegend)
                self.removeCurves([legend], replot=False)
                self.newCurve(self.dataObjectsDict[newlegend].x[0],
                              self.dataObjectsDict[newlegend].y[0],
                              legend=self.dataObjectsDict[newlegend].info['legend'])
            return

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
            
    def _fitIconMenu(self):
        self.fitButtonMenu.exec_(self.cursor().pos())        

    def _fitIconSignal(self):
        if DEBUG:
            print("_fitIconSignal")
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

    def getActiveCurveLegend(self):
        return super(ScanWindow,self).getActiveCurve(just_legend=True)

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
        if False:
            #The legend menu was not working with the next line
            #but if works if I add the list
            self._replaceSelection(sel_list)
        else:
            oldlist = list(self.dataObjectsDict.keys())
            self._addSelection(sel_list)
            self.removeCurves(oldlist)

    #The plugins interface
    def getActiveCurve(self, just_legend=False):
        #get active curve
        legend = self.getActiveCurveLegend()
        if legend is None:
            return None
        if just_legend:
            return legend

        found = False
        for key in self._curveList:
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
        info = copy.deepcopy(dataObject.info)
        info['xlabel'] = xlabel
        info['ylabel'] = ylabel
        return x, y, legend, info

    def getGraphYLimits(self):
        #if the active curve is mapped to second axis
        #I should give the second axis limits
        return super(ScanWindow, self).getGraphYLimits()

    def setGraphXTitle(self, title):
        print("DEPRECATED")
        return self.setGraphXLabel(title)

    def setGraphYTitle(self, title):
        print("DEPRECATED")
        return self.setGraphYLabel(title)

    def getGraphXTitle(self):
        print("getGraphXTitle DEPRECATED")
        return self.getGraphXLabel()

    def getGraphYTitle(self):
        print("getGraphYTitle DEPRECATED")
        return self.getGraphYLabel()

    #end of plugins interface
    def addCurve(self, x, y, legend=None, info=None, **kw):
        if legend in self.dataObjectsDict:
            super(ScanWindow, self).addCurve(x, y, legend=legend, info=info, **kw)
        else:
            # create the data object
            self.newCurve(x, y, legend=legend, info=info, **kw)
    
    def newCurve(self, x, y, legend=None, xlabel=None, ylabel=None,
                 replace=False, replot=True, info=None, **kw):
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
        sel['scanselection'] = True
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
        pixmap = qt.QPixmap.grabWidget(self)

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


def test():
    w = ScanWindow()
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
