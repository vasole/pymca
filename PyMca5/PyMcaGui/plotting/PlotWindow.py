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
__doc__ = """
This window handles plugins and adds a toolbar to the PlotWidget.

Currently the only dependency on PyMca is through the Icons.

"""
import copy
import sys
import os
import traceback
import numpy
from numpy import argsort, nonzero, take
from . import LegendSelector
from .ObjectPrintConfigurationDialog import ObjectPrintConfigurationDialog
from . import McaROIWidget
from . import PlotWidget

if "PySide" in sys.argv:
    import PySide

# pyqtgraph has a SciPy dependency
PYQTGRAPH = False
if 'pyqtgraph' in sys.argv:
    from PyMca5.PyMcaGraph.backends import PyQtGraphBackend
    PYQTGRAPH = True

MATPLOTLIB = False
if ("matplotlib" in sys.argv) or (not PYQTGRAPH):
    from PyMca5.PyMcaGraph.backends import MatplotlibBackend
    MATPLOTLIB = True

from .PyMca_Icons import IconDict
from .. import PyMcaQt as qt

if hasattr(qt, 'QString'):
    QString = qt.QString
else:
    QString = qt.safe_str

DEBUG = 0

class PlotWindow(PlotWidget.PlotWidget):
    sigROISignal = qt.pyqtSignal(object)
    
    def __init__(self, parent=None, backend=None, plugins=True, newplot=False,
                 control=False, position=False, **kw):
        if backend is None:
            if MATPLOTLIB:
                backend = MatplotlibBackend.MatplotlibBackend
            elif PYQTGRAPH:
                backend = PyQtGraphBackend.PyQtGraphBackend
        super(PlotWindow, self).__init__(parent=parent, backend=backend)
        self.pluginsIconFlag = plugins
        self.newplotIconsFlag = newplot
        self.setWindowType(None)      # None, "SCAN", "MCA"
        self._initIcons()
        self._buildToolBar(kw)
        self.setIconSize(qt.QSize(16, 16))
        self._toggleCounter = 0
        self._keepDataAspectRatioFlag = False
        self.gridLevel = 0
        self.legendWidget = None
        self.setCallback(self.graphCallback)
        if control or position:
            self._buildGraphBottomWidget(control, position)
            self._controlMenu = None

        # default print configuration (uses full page)
        self._printMenu = None
        self._printConfigurationDialog = None
        self._printConfiguration = {"xOffset": 0.1,
                                    "yOffset": 0.1,
                                    "width": 0.9,
                                    "height": 0.9,
                                    "units": "page",
                                    "keepAspectRatio": True}


        # activeCurve handling
        self.enableActiveCurveHandling(True)
        self.setActiveCurveColor('black')

        # default ROI handling
        self.roiWidget = None
        self._middleROIMarkerFlag = False
        
    def _buildGraphBottomWidget(self, control, position):
        widget = self.centralWidget()
        self.graphBottom = qt.QWidget(widget)
        self.graphBottomLayout = qt.QHBoxLayout(self.graphBottom)
        self.graphBottomLayout.setContentsMargins(0, 0, 0, 0)
        self.graphBottomLayout.setSpacing(0)

        if control:
            self.graphControlButton = qt.QPushButton(self.graphBottom)
            self.graphControlButton.setText("Options")
            self.graphControlButton.setAutoDefault(False)
            self.graphBottomLayout.addWidget(self.graphControlButton)
            self.graphControlButton.clicked.connect(self._graphControlClicked)

        if position:
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
        widget.layout().addWidget(self.graphBottom)

    def setPrintMenu(self, menu):
        self._printMenu = menu
        
    def setWindowType(self, wtype=None):
        if wtype not in [None, "SCAN", "MCA"]:
            print("Unsupported window type. Default to None")
        self._plotType = wtype

    def _graphControlClicked(self):
        if self._controlMenu is None:
            #create a default menu
            controlMenu = qt.QMenu()
            controlMenu.addAction(QString("Show/Hide Legends"),
                                       self.toggleLegendWidget)
            controlMenu.exec_(self.cursor().pos())
        else:
            self._controlMenu.exec_(self.cursor().pos())

    def setControlMenu(self, menu=None):
        self._controlMenu = menu

    def _initIcons(self):
        self.normalIcon	= qt.QIcon(qt.QPixmap(IconDict["normal"]))
        self.zoomIcon	= qt.QIcon(qt.QPixmap(IconDict["zoom"]))
        self.roiIcon	= qt.QIcon(qt.QPixmap(IconDict["roi"]))
        self.peakIcon	= qt.QIcon(qt.QPixmap(IconDict["peak"]))
        self.energyIcon = qt.QIcon(qt.QPixmap(IconDict["energy"]))

        self.zoomResetIcon	= qt.QIcon(qt.QPixmap(IconDict["zoomreset"]))
        self.roiResetIcon	= qt.QIcon(qt.QPixmap(IconDict["roireset"]))
        self.peakResetIcon	= qt.QIcon(qt.QPixmap(IconDict["peakreset"]))
        self.refreshIcon	= qt.QIcon(qt.QPixmap(IconDict["reload"]))

        self.logxIcon	= qt.QIcon(qt.QPixmap(IconDict["logx"]))
        self.logyIcon	= qt.QIcon(qt.QPixmap(IconDict["logy"]))
        self.xAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["xauto"]))
        self.yAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["yauto"]))
        self.gridIcon	= qt.QIcon(qt.QPixmap(IconDict["grid16"]))
        self.hFlipIcon	= qt.QIcon(qt.QPixmap(IconDict["gioconda16mirror"]))
        self.togglePointsIcon = qt.QIcon(qt.QPixmap(IconDict["togglepoints"]))

        self.solidCircleIcon = qt.QIcon(qt.QPixmap(IconDict["solidcircle"]))
        self.solidEllipseIcon = qt.QIcon(qt.QPixmap(IconDict["solidellipse"]))

        self.fitIcon	= qt.QIcon(qt.QPixmap(IconDict["fit"]))
        self.searchIcon	= qt.QIcon(qt.QPixmap(IconDict["peaksearch"]))

        self.averageIcon	= qt.QIcon(qt.QPixmap(IconDict["average16"]))
        self.deriveIcon	= qt.QIcon(qt.QPixmap(IconDict["derive"]))
        self.smoothIcon     = qt.QIcon(qt.QPixmap(IconDict["smooth"]))
        self.swapSignIcon	= qt.QIcon(qt.QPixmap(IconDict["swapsign"]))
        self.yMinToZeroIcon	= qt.QIcon(qt.QPixmap(IconDict["ymintozero"]))
        self.subtractIcon	= qt.QIcon(qt.QPixmap(IconDict["subtract"]))
        
        self.printIcon	= qt.QIcon(qt.QPixmap(IconDict["fileprint"]))
        self.saveIcon	= qt.QIcon(qt.QPixmap(IconDict["filesave"]))            

        self.pluginIcon     = qt.QIcon(qt.QPixmap(IconDict["plugin"])) 

    def _buildToolBar(self, kw=None):
        if kw is None:
            kw = {}
        self.toolBar = qt.QToolBar(self)
        #Autoscale
        self._addToolButton(self.zoomResetIcon,
                            self._zoomReset,
                            'Auto-Scale the Graph')

        #y Autoscale
        self.yAutoScaleButton = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True)
        self.yAutoScaleButton.setChecked(True)
        self.yAutoScaleButton.setDown(True)


        #x Autoscale
        self.xAutoScaleButton = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True)
        self.xAutoScaleButton.setChecked(True)
        self.xAutoScaleButton.setDown(True)

        #y Logarithmic
        if kw.get('logy', True):
            self.yLogButton = self._addToolButton(self.logyIcon,
                                self._toggleLogY,
                                'Toggle Logarithmic Y Axis (On/Off)',
                                toggle = True)
            self.yLogButton.setChecked(False)
            self.yLogButton.setDown(False)

        #x Logarithmic
        if kw.get('logx', True):
            self.xLogButton = self._addToolButton(self.logxIcon,
                                self._toggleLogX,
                                'Toggle Logarithmic X Axis (On/Off)',
                                toggle = True)
            self.xLogButton.setChecked(False)
            self.xLogButton.setDown(False)

        #Aspect ratio
        if kw.get('aspect', False):
            self.aspectButton = self._addToolButton(self.solidCircleIcon,
                                self._aspectButtonSignal,
                                'Keep data aspect ratio',
                                toggle = False)
            self.aspectButton.setChecked(False)
            #self.aspectButton.setDown(False)
        #colormap
        if kw.get('colormap', False):
            tb = self._addToolButton(self.colormapIcon,
                                     self._colormapIconSignal,
                                    'Change Colormap')
            self.colormapToolButton = tb

        #flip
        if kw.get('flip', False):
            tb = self._addToolButton(self.hFlipIcon,
                                 self._hFlipIconSignal,
                                 'Flip Horizontal')
            self.hFlipToolButton = tb

        #grid
        if kw.get('grid', True):
            tb = self._addToolButton(self.gridIcon,
                                self.changeGridLevel,
                                'Change Grid',
                                toggle = False)
            self.gridTb = tb


        #toggle Points/Lines
        tb = self._addToolButton(self.togglePointsIcon,
                             self._togglePointsSignal,
                             'Toggle Points/Lines')

        #energy icon
        if kw.get('energy', False):
            self._addToolButton(self.energyIcon,
                            self._energyIconSignal,
                            'Toggle Energy Axis (On/Off)',
                            toggle=True)            

        #roi icon
        if kw.get('roi', False):
            self.roiButton = self._addToolButton(self.roiIcon,
                                         self._toggleROI,
                                         'Show/Hide ROI widget',
                                         toggle=False)
            self.currentROI = None
            self.middleROIMarkerFlag = False

        #fit icon
        if kw.get('fit', False):
            self.fitButton = self._addToolButton(self.fitIcon,
                                         self._fitIconSignal,
                                         'Fit of Active Curve')

        if self.newplotIconsFlag:
            tb = self._addToolButton(self.averageIcon,
                                self._averageIconSignal,
                                 'Average Plotted Curves')

            tb = self._addToolButton(self.deriveIcon,
                                self._deriveIconSignal,
                                 'Take Derivative of Active Curve')

            tb = self._addToolButton(self.smoothIcon,
                                 self._smoothIconSignal,
                                 'Smooth Active Curve')

            tb = self._addToolButton(self.swapSignIcon,
                                self._swapSignIconSignal,
                                'Multiply Active Curve by -1')

            tb = self._addToolButton(self.yMinToZeroIcon,
                                self._yMinToZeroIconSignal,
                                'Force Y Minimum to be Zero')

            tb = self._addToolButton(self.subtractIcon,
                                self._subtractIconSignal,
                                'Subtract Active Curve')
        #save
        infotext = 'Save Active Curve or Widget'
        tb = self._addToolButton(self.saveIcon,
                                 self._saveIconSignal,
                                 infotext)

        if self.pluginsIconFlag:
            infotext = "Call/Load 1D Plugins"
            tb = self._addToolButton(self.pluginIcon,
                                 self._pluginClicked,
                                 infotext)

        self.toolBar.addWidget(qt.HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self._printGraph,
                                 'Prints the Graph')

        self.addToolBar(self.toolBar)

    def _printGraph(self):
        if self._printMenu is None:
            printMenu = qt.QMenu()
            #printMenu.addAction(QString("Select printer"),
            #                        self._printerSelect)
            printMenu.addAction(QString("Customize printing"),
                            self._getPrintConfigurationFromDialog)
            printMenu.addAction(QString("Print"),
                                       self.printGraph)
            printMenu.exec_(self.cursor().pos())
        else:
            self._printMenu.exec_(self.cursor().pos())

    def printGraph(self, *var, **kw):
        config = self.getPrintConfiguration()
        PlotWidget.PlotWidget.printGraph(self,
                            width=config['width'],
                            height=config['height'],
                            xOffset=config['xOffset'],
                            yOffset=config['yOffset'],
                            units=config['units'],
                            keepAspectRatio=config['keepAspectRatio'],
                            printer=self._printer)

    def setPrintConfiguration(self, configuration, printer=None):
        for key in self._printConfiguration:
            if key in configuration:
                self._printConfiguration[key] = configuration[key]
        if printer is not None:
            # printer should be a global thing ...
            self._printer = printer

    def getPrintConfiguration(self, dialog=False):
        if dialog:
            self._getPrintConfigurationFromDialog()
        return copy.deepcopy(self._printConfiguration)


    def _getPrintConfigurationFromDialog(self):
        if self._printConfigurationDialog is None:
            self._printConfigurationDialog = \
                                ObjectPrintConfigurationDialog(self)
        oldConfig = self.getPrintConfiguration()
        self._printConfigurationDialog.setPrintConfiguration(oldConfig,
                                                    printer=self._printer)
        if self._printConfigurationDialog.exec_():
            self.setPrintConfiguration(\
                self._printConfigurationDialog.getPrintConfiguration())

    def _addToolButton(self, icon, action, tip, toggle=None):
        tb      = qt.QToolButton(self.toolBar)            
        tb.setIcon(icon)
        tb.setToolTip(tip)
        if toggle is not None:
            if toggle:
                tb.setCheckable(1)
        self.toolBar.addWidget(tb)
        tb.clicked[()].connect(action)
        return tb

    def _aspectButtonSignal(self):
        if DEBUG:
            print("_aspectButtonSignal")
        if self._keepDataAspectRatioFlag:
            self.keepDataAspectRatio(False)
        else:
            self.keepDataAspectRatio(True)

    def keepDataAspectRatio(self, flag=True):
        if flag:
            self._keepDataAspectRatioFlag = True
            self.aspectButton.setIcon(self.solidEllipseIcon)
            self.aspectButton.setToolTip("Set free data aspect ratio")
        else:
            self._keepDataAspectRatioFlag = False
            self.aspectButton.setIcon(self.solidCircleIcon)
            self.aspectButton.setToolTip("Keep data aspect ratio")
        super(PlotWindow, self).keepDataAspectRatio(self._keepDataAspectRatioFlag)
                
    def _zoomReset(self):
        if DEBUG:
            print("_zoomReset")
        self.resetZoom()

    def _yAutoScaleToggle(self):
        if DEBUG:
            print("toggle Y auto scaling")
        if self.isYAxisAutoScale():
            self.setYAxisAutoScale(False)
            self.yAutoScaleButton.setDown(False)
            self.yAutoScaleButton.setChecked(False)
            ymin, ymax = self.getGraphYLimits()
            self.setGraphYLimits(ymin, ymax)
        else:
            self.setYAxisAutoScale(True)
            self.yAutoScaleButton.setDown(True)
            self.resetZoom()

    def _xAutoScaleToggle(self):
        if DEBUG:
            print("toggle X auto scaling")
        if self.isXAxisAutoScale():
            self.setXAxisAutoScale(False)
            self.xAutoScaleButton.setDown(False)
            self.xAutoScaleButton.setChecked(False)
            xmin, xmax = self.getGraphXLimits()
            self.setGraphXLimits(xmin, xmax)
        else:
            self.setXAxisAutoScale(True)
            self.xAutoScaleButton.setDown(True)
            self.resetZoom()
                       
    def _toggleLogX(self):
        if DEBUG:
            print("toggle logarithmic X scale")
        if self.isXAxisLogarithmic():
            self.setXAxisLogarithmic(False)
        else:
            self.setXAxisLogarithmic(True)

    def setXAxisLogarithmic(self, flag=True):
        super(PlotWindow, self).setXAxisLogarithmic(flag) 
        self.xLogButton.setChecked(flag)
        self.xLogButton.setDown(flag)
        self.replot()
        self.resetZoom()

    def _toggleLogY(self):
        if DEBUG:
            print("_toggleLogY")
        if self.isYAxisLogarithmic():
            self.setYAxisLogarithmic(False)
        else:
            self.setYAxisLogarithmic(True)

    def setYAxisLogarithmic(self, flag=True):
        super(PlotWindow, self).setYAxisLogarithmic(flag) 
        self.yLogButton.setChecked(flag)
        self.yLogButton.setDown(flag)
        self.replot()
        self.resetZoom()

    def _togglePointsSignal(self):
        if DEBUG:
            print("toggle points signal")
        self._toggleCounter = (self._toggleCounter + 1) % 3
        if self._toggleCounter == 1:
            self.setDefaultPlotLines(True)
            self.setDefaultPlotPoints(True)
        elif self._toggleCounter == 2:
            self.setDefaultPlotPoints(True)
            self.setDefaultPlotLines(False)
        else:
            self.setDefaultPlotLines(True)
            self.setDefaultPlotPoints(False)
        self.replot()

    def _hFlipIconSignal(self):
        if DEBUG:
            print("_hFlipIconSignal called")
        if self.isYAxisInverted():
            self.invertYAxis(False)
        else:
            self.invertYAxis(True)
        
    def _colormapIconSignal(self):
        if DEBUG:
            print("_colormapIconSignal called")

    def showRoiWidget(self, position=None):
        self._toggleROI(position)

    def _toggleROI(self, position=None):
        if DEBUG:
            print("_toggleROI called")
        if self.roiWidget is None:
            self.roiWidget = McaROIWidget.McaROIWidget()
            self.roiDockWidget = qt.QDockWidget(self)
            self.roiDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.roiDockWidget.setWidget(self.roiWidget)
            if position is None:
                w = self.centralWidget().width()
                h = self.centralWidget().height()
                if w > (1.25 * h):
                    self.addDockWidget(qt.Qt.RightDockWidgetArea,
                                       self.roiDockWidget)
                else:
                    self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                                       self.roiDockWidget)
            else:
                self.addDockWidget(position, self.roiDockWidget)
            if hasattr(self, "legendDockWidget"):
                self.tabifyDockWidget(self.legendDockWidget,
                                      self.roiDockWidget)
            self.roiWidget.sigMcaROIWidgetSignal.connect(self._roiSignal)
            self.roiDockWidget.setWindowTitle(self.windowTitle()+(" ROI"))
        if self.roiDockWidget.isHidden():
            self.roiDockWidget.show()
        else:
            self.roiDockWidget.hide()

    def changeGridLevel(self):
        self.gridLevel += 1
        #self.gridLevel = self.gridLevel % 3
        self.gridLevel = self.gridLevel % 2
        if self.gridLevel == 0:
            self.showGrid(False)
        elif self.gridLevel == 1:
            self.showGrid(1)
        elif self.gridLevel == 2:
            self.showGrid(2)
        self.replot()

    def _energyIconSignal(self):
        print("energy icon signal")

    def _fitIconSignal(self):
        print("fit icon signal")

    def _averageIconSignal(self):
        print("average icon signal")

    def _deriveIconSignal(self):
        print("deriveIconSignal")

    def _smoothIconSignal(self):
        print("smoothIconSignal")

    def _swapSignIconSignal(self):
        print("_swapSignIconSignal")

    def _yMinToZeroIconSignal(self):
        print("_yMinToZeroIconSignal")

    def _subtractIconSignal(self):
        print("_subtractIconSignal")

    def _saveIconSignal(self):
        print("_saveIconSignal")

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
            if m in ["PyMcaPlugins.Plugin1DBase", "Plugin1DBase"]:
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
            methods = self.pluginInstanceDict[m].getMethods(plottype=self._plotType) 
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
            return
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods(plottype=self._plotType)
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
            #qt.QObject.connect(menu, qt.SIGNAL("hovered(QAction *)"), self._actionHovered)
            menu.hovered.connect(self._actionHovered)
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
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)

    ########### ROI HANDLING ###############
    def graphCallback(self, ddict):
        if DEBUG:
            print("_graphSignalReceived", ddict)
        if ddict['event'] in ['markerMoved', 'markerSelected']:
            label = ddict['label'] 
            if label in ['ROI min', 'ROI max', 'ROI middle']:            
                self._handleROIMarkerEvent(ddict)
        if ddict['event'] in ["curveClicked", "legendClicked"]:
            legend = ddict["label"]
            self.setActiveCurve(legend)
        if ddict['event'] in ['mouseMoved']:
            if hasattr(self, "_xPos"):
                self._xPos.setText('%.7g' % ddict['x'])
                self._yPos.setText('%.7g' % ddict['y'])
        #make sure the signal is forwarded
        #super(PlotWindow, self).graphCallback(ddict)
        self.sigPlotSignal.emit(ddict)   

    def setActiveCurve(self, legend):
        PlotWidget.PlotWidget.setActiveCurve(self, legend)
        self.calculateROIs()
        self.updateLegends()
        
    def _handleROIMarkerEvent(self, ddict):
        if ddict['event'] == 'markerMoved':
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            if self.currentROI is None:
                return
            if self.currentROI not in roiDict:
                return
            x = ddict['x']
            label = ddict['label'] 
            if label == 'ROI min':
                roiDict[self.currentROI]['from'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +\
                                 roiDict[self.currentROI]['from'])
                    self.insertXMarker(pos,
                                      'ROI middle',
                                       label='',
                                       color='yellow',
                                       draggable=True)
            elif label == 'ROI max':
                roiDict[self.currentROI]['to'] = x
                if self._middleROIMarkerFlag:
                    pos = 0.5 * (roiDict[self.currentROI]['to'] +\
                                 roiDict[self.currentROI]['from'])
                    self.insertXMarker(pos,
                                      'ROI middle',
                                       label='',
                                       color='yellow',
                                       draggable=True)
            elif label == 'ROI middle':
                delta = x - 0.5 * (roiDict[self.currentROI]['from'] + \
                                   roiDict[self.currentROI]['to'])
                roiDict[self.currentROI]['from'] += delta
                roiDict[self.currentROI]['to'] += delta
                self.insertXMarker(roiDict[self.currentROI]['from'],
                                   'ROI min',
                                   label='ROI min',
                                   color='blue',
                                   draggable=True)
                self.insertXMarker(roiDict[self.currentROI]['to'],
                                   'ROI max',
                                   label='ROI max',
                                   color='blue',
                                   draggable=True)
            else:
                return
            self.calculateROIs(roiList, roiDict)
            self.emitCurrentROISignal()
        
    def _roiSignal(self, ddict):
        if ddict['event'] == "AddROI":
            xmin,xmax = self.getGraphXLimits()
            fromdata = xmin + 0.25 * (xmax - xmin)
            todata   = xmin + 0.75 * (xmax - xmin)
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if self._middleROIMarkerFlag:
                self.removeMarker('ROI middle')
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            nrois = len(roiList)
            if nrois == 0:
                newroi = "ICR"
                fromdata, dummy0, todata, dummy1 = self._getAllLimits()
                draggable = False
                color = 'black'
            else:
                for i in range(nrois):
                    i += 1
                    newroi = "newroi %d" % i
                    if newroi not in roiList:
                        break
                color = 'blue'
                draggable = True
            self.insertXMarker(fromdata, 'ROI min',
                               label='ROI min',
                               color=color,
                               draggable=draggable)
            self.insertXMarker(todata,'ROI max',
                               label='ROI max',
                               color=color,
                               draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.insertXMarker(pos, 'ROI middle',
                                   label="",
                                   color='yellow',
                                   draggable=draggable)                
            roiList.append(newroi)
            roiDict[newroi] = {}
            roiDict[newroi]['type'] = self.getGraphXLabel()
            roiDict[newroi]['from'] = fromdata
            roiDict[newroi]['to'] = todata
            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=newroi)
            self.currentROI = newroi
            self.calculateROIs()
        elif ddict['event'] in ['DelROI', "ResetROI"]:
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if self._middleROIMarkerFlag:
                self.removeMarker('ROI middle')
            roiList, roiDict = self.roiWidget.getROIListAndDict()
            currentroi = list(roiDict.keys())[0]
            self.roiWidget.fillFromROIDict(roilist=roiList,
                                           roidict=roiDict,
                                           currentroi=currentroi)
            self.currentROI = currentroi
        elif ddict['event'] == 'ActiveROI':
            print("ActiveROI event")
            pass
        elif ddict['event'] == 'selectionChanged':
            if DEBUG:
                print("Selection changed")
            self.roilist, self.roidict = self.roiWidget.getROIListAndDict()
            fromdata = ddict['roi']['from']
            todata   = ddict['roi']['to']
            self.removeMarker('ROI min')
            self.removeMarker('ROI max')
            if ddict['key'] == 'ICR':
                draggable = False
                color = 'black'
            else:
                draggable = True
                color = 'blue'
            self.insertXMarker(fromdata, label = 'ROI min',
                               color=color,
                               draggable=draggable)
            self.insertXMarker(todata, label = 'ROI max',
                               color=color,
                               draggable=draggable)
            if draggable and self._middleROIMarkerFlag:
                pos = 0.5 * (fromdata + todata)
                self.insertXMarker(pos, 'ROI middle',
                                   label="",
                                   color='yellow',
                                   draggable=True)
            self.currentROI = ddict['key'] 
            if ddict['colheader'] in ['From', 'To']:
                dict0 ={}
                dict0['event']  = "SetActiveCurveEvent"
                dict0['legend'] = self.graph.getactivecurve(justlegend=1)
                self.__graphsignal(dict0)
            elif ddict['colheader'] == 'Raw Counts':    
                pass
            elif ddict['colheader'] == 'Net Counts':    
                pass
            else:
                self.emitCurrentROISignal()
        else:
            if DEBUG:
                print("Unknown or ignored event", ddict['event'])

    def emitCurrentROISignal(self):
        ddict = {}
        ddict['event'] = "currentROISignal"
        roiList, roiDict = self.roiWidget.getROIListAndDict()
        if self.currentROI in roiDict:
            ddict['ROI'] = roiDict[self.currentROI]
        else:
            self.currentROI = None
        ddict['current'] = self.currentROI
        self.sigROISignal.emit(ddict)        

    def calculateROIs(self, *var, **kw):
        if not hasattr(self, "roiWidget"):
            return
        if len(var) == 0:
            roiList, roiDict = self.roiWidget.getROIListAndDict()
        elif len(var) == 2:
            roiList = var[0]
            roiDict = var[1]
        else:
            raise ValueError("Expected roiList and roiDict or nothing")
        update = kw.get("update", True)
        activeCurve = self.getActiveCurve(just_legend=False)
        if activeCurve is None:
            xproc = None
            yproc = None
            self.roiWidget.setHeader('<b>ROIs of XXXXXXXXXX<\b>')
        elif len(activeCurve):
            x, y, legend = activeCurve[0:3]
            idx = argsort(x, kind='mergesort')
            xproc = take(x, idx)
            yproc = take(y, idx)
            self.roiWidget.setHeader('<b>ROIs of %s<\b>' % legend)
        else:
            xproc = None
            yproc = None
            self.roiWidget.setHeader('<b>ROIs of XXXXXXXXXX<\b>')
        for key in roiList:
            #roiDict[key]['rawcounts'] = " ?????? "
            #roiDict[key]['netcounts'] = " ?????? "
            if key == 'ICR':
                if xproc is not None:
                    roiDict[key]['from'] = xproc.min()
                    roiDict[key]['to'] = xproc.max()
                else:
                    roiDict[key]['from'] = 0
                    roiDict[key]['to'] = -1
            fromData  = roiDict[key]['from']
            toData = roiDict[key]['to']
            if xproc is not None:
                idx = nonzero((fromData <= xproc) &\
                                   (xproc <= toData))[0]
                if len(idx):
                    xw = x[idx]
                    yw = y[idx]
                    rawCounts = yw.sum(dtype=numpy.float)
                    deltaX = xw[-1] - xw[0]
                    deltaY = yw[-1] - yw[0]
                    if deltaX > 0.0:
                        slope = (deltaY/deltaX)
                        background = yw[0] + slope * (xw - xw[0])
                        netCounts = rawCounts -\
                                    background.sum(dtype=numpy.float)
                    else:
                        netCounts = 0.0
                else:
                    rawCounts = 0.0
                    netCounts = 0.0
                roiDict[key]['rawcounts'] = rawCounts
                roiDict[key]['netcounts'] = netCounts
        if update:
            if self.currentROI in roiList:
                self.roiWidget.fillFromROIDict(roilist=roiList,
                                               roidict=roiDict,
                                               currentroi=self.currentROI)
            else:
                self.roiWidget.fillFromROIDict(roilist=roiList,
                                               roidict=roiDict)
        else:
            return roiList, roiDict

    def _buildLegendWidget(self):
        if self.legendWidget is None:
            self.legendWidget = LegendSelector.LegendListView()
            self.legendDockWidget = qt.QDockWidget(self)
            self.legendDockWidget.layout().setContentsMargins(0, 0, 0, 0)
            self.legendDockWidget.setWidget(self.legendWidget)
            w = self.centralWidget().width()
            h = self.centralWidget().height()
            if w > (1.25 * h):
                self.addDockWidget(qt.Qt.RightDockWidgetArea,
                                   self.legendDockWidget)
            else:
                self.addDockWidget(qt.Qt.BottomDockWidgetArea,
                                   self.legendDockWidget)
            if hasattr(self, "roiDockWidget"):
                if self.roiDockWidget is not None:
                    self.tabifyDockWidget(self.roiDockWidget,
                                      self.legendDockWidget)
            self.legendWidget.sigLegendSignal.connect(self._legendSignal)
            self.legendDockWidget.setWindowTitle(self.windowTitle()+(" Legend"))
        
    def _legendSignal(self, ddict):
        if DEBUG:
            print("Legend signal ddict = ", ddict)
        if ddict['event'] == "legendClicked":
            if ddict['button'] == "left":
                ddict['label'] = ddict['legend']
                self.graphCallback(ddict)
        elif ddict['event'] == "removeCurve":
            ddict['label'] = ddict['legend']
            self.removeCurve(ddict['legend'], replot=True)
        elif ddict['event'] == "setActiveCurve":
            ddict['event'] = 'legendClicked'
            ddict['label'] = ddict['legend']
            self.graphCallback(ddict)
        elif ddict['event'] == "checkBoxClicked":
            if ddict['selected']:
                self.hideCurve(ddict['legend'], False)
            else:
                self.hideCurve(ddict['legend'], True)
        elif ddict['event'] in ["mapToRight", "mapToLeft"]:
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            if ddict['event'] == "mapToRight":
                self.addCurve(x, y, legend=legend, info=info, yaxis="right")
            else:
                self.addCurve(x, y, legend=legend, info=info, yaxis="left")
        elif ddict['event'] == "togglePoints":
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            if ddict['points']:
                self.addCurve(x, y, legend=legend, info=info, symbol='o')
            else:
                self.addCurve(x, y, legend, info, symbol='')
            self.updateLegends()
        elif ddict['event'] == "toggleLine":
            legend = ddict['legend']
            x, y, legend, info = self._curveDict[legend][0:4]
            if ddict['line']:
                self.addCurve(x, y, legend=legend, info=info, line_style="-")
            else:
                self.addCurve(x, y, legend, info=info, line_style="")
            self.updateLegends()
        elif DEBUG:
            print("unhandled event", ddict['event'])

    def toggleLegendWidget(self):
        if self.legendWidget is None:
            self.showLegends(True)
        elif self.legendDockWidget.isHidden():
            self.showLegends(True)
        else:
            self.showLegends(False)

    def showLegends(self, flag=True):
        if self.legendWidget is None:
            self._buildLegendWidget()
            self.updateLegends()
        if flag:
            self.legendDockWidget.show()
            self.updateLegends()
        else:
            self.legendDockWidget.hide()

    def updateLegends(self):
        if self.legendWidget is None:
            return
        if self.legendDockWidget.isHidden():
            return
        legendList = [] * len(self._curveList)
        for i in range(len(self._curveList)):
            legend = self._curveList[i]
            color = self._curveDict[legend][3].get('plot_color',
                                                         '#000000')
            color = qt.QColor(color)
            linewidth = self._curveDict[legend][3].get('plot_line_width',
                                                             2)
            symbol = self._curveDict[legend][3].get('plot_symbol',
                                                    None)
            if self.isCurveHidden(legend):
                selected = False
            else:
                selected = True
            ddict={'color':color,
                   'linewidth':linewidth,
                   'symbol':symbol,
                   'selected':selected}
            legendList.append((legend, ddict))
        self.legendWidget.setLegendList(legendList)


    def setMiddleROIMarkerFlag(self, flag=True):
        if flag:
            self._middleROIMarkerFlag = True
        else:
            self._middleROIMarkerFlag= False

if __name__ == "__main__":
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = PlotWindow(roi=True, control=True, position=True)#uselegendmenu=True)
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, x*x)
    plot.addCurve(x, -y, "- dummy")
    print("Active curve = ", plot.getActiveCurve(just_legend=True))
    print("X Limits = ",     plot.getGraphXLimits())
    print("Y Limits = ",     plot.getGraphYLimits())
    print("All curves = ",   plot.getAllCurves(just_legend=True))
    #plot.removeCurve("dummy")
    #plot.addCurve(x, 2 * y, "dummy 2")
    #print("All curves = ",   plot.getAllCurves())
    app.exec_()
