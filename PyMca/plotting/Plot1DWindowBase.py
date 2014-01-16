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
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
import traceback
from PyMca import PyMcaQt as qt
import Plot
from PyMca.PyMca_Icons import IconDict
QTVERSION = qt.qVersion()
DEBUG = 0

class Plot1DWindowBase(qt.QWidget):
    def __init__(self, parent=None, backend=None, plugins=True, newplot=False,**kw):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.pluginsIconFlag = plugins
        self.newplotIconsFlag = newplot
        self.plotType=None      # None, "SCAN", "MCA"
        self._initIcons()
        self._buildToolBar()
        self.graph = Plot.Plot(parent=self, backend=backend)
        # initialize title and labels
        self.graph.setGraphTitle("Title")
        self.graph.setGraphXLabel("X")
        self.graph.setGraphYLabel("Y")

        try:
            self.mainLayout.addWidget(self.graph.widget)
        except:
        #    #testing
            pass
        self._toggleCounter = 0

    def _initIcons(self):
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
        self.xAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["xauto"]))
        self.yAutoIcon	= qt.QIcon(qt.QPixmap(IconDict["yauto"]))
        self.togglePointsIcon = qt.QIcon(qt.QPixmap(IconDict["togglepoints"]))

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

    def _buildToolBar(self):
        self.toolBar = qt.QWidget(self)
        self.toolBarLayout = qt.QHBoxLayout(self.toolBar)
        self.toolBarLayout.setMargin(0)
        self.toolBarLayout.setSpacing(2)
        self.layout().insertWidget(0, self.toolBar)
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
        self.yLogButton = self._addToolButton(self.logyIcon,
                            self._toggleLogY,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle = True)
        self.yLogButton.setChecked(False)
        self.yLogButton.setDown(False)

        #x Logarithmic
        self.xLogButton = self._addToolButton(self.logxIcon,
                            self._toggleLogX,
                            'Toggle Logarithmic X Axis (On/Off)',
                            toggle = True)
        self.xLogButton.setChecked(False)
        self.xLogButton.setDown(False)

        #toggle Points/Lines
        tb = self._addToolButton(self.togglePointsIcon,
                             self._togglePointsSignal,
                             'Toggle Points/Lines')


        #fit icon
        self.fitButton = self._addToolButton(self.fitIcon,
                                 self._fitIconSignal,
                                 'Simple Fit of Active Curve')


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

        self.toolBarLayout.addWidget(qt.HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self.printGraph,
                                 'Prints the Graph')

    def _addToolButton(self, icon, action, tip, toggle=None):
        tb      = qt.QToolButton(self.toolBar)            
        tb.setIcon(icon)
        tb.setToolTip(tip)
        if toggle is not None:
            if toggle:
                tb.setCheckable(1)
        self.toolBarLayout.addWidget(tb)
        #self.connect(tb,qt.SIGNAL('clicked()'), action)
        tb.clicked.connect(action)
        return tb
                
    def _zoomReset(self):
        print("reset zoom")

    def _yAutoScaleToggle(self):
        print("toggle Y auto scaling")

    def _xAutoScaleToggle(self):
        print("toggle X auto scaling")
                       
    def _toggleLogX(self):
        print("toggle logarithmic X scale")

    def _toggleLogY(self):
        print("toggle logarithmic Y scale")

    def _togglePointsSignal(self):
        print("toggle points signal")

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
        text = qt.QString("Reload")
        menu.addAction(text)
        actionList.append(text)
        menu.addSeparator()
        callableKeys = ["Dummy"]
        for m in self.pluginList:
            if m == "PyMcaPlugins.Plugin1DBase":
                continue
            module = sys.modules[m]
            if hasattr(module, 'MENU_TEXT'):
                text = qt.QString(module.MENU_TEXT)
            else:
                text = os.path.basename(module.__file__)
                if text.endswith('.pyc'):
                    text = text[:-4]
                elif text.endswith('.py'):
                    text = text[:-3]
                text = qt.QString(text)
            methods = self.pluginInstanceDict[m].getMethods(plottype=self.plotType) 
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
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods(plottype="SCAN")
        if len(methods) == 1:
            idx = 0
        else:
            actionList = []
            methods.sort()
            menu = qt.QMenu(self)
            for method in methods:
                text = qt.QString(method)
                pixmap = self.pluginInstanceDict[key].getMethodPixmap(method)
                tip = qt.QString(self.pluginInstanceDict[key].getMethodToolTip(method))
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
            if QTVERSION < '4.0.0':
                msg.setText("%s" % sys.exc_info()[1])
            else:
                msg.setWindowTitle("Plugin error")
                msg.setText("An error has occured while executing the plugin:")
                msg.setInformativeText(str(sys.exc_info()[1]))
                msg.setDetailedText(traceback.format_exc())
            msg.exec_()

    def _actionHovered(self, action):
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)

    def printGraph(self):
        print("prints the graph")

if __name__ == "__main__":
    import numpy
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = Plot1DWindowBase(uselegendmenu=True)
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, x*x)
    print("Active curve = ", plot.getActiveCurve())
    print("X Limits = ",     plot.getGraphXLimits())
    print("Y Limits = ",     plot.getGraphYLimits())
    print("All curves = ",   plot.getAllCurves())
    plot.removeCurve("dummy")
    plot.addCurve(x, y, "dummy 2")
    print("All curves = ",   plot.getAllCurves())
    app.exec_()

    
