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
from PyMca.PyMca_Icons import IconDict
import Plot1DWindowBase
import Plot

QTVERSION = qt.qVersion()
DEBUG = 0

class Plot1DWindow(Plot1DWindowBase.Plot1DWindowBase):
    def __init__(self, parent=None, **kw):
        super(Plot1DWindow, self).__init__(parent=parent,
                                           backend=backend,
                                           **kw)
        #self.graph is the  Plot.Plot(parent=self, backend=backend)
        #self.graph.widget is the widget but it should not be needed
        #self.graph.setGraphTitle("Title")
        #self.graph.setGraphXLabel("X")
        #self.graph.setGraphYLabel("Y")
        self._toggleCounter = 0
        self._logY = False
        self._logX = False
        
    def __getattr__(self, attr):
        # implicitly wrap methods from Plot
        if hasattr(self.graph, attr):
            m = getattr(self.graph, attr)
            if hasattr(m, '__call__'):
                return m
        return super(Plot1DWindow, self).__getattr__(attr)
        #raise NameError(attr)

    def addCurve(self, *var, **kw):
        return self.graph.addCurve(*var, **kw)
                
    def _zoomReset(self):
        if DEBUG:
            print("_zoomReset")
        self.graph.resetZoom()

    def _yAutoScaleToggle(self):
        if DEBUG:
            print("_yAutoScaleToggle")
        if self.graph.isYAxisAutoScale():
            self.graph.setYAxisAutoScale(False)
            self.yAutoScaleButton.setDown(False)
            self.yAutoScaleButton.setChecked(False)
            self.graph.setGraphYLimits(*self.graph.getGraphYLimits())
        else:
            self.graph.setYAxisAutoScale(True)
            self.yAutoScaleButton.setDown(True)
            self.graph.resetZoom()
                       
    def _xAutoScaleToggle(self):
        if DEBUG:
            print("_xAutoScaleToggle")
        if self.graph.isXAxisAutoScale():
            self.graph.setXAxisAutoScale(False)
            self.xAutoScaleButton.setDown(False)
            self.xAutoScaleButton.setChecked(False)
            self.graph.setGraphXLimits(*self.graph.getGraphXLimits())
        else:
            self.graph.setXAxisAutoScale(True)
            self.xAutoScaleButton.setDown(True)
            self.graph.resetZoom()
                       
    def _toggleLogY(self):
        if DEBUG:
            print("_toggleLogY")
        if self._logY:
            self._logY = False
        else:
            self._logY = True
        self.graph.setYAxisLogarithmic(self._logY)

    def _toggleLogX(self):
        if DEBUG:
            print("_toggleLogX")
        if self._logX:
            self._logX = False
        else:
            self._logX = True
        self.graph.setXAxisLogarithmic(self._logX)

    def _togglePointsSignal(self):
        self._toggleCounter = (self._toggleCounter + 1) % 3
        if self._toggleCounter == 1:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(True)
        elif self._toggleCounter == 2:
            self.graph.setDefaultPlotPoints(True)
            self.graph.setDefaultPlotLines(False)
        else:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(False)
        self.graph.setActiveCurve(self.graph.getActiveCurve(just_legend=1))
        self.graph.replot()

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
            print("TO NEW STYLE?????")
            qt.QObject.connect(menu,
                               qt.SIGNAL("hovered(QAction *)"),
                               self._actionHovered)
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

    def printGraph(self):
        print("prints the graph")


if __name__ == "__main__":
    import numpy
    import time
    if "matplotlib" in sys.argv:
        from MatplotlibBackend import MatplotlibBackend as backend
        print("USING matplotlib")
        time.sleep(1)
    else:
        from PyQtGraphBackend import PyQtGraphBackend as backend
        print("USING PyQtGraph")
        time.sleep(1)

    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = Plot1DWindow(backend=backend, uselegendmenu=True)
    plot.show()
    if 1:
        plot.addCurve(x, y, "dummy")
        plot.addCurve(x+100, x*x)
        plot.addCurve(x, -y, "dummy 2")
        print("Active curve = ", plot.getActiveCurve())
        print("X Limits = ",     plot.getGraphXLimits())
        print("Y Limits = ",     plot.getGraphYLimits())
        print("All curves = ",   plot.getAllCurves())
        #print("REMOVING dummy")
        #plot.removeCurve("dummy")
        plot.insertXMarker(50., draggable=True)
        #plot.insertYMarker(50., draggable=True)
    else:
        # insert a few curves
        cSin={}
        cCos={}
        nplots=50
        for i in range(nplots):
            # calculate 3 NumPy arrays
            x = numpy.arange(0.0, 10.0, 0.1)
            y = 10*numpy.sin(x+(i/10.0) * 3.14)
            z = numpy.cos(x+(i/10.0) * 3.14)
            #build a key
            a="%d" % i
            #plot the data
            cSin[a] = plot.addCurve(x, y, 'y = sin(x)' + a, replot=False)
            cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=False)
        cCos[a] = plot.addCurve(x, z, 'y = cos(x)' + a, replot=True)
        plot.insertXMarker(5., draggable=True)
        plot.insertYMarker(5., draggable=True)
    print("All curves = ", plot.getAllCurves(just_legend=True))
    app.exec_()
