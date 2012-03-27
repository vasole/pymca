#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
from PyMca import Plot1DWindowBase
qt = Plot1DWindowBase.qt
QTVERSION = qt.qVersion()
if QTVERSION < '4.0.0':
    raise ImportError("Plot1DMatplotlib.py plotting module expects Qt4")
from PyMca.PyMca_Icons import IconDict
from PyMca import PyMcaPrintPreview
from PyMca import PyMcaDirs
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import matplotlib.patches as patches
Rectangle = patches.Rectangle
Polygon = patches.Polygon
import numpy

DEBUG = 0

colordict = {}
colordict['blue']   = '#0000ff'
colordict['red']    = '#ff0000'
colordict['green']  = '#00ff00'
colordict['black']  = '#000000'
colordict['white']  = '#ffffff'
colordict['pink']   = '#ff66ff'
colordict['brown']  = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['yellow'] = '#ffff00'
colordict['darkgreen'] = 'g'
colordict['darkbrown'] = '#660000' 
colordict['magenta']   = 'm' 
colordict['cyan']      = 'c'
colordict['bluegreen'] = '#33ffff'
colorlist  = [colordict['black'],
              colordict['red'],
              colordict['blue'],
              colordict['green'],
              colordict['pink'],
              colordict['brown'],
              colordict['cyan'],
              colordict['orange'],
              colordict['violet'],
              colordict['bluegreen'],
              colordict['grey'],
              colordict['magenta'],
              colordict['darkgreen'],
              colordict['darkbrown'],
              colordict['yellow']]

class MatplotlibGraph(FigureCanvas):
    def __init__(self, parent, **kw):
       	#self.figure = Figure(figsize=size, dpi=dpi) #in inches
        self.fig = Figure()
        self._canvas = FigureCanvas.__init__(self, self.fig)
        self.ax = self.fig.add_axes([.15, .15, .75, .75])
        FigureCanvas.setSizePolicy(self,
                                   qt.QSizePolicy.Expanding,
                                   qt.QSizePolicy.Expanding)

        self.colorList = colorlist
        self.styleList = ['-', '-.', ':']
        self.nColors   = len(colorlist)
        self.nStyles   = len(self.styleList)

        self.colorIndex = 0
        self.styleIndex = 0

        self._legendList = []
        self._dataCounter = 0

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.limitsSet = False

        self._logY = False

        #default settings
        self.setDefaultPlotPoints(False)
        self.setDefaultPlotLines(True)

        #zoom handling
        self.enableZoom = self.setZoomModeEnabled
        self.setZoomModeEnabled(True)
        self.__zooming = False
        self._zoomStack = []
        self.xAutoScale = True
        self.yAutoScale = True

        #drawingmode handling
        self.setDrawModeEnabled(False)
        self.__drawModeList = ['line', 'polygon']
        self.__drawing = False
        self._drawingPatch = None
        self._drawModePatch = 'line'

        #event handling
        self._x0 = None
        self._y0 = None
        self._zoomRectangle = None
        self.fig.canvas.mpl_connect('button_press_event',
                                    self.onMousePressed)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.onMouseReleased)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.onMouseMoved)

    def setDefaultPlotPoints(self, flag=False):
        if flag:
            self.__plotPoints = True
        else:
            self.__plotPoints = False

    def setDefaultPlotLines(self, flag=False):
        if flag:
            self.__plotLines = True
        else:
            self.__plotLines = False

    def setZoomModeEnabled(self, flag=True):
        self.__zoomEnabled = flag
        if flag:
            #cannot draw and zoom simultaneously
            self.setDrawModeEnabled(False)
            self._selecting = False

    def setDrawModeEnabled(self, flag=True):
        self.__drawModeEnabled = flag
        if flag:
            #cannot draw and zoom simultaneously
            self.setZoomModeEnabled(False)

    def setDrawModePatch(self, mode=None):
        if mode is None:
            mode = self.__drawModeList[0]

        mode = mode.lower()
        #raise an error in case of an invalid mode
        modeIndex = self.__drawModeList.index(mode)

        self._drawModePatch = mode

    def onMousePressed(self, event):
        if DEBUG:
            print("onMousePressed, event = ",event.xdata, event.ydata)
        if event.inaxes != self.ax:
            if DEBUG:
                print("RETURNING")
            return
        if event.button == 3:
            #right click
            if self._drawingPatch is not None:
                self._drawingPatch.remove()
                self.draw()
            self._drawingPatch = None
            self.__zooming = False
            return

        self.__zooming = self.__zoomEnabled
        self._zoomRect = None
        self._x0 = event.xdata
        self._y0 = event.ydata
        self._xmin, self._xmax  = self.ax.get_xlim()
        self._ymin, self._ymax  = self.ax.get_ylim()

        self.__drawing = self.__drawModeEnabled
            
    def onMouseMoved(self, event):
        if DEBUG:
            print("onMouseMoved, event = ",event.xdata, event.ydata)
        if (not self.__zooming) and (not self.__drawing):
            return
        elif event.inaxes != self.ax:
            if DEBUG:
                print("RETURNING")
            return

        if self._x0 is None:
            return
        self._x1 = event.xdata
        self._y1 = event.ydata
        
        if self.__zooming:
            if self._x1 < self._xmin:
                self._x1 = self._xmin
            elif self._x1 > self._xmax:
                self._x1 = self._xmax
     
            if self._y1 < self._ymin:
                self._y1 = self._ymin
            elif self._y1 > self._ymax:
                self._y1 = self._ymax
     
            if self._x1 < self._x0:
                x = self._x1
                w = self._x0 - self._x1
            else:
                x = self._x0
                w = self._x1 - self._x0
            if self._y1 < self._y0:
                y = self._y1
                h = self._y0 - self._y1
            else:
                y = self._y0
                h = self._y1 - self._y0

            if self._zoomRectangle is None:
                self._zoomRectangle = Rectangle(xy=(x,y),
                                               width=w,
                                               height=h,
                                               fill=False)
                self.ax.add_patch(self._zoomRectangle)
            else:
                self._zoomRectangle.set_bounds(x, y, w, h)
                #self._zoomRectangle._update_patch_transform()
            self.fig.canvas.draw()
            return
        
        if self.__drawing:
            if self._drawingPatch is None:
                self._mouseData = numpy.zeros((2,2), numpy.float32)
                self._mouseData[0,0] = self._x0
                self._mouseData[0,1] = self._y0
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = self._y1
                self._drawingPatch = Polygon(self._mouseData,
                                             closed=True,
                                             fill=False)
                self.ax.add_patch(self._drawingPatch)
            elif self._drawModePatch == 'line':
                self._mouseData[1,0] = self._x1
                self._mouseData[1,1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
            elif self._drawModePatch == 'polygon':
                self._mouseData[-1,0] = self._x1
                self._mouseData[-1,1] = self._y1
                self._drawingPatch.set_xy(self._mouseData)
                self._drawingPatch.set_hatch('/')
                self._drawingPatch.set_closed(True)
            self.fig.canvas.draw()
        
    def onMouseReleased(self, event):
        if DEBUG:
            print("onMouseReleased, event = ",event.xdata, event.ydata)
        if event.button == 3:
            #right click
            if self.__drawing:
                self.__drawing = False
                self._drawingPatch = None
                ddict = {}
                ddict['event'] = 'drawingFinished'
                ddict['type']  = '%s' % self._drawModePatch
                ddict['data']  = self._mouseData * 1
                self.mySignal(ddict)
                return

            self.__zooming = False
            if len(self._zoomStack):
                xmin, xmax, ymin, ymax = self._zoomStack.pop()
                self.setLimits(xmin, xmax, ymin, ymax)
                self.draw()

        if self._x0 is None:
            return

        if self.__drawing and (self._drawingPatch is not None):
            nrows, ncols = self._mouseData.shape                
            self._mouseData = numpy.resize(self._mouseData, (nrows+1,2))
            self._mouseData[-1,0] = self._x1
            self._mouseData[-1,1] = self._y1
            self._drawingPatch.set_xy(self._mouseData)

        if (self._zoomRectangle is None):
            return

        if self._zoomRectangle is not None:
            x, y = self._zoomRectangle.get_xy()
            w = self._zoomRectangle.get_width()
            h = self._zoomRectangle.get_height()
            self._zoomRectangle.remove()
            self._x0 = None
            self._y0 = None
            self._zoomRectangle = None
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self._zoomStack.append((xmin, xmax, ymin, ymax))
            self.setLimits(x, x+w, y, y+h)
            self.draw()


    def mySignal(self, ddict):
        self.emit(qt.SIGNAL('MatplotlibGraphSignal'), ddict)

    def setLimits(self, xmin, xmax, ymin, ymax):
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.limitsSet = True

    def getX1AxisLimits(self):
        if DEBUG:
            print("getX1AxisLimitsCalled")
        xlim = self.ax.get_xlim()
        return xlim

    def getY1AxisLimits(self):
        if DEBUG:
            print("getY1AxisLimitsCalled")
        ylim = self.ax.get_ylim()

    def _filterData(self, x, y):
        index = numpy.flatnonzero((self.xmin <= x) & (x <= self.xmax))
        x = numpy.take(x, index)
        y = numpy.take(y, index)
        index = len(index)
        if index:
            index = numpy.flatnonzero((self.ymin <= y) & (y <= self.ymax))
            index = len(index)
        return index

    def _getColorAndStyle(self):
        self.__lastColorIndex = self.colorIndex * 1
        self.__lastStyleIndex = self.styleIndex * 1
        color = self.colorList[self.colorIndex]
        style = self.styleList[self.styleIndex]
        self.colorIndex += 1
        if self.colorIndex >= self.nColors:
            self.colorIndex = 0
            self.styleIndex += 1
            if self.styleIndex >= self.nStyles:
                self.styleIndex = 0        
        return color, style

    def addDataToPlot(self, x, y, legend = None,
                      color = None,
                      linewidth = None,
                      linestyle = None, **kw):
        if self.limitsSet:
            n = max(x.shape)
            if self.limitsSet is not None:
                n = self._filterData(x, y)
            if n == 0:
                #nothing to plot
                if DEBUG:
                    print("nothing to plot")
                return
        style = None
        if color is None:
            color, style = self._getColorAndStyle()
        if linestyle is None:
            if style is None:
                style = '-'
        else:
            style = linestyle
        if legend is None:
            #legend = "%02d" % self._dataCounter    #01, 02, 03, ...
            legend = "%c" % (96+self._dataCounter)  #a, b, c, ..
        if linewidth is None:linewidth = 1.0
        #self.ax = self.fig.add_axes([.15, .15, .75, .75])
        label = None
        if legend in self._legendList:
            for line2D in self.ax.lines:
                label = line2D.get_label()
                if label == legend:
                    break
        if label is not None:
           if kw.has_key('marker'):
               line2D.set_marker(kw['marker'])
           line2D.set_linestyle(style)
           #line2D.set_xdata(x)
           #line2D.set_ydata(y)
           line2D.set_data(x, y)
           #restore color and style index
           self.colorIndex = self.__lastColorIndex * 1
           self.styleIndex = self.__lastStyleIndex * 1
           return
        if self._logY:
            curveList = self.ax.semilogy( x, y, label=legend, linestyle =style, color=color, linewidth = linewidth, **kw)
            #self.ax.set_yscale('log')
        else:
            curveList = self.ax.plot( x, y, label=legend, linestyle =style, color=color, linewidth = linewidth, **kw)
        #curveList[0].set_linestyle(style)
        self._dataCounter += 1
        self._legendList.append(legend)

    def newCurve(self, legend, x, y, **kw):
        if self.__plotPoints:
            marker = 'o'
        else:
            marker = 'None'
        if self.__plotLines:
            linestyle = None
        else:
            linestyle = ''
        self.addDataToPlot( x, y, legend=legend, linewidth=1.5,
                            linestyle=linestyle, marker=marker)

    def removeCurve(self, legend):
        del self._legendList[self._legendList.index(legend)]
        for line2D in self.ax.lines:
            label = line2D.get_label()
            if label == legend:
                line2D.remove()
                break


    #QtBlissGraph like 
    def isZoomEnabled(self):
        if self.__zoomEnabled:
            return True
        else:
            return False

    def setTitle(self, text=""):
        self.ax.set_title(text)

    def y1Label(self, label=None):
        if label is None:
            return self.ax.get_ylabel()
        else:
            return self.ax.set_ylabel(label)

    def x1Label(self, label=None):
        if label is None:
            return self.ax.get_xlabel()
        else:
            return self.ax.set_xlabel(label)
        
    def zoomReset(self):
        self._zoomStack = []
        xmin = None
        for line2D in self.ax.lines:
            x = line2D.get_xdata()
            y = line2D.get_ydata()
            if xmin is None:
                xmin = x.min()
                xmax = x.max()
                ymin = y.min()
                ymax = y.max()
                continue
            xmin = min(xmin, x.min())
            xmax = max(xmax, x.max())
            ymin = min(ymin, y.min())
            ymax = max(ymax, y.max())
        if xmin is None:
            xmin = 0
            xmax = 1
            ymin = 0
            ymax = 1
        self.setLimits(xmin, xmax, ymin, ymax)

class Plot1DMatplotlib(Plot1DWindowBase.Plot1DWindowBase):
    def __init__(self, parent=None,**kw):
        Plot1DWindowBase.Plot1DWindowBase.__init__(self, **kw)
        mainLayout = self.layout()
        self.graph = MatplotlibGraph(self, **kw)
        mainLayout.addWidget(self.graph)
        self._logY = False
        self.newCurve = self.addCurve
        self.setTitle = self.graph.setTitle
        self.__toggleCounter = 0

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True, **kw):
        """
        Add the 1D curve given by x an y to the graph.
        """
        if legend is None:
            key = "Unnamed curve 1.1"
        else:
            key = str(legend)
        if info is None:
            info = {}
        xlabel = info.get('xlabel', 'X')
        ylabel = info.get('ylabel', 'Y')
        if kw.has_key('xlabel'):
            info['xlabel'] = kw['xlabel'] 
        if kw.has_key('ylabel'):
            info['ylabel'] = kw['ylabel'] 
        Plot1DWindowBase.Plot1DWindowBase.addCurve(self, x, y, legend=legend,
                               info=info, replace=replace, replot=replot)
        if replot:
            if replace:
                self.replot('REPLACE')
            else:
                self.replot('ADD')

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        """
        Plot1DWindowBase.Plot1DWindowBase.removeCurve(self, legend, replot=replot)
        if legend in self.graph._legendList:
            self.graph.removeCurve(legend)
        self._updateActiveCurve()
        #if replot:
        #    self.graph.replot()
        return
        
    def replot(self, mode=None):
        if mode in [None, 'REPLACE']:
            if mode == 'REPLACE':                
                #self.graph.fig.clear()
                self.graph.ax.cla()
                self.graph._dataCounter = 0
            for curve in self.curveList:
                x, y, legend, info = self.curveDict[curve]
                self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self._updateActiveCurve()
            self.graph.draw()
            return
        
        if mode.upper() == 'ADD':
            #currentCurves = self.graph.curves.keys()
            for curve in self.curveList:
                #if curve not in currentCurves:
                    x, y, legend, info = self.curveDict[curve] 
                    self.graph.newCurve(curve, x, y, logfilter=self._logY)
            self.graph.draw()
            return

    def _zoomReset(self):
        if True:
            #looking at own data
            xmin, xmax = self.getGraphXLimits()
            ymin, ymax = self.getGraphYLimits()
            self.graph.setLimits(xmin, xmax, ymin, ymax)
            self.graph._zoomStack = []
        else:
            #use graph data themselves
            self.graph.zoomReset()
        self.graph.draw()


    def _updateActiveCurve(self):
        self.activeCurve = self.getActiveCurve(just_legend=True)
        if self.activeCurve not in self.curveList:
            self.activeCurve = None
        if self.activeCurve is None:
            if len(self.curveList):
                self.activeCurve = self.curveList[0]
        if self.activeCurve is not None:
            #self.graph.setActiveCurve(self.activeCurve)
            info = self.curveDict[self.activeCurve][-1]
            xlabel = info['xlabel']
            ylabel = info['ylabel']
            self.graph.x1Label(xlabel)
            self.graph.y1Label(ylabel)

    def _toggleLogY(self):
        if self._logY:
            self._logY = False
        else:
            self._logY = True
        #self.graph.fig.clear()
        #self.graph.ax.clear()
        self.graph._logY = self._logY
        if self._logY:
            self.graph.ax.set_yscale('log')
        else:
            self.graph.ax.set_yscale('linear')
        self.graph.draw()

    def _togglePointsSignal(self):
        self.__toggleCounter = (self.__toggleCounter + 1) % 3
        if self.__toggleCounter == 1:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(True)
        elif self.__toggleCounter == 2:
            self.graph.setDefaultPlotPoints(True)
            self.graph.setDefaultPlotLines(False)
        else:
            self.graph.setDefaultPlotLines(True)
            self.graph.setDefaultPlotPoints(False)
        #self.graph.setActiveCurve(self.graph.getActiveCurve(justlegend=1))
        self.replot()

if 0:
    def getGraphXLimits(self):
        """
        Get the graph X limits. 
        """
        xmin, ymin, xmax, ymax = self.graph.getX1AxisLimits()
        return xmin, xmax
        
    def getGraphYLimits(self):
        """
        Get the graph Y (left) limits. 
        """
        xmin, ymin, xmax, ymax = self.graph.getY1AxisLimits()
        return ymin, ymax

    def setActiveCurve(self, legend):
        """
        Funtion to request the plot window to set the curve with the specified
        legend as the active curve.
        It returns the active curve legend.
        """
        key = str(legend)
        if key in self.curveDict.keys():
            self.activeCurve = key
            self.graph.setActiveCurve(key)
        return self.activeCurve

class Plot1DMatplotlibDialog(qt.QDialog):
    def __init__(self, parent=None, **kw):
        qt.QDialog.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.plot1DWindow = Plot1DMatplotlib(self)
        self.mainLayout.addWidget(self.plot1DWindow)


if __name__ == "__main__":
    x = numpy.arange(100.)
    y = x * x
    app = qt.QApplication([])
    plot = Plot1DMatplotlib(uselegendmenu=True)
    if 0:
        plot.graph.setZoomModeEnabled(True)
    else:
        def mySlot(ddict):
            print(ddict['event'])
            print(ddict['data'])
        qt.QObject.connect(plot.graph,
                           qt.SIGNAL('MatplotlibGraphSignal'),
                           mySlot)
        plot.graph.setDrawModeEnabled(True)
        plot.graph.setDrawModePatch('polygon')
    plot.show()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x+100, -x*x)
    print("Active curve = ", plot.getActiveCurve())
    print("X Limits = ",     plot.getGraphXLimits())
    print("Y Limits = ",     plot.getGraphYLimits())
    print("All curves = ",   plot.getAllCurves())
    plot.removeCurve("dummy")
    plot.addCurve(x, y, "dummy2")
    plot.graph.setTitle('Title')
    plot.graph.x1Label('X')
    plot.graph.y1Label('Y')
    print("All curves = ",   plot.getAllCurves())
    app.exec_()
