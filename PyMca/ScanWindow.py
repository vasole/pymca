#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
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
# is a problem for you.
#############################################################################*/
import sys
import os
import time
from QtBlissGraph import qt
if __name__ == "__main__":
    app = qt.QApplication([])
import QtBlissGraph
from PyMca_Icons import IconDict
import numpy.oldnumeric as Numeric
import ScanFit
import SimpleMath
import DataObject
import copy
import PyMcaPrintPreview
import PyMcaDirs
import ScanWindowInfoWidget

QTVERSION = qt.qVersion()


DEBUG = 0

MATPLOTLIB = 0
#QtBlissgraph should have the same colors
colordict = {}
colordict['blue']  = '#0000ff'
colordict['red']   = '#ff0000'
colordict['green'] = '#00ff00'
colordict['black'] = '#000000'
colordict['white'] = '#ffffff'
colordict['pink'] = '#ff66ff'
colordict['brown'] = '#a52a2a'
colordict['orange'] = '#ff9900'
colordict['violet'] = '#6600ff'
colordict['grey']   = '#808080'
colordict['yellow'] = '#ffff00'
colordict['darkgreen']  = 'g'
colordict['darkbrown'] = '#660000' 
colordict['magenta'] = 'm' 
colordict['cyan']  = 'c'
colordict['bluegreen']  = '#33ffff'
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
#qt is not necessary, but I use it in the rest of the application
#without QT:
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#
try:
    from matplotlib import rcParams
    #rcParams['numerix'] = "numeric"
    from matplotlib.font_manager import FontProperties
    #2D stuff
    if QTVERSION < '4.0.0':
        from matplotlib.backends.backend_qtagg import FigureCanvasQT, FigureCanvasQTAgg
    else:
        from matplotlib.backends.backend_qt4agg import  FigureCanvasQT, FigureCanvasQTAgg
    #3D stuff
    from matplotlib.figure import Figure
    import matplotlib.axes3d as axes3d
    MATPLOTLIB = 1
    class FigureCanvas(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=None, height=None, dpi=None,
                     xlist=None, ylist = None,
                     dataobject = None, fixed_print_size = False, toolbar = False,
                     **kw):
            if (width == None) or (height == None):
                if dpi is not None:
                    self.fig = Figure(dpi=dpi)
                else:
                    self.fig = Figure()
            FigureCanvasQTAgg.__init__(self, self.fig)
            self.p = None
            if toolbar:
                if parent is None:
                    self.p = qt.QWidget()
                    layout = qt.QVBoxLayout(self.p)
                layout = self.p.layout()
                layout.setMargin(0)
                layout.setSpacing(0)
                if 0:   #they are not implemented yet
                    button1 = qt.QPushButton(parent)
                    button1.setText("Save")
                    layout.addWidget(button1)
                    #layout.addWidget(HorizontalSpacer(self.p))
                    #button2 = qt.QPushButton(parent)
                    #button2.setText("Print")
                    #layout.addWidget(button2)
            if QTVERSION < '4.0.0':
                if parent is not None:
                    self.reparent(parent, QPoint(0, 0))
                else:
                    #self.reparent(parent, QPoint(0, 0))
                    layout.addWidget(self)
            else:
                if parent is not None:
                    self.setParent(parent)
                    self.move(QPoint(0,0))
                else:
                    if toolbar:
                        layout.addWidget(self)
            self._fixed_print_size = fixed_print_size
            if dataobject is not None:
                self.plotDataObject(dataobject)
            else:
                self.dataObject = None

            if xlist is not None:
                if ylist is not None:
                    self.plotXYList(xlist, ylist, **kw)

            if toolbar:self.p.show()

        def plotDataObject(self, dataObject, **kw):
            if len(dataObject.x) == 1: self._plot2DDataObject(dataObject, **kw)
            else: self._plot3DDataObject(dataObject, **kw)

        def plotXYList(self, xlist, ylist, **kw):
            logy    = False
            legends = True
            legendslist = []
            loc = (1.01, 0.0)
            labelsep = 0.02
            xlabel = None
            ylabel = None
            for key in kw:
                if key.upper() == "LOC":
                    loc = kw[key]
                    continue
                if key.upper() == "LOGY":
                    if kw[key]:logy = True
                    else:logy = False
                    continue
                if key.upper() == "XLABEL":
                    xlabel = kw[key]
                    continue
                if key.upper() == "YLABEL":
                    ylabel = kw[key]
                    continue
                if key.upper() == "LEGENDS":
                    if kw[key]:legends = True
                    else:      legends = False
                    continue
                if key.upper() == "LEGENDSLIST":
                    legendslist = kw[key]
                    continue
                if key.upper() == "XMIN":
                    if kw[key]:xmin = kw[key]
                    else:      xmin = xlist[0]
                    continue
                if key.upper() == "XMAX":
                    if kw[key]:xmax = kw[key]
                    else:      xmax = xlist[-1]
                    continue
                if key.upper() == "YMIN":
                    if kw[key]:ymin = kw[key]
                    else:      ymin = None
                    continue
                if key.upper() == "YMAX":
                    if kw[key]:ymax = kw[key]
                    else:      ymax = None

            icolor = 0
            if len(legendslist) == 0:legends = False
            if not legends:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .85, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .85, .8])
            else:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .7, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .7, .8])
            if logy:
                axfunction = ax.semilogy
            else:
                axfunction = ax.plot
            ax.set_axisbelow(True)
            i = 0
            lshape = "-"
            for iplot in range(len(xlist)):
                xplot = xlist[iplot]
                yplot = ylist[iplot]
                axfunction(xplot, yplot, ls=lshape, color=colorlist[i],lw=1.5)
                i += 1
                if i == len(colorlist):
                    i = 2   #black and red used only once
                    lshape = ".."

            if legends and len(legendslist):
                if len(legendslist) > 14:
                    #loc = (1.01, 0.0)  # outside at plot bottom
                    fontsize = 8
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                else:
                    fontsize = 10
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                    
                legend = ax.legend(legendslist,
                                   loc = loc,
                                   #fontname = "Times",
                                   prop = fontproperties,
                                   labelsep = labelsep,
                                   pad = 0.15)
                legend.draw_frame(drawframe)
            else:
                fontsize = 10
                fontproperties = FontProperties(size=fontsize)
            #if xlabel is not None:ax.set_xlabel(xlabel) #, prop=fontproperties)
            #if ylabel is not None:ax.set_ylabel(ylabel) #, prop=fontproperties)
            ax.set_xlim(xmin, xmax)
            self.fig.canvas.show()
            self.print_figure("test"+".png", dpi=300) #print quality independent
            #canvas.print_figure(filename+".eps", dpi=300) #of screen quality


        def _plot2DDataObject(self, dataObject, **kw):
            if self.p is not None:
                if QTVERSION < '4.0.0':
                    self.p.setCaption(dataObject.info['legend'])
                else:
                    self.p.setWindowTitle(dataObject.info['legend'])
            xdata = dataObject.x[0]
            ydata = dataObject.y[0]
            index = None
            if dataObject.m is None:
                mdata = [Numeric.ones(len(zdata)).astype(Numeric.Float)]
            elif len(dataObject.m[0]) > 0:
                if len(dataObject.m[0]) == len(ydata):
                    index = Numeric.nonzero(dataObject.m[0])
                    if not len(index): return
                    xdata = Numeric.take(xdata, index)
                    mdata = Numeric.take(dataObject.m[0], index)
                else:
                    raise "ValueError", "Monitor data length different than counter data"

            logy    = False
            legends = True
            legendslist = None
            loc = (1.01, 0.0)
            labelsep = 0.02
            for key in kw:
                if key.upper() == "LOC":
                    loc = kw[key]
                    continue
                if key.upper() == "LOGY":
                    if kw[key]:logy = True
                    else:logy = False
                    continue
                if key.upper() == "LEGENDS":
                    if kw[key]:legends = True
                    else:      legends = False
                    continue
                if key.upper() == "LEGENDSLIST":
                    legendslist = kw[key]
                    continue
                if key.upper() == "XMIN":
                    if kw[key]:xmin = kw[key]
                    else:      xmin = xdata[0]
                    continue
                if key.upper() == "XMAX":
                    if kw[key]:xmax = kw[key]
                    else:      xmax = xdata[-1]
                    continue
                if key.upper() == "YMIN":
                    if kw[key]:ymin = kw[key]
                    else:      ymin = None
                    continue
                if key.upper() == "YMAX":
                    if kw[key]:ymax = kw[key]
                    else:      ymax = None

            xplot = Numeric.take(xdata, index)
            mplot = Numeric.take(mdata, index)
            icolor = 0
            xlabel = 'X'
            ylabel = 'Y'
            if legendslist is None:
                legendslist = []
                sel = {}
                sel['selection'] = dataObject.info['selection']
                if sel['selection'] is not None:
                    if type(sel['selection']) == type({}):
                        if xlabel is None:
                            if sel['selection'].has_key('x'):
                                ilabel = sel['selection']['x'][0]
                                xlabel = dataObject.info['LabelNames'][ilabel]
                        if ylabel is None:
                            if sel['selection'].has_key('y'):
                                ilabel = sel['selection']['y'][0]
                                ylabel = dataObject.info['LabelNames'][ilabel]
                        if sel['selection'].has_key('y'):
                            for ilabel in  sel['selection']['y']:
                                legendslist.append(dataObject.info['LabelNames'][ilabel])

                else:legends = False
                
            if not legends:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .85, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .85, .8])
            else:
                if logy:
                    ax = self.fig.add_axes([.1, .15, .7, .8])
                else:
                    ax = self.fig.add_axes([.15, .15, .7, .8])
            if logy:
                axfunction = ax.semilogy
            else:
                axfunction = ax.plot
            ax.set_axisbelow(True)

            self.dataObject = dataObject
            


            i = 0
            lshape = "-"
            for y in dataObject.y:
                yplot = Numeric.take(y, index)/mplot
                axfunction(xplot, yplot, ls=lshape, color=colorlist[i],lw=1.5)
                i += 1
                if i == len(colorlist):
                    i = 2   #black and red used only once
                    lshape = ".."

            if legends and len(legendslist):
                if len(legendslist > 14):
                    #loc = (1.01, 0.0)  # outside at plot bottom
                    fontsize = 8
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                else:
                    fontsize = 10
                    fontproperties = FontProperties(size=fontsize)
                    labelsep = 0.015
                    drawframe = True   # with frame
                    
                legend = ax.legend(legendslist,
                                   loc = loc,
                                   fontname = "Times",
                                   prop = fontproperties,
                                   labelsep = labelsep,
                                   pad = 0.15)
                legend.draw_frame(drawframe)
            else:
                fontsize = 10
                fontproperties = FontProperties(size=fontsize)
            if xlabel is not None:ax.set_xlabel(xlabel, prop=fontproperties)
            if ylabel is not None:ax.set_ylabel(ylabel, prop=fontproperties)
            ax.set_xlim(xmin, xmax)
            self.fig.canvas.show()
            canvas.print_figure("test"+".png", dpi=300) #print quality independent
            #canvas.print_figure(filename+".eps", dpi=300) #of screen quality
        
        def _plot3DDataObject(self, dataObject, **kw):
            if self.p is not None:
                if QTVERSION < '4.0.0':
                    self.p.setCaption(dataObject.info['legend'])
                else:
                    self.p.setWindowTitle(dataObject.info['legend'])
            #let's try to generate a plot
            xdata = dataObject.x[0]
            ydata = dataObject.x[1]
            zdata = dataObject.y[0]
            index = None
            if dataObject.m is None:
                mdata = [Numeric.ones(len(zdata)).astype(Numeric.Float)]
            elif len(dataObject.m[0]) > 0:
                if len(dataObject.m[0]) == len(zdata):
                    index = Numeric.nonzero(dataObject.m[0])
                    if not len(index): return
                    xdata = Numeric.take(xdata, index)
                    ydata = Numeric.take(ydata, index)
                    zdata = Numeric.take(zdata, index)
                    mdata = Numeric.take(dataObject.m[0], index)
                else:
                    raise "ValueError", "Monitor data length different than counter data"
            zdata = zdata / mdata
            #self.fig = Figure(figsize=(6,3), dpi = 150)
            #self.fig = Figure()
            #self.fig.canvas = FigureCanvas(self.fig)
            #layout  = qt.QVBoxLayout(self.fig.canvas)
            #toolBar = qt.QWidget(self.fig.canvas)
            #toolBarLayout = qt.QHBoxLayout(toolBar)
            #tb      = qt.QToolButton(toolBar)
            #tb.setIconSet(self.savIcon)
            #self.connect(tb,qt.SIGNAL('clicked()'),self.printps)
            #qt.QToolTip.add(tb,'Prints the Graph')
            #self.toolBarLayout.addWidget(tb)
            #layout.addWidget(toolBar)
            ax = axes3d.Axes3D(self.fig)
            #f = ax.scatter3D
            f = ax.plot3D
            #f = ax.contourf
            for z in dataObject.y:
                if index is None:
                    zdata = z
                else:
                    zdata = Numeric.take(z, index)
                f(Numeric.ravel(xdata),
                         Numeric.ravel(ydata),
                         Numeric.ravel(zdata))
            ax.toolbar = MatplotlibToolbar()
            xlabel = 'X'
            ylabel = 'Y'
            zlabel = 'Z'
            sel = {}
            sel['selection'] = dataObject.info['selection']
            if sel['selection'] is not None:
                if type(sel['selection']) == type({}):
                    if sel['selection'].has_key('x'):
                        #proper scan selection
                        ilabel = sel['selection']['x'][0]
                        xlabel = dataObject.info['LabelNames'][ilabel]
                        ilabel = sel['selection']['x'][1]
                        ylabel = dataObject.info['LabelNames'][ilabel]
                        ilabel = sel['selection']['y'][0]
                        zlabel = dataObject.info['LabelNames'][ilabel]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            self.fig.canvas.show()
            self.dataObject = dataObject

        def resizeEvent( self, e ):
            FigureCanvasQT.resizeEvent( self, e )
            w = e.size().width()
            h = e.size().height()
            if DEBUG: print "FigureCanvasQtAgg.resizeEvent(", w, ",", h, ")"
            dpival = self.figure.dpi.get()
            if not self._fixed_print_size:
                winch = w/dpival
                hinch = h/dpival
                #self.figure.set_figsize_inches( winch, hinch )
                self.figure.set_size_inches( winch, hinch )
            self.draw()

        def sizeHint(self):
            w, h = self.get_width_height()
            return qt.QSize(w, h)

        def motion_notify_event(self, *var):
            print "catching"
            print var
except:
    if DEBUG:print "matplotlib not loaded"



class ScanWindow(qt.QWidget):
    def __init__(self, parent=None, name="Scan Window", specfit=None):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setCaption(name)
        else:
            self.setWindowTitle(name)
        self._initIcons()
        self._build()
        self.fig = None
        self.scanFit = ScanFit.ScanFit(specfit=specfit)
        self.printPreview = PyMcaPrintPreview.PyMcaPrintPreview(modal = 0)
        self.simpleMath = SimpleMath.SimpleMath()
        self.graph.canvas().setMouseTracking(1)
        self.graph.setCanvasBackground(qt.Qt.white)
        self.outputDir = None
        self.outputFilter = None
        self.__toggleCounter = 0
        if QTVERSION < '4.0.0':
            self.connect(self.graph,
                         qt.PYSIGNAL("QtBlissGraphSignal"),
                         self._graphSignalReceived)
            self.connect(self.scanFit,
                         qt.PYSIGNAL('ScanFitSignal') ,
                         self._scanFitSignalReceived)
        else:
            self.connect(self.graph,
                         qt.SIGNAL("QtBlissGraphSignal"),
                         self._graphSignalReceived)
            self.connect(self.scanFit,
                         qt.SIGNAL('ScanFitSignal') ,
                         self._scanFitSignalReceived)
        self.dataObjectsDict = {}
        self.dataObjectsList = []

    def _initIcons(self):
        if QTVERSION < '4.0.0':
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
            self.togglePointsIcon = qt.QIconSet(qt.QPixmap(IconDict["togglepoints"]))
            self.fitIcon	= qt.QIconSet(qt.QPixmap(IconDict["fit"]))
            self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))

            self.averageIcon	= qt.QIconSet(qt.QPixmap(IconDict["average16"]))
            self.deriveIcon	= qt.QIconSet(qt.QPixmap(IconDict["derive"]))
            self.smoothIcon     = qt.QIconSet(qt.QPixmap(IconDict["smooth"]))
            self.swapSignIcon	= qt.QIconSet(qt.QPixmap(IconDict["swapsign"]))
            self.yMinToZeroIcon	= qt.QIconSet(qt.QPixmap(IconDict["ymintozero"]))
            self.subtractIcon	= qt.QIconSet(qt.QPixmap(IconDict["subtract"]))
            
            self.printIcon	= qt.QIconSet(qt.QPixmap(IconDict["fileprint"]))
            self.saveIcon	= qt.QIconSet(qt.QPixmap(IconDict["filesave"]))
        else:
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

    def _build(self):
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self._logY = False
        self._buildToolBar()
        self._buildGraph()
        self.scanWindowInfoWidget = ScanWindowInfoWidget.\
                                        ScanWindowInfoWidget(self)
        self.mainLayout.addWidget(self.scanWindowInfoWidget)


    def _buildToolBar(self):
        self.toolBar = qt.QWidget(self)
        self.toolBarLayout = qt.QHBoxLayout(self.toolBar)
        self.mainLayout.addWidget(self.toolBar)
        #Autoscale
        self._addToolButton(self.zoomResetIcon,
                            self._zoomReset,
                            'Auto-Scale the Graph')


        #y Autoscale
        self.yAutoScaleButton = self._addToolButton(self.yAutoIcon,
                            self._yAutoScaleToggle,
                            'Toggle Autoscale Y Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.yAutoScaleButton.setState(qt.QButton.On)
        else:
            self.yAutoScaleButton.setChecked(True)
            self.yAutoScaleButton.setDown(True)


        #x Autoscale
        self.xAutoScaleButton = self._addToolButton(self.xAutoIcon,
                            self._xAutoScaleToggle,
                            'Toggle Autoscale X Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.xAutoScaleButton.setState(qt.QButton.On)
        else:
            self.xAutoScaleButton.setChecked(True)
            self.xAutoScaleButton.setDown(True)

        #y Logarithmic
        self.yLogButton = self._addToolButton(self.logyIcon,
                            self._toggleLogY,
                            'Toggle Logarithmic Y Axis (On/Off)',
                            toggle = True)
        if QTVERSION < '4.0.0':
            self.yLogButton.setState(qt.QButton.Off)
        else:
            self.yLogButton.setChecked(False)
            self.yLogButton.setDown(False)

        #toggle Points/Lines
        tb = self._addToolButton(self.togglePointsIcon,
                             self._togglePointsSignal,
                             'Toggle Points/Lines')


        #fit icon
        tb = self._addToolButton(self.fitIcon,
                             self._fitIconSignal,
                             'Simple Fit of Active Curve')


        self.newplotIcons = True
        if self.newplotIcons:
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


        self.toolBarLayout.addWidget(HorizontalSpacer(self.toolBar))

        # ---print
        tb = self._addToolButton(self.printIcon,
                                 self.printGraph,
                                 'Prints the Graph')

    def _addToolButton(self, icon, action, tip, toggle=None):
        tb      = qt.QToolButton(self.toolBar)            
        if QTVERSION < '4.0.0':
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
        self.toolBarLayout.addWidget(tb)
        self.connect(tb,qt.SIGNAL('clicked()'), action)
        return tb
        

    def _buildGraph(self):
        self.graph = QtBlissGraph.QtBlissGraph(self, uselegendmenu=True,
                                               legendrename=True,
                                               usecrosscursor=True)
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
        self.graphBottomLayout.addWidget(HorizontalSpacer(self.graphBottom))
        self.mainLayout.addWidget(self.graphBottom)



    def setDispatcher(self, w):
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
            
    def _addSelection(self, selectionlist, replot=True):
        if DEBUG:print "_addSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            if not sel.has_key("scanselection"): continue
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
                    xdata = Numeric.arange(ylen).astype(Numeric.Float)
                else:
                    #nothing to be plot
                    continue
            if dataObject.x is None:
                ylen = len(dataObject.y[0]) 
                if ylen:
                    xdata = Numeric.arange(ylen).astype(Numeric.Float)
                else:
                    #nothing to be plot
                    continue                    
            elif len(dataObject.x) > 1:
                if DEBUG:print "Mesh plots"
                if not MATPLOTLIB:continue
                else:
                    dataObject.info['legend'] = legend
                    FigureCanvas(dataobject=dataObject,toolbar=True)
                    continue
            else:
                xdata = dataObject.x[0]

            sps_source = False
            if sel.has_key('SourceType'):
                if sel['SourceType'] == 'SPS':
                    sps_source = True

            if sps_source:
                ycounter = -1
                dataObject.info['selection'] = copy.deepcopy(sel['selection'])
                for ydata in dataObject.y:
                    ycounter += 1
                    if dataObject.m is None:
                        mdata = [Numeric.ones(len(ydata)).astype(Numeric.Float)]
                    elif len(dataObject.m[0]) > 0:
                        if len(dataObject.m[0]) == len(ydata):
                            index = Numeric.nonzero(dataObject.m[0])
                            if not len(index): continue
                            xdata = Numeric.take(xdata, index)
                            ydata = Numeric.take(ydata, index)
                            mdata = Numeric.take(dataObject.m[0], index)
                            #A priori the graph only knows about plots
                            ydata = ydata/mdata
                        else:
                            raise ValueError, "Monitor data length different than counter data"
                    else:
                        mdata = [Numeric.ones(len(ydata)).astype(Numeric.Float)]
                    ylegend = 'y%d' % ycounter
                    if sel['selection'] is not None:
                        if type(sel['selection']) == type({}):
                            if sel['selection'].has_key('x'):
                                #proper scan selection
                                ilabel = dataObject.info['selection']['y'][ycounter]
                                ylegend = dataObject.info['LabelNames'][ilabel]
                    newLegend = legend + " " + ylegend
                    #here I should check the log or linear status
                    if newLegend not in self.dataObjectsList:
                        self.dataObjectsList.append(newLegend)
                    self.dataObjectsDict[newLegend] = dataObject
                    if self.__toggleCounter in [1, 2]:
                        symbol = 'o'
                    else:
                        symbol = None
                    self.graph.newCurve(newLegend,
                                        x=xdata,
                                        y=ydata,
                                        logfilter=self._logY,
                                        symbol=symbol)
                    if self.scanWindowInfoWidget is not None:
                        activeLegend = self.getActiveCurve()
                        if activeLegend is not None:
                            if activeLegend == newLegend:
                                self.scanWindowInfoWidget.updateFromDataObject\
                                                            (dataObject)
            else:
                #we have to loop for all y values
                ycounter = -1
                for ydata in dataObject.y:
                    ycounter += 1
                    newDataObject   = DataObject.DataObject()
                    newDataObject.info = copy.deepcopy(dataObject.info)
                    if dataObject.m is None:
                        mdata = Numeric.ones(len(ydata)).astype(Numeric.Float)
                    elif len(dataObject.m[0]) > 0:
                        if len(dataObject.m[0]) == len(ydata):
                            index = Numeric.nonzero(dataObject.m[0])
                            if not len(index): continue
                            xdata = Numeric.take(xdata, index)
                            ydata = Numeric.take(ydata, index)
                            mdata = Numeric.take(dataObject.m[0], index)
                            #A priori the graph only knows about plots
                            ydata = ydata/mdata
                        else:
                            raise ValueError, "Monitor data length different than counter data"
                    else:
                        mdata = Numeric.ones(len(ydata)).astype(Numeric.Float)
                    newDataObject.x = [xdata]
                    newDataObject.y = [ydata]
                    newDataObject.m = [mdata]
                    newDataObject.info['selection'] = copy.deepcopy(sel['selection'])
                    ylegend = 'y%d' % ycounter
                    if sel['selection'] is not None:
                        if type(sel['selection']) == type({}):
                            if sel['selection'].has_key('x'):
                                #proper scan selection
                                newDataObject.info['selection']['x'] = sel['selection']['x'] 
                                newDataObject.info['selection']['y'] = [sel['selection']['y'][ycounter]]
                                newDataObject.info['selection']['m'] = sel['selection']['m']
                                ilabel = newDataObject.info['selection']['y'][0]
                                ylegend = newDataObject.info['LabelNames'][ilabel]
                    if dataObject.info.has_key('operations') and len(dataObject.y) == 1:
                        newDataObject.info['legend'] = legend
                        symbol = 'x'
                    else:
                        newDataObject.info['legend'] = legend + " " + ylegend
                        newDataObject.info['selectionlegend'] = legend
                        if self.__toggleCounter in [1, 2]:
                            symbol = 'o'
                        else:
                            symbol = None
                    maptoy2 = False
                    if dataObject.info.has_key('operations'):
                        if dataObject.info['operations'][-1] == 'derivate':
                            maptoy2 = True
                        
                    #here I should check the log or linear status
                    if newDataObject.info['legend'] not in self.dataObjectsList:
                        self.dataObjectsList.append(newDataObject.info['legend'])
                    self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
                    self.graph.newCurve(newDataObject.info['legend'],
                                        x=xdata,
                                        y=ydata,
                                        logfilter=self._logY,
                                        symbol=symbol,
                                        maptoy2=maptoy2)
        if replot:
            self.graph.replot()

            
    def _removeSelection(self, selectionlist):
        if DEBUG:print "_removeSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        removelist = []
        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            if not sel.has_key("scanselection"): continue
            if not sel["scanselection"]:continue
            if len(key.split(".")) > 2: continue

            legend = sel['legend'] #expected form sourcename + scan key
            if type(sel['selection']) == type({}):
                if sel['selection'].has_key('y'):
                    if sel['selection'].has_key('cntlist'):
                        for index in sel['selection']['y']:
                            removelist.append(legend +" "+sel['selection']['cntlist'][index])

        if not len(removelist):return
        self.removeCurves(removelist)

    def removeCurves(self, removelist):
        for legend in removelist:
            if legend in self.dataObjectsList:
                del self.dataObjectsList[self.dataObjectsList.index(legend)]
            if legend in self.dataObjectsDict.keys():
                del self.dataObjectsDict[legend]
            self.graph.delcurve(legend)
        self.graph.replot()

    def _replaceSelection(self, selectionlist):
        if DEBUG:print "_replaceSelection(self, selectionlist)",selectionlist
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        doit = 0
        for sel in sellist:
            if not sel.has_key("scanselection"): continue
            if not sel["scanselection"]:continue
            if len(sel["Key"].split(".")) > 2: continue
            dataObject = sel['dataobject']
            if dataObject.info["selectiontype"] == "1D":
                if hasattr(dataObject, 'y'):
                    doit = 1
                    break
        if not doit:return
        self.graph.clearcurves()
        #self.graph.replot()    #the scan addition will do the replot
        self.dataObjectsDict={}
        self.dataObjectsList=[]
        self._addSelection(selectionlist)

    def _graphSignalReceived(self, ddict):
        if DEBUG:print "_graphSignalReceived", ddict            
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
                if DEBUG:print "unknown legend %s" % legend
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
            self.graph.ylabel(ylabel)
            self.graph.xlabel(xlabel)
            if self.scanWindowInfoWidget is not None:
                self.scanWindowInfoWidget.updateFromDataObject\
                                                            (dataObject)
            return

        if ddict['event'] == "RemoveCurveEvent":
            legend = ddict['legend']
            self.graph.delcurve(legend)
            if self.dataObjectsDict.has_key(legend):
                del self.dataObjectsDict[legend]
                del self.dataObjectsList[self.dataObjectsList.index(legend)]
            self.graph.replot()
            return
        
        if ddict['event'] == "RenameCurveEvent":
            legend = ddict['legend']
            newlegend = ddict['newlegend']
            if self.dataObjectsDict.has_key(legend):
                self.dataObjectsDict[newlegend]= copy.deepcopy(self.dataObjectsDict[legend])
                self.dataObjectsDict[newlegend].info['legend'] = newlegend
                self.dataObjectsList.append(newlegend)
                self.graph.delcurve(legend)
                self.graph.newCurve(self.dataObjectsDict[newlegend].info['legend'],
                                    self.dataObjectsDict[newlegend].x[0],
                                    self.dataObjectsDict[newlegend].y[0],
                                    logfilter=self._logY)
                del self.dataObjectsDict[legend]
                del self.dataObjectsList[self.dataObjectsList.index(legend)]
            self.graph.replot()
            return

    def _scanFitSignalReceived(self, ddict):
        if DEBUG:print "_graphSignalReceived", ddict
        if ddict['event'] == "EstimateFinished":
            return
        if ddict['event'] == "FitFinished":
            newDataObject = self.__fitDataObject

            xplot = self.scanFit.specfit.xdata * 1.0
            yplot = self.scanFit.specfit.gendata(parameters=ddict['data'])
            newDataObject.x = [xplot]
            newDataObject.y = [yplot]
            newDataObject.m = [Numeric.ones(len(yplot)).astype(Numeric.Float)]            

            #here I should check the log or linear status
            self.graph.newcurve(newDataObject.info['legend'],
                                x=xplot,
                                y=yplot,
                                logfilter=self._logY)
            if newDataObject.info['legend'] not in self.dataObjectsList:
                self.dataObjectsList.append(newDataObject.info['legend'])
            self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
        self.graph.replot()

            
    def _zoomReset(self):
        if DEBUG:print "_zoomReset"
        self.graph.zoomReset()
        #self.graph.replot()

    def _yAutoScaleToggle(self):
        if DEBUG:print "_yAutoScaleToggle"
        if self.graph.yAutoScale:
            self.graph.yAutoScale = False
            self.yAutoScaleButton.setDown(False)
            if QTVERSION < '4.0.0':
                self.yButton.setState(qt.QButton.Off)
            else:
                self.yAutoScaleButton.setChecked(False)
            self.graph.setY1AxisLimits(*self.graph.getY1AxisLimits())
            y2limits = self.graph.getY2AxisLimits()
            if y2limits is not None:self.graph.setY2AxisLimits(*y2limits)
        else:
            self.graph.yAutoScale = True
            if QTVERSION < '4.0.0':
                self.yAutoScaleButton.setState(qt.QButton.On)
            else:
                self.yAutoScaleButton.setDown(True)
            self.graph.zoomReset()
                       
    def _xAutoScaleToggle(self):
        if DEBUG:print "_xAutoScaleToggle"
        if self.graph.xAutoScale:
            self.graph.xAutoScale = False
            self.xAutoScaleButton.setDown(False)
            if QTVERSION < '4.0.0':
                self.xAutoScaleButton.setState(qt.QButton.Off)
            else:
                self.xAutoScaleButton.setChecked(False)
            self.graph.setX1AxisLimits(*self.graph.getX1AxisLimits())
        else:
            self.graph.xAutoScale = True
            self.xAutoScaleButton.setDown(True)
            if QTVERSION < '4.0.0':
                self.xAutoScaleButton.setState(qt.QButton.On)
            else:
                self.xAutoScaleButton.setChecked(True)
            self.graph.zoomReset()
                       
    def _toggleLogY(self):
        if DEBUG:print "_toggleLogY"
        if self._logY:
            self._logY = False
        else:
            self._logY = True
        activecurve = self.graph.getactivecurve(justlegend=1)

        self.graph.clearCurves()    
        self.graph.toggleLogY()

        sellist = []
        i = 0
        for key in self.dataObjectsList:
            if key in self.dataObjectsDict.keys():
                sel ={}
                sel['SourceName'] = self.dataObjectsDict[key].info['SourceName']
                sel['dataobject'] = self.dataObjectsDict[key]
                sel['Key'] = self.dataObjectsDict[key].info['Key']
                if self.dataObjectsDict[key].info.has_key('selectionlegend'):
                    sel['legend'] = self.dataObjectsDict[key].info['selectionlegend']
                else:
                    sel['legend'] = self.dataObjectsDict[key].info['legend']
                sel['scanselection'] = True
                sel['selection'] = self.dataObjectsDict[key].info['selection']
                sellist.append(sel)
            i += 1
        self._addSelection(sellist, replot=False)
        self.graph.setactivecurve(activecurve)

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
        self.graph.setActiveCurve(self.graph.getActiveCurve(justlegend=1))
        self.graph.replot()

    def _fitIconSignal(self):
        if DEBUG:print "_fitIconSignal"
        self.__QSimpleOperation("fit")

    def _saveIconSignal(self):
        if DEBUG:print "_saveIconSignal"
        self.__QSimpleOperation("save")
        
    def _averageIconSignal(self):
        if DEBUG:print "_averageIconSignal"
        self.__QSimpleOperation("average")
        
    def _smoothIconSignal(self):
        if DEBUG:print "_smoothIconSignal"
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
            filterlist = ['Specfile MCA  *.mca',
                          'Specfile Scan *.dat',
                          'Specfile MultiScan *.dat',
                          'Raw ASCII *.txt',
                          '","-separated CSV *.csv',
                          '";"-separated CSV *.csv',
                          '"tab"-separated CSV *.csv',
                          'Widget PNG *.png',
                          'Widget JPG *.jpg']
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
        self.outputFilter = str(outfile.selectedFilter())
        filterused = self.outputFilter.split()
        filetype  = filterused[1]
        extension = filterused[2]
        if QTVERSION < '4.0.0':
            outdir=str(outfile.selectedFile())
        else:
            outdir=str(outfile.selectedFiles()[0])
        try:            
            self.outputDir  = os.path.dirname(outdir)
            PyMcaDirs.outputDir = os.path.dirname(outdir)
        except:
            print "setting output directory to default"
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
                    tmpstr += "%.4f " % data[idx+i]
                if idx+16 != length:
                    tmpstr += "\\"
            else:
                for i in range(idx, length):
                    tmpstr += "%.4f " % data[i]
            tmpstr += "\n"
        return tmpstr
        
    def __QSimpleOperation(self, operation):
        try:
            self.__simpleOperation(operation)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("%s" % sys.exc_info()[1])
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()

    def __simpleOperation(self, operation):
        if operation == 'subtract':
            self._subtractOperation()
            return
        if operation != "average":
            #get active curve
            legend = self.getActiveCurve()
            if legend is None:return

            found = False
            for key in self.dataObjectsList:
                if key == legend:
                    found = True
                    break

            if found:
                dataObject = self.dataObjectsDict[legend]
            else:
                print "I should not be here"
                print "active curve =",legend
                print "but legend list = ",self.dataObjectsList
                return
            y = dataObject.y[0]
            if dataObject.x is not None:
                x = dataObject.x[0]
            else:
                x = Numeric.arange(len(y)).astype(Numeric.Float)
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
            for key in self.graph.curves.keys():
                if DEBUG:print "key -> ", key
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
                                if DEBUG: print "LABEL = ", label, "ilabel = ", ilabel
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
                format = filename[-3:].upper()
                pixmap = qt.QPixmap.grabWidget(self.graph)
                if not pixmap.save(filename, format):
                    qt.QMessageBox.critical(self,
                                        "Save Error",
                                        "%s" % sys.exc_info()[1])
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
                        for key in self.graph.curves.keys():
                            if key not in self.dataObjectsDict.keys():
                                continue
                            if key == legend: continue
                            dataObject = self.dataObjectsDict[key]
                            y = dataObject.y[0]
                            if dataObject.x is not None:
                                x = dataObject.x[0]
                            else:
                                x = Numeric.arange(len(y)).astype(Numeric.Float)
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
                    else:
                        csvseparator = "\t"
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
        if newDataObject.info.has_key('selectionlegend'):
            del newDataObject.info['selectionlegend']
        if not newDataObject.info.has_key('operations'):
            newDataObject.info['operations'] = []
        newDataObject.info['operations'].append(operation)

        sel = {}
        sel['SourceType'] = "Operation"
        #get new x and new y
        if operation == "derivate":
            xplot, yplot = self.simpleMath.derivate(x, y)
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
            if dataObject.info.has_key('operations'):
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
            xmin,xmax=self.graph.getX1AxisLimits()
            outputlegend = legend + " Fit"
            for key in self.graph.curves.keys():
                if key == outputlegend:
                    self.graph.delcurve(outputlegend)
                    break
            if outputlegend in self.dataObjectsDict.keys():
                del self.dataObjectsDict[key]
            if outputlegend in self.dataObjectsList:
                i = self.dataObjectsList.index(key)
                del self.dataObjectsList[i]
            self.scanFit.setData(x = x,
                                 y = y,
                                 xmin = xmin,
                                 xmax = xmax,
                                 legend = legend)
            if self.scanFit.isHidden():
                self.scanFit.show()
            if QTVERSION < '4.0.0':
                self.scanFit.raiseW()
            else:
                self.scanFit.raise_()
        else:
            raise "ValueError","Unknown operation %s" % operation
        if operation != "fit":
            newDataObject.x = [xplot]
            newDataObject.y = [yplot]
            newDataObject.m = [Numeric.ones(len(yplot)).astype(Numeric.Float)]

        #and add it to the plot
        if True and (operation != 'fit'):
            sel['dataobject'] = newDataObject
            sel['scanselection'] = True
            sel['selection'] = copy.deepcopy(dataObject.info['selection'])
            sel['selectiontype'] = "1D"
            if operation != 'fit':
                self._addSelection([sel])
            else:
                self.__fitDataObject = newDataObject
                return
        else:
            newDataObject.info['legend'] = outputlegend
            if operation == 'fit':
                self.__fitDataObject = newDataObject
                return

            if newDataObject.info['legend'] not in self.dataObjectsList:
                self.dataObjectsList.append(newDataObject.info['legend'])
            self.dataObjectsDict[newDataObject.info['legend']] = newDataObject
            #here I should check the log or linear status
            self.graph.newcurve(newDataObject.info['legend'],
                                x=xplot,
                                y=yplot,
                                logfilter=self._logY)
        self.graph.replot()

    def getActiveCurve(self):
        #get active curve
        legend = self.graph.getactivecurve(justlegend=1)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please Select an active curve")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.setWindowTitle('Scan window')
                msg.exec_()
        return legend

    def _deriveIconSignal(self):
        if DEBUG:print "_deriveIconSignal"
        self.__QSimpleOperation('derivate')

    def _swapSignIconSignal(self):
        if DEBUG:print "_swapSignIconSignal"
        self.__QSimpleOperation('swapsign')

    def _yMinToZeroIconSignal(self):
        if DEBUG:print "_yMinToZeroIconSignal"
        self.__QSimpleOperation('forceymintozero')

    def _subtractIconSignal(self):
        if DEBUG:print "_subtractIconSignal"
        self.__QSimpleOperation('subtract')

    def _subtractOperation(self):
        #identical to twice the average with the negative active curve
        #get active curve
        legend = self.getActiveCurve()
        if legend is None:return

        found = False
        for key in self.dataObjectsList:
            if key == legend:
                found = True
                break

        if found:
            dataObject = self.dataObjectsDict[legend]
        else:
            print "I should not be here"
            print "active curve =",legend
            print "but legend list = ",self.dataObjectsList
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
        for key in self.graph.curves.keys():
            legend = ""
            x = [xActive]
            y = [-yActive]
            if DEBUG:print "key -> ", key
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
                            if DEBUG: print "LABEL = ", label, "ilabel = ", ilabel
                    y.append(self.dataObjectsDict[key].y[ilabel])
                outputlegend = "(%s - %s)" %  (key, yActiveLegend)
                ndata += 1
                xplot, yplot = self.simpleMath.average(x, y)
                yplot *= 2
                #create the output data object
                newDataObject = DataObject.DataObject()
                newDataObject.data = None
                newDataObject.info.update(self.dataObjectsDict[key].info)
                if not newDataObject.info.has_key('operations'):
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
            oldlist = self.dataObjectsDict.keys()
            self._addSelection(sel_list)
            self.removeCurves(oldlist)
        

    def newCurve(self, x, y, legend=None, xlabel=None, ylabel=None, replace=False):
        if legend is None:
            legend = "Unnamed curve 1.1"
        if xlabel is None:
            xlabel = "X"
        if ylabel is None:
            ylabel = "Y"
        newDataObject = DataObject.DataObject()
        newDataObject.x = [x]
        newDataObject.y = [y]
        newDataObject.m = None
        newDataObject.info['legend'] = legend
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
            self._addSelection(sel_list)

    def printGraph(self):
        #temporary print
        pixmap = qt.QPixmap.grabWidget(self.graph)
        self.printPreview.addPixmap(pixmap)
        if self.printPreview.isHidden():
            self.printPreview.show()
        if QTVERSION < '4.0.0':
            self.printPreview.raiseW()
        else:
            self.printPreview.raise_()
        
    def printGraphWork(self):
        if DEBUG:print "printGraphSignal"

        #get graph curvelist (not dataObjects list)??
        legendslist = []
        xdata      = []
        ydata      = []
        xlabel = self.graph.xlabel()
        ylabel = self.graph.ylabel()
        xmin, xmax   = self.graph.getx1axislimits()
        ymin, ymax   = self.graph.gety1axislimits()
        for legend in self.dataObjectsList:
            if legend in self.graph.curves.keys():
                legendslist.append(legend)
                xdata.append(self.dataObjectsDict[legend].x[0])
                ydata.append(self.dataObjectsDict[legend].y[0])
        if len(legendslist):
            FigureCanvas(xlist = xdata, ylist = ydata,
                         xmin = xmin, xmax = xmax,
                         ymin = ymin, ymax = ymax,
                         xlabel = xlabel, ylabel = ylabel,
                         legendslist = legendslist)


if MATPLOTLIB:
    class QMatplotlibCanvas(FigureCanvasQTAgg):
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
        
    

    class MatplotlibToolbar:
        def __init__(self):
            self._active = None
        
        def set_message(self, msg):
            if DEBUG:print "msg = ",msg

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))

def test():
    w = ScanWindow()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    if QTVERSION < '4.0.0':
        app.setMainWidget(w)
        app.exec_loop()
    else:
        app.exec_()


if __name__ == "__main__":
    test()
