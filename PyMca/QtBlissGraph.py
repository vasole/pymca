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
# The Python version of qwt-*/examples/simple_plot/simple.cpp
#from QMdiApp.mdi_icons import IconDict
import copy
import sys
import string
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        from PyQt4 import Qwt5 as qwt
    except:
        import qt
        try:
            import Qwt5 as qwt
        except:
            try:
                import Qwt4 as qwt 
            except:
                import qwt
else:
    import qt
    try:
        import Qwt5 as qwt
    except:
        try:
            import Qwt4 as qwt 
        except:
            import qwt

QTVERSION = qt.qVersion()
    
if qwt.QWT_VERSION_STR[0] > '4':
    Qwt = qwt
    qwt.QwtPlotMappedItem = Qwt.QwtPlotItem
    qwt.QwtCurve          = Qwt.QwtPlotCurve
    QWTVERSION4 = False
else:
    QWTVERSION4 = True
try:
    from Icons import IconDict
except:
    pass
import time
from Numeric import *
DEBUG = 0
USE_SPS_LUT = 1
if USE_SPS_LUT:
    try:
        import spslut
        COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                        spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]
    except:
        USE_SPS_LUT = 0

#import arrayfns
if QTVERSION < '3.0.0':
    qt.QCursor.ArrowCursor   = qt.ArrowCursor
    qt.QCursor.UpArrowCursor = qt.UpArrowCursor
    qt.QCursor.WaitCursor    = qt.WaitCursor
    qt.QCursor.CrossCursor   = qt.CrossCursor
    qt.QCursor.SizeVerCursor = qt.SizeVerCursor
    qt.QCursor.SizeHorCursor = qt.SizeHorCursor
    qt.QCursor.PointingHandCursor = qt.ArrowCursor

elif QTVERSION > '4.0.0':
    qt.QCursor.ArrowCursor   = qt.Qt.ArrowCursor
    qt.QCursor.UpArrowCursor = qt.Qt.UpArrowCursor
    qt.QCursor.WaitCursor    = qt.Qt.WaitCursor
    qt.QCursor.CrossCursor   = qt.Qt.CrossCursor
    qt.QCursor.SizeVerCursor = qt.Qt.SizeVerCursor
    qt.QCursor.SizeHorCursor = qt.Qt.SizeHorCursor
    qt.QCursor.PointingHandCursor = qt.Qt.PointingHandCursor

if not USE_SPS_LUT:
    # from scipy.pilutil
    def bytescale(data, cmin=None, cmax=None, high=255, low=0):
        if data.typecode == UInt8:
            return data
        high = high - low
        if cmin is None:
            cmin = min(ravel(data))
        if cmax is None:
            cmax = max(ravel(data))
        scale = high *1.0 / (cmax-cmin or 1)
        bytedata = ((data*1.0-cmin)*scale + 0.4999).astype(UInt8)
        return bytedata + asarray(low).astype(UInt8)

    def fuzzypalette():
        #calculare bit mapdef
        # In order to make more easy the reckonings, all the calculations 
        # will be done in normalized values and later we translate them to
        # practical values (here we describe the limits of these real values)
        init_num_indexed_color = 0
        end_num_indexed_color = 255
        total_indexed_colors = end_num_indexed_color - init_num_indexed_color 
        init_range_color = 0
        end_range_color = 255
        total_range_color = end_range_color - init_range_color


        # We will deal with the colors in this schem like if they were 
        # fuzzy variables, so we will represent them like trapezoids
        # that are centered in the crisp value
        fuzziness = 0.75
        slope_factor = 4 # We always supposse that 2*(1.0/slope)<fuzziness  


        # Settiings for palette
        R_center = 0.75
        G_center = 0.5
        B_center = 0.25
    
        myColorMap = []
        for i in range(init_num_indexed_color, end_num_indexed_color+1):
            normalized_i = (i*1.0)/total_indexed_colors
            R_value = init_range_color + int(trapezoid_fuction(R_center,
                    fuzziness,slope_factor,normalized_i) * total_range_color)
            G_value = init_range_color + int(trapezoid_fuction(G_center,
                    fuzziness,slope_factor,normalized_i) * total_range_color)
            B_value = init_range_color + int(trapezoid_fuction(B_center,
                    fuzziness,slope_factor,normalized_i) * total_range_color)
            myColorMap.append(qt.qRgb(R_value, G_value, B_value))
        return myColorMap

    def trapezoid_fuction(center, fuzziness, slope, point_to_calculate):
        init_point = center - (fuzziness/2.0)
        end_point = center + (fuzziness/2.0)
        triangular_margin = 1.0/slope # We supposse that the highness
                                  # is normalized to 1
        if (point_to_calculate < init_point):
            value = 0
        elif (point_to_calculate < (init_point + triangular_margin)):   
            value = (point_to_calculate - init_point) * slope
        elif (point_to_calculate < (end_point - triangular_margin)):
            value = 1
        elif (point_to_calculate < end_point):
            value = (end_point - point_to_calculate) * slope
        else:   
            value = 0
        return value

    
class QtBlissGraphWindow(qt.QMainWindow):
    if qt.qVersion() < '4.0.0':
        def __init__(self, parent=None, name="Graph", fl=qt.Qt.WDestructiveClose):
            qt.QMainWindow.__init__(self, parent, name, fl)
            self.name= name
            self.setCaption(self.name)
            self.container= QtBlissGraphContainer(self)
            self.view  =self.container.graph
            self.graph =self.container.graph 
            self.view.DeleteAllRoi= None
            self.view.DeleteAllPeak= None
            self.view.Refresh= None
            self.view.ToggleLogXs= None

            self.setCentralWidget(self.container)
            self.initIcons()
            self.initToolBar()

            self.resize(400,300)
    else:
        def __init__(self, parent=None, name="Graph", fl=0):
            qt.QMainWindow.__init__(self, parent, name, fl)
            self.name= name
            self.setCaption(self.name)
            self.container= QtBlissGraphContainer(self)
            self.view  =self.container.graph
            self.graph =self.container.graph 
            self.view.DeleteAllRoi= None
            self.view.DeleteAllPeak= None
            self.view.Refresh= None
            self.view.ToggleLogXs= None

            self.setCentralWidget(self.container)
            self.initIcons()
            self.initToolBar()
            self.resize(400,300)

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
        self.searchIcon	= qt.QIconSet(qt.QPixmap(IconDict["peaksearch"]))

    def initToolBar(self):
        toolbar= qt.QToolBar(self, "Graph Commands")
        """
        self.normalButton= qt.QToolButton(self.normalIcon, "Normal Mode", qt.QString.null,
                                self.__toggleNormal, toolbar, "Normal Mode")
        self.normalButton.setToggleButton(1)
        self.zoomButton= qt.QToolButton(self.zoomIcon, "Zoom Mode", qt.QString.null,
                                self.__toggleZoom, toolbar, "Zoom Mode")
        self.zoomButton.setToggleButton(1)
        self.roiButton= qt.QToolButton(self.roiIcon, "Roi Mode", qt.QString.null,
                                self.__toggleRoi, toolbar, "Roi Mode")
        self.roiButton.setToggleButton(1)
        self.peakButton= qt.QToolButton(self.peakIcon, "Peak Mode", qt.QString.null,
                                self.__togglePeak, toolbar, "Peak Mode")
        self.peakButton.setToggleButton(1)
        """
        toolbar.addSeparator()
        tb= qt.QToolButton(self.zoomResetIcon, "Reset Zoom", qt.QString.null,
                        self.view.ResetZoom, toolbar, "Reset Zoom")
        """
        tb= qt.QToolButton(self.roiResetIcon, "Remove ROIs",qt.QString.null,
                        self.view.DeleteAllRoi, toolbar, "Remove ROIs")
        tb= qt.QToolButton(self.peakResetIcon, "Remove Peaks", qt.QString.null,
                        self.view.DeleteAllPeak, toolbar, "Remove Peaks")
        toolbar.addSeparator()
        
        tb= qt.QToolButton(self.refreshIcon, "Refresh Graph", qt.QString.null,
                        self.view.Refresh, toolbar, "Refresh Graph")
        self.logxButton= qt.QToolButton(self.logxIcon, "Log X Axis", qt.QString.null,
                        self.view.ToggleLogX, toolbar, "Log X Axis")
        self.logxButton.setToggleButton(1)
        """
        self.logyButton= qt.QToolButton(self.logyIcon, "Log Y Axis", qt.QString.null,
                        self.view.ToggleLogY, toolbar, "Log Y Axis")
        self.logyButton.setToggleButton(1)
        toolbar.addSeparator()
        tb= qt.QToolButton(self.fitIcon, "Fit Active Spectrum", qt.QString.null,
                        self.fitSpectrum, toolbar, "Fit Active Spectrum")
        tb= qt.QToolButton(self.searchIcon, "Peak Search on Active Spectrum", qt.QString.null,
                        self.peakSearch, toolbar, "Peak Search")
    
    def __toggleNormal(self):
        pass
    def __toggleZoom(self):
        pass
    def __toggleRoi(self):
        pass
    def __togglePeak(self):
        pass

    def fitSpectrum(self):
        pass
    def peakSearch(self):
        pass


class QtBlissGraphContainer(qt.QWidget):
    def __init__(self, parent = None, name = 'Container', fl = 0, *args, **kw):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
        else:
            qt.QWidget.__init__(self,parent)
        if 0:
            self.layout=qt.QVBoxLayout(self)
        else:
            self.layout=qt.QHBoxLayout(self)
        if qt.qVersion() < '4.0.0':self.layout.setAutoAdd(1)
        self.graph = QtBlissGraph(self, *args, **kw)
        if qt.qVersion() > '4.0.0':self.layout.addWidget(self.graph)

GRAPHEVENT = qt.QEvent.User


if qt.qVersion() < '4.0.0':
    class GraphCustomEvent(qt.QCustomEvent):
        def __init__(self,dict=None):
            if dict is None: dict = {}
            qt.QCustomEvent.__init__(self,GRAPHEVENT)
            self.dict = dict
else:
    class GraphCustomEvent(qt.QEvent):
        def __init__(self,dict=None):
            if dict is None: dict = {}
            self.dict = dict
            qt.QEvent.__init__(self,GRAPHEVENT)

    class CrappyMouseEvent:
        def __init__(self, types, x, y, button):
            self.info = (types, x, y, button)

        def type(self):
            return self.info[0]

        def pos(self):
            return self

        def x(self):
            return self.info[1]

        def y(self):
            return self.info[2]

        def button(self):
            return self.info[3]

class QtBlissGraph(qwt.QwtPlot):
    def __init__(self, *args,**kw):
        apply(qwt.QwtPlot.__init__, (self,) + args)
        #font = self.parent().font()
        #font.setFamily(qt.QString('Helvetica'))
        #self.setFont(font)
        self.plotLayout().setMargin(0)
        self.plotLayout().setCanvasMargin(0)
        if not QWTVERSION4:
            self.setAutoReplot(False)
            if QTVERSION > '4.0.0':
                self.plotLayout().setAlignCanvasToScales(True)
        #self.setCanvasLineWidth(0)
        #self.setTitle('QtBlissGraph')
        self.setTitle('   ')
        self.setAxisTitle(qwt.QwtPlot.xBottom,"x")        
        self.setAxisTitle(qwt.QwtPlot.yLeft,"y")
        #On windows there are problems with the first plot when passing to log
        self.__firstplot = 1
        self.__timer      = time.time()
        #legend
        if qwt.QWT_VERSION_STR[0] < '5':
            self.enableLegend(1)
            # font needed for Q.. version
            if sys.platform == "win32":
                if qt.qVersion() > "3.3.2":  #I have that commercial version
                    oqtFont = qt.QFont( 'MS Sans Serif', 8)
                    if oqtFont.exactMatch():
                       self.setLegendFont( oqtFont )
                       self.setLegendFrameStyle( 2 );
            self.setAutoLegend(1)
            
            if kw.has_key('LegendPos'):
                self.setLegendPos(kw['LegendPos'])
            else:
                self.setLegendPos(qwt.Qwt.Bottom)
        else:
            legend = Qwt.QwtLegend()
            legend.setItemMode(Qwt.QwtLegend.ClickableItem)
            if kw.has_key('LegendPos'):
                self.insertLegend(legend, kw['LegendPos'])
            else:
                self.insertLegend(legend, Qwt.QwtPlot.BottomLegend)            
        self.__uselegendmenu = 0
        if kw.has_key('uselegendmenu'):
            if kw['uselegendmenu']:
                self.__uselegendmenu = 1

        if kw.has_key('keepimageratio'):
            self.__keepimageratio = 1
        else:
            self.__keepimageratio = 0
        self.legendmenu    = None
        if qwt.QWT_VERSION_STR[0] < '5':
            self.setLegendFrameStyle(qt.QFrame.Box | qt.QFrame.Sunken)
        self.enableZoom(True)
        self._zooming = 0
        self._selecting = 0
        self.__zoomback   = 1
        self.__markermode = 0
        self.__markermoving= 0
        self._xImageMirror = 0
        self._yImageMirror = 1
        self.markersdict = {}       
        self.curves={}
        self.curveslist=[]
        self.__activecurves = []
        self.__oldcursor = self.canvas().cursor().shape()
        #colors
        self.colorslist=['blue','red','green','pink','brown',
                        'orange','violet','yellow']
        self.colors ={}
        if QTVERSION < '4.0.0':
            self.colors['blue']     = qt.Qt.blue 
            self.colors['red']      = qt.Qt.red 
            self.colors['yellow']   = qt.Qt.yellow 
            self.colors['black']    = qt.Qt.black
            self.colors['green']    = qt.Qt.green
            self.colors['white']    = qt.Qt.white
        else:
            self.colors['blue']     = qt.QColor(0, 0, 0xFF)
            self.colors['red']      = qt.QColor(0xFF, 0, 0)
            self.colors['yellow']   = qt.QColor(0xFF, 0xFF, 0)
            self.colors['black']    = qt.QColor(0, 0, 0)
            self.colors['green']    = qt.QColor(0, 0xFF, 0)
            self.colors['white']    = qt.QColor(0xFF, 0xFF, 0xFF)
        # added colors #
        self.colors['pink']     = qt.QColor(255,20,147)
        self.colors['brown']    = qt.QColor(165,42,42)
        self.colors['orange']   = qt.QColor(255,165,0)
        self.colors['violet']   = qt.QColor(148,0,211)
        
        
        self.__activecolor = self.colors['black']
        #self.__activecolor = self.colors['blue']
      
        #styles
        ##self.styleslist=['line','spline','steps','sticks','none']
        self.styleslist=['line','steps','sticks','none']
        self.styles={}
        self.styles['line'] = qwt.QwtCurve.Lines
        ##self.styles['spline'] = qwt.QwtCurve.Spline
        self.styles['steps'] = qwt.QwtCurve.Steps
        self.styles['sticks'] = qwt.QwtCurve.Sticks
        
        self.styles['none'] = qwt.QwtCurve.NoCurve
        #symbols
        self.symbolslist=['cross','ellipse','xcross','none']
        self.symbols={}
        self.symbols['cross'] = qwt.QwtSymbol.Cross
        self.symbols['ellipse'] = qwt.QwtSymbol.Ellipse
        self.symbols['xcross'] = qwt.QwtSymbol.XCross
        if hasattr(qwt.QwtSymbol,"None"):
            self.symbols['none'] = qwt.QwtSymbol.None
        else:
            #latest API
            self.symbols['none'] = qwt.QwtSymbol.NoSymbol
        #types
        self.linetypeslist=['solid','dot','dash','dashdot','dashdotdot']
        self.linetypes={}
        if qt.qVersion() < '4.0.0':
            self.linetypes['solid']     = qt.Qt.SolidLine
            self.linetypes['dash']      = qt.Qt.DashLine
            self.linetypes['dashdot']   = qt.Qt.DashDotLine
            self.linetypes['dashdotdot']= qt.Qt.DashDotDotLine
            self.linetypes['dot']       = qt.Qt.DotLine
        else:
            self.linetypes['solid']     = qt.Qt.SolidLine
            self.linetypes['dash']      = qt.Qt.DashLine
            self.linetypes['dashdot']   = qt.Qt.DashDotLine
            self.linetypes['dashdotdot']= qt.Qt.DashDotDotLine
            self.linetypes['dot']       = qt.Qt.DotLine        
        #color counter
        self.color   = 0
        #symbol counter
        self.symbol  = 0
        #line type
        self.linetype= 0
        #width
        self.linewidth     = 2
        self.__activelinewidth = 2
        
        #perhaps this should be somewhere else
        self.plotImage=None        
        self.onlyoneactive = 1
        self.xAutoScale = True
        self.yAutoScale = True
        #connections and functions
        self.zoomStack  = []
        self.zoomStack2 = []
        if qwt.QWT_VERSION_STR[0] < '5':
            self.connect(self, qt.SIGNAL("legendClicked(long)"), self._legendClicked)
        else:
            self.connect(self, qt.SIGNAL("legendClicked(QwtPlotItem *)"), self._legendClicked)
            
        #replot
        self.__logy1 = 0
        self.__logy2 = 0
        #self.plotimage()
        self.replot()

        #if QTVERSION < '4.0.0':
        if QWTVERSION4:
            self.connect(self,
                     qt.SIGNAL('plotMouseMoved(const QMouseEvent&)'),
                     self.onMouseMoved)
            self.connect(self,
                     qt.SIGNAL('plotMousePressed(const QMouseEvent&)'),
                     self.onMousePressed)
            self.connect(self,
                     qt.SIGNAL('plotMouseReleased(const QMouseEvent&)'),
                     self.onMouseReleased)
            self.picker = None
        else:
            if QTVERSION < '4.0.0':
                self.picker = MyPicker(self.canvas())
                self.connect(self.picker,
                     qt.PYSIGNAL('MouseMoved(const QMouseEvent&)'),
                     self.onMouseMoved)
                self.connect(self.picker,
                     qt.PYSIGNAL('MousePressed(const QMouseEvent&)'),
                     self.onMousePressed)
                self.connect(self.picker,
                     qt.PYSIGNAL('MouseReleased(const QMouseEvent&)'),
                     self.onMouseReleased)
                self.picker.setSelectionFlags(Qwt.QwtPicker.DragSelection  |
                                              Qwt.QwtPicker.RectSelection)

                self.picker.setRubberBand(Qwt.QwtPicker.NoRubberBand)
                self.picker.setRubberBandPen(qt.QPen(qt.Qt.green))
                self.picker.setEnabled(1)
            elif 1:
                if sys.platform == "win32":
                    #self.canvas().setPaintAttribute(Qwt.QwtPlotCanvas.PaintPacked,
                    #                                False)
                    #this get rid of ugly black rectangles during painting
                    self.canvas().setAttribute(qt.Qt.WA_PaintOnScreen, False)
                self.picker = MyPicker(self.canvas())
                self.connect(self.picker,
                     qt.SIGNAL('MouseMoved(const QMouseEvent&)'),
                     self.onMouseMoved)
                self.connect(self.picker,
                     qt.SIGNAL('MousePressed(const QMouseEvent&)'),
                     self.onMousePressed)
                self.connect(self.picker,
                     qt.SIGNAL('MouseReleased(const QMouseEvent&)'),
                     self.onMouseReleased)
                self.picker.setSelectionFlags(Qwt.QwtPicker.DragSelection  |
                                              Qwt.QwtPicker.RectSelection)

                self.picker.setRubberBand(Qwt.QwtPicker.NoRubberBand)
                self.picker.setRubberBandPen(qt.QPen(qt.Qt.green))
                self.picker.setEnabled(1)
            else:
                if 0:
                    self.picker = Qwt.QwtPicker(self.canvas())
                    self.picker.setTrackerMode(Qwt.QwtPicker.ActiveOnly)
                    self.connect(self.picker,
                                 qt.SIGNAL('selected(const selectedPoints &)'),
                                 self.testmethod)
                    #click selection
                    self.picker.setSelectionFlags(Qwt.QwtPicker.PointSelection| \
                                                  Qwt.QwtPicker.ClickSelection)
                    self.picker.setRubberBandPen(qt.QPen(qt.Qt.green))
                else:
                    self.picker = Qwt.QwtPlotPicker(self.canvas())
                    self.picker.setAxis(Qwt.QwtPlot.xBottom,
                                    Qwt.QwtPlot.yLeft)
                    #,
                    if 0:
                        self.picker.setSelectionFlags(Qwt.QwtPicker.PointSelection|
                                                      Qwt.QwtPicker.DragSelection)
                    else:
                        self.picker.setSelectionFlags(Qwt.QwtPicker.DragSelection  |
                                                      Qwt.QwtPicker.RectSelection)
                    #this shows x and y coordenates
                    self.picker.setTrackerMode(Qwt.QwtPicker.AlwaysOn)
                    #,
                    #                    self.canvas())
                    self.picker.setRubberBand(Qwt.QwtPicker.RectRubberBand)
                    self.picker.setRubberBandPen(qt.QPen(qt.Qt.green))
                    self.picker.setEnabled(1)
                    #only these two signals work
                    self.connect(
                                self.picker,
                                qt.SIGNAL('appended(const QPoint &)'),
                                self.pickerAppendedSlot)
                    self.connect(
                                self.picker,
                                qt.SIGNAL('moved(const QPoint &)'),
                                self.pickerMovedSlot)
                    
    def pickerMovedSlot(self, *var):
        print "Moved"
        print "point",var[0].x(),var[0].y()

    def pickerAppendedSlot(self, *var):
        print "Appended"
        print "point",var[0].x(),var[0].y()


    def setx1timescale(self,on=0):
        if on:
            self.setAxisScaleDraw(qwt.QwtPlot.xBottom,TimeScaleDraw())
        else:
            self.setAxisScaleDraw(qwt.QwtPlot.xBottom,qwt.QwtScaleDraw())

    def ToggleLogY(self):
        if DEBUG:print "use toggleLogY"
        return self.toggleLogY()

    def toggleLogY(self):
        if self.__logy1: 
            #get the current limits
            if QWTVERSION4:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).d1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).d2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).d1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).d2()
            else:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
            self.__logy1 =0
            self.__logy2 =0
            if QWTVERSION4:
                self.setAxisOptions(qwt.QwtPlot.yLeft, qwt.QwtAutoScale.None)
                self.setAxisMargins(qwt.QwtPlot.yLeft,0,0)
                self.setAxisOptions(qwt.QwtPlot.yRight, qwt.QwtAutoScale.None)
            if self.yAutoScale:
                self.setAxisAutoScale(qwt.QwtPlot.yLeft)
                self.setAxisAutoScale(qwt.QwtPlot.yRight)
            if QWTVERSION4:
                self.setAxisMargins(qwt.QwtPlot.yRight,0,0)
            else:    
                self.setAxisScaleEngine(Qwt.QwtPlot.yLeft, Qwt.QwtLinearScaleEngine())
                self.setAxisScaleEngine(Qwt.QwtPlot.yRight, Qwt.QwtLinearScaleEngine())
        else:
            #get current margins
            #get the current limits
            if QWTVERSION4:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).d1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).d2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).d1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).d2()
            else:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
            self.__logy1 =1
            self.__logy2 =1
            self.enableAxis(qwt.QwtPlot.yLeft)
            self.enableAxis(qwt.QwtPlot.yRight)
            if self.yAutoScale:
                self.setAxisAutoScale(qwt.QwtPlot.yLeft)
                self.setAxisAutoScale(qwt.QwtPlot.yRight)
            if QWTVERSION4:
                self.setAxisOptions(qwt.QwtPlot.yLeft ,qwt.QwtAutoScale.Logarithmic)
                self.setAxisOptions(qwt.QwtPlot.yRight,qwt.QwtAutoScale.Logarithmic)
            else:
                self.setAxisScaleEngine(Qwt.QwtPlot.yLeft, Qwt.QwtLog10ScaleEngine())
                self.setAxisScaleEngine(Qwt.QwtPlot.yRight, Qwt.QwtLog10ScaleEngine())
        self.replot()
        if self.checky1scale() or self.checky2scale():
            self.replot()
            
    def zoomReset(self):
        if self.yAutoScale:
            self.setAxisAutoScale(qwt.QwtPlot.yLeft) 
            self.setAxisAutoScale(qwt.QwtPlot.yRight)
        
        if self.xAutoScale:
            self.setAxisAutoScale(qwt.QwtPlot.xTop) 
            self.setAxisAutoScale(qwt.QwtPlot.xBottom)

        #the above part is enough is there are some curves
        #but is not enough is there is just an image
        if len(self.zoomStack):
            autoreplot = self.autoReplot()
            self.setAutoReplot(False)
            if len(self.zoomStack):
                xmin, xmax, ymin, ymax = self.zoomStack[0]
                if self.xAutoScale:
                    self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)
                if self.yAutoScale:
                    if QWTVERSION4:
                        self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)
                    else:
                        self.setAxisScale(qwt.QwtPlot.yLeft, ymax, ymin)
                xmin, xmax, ymin, ymax = self.zoomStack2[0]
                if self.axisEnabled(qwt.QwtPlot.xTop):
                    if self.xAutoScale:
                        self.setAxisScale(qwt.QwtPlot.xTop, xmin, xmax)
                if self.axisEnabled(qwt.QwtPlot.yRight):
                    if self.yAutoScale:
                        if QWTVERSION4:
                            self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
                        else:
                            self.setAxisScale(qwt.QwtPlot.yRight, ymax, ymin)                            
            self.setAutoReplot(autoreplot)

        self.zoomStack =  []
        self.zoomStack2 = []

        self.replot()
        if self.checky1scale() or self.checky2scale():
            self.replot()
            
    def ResetZoom(self):
        if DEBUG:print "ResetZoom kept for compatibility, use zoomReset"
        self.zoomReset()

    def enableZoom(self, flag=True):
        self.__zoomEnabled = flag
        if flag:self._selecting = False

    def enableSelection(self, flag=True):
        self._selecting    = flag
        if flag: self.__zoomEnabled = False

    def checky2scale(self):
        flag = 0
        if len(self.curves.keys()):
            flag = 1
            for key in self.curves.keys():
                if self.curves[key]['maptoy2']:
                    flag = 0
                    break
            if flag:
                if QWTVERSION4:
                    ymin = self.axisScale(qwt.QwtPlot.yLeft).lBound()
                    ymax = self.axisScale(qwt.QwtPlot.yLeft).hBound()
                    self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
                else:
                    if 1:
                        ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
                        ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                        self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
        return flag

    def checky1scale(self):
        flag = 0
        if len(self.curves.keys()):
            flag = 1
            for key in self.curves.keys():
                if not self.curves[key]['maptoy2']:
                    flag = 0
                    break
            if flag:
                if QWTVERSION4:
                    ymin = self.axisScale(qwt.QwtPlot.yRight).lBound()
                    ymax = self.axisScale(qwt.QwtPlot.yRight).hBound()
                    self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)
                else:
                    if 1:
                        ymax = self.canvasMap(qwt.QwtPlot.yRight).s2()
                        ymin = self.canvasMap(qwt.QwtPlot.yRight).s1()
                        self.setAxisScale(qwt.QwtPlot.yLeft, ymax, ymin)
        return flag

    def pixmapPlot(self, pixmap, size, xmirror = None, ymirror = None):
        if xmirror is None: xmirror = self._xImageMirror
        if ymirror is None: ymirror = self._yImageMirror
        if self.plotImage is None:
            if QWTVERSION4:
                self.plotImage=QwtPlotImage(self)
            else:
                self.plotImage=Qwt5PlotImage(self)
        self.plotImage.setPixmap(pixmap, size,
                                 xmirror = xmirror, ymirror=ymirror)
        
    def imagePlot(self, data=None, colormap=None, 
                xmirror=None, ymirror=None):
        if data is not None:           
            #get the current limits
            if QWTVERSION4:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).d1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).d2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).d1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).d2()
            else:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
            if self.plotImage is None:
                if QWTVERSION4:
                    self.plotImage=QwtPlotImage(self)
                else:
                    self.plotImage=Qwt5PlotImage(self)
            if xmirror is None: xmirror = self._xImageMirror        
            if ymirror is None: ymirror = self._yImageMirror
            if len(self.curveslist):
                self.plotImage.setData(data,(xmin,xmax),
                                   (ymin, ymax),
                                   colormap=colormap,
                                   xmirror = xmirror,
                                   ymirror = ymirror)
            else:
                #let the scale be governed by the image
                self.plotImage.setData(data,None,None,
                                   colormap=colormap,
                                   xmirror = xmirror,
                                   ymirror = ymirror)
                self.replot()
                if QWTVERSION4:
                    xmin = self.canvasMap(qwt.QwtPlot.xBottom).d1()
                    xmax = self.canvasMap(qwt.QwtPlot.xBottom).d2()
                    ymin = self.canvasMap(qwt.QwtPlot.yLeft).d1()
                    ymax = self.canvasMap(qwt.QwtPlot.yLeft).d2()
                else:
                    xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
                    xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
                    ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                    ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
            self.imageratio = (ymax - ymin)/(xmax - xmin)


    def plotimage(self, *var, **kw):
        if DEBUG:print "plotimage obsolete, use imagePlot instead"
        return self.imagePlot(*var, **kw)

    def drawCanvasItems(self, painter, rectangle, maps, filter):
        if DEBUG:
            print "drawCanvasItems"
        if self.plotImage is not None:
            self.plotImage.drawImage(
                    #painter, maps[qwt.QwtPlot.xBottom], maps[qwt.QwtPlot.yLeft],lgy1=self.__logy1)
                    painter, maps[qwt.QwtPlot.xBottom], maps[qwt.QwtPlot.yLeft])
        qwt.QwtPlot.drawCanvasItems(self, painter, rectangle, maps, filter)
    
    def removeImage(self, legend=None):
        if not QWTVERSION4:
            self.plotImage.detach()
        self.plotImage = None
    
    def onMouseMoved(self,event):
        #method to be overwritten
        if DEBUG:
            print "onMouseMoved, event = ",event
        xpixel = event.pos().x()
        ypixel = event.pos().y()
        x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
        y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
        if self.__markermode:
            if self.__markermoving in self.markersdict.keys():
                xmarker = self.markersdict[self.__markermoving]['xmarker']
                if xmarker:
                    if QWTVERSION4:
                        self.setMarkerXPos(self.__markermoving, x)
                    else:
                        self.setMarkerXPos(self.markersdict[self.__markermoving]['marker'], x)
                else:
                    if QWTVERSION4:
                        self.setMarkerYPos(self.__markermoving, y)
                    else:
                        self.setMarkerYPos(self.markersdict[self.__markermoving]['marker'], y)
                ddict ={}
                ddict['event']    = "markerMoving"
                ddict['marker']   = self.__markermoving
                ddict['x']        = x
                ddict['y']        = y
                if qt.qVersion() < '4.0.0':
                    self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(ddict,))
                else:
                    self.emit(qt.SIGNAL("QtBlissGraphSignal"),(ddict))
                self.replot()
            else:
                (marker,distance)=self.closestMarker(xpixel,ypixel)
                if distance < 4:
                    if marker is None:
                        pass
                    elif marker not in self.markersdict.keys():
                        print "Wrong Marker selection"
                    else:
                        if not self.markersdict[marker].has_key('xmarker'):
                            self.markersdict[marker]['xmarker'] = True
                        else:
                            xmarker = self.markersdict[marker]['xmarker']
                        if self.markersdict[marker]['followmouse']:
                            #self.canvas().setCursor(qt.QCursor(qt.QCursor.PointingHandCursor))
                            if (self.canvas().cursor().shape() != qt.QCursor.SizeHorCursor) and \
                               (self.canvas().cursor().shape() != qt.QCursor.SizeVerCursor):
                                self.__oldcursor = self.canvas().cursor().shape()
                            if xmarker:
                                #for x marker
                                self.canvas().setCursor(qt.QCursor(qt.QCursor.SizeHorCursor))
                            else:
                                #for y marker
                                self.canvas().setCursor(qt.QCursor(qt.QCursor.SizeVerCursor))
                        else:
                            #the marker is selectable because we are in markermode
                            if (self.canvas().cursor().shape() != qt.QCursor.SizeHorCursor) and \
                               (self.canvas().cursor().shape() != qt.QCursor.SizeVerCursor) and \
                               (self.canvas().cursor().shape() != qt.QCursor.PointingHandCursor):
                                self.__oldcursor = self.canvas().cursor().shape()
                            self.canvas().setCursor(qt.QCursor(qt.QCursor.PointingHandCursor))
                else:
                    self.canvas().setCursor(qt.QCursor(self.__oldcursor))
        #as default, export the mouse in graph coordenates
        dict= {'event':'MouseAt',
              'x':x,
              'y':y,
              'xpixel':xpixel,
              'ypixel':ypixel}
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('QtBlissGraphSignal'),(dict))
        
    def onMousePressed(self,e):
        #method to be overwritten
        if DEBUG:
            print "onMousePressed, event = ",e.pos().x(),e.pos().y()
        self.movingmarker = 0
        if qt.Qt.LeftButton == e.button():
            # Python semantics: self.pos = e.pos() does not work; force a copy
            #decide zooming or marker
            self._zooming = self.__zoomEnabled
            self.xpos = e.pos().x()
            self.ypos = e.pos().y()
            self.__timer      = time.time()
            if self.__markermode:
                if len(self.markersdict.keys()):
                    xpixel = e.pos().x()
                    ypixel = e.pos().y()
                    x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
                    y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
                    (marker,distance)=self.closestMarker(xpixel,ypixel)
                    if distance < 4:
                        if marker not in self.markersdict.keys():
                            print "Wrong Marker selection"
                        else:
                            if not self.markersdict[marker].has_key('xmarker'):
                                self.markersdict[marker]['xmarker'] = True
                            else:
                                xmarker = self.markersdict[marker]['xmarker']
                            if self.markersdict[marker]['followmouse']:
                                self.__markermoving = marker
                                if  QWTVERSION4:
                                    if xmarker:
                                        self.setMarkerXPos(marker, x)
                                    else:
                                        self.setMarkerYPos(marker, y)
                                else:
                                    if xmarker:
                                        self.setMarkerXPos(self.markersdict[marker]['marker'],
                                                       x)
                                    else:
                                        self.setMarkerYPos(self.markersdict[marker]['marker'],
                                                       y)
                        self._zooming = 0
                                
            if self._zooming:
                self.enableOutline(1)
                self.setOutlinePen(qt.QPen(qt.Qt.black))
                if  QWTVERSION4:
                    self.setOutlineStyle(qwt.Qwt.Rect)
                else:
                    self.picker.setRubberBand(Qwt.QwtPicker.RectRubberBand)                    
                if self.zoomStack == []:
                    if QWTVERSION4:
                        self.zoomState = (
                            self.axisScale(qwt.QwtPlot.xBottom).lBound(),
                            self.axisScale(qwt.QwtPlot.xBottom).hBound(),
                            self.axisScale(qwt.QwtPlot.yLeft).lBound(),
                            self.axisScale(qwt.QwtPlot.yLeft).hBound(),
                            )
                    else:
                        self.zoomState = (
                                self.canvasMap(qwt.QwtPlot.xBottom).s1(),
                                self.canvasMap(qwt.QwtPlot.xBottom).s2(),
                                self.canvasMap(qwt.QwtPlot.yLeft).s2(),
                                self.canvasMap(qwt.QwtPlot.yLeft).s1())
                if self.zoomStack2 == []:
                    if QWTVERSION4:
                        self.zoomState2 = (
                            self.axisScale(qwt.QwtPlot.xTop).lBound(),
                            self.axisScale(qwt.QwtPlot.xTop).hBound(),
                            self.axisScale(qwt.QwtPlot.yRight).lBound(),
                            self.axisScale(qwt.QwtPlot.yRight).hBound(),
                            )
                    else:
                        self.zoomState2 = (
                                self.canvasMap(qwt.QwtPlot.xTop).s1(),
                                self.canvasMap(qwt.QwtPlot.xTop).s2(),
                                self.canvasMap(qwt.QwtPlot.yRight).s2(),
                                self.canvasMap(qwt.QwtPlot.yRight).s1())
            elif self._selecting:
                self.enableOutline(1)
                self.setOutlinePen(qt.QPen(qt.Qt.black))
                if  QWTVERSION4:
                    self.setOutlineStyle(qwt.Qwt.Rect)
                else:
                    self.picker.setRubberBand(Qwt.QwtPicker.RectRubberBand)                    
        elif qt.Qt.RightButton == e.button():
            self._zooming = 0
            """
            if self.__markermode:
                if len(self.markersdict.keys()):
                    xpixel = e.pos().x()
                    ypixel = e.pos().y()
                    x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
                    y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
                    (marker,distance)=self.closestMarker(xpixel,ypixel)
                    if marker not in self.markersdict.keys():
                        print "Wrong Marker selection"
                    else:
                        if self.markersdict[marker]['followmouse']:
                            self.__markermoving = marker 
                            self.setMarkerXPos(marker, x)
                            self.replot()
                            dict = {}
                            dict['event']    = "markerMoved"
                            dict['distance'] = distance
                            dict['marker']   = marker
                            dict['x']        = x
                            dict['xpixel']   = xpixel
                            dict['y']        = y
                            dict['ypixel']   = ypixel
                            self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(dict,))
            """
        # fake a mouse move to show the cursor position
        self.onMouseMoved(e)

    def onMouseReleased(self,e):
        #method to be overwritten
        if DEBUG:
            print "onMouseRealeased, event = ",e
        if qt.Qt.LeftButton == e.button():
            #this is to solve a strange problem under darwin platform
            #where the signals were sent twice
            if self.__timer == -1:return
            etime = time.time() - self.__timer
            self.__timer = -1
            if (etime < 0.2) and ((self.xpos - e.pos().x()) == 0) and ((self.xpos - e.pos().x()) == 0):
                xpixel = e.pos().x()
                ypixel = e.pos().y()
                x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
                y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
                dict = {}
                dict['event']    = "MouseClick"
                dict['x']        = x
                dict['xpixel']   = xpixel
                dict['y']        = y
                dict['ypixel']   = ypixel
                if QTVERSION < '4.0.0':
                    self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(dict,))
                else:
                    self.emit(qt.SIGNAL("QtBlissGraphSignal"),(dict))
            if self._zooming:
                xmin0 = min(self.xpos, e.pos().x())
                xmax0 = max(self.xpos, e.pos().x())
                ymin0 = min(self.ypos, e.pos().y())
                ymax0 = max(self.ypos, e.pos().y())
                if QWTVERSION4:
                    self.setOutlineStyle(qwt.Qwt.Cross)
                else:
                    self.picker.setRubberBand(Qwt.QwtPicker.NoRubberBand)
                xmin = self.invTransform(qwt.QwtPlot.xBottom, xmin0)
                xmax = self.invTransform(qwt.QwtPlot.xBottom, xmax0)
                ymin = self.invTransform(qwt.QwtPlot.yLeft, ymin0)
                ymax = self.invTransform(qwt.QwtPlot.yLeft, ymax0)
                if xmin == xmax or ymin == ymax:
                    return
                if self.__keepimageratio:
                    ysize = abs(ymin-ymax)
                    xsize = abs(xmin-xmax)
                    a = min(xsize,ysize)
                    xsize = a
                    ysize = a * self.imageratio
                    xmin = int(0.5*(xmin+xmax)-0.5*xsize)
                    xmax = xmin + xsize -1
                    ymin = int(0.5*(ymin+ymax)-0.5*ysize)
                    ymax = ymin + round(ysize) -1
                    
                self.zoomStack.append(self.zoomState)
                self.zoomState = (xmin, xmax, ymin, ymax)
                self.enableOutline(0)
                autoreplot = self.autoReplot()
                self.setAutoReplot(False)
                self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)
                if QWTVERSION4:
                    self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)
                else:
                    self.setAxisScale(qwt.QwtPlot.yLeft, ymax, ymin)
                if self.axisEnabled(qwt.QwtPlot.xTop):
                    xmin = self.invTransform(qwt.QwtPlot.xTop, xmin0)
                    xmax = self.invTransform(qwt.QwtPlot.xTop, xmax0)
                    self.setAxisScale(qwt.QwtPlot.xTop, xmin, xmax)                
                if self.axisEnabled(qwt.QwtPlot.yRight):
                    ymin = self.invTransform(qwt.QwtPlot.yRight, ymin0)
                    ymax = self.invTransform(qwt.QwtPlot.yRight, ymax0)
                    if QWTVERSION4:
                        self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
                    else:
                        self.setAxisScale(qwt.QwtPlot.yRight, ymax, ymin)                
                self.zoomStack2.append(self.zoomState2)  
                self.zoomState2 = (xmin, xmax, ymin, ymax)
                self.setAutoReplot(autoreplot)
                self.replot()                
                dict = {}
                dict['event']    = "MouseZoom"
                dict['xmin']        = min(xmin,xmax)
                dict['xpixel_min']   = min(xmin0,xmax0)
                dict['ymin']        = min(ymin,ymax)
                dict['ypixel_min']   = min(ymin0,ymax0)
                dict['xmax']        = max(xmin,xmax)
                dict['xpixel_max']   = max(xmin0,xmax0)
                dict['ymax']        = max(ymin,ymax)
                dict['ypixel_max']   = max(ymin0,ymax0)

                if qt.qVersion() < '4.0.0':
                    self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(dict,))
                else:
                    self.emit(qt.SIGNAL("QtBlissGraphSignal"),(dict))
            elif self._selecting:
                xmin0 = min(self.xpos, e.pos().x())
                xmax0 = max(self.xpos, e.pos().x())
                ymin0 = min(self.ypos, e.pos().y())
                ymax0 = max(self.ypos, e.pos().y())
                if QWTVERSION4:
                    self.setOutlineStyle(qwt.Qwt.Cross)
                else:
                    self.picker.setRubberBand(Qwt.QwtPicker.NoRubberBand)
                xmin = self.invTransform(qwt.QwtPlot.xBottom, xmin0)
                xmax = self.invTransform(qwt.QwtPlot.xBottom, xmax0)
                ymin = self.invTransform(qwt.QwtPlot.yLeft, ymin0)
                ymax = self.invTransform(qwt.QwtPlot.yLeft, ymax0)
                self.enableOutline(0)
                self.replot()                
                ddict = {}
                ddict['event']    = "MouseSelection"
                ddict['xmin']        = min(xmin,xmax)
                ddict['xpixel_min']   = min(xmin0,xmax0)
                ddict['ymin']        = min(ymin,ymax)
                ddict['ypixel_min']   = min(ymin0,ymax0)
                ddict['xmax']        = max(xmin,xmax)
                ddict['xpixel_max']   = max(xmin0,xmax0)
                ddict['ymax']        = max(ymin,ymax)
                ddict['ypixel_max']   = max(ymin0,ymax0)
                if qt.qVersion() < '4.0.0':
                    self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),(ddict,))
                else:
                    self.emit(qt.SIGNAL("QtBlissGraphSignal"),(ddict))
            else:
                if self.__markermode:
                    if len(self.markersdict.keys()):
                        xpixel = e.pos().x()
                        ypixel = e.pos().y()
                        (marker,distance)=self.closestMarker(xpixel,ypixel)
                        if marker is None:
                            pass
                        elif marker not in self.markersdict.keys():
                            print "Wrong Marker selection"
                        else:
                            dict = {}
                            if self.markersdict[marker]['followmouse']:
                                dict['event']    = "markerMoved"
                            else:
                                dict['event']    = "markerSelected"
                            dict['distance'] = distance
                            dict['marker']   = marker
                            if QWTVERSION4:
                                x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
                                y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
                            else:
                                x = self.markersdict[marker]['marker'].xValue()
                                y = self.markersdict[marker]['marker'].yValue()
                            dict['x']        = x
                            dict['xpixel']   = xpixel
                            dict['y']        = y
                            dict['ypixel']   = ypixel
                            if qt.qVersion() < '4.0.0':
                                self.emit(qt.PYSIGNAL("QtBlissGraphSignal"),
                                          (dict,))
                            else:
                                self.emit(qt.SIGNAL("QtBlissGraphSignal"),
                                          (dict))
                    #self.canvas().setCursor(qt.QCursor(qt.QCursor.CrossCursor))
                    #self.canvas().setCursor(qt.QCursor(self.__oldcursor))
                    else:
                        #print "not in marker mode"
                        pass
                    self.__markermoving = 0                    
        elif qt.Qt.RightButton == e.button():
            if self.__zoomback:
                autoreplot = self.autoReplot()
                self.setAutoReplot(False)
                if len(self.zoomStack):
                    xmin, xmax, ymin, ymax = self.zoomStack.pop()
                    self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)
                    if QWTVERSION4:
                        self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)
                    else:
                        self.setAxisScale(qwt.QwtPlot.yLeft, ymax, ymin)
                    xmin, xmax, ymin, ymax = self.zoomStack2.pop()
                    if self.axisEnabled(qwt.QwtPlot.xTop):
                        self.setAxisScale(qwt.QwtPlot.xTop, xmin, xmax)
                    if self.axisEnabled(qwt.QwtPlot.yRight):
                        if QWTVERSION4:
                            self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
                        else:
                            self.setAxisScale(qwt.QwtPlot.yRight, ymax, ymin)                            
                    self.replot()
                self.setAutoReplot(autoreplot)
            return                    

    def enablezoomback(self):
        self.__zoomback = 1

    def disablezoomback(self):
        self.__zoomback = 0        

    def enablemarkermode(self):
        self.__markermode = 1

    def disablemarkermode(self):
        self.__markermode = 0        

    def setmarkerfollowmouse(self,marker,boolean):
        if marker in self.markersdict.keys():
            if boolean:
                self.markersdict[marker]['followmouse'] = 1
            else:
                self.markersdict[marker]['followmouse'] = 0
            
    def __legendsetactive(self):
        self.setactivecurve(self.__activelegendname)
        pass
    
    def __legendmaptoy1(self):
        self.mapToY1(self.__activelegendname)   
        self.setactivecurve(self.getactivecurve(justlegend=1))
    
    def __legendmaptoy2(self):
        self.mapToY2(self.__activelegendname)   
        self.setactivecurve(self.getactivecurve(justlegend=1))

    def __legendremovesignal(self):
        if DEBUG: print  "__legendremovesignal"
        self.__removecurveevent = {}
        self.__removecurveevent['event'] = "RemoveCurveEvent"
        self.__removecurveevent['legend'] = self.__activelegendname
 
    def customEvent(self, event):
        if event.dict.has_key('legend'):
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),(event.dict,))
            else:
                self.emit(qt.SIGNAL('QtBlissGraphSignal'), (event.dict))
        else:
            newevent = CrappyMouseEvent(event.dict['event'],
                                        event.dict['x'],
                                        event.dict['y'],
                                        event.dict['button'])
            if event.dict['event'] == qt.QEvent.MouseMove:
                self.onMouseMoved(newevent)
            elif event.dict['event'] == qt.QEvent.MouseButtonPress:
                self.onMousePressed(newevent)
            elif event.dict['event'] == qt.QEvent.MouseButtonRelease:
                self.onMouseReleased(newevent)
                    
    def maptoy1(self,keyorindex):
        print "maptoy1 deprecated, use mapToY1"
        self.mapToY1(keyorindex)

    def mapToY1(self, keyorindex):
        if type(keyorindex) == type(" "):
            if keyorindex in self.curveslist:
                index = self.curveslist.index(keyorindex) + 1
                key   = keyorindex
            else:
                return -1
        elif keyorindex >0:
            index = keyorindex
            key   = self.curveslist[index-1]
        else:
            return -1
        if not self.axisEnabled(qwt.QwtPlot.yLeft):
            self.enableAxis(qwt.QwtPlot.yLeft,1)
        self.curves[key]["maptoy2"] = 0
        self.setCurveYAxis(index,qwt.QwtPlot.yLeft)

    def maptoy2(self,keyorindex):
        print "maptoy2 deprecated, use mapToY2"
        self.mapToY2(keyorindex)

    def mapToY2(self, keyorindex):
        if type(keyorindex) == type(" "):
            if keyorindex in self.curveslist:
                index = self.curveslist.index(keyorindex) + 1
                key   = keyorindex
            else:
                return -1
        elif keyorindex >0:
            index = keyorindex
            key   = self.curveslist[index-1]
        else:
            return -1
        if not self.axisEnabled(qwt.QwtPlot.yRight):
            self.enableAxis(qwt.QwtPlot.yRight,1)
        self.curves[key]["maptoy2"] = 1
        self.setCurveYAxis(index,qwt.QwtPlot.yRight)        

    if not QWTVERSION4:
        def setCurveYAxis(self, index, axis):
            self.curves[self.curveslist[index-1]]['curve'].setYAxis(axis)


    def _legendClicked(self, itemorindex):
        if QWTVERSION4:
            index = itemorindex
            self.legendClicked
            listindex = 0
            legendtext = None
            for curve in self.curveslist:
                if self.curves[curve]['curve'] == index:
                    legendtext= curve
                    listindex = self.curveslist.index(legendtext) + 1
                    if DEBUG:
                        print "legend clicked with name = ",legendtext
                    break
            if legendtext is not  None:
                item = self.curves[curve]['curve']
            else:
                item = None
            self.legendClicked(item, index)
        else:
            self.legendClicked(itemorindex)


    def legendClicked(self, item, index=None):
        if QWTVERSION4:
            self.legendclicked(index)
            return
        legendtext = str(item.title().text())
        if self.onlyoneactive:
            self.setactivecurve(legendtext)
        else:
            self.toggleCurve(legendtext)

    def legendclicked(self,index):
        if DEBUG:
            print "legendclicked with index = ",index
        
        listindex = 0
        for curve in self.curveslist:
            if self.curves[curve]['curve'] == index:
                legendtext= curve
                listindex = self.curveslist.index(legendtext) + 1
                if DEBUG:
                    print "legendclicked with name = ",legendtext
                break

        if listindex > 0:
            if self.onlyoneactive:
                self.setactivecurve(legendtext)
            else:
                self.toggleCurve(legendtext)
        
        legend = self.legend()
        n = legend.itemCount()

        if n > 1:
            if 0:
                for i in range(n):
                #if i != index:
                    item=self.legend().findItem(i+1)
                    item.setFocusPolicy(qt.QWidget.ClickFocus)
            else:
                for curve in self.curves.keys():
                    item=self.legend().findItem(self.curves[curve] ["curve"])
                    if QTVERSION < '4.0.0':
                        item.setFocusPolicy(qt.QWidget.ClickFocus)    

        
    def setactivecolor(self,color):
        if color != self.__activecolor:    
            if color in self.colors.keys():
                if color in self.colorslist:
                    colorindex  = self.colorslist.index(color)
                    colorbuffer = self.__activecolor                
                    self.__activecolor = color
                    self.colorslist[colorindex] = colorbuffer
        return self.__activecolor

    def setactivelinewidth(self,linewidth):
        self.__activelinewidth = linewidth
        return self.__activelinewidth
        
    def xlabel(self,label=None):
        if DEBUG:"print xlabel deprecated, use x1Label"
        return self.x1Label(label)

    def x1Label(self,label=None):
        # set axis titles
        if label is None:
            if qwt.QWT_VERSION_STR[0] > '4':
                return self.axisTitle(qwt.QwtPlot.xBottom).text()
            else:
                return self.axisTitle(qwt.QwtPlot.xBottom)
        else:
            return self.setAxisTitle(qwt.QwtPlot.xBottom, label)
    
    def ylabel(self,label=None):
        if DEBUG:"print ylabel deprecated, use y1Label"
        self.y1Label(label)

    def y1Label(self,label=None):
        # set axis titles
        if label is None:
            if qwt.QWT_VERSION_STR[0] > '4':
                return self.axisTitle(qwt.QwtPlot.yLeft).text()
            else:
                return self.axisTitle(qwt.QwtPlot.yLeft)
        else:
            return self.setAxisTitle(qwt.QwtPlot.yLeft, label)
    

    def enableOutline(self, value):
        if QWTVERSION4:
            qwt.QwtPlot.enableOutline(self, value)
            
    def setOutlinePen(self, value):    
        if QWTVERSION4:
            qwt.QwtPlot.setOutlinePen(self, value)
            
    def setOutlineStyle(self, value):    
        if QWTVERSION4:
            qwt.QwtPlot.setOutlineStyle(self, value)


    if QWTVERSION4:
        def eventFilter(self,object,event):
            if object == self.canvas():
                if qt.qVersion() < '4.0.0':return 0
                if event.type() == qt.QEvent.MouseMove:
                    e = GraphCustomEvent()
                    e.dict['event']  = qt.QEvent.MouseMove
                    e.dict['x']      = event.pos().x()
                    e.dict['y']      = event.pos().y()
                    e.dict['button'] = event.button()
                    qt.QApplication.postEvent(self, e)
                elif event.type() == qt.QEvent.MouseButtonPress:
                    e = GraphCustomEvent()
                    e.dict['event']  = qt.QEvent.MouseButtonPress
                    e.dict['x']      = event.pos().x()
                    e.dict['y']      = event.pos().y()
                    e.dict['button'] = event.button()
                    qt.QApplication.postEvent(self, e)
                elif event.type() == qt.QEvent.MouseButtonRelease:
                    e = GraphCustomEvent()
                    e.dict['event']  = qt.QEvent.MouseButtonRelease
                    e.dict['x']      = event.pos().x()
                    e.dict['y']      = event.pos().y()
                    e.dict['button'] = event.button()
                    qt.QApplication.postEvent(self, e)
                else:return 0
                return 1
            elif event.type() == qt.QEvent.MouseButtonRelease:
                #if event.button() == qt.Qt.LeftButton:
                #    if qt.qVersion() > '4.0.0':
                #        legendname = str(object.text().text())
                #        self.setactivecurve(legendname)
                #        return 1
                if event.button() == qt.Qt.RightButton:
                    if QWTVERSION4:
                        self.__activelegendname = str(object.text())
                    else:
                        #print type(object.text())
                        #print type(object.text().text())
                        self.__activelegendname = str(object.text().text())
                    if not self.__uselegendmenu:
                            return 0
                    self.__event = None
                    if self.legendmenu is None:
                        if qt.qVersion() < '4.0.0':
                             self.legendmenu = qt.QPopupMenu()
                             self.legendmenu.insertItem(qt.QString("Set Active"),self.__legendsetactive)
                             self.legendmenu.insertSeparator()
                             self.legendmenu.insertItem(qt.QString("Map to y1") ,self.__legendmaptoy1)
                             self.legendmenu.insertItem(qt.QString("Map to y2") ,self.__legendmaptoy2)
                             self.legendmenu.insertSeparator()
                             self.legendmenu.insertItem(qt.QString("Remove curve") ,self.__legendremovesignal)
                        else:
                             self.legendmenu = qt.QMenu()
                             self.legendmenu.addAction(qt.QString("Set Active"),self.__legendsetactive)
                             self.legendmenu.addSeparator()
                             self.legendmenu.addAction(qt.QString("Map to y1") ,self.__legendmaptoy1)
                             self.legendmenu.addAction(qt.QString("Map to y2") ,self.__legendmaptoy2)
                             self.legendmenu.addSeparator()
                             self.legendmenu.addAction(qt.QString("Remove curve") ,self.__legendremovesignal)
                    if qt.qVersion() < '4.0.0':
                        self.legendmenu.exec_loop(self.cursor().pos())
                    else:
                        self.legendmenu.exec_(self.cursor().pos())
                    if self.__removecurveevent is not None:
                        if 0:
                            #This crashes windows Qt 3.3.2
                            if qt.qVersion() < '4.0.0':
                                self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),
                                                   (self.__removecurveevent,))
                            else:
                                self.emit(qt.SIGNAL('QtBlissGraphSignal'),
                                                    self.__removecurveevent)
                        else:
                            #This does not crash Qt 3.3.2
                            event = GraphCustomEvent()
                            event.dict['event' ]  = "RemoveCurveEvent"
                            event.dict['legend']  = self.__removecurveevent['legend']
                            qt.QApplication.postEvent(self, event)
                    return 1
            return 0
    else:
        def legendItemSlot(self, ddict):
            if ddict['event'] == "leftMousePressed": return self.setactivecurve(ddict['legend'])
            if ddict['event'] != "rightMouseReleased": return
            if not self.__uselegendmenu: return
            self.__activelegendname = ddict['legend']
            self.__event = None
            self.__removecurveevent = None
            if self.legendmenu is None:
                if QTVERSION < '4.0.0':
                     self.legendmenu = qt.QPopupMenu()
                     self.legendmenu.insertItem(qt.QString("Set Active"),self.__legendsetactive)
                     self.legendmenu.insertSeparator()
                     self.legendmenu.insertItem(qt.QString("Map to y1") ,self.__legendmaptoy1)
                     self.legendmenu.insertItem(qt.QString("Map to y2") ,self.__legendmaptoy2)
                     self.legendmenu.insertSeparator()
                     self.legendmenu.insertItem(qt.QString("Remove curve") ,self.__legendremovesignal)
                else:
                     self.legendmenu = qt.QMenu()
                     self.legendmenu.addAction(qt.QString("Set Active"),self.__legendsetactive)
                     self.legendmenu.addSeparator()
                     self.legendmenu.addAction(qt.QString("Map to y1") ,self.__legendmaptoy1)
                     self.legendmenu.addAction(qt.QString("Map to y2") ,self.__legendmaptoy2)
                     self.legendmenu.addSeparator()
                     self.legendmenu.addAction(qt.QString("Remove curve") ,self.__legendremovesignal)
            if QTVERSION < '4.0.0':
                self.legendmenu.exec_loop(self.cursor().pos())
            else:
                self.legendmenu.exec_(self.cursor().pos())
            if self.__removecurveevent is not None:
                event = GraphCustomEvent()
                event.dict['event' ]  = "RemoveCurveEvent"
                event.dict['legend']  = self.__removecurveevent['legend']
                qt.QApplication.postEvent(self, event)
    
    def newcurve(self,*var,**kw):
        if DEBUG: print "newcurve obsolete, use newCurve instead"
        return self.newCurve(*var,**kw)
        
    def newCurve(self,key,x=None,y=None,logfilter=0,curveinfo=None,**kw):
        if key not in self.curves.keys():
            self.curveinit(key,**kw)
            n = self.legend().itemCount()
            if n > 0:
                if QWTVERSION4:
                    #do it only for the last curve
                    item=self.legend().findItem(self.curves[key]['curve'])
                    item.setFocusPolicy(qt.QWidget.ClickFocus)
                    if self.__uselegendmenu:
                        item.installEventFilter(self)
                else:
                    item = self.legend().find(self.curves[key]['curve'])
                    if QTVERSION < '4.0.0':
                        self.connect(item,qt.PYSIGNAL("MyQwtLegendItemSignal"),
                                     self.legendItemSlot)
                    else:
                        self.connect(item,qt.SIGNAL("MyQwtLegendItemSignal"),
                                     self.legendItemSlot)
        else:
            #curve already exists
            pass
        if curveinfo is None:
            self.curves[key]['curveinfo'] = {}
        else:
            self.curves[key]['curveinfo'] = copy.deepcopy(curveinfo)
        if y is not None:
            if len(y):
                if type(y) == ArrayType:
                    if y.shape == (len(y), 1):
                       y.shape =  [len(y),] 
                if x is None:
                    x=arange(len(y))
                if logfilter:
                    i1=nonzero(y>0.0)
                    x= take(x,i1)
                    y= take(y,i1)
            if len(y):
                ymin = min(y)
            else:
                self.delcurve(key)
                return
                #if len(self.curves.keys()) != 1:
                #    self.delcurve(key)
                #    return                    
            self.setCurveData(self.curves[key]['curve'], x, y)
            if kw.has_key('baseline'):
                ybase = take(kw['baseline'],i1)
                if logfilter:
                    i1 = nonzero(ybase<=0)
                    for i in i1:
                        ybase[i] = ymin
                if QWTVERSION4:
                    self.curve(self.curves[key]['curve']).setbaseline(ybase)
                else:
                    self.curves[key]['curve'].setbaseline(ybase)
            if kw.has_key('regions'):
                #print "x0 = ",x[0]
                #print "x-1= ",x[-1]
                regions = []
                for region in kw['regions']:
                    #print region[0]
                    #print region[1]
                    i1 = min(nonzero(x>=region[0]),0)
                    i2 = max(nonzero(x<=region[1]))
                    #print "i1,i2 ",i1,i2
                    #print len(x)
                    regions.append([int(i1),int(i2)])
                if QWTVERSION4:
                    self.curve(self.curves[key]['curve']).setregions(regions)
                else:
                    self.curves[key]['curve'].setregions(regions)
            if len(self.curves.keys()) == 1:
                #set the active curve
                #self.legendclicked(1)
                self.setactivecurve(self.curves.keys()[0])
                #This almost work self.maptoy2(self.curves.keys()[0])                
        else:
            self.delcurve(key)
                
    def setxofy(self,legend):
        if legend in self.curves.keys():
            self.setCurveOptions(self.curves[legend]['curve'],qwt.QwtCurve.Xfy)

    def delcurve(self,key):
        index = None
        if key in self.curves.keys():
            del_index = self.curves[key]['curve']
            del self.curves[key]
        if key in self.curveslist:
            index = self.curveslist.index(key)
            del self.curveslist[index]
        if index is not None:
            try:
                if QWTVERSION4:
                    self.removeCurve(del_index)
                else:
                    del_index.detach()
            except:
                print "del_index = ",del_index,"error"
        if not len(self.curves.keys()):
            self.clearcurves()
            dict = {}
            dict['event' ]  = "SetActiveCurveEvent"
            dict['legend']  = None
            dict['index' ]  = -1
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),(dict,))
            else:
                self.emit(qt.SIGNAL('QtBlissGraphSignal'),(dict))        
        elif self.__activecurves is not None:
            if key in self.__activecurves:
                del self.__activecurves[self.__activecurves.index(key)]
                if self.__activecurves == []:
                    dict = {}
                    dict['event' ]  = "SetActiveCurveEvent"
                    dict['legend']  = None
                    dict['index' ]  = -1
                    if qt.qVersion() < '4.0.0':
                        self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),(dict,))
                    else:
                        self.emit(qt.SIGNAL('QtBlissGraphSignal'),(dict))
        
    def clearcurves(self):
        for key in self.curves.keys():
            self.delcurve(key) 
        self.__activecurves=[]
        #color counter
        self.color   = 0
        #symbol counter
        self.symbol  = 0
        #line type
        self.linetype= 0
        if QWTVERSION4: self.removeCurves()
        if 0:
            self.removeMarkers() #necessary because clear() will get rid of them
            self.clear()         #this deletes also plot items in Qwt5!
        else:                    #try to remove just the curves
            self.replot()
        
    if not QWTVERSION4:
        def removeCurves(self):
            for key in self.curves.keys():
                self.delcurve(key) 
                
        def removeMarkers(self):
            for key in self.markersdict.keys():
                self.markersdict[key]['marker'].detach()
                del self.markersdict[key]

        def closestMarker(self, xpixel, ypixel):
            x = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
            y = self.invTransform(qwt.QwtPlot.yLeft, ypixel)
            (marker, distance) = (None, None)
            xmarker = True
            for key in self.markersdict.keys():
                if marker is None:
                    marker   = key
                    if self.markersdict[key]['marker'].lineStyle() == Qwt.QwtPlotMarker.HLine:
                        #ymarker
                        distance = abs(y - self.markersdict[key]['marker'].yValue())
                        xmarker = False
                    else:
                        #xmarker
                        distance = abs(x - self.markersdict[key]['marker'].xValue())
                        xmarker = True
                else:
                    if self.markersdict[key]['marker'].lineStyle() == Qwt.QwtPlotMarker.HLine:
                        #ymarker
                        distancew = abs(y - self.markersdict[key]['marker'].yValue())
                        xmarker = False
                    else:
                        #xmarker
                        distancew = abs(x - self.markersdict[key]['marker'].xValue())
                        xmarker = True
                    if distancew < distance:
                        distance = distancew
                        marker   = key
            #this distance is in x coordenates
            #but I decide on distance in pixels ...
            if distance is not None:
                if xmarker:
                    x1pixel = abs(self.invTransform(qwt.QwtPlot.xBottom, xpixel+4)-x)/4.0
                    distance = distance / x1pixel
                else:
                    y1pixel = abs(self.invTransform(qwt.QwtPlot.xBottom, ypixel+4)-y)/4.0
                    distance = distance / y1pixel
            return (marker, distance)

    def toggleCurve(self, keyorindex):
        if type(keyorindex) == type(" "):
            if keyorindex in self.curveslist:
                index = self.curveslist.index(keyorindex) + 1
                key   = keyorindex
            else:
                return -1
        elif keyorindex >0:
            index = keyorindex
            key   = self.curveslist[index-1]
        else:
            return -1
        if 0:
          #this is to hide a curve
          if curve:
            curve.setEnabled(not curve.enabled())
            self.replot()
        if key in self.__activecurves:
            del self.__activecurves[self.__activecurves.index(key)]
            color = self.curves[key] ["pen"]
            linetype = self.curves[key] ["linetype"] 
            pen = qt.QPen(color,self.linewidth,linetype)
            self.setCurvePen(self.curves[key]['curve'],pen ) 
        else:
            linetype = self.curves[key] ["linetype"] 
            pen = qt.QPen(self.__activecolor,self.__activelinewidth,linetype)
            self.setCurvePen(self.curves[key]['curve'],pen ) 
            self.__activecurves.append(key)
        if 1:
            #plot just that curve?
            self.drawCurve(index,0,-1)
        else:
            self.replot()


    def getactivecurve(self,justlegend=0):
        #check the number of curves
        if len(self.curves.keys()) > 1:
            if not len(self.__activecurves):
                if justlegend:return None
                else:return None,None,None
            else:
                legend = self.__activecurves[0]    
        elif  len(self.curves.keys()) == 1:
            legend = self.curves.keys()[0]
        else:
            if justlegend:return None
            else: return None,None,None
        if legend in self.curves.keys():
            if justlegend:
                return legend
            index     = self.curves[legend]['curve']
            x=[]
            y=[]
            if QWTVERSION4:
                for i in range(self.curve(index).dataSize()):
                    x.append(self.curve(index).x(i))
                    y.append(self.curve(index).y(i))
            else:
                print "QtBlissGraph: get curve data to be implemented"
            return legend,array(x).astype(Float),array(y).astype(Float)
        else:
            return None,None,None

    def getcurveinfo(self,legend):
        if legend is None:return {}
        dict={}
        if legend in self.curves.keys():dict=copy.deepcopy(self.curves[legend]['curveinfo'].copy())
        return dict
        

            
    def setactivecurve(self,keyorindex):
        if type(keyorindex) == type(" "):
            if keyorindex in self.curves.keys():
                #index = self.curveslist.index(keyorindex) + 1
                index = self.curves[keyorindex]['curve']
                key   = keyorindex
            else:
                return -1
        elif keyorindex >0:
            index = keyorindex
            key   = self.curveslist[index-1]
        else:
            return -1
        for ckey in  self.__activecurves:
            del self.__activecurves[self.__activecurves.index(ckey)]
            if ckey in self.curves.keys():
                color = self.curves[ckey] ["pen"]
                linetype = self.curves[ckey] ["linetype"] 
                pen = qt.QPen(color,self.linewidth,linetype)
                self.setCurvePen(self.curves[ckey]['curve'],pen )
        linetype = self.curves[key] ["linetype"] 
        pen = qt.QPen(self.__activecolor,self.__activelinewidth,linetype)
        self.setCurvePen(self.curves[key]['curve'],pen ) 
        self.__activecurves.append(key)



        actualindex = self.curves[key] ["curve"] 
        n = self.legend().itemCount()
        if n > 1:
            if 0:
                for i in range(n):
                        item=self.legend().findItem(i+1)
                        item.setFocusPolicy(qt.QWidget.ClickFocus)
            else:
                for curve in self.curves.keys():
                    if QWTVERSION4:
                        item=self.legend().findItem(self.curves[curve] ["curve"])
                        item.setFocusPolicy(qt.QWidget.ClickFocus)
                    else:
                        if DEBUG:
                            print "findItem also missing"

        #self.legend().findItem(index).setFocus()
        self.replot()
        dict = {}
        dict['event' ]  = "SetActiveCurveEvent"
        dict['legend']  = key
        dict['index' ]  = index
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('QtBlissGraphSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('QtBlissGraphSignal'),(dict))
        return 0
    
    def setmarkercolor(self,marker,color,label=None):
        if marker not in self.markersdict.keys():
            print "ERROR, returning"
            return -1
        if color in self.colors.keys():
            pen = self.colors[color]
        else:
            pen = self.colors[self.colorslist[0]]
        #self.setMarkerPen(self.markerKeys()[marker-1],pen)
        if QWTVERSION4:
            self.setMarkerPen(self.markersdict[marker]['marker'],
                              qt.QPen(pen,0,qt.Qt.SolidLine))
        else:
            self.markersdict[marker]['marker'].setLinePen(qt.QPen(pen,0,qt.Qt.SolidLine))
        if label is not None:
            if QWTVERSION4:
                self.setMarkerLabel(self.markersdict[marker]['marker'],
                                    qt.QString(label) )
            else:
                self.markersdict[marker]['marker'].setLabel(Qwt.QwtText(label))
                
        return 0

    def curveinit(self,key,**kw):
        self.curves[key] =  {}
        #self.curves[key] =  self.insertCurve(key)
        if len(kw.keys()):
            if QWTVERSION4:
                curve = BlissCurve(self,key,**kw)
                self.curves[key] ["curve"] = self.insertCurve(curve)
            else:
                curve = BlissCurve(key,**kw)
                curve.attach(self)
                self.curves[key] ["curve"] = curve
        else:
            self.curves[key] ["curve"] = self.insertCurve(key)
        self.curves[key] ["name"] = qt.QString(str(key))
        self.curveslist.append(key)
        self.curves[key] ["symbol"] = self.getnewsymbol()        
        self.curves[key] ["maptoy2"]  = 0
        color = self.colors[self.colorslist[self.color]]
        linetype = self.linetypes[self.linetypeslist[self.linetype]]
        self.curves[key] ["pen"]    = color
        pen = qt.QPen(color,self.linewidth,linetype)
        self.curves[key] ["linetype"]  = linetype
        self.curves[key] ["curveinfo"] = {}
        self.getnewpen()
        self.setCurvePen(self.curves[key]['curve'],pen )

    def insertCurve(self, key):
        if QWTVERSION4:
            return qwt.QwtPlot.insertCurve(self, key)
        else:
            #curve = qwt.Qwt.QwtPlotCurve(key)
            curve = MyQwtPlotCurve(key)
            curve.attach(self)
            return curve
        
    def setCurvePen(self, curve, pen):
        if QWTVERSION4:
            qwt.QwtPlot.setCurvePen(self, curve, pen)
        else:
            curve.setPen(pen)

    def setCurveData(self, curve, x, y):
        if QWTVERSION4:
            qwt.QwtPlot.setCurveData(self, curve, x, y)
        else:
            curve.setData(x, y)
                      
    def insertLineMarker(self, key, position):
        if QWTVERSION4:
            return qwt.QwtPlot.insertLineMarker(self, key, position)
        else:
            m = Qwt.QwtPlotMarker()
            m.setLabel(Qwt.QwtText(key))
            try:
                m.setLabelAlignment(position)
            except:
                #print "marker label error"
                m.setLabelAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
            m.attach(self)
            return m

    def setMarkerYPos(self, m, value):
        if QWTVERSION4:
            return qwt.QwtPlot.setMarkerYPos(self, m, value)
        else:
            if m in self.markersdict.keys():
                m = self.markersdict[m]['marker']
            return m.setYValue(value)
                      
    def setMarkerXPos(self, m, value):
        if QWTVERSION4:
            return qwt.QwtPlot.setMarkerXPos(self, m, value)
        else:
            if m in self.markersdict.keys():
                m = self.markersdict[m]['marker']
            return m.setXValue(value)
                      
    def getnewpen(self):
        self.color = self.color + 1
        if self.color > (len(self.colorslist)-1):
            self.color = 0
            self.linetype += 1
            if self.linetype > (len(self.linetypeslist)-1):
                self.linetype = 0  
        return    
    
    def getnewstyle(self):
        return self.linetype
    
    def getnewsymbol(self):
        return
    
    def getcurveaxes(self,key):
        if type(key) == type(" "):
            if key in self.curveslist:
                index = self.curveslist.index(key) + 1
            else:
                return -1,-1
        elif key >0:
            index = key
        else:
            return -1,-1
        x = self.curveXAxis(index)
        y = self.curveXAxis(index)
        return x,y

    def getX1AxisLimits(self):
        #get the current limits
        if QWTVERSION4:
            xmin = self.canvasMap(qwt.QwtPlot.xBottom).d1()
            xmax = self.canvasMap(qwt.QwtPlot.xBottom).d2()
        else:
            xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
            xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
        return xmin,xmax

    def getx1axislimits(self):
        if DEBUG:print "getx1axislimits deprecated, use getX1AxisLimits instead"
        return self.getX1AxisLimits()
        
    def getY1AxisLimits(self):
        #get the current limits
        if QWTVERSION4:
            ymin = self.canvasMap(qwt.QwtPlot.yLeft).d1()
            ymax = self.canvasMap(qwt.QwtPlot.yLeft).d2()
        else:
            ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
            ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
        return ymin,ymax

    def gety1axislimits(self):
        if DEBUG:print "gety1axislimits deprecated, use getY1AxisLimits instead"
        return self.getY1AxisLimits()

    def setX1AxisInverted(self, flag):
        self.axisScaleEngine(qwt.QwtPlot.xBottom).setAttribute(qwt.QwtScaleEngine.Inverted,
                                                               flag)
        
    def setY1AxisInverted(self, flag):
        self.axisScaleEngine(qwt.QwtPlot.yLeft).setAttribute(qwt.QwtScaleEngine.Inverted,
                                                             flag)

    def isX1AxisInverted(self):
        return self.axisScaleEngine(qwt.QwtPlot.xBottom).    \
                                   testAttribute(qwt.QwtScaleEngine.Inverted)

    def isY1AxisInverted(self):
        return self.axisScaleEngine(qwt.QwtPlot.yLeft).    \
                                   testAttribute(qwt.QwtScaleEngine.Inverted)

    def isY2AxisInverted(self):
        return self.axisScaleEngine(qwt.QwtPlot.yRight).    \
                                   testAttribute(qwt.QwtScaleEngine.Inverted)

    def setY2AxisInverted(self, flag):
        self.axisScaleEngine(qwt.QwtPlot.yRight).setAttribute(qwt.QwtScaleEngine.Inverted,
                                                              flag)

    def setx1axislimits(self, *var, **kw):
        if DEBUG:print "setx1axislimits deprecated, use setX1AxisLimits instead"
        return self.setX1AxisLimits(*var, **kw)

    def setX1AxisLimits(self, xmin, xmax, replot=None):
        if replot is None: replot = True
        self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)
        if replot:self.replot()

    def sety1axislimits(self, *var, **kw):
        if DEBUG:print "sety1axislimits deprecated, use setY1AxisLimits instead"
        return self.setY1AxisLimits(*var, **kw)
        
    def setY1AxisLimits(self, ymin, ymax, replot=None):
        if replot is None: replot = True
        if self.__logy1:
            if ymin <= 0:
                ymin = 1
            #else:
            #    ymin = log(ymin)            
            if ymax <= 0:
                ymax = ymin+1
            #else:
            #    ymax = log(ymax)
        self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)
        if replot:self.replot()

    def sety2axislimits(self,*var):
        if DEBUG:print "Deprecation warning: use setY2AxisLimits instead"
        return self.setY2AxisLimits(*var)

    def setY2AxisLimits(self, ymin, ymax):
        if self.__logy2:
            if ymin <= 0:
                ymin = 1
            #else:
            #    ymin = log(ymin)            
            if ymax <= 0:
                ymax = ymin+1
            #else:
            #    ymax = log(ymax)            
        self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)
        self.replot()

    def insertx1marker(self,*var,**kw):
        if DEBUG:"print insertx1marker deprecated, use insertX1Marker"
        return self.insertX1Marker(*var, **kw)

    def inserty1marker(self,*var,**kw):
        if DEBUG:"print inserty1marker deprecated, use insertY1Marker"
        return self.insertY1Marker(*var, **kw)

    def insertX1Marker(self,*var,**kw):
        if len(var) < 1:
            return -1
        elif len(var) ==1:
            x = var[0]
            y = None
        else:
            x=var[0]
            y=var[1]
        if kw.has_key('label'):
            label = kw['label']
        else:
            label = ""
        if kw.has_key('noline'):
            noline = kw['noline']
        else:
            noline = False

        if QWTVERSION4:
            if not noline:
                marker = self.insertLineMarker(label,qwt.QwtPlot.xBottom)
            else:
                marker = self.insertMarker()
            mX = marker
        else:
            mX = Qwt.QwtPlotMarker()
            mX.setLabel(Qwt.QwtText(label))
            mX.setLabelAlignment(qt.Qt.AlignRight | qt.Qt.AlignTop)
            if not noline:
                mX.setLineStyle(Qwt.QwtPlotMarker.VLine)
            marker = id(mX)

        if marker == 0:
            print "Error inserting marker!!!!"
            return -1
        if y is None:
            if QWTVERSION4:
                self.setMarkerXPos(marker,x)
            else:
                mX.setXValue(x)
        else:
            if QWTVERSION4:
                self.setMarkerPos(marker, x,y)
            else:
                mX.setXValue(x)
                mX.setYValue(y)
        if marker in self.markersdict.keys():
            self.markersdict[marker]['marker'] = mX
        else:
            self.markersdict[marker]={}
            self.markersdict[marker]['marker']      = mX
            self.markersdict[marker]['followmouse'] = 0
        self.markersdict[marker]['xmarker'] = True
        if not QWTVERSION4:    mX.attach(self)
        return marker

    def insertY1Marker(self,*var,**kw):
        if len(var) < 1:
            return -1
        elif len(var) ==1:
            y = var[0]
            x = None
        else:
            x=var[0]
            y=var[1]
        if kw.has_key('label'):
            label = kw['label']
        else:
            label = ""

        if QWTVERSION4:
            marker = self.insertLineMarker(label,qwt.QwtPlot.yLeft)
            mX = marker
        else:
            mX = Qwt.QwtPlotMarker()
            mX.setLabel(Qwt.QwtText(label))
            mX.setLabelAlignment(qt.Qt.AlignRight | qt.Qt.AlignTop)
            mX.setLineStyle(Qwt.QwtPlotMarker.HLine)
            marker = id(mX)

        if marker == 0:
            print "Error inserting marker!!!!"
            return -1
        if x is None:
            if QWTVERSION4:
                self.setMarkerYPos(marker,y)
            else:
                mX.setYValue(y)
        else:
            if QWTVERSION4:
                self.setMarkerPos(marker, x,y)
            else:
                mX.setXValue(x)
                mX.setYValue(y)
        if marker in self.markersdict.keys():
            self.markersdict[marker]['marker'] = mX
        else:
            self.markersdict[marker]={}
            self.markersdict[marker]['marker']      = mX
            self.markersdict[marker]['followmouse'] = 0
        self.markersdict[marker]['xmarker'] = False
        if not QWTVERSION4:    mX.attach(self)
        return marker

    def setx1markerpos(self,marker,x,y=None):
        if QWTVERSION4:
            if y is None:
                self.setMarkerXPos(marker, x)
            else:
                self.setMarkerPos(marker, x,y)
        else:
            if marker in self.markersdict.keys():
                marker = self.markersdict[marker]['marker']

            if y is None:
                self.setMarkerXPos(marker, x)
            else:
                self.setMarkerPos(marker, x,y)
         
    def clearmarkers(self):
        if DEBUG:print "Deprecation warning: use clearMarkers instead"
        return self.clearMarkers()

    def clearMarkers(self):
        self.removeMarkers()
        self.markersdict = {}
        
    def removeMarker(self,marker):
        if marker in self.markersdict.keys():
            if QWTVERSION4:
                qwt.QwtPlot.removeMarker(self,marker)
                del self.markersdict[marker]
            else:
                self.markersdict[marker]['marker'].detach()
                del self.markersdict[marker]
        else:
            for m in self.markerdict.keys():
                if id(self.markerdict[m]) == id(marker):
                    marker.detach()
                    del self.markersdict[m]
                    break

    def removemarker(self,marker):
        print "Deprecation warning: use removeMarker instead"
        return self.removeMarker(marker)

    if QTVERSION < '4.0.0':
        def printps(self):
            printer = qt.QPrinter()
            if printer.setup(self):
                painter = qt.QPainter()
                if not(painter.begin(printer)):
                    return 0
                metrics = qt.QPaintDeviceMetrics(printer)
                dpiy    = metrics.logicalDpiY()
                margin  = int((2/2.54) * dpiy) #2cm margin
                body = qt.QRect(0.5*margin, margin, metrics.width()- 1 * margin, metrics.height() - 2 * margin)
                fil  = qwt.QwtPlotPrintFilter()
                fil.setOptions(fil.PrintAll)
                fil.apply(self)
                self.printPlot(painter,body)
                painter.end()
    else:
        def printps(self):
            printer = qt.QPrinter()
            printDialog = qt.QPrintDialog(printer, self)
            if printDialog.exec_():
                try:
                    painter = qt.QPainter()
                    if not(painter.begin(printer)):
                        return 0
                    dpiy    = printer.logicalDpiY()
                    margin  = int((2/2.54) * dpiy) #2cm margin
                    body = qt.QRect(0.5*margin,
                                    margin,
                                    printer.width()- 1 * margin,
                                    printer.height() - 2 * margin)
                    fil  = qwt.QwtPlotPrintFilter()
                    fil.setOptions(fil.PrintAll)
                    fil.apply(self)
                    self.print_(painter,body)
                finally:
                    painter.end()
        
class Qwt5PlotImage(qwt.QwtPlotItem):
    def __init__(self, parent, palette=None):
        qwt.QwtPlotItem.__init__(self)
        self.xyzs = None
        self.attach(parent)
        self.image = None
        if not USE_SPS_LUT:
            if palette is None:
                self.palette = fuzzypalette()
            else:
                self.palette = palette
                
    def draw(self, painter, xMap, yMap,lgx1=0,lgy1=0):
        """Paint image to zooming to xMap, yMap

        Calculate (x1, y1, x2, y2) so that it contains at least 1 pixel,
        and copy the visible region to scale it to the canvas.
        """
        # calculate y1, y2
        if self.image is None:
            return
        if lgy1:
            if yMap.s1() <= 1:
                yMaps1 = 0.0
            else:
                yMaps1 = pow(10.,yMap.s1())
            yMaps2 = pow(10.,yMap.s2())
        else:
            yMaps2 = yMap.s2()
            yMaps1 = yMap.s1()
        if DEBUG:
            print "working drawImage xMap,yMap",xMap.s1(),xMap.s2(), yMaps1,yMaps2
        y1 = y2 = self.image.height()
        y1 *= (self.yMap.s2() - yMaps2)
        y1 /= (self.yMap.s2() - self.yMap.s1())
        y1 = max(0, int(y1-0.5))
        y2 *= (self.yMap.s2() - yMaps1)
        y2 /= (self.yMap.s2() - self.yMap.s1())
        y2 = min(self.image.height(), int(y2+0.5))
        # calculate x1, x1
        x1 = x2 = self.image.width()
        #x1 *= (self.xMap.d2() - xMap.d2())
        x1 *= (xMap.s1() - self.xMap.s1())
        x1 /= (self.xMap.s2() - self.xMap.s1())
        x1 = max(0, int(x1-0.5))
        x2 *= (xMap.s2()-self.xMap.s1())
        x2 /= (self.xMap.s2() - self.xMap.s1())
        x2 = min(self.image.width(), int(x2+0.5))
        #print x1,x2,y1,y2
        # copy
        image = self.image.copy(x1, y1, x2-x1, y2-y1)
        # zoom
        if QTVERSION < '4.0.0':
            image = image.smoothScale(xMap.p2()-xMap.p1()+1, yMap.p1()-yMap.p2()+1)
        else:
            image = image.scaled(xMap.p2()-xMap.p1()+1, yMap.p1()-yMap.p2()+1)
        # draw
        painter.drawImage(xMap.p1(), yMap.p2(), image)

    def setData(self, xyzs, xScale = None, yScale = None, colormap=None,
                xmirror=0, ymirror=1):
        self.xyzs = xyzs
        shape = xyzs.shape
        if xScale is None:
            xRange = (0, shape[1])
        else:
            xRange = xScale * 1

        if yScale is None:
            yRange = (0, shape[0])
        else:
            yRange = yScale * 1

        if self.plot().isX1AxisInverted():
            self.xMap = Qwt.QwtScaleMap(0, shape[0], xRange[1], xRange[0])
            self.plot().setAxisScale(Qwt.QwtPlot.xBottom, xRange[1], xRange[0])
        else:
            self.xMap = Qwt.QwtScaleMap(0, shape[0], xRange[0], xRange[1])
            self.plot().setAxisScale(Qwt.QwtPlot.xBottom,  xRange[0], xRange[1])

        if self.plot().isY1AxisInverted():
            self.yMap = Qwt.QwtScaleMap(0, shape[1], yRange[1], yRange[0])
            self.plot().setAxisScale(Qwt.QwtPlot.yLeft, yRange[1], yRange[0])
        else:
            self.yMap = Qwt.QwtScaleMap(0, shape[1], yRange[0], yRange[1])
            self.plot().setAxisScale(Qwt.QwtPlot.yLeft, yRange[0], yRange[1])

        if not USE_SPS_LUT:
            #calculated palette
            if QWTVERSION4:
                self.image = qwt.toQImage(bytescale(self.xyzs)).mirror(xmirror, ymirror)
                for i in range(0, 256):
                    self.image.setColor(i, self.palette[i])
            else:
                self.image = qwt.toQImage(bytescale(self.xyzs)).mirrored(xmirror, ymirror)
                for i in range(0, 256):
                    self.image.setColor(i, self.palette[i])
            
        else:
            if colormap is None:
                (self.image_buffer,size,minmax)= spslut.transform(self.xyzs, (1,0),
                                         (spslut.LINEAR,3.0), "BGRX", spslut.TEMP,
                                          1, (min(ravel(self.xyzs)),max(ravel(self.xyzs))))
            else:
                (self.image_buffer,size,minmax)= spslut.transform(self.xyzs, (1,0),
                                         (spslut.LINEAR,3.0),
                                         "BGRX", COLORMAPLIST[int(str(colormap[0]))],
                                          colormap[1], (colormap[2],colormap[3]))
            if QTVERSION < '4.0.0':
                self.image=qt.QImage(self.image_buffer,size[0], size[1],
                                    32, None,
                                    0, qt.QImage.IgnoreEndian).mirror(xmirror,
                                                                      ymirror)
            else:
                self.image = qt.QImage(self.image_buffer,size[0], size[1],
                                       qt.QImage.Format_RGB32).mirrored(xmirror,
                                                                        ymirror)

    def setPixmap(self, pixmap, size = None, xScale = None, yScale = None,
                  xmirror = 0, ymirror = 1):
        #I have to receive an array
        if type(pixmap) == type(""):
            shape = size
        else:
            shape = size
       #     shape = pixmap.shape
       #     if len(shape) == 1:
       #         shape = (shape[0], 1)
        if xScale is None:
            xRange = (0, shape[0])
        else:
            xRange = xScale * 1

        if yScale is None:
            yRange = (0, shape[1])
        else:
            yRange = yScale * 1

        if self.plot().isX1AxisInverted():
            self.xMap = Qwt.QwtScaleMap(0, shape[0], xRange[1], xRange[0])
            self.plot().setAxisScale(Qwt.QwtPlot.xBottom, xRange[1], xRange[0])
        else:
            self.xMap = Qwt.QwtScaleMap(0, shape[0], xRange[0], xRange[1])
            self.plot().setAxisScale(Qwt.QwtPlot.xBottom,  xRange[0], xRange[1])
            
        if self.plot().isY1AxisInverted():
            self.yMap = Qwt.QwtScaleMap(0, shape[1], yRange[1], yRange[0])
            self.plot().setAxisScale(Qwt.QwtPlot.yLeft, yRange[1], yRange[0])
        else:
            self.yMap = Qwt.QwtScaleMap(0, shape[1], yRange[0], yRange[1])
            self.plot().setAxisScale(Qwt.QwtPlot.yLeft, yRange[0], yRange[1])

        if QTVERSION < '4.0.0':
            if type(pixmap) == type(""):
                self.image=qt.QImage(pixmap,
                                     size[0],
                                     size[1],
                                     32, None, 0,
                                     qt.QImage.IgnoreEndian).mirror(xmirror,
                                                                    ymirror)
            else:
                self.image=qt.QImage(pixmap.tostring(),
                                     size[0],
                                     size[1],
                                     32, None, 0,
                                     qt.QImage.IgnoreEndian).mirror(xmirror,
                                                                    ymirror)
        else:
            if type(pixmap) == type(""):
                self.image = qt.QImage(pixmap,
                                   size[0],
                                   size[1],
                                   qt.QImage.Format_RGB32).mirrored(xmirror,
                                                                    ymirror)
            else:
                self.image = qt.QImage(pixmap.tostring(),
                                   size[0],
                                   size[1],
                                   qt.QImage.Format_RGB32).mirrored(xmirror,
                                                                    ymirror)


class QwtPlotImage(qwt.QwtPlotMappedItem):
    def __init__(self, parent, palette=None):
        qwt.QwtPlotItem.__init__(self, parent)
        self.xyzs = None
        self.plot = parent
        self.image = None
        if not USE_SPS_LUT:
            if palette is None:
                self.palette = fuzzypalette()
            else:
                self.palette = palette
                
    def drawImage(self, painter, xMap, yMap,lgx1=0,lgy1=0):
        """Paint image to zooming to xMap, yMap

        Calculate (x1, y1, x2, y2) so that it contains at least 1 pixel,
        and copy the visible region to scale it to the canvas.
        """
        if DEBUG:
            print "drawImage xMap,yMap",xMap.d1(),xMap.d2(), yMap.d1(),yMap.d2()
            print "drawImage self.xMap,yMap",self.xMap.d1(),self.xMap.d2(), self.yMap.d1(),self.yMap.d2()
            print "drawImage xMap,yMap",xMap.i1(),xMap.i2(), yMap.i1(),yMap.i2()
        # calculate y1, y2
        if self.image is None:
            return
        if lgy1:
            if yMap.d1() <= 1:
                yMapd1 = 0.0
            else:
                yMapd1 = pow(10.,yMap.d1())
            yMapd2 = pow(10.,yMap.d2())
        else:
            yMapd2 = yMap.d2()
            yMapd1 = yMap.d1()
        if DEBUG:
            print "working drawImage xMap,yMap",xMap.d1(),xMap.d2(), yMapd1,yMapd2
        y1 = y2 = self.image.height()
        y1 *= (self.yMap.d2() - yMapd2)
        y1 /= (self.yMap.d2() - self.yMap.d1())
        y1 = max(0, int(y1-0.5))
        y2 *= (self.yMap.d2() - yMapd1)
        y2 /= (self.yMap.d2() - self.yMap.d1())
        y2 = min(self.image.height(), int(y2+0.5))
        # calculate x1, x1
        x1 = x2 = self.image.width()
        #x1 *= (self.xMap.d2() - xMap.d2())
        x1 *= (xMap.d1() - self.xMap.d1())
        x1 /= (self.xMap.d2() - self.xMap.d1())
        x1 = max(0, int(x1-0.5))
        x2 *= (xMap.d2()-self.xMap.d1())
        x2 /= (self.xMap.d2() - self.xMap.d1())
        x2 = min(self.image.width(), int(x2+0.5))
        #print x1,x2,y1,y2
        # copy
        image = self.image.copy(x1, y1, x2-x1, y2-y1)
        # zoom
        #print "zoom = ",xMap.i2()-xMap.i1()+1, yMap.i1()-yMap.i2()+1
        image = image.smoothScale(xMap.i2()-xMap.i1()+1, yMap.i1()-yMap.i2()+1)
        # draw
        painter.drawImage(xMap.i1(), yMap.i2(), image)

    def setData(self, xyzs, xScale = None, yScale = None, colormap=None,
                xmirror=0, ymirror=1):
        self.xyzs = xyzs
        shape = xyzs.shape
        if xScale is not None:
            self.xMap = qwt.QwtDiMap(0, shape[0], xScale[0], xScale[1])
            self.plot.setAxisScale(qwt.QwtPlot.xBottom, *xScale)
        else:
            self.xMap = qwt.QwtDiMap(0, shape[0], 0, shape[0])
        if yScale is not None:
            self.yMap = qwt.QwtDiMap(0, shape[1], yScale[0], yScale[1])
            self.plot.setAxisScale(qwt.QwtPlot.yLeft, *yScale)
        else:
            self.yMap = qwt.QwtDiMap(0, shape[1], 0, shape[1])
        if not USE_SPS_LUT:
            #calculated palette
            self.image = qwt.toQImage(bytescale(self.xyzs)).mirror(xmirror, ymirror)
            for i in range(0, 255):
                self.image.setColor(i, self.palette[i])
        else:
            if colormap is None:
                (self.image_buffer,size,minmax)= spslut.transform(self.xyzs, (1,0),
                                         (spslut.LINEAR,3.0), "BGRX", spslut.TEMP,
                                          1, (min(ravel(self.xyzs)),max(ravel(self.xyzs))))
            else:
                (self.image_buffer,size,minmax)= spslut.transform(self.xyzs, (1,0),
                                         (spslut.LINEAR,3.0), "BGRX", COLORMAPLIST[colormap[0]],
                                          colormap[1], (colormap[2],colormap[3]))
            self.image=qt.QImage(self.image_buffer,size[0], size[1],32, None, 0,
                                 qt.QImage.IgnoreEndian).mirror(xmirror,ymirror)

    def setPixmap(self, pixmap, size = None, xScale = None, yScale = None,
                  xmirror = 0, ymirror=1):
        #I have to receive an array
        if type(pixmap) == type(""):
            shape = size
        else:
            shape = size
       #     shape = pixmap.shape
       #     if len(shape) == 1:
       #         shape = (shape[0], 1)
        if xScale is None:
            xRange = (0, shape[0])
        else:
            xRange = xScale * 1

        if yScale is None:
            yRange = (0, shape[1])
        else:
            yRange = yScale * 1
        
        self.xMap = Qwt.QwtScaleMap(0, shape[0], *xRange)
        self.plot().setAxisScale(Qwt.QwtPlot.xBottom, *xRange)
        self.yMap = Qwt.QwtScaleMap(0, shape[1], *yRange)
        self.plot().setAxisScale(Qwt.QwtPlot.yLeft, *yRange)
        if type(pixmap) == type(""):
            self.image=qt.QImage(pixmap,
                                 size[0],
                                 size[1],
                                 32, None, 0,
                                 qt.QImage.IgnoreEndian).mirror(xmirror,ymirror)
        else:
            self.image=qt.QImage(pixmap.tostring(),
                                 size[0],
                                 size[1],
                                 32, None, 0,
                                 qt.QImage.IgnoreEndian).mirror(xmirror,ymirror)

if qwt.QWT_VERSION_STR[0] > '4':
    class MyPicker(Qwt.QwtPicker):
        def widgetMousePressEvent(self, event):
            if DEBUG:print "mouse press"
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("MousePressed(const QMouseEvent&)"),
                          (event,))
            else:
                self.emit(qt.SIGNAL("MousePressed(const QMouseEvent&)"), event)
            Qwt.QwtPicker.widgetMousePressEvent(self, event)

        def widgetMouseReleaseEvent(self, event):
            if DEBUG:print "mouse release"
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("MouseReleased(const QMouseEvent&)"),
                          (event,))
            else:
                self.emit(qt.SIGNAL("MouseReleased(const QMouseEvent&)"), event)
            Qwt.QwtPicker.widgetMouseReleaseEvent(self, event)

        def widgetMouseDoubleClickEvent(self, event):
            if DEBUG:print "mouse doubleclick"
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("MouseDoubleClicked(const QMouseEvent&)"),
                          (event,))
            else:
                self.emit(qt.SIGNAL("MouseDoubleClicked(const QMouseEvent&)"), event)
            Qwt.QwtPicker.widgetMouseDoubleClickEvent(self, event)

        def widgetMouseMoveEvent(self, event):
            if DEBUG:print "mouse move"
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("MouseMoved(const QMouseEvent&)"), (event,))
            else:
                self.emit(qt.SIGNAL("MouseMoved(const QMouseEvent&)"), event)
            Qwt.QwtPicker.widgetMouseMoveEvent(self, event)

    class MyQwtPlotCurve(Qwt.QwtPlotCurve):
        def __init__(self, key):
            Qwt.QwtPlotCurve.__init__(self, key)

        #These two methods to be overloaded to use ANY widget as legend item.
        #Default implementation is a QwtLegendItem            
        def legendItem(self):
            if DEBUG:print "in legend item"
            #return Qwt.QwtPlotCurve.legendItem(self)
            return MyQwtLegendItem(self.plot().legend())

        def updateLegend(self, legend):
            if DEBUG:print "in update legend!"
            return Qwt.QwtPlotCurve.updateLegend(self, legend)

    class MyQwtLegendItem(Qwt.QwtLegendItem):
        def __init__(self, *var):
            Qwt.QwtLegendItem.__init__(self, *var)
            self.legendMenu = None
                     
        def mousePressEvent(self, event):
            if DEBUG:print "MyQwtLegendItem mouse pressed"
            if event.button() == qt.Qt.LeftButton:
                text = "leftMousePressed"
            elif event.button() == qt.Qt.RightButton:
                text = "rightMousePressed"
            else:
                text = "centralMousePressed" 
            self.__emitSignal(text)

        def mouseReleaseEvent(self, event):
            if DEBUG:print "MyQwtLegendItem mouse released"
            if event.button() == qt.Qt.LeftButton:
                text = "leftMouseReleased"
            elif event.button() == qt.Qt.RightButton:
                text = "rightMouseReleased"
            else:
                text = "centralMouseReleased" 
            self.__emitSignal(text)

        def __emitSignal(self, eventText):
            ddict = {}
            ddict['legend'] = str(self.text().text())
            ddict['event']  = eventText 
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("MyQwtLegendItemSignal"), (ddict,))
            else:
                self.emit(qt.SIGNAL("MyQwtLegendItemSignal"), ddict)
if QWTVERSION4:
    class BlissCurve(qwt.QwtPlotCurve):
        def __init__(self,parent=None,name="",regions=[[0,-1]],baseline=[]):
            qwt.QwtPlotCurve.__init__(self,parent,name)
            self.regions = regions
            self.setStyle(qwt.QwtCurve.Sticks)
            self.baselinedata = baseline 
            #self.setOptions(qwt.QwtCurve.Xfy)
            
        def draw(self,painter,xMap,yMap,a,b):
            for region in self.regions:
                if len(self.baselinedata):
                    for i in range(int(region[0]),int(region[1])):
                        #get the point
                        self.setBaseline(self.baselinedata[i])
                        qwt.QwtCurve.draw(self,painter,xMap,yMap,i,i)
                else: 
                    qwt.QwtCurve.draw(self,painter,xMap,yMap,region[0],region[1])

        def setregions(self,regions):
            self.regions = regions

        def setbaseline(self,baseline):
            self.baselinedata = baseline
else:
    class BlissCurve(MyQwtPlotCurve):
        def __init__(self,name="",regions=[[0,-1]],baseline=[]):
            MyQwtPlotCurve.__init__(self,name)
            self.regions = regions
            self.setStyle(qwt.QwtCurve.Sticks)
            self.baselinedata = baseline
            #self.setOptions(qwt.QwtCurve.Xfy)

        def drawFromTo(self,painter,xMap,yMap,a,b):
            for region in self.regions:
                if len(self.baselinedata):
                    for i in range(int(region[0]),int(region[1])):
                        #get the point
                        self.setBaseline(self.baselinedata[i])
                        MyQwtPlotCurve.drawFromTo(self,painter,xMap,yMap,i,i)
                else: 
                    MyQwtPlotCurve.drawFromTo(self,painter,xMap,yMap,region[0],region[1])

        def setregions(self,regions):
            self.regions = regions

        def setbaseline(self,baseline):
            self.baselinedata = baseline

class TimeScaleDraw(qwt.QwtScaleDraw):
    def label(self, value):
        if int(value) - value == 0:
            value = abs(int(value))
            h = value / 3600
            m = (value % 3600) / 60
            s = value - 3600*h - 60*m


            if h == 0 and m == 0:
                return '%ds' % s
            else:
                if h == 0:
                    return '%dm%02ds' % (m, s)
                else:
                    return '%dh%02dm%02ds' % (h, m, s)
        else:
            return ''

def make0():
    demo = QtBlissGraphContainer(uselegendmenu=1)
    demo.resize(500, 300)
    # set axis titles
    demo.graph.setAxisTitle(qwt.QwtPlot.xBottom, 'x -->')
    demo.graph.setAxisTitle(qwt.QwtPlot.yLeft, 'y -->')
    demo.graph.uselegendmenu = 1
    # insert a few curves
    cSin={}
    cCos={}
    nplots=10
    for i in range(nplots):
        # calculate 3 NumPy arrays
        x = arrayrange(0.0, 10.0, 0.1)
        y = 10*sin(x+(i/10.0) * 3.14)
        z = cos(x+(i/10.0) * 3.14)
        #build a key
        a=`i`
        #plot the data
        cSin[a] = demo.graph.newcurve('y = sin(x)'+a,x=x,y=y)
        cCos[a] = demo.graph.newcurve('y = cos(x)'+a,x=x,y=z)
    # insert a horizontal marker at y = 0
    mY = demo.graph.inserty1marker(0.0, 0.0, 'y = 0')
    demo.graph.setmarkerfollowmouse(mY,True)
    # insert a vertical marker at x = 2 pi
    mX = demo.graph.insertx1marker(2*pi, 0.0, label='x = 2 pi')
    demo.graph.setmarkerfollowmouse(mX,True)
    demo.graph.enablemarkermode()
    demo.graph.canvas().setMouseTracking(True)
    # replot
    #print dir(demo.graph)
    #print dir(demo.graph.curve)
    #print dir(demo.graph.curve(demo.graph.curves['y = sin(x)0']['curve']))
    #print dir(demo.graph.legend())

    #print demo.graph.curve(cSin[a]).dataSize()
    #first = 0
    #last  = 0
    #length,first,last = demo.graph.curve(cSin[a]).verifyRange(first,last)
    #print length,first,last
    
    #how to retrieve the data
    #print demo.graph.curve(cSin[a]).x
    if 0:
        for i in range(demo.graph.curve(1).dataSize()):
            print demo.graph.curve(1).x(i)
    
    demo.graph.replot()
    idata = arange(10000.)
    idata.shape = [100,100]
    demo.graph.imagePlot(idata)
    demo.graph.replot()
    demo.graph.setactivecurve('y = sin(x)3')
    print demo.graph.getactivecurve()
    
    #demo.graph.show()
    demo.show()
    return demo


def main(args):
    app = qt.QApplication(args)
    if qt.qVersion() < '4.0.0':
        demo = make0()
        def myslot(ddict):
            if ddict['event'] == "RemoveCurveEvent":
                #self.removeCurve
                demo.graph.delcurve(ddict['legend'])
                demo.graph.replot()
        qt.QObject.connect(demo.graph,qt.PYSIGNAL('QtBlissGraphSignal'), myslot)
        app.setMainWidget(demo)
        app.exec_loop()
    else:
        demo = make0()
        demo.resize(500, 300)
        # set axis titles
        def myslot(ddict):
            if ddict['event'] == "RemoveCurveEvent":
                #self.removeCurve
                demo.graph.delcurve(ddict['legend'])
                demo.graph.replot()
        demo.show()
        qt.QObject.connect(demo.graph,qt.SIGNAL('QtBlissGraphSignal'), myslot)
        sys.exit(app.exec_())


# Admire
if __name__ == '__main__':
    main(sys.argv)

