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
try:
    import PyQt4.Qt as qt
    #from PyQt4.Qwt5 import Qwt as qwt
    qt.PYSIGNAL = qt.SIGNAL
except:
    import qt
    #import qwt
from QtBlissGraph import qwt
import sys
import os
#from GraphWidget import *
from copy import *
import Numeric


class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=""):
        qt.QLineEdit.__init__(self,parent)

    def focusInEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('yellow'))

    def focusOutEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('white'))
        self.emit(qt.SIGNAL("returnPressed()"),())
        
    def setPaletteBackgroundColor(self, qcolor):
        if qt.qVersion() < '3.0.0':
            palette = self.palette()
            palette.setColor(qt.QColorGroup.Base,qcolor)
            self.setPalette(palette)
            text = self.text()
            self.setText(text)
        else:
            qt.QLineEdit.setPaletteBackgroundColor(self,qcolor)

"""
Manage colormap Widget class
"""
class ColormapDialog(qt.QDialog):
    def __init__(self, parent=None, name="Colormap Dialog"):
        if qt.qVersion() < '4.0.0':
            qt.QDialog.__init__(self, parent, name)
            self.setCaption(name)
        else:
            qt.QDialog.__init__(self, parent)
            self.setWindowTitle(name)
        self.title = name
                 
        
        self.colormapList = ["Greyscale", "Reverse Grey", "Temperature",
                             "Red", "Green", "Blue", "Many"]
                         

        # defaults values
        self.dataMin   = -10
        self.dataMax   = 10
        self.minValue  = 0
        self.maxValue  = 1
        self.colormapIndex  = 2

        self.autoscale   = False
        self.autoscale90 = False
        # main layout
        if qt.qVersion() < '4.0.0':
            vlayout = qt.QVBoxLayout(self, 0, -1, "Main ColormapDialog Layout")
        else:
            vlayout = qt.QVBoxLayout(self)
        vlayout.setMargin(10)

        # layout 1 : -combo to choose colormap
        #            -autoscale button
        #            -autoscale 90% button
        hbox1    = qt.QWidget(self)
        hlayout1 = qt.QHBoxLayout(hbox1)
        vlayout.addWidget(hbox1)
        hlayout1.setSpacing(10)

        # combo
        self.combo = qt.QComboBox(hbox1)
        for colormap in self.colormapList:
            if qt.qVersion() < '4.0.0':
                self.combo.insertItem(colormap)
            else:
                self.combo.addItem(colormap)
        self.connect(self.combo,
                     qt.SIGNAL("activated(int)"),
                     self.colormapChange)
        hlayout1.addWidget(self.combo)

        # autoscale
        self.autoScaleButton = qt.QPushButton("Autoscale", hbox1)
        if qt.qVersion() < '4.0.0':
            self.autoScaleButton.setToggleButton(True)
        else:
            self.autoScaleButton.setCheckable(True)
        self.connect(self.autoScaleButton,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscaleChange)
        hlayout1.addWidget(self.autoScaleButton)

        # autoscale 90%
        self.autoScale90Button = qt.QPushButton("Autoscale 90%", hbox1)
        if qt.qVersion() < '4.0.0':
            self.autoScale90Button.setToggleButton(True)
        else:
            self.autoScale90Button.setCheckable(True)
                
        self.connect(self.autoScale90Button,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscale90Change)
        hlayout1.addWidget(self.autoScale90Button)

        vlayout.addSpacing(20)

        # hlayout 2 : - min label
        #             - min texte
        hbox2    = qt.QWidget(self)
        hlayout2 = qt.QHBoxLayout(hbox2)
        vlayout.addWidget(hbox2)
        hlayout2.addStretch(10)
        
        self.minLabel  = qt.QLabel(hbox2)
        self.minLabel.setText("Minimum")
        hlayout2.addWidget(self.minLabel)
        
        hlayout2.addSpacing(5)
        hlayout2.addStretch(1)
        self.minText  = MyQLineEdit(hbox2)
        self.minText.setFixedWidth(150)
        self.minText.setAlignment(qt.Qt.AlignRight)
        self.connect(self.minText,
                     qt.SIGNAL("returnPressed()"),
                     self.minTextChanged)
        hlayout2.addWidget(self.minText)
        
        # hlayout 3 : - min label
        #             - min text
        hbox3    = qt.QWidget(self)
        hlayout3 = qt.QHBoxLayout(hbox3)
        vlayout.addWidget(hbox3)

        hlayout3.addStretch(10)
        self.maxLabel = qt.QLabel(hbox3)
        self.maxLabel.setText("Maximum")
        hlayout3.addWidget(self.maxLabel)
        
        hlayout3.addSpacing(5)
        hlayout3.addStretch(1)
                
        self.maxText = MyQLineEdit(hbox3)
        self.maxText.setFixedWidth(150)
        self.maxText.setAlignment(qt.Qt.AlignRight)

        self.connect( self.maxText,
                      qt.SIGNAL("returnPressed()"),
                      self.maxTextChanged)
        hlayout3.addWidget(self.maxText)


        # Graph widget for color curve...
        self.c = GraphWidget(self)
        self.c.enableXBottomAxis(1)
        self.c.setXLabel("Data Values")
        self.c.enableXTopAxis(0)
        self.c.enableYLeftAxis(0)
        self.c.enableYRightAxis(0)
        
        if qt.qVersion() < '4.0.0':
            self.c.setAutoLegend(0)
            self.c.enableLegend(0)
        
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge
        self.c.setZoom(self.minmd, self.maxpd, -11.5, 11.5)

        self.c.setMarkedCurve( "ConstrainedCurve",
                               [self.minmd, self.dataMin, self.dataMax, self.maxpd],
                               [-10, -10, 10, 10 ]
                             )
        self.c.markedCurves["ConstrainedCurve"].defConstraints(
                [(self.minmd,    self.minmd,   -10, -10 ),
                 (self.dataMin,  self.dataMax, -10, -10 ),
                (self.dataMin,  self.dataMax,  10,  10 ),
                (self.maxpd,    self.maxpd,    10,  10 )])
        
        self.c.setMinimumSize(qt.QSize(250,200))
        vlayout.addWidget(self.c)

        self.connect (self.c , qt.PYSIGNAL("PointMoved"),
                      self.chval)

        self.connect (self.c , qt.PYSIGNAL("PointReleased"),
                      self.chmap)

        # colormap window can not be resized
        self.setFixedSize(vlayout.minimumSize())

    def _update(self):
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge
        self.c.setZoom(self.minmd, self.maxpd, -11.5, 11.5)

        self.c.markedCurves["ConstrainedCurve"].defConstraints(
            [(self.minmd,    self.minmd,   -10, -10 ),
             (self.dataMin,  self.dataMax, -10, -10 ),
             (self.dataMin,  self.dataMax,  10,  10 ),
             (self.maxpd,    self.maxpd,    10,  10 )])

        self.c.markedCurves["ConstrainedCurve"].deplace(0, self.minmd, -10)
        self.c.markedCurves["ConstrainedCurve"].deplace(3, self.maxpd,  10)

    def chval(self, *args):
        (diam , x ,y) = (args[0], args[1], args[2])

        if diam == 2:
            self.setDisplayedMinValue(x)
        if diam == 3:
            self.setDisplayedMaxValue(x)

    def chmap(self, *args):
        (diam , x ,y) = (args[0], args[1], args[2])

        if diam == 2:
            self.setMinValue(x)
        if diam == 3:
            self.setMaxValue(x)
        
    """
    Colormap
    """
    def setColormap(self, colormap):
        self.colormapIndex = colormap
        if qt.qVersion() < '4.0.0':
            self.combo.setCurrentItem(colormap)
        else:
            self.combo.setCurrentIndex(colormap)
    
    def colormapChange(self, colormap):
        self.colormapIndex = colormap
        self.sendColormap()

    # AUTOSCALE
    """
    Autoscale
    """
    def autoscaleChange(self, val):
        self.autoscale = val
        self.setAutoscale(val)        
        self.sendColormap()

    def setAutoscale(self, val):
        if val:
            if qt.qVersion() < '4.0.0':
                self.autoScaleButton.setOn(True)
                self.autoScale90Button.setOn(False)
            else:
                self.autoScaleButton.setChecked(True)
                self.autoScale90Button.setChecked(False)
                #self.autoScale90Button.setDown(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax)

            self.maxText.setEnabled(0)
            self.minText.setEnabled(0)
            self.c.setEnabled(0)
        else:
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(1)
    """
    set rangeValues to dataMin ; dataMax-10%
    """
    def autoscale90Change(self, val):
        self.autoscale90 = val
        self.setAutoscale90(val)
        self.sendColormap()

    def setAutoscale90(self, val):
        if val:
            if qt.qVersion() < '4.0.0':
                self.autoScaleButton.setOn(False)
            else:
                self.autoScaleButton.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax - abs(self.dataMax/10))

            self.minText.setEnabled(0)
            self.maxText.setEnabled(0)
            self.c.setEnabled(0)
        else:
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            self.c.setEnabled(1)



    # MINIMUM
    """
    change min value and update colormap
    """
    def setMinValue(self, val):
        v = float(str(val))
        self.minValue = v
        self.minText.setText("%g"%v)
        self.c.markedCurves["ConstrainedCurve"].deplace(1, v, -10)
        self.sendColormap()

    """
    min value changed by text
    """
    def minTextChanged(self):
        val = float(str(self.minText.text()))
        self.setMinValue(val)
        if self.minText.hasFocus():
            self.c.setFocus()
        
    """
    change only the displayed min value
    """
    def setDisplayedMinValue(self, val):
        self.minValue = val
        self.minText.setText("%g"%val)

    # MAXIMUM
    """
    change max value and update colormap
    """
    def setMaxValue(self, val):
        v = float(str(val))
        self.maxValue = v
        self.maxText.setText("%g"%v)
        self.c.markedCurves["ConstrainedCurve"].deplace(2, v, 10)
        self.sendColormap()

    """
    max value changed by text
    """
    def maxTextChanged(self):
        val = float(str(self.maxText.text()))
        self.setMaxValue(val)
        if self.maxText.hasFocus():
            self.c.setFocus()
            
    """
    change only the displayed max value
    """
    def setDisplayedMaxValue(self, val):
        self.maxValue = val
        self.maxText.setText("%g"%val)

    # DATA values
    """
    set min/max value of data source
    """
    def setDataMinMax(self, minVal, maxVal):
        if minVal is not None:
            vmin = float(str(minVal))
            self.dataMin = vmin
        if maxVal is not None:
            vmax = float(str(maxVal))
            self.dataMax = vmax

        # are current values in the good range ?
        self._update()

    """
    send 'ColormapChanged' signal
    """
    def sendColormap(self):
        try:
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL("ColormapChanged"),
                        (self.colormapIndex, self.autoscale,
                         self.minValue, self.maxValue,
                         self.dataMin, self.dataMax))
            else:
                self.emit(qt.PYSIGNAL("ColormapChanged"),
                        self.colormapIndex, self.autoscale,
                         self.minValue, self.maxValue,
                         self.dataMin, self.dataMax)
            
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])

############## Graph Widget ############
#import qt
#import qtcanvas

if qt.qVersion() < '3.0.0':
    qt.QCursor.ArrowCursor   = qt.ArrowCursor
    qt.QCursor.UpArrowCursor = qt.UpArrowCursor
    qt.QCursor.WaitCursor    = qt.WaitCursor
    qt.QCursor.CrossCursor   = qt.CrossCursor
    qt.QCursor.SizeVerCursor = qt.SizeVerCursor
    qt.QCursor.SizeHorCursor = qt.SizeHorCursor
    qt.QCursor.PointingHandCursor = qt.ArrowCursor

# TODO
# enableOutline() is deprecated ...
# setMarkedCure()
#
# - possibility to move a line (ie: 2 egdes and a vertex)
# - (en/dis)able zoom / move

# FIXED :
# outline for the zoom but...
# zoomback to previous zoom ok 

class GraphError(Exception):
    def __init__(self, message):
        self.message= message
        
    def __str__(self):
        return "GraphError: %s"%self.message

class MarkedCurve:
    def __init__(self, graph, name, data1, data2, diams):
        self.name  = name
        self.data1 = data1
        self.data2 = data2
        self.diams = diams
        self.graph = graph
        self.constraints = None
        
    def getData(self):
        return (self.name, self.data1, self.data2, self.diams)

    def reDraw(self):
        self.graph.setCurve(self.name, self.data1, self.data2)

    def defConstraints(self, constraints):
        """constraints is a list of 4-uples constraints
        [(xmin, xmax, ymin, ymax),(....),]
        """
        self.constraints = constraints

    def movePoint(self, diam, x, y):
        a = self.diams.index(diam)
        if self.constraints != None:
            (xmin, xmax, ymin, ymax) = self.constraints[a]
            #print "contraintes :" , xmin, xmax, ymin, ymax
            if (x < xmin):
                x = xmin
            if (x > xmax):
                x = xmax
            if (y < ymin):
                y = ymin
            if (y > ymax):
                y = ymax
                
        self.data1[a] = x
        self.data2[a] = y

        return (diam, x, y)

    def deplace(self, point, x, y):
        diam = self.diams[point]
        self.movePoint(diam,x,y)
        self.graph._movedDiam = (diam, x, y)
        if qt.qVersion() < '4.0.0':
            self.graph.setMarkerPos(diam, x, y)
        else:
            diam.setXValue(x)
            diam.setYValue(y)
        self.reDraw()
        

class GraphWidget(qwt.QwtPlot):
    MouseActions  = [None, "zoom",  "xzoom"]
    CursorMarkers = [None, "cross", "vline", "hline"]

    modeList = [None, "selectdiamond", "movediamond", "diamond",
                "zoom", "zooming"]

    def __init__(self, parent = None, name = "GraphWidget", actions = None):
        qwt.QwtPlot.__init__(self, parent)
        
        self.parent  = parent
        self.actions = actions
        self.contextMenu = None
        self.actionColor = qt.QColor(qt.Qt.red)
        
        self.curveKeys  = {}
        self.activeName = None
        self.legendMenu = None
        
        self.isZooming = 0
        self.zoomStart = None
        self.zoomEnd   = None
        self.zoomState = None
        self.zoomStack = []
        
        self.zoomMarker    = None
        self.zoomMarkerKey = None
        
        self.cursorType    = None
        self.cursorVMarker = None
        self.cursorHMarker = None
        
        self.setDefaults()

        # allow to receive mouse move events even if no button pressed...
        self.canvas().setMouseTracking(1)

        # mouse signals
        self.connect(self,qt.SIGNAL("plotMouseMoved(const QMouseEvent&)"),
                     self._onMouseMoved)
        self.connect(self,qt.SIGNAL('plotMousePressed(const QMouseEvent&)'),
                     self._onMousePressed)
        self.connect(self,qt.SIGNAL('plotMouseReleased(const QMouseEvent&)'),
                     self._onMouseReleased)
        self.connect(self,qt.SIGNAL("legendClicked(long)"),
                     self._onLegend)

        # default mode
        # self.setMode("zoom")
        self.setMode("selectdiamond")
        
        # diamonds init
        self._selectedDiamond = None

        # list of (id marker , curve name) 2-uples
        self.diamondList = []
        
        self.markedCurves = {}
        

    def setModeCursor(self):
        """
        available cursors:
        
        Qt.ArrowCursor     Qt.UpArrowCursor      Qt.CrossCursor
        Qt.WaitCursor      Qt.SizeHorCursor      Qt.WhatsThisCursor
        Qt.SizeVerCursor   Qt.BusyCursor         Qt.SizeBDiagCursor
        Qt.SizeFDiagCursor Qt.SizeAllCursor      Qt.SplitVCursor
        Qt.SplitHCursor    Qt.PointingHandCursor Qt.ForbiddenCursor
        Qt.IbeamCursor

        """
        if qt.qVersion() > '4.0.0':
            qt.QCursor.CrossCursor    = qt.Qt.CrossCursor
            qt.QCursor.SizeHorCursor  = qt.Qt.SizeHorCursor
            qt.QCursor.UpArrowCursor  = qt.Qt.UpArrowCursor
            qt.QCursor.SizeAllCursor  = qt.Qt.SizeAllCursor
        if self.mode == "selectdiamond":
            self.canvas().setCursor(qt.QCursor(qt.QCursor.CrossCursor))

        if self.mode == "movediamond":
            if qt.qVersion() < '3.0.0':
                self.canvas().setCursor(qt.QCursor(qt.QCursor.SizeHorCursor))
            else:
                self.canvas().setCursor(qt.QCursor(qt.QCursor.SizeAllCursor))
            
        if self.mode == "diamond":
            self.canvas().setCursor(qt.QCursor(qt.QCursor.UpArrowCursor))
        
        if self.mode == "zooming":
            self.canvas().setCursor(qt.QCursor(qt.QPixmap(
                "../Icons/IconsLibrary/zoomrect.png")))
        
        if self.mode == "zoom":
            self.canvas().setCursor(qt.QCursor(qt.QPixmap(
                "../Icons/IconsLibrary/zoomrect.png")))
        

    def setMode(self, mode):
        """ define the working mode : selectdiamond, movediamond, diamond, zoom
        """
        if mode in self.modeList:
            self.mode = mode
            self.setModeCursor()

            if self.mode == "zoom" or  self.mode == "zooming" :
                self.enableOutline(1)
            else:
                if qt.qVersion() < '4.0.0':
                    self.enableOutline(0)
                        
        else:
            print "error invalid mode in setMode"
        


    def setContextMenu(self, menu):
        self.contextMenu = menu 

    def contentsContextMenuEvent(self, event):
        if self.contextMenu is not None:
            if event.reason() == qt.QContextMenuEvent.Mouse:
                self.contextMenu.exec_loop(QCursor.pos())

    # fonction a remonter dans MdiGraph ?
    def getPreviewPixmap(self):
        return qt.QPixmap(qt.QPixmap.grabWidget(self))


    def setDefaults(self):
        self.plotLayout().setMargin(0)
        self.plotLayout().setCanvasMargin(0)
        
        self.setCanvasBackground(qt.Qt.white)
        
        self.setTitle(" ")
        self.setXLabel("X axis")
        self.setYLabel("Y axis")
        
        self.enableXBottomAxis(1)
        self.enableXTopAxis(0)
        self.enableYLeftAxis(1)
        self.enableYRightAxis(0)
        
        self.setAxisAutoScale(qwt.QwtPlot.xBottom)
        self.setAxisAutoScale(qwt.QwtPlot.yLeft)
        
        if qt.qVersion() < '4.0.0':
            self.setLegendFrameStyle(qt.QFrame.Box)
            self.setLegendPos(qwt.Qwt.Bottom)
        
            self.useLegend(1)
            self.useLegendMenu(1)
        
        self.activePen= qt.QPen(qt.Qt.red, 2, qt.Qt.SolidLine)
        self.actionPen= qt.QPen(qt.Qt.black, 2, qt.Qt.DotLine)
        
        self.setMouseAction("xzoom")
        
        #        self.setCursorMarker("vline", 1)
        
    # 
    # AXIS
    #
    def setXLabel(self, label):
        self.setAxisTitle(qwt.QwtPlot.xBottom, label)
        
    def getXLabel(self):
        return str(self.axisTitle(qwt.QwtPlot.xBottom))

    def setYLabel(self, label):
        self.setAxisTitle(qwt.QwtPlot.yLeft, label)
        
    def getYLabel(self):
        return str(self.axisTitle(qwt.QwtPlot.yLeft))

    def getTitle(self):
        return str(self.title())

    def mapToY1(self, name):
        key = self.curveKeys[name]
        self.setCurveYAxis(key, qwt.QwtPlot.yLeft)
        self.__checkYAxis()
        self.replot()

    def mapToY2(self, name):
        key = self.curveKeys[name]
        if not self.axisEnabled(qwt.QwtPlot.yRight):
            self.enableAxis(qwt.QwtPlot.yRight, 1)

        self.setCurveYAxis(key, qwt.QwtPlot.yRight)
        self.__checkYAxis()
        self.resetZoom()

    def __checkYAxis(self):
        keys = self.curveKeys.values()
        right = 0
        left = 0
        for key in self.curveKeys.values():
            if self.curveYAxis(key) == qwt.QwtPlot.yRight:
                right += 1
            else:
                left += 1

        if not left:
            for key in keys:
                self.setCurveYAxis(key, qwt.QwtPlot.yLeft)
                self.enableAxis(qwt.QwtPlot.yRight, 0)
        else:
            if not right:
                self.enableAxis(qwt.QwtPlot.yRight, 0)

    #
    # LEGEND
    #
    def useLegend(self, yesno):
        if qt.qVersion() < '4.0.0':
            self.setAutoLegend(yesno)
            self.enableLegend(yesno)

    def __setLegendBold(self, name, yesno):
        """ <LEGEND> Set legend font to bold or not
        """
        key= self.curveKeys[name]
        if self.legendEnabled(key):
            item= self.legend().findItem(key)
            font= item.font()
            font.setBold(yesno)
            item.setFont(font)

    def useLegendMenu(self, yesno):
        if yesno:
            self.__useLegendMenu= 1
            if self.legendMenu is None:
                self.__createLegendMenu()
            else:
                pass
        else:
            self.__useLegendMenu= 0

    def __createLegendMenu(self):
        self.legendMenu= qt.QPopupMenu()
        self.legendMenuItems= {}
        self.legendMenuItems["active"]= self.legendMenu.insertItem(
            qt.QString("Set Active"),
            self.__legendSetActive
            )
        self.legendMenu.insertSeparator()
        self.legendMenuItems["mapy1"]= self.legendMenu.insertItem(
            qt.QString("Map to y1") ,self.__legendMapToY1
            )
        self.legendMenuItems["mapy2"]= self.legendMenu.insertItem(
            qt.QString("Map to y2") ,self.__legendMapToY2
            )
        self.legendMenu.insertSeparator()
        self.legendMenuItems["remove"]= self.legendMenu.insertItem(
            qt.QString("Remove"), self.__legendRemove
            )
        
    def __checkLegendMenu(self, name):
        self.legendMenu.setItemEnabled(
            self.legendMenuItems["active"], not (name==self.activeName))
        yaxis = self.curveYAxis(self.curveKeys[name])
        self.legendMenu.setItemEnabled( \
            self.legendMenuItems["mapy1"], \
            yaxis==qwt.QwtPlot.yRight and len(self.curveKeys.keys())>1)
        self.legendMenu.setItemEnabled(
            self.legendMenuItems["mapy2"],
            yaxis==qwt.QwtPlot.yLeft)

    
    def eventFilter(self, object, event):
        """ <LEGEND> Look for mouse event on legend item to display legend menu
        """
        if self.__useLegendMenu and \
               event.type() == qt.QEvent.MouseButtonRelease and \
               event.button() == qt.Qt.RightButton:
                   self.__legendName = str(object.text())
                   self.__checkLegendMenu(self.__legendName)
                   self.legendMenu.exec_loop(self.cursor().pos())
                   return 1
        return 0

    def __legendAddCurve(self, key):
        """ <LEGEND> To be called on curve creation to catch
        mouse event for legend menu
        """
        item = self.legend().findItem(key)
        if item is None:return
        item.setFocusPolicy(qt.QWidget.ClickFocus)
        if self.__useLegendMenu:
            item.installEventFilter(self)

    def __legendSetActive(self):
        """ <LEGEND>  menu callback
        """
        self.setActiveCurve(self.__legendName)

    def __legendMapToY1(self):
        """ <LEGEND> menu callback
        """
        self.mapToY1(self.__legendName)

    def __legendMapToY2(self):
        """ <LEGEND> menu callback
        """
        self.mapToY2(self.__legendName)

    def __legendRemove(self):
        """ <LEGEND> menu callback
        """
        self.delCurve(self.__legendName)

    #
    # CURVE
    #
    
    def setCurve(self, name, data1, data2=None):
        """ name = name of the curve also used in the legend
        data1 = x data if data2 is set, y data otherwise
        data2 = y data
        """
        if not self.curveKeys.has_key(name):
            curve = GraphCurve(self, name)
            if qt.qVersion() < '4.0.0':
                self.curveKeys[name] = self.insertCurve(curve)
        else:
            curve = self.curve(self.curveKeys[name])
            
        if data2 is None:
            y = Numeric.array(data1)
            x = Numeric.arange(len(y)).astype(Numeric.Float)
        else:
            x = Numeric.array(data1)
            y = Numeric.array(data2)

        curve.setData(x, y)
        if qt.qVersion() < '4.0.0':self.__legendAddCurve(self.curveKeys[name])
        self.setActiveCurve(name)

    def setMarkedCurve(self, name, data1, data2=None):
        diamTab = []
        
        if len(data1) > 100:
            print "warning it may be big..."
        
        if data2 is None:
            y = data1
            x = Numeric.arange(len(y)).astype(Numeric.Float)
        else:
            x = data1
            y = data2

        for i in range(len(x)):
            diam = self._drawDiamond(name, x[i], y[i])
            diamTab.append(diam)

        self.markedCurves[name] = MarkedCurve(self, name, x, y, diamTab) 

        self.markedCurves[name].reDraw()
        
    def delCurve(self, name):
        if self.curveKeys.has_key(name):

            if name == self.activeName:
                self.activeName= None

            self.removeCurve(self.curveKeys[name])
            del self.curveKeys[name]
            self.__checkYAxis()
            self.setActiveCurve(self.activeName)

    def setActiveCurve(self, name=None):
        if name is None:
            if len(self.curveKeys.keys()):
                name= self.curveKeys.keys()[0]

        if self.curveKeys.has_key(name):
            if self.activeName is not None:
                key= self.curveKeys[self.activeName]
                pen= self.curve(key).getSavePen()
                self.setCurvePen(key, pen)
                self.__setLegendBold(self.activeName, 0)
                
            key= self.curveKeys[name]
            curve= self.curve(key)
            curve.savePen()
            self.setCurvePen(key, self.activePen)
            self.__setLegendBold(name, 1)
            self.activeName= name
            self.replot()
            self.emit(qt.PYSIGNAL("GraphActive"), (self.activeName,))

    if qt.qVersion() > '4.0.0':
        #Qwt5
        def insertMarker(self):
            mX = qwt.QwtPlotMarker()
            #mX.setLabel(Qwt.QwtText(label))
            #mX.setLabelAlignment(qt.Qt.AlignRight | qt.Qt.AlignTop)
            #mX.setLineStyle(Qwt.QwtPlotMarker.VLine)
            return mX
            


    #
    # Diamonds
    #
    def _drawDiamond(self, curveName=None, x=0, y=0):
        """ add a diamond marker
        """
        self.marker = self.insertMarker()
        self.diamondList.append((self.marker, curveName))
        # print "self.marker, x, y", self.marker, x, y
        if qt.qVersion() < '4.0.0':
            self.setMarkerPos(self.marker, x, y)
            self.setMarkerLinePen(self.marker, 
                            qt.QPen(qt.Qt.green, 2, qt.Qt.DashDotLine))
            self.setMarkerSymbol(self.marker,
                                 qwt.QwtSymbol (qwt.QwtSymbol.Diamond, 
                                               qt.QBrush(qt.Qt.blue),
                                 qt.QPen(qt.Qt.red), qt.QSize(15,15)))

        else:
            self.marker.setXValue(x)
            self.marker.setYValue(y)

        self.replot()
        
        return self.marker

    def _delDiamond(self, markerRef):
        """ Delete the marker with ref markerRef
        """
        pass

    def _moveDiamond(self, diam, x, y):
        """ move the marker diam to position (x, y)
            Update the related curve
        """
        # update the data of the marked curve...
        for (diamond, curve) in self.diamondList:
            if diamond == diam:
                (diam, x, y) = self.markedCurves[curve].movePoint(diam, x, y)
                self.markedCurves[curve].reDraw()
                self._movedDiam = (diam, x, y)
        # move the marker 
        self.setMarkerPos(diam, x, y)

    def _highlightDiamondOn(self, marker):
        """ highlight a marker in green"""
        self.setMarkerSymbol( marker,
                              qwt.QwtSymbol (qwt.QwtSymbol.Diamond,
                                         qt.QBrush(qt.Qt.green),
                                         qt.QPen(qt.Qt.green),
                                         qt.QSize(15,15))
                              )
        self.replot()

    def _highlightDiamondOff(self, marker):
        """ redraw a marker back in red/blue"""
        self.setMarkerSymbol( marker,
                              qwt.QwtSymbol (qwt.QwtSymbol.Diamond,
                                         qt.QBrush(qt.Qt.blue),
                                         qt.QPen(qt.Qt.red),
                                         qt.QSize(15,15))
                              )
        self.replot()

    #
    # MOUSE CALLBACKS
    #

    # mouse MOVE
    def _onMouseMoved(self, event):
        xpixel = event.pos().x()
        ypixel = event.pos().y()

        xdata = self.invTransform(qwt.QwtPlot.xBottom, xpixel)
        ydata = self.invTransform(qwt.QwtPlot.yLeft, ypixel)

        replot = 0
        
        if self.mode == "selectdiamond":
            pass

        if self.mode == "movediamond":
            self._moveDiamond(self._selectedDiamond, xdata, ydata)
            self.emit(qt.PYSIGNAL("PointMoved"), self._movedDiam)
            replot = 1
            
        if self.__updateCursorMarker(xdata, ydata):
            replot = 1

        if self.isZooming and self.zoomMarker is not None:
            self.__updateZoomMarker(xdata, ydata)
            replot = 1
                
        if self.mode == "zooming":
            pass
            
        # replot if needed
        if replot:
            self.replot()
            
        pos = {"data": (xdata, ydata), "pixel": (xpixel, ypixel)}
        self.emit(qt.PYSIGNAL("GraphPosition"), (pos,))

    # mouse PRESSED
    def _onMousePressed(self, event):
        if event.button() == qt.Qt.LeftButton:
            self._onMouseLeftPressed(event)
        elif event.button() == qt.Qt.RightButton:
            self._onMouseRightPressed(event)
        elif event.button() == qt.Qt.MidButton:
            self._onMouseMidPressed(event)
        
    def _onMouseLeftPressed(self, event):

        x = self.invTransform(qwt.QwtPlot.xBottom, event.x())
        y = self.invTransform(qwt.QwtPlot.yLeft, event.y())

        # print "mouseLeftPressed:", event.x(), event.y(),
        #         "data:", x, y, "mode:", self.mode

        if self.mode == "diamond":
            self._drawDiamond(None, x, y)

        if self.mode == "movediamond" or self.mode == "zooming":
            print " y a comme une erreur...  mode=", self.mode

        if self.mode == "selectdiamond":
            (cmarker, distance) = self.closestMarker(event.x(), event.y())
            if distance < 8:
                self.setMode("movediamond")
                self._selectedDiamond = cmarker
                self._highlightDiamondOn(self._selectedDiamond)
                
        if self.mode == "zoom":
            self._zoomPointOne = (x,y)
            self.setMode ("zooming")
            
        if self.mode == "selection":
            pass
    
    def _onMouseRightPressed(self, event):
        self.zoomBack()
        
    def _onMouseMidPressed(self, event):
        self.nextMode()

    def nextMode(self):
        print " old mode = " , self.mode
        oldmode = self.mode
        if oldmode == "zoom":
            self.setMode("selectdiamond")
        elif oldmode == "selectdiamond":
            self.setMode("zoom")
            
        print " new mode = " , self.mode

        
    # mouse RELEASED
    def _onMouseReleased(self, event):
        if event.button() == qt.Qt.LeftButton:
            self._onMouseLeftReleased(event)
        elif event.button() == qt.Qt.RightButton:
            self._onMouseRightReleased(event)

    def _onMouseLeftReleased(self, event):
        if self.mode == "movediamond":
            self.setMode ("selectdiamond")
            self._highlightDiamondOff(self._selectedDiamond)
            self._movingDiamond = None
            self.emit(qt.PYSIGNAL("PointReleased"), self._movedDiam)

        if self.mode == "zooming":
            (xmin, ymin) = self._zoomPointOne
            xmax = self.invTransform(qwt.QwtPlot.xBottom, event.x())
            ymax = self.invTransform(qwt.QwtPlot.yLeft, event.y())
            self.setZoom(xmin, xmax, ymin, ymax)
            self.setMode("zoom")

    def _onMouseRightReleased(self, event):
        pass

    #
    # legend ?
    #
    def _onLegend(self, item):
        curve = self.curve(item)
        name  = str(curve.title())
        self.setActiveCurve(name)

    #
    # ZOOM
    #

    # ??? unused ?
    def __startZoom(self, x, y):
        self.isZooming= 1
        xpos = self.invTransform(qwt.QwtPlot.xBottom, x)
        ypos = self.invTransform(qwt.QwtPlot.yLeft, y)

        if self.axisEnabled(qwt.QwtPlot.yRight):
            y2pos = self.invTransform(qwt.QwtPlot.yRight, y)
        else:
            y2pos = 0

        self.zoomStart = (xpos, ypos, y2pos)

        marker = self.zoomMarkerClass(self)
        self.zoomMarker = self.insertMarker(marker)

    # ??? unused ?
    def __stopZoom(self, x, y):
        self.isZooming = 0
        xpos = self.invTransform(qwt.QwtPlot.xBottom, x)
        ypos = self.invTransform(qwt.QwtPlot.yLeft, y)
        if self.axisEnabled(qwt.QwtPlot.yRight):
            y2pos = self.invTransform(qwt.QwtPlot.yRight, y)
        else:
            y2pos = 0
        self.zoomEnd = (xpos, ypos, y2pos)

        self.removeMarker(self.zoomMarker)
        self.zoomMarker = None

    def __updateZoomMarker(self, x, y):
        self.setMarkerPos(self.zoomMarker, x, y)

    def getZoom(self):
        if self.axisAutoScale(qwt.QwtPlot.xBottom):
            xmin= None
            xmax= None
        else:
            if qt.qVersion() < '4.0.0':
                xmin= self.axisScale(qwt.QwtPlot.xBottom).lBound()
                xmax= self.axisScale(qwt.QwtPlot.xBottom).hBound()
            else:
                xmin = self.canvasMap(qwt.QwtPlot.xBottom).s1()
                xmax = self.canvasMap(qwt.QwtPlot.xBottom).s2()
                self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)


        if self.axisAutoScale(qwt.QwtPlot.yLeft):
            ymin= None
            ymax= None
        else:
            if qt.qVersion() < '4.0.0':
                ymin= self.axisScale(qwt.QwtPlot.yLeft).lBound()
                ymax= self.axisScale(qwt.QwtPlot.yLeft).hBound()
            else:
                ymax = self.canvasMap(qwt.QwtPlot.yLeft).s2()
                ymin = self.canvasMap(qwt.QwtPlot.yLeft).s1()
                self.setAxisScale(qwt.QwtPlot.yRight, ymin, ymax)



        if 0:
        #if self.axisEnabled(qwt.QwtPlot.yRight):
            if self.axisAutoScale(qwt.QwtPlot.yRight):
                y2min= None
                y2max= None
            else:
                y2min= self.axisScale(qwt.QwtPlot.yRight).lBound()
                y2max= self.axisScale(qwt.QwtPlot.yRight).hBound()
        else:
            y2min= 0
            y2max= 0

        return (xmin, xmax, ymin, ymax, y2min, y2max)

    def setZoom(self, xmin, xmax, ymin, ymax, y2min=0, y2max=0):
        self.zoomStack.append(self.getZoom())
        self.__setZoom(xmin, xmax, ymin, ymax, y2min, y2max)

    def __setZoom(self, xmin, xmax, ymin, ymax, y2min=0, y2max=0):
        if xmin is None or xmax is None:
            self.setAxisAutoScale(qwt.QwtPlot.xBottom)
        else:
            self.setAxisScale(qwt.QwtPlot.xBottom, xmin, xmax)

        if ymin is None or ymax is None:
            self.setAxisAutoScale(qwt.QwtPlot.yLeft)
        else:
            self.setAxisScale(qwt.QwtPlot.yLeft, ymin, ymax)

        if self.axisEnabled(qwt.QwtPlot.yRight):
            if y2min is None or y2max is None:
                self.setAxisAutoScale(qwt.QwtPlot.yRight)
            else:
                self.setAxisScale(qwt.QwtPlot.yRight, y2min, y2max)
                
        self.replot()

    def setXZoom(self, xmin, xmax):
        self.setZoom(xmin, xmax, None, None)

    def setYZoom(self, ymin, ymax):
        self.setZoom(None, None, ymin, ymax)

    def zoomBack(self):
        if len(self.zoomStack)>1:
            zoom= self.zoomStack[-1]
            self.__setZoom(zoom[0], zoom[1], zoom[2], zoom[3], zoom[4], zoom[5])
            self.zoomStack.remove(zoom)
        else:
            self.resetZoom()

    def resetZoom(self):
        self.setAxisAutoScale(qwt.QwtPlot.xBottom)
        self.setAxisAutoScale(qwt.QwtPlot.yLeft)
        if self.axisEnabled(qwt.QwtPlot.yRight):
            self.setAxisAutoScale(qwt.QwtPlot.yRight)

        self.zoomStack= []
        self.replot()

    #
    # MOUSE ACTION
    #
    def setMouseAction(self, action):
        if action in self.MouseActions:
            self.mouseAction = self.MouseActions.index(action)
            self.isZooming = 0
            
            if self.mouseAction == 1:
                self.zoomMarkerClass = GraphRectMarker
            elif self.mouseAction == 2:
                self.zoomMarkerClass = GraphXRectMarker

    #
    # GRIDLINES
    #
    def setGridLines(self, xgrid, ygrid=None):
        if ygrid is None:
            ygrid = xgrid

        self.enableGridX(xgrid)
        self.enableGridY(ygrid)

    def getGridLines(self):
        return (self.gridXEnabled(), self.gridYEnabled())
    

    #
    # GRAPHIC CURSOR
    #
    def setCursorMarker(self, type = None, text = 0):
        if type not in self.CursorMarkers:
            raise GraphError("Invalid Cursor type <%s>"%type)

        self.cursorType = type
        self.cursorText = text

        if self.cursorVMarker is not None:
            self.removeMarker(self.cursorVMarker)

        if self.cursorHMarker is not None:
            self.removeMarker(self.cursorHMarker)

        if self.cursorType == "cross" or self.cursorType == "vline":
            self.cursorVMarker= self.insertLineMarker("", qwt.QwtPlot.xBottom)

        if self.cursorType == "cross" or self.cursorType == "hline":
            self.cursorHMarker= self.insertLineMarker("", qwt.QwtPlot.yLeft)

        self.replot()

    def __updateCursorMarker(self, xdata, ydata):

        if self.cursorType is None:
            return 0

        replot = 0
    
        if self.cursorVMarker is not None:
            self.setMarkerXPos(self.cursorVMarker, xdata)
            if self.cursorText:
                self.setMarkerLabelText(self.cursorVMarker, "%g"%xdata )
            replot = 1
                
        if self.cursorHMarker is not None:
            self.setMarkerYPos(self.cursorHMarker, ydata)
            if self.cursorText:
                self.setMarkerLabelText(self.cursorHMarker, "%g"%ydata )
            replot = 1
                
        return replot

class GraphRectMarker(qwt.QwtPlotMarker):
    def __init__(self, plot):
        qwt.QwtPlotMarker.__init__(self, plot)
        self.x0= None
        self.x1= None
        self.y0= None
        self.y1= None

    def draw(self, painter, x, y, rect):
        if self.x0 is None:
            self.x0= x

        self.y0= y
        self.x1= x
        self.y1= y
        painter.drawRect(self.x0, self.y0, self.x1-self.x0, self.y1-self.y0)

class GraphXRectMarker(qwt.QwtPlotMarker):
    def __init__(self, plot):
        qwt.QwtPlotMarker.__init__(self, plot)
        self.x0= None
        self.x1= None

    def draw(self, painter, x, y, rect):
        if self.x0 is None:
            self.x0= x

        self.x1= x
        painter.drawRect(self.x0, rect.top(), self.x1-self.x0, rect.height())

class GraphCurve(qwt.QwtPlotCurve):
    def __init__(self, parent, name):
        if qt.qVersion() < '4.0.0':
            qwt.QwtPlotCurve.__init__(self, parent, name)
        else:
            qwt.QwtPlotCurve.__init__(self, name)
        self.save= None

    def savePen(self):
        self.save= qt.QPen(self.pen())

    def getSavePen(self):
        return self.save

def test():
    import sys

    app = qt.QApplication(sys.argv)
    app.connect(app,qt.SIGNAL("lastWindowClosed()"), app.quit)
    wid = GraphWidget()
    
##     for i in range(2):
##         x = arrayrange(0.0, 10.0, 0.1)
##         y = sin(x+(i/10.0) * 3.14) * (i+1)
##         z = cos(x+(i/10.0) * 3.14) * (i+1)
    
##         wid.setCurve("sin%d"%i, x, y)
##         wid.setCurve("cos%d"%i, x, z)

    # sample of curve 
#    wid.setCurve("matrace", [ 0,1,2,3,4,5,6,7,8,9] , [ 0,1,4,9,10,8,7,5,2,5])

    # sample of marked curve
#    a = array([0.0, 1.0, 2.0, 4.0, 8.0, 10.0 ])
#    wid.setMarkedCurve( "mysampleTrace", a )

    # sample of constrained marked curve
    wid.setMarkedCurve( "ConstrainedCurve", [0,2,8,10], [0,0,4,4] )

    wid.markedCurves["ConstrainedCurve"].defConstraints(
        [(0,0,0,0),(2,8,0,0),(2,8,4,4),(10,10,4,4)] )

    # move point 2 to position 5,5
    wid.markedCurves["ConstrainedCurve"].deplace( 2, 5, 4)

    wid.setZoom(-0.5, 10.5, -0.5, 4.5)
    wid.show()

    app.setMainWidget(wid)
    app.exec_loop()



#if __name__=="__main__":
#    test()

############## end graphwidget #########        
if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    demo = ColormapDialog()
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(demo)
    #demo.exec_loop()
    demo.show()
    if qt.qVersion() < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()
