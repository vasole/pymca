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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import os
import numpy
from PyMca import RGBCorrelatorGraph
qt = RGBCorrelatorGraph.qt
IconDict = RGBCorrelatorGraph.IconDict
QTVERSION = qt.qVersion()
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str
MATPLOTLIB = False
if QTVERSION > '4.0.0':
    try:
        from PyMca import QPyMcaMatplotlibSave
        MATPLOTLIB = True
    except ImportError:
        MATPLOTLIB = False
else:
    qt.QIcon = qt.QIconSet
from PyMca import ColormapDialog
from PyMca import spslut
from PyMca import PyMcaDirs
from PyMca import ArraySave
try:
    from PyMca import ProfileScanWidget
except ImportError:
    print("MaskImageWidget importing ProfileScanWidget directly")
    import ProfileScanWidget
try:
    from PyMca import SpecfitFuns
except ImportError:
    print("MaskImageWidget importing SpecfitFuns directly")
    import SpecfitFuns


COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]

if QTVERSION > '4.0.0':
    from PyQt4 import Qwt5
    try:
        from PyMca import QwtPlotItems
        OVERLAY_DRAW = True
    except ImportError:
        OVERLAY_DRAW = False
else:
    OVERLAY_DRAW = False
    import Qwt5

DEFAULT_COLORMAP_INDEX = 2
DEFAULT_COLORMAP_LOG_FLAG = False
DEBUG = 0


# set this variable to false if you get crashes when moving the mouse
# over the images.
# Before I thought it had to do with the Qt version used, but it seems
# to be related to old PyQwt versions. (In fact, the 5.2.1 version is
# a recent snapshot)
if Qwt5.QWT_VERSION_STR < '5.2.1':
    USE_PICKER = False
else:
    USE_PICKER = True

# Uncomment next line if you experience crashes moving the mouse on
# top of the images
#USE_PICKER = True

def convertToRowAndColumn(x, y, shape, xScale=None, yScale=None, safe=True):
    if xScale is None:
        c = x
    else:
        if x < xScale[0]:
            x = xScale[0]        
        c = shape[1] *(x - xScale[0]) / (xScale[1] - xScale[0])
    if yScale is None:
        r = y
    else:
        if y < yScale[0]:
            y = yScale[0]        
        r = shape[0] *(y - yScale[0]) / (yScale[1] - yScale[0])

    if safe:
        c = min(int(c), shape[1] - 1)
        r = min(int(r), shape[0] - 1)
    return r, c

class MyPicker(Qwt5.QwtPlotPicker):
    def __init__(self, *var):
        Qwt5.QwtPlotPicker.__init__(self, *var)
        self.__text = Qwt5.QwtText()
        self.data = None
        self.xScale = None
        self.yScale = None

    if USE_PICKER:
        def trackerText(self, var):
            d=self.invTransform(var)
            if self.data is None:
                self.__text.setText("%g, %g" % (d.x(), d.y()))
            else:
                x = d.x()
                y = d.y()
                r, c = convertToRowAndColumn(x, y, self.data.shape,
                                             xScale=self.xScale,
                                             yScale=self.yScale, safe=True)
                z = self.data[r, c]
                self.__text.setText("%.1f, %.1f, %.4g" % (x, y, z))
            return self.__text
    
class MaskImageWidget(qt.QWidget):
    def __init__(self, parent = None, rgbwidget=None, selection=True, colormap=False,
                 imageicons=True, standalonesave=True, usetab=False,
                 profileselection=False, scanwindow=None):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
            self.setCaption("PyMca - Image Selection Tool")
            profileselection = False
        else:
            self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
            self.setWindowTitle("PyMca - Image Selection Tool")
            if 0:
                screenHeight = qt.QDesktopWidget().height()
                if screenHeight > 0:
                    self.setMaximumHeight(int(0.99*screenHeight))
                    self.setMinimumHeight(int(0.5*screenHeight))
                screenWidth = qt.QDesktopWidget().width()
                if screenWidth > 0:
                    self.setMaximumWidth(int(screenWidth)-5)
                    self.setMinimumWidth(min(int(0.5*screenWidth),800))

        self._y1AxisInverted = False
        self.__selectionMask = None
        self.__imageData = None
        self.__image = None
        self._xScale = None
        self._yScale = None

        self.colormap = None
        self.colormapDialog = None
        self.setDefaultColormap(DEFAULT_COLORMAP_INDEX,
                                DEFAULT_COLORMAP_LOG_FLAG)
        self.rgbWidget = rgbwidget

        self.__imageIconsFlag = imageicons
        self.__selectionFlag = selection
        self.__useTab = usetab
        self.mainTab = None

        self._build(standalonesave, profileselection=profileselection)
        self._profileSelectionWindow = None
        self._profileScanWindow = scanwindow

        self.__brushMenu  = None
        self.__brushMode  = False
        self.__eraseMode  = False
        self.__connected = True

        self.__setBrush2()
        
        self.outputDir   = None
        self._saveFilter = None
        
        self._buildConnections()
        self._matplotlibSaveImage = None

        # the overlay items to be drawn
        self._overlayItemsDict = {}
        # the last overlay legend used
        self.__lastOverlayLegend = None
        # the projection mode
        self.__lineProjectionMode = 'D'

    def _build(self, standalonesave, profileselection=False):
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        if self.__useTab:
            self.mainTab = qt.QTabWidget(self)
            #self.graphContainer =qt.QWidget()
            #self.graphContainer.mainLayout = qt.QVBoxLayout(self.graphContainer)
            #self.graphContainer.mainLayout.setMargin(0)
            #self.graphContainer.mainLayout.setSpacing(0)
            self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self,
                                                   selection = self.__selectionFlag,
                                                   colormap=True,
                                                   imageicons=self.__imageIconsFlag,
                                                   standalonesave=False,
                                                   standalonezoom=False,
                                                   profileselection=profileselection)
            self.mainTab.addTab(self.graphWidget, 'IMAGES')
        else:
            if QTVERSION < '4.0.0':
                self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self,
                                                   selection = self.__selectionFlag,
                                                   colormap=True,
                                                   imageicons=self.__imageIconsFlag,
                                                   standalonesave=True,
                                                   standalonezoom=False)
                standalonesave = False
            else:
                self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self,
                                                   selection =self.__selectionFlag,
                                                   colormap=True,
                                                   imageicons=self.__imageIconsFlag,
                                                   standalonesave=False,
                                                   standalonezoom=False,
                                                   profileselection=profileselection)

        #for easy compatibility with RGBCorrelatorGraph
        self.graph = self.graphWidget.graph

        if profileselection:
            self.connect(self.graphWidget,
                         qt.SIGNAL('PolygonSignal'), 
                         self._polygonSignalSlot)

        if standalonesave:
            self.buildStandaloneSaveMenu()

        self.connect(self.graphWidget.zoomResetToolButton,
                     qt.SIGNAL("clicked()"), 
                     self._zoomResetSignal)
        self.graphWidget.picker = MyPicker(Qwt5.QwtPlot.xBottom,
                           Qwt5.QwtPlot.yLeft,
                           Qwt5.QwtPicker.NoSelection,
                           Qwt5.QwtPlotPicker.CrossRubberBand,
                           Qwt5.QwtPicker.AlwaysOn,
                           self.graphWidget.graph.canvas())
        self.graphWidget.picker.setTrackerPen(qt.QPen(qt.Qt.black))
        self.graphWidget.graph.enableSelection(False)
        self.graphWidget.graph.enableZoom(True)
        if self.__selectionFlag:
            if self.__imageIconsFlag:
                self.setSelectionMode(False)
            else:
                self.setSelectionMode(True)
            self._toggleSelectionMode()
        if self.__useTab:
            self.mainLayout.addWidget(self.mainTab)
        else:
            self.mainLayout.addWidget(self.graphWidget)

    def buildStandaloneSaveMenu(self):
        self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
        self._saveMenu = qt.QMenu()
        self._saveMenu.addAction(QString("Image Data"),
                                 self.saveImageList)
        self._saveMenu.addAction(QString("Colormap Clipped Seen Image Data"),
                                 self.saveClippedSeenImageList)
        self._saveMenu.addAction(QString("Clipped and Subtracted Seen Image Data"),
                                 self.saveClippedAndSubtractedSeenImageList)

        self._saveMenu.addAction(QString("Standard Graphics"),
                                 self.graphWidget._saveIconSignal)
        if QTVERSION > '4.0.0':
            if MATPLOTLIB:
                self._saveMenu.addAction(QString("Matplotlib") ,
                                 self._saveMatplotlibImage)

    def _buildConnections(self, widget = None):
        self.connect(self.graphWidget.hFlipToolButton,
                 qt.SIGNAL("clicked()"),
                 self._hFlipIconSignal)

        self.connect(self.graphWidget.colormapToolButton,
                     qt.SIGNAL("clicked()"),
                     self.selectColormap)

        if self.__selectionFlag:
            self.connect(self.graphWidget.selectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self._toggleSelectionMode)
            text = "Toggle between Selection\nand Zoom modes"
            if QTVERSION > '4.0.0':
                self.graphWidget.selectionToolButton.setToolTip(text)

        if self.__imageIconsFlag:
            self.connect(self.graphWidget.imageToolButton,
                     qt.SIGNAL("clicked()"),
                     self._resetSelection)

            self.connect(self.graphWidget.eraseSelectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self._setEraseSelectionMode)

            self.connect(self.graphWidget.rectSelectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self._setRectSelectionMode)

            self.connect(self.graphWidget.brushSelectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self._setBrushSelectionMode)

            self.connect(self.graphWidget.brushToolButton,
                     qt.SIGNAL("clicked()"),
                     self._setBrush)

        if QTVERSION < "4.0.0":
            self.connect(self.graphWidget.graph,
                         qt.PYSIGNAL("QtBlissGraphSignal"),
                         self._graphSignal)
        else:
            if self.__imageIconsFlag:
                self.connect(self.graphWidget.additionalSelectionToolButton,
                         qt.SIGNAL("clicked()"),
                         self._additionalSelectionMenuDialog)
                self._additionalSelectionMenu = qt.QMenu()
                self._additionalSelectionMenu.addAction(QString("Reset Selection"),
                                                        self._resetSelection)
                self._additionalSelectionMenu.addAction(QString("Invert Selection"),
                                                        self._invertSelection)
                self._additionalSelectionMenu.addAction(QString("I >= Colormap Max"),
                                                        self._selectMax)
                self._additionalSelectionMenu.addAction(QString("Colormap Min < I < Colormap Max"),
                                                        self._selectMiddle)
                self._additionalSelectionMenu.addAction(QString("I <= Colormap Min"),
                                                        self._selectMin)

            self.connect(self.graphWidget.graph,
                     qt.SIGNAL("QtBlissGraphSignal"),
                     self._graphSignal)

    def _polygonSignalSlot(self, ddict):
        if DEBUG:
            print("_polygonSignalSLot, event = %s" % ddict['event'])
            print("Received ddict = ", ddict)

        if ddict['event'] in [None, "NONE"]:
            #Nothing to be made
            return

        if ddict['event'] == "PolygonWidthChanged":
            if self.__lastOverlayLegend is not None:
                legend = self.__lastOverlayLegend
                if legend in self._overlayItemsDict:
                    info = self._overlayItemsDict[legend]['info']
                    if info['mode'] == ddict['mode']:
                        newDict = {}
                        newDict.update(info)
                        newDict['pixelwidth'] = ddict['pixelwidth']
                        self._polygonSignalSlot(newDict)
            return

        if self._profileSelectionWindow is None:
            if self._profileScanWindow is None:
                #identical to the standard scan window
                self._profileSelectionWindow = ProfileScanWidget.ProfileScanWidget(actions=False)
            else:
                self._profileSelectionWindow = ProfileScanWidget.ProfileScanWidget(actions=True)
                self.connect(self._profileSelectionWindow,
                             qt.SIGNAL('addClicked'),
                             self._profileSelectionSlot)
                self.connect(self._profileSelectionWindow,
                             qt.SIGNAL('removeClicked'),
                             self._profileSelectionSlot)
                self.connect(self._profileSelectionWindow,
                             qt.SIGNAL('replaceClicked'),
                             self._profileSelectionSlot)
            self._interpolate =  SpecfitFuns.interpol
            #if I do not return here and the user interacts with the graph while
            #the profileSelectionWindow is not shown, I get crashes under Qt 4.5.3 and MacOS X
            #when calling _getProfileCurve
            return


        self._updateProfileCurve(ddict)

    def _updateProfileCurve(self, ddict):
        curve = self._getProfileCurve(ddict)
        if curve is None:
            return
        xdata, ydata, legend, info = curve            
        replot=True
        replace=True
        idx = numpy.isfinite(ydata)
        xdata = xdata[idx]
        ydata = ydata[idx]
        self._profileSelectionWindow.addCurve(xdata, ydata,
                                              legend=legend,
                                              info=info,
                                              replot=replot,
                                              replace=replace)

    def getGraphTitle(self):
        try:
            title = self.graphWidget.graph.title().text()
            if sys.version < '3.0':
                title = qt.safe_str(title)
        except:
            title = ""
        return title

    def setLineProjectionMode(self, mode):
        """
        Set the line projection mode.

        mode: 1 character string. Allowed options 'D', 'X' 'Y'
        D - Plot the intensity over the drawn line over as many intervals as pixels over the axis
            containing the longest projection in pixels.
        X - Plot the intensity over the drawn line over as many intervals as pixels over the X axis
        Y - Plot the intensity over the drawn line over as many intervals as pixels over the Y axis
        """
        m = mode.upper()
        if m not in ['D', 'X', 'Y']:
            raise ValueError("Invalid mode %s. It has to be 'D', 'X' or 'Y'")
        self.__lineProjectionMode = m

    def getLineProjectionMode(self):
        return self.__lineProjectionMode

    def _getProfileCurve(self, ddict, image=None, overlay=OVERLAY_DRAW):
        if image is None:
            imageData = self.__imageData
        else:
            imageData = image
        if imageData is None:
            return None

        title = self.getGraphTitle()

        self._profileSelectionWindow.setTitle(title)
        if self._profileScanWindow is not None:
            self._profileSelectionWindow.label.setText(title)

        #showing the profileSelectionWindow now can make the program crash if the workaround mentioned above
        #is not implemented
        self._profileSelectionWindow.show()
        #self._profileSelectionWindow.raise_()

        if ddict['event'] == 'PolygonModeChanged':
            return

        #if I show the image here it does not crash, but it is not nice because
        #the user would get the profileSelectionWindow under his mouse
        #self._profileSelectionWindow.show()

        shape = imageData.shape
        width = ddict['pixelwidth'] - 1
        if ddict['mode'].upper() in ["HLINE", "HORIZONTAL"]:
            xLabel = self.getXLabel()
            deltaDistance = 1.0
            if width < 1:
                row = int(ddict['row'][0])
                if row < 0:
                    row = 0
                if row >= shape[0]:
                    row = shape[0] - 1
                ydata  = imageData[row, :]
                legend = "Row = %d"  % row
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([0.0, shape[1], shape[1], 0.0],
                                         [row, row, row+1, row+1],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            else:
                row0 = int(ddict['row'][0]) - 0.5 * width
                if row0 < 0:
                    row0 = 0
                    row1 = row0 + width
                else:
                    row1 = int(ddict['row'][0]) + 0.5 * width
                if row1 >= shape[0]:
                    row1 = shape[0] - 1
                    row0 = max(0, row1 - width)
                ydata = imageData[row0:int(row1+1), :].sum(axis=0)
                legend = "Row = %d to %d"  % (row0, row1)
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([0.0, 0.0, shape[1], shape[1]],
                                         [row0, row1, row1, row0],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            xdata  = numpy.arange(shape[1]).astype(numpy.float)
            if self._xScale is not None:
                xdata = self._xScale[0] + xdata * (self._xScale[1] - self._xScale[0]) / float(shape[1])
        elif ddict['mode'].upper() in ["VLINE", "VERTICAL"]:
            xLabel = self.getYLabel()
            deltaDistance = 1.0
            if width < 1:
                column = int(ddict['column'][0])
                if column < 0:
                    column = 0
                if column >= shape[1]:
                    column = shape[1] - 1
                ydata  = imageData[:, column]
                legend = "Column = %d"  % column
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([column, column, column+1, column+1],
                                         [0.0, shape[0], shape[0], 0.0],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            else:
                col0 = int(ddict['column'][0]) - 0.5 * width
                if col0 < 0:
                    col0 = 0
                    col1 = col0 + width
                else:
                    col1 = int(ddict['column'][0]) + 0.5 * width
                if col1 >= shape[1]:
                    col1 = shape[1] - 1
                    col0 = max(0, col1 - width)
                ydata = imageData[:, col0:int(col1+1)].sum(axis=1)
                legend = "Col = %d to %d"  % (col0, col1)
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([col0, col0, col1, col1],
                                         [0, shape[0], shape[0], 0.],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            xdata  = numpy.arange(shape[0]).astype(numpy.float)
            if self._yScale is not None:
                xdata = self._yScale[0] + xdata * (self._yScale[1] - self._yScale[0]) / float(shape[0])
        elif ddict['mode'].upper() in ["LINE"]:
            if len(ddict['column']) == 1:
                #only one point given
                return
            #the coordinates of the reference points
            x0 = numpy.arange(float(shape[0]))
            y0 = numpy.arange(float(shape[1]))
            #get the interpolation points
            col0, col1 = [int(x) for x in ddict['column']]
            row0, row1 = [int(x) for x in ddict['row']]
            deltaCol = abs(col0 - col1)
            deltaRow = abs(row0 - row1)
            if self.__lineProjectionMode == 'D':
                if deltaCol >= deltaRow:
                    npoints = deltaCol + 1
                else:    
                    npoints = deltaRow + 1
            elif self.__lineProjectionMode == 'X':
                npoints = deltaCol + 1
            else:
                npoints = deltaRow + 1
            if npoints == 1:
                #all points are the same
                if DEBUG:
                    print("START AND END POINT ARE THE SAME!!")
                return

            if width < 1:
                x = numpy.zeros((npoints, 2), numpy.float)
                x[:, 0] = numpy.linspace(row0, row1, npoints)
                x[:, 1] = numpy.linspace(col0, col1, npoints)
                legend = "From (%.3f, %.3f) to (%.3f, %.3f)" % (col0, row0, col1, row1)
                #perform the interpolation
                ydata = self._interpolate((x0, y0), imageData, x)
                xdata = numpy.arange(float(npoints))
                
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([col0, col1],
                                         [row0, row1],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            elif deltaCol == 0:
                #vertical line
                col0 = int(ddict['column'][0]) - 0.5 * width
                if col0 < 0:
                    col0 = 0
                    col1 = col0 + width
                else:
                    col1 = int(ddict['column'][0]) + 0.5 * width
                if col1 >= shape[1]:
                    col1 = shape[1] - 1
                    col0 = max(0, col1 - width)
                row0 = int(ddict['row'][0])
                row1 = int(ddict['row'][1])
                if row0 > row1:
                    tmp = row0
                    row0 = row1
                    row1 = tmp
                if row0 < 0:
                    row0 = 0
                if row1 >= shape[0]:
                    row1 = shape[0] - 1                    
                ydata = imageData[row0:row1+1, col0:int(col1+1)].sum(axis=1)
                legend = "Col = %d to %d"  % (col0, col1)
                npoints = max(ydata.shape)
                xdata = numpy.arange(float(npoints))
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([col0, col0, col1, col1],
                                         [row0, row1, row1, row0],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            elif deltaRow == 0:
                #horizontal line
                row0 = int(ddict['row'][0]) - 0.5 * width
                if row0 < 0:
                    row0 = 0
                    row1 = row0 + width
                else:
                    row1 = int(ddict['row'][0]) + 0.5 * width
                if row1 >= shape[0]:
                    row1 = shape[0] - 1
                    row0 = max(0, row1 - width)
                col0 = int(ddict['column'][0])
                col1 = int(ddict['column'][1])
                if col0 > col1:
                    tmp = col0
                    col0 = col1
                    col1 = tmp
                if col0 < 0:
                    col0 = 0
                if col1 >= shape[1]:
                    col1 = shape[1] - 1                    
                ydata = imageData[row0:int(row1+1), col0:col1+1].sum(axis=0)
                legend = "Row = %d to %d"  % (row0, row1)
                npoints = max(ydata.shape)
                xdata = numpy.arange(float(npoints))
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem([col0, col0, col1, col1],
                                         [row0, row1, row1, row0],
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            else:
                #restore original value of width
                width = ddict['pixelwidth']
                #find m and b in the line y = mx + b
                m = (row1 - row0) / float((col1 - col0))
                b = row0 - m * col0
                alpha = numpy.arctan(m)
                #imagine the following sequence
                # - change origin to the first point
                # - clock-wise rotation to bring the line on the x axis of a new system
                # so that the points (col0, row0) and (col1, row1) become (x0, 0) (x1, 0)
                # - counter clock-wise rotation to get the points (x0, -0.5 width),
                # (x0, 0.5 width), (x1, 0.5 * width) and (x1, -0.5 * width) back to the
                # original system.
                # - restore the origin to (0, 0)
                # - if those extremes are inside the image the selection is acceptable
                cosalpha = numpy.cos(alpha)
                sinalpha = numpy.sin(alpha)
                newCol0 = 0.0
                newCol1 = (col1-col0) * cosalpha + (row1-row0) * sinalpha
                newRow0 = 0.0
                newRow1 = -(col1-col0) * sinalpha + (row1-row0) * cosalpha

                if DEBUG:
                    print("new X0 Y0 = %f, %f  " % (newCol0, newRow0))
                    print("new X1 Y1 = %f, %f  " % (newCol1, newRow1))

                tmpX   = numpy.linspace(newCol0, newCol1, npoints).astype(numpy.float)
                rotMatrix = numpy.zeros((2,2), numpy.float)
                rotMatrix[0,0] =   cosalpha
                rotMatrix[0,1] = - sinalpha
                rotMatrix[1,0] =   sinalpha
                rotMatrix[1,1] =   cosalpha
                if DEBUG:
                    #test if I recover the original points
                    testX = numpy.zeros((2, 1) , numpy.float)
                    colRow = numpy.dot(rotMatrix, testX)
                    print("Recovered X0 = %f" % (colRow[0,0] + col0))
                    print("Recovered Y0 = %f" % (colRow[1,0] + row0))
                    print("It should be = %f, %f" % (col0, row0))
                    testX[0,0] = newCol1
                    testX[1,0] = newRow1
                    colRow = numpy.dot(rotMatrix, testX)
                    print("Recovered X1 = %f" % (colRow[0,0] + col0))
                    print("Recovered Y1 = %f" % (colRow[1,0] + row0))
                    print("It should be = %f, %f" % (col1, row1))

                #find the drawing limits
                testX = numpy.zeros((2, 4) , numpy.float)
                testX[0,0] = newCol0
                testX[0,1] = newCol0
                testX[0,2] = newCol1
                testX[0,3] = newCol1
                testX[1,0] = newRow0 - 0.5 * width
                testX[1,1] = newRow0 + 0.5 * width
                testX[1,2] = newRow1 + 0.5 * width
                testX[1,3] = newRow1 - 0.5 * width
                colRow = numpy.dot(rotMatrix, testX)
                colLimits0 = colRow[0, :] + col0
                rowLimits0 = colRow[1, :] + row0

                for a in rowLimits0:
                    if (a >= shape[0]) or (a < 0):
                        print("outside row limits",a)
                        return
                for a in colLimits0:
                    if (a >= shape[1]) or (a < 0):
                        print("outside column limits",a)
                        return

                r0 = rowLimits0[0]
                r1 = rowLimits0[1]

                if r0 > r1:
                    print("r0 > r1", r0, r1)
                    raise ValueError("r0 > r1")                    

                x = numpy.zeros((2, npoints) , numpy.float)
                tmpMatrix = numpy.zeros((npoints, 2) , numpy.float)

                if 0:
                    #take only the central point
                    oversampling = 1
                    x[0, :] = tmpX
                    x[1, :] = 0.0
                    colRow = numpy.dot(rotMatrix, x)
                    colRow[0, :] += col0
                    colRow[1, :] += row0
                    tmpMatrix[:,0] = colRow[1,:]
                    tmpMatrix[:,1] = colRow[0,:]
                    ydataCentral = self._interpolate((x0, y0),\
                                    imageData, tmpMatrix)
                    #multiply by width too have the equivalent scale
                    ydata = ydataCentral
                else:
                    if ddict['event'] == "PolygonSelected":
                        #oversampling solves noise introduction issues
                        oversampling = width + 1
                        oversampling = min(oversampling, 21) 
                    else:
                        oversampling = 1
                    ncontributors = width * oversampling
                    iterValues = numpy.linspace(-0.5*width, 0.5*width, ncontributors)
                    tmpMatrix = numpy.zeros((npoints*len(iterValues), 2) , numpy.float)
                    x[0, :] = tmpX
                    offset = 0
                    for i in iterValues:
                        x[1, :] = i
                        colRow = numpy.dot(rotMatrix, x)
                        colRow[0, :] += col0
                        colRow[1, :] += row0
                        """
                        colLimits = [colRow[0, 0], colRow[0, -1]]
                        rowLimits = [colRow[1, 0], colRow[1, -1]]
                        for a in rowLimits:
                            if (a >= shape[0]) or (a < 0):
                                print("outside row limits",a)
                                return
                        for a in colLimits:
                            if (a >= shape[1]) or (a < 0):
                                print("outside column limits",a)
                                return
                        """
                        #it is much faster to make one call to the interpolating
                        #routine than making many calls
                        tmpMatrix[offset:(offset+npoints),0] = colRow[1,:]
                        tmpMatrix[offset:(offset+npoints),1] = colRow[0,:]                    
                        offset += npoints
                    ydata = self._interpolate((x0, y0),\
                                   imageData, tmpMatrix)
                    ydata.shape = len(iterValues), npoints
                    ydata = ydata.sum(axis=0)
                    #deal with the oversampling
                    ydata /= oversampling
                          
                xdata = numpy.arange(float(npoints))
                legend = "y = %f (x-%.1f) + %f ; width=%d" % (m, col0, b, width)
                if overlay:
                    #self.drawOverlayItem(x, y, legend=name, info=info, replot, replace)
                    self.drawOverlayItem(colLimits0,
                                         rowLimits0,
                                         legend=ddict['mode'],
                                         info=ddict,
                                         replace=True,
                                         replot=True)
            if self.__lineProjectionMode == 'X':
                xLabel = self.getXLabel()
                xdata += col0
                if self._xScale is not None:
                    xdata = self._xScale[0] + xdata * (self._xScale[1] - self._xScale[0]) / float(shape[1])
            elif self.__lineProjectionMode == 'Y':
                xLabel = self.getYLabel()
                xdata += row0
                if self._xScale is not None:
                    xdata = self._yScale[0] + xdata * (self._yScale[1] - self._yScale[0]) / float(shape[0])
            else:
                xLabel = "Distance"
                if self._xScale is not None:
                    deltaCol *= (self._xScale[1] - self._xScale[0])/float(shape[1])
                    deltaRow *= (self._yScale[1] - self._yScale[0])/float(shape[0])
                #get the abscisa in distance units
                deltaDistance = numpy.sqrt(float(deltaCol) * deltaCol +
                                    float(deltaRow) * deltaRow)/(npoints-1.0)
                xdata *= deltaDistance
        else:
            if DEBUG:
                print("Mode %s not supported yet" % ddict['mode'])
            return

        info = {}
        info['xlabel'] = xLabel
        info['ylabel'] = "Z"
        return xdata, ydata, legend, info

    def _profileSelectionSlot(self, ddict):
        if DEBUG:
            print(ddict)
        # the curves as [[x0, y0, legend0, info0], ...] 
        curveList = ddict['curves']
        label = ddict['label']
        n = len(curveList)
        if ddict['event'] == 'ADD':
            for i in range(n):
                x, y, legend, info = curveList[i]
                info['profilelabel'] = label
                if i == (n-1):
                    replot = True
                self._profileScanWindow.addCurve(x, y, legend=legend, info=info,
                                                 replot=replot, replace=False)
        elif ddict['event'] == 'REPLACE':
            for i in range(n):
                x, y, legend, info = curveList[i]
                info['profilelabel'] = label
                if i in [0, n-1]:
                    replace = True
                else:
                    replace = False
                if i == (n-1):
                    replot = True
                else:
                    replot = False
                self._profileScanWindow.addCurve(x, y, legend=legend, info=info,
                                                 replot=replot, replace=replace)
        elif ddict['event'] == 'REMOVE':
            curveList = self._profileScanWindow.getAllCurves()
            if curveList in [None, []]:
                return
            toDelete = []
            n = len(curveList)                
            for i in range(n):
                x, y, legend, info = curveList[i]
                curveLabel = info.get('profilelabel', None)
                if curveLabel is not None:
                    if label == curveLabel:
                        toDelete.append(legend)
            n = len(toDelete)
            for i in range(n):
                legend = toDelete[i]
                if i == (n-1):
                    replot = True
                else:
                    replot = False
                self._profileScanWindow.removeCurve(legend, replot=replot)        

    def drawOverlayItem(self, x, y, legend=None, info=None, replace=False, replot=True):
        #same call as the plot1D addCurve command
        if legend is None:
            legend="UnnamedOverlayItem"
        if legend not in self._overlayItemsDict:
            overlayItem = QwtPlotItems.PolygonItem(legend)
            overlayItem.attach(self.graphWidget.graph)
            self._overlayItemsDict[legend] = {}
            self._overlayItemsDict[legend]['item'] = overlayItem
        else:
            overlayItem = self._overlayItemsDict[legend]['item']
        if replace:
            iterKeys = list(self._overlayItemsDict.keys())
            for name in iterKeys:
                if name == legend:
                    continue
                self._overlayItemsDict[name]['item'].detach()
                delKeys = list(self._overlayItemsDict[name].keys())
                for key in delKeys:
                    del self._overlayItemsDict[name][key]
                del self._overlayItemsDict[name]
        #the type of x can be list or array
        shape = self.__imageData.shape
        if self._xScale is None:
            xList = x
        else:
            xList = []
            for i in x:
                xList.append(self._xScale[0] + i * (self._xScale[1] - self._xScale[0])/float(shape[1]))

        if self._yScale is None:
            yList = y
        else:
            yList = []
            for i in y:
                yList.append(self._yScale[0] + i * (self._yScale[1] - self._yScale[0])/float(shape[0]))
        overlayItem.setData(xList, yList)
        self._overlayItemsDict[legend]['x'] = xList
        self._overlayItemsDict[legend]['y'] = yList
        self._overlayItemsDict[legend]['info'] = info
        if replot:
            self.graphWidget.graph.replot()
        self.__lastOverlayLegend = legend

    def _hFlipIconSignal(self):
        if not self.graphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set Y Axis to AutoScale first")
            return
        if not self.graphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set X Axis to AutoScale first")
            return

        if self._y1AxisInverted:
            self._y1AxisInverted = False
        else:
            self._y1AxisInverted = True
        self.graphWidget.graph.zoomReset()
        self.graphWidget.graph.setY1AxisInverted(self._y1AxisInverted)
        self.plotImage(True)

        #inform the other widgets
        ddict = {}
        ddict['event'] = "hFlipSignal"
        ddict['current'] = self._y1AxisInverted * 1
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)        

    def setY1AxisInverted(self, value):
        self._y1AxisInverted = value
        self.graphWidget.graph.setY1AxisInverted(self._y1AxisInverted)

    def setXLabel(self, label="Column"):
        return self.graphWidget.setXLabel(label)

    def setYLabel(self, label="Row"):
        return self.graphWidget.setYLabel(label)

    def getXLabel(self):
        return self.graphWidget.getXLabel()

    def getYLabel(self):
        return self.graphWidget.getYLabel()

    def buildAndConnectImageButtonBox(self, replace=True):
        # The IMAGE selection
        self.imageButtonBox = qt.QWidget(self)
        buttonBox = self.imageButtonBox
        self.imageButtonBoxLayout = qt.QHBoxLayout(buttonBox)
        self.imageButtonBoxLayout.setMargin(0)
        self.imageButtonBoxLayout.setSpacing(0)
        self.addImageButton = qt.QPushButton(buttonBox)
        icon = qt.QIcon(qt.QPixmap(IconDict["rgb16"]))
        self.addImageButton.setIcon(icon)
        self.addImageButton.setText("ADD IMAGE")
        self.removeImageButton = qt.QPushButton(buttonBox)
        self.removeImageButton.setIcon(icon)
        self.removeImageButton.setText("REMOVE IMAGE")
        self.imageButtonBoxLayout.addWidget(self.addImageButton)
        self.imageButtonBoxLayout.addWidget(self.removeImageButton)

        
        self.mainLayout.addWidget(buttonBox)
        
        self.connect(self.addImageButton, qt.SIGNAL("clicked()"), 
                    self._addImageClicked)
        self.connect(self.removeImageButton, qt.SIGNAL("clicked()"), 
                    self._removeImageClicked)
        if replace:
            self.replaceImageButton = qt.QPushButton(buttonBox)
            self.replaceImageButton.setIcon(icon)
            self.replaceImageButton.setText("REPLACE IMAGE")
            self.imageButtonBoxLayout.addWidget(self.replaceImageButton)
            self.connect(self.replaceImageButton,
                         qt.SIGNAL("clicked()"), 
                         self._replaceImageClicked)
    
    def _setEraseSelectionMode(self):
        if DEBUG:
            print("_setEraseSelectionMode")
        self.__eraseMode = True
        self.__brushMode = True
        self.graphWidget.picker.setTrackerMode(Qwt5.QwtPicker.ActiveOnly)
        self.graphWidget.graph.enableSelection(False)

    def _setRectSelectionMode(self):
        if DEBUG:
            print("_setRectSelectionMode")
        self.__eraseMode = False
        self.__brushMode = False
        self.graphWidget.picker.setTrackerMode(Qwt5.QwtPicker.AlwaysOn)
        self.graphWidget.graph.enableSelection(True)
        
    def _setBrushSelectionMode(self):
        if DEBUG:
            print("_setBrushSelectionMode")
        self.__eraseMode = False
        self.__brushMode = True
        self.graphWidget.picker.setTrackerMode(Qwt5.QwtPicker.ActiveOnly)
        self.graphWidget.graph.enableSelection(False)
        
    def _setBrush(self):
        if DEBUG:
            print("_setBrush")
        if self.__brushMenu is None:
            if QTVERSION < '4.0.0':
                self.__brushMenu = qt.QPopupMenu()
                self.__brushMenu.insertItem(QString(" 1 Image Pixel Width"),
                                            self.__setBrush1)
                self.__brushMenu.insertItem(QString(" 2 Image Pixel Width"),
                                            self.__setBrush2)
                self.__brushMenu.insertItem(QString(" 3 Image Pixel Width"),
                                            self.__setBrush3)
                self.__brushMenu.insertItem(QString(" 5 Image Pixel Width"),
                                            self.__setBrush4)
                self.__brushMenu.insertItem(QString("10 Image Pixel Width"),
                                            self.__setBrush5)
                self.__brushMenu.insertItem(QString("20 Image Pixel Width"),
                                            self.__setBrush6)
            else:
                self.__brushMenu = qt.QMenu()
                self.__brushMenu.addAction(QString(" 1 Image Pixel Width"),
                                           self.__setBrush1)
                self.__brushMenu.addAction(QString(" 2 Image Pixel Width"),
                                           self.__setBrush2)
                self.__brushMenu.addAction(QString(" 3 Image Pixel Width"),
                                           self.__setBrush3)
                self.__brushMenu.addAction(QString(" 5 Image Pixel Width"),
                                           self.__setBrush4)
                self.__brushMenu.addAction(QString("10 Image Pixel Width"),
                                           self.__setBrush5)
                self.__brushMenu.addAction(QString("20 Image Pixel Width"),
                                           self.__setBrush6)
        if QTVERSION < '4.0.0':
            self.__brushMenu.exec_loop(self.cursor().pos())
        else:
            self.__brushMenu.exec_(self.cursor().pos())

    def __setBrush1(self):
        self.__brushWidth = 1

    def __setBrush2(self):
        self.__brushWidth = 2

    def __setBrush3(self):
        self.__brushWidth = 3

    def __setBrush4(self):
        self.__brushWidth = 5

    def __setBrush5(self):
        self.__brushWidth = 10

    def __setBrush6(self):
        self.__brushWidth = 20

    def _toggleSelectionMode(self):
        if self.graphWidget.graph._selecting:
            self.setSelectionMode(False)
        else:
            self.setSelectionMode(True)

    def setSelectionMode(self, mode = None):
        #does it have sense to enable the selection without the image selection icons?
        #if not self.__imageIconsFlag:
        #    mode = False
        if mode:
            self.graphWidget.graph.enableSelection(True)
            self.__brushMode  = False
            self.graphWidget.picker.setTrackerMode(Qwt5.QwtPicker.AlwaysOn)
            if QTVERSION < '4.0.0':
                self.graphWidget.selectionToolButton.setState(qt.QButton.On)
            else:
                self.graphWidget.hideProfileSelectionIcons()
                self.graphWidget.selectionToolButton.setChecked(True)
            self.graphWidget.graph.enableZoom(False)
            self.graphWidget.selectionToolButton.setDown(True)
            self.graphWidget.showImageIcons()            
        else:
            self.graphWidget.picker.setTrackerMode(Qwt5.QwtPicker.AlwaysOff)
            self.graphWidget.showProfileSelectionIcons()
            self.graphWidget.graph.enableZoom(True)
            if QTVERSION < '4.0.0':
                self.graphWidget.selectionToolButton.setState(qt.QButton.Off)
            else:
                self.graphWidget.selectionToolButton.setChecked(False)
            self.graphWidget.selectionToolButton.setDown(False)
            self.graphWidget.hideImageIcons()
        if self.__imageData is None: return
        #do not reset the selection
        #self.__selectionMask = numpy.zeros(self.__imageData.shape, numpy.UInt8)

    def _additionalSelectionMenuDialog(self):
        if self.__imageData is None:
            return
        self._additionalSelectionMenu.exec_(self.cursor().pos())

    def _getSelectionMinMax(self):
        if self.colormap is None:
            goodData = self.__imageData[numpy.isfinite(self.__imageData)]
            maxValue = goodData.max()
            minValue = goodData.min()
        else:
            minValue = self.colormap[2]
            maxValue = self.colormap[3]

        return minValue, maxValue

    def _selectMax(self):
        selectionMask = numpy.zeros(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self.__imageData, copy=True)
        tmpData[True - numpy.isfinite(self.__imageData)] = minValue
        selectionMask[tmpData >= maxValue] = 1
        self.setSelectionMask(selectionMask, plot=False)
        self.plotImage(update=False)
        self._emitMaskChangedSignal()
        
    def _selectMiddle(self):
        selectionMask = numpy.ones(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self.__imageData, copy=True)
        tmpData[True - numpy.isfinite(self.__imageData)] = maxValue
        selectionMask[tmpData >= maxValue] = 0
        selectionMask[tmpData <= minValue] = 0
        self.setSelectionMask(selectionMask, plot=False)
        self.plotImage(update=False)
        self._emitMaskChangedSignal()        

    def _selectMin(self):
        selectionMask = numpy.zeros(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        tmpData = numpy.array(self.__imageData, copy=True)
        tmpData[True - numpy.isfinite(self.__imageData)] = maxValue
        selectionMask[tmpData <= minValue] = 1
        self.setSelectionMask(selectionMask, plot=False)
        self.plotImage(update=False)
        self._emitMaskChangedSignal()
        
    def _invertSelection(self):
        if self.__imageData is None:
            return
        mask = numpy.ones(self.__imageData.shape,
                                             numpy.uint8)
        if self.__selectionMask is not None:
            mask[self.__selectionMask > 0] = 0

        self.setSelectionMask(mask, plot=True)
        self._emitMaskChangedSignal()
        
    def _resetSelection(self, owncall=True):
        if DEBUG:
            print("_resetSelection")
        self.__selectionMask = None
        if self.__imageData is None:
            return
        #self.__selectionMask = numpy.zeros(self.__imageData.shape, numpy.uint8)
        self.plotImage(update = True)

        #inform the others
        if owncall:
            ddict = {}
            ddict['event'] = "resetSelection"
            ddict['id'] = id(self)
            self.emitMaskImageSignal(ddict)
            
    def setSelectionMask(self, mask, plot=True):
        if mask is not None:
            if self.__imageData is not None:
                mask *=  numpy.isfinite(self.__imageData)
        self.__selectionMask = mask
        if plot:
            self.plotImage(update=False)

    def getSelectionMask(self):
        if self.__imageData is None:
            return None
        if self.__selectionMask is None:
            return numpy.zeros(self.__imageData.shape, numpy.uint8)
        return self.__selectionMask

    def setImageData(self, data, clearmask=False, xScale=None, yScale=None):
        self.__image = None
        self._xScale = xScale
        self._yScale = yScale
        if data is None:
            self.__imageData = data
            self.__selectionMask = None
            self.plotImage(update = True)
            return
        else:
            self.__imageData = data
        if clearmask:
            self.__selectionMask = None
        if self.colormapDialog is not None:
            goodData = self.__imageData[numpy.isfinite(self.__imageData)]
            minData = goodData.min()
            maxData = goodData.max()
            if self.colormapDialog.autoscale:
                self.colormapDialog.setDisplayedMinValue(minData) 
                self.colormapDialog.setDisplayedMaxValue(maxData) 
            self.colormapDialog.setDataMinMax(minData, maxData, update=True)
        else:
            self.plotImage(update = True)

    def getImageData(self):
        return self.__imageData

    def getQImage(self):
        return self.__image

    def setQImage(self, qimage, width, height, clearmask=False, data=None):
        #This is just to get it different than None
        if (qimage.width() != width) or (qimage.height() != height):
            if 1 or (qimage.width() > width) or (qimage.height() > height):
                transformation = qt.Qt.SmoothTransformation
            else:
                transformation = qt.Qt.FastTransformation
            self.__image = qimage.scaled(qt.QSize(width, height),
                                     qt.Qt.IgnoreAspectRatio,
                                     transformation)
        else:
            self.__image = qimage

        if self.__image.format() == qt.QImage.Format_Indexed8:
            pixmap0 = numpy.fromstring(qimage.bits().asstring(width * height),
                                 dtype = numpy.uint8)
            pixmap = numpy.zeros((height * width, 4), numpy.uint8)
            pixmap[:,0] = pixmap0[:]
            pixmap[:,1] = pixmap0[:]
            pixmap[:,2] = pixmap0[:]
            pixmap[:,3] = 255
            pixmap.shape = height, width, 4
        else:
            self.__image = self.__image.convertToFormat(qt.QImage.Format_ARGB32) 
            pixmap = numpy.fromstring(self.__image.bits().asstring(width * height * 4),
                                 dtype = numpy.uint8)
        pixmap.shape = height, width,-1
        if data is None:
            self.__imageData = numpy.zeros((height, width), numpy.float)
            self.__imageData = pixmap[:,:,0] * 0.114 +\
                               pixmap[:,:,1] * 0.587 +\
                               pixmap[:,:,2] * 0.299
        else:
            self.__imageData = data
            self.__imageData.shape = height, width
        self._xScale = None
        self._yScale = None
        self.__pixmap0 = pixmap
        if clearmask:
            self.__selectionMask = None
        self.plotImage(update = True)
        
    def plotImage(self, update=True):
        if self.__imageData is None:
            self.graphWidget.graph.clear()
            self.graphWidget.picker.data = None
            self.graphWidget.picker.xScale = None
            self.graphWidget.picker.yScale = None
            return

        if update:
            self.getPixmapFromData()
            self.__pixmap0 = self.__pixmap.copy()
            self.graphWidget.picker.data = self.__imageData
            self.graphWidget.picker.xScale = self._xScale
            self.graphWidget.picker.yScale = self._yScale
            if self.colormap is None:
                if self.__defaultColormap < 2:
                    self.graphWidget.picker.setTrackerPen(qt.QPen(qt.Qt.green))
                else:
                    self.graphWidget.picker.setTrackerPen(qt.QPen(qt.Qt.black))
            elif int(str(self.colormap[0])) > 1:     #color
                self.graphWidget.picker.setTrackerPen(qt.QPen(qt.Qt.black))
            else:
                self.graphWidget.picker.setTrackerPen(qt.QPen(qt.Qt.green))
        self.__applyMaskToImage()
        if not self.graphWidget.graph.yAutoScale:
            ylimits = self.graphWidget.graph.getY1AxisLimits()
        if not self.graphWidget.graph.xAutoScale:
            xlimits = self.graphWidget.graph.getX1AxisLimits()
        self.graphWidget.graph.pixmapPlot(self.__pixmap.tostring(),
            (self.__imageData.shape[1], self.__imageData.shape[0]),
                                        xmirror = 0,
                                        ymirror = not self._y1AxisInverted,
                                        xScale = self._xScale,
                                        yScale = self._yScale)

        if not self.graphWidget.graph.yAutoScale:
            self.graphWidget.graph.setY1AxisLimits(ylimits[0], ylimits[1],
                                                   replot=False)
        if not self.graphWidget.graph.xAutoScale:
            self.graphWidget.graph.setX1AxisLimits(xlimits[0], xlimits[1],
                                                   replot=False)
        self.graphWidget.graph.replot()

    def getPixmapFromData(self):
        colormap = self.colormap
        if self.__image is not None:
            self.__pixmap = self.__pixmap0.copy()
            return

        if hasattr(self.__imageData, 'mask'):
            data = self.__imageData.data
        else:
            data = self.__imageData

        finiteData = numpy.isfinite(data)
        goodData = finiteData.min()
        
        if self.colormapDialog is not None:
            minData = self.colormapDialog.dataMin
            maxData = self.colormapDialog.dataMax
        else:
            if goodData:
                minData = data.min()
                maxData = data.max()
            else:
                tmpData = data[finiteData]
                minData = tmpData.min()
                maxData = tmpData.max()
                tmpData = None
        if colormap is None:
            (self.__pixmap,size,minmax)= spslut.transform(\
                                data,
                                (1,0),
                                (self.__defaultColormapType,3.0),
                                "BGRX",
                                self.__defaultColormap,
                                0,
                                (minData,maxData),
                                (0, 255), 1)
        else:
            if len(colormap) < 7:
                colormap.append(spslut.LINEAR)
            if goodData:
                (self.__pixmap,size,minmax)= spslut.transform(\
                                data,
                                (1,0),
                                (colormap[6],3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]),
                                (0,255), 1)                
            elif colormap[1]:
                #autoscale
                (self.__pixmap,size,minmax)= spslut.transform(\
                                data,
                                (1,0),
                                (colormap[6],3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                0,
                                (minData,maxData),
                                (0,255), 1)
            else:
                (self.__pixmap,size,minmax)= spslut.transform(\
                                data,
                                (1,0),
                                (colormap[6],3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]),
                                (0,255), 1)

        self.__pixmap = self.__pixmap.astype(numpy.ubyte)

        self.__pixmap.shape = [data.shape[0], data.shape[1], 4]

        if not goodData:
            self.__pixmap[finiteData < 1] = 255

    def __applyMaskToImage(self):
        if self.__selectionMask is None:
            return
        #if not self.__selectionFlag:
        #    print("Return because of selection flag")
        #    return

        if self.colormap is None:
            if self.__image is not None:
                if self.__image.format() == qt.QImage.Format_ARGB32:
                    for i in range(4):
                        self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] *\
                                (1 - (0.2 * self.__selectionMask))).astype(numpy.uint8)
                else:
                    self.__pixmap = self.__pixmap0.copy()
                    self.__pixmap[self.__selectionMask>0,0]    = 0x40
                    self.__pixmap[self.__selectionMask>0,2]    = 0x70
                    self.__pixmap[self.__selectionMask>0,3]    = 0x40
            else:
                if self.__defaultColormap > 1:
                    tmp = 1 - 0.2 * self.__selectionMask
                    for i in range(3):
                        self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] *\
                                tmp)
                    if 0:
                        #this is to recolor non finite points
                        tmpMask = numpy.isfinite(self.__imageData)
                        goodData = numpy.isfinite(self.__imageData).min()
                        if not goodData:
                            for i in range(3):
                                self.__pixmap[:,:,i] *= tmpMask
                else:
                    self.__pixmap = self.__pixmap0.copy()
                    self.__pixmap[self.__selectionMask>0,0]    = 0x40
                    self.__pixmap[self.__selectionMask>0,2]    = 0x70
                    self.__pixmap[self.__selectionMask>0,3]    = 0x40
                    if 0:
                        #this is to recolor non finite points
                        tmpMask = ~numpy.isfinite(self.__imageData)
                        badData = numpy.isfinite(self.__imageData).max()
                        if badData:
                            self.__pixmap[tmpMask,0]    = 0x00
                            self.__pixmap[tmpMask,1]    = 0xff
                            self.__pixmap[tmpMask,2]    = 0xff
                            self.__pixmap[tmpMask,3]    = 0xff
        elif int(str(self.colormap[0])) > 1:     #color
            tmp = 1 - 0.2 * self.__selectionMask
            for i in range(3):
                self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] *\
                        tmp)
            if 0:
                tmpMask = numpy.isfinite(self.__imageData)
                goodData = numpy.isfinite(self.__imageData).min()
                if not goodData:
                    if not goodData:
                        for i in range(3):
                            self.__pixmap[:,:,i] *= tmpMask                
        else:
            self.__pixmap = self.__pixmap0.copy()
            tmp  = 1 - self.__selectionMask
            self.__pixmap[:,:, 2] = (0x70 * self.__selectionMask) +\
                                  tmp * self.__pixmap0[:,:,2]
            self.__pixmap[:,:, 3] = (0x40 * self.__selectionMask) +\
                                  tmp * self.__pixmap0[:,:,3]
            if 0:
                tmpMask = ~numpy.isfinite(self.__imageData)
                badData = numpy.isfinite(self.__imageData).max()
                if badData:
                    self.__pixmap[tmpMask,0]    = 0x00
                    self.__pixmap[tmpMask,1]    = 0xff
                    self.__pixmap[tmpMask,2]    = 0xff
                    self.__pixmap[tmpMask,3]    = 0xff
        return

    def selectColormap(self):
        if self.__imageData is None:
            return
        if self.colormapDialog is None:
            self.__initColormapDialog()
        if self.colormapDialog.isHidden():
            self.colormapDialog.show()
        if QTVERSION < '4.0.0':
            self.colormapDialog.raiseW()
        else:
            self.colormapDialog.raise_()
        self.colormapDialog.show()

    def __initColormapDialog(self):
        goodData = self.__imageData[numpy.isfinite(self.__imageData)]
        maxData = goodData.max()
        minData = goodData.min()
        self.colormapDialog = ColormapDialog.ColormapDialog()
        colormapIndex = self.__defaultColormap
        if colormapIndex == 1:
            colormapIndex = 0
        elif colormapIndex == 6:
            colormapIndex = 1
        self.colormapDialog.colormapIndex  = colormapIndex
        self.colormapDialog.colormapString = self.colormapDialog.colormapList[colormapIndex]
        self.colormapDialog.setDataMinMax(minData, maxData)
        self.colormapDialog.setAutoscale(1)
        self.colormapDialog.setColormap(self.colormapDialog.colormapIndex)
        self.colormapDialog.setColormapType(self.__defaultColormapType, update=False)
        self.colormap = (self.colormapDialog.colormapIndex,
                              self.colormapDialog.autoscale,
                              self.colormapDialog.minValue, 
                              self.colormapDialog.maxValue,
                              minData, maxData)
        if QTVERSION < '4.0.0':
            self.colormapDialog.setCaption("Colormap Dialog")
            self.connect(self.colormapDialog,
                         qt.PYSIGNAL("ColormapChanged"),
                         self.updateColormap)
        else:
            self.colormapDialog.setWindowTitle("Colormap Dialog")
            self.connect(self.colormapDialog,
                         qt.SIGNAL("ColormapChanged"),
                         self.updateColormap)
        self.colormapDialog._update()

    def updateColormap(self, *var):
        if len(var) > 6:
            self.colormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5],
                             var[6]]
        elif len(var) > 5:
            self.colormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        else:
            self.colormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        self.plotImage(True)

    def _addImageClicked(self):
        ddict = {}
        ddict['event'] = "addImageClicked"
        ddict['image'] = self.__imageData
        ddict['title'] = self.getGraphTitle()
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)
            
    def _removeImageClicked(self):
        ddict = {}
        ddict['event'] = "removeImageClicked"
        ddict['title'] = self.getGraphTitle()
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)

    def _replaceImageClicked(self):
        ddict = {}
        ddict['event'] = "replaceImageClicked"
        ddict['image'] = self.__imageData
        ddict['title'] = self.getGraphTitle()
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)

    def _saveToolButtonSignal(self):
        self._saveMenu.exec_(self.cursor().pos())

    def _saveMatplotlibImage(self):
        imageData = self.__imageData
        if self._matplotlibSaveImage is None:
            self._matplotlibSaveImage = QPyMcaMatplotlibSave.SaveImageSetup(None,
                                                            image=None)
        title = "Matplotlib " + self.getGraphTitle()
        self._matplotlibSaveImage.setWindowTitle(title)        
        ddict = self._matplotlibSaveImage.getParameters()
        if self.colormap is not None:
            colormapType = ddict['linlogcolormap']
            try:
                colormapIndex, autoscale, vmin, vmax,\
                        dataMin, dataMax, colormapType = self.colormap
                if colormapType == spslut.LOG:
                    colormapType = 'logarithmic'
                else:
                    colormapType = 'linear'
            except:
                colormapIndex, autoscale, vmin, vmax = self.colormap[0:4]
            ddict['linlogcolormap'] = colormapType
            if not autoscale:
                ddict['valuemin'] = vmin
                ddict['valuemax'] = vmax
            else:
                ddict['valuemin'] = 0
                ddict['valuemax'] = 0
            
        #this sets the actual dimensions
        if self._xScale is not None:
            ddict['xorigin'] = self._xScale[0]
            ddict['xpixelsize'] = (self._xScale[1] - self._xScale[0])/\
                                      float(imageData.shape[1])
        if self._yScale is not None:
            ddict['yorigin'] = self._yScale[0]
            ddict['ypixelsize'] = (self._yScale[1] - self._yScale[0])/\
                                      float(imageData.shape[0])
        ddict['xlabel'] = self.getXLabel()
        ddict['ylabel'] = self.getYLabel()
        limits = self.graphWidget.graph.getX1AxisLimits()
        ddict['zoomxmin'] = limits[0]
        ddict['zoomxmax'] = limits[1]
        limits = self.graphWidget.graph.getY1AxisLimits()
        ddict['zoomymin'] = limits[0] 
        ddict['zoomymax'] = limits[1]
        
        self._matplotlibSaveImage.setParameters(ddict)
        self._matplotlibSaveImage.setImageData(imageData)
        self._matplotlibSaveImage.show()
        self._matplotlibSaveImage.raise_()

    def _otherWidgetGraphSignal(self, ddict):
        self._graphSignal(ddict, ownsignal = False)

    def _graphSignal(self, ddict, ownsignal = None):
        if ownsignal is None:
            ownsignal = True
        emitsignal = False
        if self.__imageData is None:
            return
        if ddict['event'] == "MouseSelection":
            if ddict['column_min'] < ddict['column_max']:
                xmin = ddict['column_min']
                xmax = ddict['column_max']
            else:
                xmin = ddict['column_max']
                xmax = ddict['column_min']
            if ddict['row_min'] < ddict['row_max']:
                ymin = ddict['row_min']
                ymax = ddict['row_max']
            else:
                ymin = ddict['row_max']
                ymax = ddict['row_min']
            """
            if not (self._xScale is None and self._yScale is None):
                ymin, xmin = convertToRowAndColumn(xmin, ymin, self.__imageData.shape,
                                                  xScale=self._xScale,
                                                  yScale=self._yScale,
                                                  safe=True)
                ymax, xmax = convertToRowAndColumn(xmax, ymax, self.__imageData.shape,
                                                  xScale=self._xScale,
                                                  yScale=self._yScale,
                                                  safe=True)
            """
            i1 = max(int(round(xmin)), 0)
            i2 = min(abs(int(round(xmax))) + 1, self.__imageData.shape[1])
            j1 = max(int(round(ymin)),0)
            j2 = min(abs(int(round(ymax))) + 1, self.__imageData.shape[0])
            if self.__selectionMask is None:
                self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                     numpy.uint8)
            self.__selectionMask[j1:j2, i1:i2] = 1
            emitsignal = True

        elif ddict['event'] == "MouseAt":
            if ownsignal:
                pass
            if self.__brushMode:
                if self.graphWidget.graph.isZoomEnabled():
                    return
                #if follow mouse is not activated
                #it only enters here when the mouse is pressed.
                #Therefore is perfect for "brush" selections.
                """
                if not (self._xScale is None and self._yScale is None):
                    y, x = convertToRowAndColumn(ddict['x'], ddict['y'], self.__imageData.shape,
                                                      xScale=self._xScale,
                                                      yScale=self._yScale,
                                                      safe=True)
                else:
                    x = ddict['x']
                    y = ddict['y']
                """
                y = ddict['row']
                x = ddict['column']
                width = self.__brushWidth   #in (row, column) units
                r = self.__imageData.shape[0]
                c = self.__imageData.shape[1]

                xmin = max((x-0.5*width), 0)
                xmax = min((x+0.5*width), c)
                ymin = max((y-0.5*width), 0)
                ymax = min((y+0.5*width), r)
                
                i1 = min(int(round(xmin)), c-1)
                i2 = min(int(round(xmax)), c)
                j1 = min(int(round(ymin)),r-1)
                j2 = min(int(round(ymax)), r)
                if i1 == i2:
                    i2 = i1+1
                if j1 == j2:
                    j2 = j1+1
                if self.__selectionMask is None:
                    self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                     numpy.uint8)
                if self.__eraseMode:
                    self.__selectionMask[j1:j2, i1:i2] = 0 
                else:
                    self.__selectionMask[j1:j2, i1:i2] = 1
                emitsignal = True
        if emitsignal:
            #should this be made by the parent?
            self.plotImage(update = False)
            
            #inform the other widgets
            self._emitMaskChangedSignal()

    def _emitMaskChangedSignal(self):
        #inform the other widgets
        ddict = {}
        ddict['event'] = "selectionMaskChanged"
        ddict['current'] = self.__selectionMask * 1
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)
                            
    def emitMaskImageSignal(self, ddict):
        if QTVERSION < '4.0.0':
            qt.QObject.emit(self,
                        qt.PYSIGNAL('MaskImageWidgetSignal'),
                        ddict)
        else:
            qt.QObject.emit(self,
                        qt.SIGNAL('MaskImageWidgetSignal'),
                        ddict)

    def _zoomResetSignal(self):
        if DEBUG:
            print("_zoomResetSignal")
        self.graphWidget._zoomReset(replot=False)
        self.plotImage(True)

    def getOutputFileName(self):
        initdir = PyMcaDirs.outputDir
        if self.outputDir is not None:
            if os.path.exists(self.outputDir):
                initdir = self.outputDir
        filedialog = qt.QFileDialog(self)
        filedialog.setFileMode(filedialog.AnyFile)
        filedialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict["gioconda16"])))
        formatlist = ["ASCII Files *.dat",
                      "EDF Files *.edf",
                      'CSV(, separated) Files *.csv',
                      'CSV(; separated) Files *.csv',
                      'CSV(tab separated) Files *.csv']
        if hasattr(qt, "QStringList"):
            strlist = qt.QStringList()
        else:
            strlist = []
        for f in formatlist:
                strlist.append(f)
        if self._saveFilter is None:
            self._saveFilter =formatlist[0]
        filedialog.setFilters(strlist)
        filedialog.selectFilter(self._saveFilter)
        filedialog.setDirectory(initdir)
        ret = filedialog.exec_()
        if not ret:
            return ""
        filename = filedialog.selectedFiles()[0]
        if len(filename):
            filename = qt.safe_str(filename)
            self.outputDir = os.path.dirname(filename)
            self._saveFilter = qt.safe_str(filedialog.selectedFilter())
            filterused = "."+self._saveFilter[-3:]
            PyMcaDirs.outputDir = os.path.dirname(filename)
            if len(filename) < 4:
                filename = filename+ filterused
            elif filename[-4:] != filterused :
                filename = filename+ filterused
        else:
            filename = ""
        return filename

    def saveImageList(self, filename=None, imagelist=None, labels=None):
        imageList = []
        if labels is None:
            labels = []
        if imagelist is None:
            if self.__imageData is not None:
                imageList.append(self.__imageData)
                label = self.getGraphTitle()
                label.replace(' ', '_')
                labels.append(label)
                if self.__selectionMask is not None:
                    if self.__selectionMask.max() > 0:
                        imageList.append(self.__selectionMask)
                        labels.append(label+"_Mask")
        else:
            imageList = imagelist
            if len(labels) == 0:
                for i in range(len(imagelist)):
                    labels.append("Image%02d" % i)

        if not len(imageList):
            qt.QMessageBox.information(self,"No Data",
                            "Image list is empty.\nNothing to be saved")
            return
        if filename is None:
            filename = self.getOutputFileName()
            if not len(filename):return

        if filename.lower().endswith(".edf"):
            ArraySave.save2DArrayListAsEDF(imageList, filename, labels)
        elif filename.lower().endswith(".csv"):
            if "," in self._saveFilter:
                csvseparator = ","
            elif ";" in self._saveFilter:
                csvseparator = ";"
            else:
                csvseparator = "\t"
            ArraySave.save2DArrayListAsASCII(imageList, filename, labels,
                                             csv=True,
                                             csvseparator=csvseparator)
        else:
            ArraySave.save2DArrayListAsASCII(imageList, filename, labels,
                                             csv=False)


    def saveClippedSeenImageList(self):
        return self.saveClippedAndSubtractedSeenImageList(subtract=False)


    def saveClippedAndSubtractedSeenImageList(self, subtract=True):
        imageData = self.__imageData
        if imageData is None:
            return
        vmin = None
        label = self.getGraphTitle()
        if not len(label):
            label = "Image01"
        if self.colormap is not None:
            colormapIndex, autoscale, vmin, vmax = self.colormap[0:4]
            if not autoscale:
                imageData = imageData.clip(vmin, vmax)
                label += ".clip(%f,%f)" % (vmin, vmax)
        if subtract:
            if vmin is None:
                vmin = imageData.min()
            imageData = imageData-vmin
            label += "-%f" % vmin
        imageList = [imageData]
        labelList = [label]
        if self.__selectionMask is not None:
            if self.__selectionMask.max() > 0:
                imageList.append(self.__selectionMask)
                labelList.append(label+"_Mask")
        self.saveImageList(filename=None,
                           imagelist=imageList,
                           labels=labelList)

    def setDefaultColormap(self, colormapindex, logflag=False):
        self.__defaultColormap = COLORMAPLIST[min(colormapindex, len(COLORMAPLIST)-1)]
        if logflag:
            self.__defaultColormapType = spslut.LOG
        else:
            self.__defaultColormapType = spslut.LINEAR

    def closeEvent(self, event):
        if self._profileSelectionWindow is not None:
            self._profileSelectionWindow.close()
        if self.colormapDialog is not None:
            self.colormapDialog.close()
        return qt.QWidget.closeEvent(self, event)

    def setInfoText(self, text):
        return self.graphWidget.setInfoText(text)

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))
    container = MaskImageWidget()
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('edf') or\
           sys.argv[1].endswith('cbf') or\
           sys.argv[1].endswith('ccd') or\
           sys.argv[1].endswith('spe') or\
           sys.argv[1].endswith('tif') or\
           sys.argv[1].endswith('tiff'):
            container = MaskImageWidget(profileselection=True)
            import EdfFile
            edf = EdfFile.EdfFile(sys.argv[1])
            data = edf.GetData(0)
            container.setImageData(data)
        else:
            image = qt.QImage(sys.argv[1])
            #container.setQImage(image, image.width(),image.height())
            container.setQImage(image, 200, 200)
    else:
        container = MaskImageWidget(profileselection=True)
        data = numpy.arange(400 * 200).astype(numpy.int32)
        data.shape = 400, 200
        #data = numpy.eye(200)
        container.setImageData(data, xScale=(200, 800), yScale=(400., 800.))
        #data.shape = 100, 400
        #container.setImageData(None)
        #container.setImageData(data)
    container.show()
    def theSlot(ddict):
        print(ddict['event'])

    if QTVERSION < '4.0.0':
        qt.QObject.connect(container,
                           qt.PYSIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.setMainWidget(container)
        app.exec_loop()
    else:
        qt.QObject.connect(container,
                           qt.SIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.exec_()

if __name__ == "__main__":
    test()

