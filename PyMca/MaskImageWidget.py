#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import RGBCorrelatorGraph
qt = RGBCorrelatorGraph.qt
IconDict = RGBCorrelatorGraph.IconDict
QWTVERSION4 = RGBCorrelatorGraph.QtBlissGraph.QWTVERSION4
QTVERSION = qt.qVersion()
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
MATPLOTLIB = False
if QTVERSION > '4.0.0':
    import RGBCorrelator
    try:
        import QPyMcaMatplotlibSave
        MATPLOTLIB = True
    except ImportError:
        MATPLOTLIB = False
else:
    qt.QIcon = qt.QIconSet
import numpy
import ColormapDialog
import spslut
import os
import PyMcaDirs
import ArraySave
try:
    from PyMca import ProfileScanWidget
except ImportError:
    import ProfileScanWidget
try:
    from PyMca import SpecfitFuns
except:
    import SpecfitFuns


COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]

if QWTVERSION4:
    raise ImportError("QImageFilterWidget needs Qwt5")

if QTVERSION > '4.0.0':
    import PyQt4.Qwt5 as Qwt
else:
    import Qwt5 as Qwt


DEBUG = 0

if QTVERSION < '4.6.0':
    USE_PICKER = True
else:
    USE_PICKER = False
class MyPicker(Qwt.QwtPlotPicker):
    def __init__(self, *var):
        Qwt.QwtPlotPicker.__init__(self, *var)
        self.__text = Qwt.QwtText()
        self.data = None

    if USE_PICKER:
        def trackerText(self, var):
            d=self.invTransform(var)
            if self.data is None:
                self.__text.setText("%g, %g" % (d.x(), d.y()))
            else:
                limits = self.data.shape
                x = round(d.y())
                y = round(d.x())
                if x < 0: x = 0
                if y < 0: y = 0
                x = min(int(x), limits[0]-1)
                y = min(int(y), limits[1]-1)
                z = self.data[x, y]
                self.__text.setText("%d, %d, %.4g" % (y, x, z))
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
        self.colormap = None
        self.colormapDialog = None
        self.setDefaultColormap(2, False)
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
            self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
            self._saveMenu = qt.QMenu()
            self._saveMenu.addAction(QString("Image Data"),
                                     self.saveImageList)
            self._saveMenu.addAction(QString("Standard Graphics"),
                                     self.graphWidget._saveIconSignal)
            if QTVERSION > '4.0.0':
                if MATPLOTLIB:
                    self._saveMenu.addAction(QString("Matplotlib") ,
                                     self._saveMatplotlibImage)

        self.connect(self.graphWidget.zoomResetToolButton,
                     qt.SIGNAL("clicked()"), 
                     self._zoomResetSignal)
        self.graphWidget.picker = MyPicker(Qwt.QwtPlot.xBottom,
                           Qwt.QwtPlot.yLeft,
                           Qwt.QwtPicker.NoSelection,
                           Qwt.QwtPlotPicker.CrossRubberBand,
                           Qwt.QwtPicker.AlwaysOn,
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

        if ddict['event'] in [None, "NONE"]:
            #Nothing to be made
            return

        #
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
        try:
            title = self.graphWidget.graph.title().text()
            if sys.version < '3.0':
                title = str(title)
        except:
            title = ""
        self._profileSelectionWindow.setTitle(title)
        if self._profileScanWindow is not None:
            self._profileSelectionWindow.label.setText(title)
        self._profileSelectionWindow.show()
        #self._profileSelectionWindow.raise_()
        if ddict['event'] == 'PolygonModeChanged':
            return
        shape = self.__imageData.shape
        width = ddict['pixelwidth'] - 1
        if ddict['mode'].upper() in ["HLINE", "HORIZONTAL"]:
            if width < 1:
                row = int(ddict['y'][0])
                if row < 0:
                    row = 0
                if row >= shape[0]:
                    row = shape[0] - 1
                ydata  = self.__imageData[row, :]
                legend = "Row = %d"  % row
            else:
                row0 = int(ddict['y'][0]) - 0.5 * width
                if row0 < 0:
                    row0 = 0
                    row1 = row0 + width
                else:
                    row1 = int(ddict['y'][0]) + 0.5 * width
                if row1 >= shape[0]:
                    row1 = shape[0] - 1
                    row0 = max(0, row1 - width)
                ydata = self.__imageData[row0:int(row1+1), :].sum(axis=0)
                legend = "Row = %d to %d"  % (row0, row1)
            xdata  = numpy.arange(shape[1]).astype(numpy.float)
        elif ddict['mode'].upper() in ["VLINE", "VERTICAL"]:
            if width < 1:
                column = int(ddict['x'][0])
                if column < 0:
                    column = 0
                if column >= shape[1]:
                    column = shape[1] - 1
                ydata  = self.__imageData[:, column]
                legend = "Column = %d"  % column
            else:
                col0 = int(ddict['x'][0]) - 0.5 * width
                if col0 < 0:
                    col0 = 0
                    col1 = col0 + width
                else:
                    col1 = int(ddict['x'][0]) + 0.5 * width
                if col1 >= shape[1]:
                    col1 = shape[1] - 1
                    col0 = max(0, col1 - width)
                ydata = self.__imageData[:, col0:int(col1+1)].sum(axis=1)
                legend = "Col = %d to %d"  % (col0, col1)
            xdata  = numpy.arange(shape[0]).astype(numpy.float)
        elif ddict['mode'].upper() in ["LINE"]:
            if len(ddict['x']) == 1:
                #only one point given
                return
            #the coordinates of the reference points
            x0 = numpy.arange(float(shape[0]))
            y0 = numpy.arange(float(shape[1]))
            #get the interpolation points
            col0, col1 = [int(x) for x in ddict['x']]
            row0, row1 = [int(x) for x in ddict['y']]
            deltaCol = abs(col0 - col1)
            deltaRow = abs(row0 - row1)
            if deltaCol > deltaRow:
                npoints = deltaCol+1
            else:    
                npoints = deltaRow+1
            if width < 1:
                x = numpy.zeros((npoints, 2), numpy.float)
                x[:, 0] = numpy.linspace(row0, row1, npoints)
                x[:, 1] = numpy.linspace(col0, col1, npoints)
                legend = "From (%.3f, %.3f) to (%.3f, %.3f)" % (col0, row0, col1, row1)
                #perform the interpolation
                ydata = self._interpolate((x0, y0), self.__imageData, x)
                xdata = numpy.arange(float(npoints))
            elif deltaCol == 0:
                #vertical line
                col0 = int(ddict['x'][0]) - 0.5 * width
                if col0 < 0:
                    col0 = 0
                    col1 = col0 + width
                else:
                    col1 = int(ddict['x'][0]) + 0.5 * width
                if col1 >= shape[1]:
                    col1 = shape[1] - 1
                    col0 = max(0, col1 - width)
                ydata = self.__imageData[:, col0:int(col1+1)].sum(axis=1)
                legend = "Col = %d to %d"  % (col0, col1)
                npoints = max(ydata.shape)
                xdata = numpy.arange(float(npoints))
            elif deltaRow == 0:
                #horizontal line
                row0 = int(ddict['y'][0]) - 0.5 * width
                if row0 < 0:
                    row0 = 0
                    row1 = row0 + width
                else:
                    row1 = int(ddict['y'][0]) + 0.5 * width
                if row1 >= shape[0]:
                    row1 = shape[0] - 1
                    row0 = max(0, row1 - width)
                ydata = self.__imageData[row0:int(row1+1), :].sum(axis=0)
                legend = "Row = %d to %d"  % (row0, row1)
                npoints = max(ydata.shape)
                xdata = numpy.arange(float(npoints))
            else:
                #find m and b in the line y = mx + b
                m = (row1 - row0) / float((col1 - col0))
                b = row0 - m * col0
                alpha = numpy.arctan(m)
                npoints = deltaCol + 1
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

                tmpX   = numpy.linspace(newCol0, newCol1, npoints)
                rotMatrix = numpy.zeros((2,2), numpy.float)
                rotMatrix[0,0] =   cosalpha
                rotMatrix[0,1] =   sinalpha
                rotMatrix[1,0] =   cosalpha
                rotMatrix[1,1] = - sinalpha
                if DEBUG:
                    #test if I recover the original points
                    testX = numpy.zeros((1, 2) , numpy.float)
                    xy = numpy.dot(testX, rotMatrix)
                    print("Recovered X0 = %f" % (xy[0,0] + col0))
                    print("Recovered Y0 = %f" % (xy[0,1] + row0))
                    print("It should be = %f, %f" % (col0, row0))
                    testX[0,0] = newCol1
                    testX[0,1] = newRow1
                    xy = numpy.dot(testX, rotMatrix)
                    print("Recovered X1 = %f" % (xy[0,0] + col0))
                    print("Recovered Y1 = %f" % (xy[0,1] + row0))
                    print("It should be = %f, %f" % (col1, row1))

                x = numpy.zeros((npoints, 2) , numpy.float)
                tmpMatrix = numpy.zeros((npoints, 2) , numpy.float)
                ydata = numpy.zeros((npoints, ) , numpy.float)
                #oversampling solves noise introduction issues
                oversampling = (width+1)
                ncontributors = int(width * oversampling)
                if ncontributors == 0:
                    x[:, 0] = tmpX
                    #x[:, 1] = 0.0
                    xy = numpy.dot(x, rotMatrix)
                    tmpMatrix[:,1] = xy[:,0] + col0
                    tmpMatrix[:,0] = xy[:,1] + row0
                    rowLimits = [tmpMatrix[0,0], tmpMatrix[-1,0]]
                    colLimits = [tmpMatrix[0,1], tmpMatrix[-1,1]]
                    for a in rowLimits:
                        if (a >= shape[0]) or (a < 0):
                            print("outside row limits",a)
                            return
                    for a in colLimits:
                        if (a >= shape[1]) or (a < 0):
                            print("outside column limits",a)
                            return
                    ydata += self._interpolate((x0, y0), self.__imageData, tmpMatrix)
                else:
                    x[:, 0] = tmpX
                    for i in range(ncontributors):
                        x[:, 1] = (i - 0.5) * (width/oversampling)
                        xy = numpy.dot(x, rotMatrix)
                        tmpMatrix[:,1] = xy[:,0] + col0
                        tmpMatrix[:,0] = xy[:,1] + row0
                        rowLimits = [tmpMatrix[0,0], tmpMatrix[-1,0]]
                        colLimits = [tmpMatrix[0,1], tmpMatrix[-1,1]]
                        for a in rowLimits:
                            if (a >= shape[0]) or (a < 0):
                                print("outside row limits",a)
                                return
                        for a in colLimits:
                            if (a >= shape[1]) or (a < 0):
                                print("outside column limits",a)
                                return
                        ydata += self._interpolate((x0, y0), self.__imageData, tmpMatrix)
                    ydata /= oversampling
                xdata = numpy.arange(float(npoints))
                legend = "y = %f (x-%.1f) + %f ; width=%d" % (m, col0, b, width+1)
        else:
            if DEBUG:
                print("Mode %s not supported yet" % ddict['mode'])
            return
        info = {}
        info['xlabel'] = "Point"
        info['ylabel'] = "Z"
        self._profileSelectionWindow.addCurve(xdata, ydata, legend=legend, info=info,
                                          replot=False, replace=True)

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

    def _hFlipIconSignal(self):
        if QWTVERSION4:
            qt.QMessageBox.information(self, "Flip Image", "Not available under PyQwt4")
            return
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
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.ActiveOnly)
        self.graphWidget.graph.enableSelection(False)

    def _setRectSelectionMode(self):
        if DEBUG:
            print("_setRectSelectionMode")
        self.__eraseMode = False
        self.__brushMode = False
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.AlwaysOn)
        self.graphWidget.graph.enableSelection(True)
        
    def _setBrushSelectionMode(self):
        if DEBUG:
            print("_setBrushSelectionMode")
        self.__eraseMode = False
        self.__brushMode = True
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.ActiveOnly)
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
        if mode:
            self.graphWidget.graph.enableSelection(True)
            self.__brushMode  = False
            self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.AlwaysOn)
            if QTVERSION < '4.0.0':
                self.graphWidget.selectionToolButton.setState(qt.QButton.On)
            else:
                self.graphWidget.hideProfileSelectionIcons()
                self.graphWidget.selectionToolButton.setChecked(True)
            self.graphWidget.graph.enableZoom(False)
            self.graphWidget.selectionToolButton.setDown(True)
            self.graphWidget.showImageIcons()
            
        else:
            self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.AlwaysOff)
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
            maxValue = self.__imageData.max()
            minValue = self.__imageData.min()
        else:
            minValue = self.colormap[2]
            maxValue = self.colormap[3]

        return minValue, maxValue

    def _selectMax(self):
        self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        self.__selectionMask[self.__imageData >= maxValue] = 1
        self.plotImage(update=False)
        self._emitMaskChangedSignal()
        
    def _selectMiddle(self):
        self.__selectionMask = numpy.ones(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        self.__selectionMask[self.__imageData >= maxValue] = 0
        self.__selectionMask[self.__imageData <= minValue] = 0
        self.plotImage(update=False)
        self._emitMaskChangedSignal()        

    def _selectMin(self):
        self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                             numpy.uint8)
        minValue, maxValue = self._getSelectionMinMax()
        self.__selectionMask[self.__imageData <= minValue] = 1
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
        if self.__imageData is None:
            return
        self.__selectionMask = numpy.zeros(self.__imageData.shape, numpy.uint8)
        self.plotImage(update = True)

        #inform the others
        if owncall:
            ddict = {}
            ddict['event'] = "resetSelection"
            ddict['id'] = id(self)
            self.emitMaskImageSignal(ddict)
            
    def setSelectionMask(self, mask, plot=True):
        self.__selectionMask = mask
        if plot:
            self.plotImage(update=False)

    def getSelectionMask(self):
        return self.__selectionMask

    def setImageData(self, data, clearmask=False):
        self.__image = None
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
            minData = self.__imageData.min()
            maxData = self.__imageData.max()
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
        self.__pixmap0 = pixmap
        if clearmask:
            self.__selectionMask = None
        self.plotImage(update = True)
        
    def plotImage(self, update=True):
        if self.__imageData is None:
            self.graphWidget.graph.clear()
            self.graphWidget.picker.data = None
            return
        
        if self.__selectionMask is None:
            self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                                 numpy.uint8)
        if update:
            self.getPixmapFromData()
            self.__pixmap0 = self.__pixmap.copy()
            self.graphWidget.picker.data = self.__imageData
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
                                        ymirror = not self._y1AxisInverted)
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

        if colormap is None:
            (self.__pixmap,size,minmax)= spslut.transform(\
                                self.__imageData,
                                (1,0),
                                (self.__defaultColormapType,3.0),
                                "BGRX",
                                self.__defaultColormap,
                                1,
                                (0,1),
                                (0, 255), 1)
        else:
            if len(colormap) < 7: colormap.append(spslut.LINEAR)
            (self.__pixmap,size,minmax)= spslut.transform(\
                                self.__imageData,
                                (1,0),
                                (colormap[6],3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]),
                                (0,255), 1)

        self.__pixmap = self.__pixmap.astype(numpy.ubyte)

        self.__pixmap.shape = [self.__imageData.shape[0],
                                    self.__imageData.shape[1],
                                    4]

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
                    for i in range(4):
                        self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] *\
                                (1 - (0.2 * self.__selectionMask))).astype(numpy.uint8)
                else:
                    self.__pixmap = self.__pixmap0.copy()
                    self.__pixmap[self.__selectionMask>0,0]    = 0x40
                    self.__pixmap[self.__selectionMask>0,2]    = 0x70
                    self.__pixmap[self.__selectionMask>0,3]    = 0x40
        elif int(str(self.colormap[0])) > 1:     #color
            for i in range(4):
                self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] *\
                        (1 - (0.2 * self.__selectionMask))).astype(numpy.uint8)
        else:
            self.__pixmap = self.__pixmap0.copy()
            self.__pixmap[self.__selectionMask>0,0]    = 0x40
            self.__pixmap[self.__selectionMask>0,2]    = 0x70
            self.__pixmap[self.__selectionMask>0,3]    = 0x40
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
        minData = self.__imageData.min()
        maxData = self.__imageData.max()
        self.colormapDialog = ColormapDialog.ColormapDialog()
        self.colormapDialog.colormapIndex  = self.colormapDialog.colormapList.index("Temperature")
        self.colormapDialog.colormapString = "Temperature"
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
        self.colormapDialog.setDataMinMax(minData, maxData)
        self.colormapDialog.setAutoscale(1)
        self.colormapDialog.setColormap(self.colormapDialog.colormapIndex)
        self.colormap = (self.colormapDialog.colormapIndex,
                              self.colormapDialog.autoscale,
                              self.colormapDialog.minValue, 
                              self.colormapDialog.maxValue,
                              minData, maxData)
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
        ddict['title'] = str(self.graphWidget.graph.title().text())
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)
            
    def _removeImageClicked(self):
        ddict = {}
        ddict['event'] = "removeImageClicked"
        ddict['title'] = str(self.graphWidget.graph.title().text())
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)

    def _replaceImageClicked(self):
        ddict = {}
        ddict['event'] = "replaceImageClicked"
        ddict['image'] = self.__imageData
        ddict['title'] = str(self.graphWidget.graph.title().text())
        ddict['id'] = id(self)
        self.emitMaskImageSignal(ddict)

    def _saveToolButtonSignal(self):
        self._saveMenu.exec_(self.cursor().pos())

    def _saveMatplotlibImage(self):
        if self._matplotlibSaveImage is None:
            self._matplotlibSaveImage = QPyMcaMatplotlibSave.SaveImageSetup(None,
                                                            self.__imageData)
            self._matplotlibSaveImage.setWindowTitle("Matplotlib Image")
        else:
            self._matplotlibSaveImage.setImageData(self.__imageData)
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
            if ddict['xmin'] < ddict['xmax']:
                xmin = ddict['xmin']
                xmax = ddict['xmax']
            else:
                xmin = ddict['xmax']
                xmax = ddict['xmin']
            if ddict['ymin'] < ddict['ymax']:
                ymin = ddict['ymin']
                ymax = ddict['ymax']
            else:
                ymin = ddict['ymax']
                ymax = ddict['ymin']
            i1 = max(int(round(xmin)), 0)
            i2 = min(abs(int(round(xmax)))+1, self.__imageData.shape[1])
            j1 = max(int(round(ymin)),0)
            j2 = min(abs(int(round(ymax)))+1,self.__imageData.shape[0])
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
                width = self.__brushWidth   #in (row, column) units
                r = self.__imageData.shape[0]
                c = self.__imageData.shape[1]
                xmin = max((ddict['x']-0.5*width), 0)
                xmax = min((ddict['x']+0.5*width), c)
                ymin = max((ddict['y']-0.5*width), 0)
                ymax = min((ddict['y']+0.5*width), r)
                i1 = min(int(round(xmin)), c-1)
                i2 = min(int(round(xmax)), c)
                j1 = min(int(round(ymin)),r-1)
                j2 = min(int(round(ymax)), r)
                if i1 == i2: i2 = i1+1
                if j1 == j2: j2 = j1+1
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

    def _saveToolButtonSignal(self):
        self._saveMenu.exec_(self.cursor().pos())

    def _saveMatplotlibImage(self):
        if self._matplotlibSaveImage is None:
            self._matplotlibSaveImage = QPyMcaMatplotlibSave.SaveImageSetup(None,
                                                            self.__imageData)
            self._matplotlibSaveImage.setWindowTitle("Matplotlib Image")
        else:
            self._matplotlibSaveImage.setImageData(self.__imageData)
        self._matplotlibSaveImage.show()
        self._matplotlibSaveImage.raise_()    


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
        if not ret: return ""
        filename = filedialog.selectedFiles()[0]
        if len(filename):
            filename = str(filename)
            self.outputDir = os.path.dirname(filename)
            self._saveFilter = str(filedialog.selectedFilter())
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
                label = str(self.graphWidget.graph.title().text())
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

        if filename[-4:].lower() == ".edf":
            ArraySave.save2DArrayListAsEDF(imageList, filename, labels)
        elif filename[-4:].lower() == ".csv":
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


    def setDefaultColormap(self, colormapindex, logflag=False):
        self.__defaultColormap = COLORMAPLIST[min(colormapindex, len(COLORMAPLIST)-1)]
        if logflag:
            self.__defaultColormapType = spslut.LOG
        else:
            self.__defaultColormapType = spslut.LINEAR

    def closeEvent(self, event):
        if self._profileSelectionWindow is not None:
            self._profileSelectionWindow.close()
        qt.QWidget.closeEvent(self, event)


def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))
    container = MaskImageWidget()
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('edf') or\
           sys.argv[1].endswith('cbf'):
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
        data = numpy.arange(40000).astype(numpy.int32)
        data.shape = 200, 200
        container.setImageData(data)
        #data.shape = 100, 400
        #container.setImageData(None)
        #container.setImageData(data)
    container.show()
    def theSlot(ddict):
        print(ddict['event'])

    if QTVERSION < '4.0.0':
        qt.QObject.connect(container,
                       qt.PYSIGNAL("MaskImageWidgetSignal"),
                       updateMask)
        app.setMainWidget(container)
        app.exec_loop()
    else:
        qt.QObject.connect(container,
                           qt.SIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.exec_()

if __name__ == "__main__":
    test()
        
