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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import RGBCorrelatorGraph
qt = RGBCorrelatorGraph.qt
QTVERSION = qt.qVersion()
if QTVERSION > '4.0.0':
    import RGBCorrelator
    from RGBCorrelatorWidget import ImageShapeDialog
from PyMca_Icons import IconDict
import numpy
import ColormapDialog
import spslut
import os
import PyMcaDirs
import ArraySave
try:
    import QPyMcaMatplotlibSave
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]
QWTVERSION4 = RGBCorrelatorGraph.QtBlissGraph.QWTVERSION4

if QWTVERSION4:
    raise "ImportError","QImageFilterWidget needs Qwt5"

if QTVERSION > '4.0.0':
    import PyQt4.Qwt5 as Qwt
else:
    import Qwt5 as Qwt


DEBUG = 0


class MyPicker(Qwt.QwtPlotPicker):
    def __init__(self, *var):
        Qwt.QwtPlotPicker.__init__(self, *var)
        self.__text = Qwt.QwtText()
        self.data = None

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
    def __init__(self, parent = None, rgbwidget=None, selection=False, colormap=False,
                 imageicons=False, standalonesave=True):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
            self.setCaption("PyMca - Image Selection Tool")
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
        self.colormap = None
        self.colormapDialog = None
        self.rgbWidget = rgbwidget

        self._build()

        self.__brushMenu  = None
        self.__brushMode  = False
        self.__eraseMode  = False
        self.__connected = True

        self.__setBrush2()
        
        self.outputDir   = None
        self._saveFilter = None
        
        self._buildConnections()
        self._matplotlibSaveImage = None

    def _build(self):        
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        standaloneSaving = True
        if QTVERSION > '4.0.0':
            if MATPLOTLIB:
                standaloneSaving = False
        self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self,
                                               selection = True,
                                               colormap=True,
                                               imageicons=True,
                                               standalonesave=standaloneSaving,
                                               standalonezoom=False)
        if not standaloneSaving:
            self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
            self._saveMenu = qt.QMenu()
            self._saveMenu.addAction(qt.QString("Data"),
                                     self.saveImageList)
            self._saveMenu.addAction(qt.QString("Standard Graphics"),
                                     self.graphWidget._saveIconSignal)
            self._saveMenu.addAction(qt.QString("Matplotlib") ,
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
        self.graphWidget.picker.setTrackerPen(qt.Qt.black)
        self.graphWidget.graph.enableSelection(False)
        self.graphWidget.graph.enableZoom(True)
        self.setSelectionMode(False)
        self._toggleSelectionMode()
        self.mainLayout.addWidget(self.graphWidget)

    def _buildConnections(self, widget = None):
        self.connect(self.graphWidget.hFlipToolButton,
                 qt.SIGNAL("clicked()"),
                 self._hFlipIconSignal)

        self.connect(self.graphWidget.colormapToolButton,
                     qt.SIGNAL("clicked()"),
                     self.selectColormap)

        self.connect(self.graphWidget.selectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self._toggleSelectionMode)
        text = "Toggle between Selection\nand Zoom modes"
        if QTVERSION > '4.0.0':
            self.graphWidget.selectionToolButton.setToolTip(text)

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
            self.connect(self.graphWidget.graph,
                     qt.SIGNAL("QtBlissGraphSignal"),
                     self._graphSignal)

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
        if QTVERSION < '4.0.0':
            qt.QObject.emit(self,
                        qt.PYSIGNAL('MaskImageWidgetSignal'),
                        ddict)
        else:
            qt.QObject.emit(self,
                        qt.SIGNAL('MaskImageWidgetSignal'),
                        ddict)

        

    def _buildAndConnectButtonBox(self):
        if self.rgbWidget is not None:
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
            self.replaceImageButton = qt.QPushButton(buttonBox)
            self.replaceImageButton.setIcon(icon)
            self.replaceImageButton.setText("REPLACE IMAGE")
            self.imageButtonBoxLayout.addWidget(self.addImageButton)
            self.imageButtonBoxLayout.addWidget(self.removeImageButton)
            self.imageButtonBoxLayout.addWidget(self.replaceImageButton)
            
            self.mainLayout.addWidget(buttonBox)
            
            self.connect(self.addImageButton, qt.SIGNAL("clicked()"), 
                        self._addImageClicked)
            self.connect(self.removeImageButton, qt.SIGNAL("clicked()"), 
                        self._removeImageClicked)
            self.connect(self.replaceImageButton, qt.SIGNAL("clicked()"), 
                        self._replaceImageClicked)

    def _setEraseSelectionMode(self):
        if DEBUG:print "_setEraseSelectionMode"
        self.__eraseMode = True
        self.__brushMode = True
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.ActiveOnly)
        self.graphWidget.graph.enableSelection(False)

    def _setRectSelectionMode(self):
        if DEBUG:print "_setRectSelectionMode"
        self.__eraseMode = False
        self.__brushMode = False
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.AlwaysOn)
        self.graphWidget.graph.enableSelection(True)
        
    def _setBrushSelectionMode(self):
        if DEBUG:print "_setBrushSelectionMode"
        self.__eraseMode = False
        self.__brushMode = True
        self.graphWidget.picker.setTrackerMode(Qwt.QwtPicker.ActiveOnly)
        self.graphWidget.graph.enableSelection(False)
        
    def _setBrush(self):
        if DEBUG:print "_setBrush"
        if self.__brushMenu is None:
            if QTVERSION < '4.0.0':
                self.__brushMenu = qt.QPopupMenu()
                self.__brushMenu.insertItem(qt.QString(" 1 Image Pixel Width"),
                                            self.__setBrush1)
                self.__brushMenu.insertItem(qt.QString(" 2 Image Pixel Width"),
                                            self.__setBrush2)
                self.__brushMenu.insertItem(qt.QString(" 3 Image Pixel Width"),
                                            self.__setBrush3)
                self.__brushMenu.insertItem(qt.QString(" 5 Image Pixel Width"),
                                            self.__setBrush4)
                self.__brushMenu.insertItem(qt.QString("10 Image Pixel Width"),
                                            self.__setBrush5)
                self.__brushMenu.insertItem(qt.QString("20 Image Pixel Width"),
                                            self.__setBrush6)
            else:
                self.__brushMenu = qt.QMenu()
                self.__brushMenu.addAction(qt.QString(" 1 Image Pixel Width"),
                                           self.__setBrush1)
                self.__brushMenu.addAction(qt.QString(" 2 Image Pixel Width"),
                                           self.__setBrush2)
                self.__brushMenu.addAction(qt.QString(" 3 Image Pixel Width"),
                                           self.__setBrush3)
                self.__brushMenu.addAction(qt.QString(" 5 Image Pixel Width"),
                                           self.__setBrush4)
                self.__brushMenu.addAction(qt.QString("10 Image Pixel Width"),
                                           self.__setBrush5)
                self.__brushMenu.addAction(qt.QString("20 Image Pixel Width"),
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
            self.graphWidget.graph.enableZoom(False)
            if QTVERSION < '4.0.0':
                self.graphWidget.selectionToolButton.setState(qt.QButton.On)
            else:
                self.graphWidget.selectionToolButton.setChecked(True)
            self.graphWidget.selectionToolButton.setDown(True)
            self.graphWidget.showImageIcons()
            
        else:
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
        
    def _resetSelection(self, owncall=True):
        if DEBUG:
            print "_resetSelection"
        if self.__imageData is None:
            return
        self.__selectionMask = numpy.zeros(self.__imageData.shape, numpy.uint8)
        self.plotImage(update = True)

        #inform the others
        if owncall:
            ddict = {}
            ddict['event'] = "resetSelection"
            ddict['id'] = id(self)
            if QTVERSION < '4.0.0':
                qt.QObject.emit(self,
                            qt.PYSIGNAL('MaskImageWidgetSignal'),
                            ddict)
            else:
                qt.QObject.emit(self,
                            qt.SIGNAL('MaskImageWidgetSignal'),
                            ddict)

    def setSelectionMask(self, mask, plot=True):
        self.__selectionMask = mask
        if plot:
            self.plotImage(update=False)

    def setImageData(self, data, clearmask=False):
        self.__imageData = data
        if clearmask:
            self.__selectionMask = None
        self.plotImage(update = True)
        
    def plotImage(self, update=True):
        if self.__imageData is None:
            self.graphWidget.graph.clear()
            self.graphWidget.picker.data = None
            return
        if update:
            if self.__selectionMask is None:
                self.__selectionMask = numpy.zeros(self.__imageData.shape,
                                                     numpy.uint8)
            self.getPixmapFromData()
            self.__pixmap0 = self.__pixmap.copy()
            self.graphWidget.picker.data = self.__imageData
            if self.colormap is None:
                self.graphWidget.picker.setTrackerPen(qt.Qt.black)
            elif int(str(self.colormap[0])) > 1:     #color
                self.graphWidget.picker.setTrackerPen(qt.Qt.black)
            else:
                self.graphWidget.picker.setTrackerPen(qt.Qt.green)
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
        if colormap is None:
            (self.__pixmap,size,minmax)= spslut.transform(\
                                self.__imageData,
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                spslut.TEMP,
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
        if self.colormap is None:
            for i in range(4):
                self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] * (1 - (0.2 * self.__selectionMask))).astype(numpy.uint8)
        elif int(str(self.colormap[0])) > 1:     #color
            for i in range(4):
                self.__pixmap[:,:,i]  = (self.__pixmap0[:,:,i] * (1 - (0.2 * self.__selectionMask))).astype(numpy.uint8)
        else:
            self.__pixmap[self.__selectionMask>0,0]    = 0x40
            self.__pixmap[self.__selectionMask>0,2]    = 0x70
            self.__pixmap[self.__selectionMask>0,1]    = self.__pixmap0[self.__selectionMask>0, 1] 
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
        self.rgbWidget.addImage(self.__imageData,
                                str(self.graphWidget.graph.title().text()))

    def _removeImageClicked(self):
        self.rgbWidget.removeImage(str(self.graphWidget.graph.title().text()))

    def _replaceImageClicked(self):
        self.rgbWidget.reset()
        self.rgbWidget.addImage(self.__imageData,
                                str(self.graphWidget.graph.title().text()))
        if self.rgbWidget.isHidden():
            self.rgbWidget.show()
        if self.tab is None:
            self.rgbWidget.show()
            self.rgbWidget.raise_()
        else:
            self.tab.setCurrentWidget(self.rgbWidget)

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
            ddict = {}
            ddict['event'] = "selectionMaskChanged"
            ddict['current'] = self.__selectionMask * 1
            ddict['id'] = id(self)
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
            print "_zoomResetSignal"
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
        strlist = qt.QStringList()
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

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    container = MaskImageWidget()
    data = numpy.arange(10000)
    data.shape = 100, 100
    container.setImageData(data)
    container.show()
    def theSlot(ddict):
        print ddict['event']

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
        
