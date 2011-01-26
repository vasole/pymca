#!/usr/bin/env python
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
import sys
import os
import numpy
import weakref

from PyMca import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
from PyMca import DataObject
from PyMca import McaWindow
from PyMca import StackBase
from PyMca import CloseEventNotifyingWidget
from PyMca import MaskImageWidget
from PyMca import StackROIWindow
from PyMca import RGBCorrelator
from PyMca.RGBCorrelatorWidget import ImageShapeDialog
from PyMca.PyMca_Icons import IconDict
from PyMca import StackSelector
from PyMca import PyMcaDirs
from PyMca import ArraySave
HDF5 = ArraySave.HDF5

DEBUG = 0
QTVERSION = qt.qVersion()
if DEBUG:
    StackBase.DEBUG = DEBUG

class QStackWidget(StackBase.StackBase,
                   CloseEventNotifyingWidget.CloseEventNotifyingWidget):
    def __init__(self, parent = None,
                 mcawidget = None,
                 rgbwidget = None,
                 vertical = False,
                 master = True):
        StackBase.StackBase.__init__(self)
        CloseEventNotifyingWidget.CloseEventNotifyingWidget.__init__(self,
                                                                     parent)
        
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.setWindowTitle("PyMCA - ROI Imaging Tool")
        screenHeight = qt.QDesktopWidget().height()
        if screenHeight > 0:
            if QTVERSION < '4.5.0':
                self.setMaximumHeight(int(0.99*screenHeight))
            self.setMinimumHeight(int(0.5*screenHeight))
        screenWidth = qt.QDesktopWidget().width()
        if screenWidth > 0:
            if QTVERSION < '4.5.0':
                self.setMaximumWidth(int(screenWidth)-5)
            self.setMinimumWidth(min(int(0.5*screenWidth),800))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.mcaWidget = mcawidget
        self.rgbWidget = rgbwidget
        self.master = master
        self._slave = None
        self._masterStack = None
        self.stackSelector = None
        self._build(vertical=vertical)
        self._buildBottom()
        self._buildConnections()
        self.__ROIConnected = True

    def _build(self, vertical=False):
        box = qt.QSplitter(self)
        if vertical:
            box.setOrientation(qt.Qt.Vertical)
        else:
            box.setOrientation(qt.Qt.Horizontal)

        self.stackWindow = qt.QWidget(box)
        self.stackWindow.mainLayout = qt.QVBoxLayout(self.stackWindow)
        self.stackWindow.mainLayout.setMargin(0)
        self.stackWindow.mainLayout.setSpacing(0)

        if HDF5:
            self.stackWidget = MaskImageWidget.MaskImageWidget(self.stackWindow,
                                                        selection=False,
                                                        standalonesave=False,
                                                        imageicons=False)
            self._stackSaveMenu = qt.QMenu()
            self._stackSaveMenu.addAction(QString("Stack as Spectra"),
                                                 self.saveStackAsNeXusSpectra)
            self._stackSaveMenu.addAction(QString("Stack as Images"),
                                                 self.saveStackAsNeXusImages)
            self._stackSaveMenu.addAction(QString("Stack as Float32 Spectra"),
                                                 self.saveStackAsFloat32NeXusSpectra)
            self._stackSaveMenu.addAction(QString("Stack as Float64 Spectra"),
                                                 self.saveStackAsFloat64NeXusSpectra)
            self._stackSaveMenu.addAction(QString("Stack as Float32 Images"),
                                                 self.saveStackAsFloat32NeXusImages)
            self._stackSaveMenu.addAction(QString("Stack as Float64 Images"),
                                                 self.saveStackAsFloat64NeXusImages)
            self._stackSaveMenu.addAction(QString("Stack as HDF5 /data"),
                                                 self.saveStackAsSimplestHDF5)
            self._stackSaveMenu.addAction(QString("Standard Graphics"),
                                self.stackWidget.graphWidget._saveIconSignal)
            self.connect(self.stackWidget.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._stackSaveToolButtonSignal)
        else:
            self.stackWidget = MaskImageWidget.MaskImageWidget(self.stackWindow,
                                                        selection=False,
                                                        imageicons=False)
            
        self.stackGraphWidget = self.stackWidget.graphWidget

        self.roiWindow = qt.QWidget(box)
        self.roiWindow.mainLayout = qt.QVBoxLayout(self.roiWindow)
        self.roiWindow.mainLayout.setMargin(0)
        self.roiWindow.mainLayout.setSpacing(0)
        standaloneSaving = True        
        self.roiWidget = MaskImageWidget.MaskImageWidget(parent=self.roiWindow,
                                                         rgbwidget=self.rgbWidget,
                                                         selection=True,
                                                         colormap=True,
                                                         imageicons=True,
                                                         standalonesave=standaloneSaving,
                                                         profileselection=True)
        infotext  = 'Toggle background subtraction from current image\n'
        infotext += 'subtracting a straight line between the ROI limits.'
        self.roiBackgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))  
        self.roiBackgroundButton = self.roiWidget.graphWidget._addToolButton(\
                                    self.roiBackgroundIcon,
                                    self._roiSubtractBackgroundClicked,
                                    infotext,
                                    toggle = True,
                                    state = False,
                                    position = 6)
        self.roiGraphWidget = self.roiWidget.graphWidget
        self.stackWindow.mainLayout.addWidget(self.stackWidget)
        self.roiWindow.mainLayout.addWidget(self.roiWidget)
        box.addWidget(self.stackWindow)
        box.addWidget(self.roiWindow)
        self.mainLayout.addWidget(box)


        #add some missing icons
        offset = 6
        infotext  = 'If checked, spectra will be added normalized to the number\n'
        infotext += 'of pixels. Be carefull if you are preparing a batch and you\n'
        infotext += 'fit the normalized spectra because the data in the batch will\n'
        infotext += 'have a different weight because they are not normalized.'
        self.normalizeIcon = qt.QIcon(qt.QPixmap(IconDict["normalize16"]))
        self.normalizeButton = self.stackGraphWidget._addToolButton(\
                                        self.normalizeIcon,
                                        self.normalizeIconChecked,
                                        infotext,
                                        toggle = True,
                                        state = False,
                                        position = 6)
        offset += 1

        if self.master:
            self.loadIcon = qt.QIcon(qt.QPixmap(IconDict["fileopen"]))  
            self.loadStackButton = self.stackGraphWidget._addToolButton(\
                                        self.loadIcon,
                                        self.loadSlaveStack,
                                        'Load another stack of same shape',
                                        position = offset)
            offset += 1
        
        self.pluginIcon     = qt.QIcon(qt.QPixmap(IconDict["plugin"]))
        infotext = "Call/Load Stack Plugins"
        self.stackGraphWidget._addToolButton(self.pluginIcon,
                                             self._pluginClicked,
                                             infotext,
                                             toggle = False,
                                             state = False,
                                             position = offset)
        
    def setStack(self, *var, **kw):
        self.stackWidget.setImageData(None)
        self.roiWidget.setImageData(None)
        StackBase.StackBase.setStack(self, *var, **kw)
        if (1 in self._stack.data.shape) and\
           isinstance(self._stack.data, numpy.ndarray):
            oldshape = self._stack.data.shape
            dialog = ImageShapeDialog(self, shape = oldshape[0:2])
            dialog.setModal(True)
            ret = dialog.exec_()
            if ret:
                shape = dialog.getImageShape()
                dialog.close()
                del dialog
                self._stack.data.shape = [shape[0], shape[1], oldshape[2]]
                self.stackWidget.setImageData(None)
                self.roiWidget.setImageData(None)
                StackBase.StackBase.setStack(self, self._stack, **kw)

    def normalizeIconChecked(self):
        pass

    def _roiSubtractBackgroundClicked(self):
        if not len(self._ROIImageList):
            return
        if self.roiBackgroundButton.isChecked():
            self.roiWidget.setImageData(self._ROIImageList[0]-\
                                        self._ROIImageList[-1])
            self.roiWidget.graphWidget.graph.setTitle(self._ROIImageNames[0]+\
                                                      " Net")
        else:
            self.roiWidget.setImageData(self._ROIImageList[0])
            self.roiWidget.graphWidget.graph.setTitle(self._ROIImageNames[0])

    def _stackSaveToolButtonSignal(self):
        self._stackSaveMenu.exec_(self.cursor().pos())

    def _getOutputHDF5Filename(self, nexus=False):
        fileTypes = "HDF5 Files (*.h5)\nHDF5 Files (*.hdf)"
        message = "Enter output filename"
        wdir = PyMcaDirs.outputDir
        filename = qt.QFileDialog.getSaveFileName(self, message, wdir, fileTypes)
        if len(filename):
            try:
                return str(filename)
            except UnicodeEncodeError:
                if 0:
                    return str(unicode(filename).encode('utf-8'))
                else:
                    msg = qt.QMessageBox(self)
                    msg.setWindowTitle("Encoding error")
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("Please use ASCII characters in file name and path")
                    msg.exec_()
        return ""

    def saveStackAsNeXus(self, dtype=None, interpretation=None):
        mcaIndex = self._stack.info.get('McaIndex', -1)
        if interpretation is None:
            if mcaIndex in [0, -1]:
                interpretation = "spectrum"
            else:
                interpretation = "image"
        if interpretation not in ["spectrum", "image"]:
            raise ValueError("Unknown data interpretation %s" % interpretation)
        filename = self._getOutputHDF5Filename()
        if not len(filename):
            return
        ArraySave.save3DArrayAsHDF5(self._stack.data,
                                    filename,
                                    labels = None,
                                    dtype=dtype,
                                    mode='nexus',
                                    mcaindex=mcaIndex,
                                    interpretation=interpretation)

    def saveStackAsNeXusSpectra(self):
        try:
            self.saveStackAsNeXus(interpretation="spectrum")
        except:
            msg = qt.QMessageBox(self)
            msg.setWindowTitle("Error saving stack")
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("%s: %s" % (sys.exc_info()[0], sys.exc_info()[1]))
            msg.exec_()
            if DEBUG:
                raise


    def saveStackAsNeXusImages(self):
        self.saveStackAsNeXus(interpretation="image")

    def saveStackAsFloat32NeXusSpectra(self):
        self.saveStackAsNeXus(dtype=numpy.float32, interpretation="spectrum")

    def saveStackAsFloat64NeXusSpectra(self):
        self.saveStackAsNeXus(dtype=numpy.float64, interpretation="spectrum")

    def saveStackAsFloat32NeXusImages(self):
        self.saveStackAsNeXus(dtype=numpy.float32, interpretation="image")

    def saveStackAsFloat64NeXusImages(self):
        self.saveStackAsNeXus(dtype=numpy.float64, interpretation="image")

    def saveStackAsNeXusPlus(self):
        filename = self._getOutputHDF5Filename()
        if not len(filename):
            return
        ArraySave.save3DArrayAsHDF5(self._stack.data, filename,
                                    labels = None, dtype=None, mode='nexus+')

    def saveStackAsSimpleHDF5(self):
        filename = self._getOutputHDF5Filename()
        if not len(filename):
            return
        ArraySave.save3DArrayAsHDF5(self._stack.data, filename,
                                    labels = None, dtype=None, mode='simple')

    def saveStackAsSimplestHDF5(self):
        filename = self._getOutputHDF5Filename()
        if not len(filename):
            return            
        ArraySave.save3DArrayAsHDF5(self._stack.data, filename, labels = None, dtype=None, mode='simplest')

    def loadStack(self):
        if self._stackImageData is not None:
            #clear with a small stack
            stack = DataObject.DataObject()
            stack.data = numpy.zeros((100,100,100), numpy.float32)
            self.setStack(stack)
        if self.stackSelector  is None:
            self.stackSelector = StackSelector.StackSelector(self)
        stack = self.stackSelector.getStack()
        if type(stack) == type([]):
            #aifira like, two stacks
            self.setStack(stack[0])
            self._slave = None
            slave = QStackWidget(master=False, rgbwidget=self.rgbWidget)
            slave.setStack(stack[1])
            self.setSlave(slave)
        else:
            self.setStack(stack)

    def setStack(self, *var, **kw):
        self.stackWidget.setImageData(None)
        self.roiWidget.setImageData(None)
        StackBase.StackBase.setStack(self, *var, **kw)
        if (1 in self._stack.data.shape) and\
           isinstance(self._stack.data, numpy.ndarray):
            oldshape = self._stack.data.shape
            dialog = ImageShapeDialog(self, shape = oldshape[0:2])
            dialog.setModal(True)
            ret = dialog.exec_()
            if ret:
                shape = dialog.getImageShape()
                dialog.close()
                del dialog
                self._stack.data.shape = [shape[0], shape[1], oldshape[2]]
                self.stackWidget.setImageData(None)
                self.roiWidget.setImageData(None)
                StackBase.StackBase.setStack(self, self._stack, **kw)

    def loadSlaveStack(self):
        if self._slave is not None:
            actionList = ['Load Slave', 'Show Slave', 'Delete Slave']
            menu = qt.QMenu(self)
            for action in actionList:
                text = QString(action)
                menu.addAction(text)
            a = menu.exec_(qt.QCursor.pos())
            if a is None:
                return None
            if str(a.text()).startswith("Load"):
                self._slave = None
            elif str(a.text()).startswith("Show"):
                self._slave.show()
                self._slave.raise_()
                return
            else:
                self._slave = None
                return
        if self.stackSelector  is None:
            self.stackSelector = StackSelector.StackSelector(self)

        try:
            stack = self.stackSelector.getStack()
        except:
            txt = "%s" % sys.exc_info()[1]
            if txt.startswith("Incomplete selection"):
                return
            msg = qt.QMessageBox(self)
            msg.setWindowTitle("Error loading slave stack")
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("%s: %s" % (sys.exc_info()[0], sys.exc_info()[1]))
            msg.exec_()
            return
        if stack is None:
            return
        if type(stack) == type([]):
            stack = stack[0]

        slave = QStackWidget(rgbwidget=self.rgbWidget,
                                  master=False)
        slave.setStack(stack)
        self.setSlave(slave)


    def setSlave(self, slave):
        self._slave = None
        self._slave = slave
        self._slave.setSelectionMask(self.getSelectionMask())
        self._slave.show()
        self._slave._setMaster(self)

    def _setMaster(self, master=None):
        if self.master:
            self._masterStack = None
            return
        if master is None:
            master = self
        self._masterStack = weakref.proxy(master)
        
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
            if m == "PyMcaPlugins.StackPluginBase":
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
            methods = self.pluginInstanceDict[m].getMethods()
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
            dirName = str(qt.QFileDialog.getExistingDirectory(self,
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
            StackBase.DEBUG = DEBUG
            return
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods()
        if len(methods) == 1:
            idx = 0
        else:
            actionList = []
            methods.sort()
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
            msg.setText("%s" % sys.exc_info()[1])
            msg.exec_()
            if DEBUG:
                raise

    def _actionHovered(self, action):
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)
        

    def _buildBottom(self):
        n = 0
        self.tab = None
        if self.mcaWidget is None:
            n += 1
        if self.rgbWidget is None:
            n += 1
        if n == 1:
            if self.mcaWidget is None:
                self.mcaWidget = McaWindow.McaWidget(self, vertical = False)
                self.mcaWidget.setWindowTitle("PyMCA - Mca Window")
                self.mainLayout.addWidget(self.mcaWidget)
            if self.rgbWidget is None:
                self.rgbWidget = RGBCorrelator.RGBCorrelator(self)
                self.mainLayout.addWidget(self.rgbWidget)
        elif n == 2:
            self.tab = qt.QTabWidget(self)
            self.mcaWidget = McaWindow.McaWidget(vertical = False)
            self.mcaWidget.graphBox.setMinimumWidth(0.5 * qt.QWidget.sizeHint(self).width())
            self.tab.setMaximumHeight(1.3 * qt.QWidget.sizeHint(self).height())
            self.mcaWidget.setWindowTitle("PyMCA - Mca Window")
            self.tab.addTab(self.mcaWidget, "MCA")
            self.rgbWidget = RGBCorrelator.RGBCorrelator()
            self.tab.addTab(self.rgbWidget, "RGB Correlator")
            self.mainLayout.addWidget(self.tab)
        self.mcaWidget.setMiddleROIMarkerFlag(True)        

    def _buildAndConnectButtonBox(self):
        #the MCA selection
        self.mcaButtonBox = qt.QWidget(self.stackWindow)
        self.mcaButtonBoxLayout = qt.QHBoxLayout(self.mcaButtonBox)
        self.mcaButtonBoxLayout.setMargin(0)
        self.mcaButtonBoxLayout.setSpacing(0)
        self.addMcaButton = qt.QPushButton(self.mcaButtonBox)
        self.addMcaButton.setText("ADD MCA")
        self.removeMcaButton = qt.QPushButton(self.mcaButtonBox)
        self.removeMcaButton.setText("REMOVE MCA")
        self.replaceMcaButton = qt.QPushButton(self.mcaButtonBox)
        self.replaceMcaButton.setText("REPLACE MCA")
        self.mcaButtonBoxLayout.addWidget(self.addMcaButton)
        self.mcaButtonBoxLayout.addWidget(self.removeMcaButton)
        self.mcaButtonBoxLayout.addWidget(self.replaceMcaButton)
        
        self.stackWindow.mainLayout.addWidget(self.mcaButtonBox)

        self.connect(self.addMcaButton, qt.SIGNAL("clicked()"), 
                    self._addMcaClicked)
        self.connect(self.removeMcaButton, qt.SIGNAL("clicked()"), 
                    self._removeMcaClicked)
        self.connect(self.replaceMcaButton, qt.SIGNAL("clicked()"), 
                    self._replaceMcaClicked)

        if self.rgbWidget is not None:
            # The IMAGE selection
            self.roiWidget.buildAndConnectImageButtonBox()

    def _buildConnections(self):
        self._buildAndConnectButtonBox()

        #self.connect(self.stackGraphWidget.hFlipToolButton,
        #     qt.SIGNAL("clicked()"),
        #     self._hFlipIconSignal)

        #ROI Image
        widgetList = [self.stackWidget, self.roiWidget]
        for widget in widgetList:
            self.connect(widget,
                 qt.SIGNAL('MaskImageWidgetSignal'),
                 self._maskImageWidgetSlot)

        #self.stackGraphWidget.graph.canvas().setMouseTracking(1)
        self.stackGraphWidget.setInfoText("    X = ???? Y = ???? Z = ????")
        self.stackGraphWidget.showInfo()

        self.connect(self.stackGraphWidget.graph,
                 qt.SIGNAL("QtBlissGraphSignal"),
                 self._stackGraphSignal)
        self.connect(self.mcaWidget,
                 qt.SIGNAL("McaWindowSignal"),
                 self._mcaWidgetSignal)
        self.connect(self.roiWidget.graphWidget.graph,
                 qt.SIGNAL("QtBlissGraphSignal"),
                 self._stackGraphSignal)

    def showOriginalImage(self):
        self.stackGraphWidget.graph.setTitle("Original Stack")
        if self._stackImageData is None:
            self.stackGraphWidget.graph.clear()
            return
        self.stackWidget.setImageData(self._stackImageData)

    def showOriginalMca(self):
        self.sendMcaSelection(self._mcaData0, action="ADD")

    def showROIImageList(self, imageList, image_names=None):
        if self.roiBackgroundButton.isChecked():
            self.roiWidget.setImageData(imageList[0]-imageList[-1])
            self.roiWidget.graphWidget.graph.setTitle(image_names[0]+\
                                                      " Net")
        else:
            self.roiWidget.setImageData(imageList[0])
            self.roiWidget.graphWidget.graph.setTitle(image_names[0])
        self._ROIImageList = imageList
        self._ROIImageNames = image_names
        self._stackROIImageListUpdated()

    def addImage(self, image, name, info=None, replace=False, replot=True):
        self.rgbWidget.addImage(image, name)
        if self.tab is None:
            if self.master:
                self.rgbWidget.show()
        else:
            self.tab.setCurrentWidget(self.rgbWidget)

    def removeImage(self, title):
        self.rgbWidget.removeImage(title)

    def replaceImage(self, image, title, info=None, replace=True, replot=True):
        self.rgbWidget.reset()
        self.rgbWidget.addImage(image, title)
        if self.rgbWidget.isHidden():
            self.rgbWidget.show()
        if self.tab is None:
            self.rgbWidget.show()
            self.rgbWidget.raise_()
        else:
            self.tab.setCurrentWidget(self.rgbWidget)
            
    def _addImageClicked(self, image, title):
        self.addImage(image, title)

    def _removeImageClicked(self, title):
        self.rgbWidget.removeImage(title)

    def _replaceImageClicked(self, image, title):
        self.replaceImage(image, title)

    def __getLegend(self):
        if self._selectionMask is None:
            legend = "Stack SUM"
        elif self._selectionMask.sum() == 0:
            legend = "Stack SUM"
        else:
            title = str(self.roiGraphWidget.graph.title().text())
            legend = "Stack " + title + " selection"
        return legend

    def _addMcaClicked(self, action=None):
        if action is None:
            action = "ADD"
        if self._stackImageData is None:
            return

        if self.normalizeButton.isChecked():        
            dataObject = self.calculateMcaDataObject(normalize=True)
        else:
            dataObject = self.calculateMcaDataObject(normalize=False)
        legend = self.__getLegend()

        if self.normalizeButton.isChecked():
            if self._selectionMask is None:
                npixels = self._stackImageData.shape[0] *\
                          self._stackImageData.shape[1]
            else:
                npixels = self._selectionMask.sum()
                if npixels == 0:
                    npixels = self._stackImageData.shape[0] *\
                              self._stackImageData.shape[1]
            legend += "/%d" % npixels
        return self.sendMcaSelection(dataObject,
                          key = "Selection",
                          legend =legend,
                          action = action)        
    
    def _removeMcaClicked(self):
        #remove the mca
        #dataObject = self.__mcaData0
        #send a dummy object
        dataObject = DataObject.DataObject()
        legend = self.__getLegend()
        if self.normalizeButton.isChecked():
            legend += "/"
            curves = self.mcaWidget.graph.curves.keys()
            for curve in curves:
                if curve.startswith(legend):
                    legend = curve
                    break
        self.sendMcaSelection(dataObject, legend = legend, action = "REMOVE")
    
    def _replaceMcaClicked(self):
        #replace the mca
        self.__ROIConnected = False
        self._addMcaClicked(action="REPLACE")
        self.__ROIConnected = True
    
    def sendMcaSelection(self, mcaObject, key = None, legend = None, action = None):
        if action is None:
            action = "ADD"
        if key is None:
            key = "SUM"
        if legend is None:
            legend = "Stack SUM"
            if self.normalizeButton.isChecked():
                npixels = self._stackImageData.shape[0] *\
                          self._stackImageData.shape[1]
                legend += "/%d" % npixels
        sel = {}
        sel['SourceName'] = "EDF Stack"
        sel['Key']        =  key
        sel['legend']     =  legend
        sel['dataobject'] =  mcaObject
        if action == "ADD":
            self.mcaWidget._addSelection([sel])
        elif action == "REMOVE":
            self.mcaWidget._removeSelection([sel])
        elif action == "REPLACE":
            self.mcaWidget._replaceSelection([sel])
        elif action == "GET_CURRENT_SELECTION":
            return sel
        if self.tab is None:
            self.mcaWidget.show()
            self.mcaWidget.raise_()
        else:
            self.tab.setCurrentWidget(self.mcaWidget)

    def setSelectionMask(self, mask, instance_id=None):
        self._selectionMask = mask
        if instance_id == id(self):
            return

        if DEBUG:
            if self._slave is not None:
                print("MASTER  setSelectionMask CALLED")
            elif self._masterStack is not None:
                print("SLAVE setSelectionMask CALLED")

        #inform built in widgets
        for widget in [self.stackWidget, self.roiWidget]:
            if instance_id != id(widget):
                if mask is None:
                    widget._resetSelection(owncall=False)
                else:
                    widget.setSelectionMask(mask, plot=True)

        #inform slave
        if self._slave is not None:
            #This is a master instance
            instanceList = [id(self._slave),
                            id(self._slave.stackWidget),
                            id(self._slave.roiWidget)]
            for key in self._slave.pluginInstanceDict.keys():
                instanceList.append(id(self._slave.pluginInstanceDict[key]))
            if instance_id not in instanceList:
                #Originated by the master
                if DEBUG:
                    print("INFORMING SLAVE")
                self._slave.setSelectionMask(mask, instance_id=id(self))

        if self._masterStack is not None:
            #This is a slave instance
            instanceList = [id(self.stackWidget),
                            id(self.roiWidget)]
            for key in self.pluginInstanceDict.keys():
                instanceList.append(id(self.pluginInstanceDict[key]))
            if instance_id in instanceList:
                #Originated by the slave
                if DEBUG:
                    print("INFORMING MASTER")
                self._masterStack.setSelectionMask(mask, instance_id=id(self))
        
        #Inform plugins
        for key in self.pluginInstanceDict.keys():
            if key == "PyMcaPlugins.StackPluginBase":
                continue
            #I remove this optimization for the case the plugin
            #does not update itself the mask
            #if id(self.pluginInstanceDict[key]) != instance_id:
            self.pluginInstanceDict[key].selectionMaskUpdated()

    def getSelectionMask(self):
        return self._selectionMask

    def _maskImageWidgetSlot(self, ddict):
        if ddict['event'] == "selectionMaskChanged":
            self.setSelectionMask(ddict['current'], instance_id=ddict['id'])
            return
        if ddict['event'] == "resetSelection":
            self.setSelectionMask(None, instance_id=ddict['id'])
            return
        if ddict['event'] == "addImageClicked":
            self._addImageClicked(ddict['image'], ddict['title'])
            return
        if ddict['event'] == "replaceImageClicked":
            self._replaceImageClicked(ddict['image'], ddict['title'])
            return
        if ddict['event'] == "removeImageClicked":
            self._removeImageClicked(ddict['title'])
            return
        if ddict['event'] == "hFlipSignal":
            if ddict['id'] != id(self.stackWidget):
                self.stackWidget.graphWidget.graph.zoomReset()
                self.stackWidget.setY1AxisInverted(ddict['current'])
                self.stackWidget.plotImage(update=True)
            if ddict['id'] != id(self.roiWidget):
                self.roiWidget.graphWidget.graph.zoomReset()
                self.roiWidget.setY1AxisInverted(ddict['current'])
                self.roiWidget.plotImage(update=True)
            return

    def _stackGraphSignal(self, ddict):
        if ddict['event'] == "MouseAt":
            x = round(ddict['y'])
            if x < 0: x = 0
            y = round(ddict['x'])
            if y < 0: y = 0
            if self._stackImageData is None:
                return
            limits = self._stackImageData.shape
            x = min(int(x), limits[0]-1)
            y = min(int(y), limits[1]-1)
            z = self._stackImageData[x, y]
            self.stackGraphWidget.setInfoText("    X = %d Y = %d Z = %.4g" %\
                                               (y, x, z))

    def _mcaWidgetSignal(self, ddict):
        if not self.__ROIConnected:
            return
        if ddict['event'] == "ROISignal":
            self.updateROIImages(ddict)

    def getActiveCurve(self):
        legend = self.mcaWidget.graph.getactivecurve(justlegend=1)
        if legend is None:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Please select an active curve")
            msg.exec_()
            return
        x, y, legend, info = self.mcaWidget.getActiveCurve()
        return x, y, legend, info

    def getGraphXLimits(self):
        return self.mcaWidget.graph.getX1AxisLimits()
        
    def getGraphYLimits(self):
        return self.mcaWidget.graph.getY1AxisLimits()
    
def test():
    #create a dummy stack
    nrows = 100
    ncols = 200
    nchannels = 1024
    a = numpy.ones((nrows, ncols), numpy.float)
    stackData = numpy.zeros((nrows, ncols, nchannels), numpy.float)
    for i in range(nchannels):
        stackData[:, :, i] = a * i
    stackData[0:10,:,:] = 0
    w = QStackWidget()
    w.setStack(stackData, mcaindex=2)
    w.show()
    return w

if __name__ == "__main__":
    import getopt
    options = ''
    longoptions = ["fileindex=","old",
                   "filepattern=", "begin=", "end=", "increment=",
                   "nativefiledialogs=", "imagestack="]
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except:
        print(sys.exc_info()[1])
        sys.exit(1)
    fileindex = 0
    filepattern=None
    begin = None
    end = None
    imagestack=False
    increment=None
    for opt, arg in opts:
        if opt in '--begin':
            if "," in arg:
                begin = map(int,arg.split(","))
            else:
                begin = [int(arg)]
        elif opt in '--end':
            if "," in arg:
                end = map(int,arg.split(","))
            else:
                end = int(arg)
        elif opt in '--increment':
            if "," in arg:
                increment = map(int,arg.split(","))
            else:
                increment = int(arg)
        elif opt in '--filepattern':
            filepattern = arg.replace('"','')
            filepattern = filepattern.replace("'","")
        elif opt in '--fileindex':
            fileindex = int(arg)
        elif opt in '--imagestack':
            imagestack = int(arg)
        elif opt in '--nativefiledialogs':
            if int(arg):
                PyMcaDirs.nativeFileDialogs=True
            else:
                PyMcaDirs.nativeFileDialogs=False
        elif opt in '--old':
            import QEDFStackWidget
            sys.exit(QEDFStackWidget.runAsMain())
    if filepattern is not None:
        if (begin is None) or (end is None):
            raise ValueError("A file pattern needs at least a set of begin and end indices")
    app = qt.QApplication([])
    widget = QStackWidget()
    w = StackSelector.StackSelector(widget)
    if filepattern is not None:
        #ignore the args even if present
        stack = w.getStackFromPattern(filepattern, begin, end, increment=increment,
                                      imagestack=imagestack)
    else:
        stack = w.getStack(args, imagestack=imagestack)
    if type(stack) == type([]):
        #aifira like, two stacks
        widget.setStack(stack[0])
        slave = QStackWidget(master=False, rgbwidget=widget.rgbWidget)
        slave.setStack(stack[1])
        widget.setSlave(slave)
        stack = None
    else:
        widget.setStack(stack)
    widget.show()
    app.exec_()
    
