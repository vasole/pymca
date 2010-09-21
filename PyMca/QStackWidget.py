#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
import numpy
from PyMca import PyMcaQt as qt
from PyMca import McaWindow
from PyMca import StackBase
from PyMca import CloseEventNotifyingWidget
from PyMca import MaskImageWidget
from PyMca import StackROIWindow
from PyMca import RGBCorrelator
from PyMca.PyMca_Icons import IconDict

DEBUG = 1
QTVERSION = qt.qVersion()

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
                                                         standalonesave=standaloneSaving)
        self.roiGraphWidget = self.roiWidget.graphWidget        
        self.stackWindow.mainLayout.addWidget(self.stackWidget)
        self.roiWindow.mainLayout.addWidget(self.roiWidget)
        box.addWidget(self.stackWindow)
        box.addWidget(self.roiWindow)
        self.mainLayout.addWidget(box)


        #add some missing icons


        self.pluginIcon     = qt.QIcon(qt.QPixmap(IconDict["plugin"]))
        infotext = "Call/Load Stack Plugins"
        self.stackGraphWidget._addToolButton(self.pluginIcon,
                                             self._pluginClicked,
                                             infotext,
                                             toggle = False,
                                             state = False,
                                             position = 6)


        #self.mainLayout.addWidget(self.roiWidget, 1, 0)
        #self.mainLayout.addWidget(self.mcaWidget, 0, 1, 2, 2)

    def _pluginClicked(self):
        actionList = []
        menu = qt.QMenu(self)
        text = qt.QString("Reload")
        menu.addAction(text)
        actionList.append(text)
        menu.addSeparator()
        callableKeys = ["Dummy"]
        for m in self.pluginList:
            if m == "PyMcaPlugins.StackPluginBase":
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
        key = callableKeys[idx]
        methods = self.pluginInstanceDict[key].getMethods()
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
            msg.setText("%s" % sys.exc_info()[1])
            msg.exec_()

    def _actionHovered(self, action):
        tip = action.toolTip()
        if str(tip) != str(action.text()):
            qt.QToolTip.showText(qt.QCursor.pos(), tip)
        

    def _buildBottom(self):
        n = 0
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
        tmp = self._selectionMask
        self._selectionMask = None
        self._addMcaClicked(action="ADD")
        self._selectionMask = tmp

    def showROIImageList(self, imageList, image_names=None):
        self.roiWidget.setImageData(imageList[0])
        self.roiWidget.graphWidget.graph.setTitle(image_names[0])

        #self.roiWidget.setImageList(imageList, image_names)

    def addImage(self, image, name, info=None, replace=False, replot=True):
        self.rgbWidget.addImage(image, name)
        if self.tab is None:
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
        title = str(self.roiGraphWidget.graph.title().text())
        return "Stack " + title + " selection"

    def _addMcaClicked(self, action=None):
        if action is None:
            action = "ADD"
        if self._stackImageData is None:
            return            
        dataObject = self.calculateMcaDataObject(normalize=False)
        legend = self.__getLegend()
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
            legend = "EDF Stack SUM"
            print "TO IMPLEMENT"
            #if self.normalizeButton.isChecked():
            #    npixels = self.__stackImageData.shape[0] * self.__stackImageData.shape[1]
            #    legend += "/%d" % npixels
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
        #inform built in widgets
        for widget in [self.stackWidget, self.roiWidget]:
            if instance_id != id(widget):
                if mask is None:
                    widget._resetSelection(owncall=False)
                else:
                    widget.setSelectionMask(mask, plot=True)
        
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

    
def test():
    #create a dummy stack
    nrows = 100
    ncols = 200
    nchannels = 1024
    a = numpy.ones((nrows, ncols), numpy.float)
    stackData = numpy.zeros((nrows, ncols, nchannels), numpy.float)
    for i in xrange(nchannels):
        stackData[:, :, i] = a * i
    stackData[0:10,:,:] = 0
    w = QStackWidget()
    w.setStack(stackData, mcaindex=2)
    w.show()
    return w

if __name__ == "__main__":
    app = qt.QApplication([])
    w = test()
    app.exec_()
    
