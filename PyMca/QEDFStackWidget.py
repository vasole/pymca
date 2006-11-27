import sys
import McaWindow
qt = McaWindow.qt
QTVERSION = qt.qVersion()
if QTVERSION > '4.0.0':
    import RGBCorrelator
import RGBCorrelatorGraph
from Icons import IconDict
import DataObject
import EDFStack
import Numeric
import ColormapDialog
import spslut
COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]
QWTVERSION4 = RGBCorrelatorGraph.QtBlissGraph.QWTVERSION4
DEBUG = 0

class QEDFStackWidget(qt.QWidget):
    def __init__(self, parent = None,
                 mcawidget = None,
                 rgbwidget = None,
                 vertical = False):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setIcon(qt.QPixmap(IconDict['gioconda16']))
            self.setCaption("ROI Imaging Tool")
        else:
            self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
            self.setWindowTitle("ROI Imaging Tool")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(6)
        self.mainLayout.setSpacing(0)
        self._build(vertical)
        if mcawidget is None:
            self.mcaWidget = McaWindow.McaWidget()
            self.mcaWidget.show()
        else:
            self.mcaWidget = mcawidget
        if rgbwidget is None:
            if QTVERSION > '4.0.0':
                #I have not implemented it for Qt3
                #self.rgbWidget = RGBCorrelator.RGBCorrelator()
                self.rgbWidget = RGBCorrelator.RGBCorrelator(self)
                self.mainLayout.addWidget(self.rgbWidget)
        else:
            self.rgbWidget = rgbwidget
        self._y1AxisInverted = False
        self.__stackImageData = None
        self.__ROIImageData  = None
        self.__stackColormap = None
        self.__stackColormapDialog = None
        self.__ROIColormap       = None
        self.__ROIColormapDialog = None
        self._buildConnections()

    def _build(self, vertical = False):
        box = qt.QWidget(self)
        if vertical:
            boxLayout  = qt.QVBoxLayout(box)
        else:
            boxLayout  = qt.QHBoxLayout(box)
        boxLayout.setMargin(0)
        boxLayout.setSpacing(6)
        self.stackWindow = qt.QWidget(box)
        self.stackWindow.mainLayout = qt.QVBoxLayout(self.stackWindow)
        self.stackWindow.mainLayout.setMargin(0)
        self.stackWindow.mainLayout.setSpacing(0)
        self.stackGraphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self.stackWindow,
                                                            colormap=True)
        self.roiWindow = qt.QWidget(box)
        self.roiWindow.mainLayout = qt.QVBoxLayout(self.roiWindow)
        self.roiWindow.mainLayout.setMargin(0)
        self.roiWindow.mainLayout.setSpacing(0)
        self.roiGraphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self.roiWindow,
                                                                selection = True,
                                                              colormap=True)
        self.roiGraphWidget.graph.enableSelection(True)
        self.roiGraphWidget.graph.enableZoom(False)
        self.stackWindow.mainLayout.addWidget(self.stackGraphWidget)
        self.roiWindow.mainLayout.addWidget(self.roiGraphWidget)
        boxLayout.addWidget(self.stackWindow)
        boxLayout.addWidget(self.roiWindow)
        self.mainLayout.addWidget(box)
        
    def toggleSelectionMode(self):
        if self.roiGraphWidget.graph._selecting:
            self.setSelectionMode(False)
        else:
            self.setSelectionMode(True)


    def setSelectionMode(self, mode = None):
        if mode:
            self.roiGraphWidget.graph.enableSelection(True)
            self.roiGraphWidget.graph.enableZoom(False)
            self.roiGraphWidget.selectionToolButton.setDown(True)
        else:
            self.roiGraphWidget.graph.enableZoom(True)
            self.roiGraphWidget.selectionToolButton.setDown(False)
            self.plotStackImage(update = True)
            self.plotROIImage(update = True)
            
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
        
        # The IMAGE selection
        self.imageButtonBox = qt.QWidget(self.roiWindow)
        buttonBox = self.imageButtonBox
        self.imageButtonBoxLayout = qt.QHBoxLayout(buttonBox)
        self.imageButtonBoxLayout.setMargin(0)
        self.imageButtonBoxLayout.setSpacing(0)
        self.addImageButton = qt.QPushButton(buttonBox)
        self.addImageButton.setText("ADD")
        self.removeImageButton = qt.QPushButton(buttonBox)
        self.removeImageButton.setText("REMOVE")
        self.replaceImageButton = qt.QPushButton(buttonBox)
        self.replaceImageButton.setText("REPLACE")
        self.imageButtonBoxLayout.addWidget(self.addImageButton)
        self.imageButtonBoxLayout.addWidget(self.removeImageButton)
        self.imageButtonBoxLayout.addWidget(self.replaceImageButton)
        
        self.roiWindow.mainLayout.addWidget(buttonBox)
        
        self.connect(self.addImageButton, qt.SIGNAL("clicked()"), 
                    self._addImageClicked)
        self.connect(self.removeImageButton, qt.SIGNAL("clicked()"), 
                    self._removeImageClicked)
        self.connect(self.replaceImageButton, qt.SIGNAL("clicked()"), 
                    self._replaceImageClicked)

    def _buildConnections(self):
        if self.rgbWidget is not None:
            self._buildAndConnectButtonBox()
        self.connect(self.stackGraphWidget.colormapToolButton,
                     qt.SIGNAL("clicked()"),
                     self.selectStackColormap)
        self.connect(self.roiGraphWidget.selectionToolButton,
                     qt.SIGNAL("clicked()"),
                     self.toggleSelectionMode)
        self.connect(self.roiGraphWidget.colormapToolButton,
                     qt.SIGNAL("clicked()"),
                     self.selectROIColormap)

        self.connect(self.stackGraphWidget.hFlipToolButton,
                     qt.SIGNAL("clicked()"),
                     self._hFlipIconSignal)
        
        self.connect(self.roiGraphWidget.hFlipToolButton,
                     qt.SIGNAL("clicked()"),
                     self._hFlipIconSignal)


        if QTVERSION < "4.0.0":
            self.connect(self.roiGraphWidget.graph,
                         qt.PYSIGNAL("QtBlissGraphSignal"),
                         self._roiGraphSignal)
            self.connect(self.mcaWidget,
                         qt.PYSIGNAL("McaWindowSignal"),
                         self._mcaWidgetSignal)
        else:
            self.connect(self.roiGraphWidget.graph,
                         qt.SIGNAL("QtBlissGraphSignal"),
                         self._roiGraphSignal)
            self.connect(self.mcaWidget,
                         qt.SIGNAL("McaWindowSignal"),
                         self._mcaWidgetSignal)

    def _roiGraphSignal(self, ddict):
        if ddict['event'] == "MouseSelection":
            ix1 = int(ddict['xmin'])
            ix2 = int(ddict['xmax'])+1
            iy1 = int(ddict['xmin'])
            iy2 = int(ddict['xmax'])+1
            if self.mcaIndex == 0:
                selectedData = Numeric.sum(Numeric.sum(self.stack.data[:,ix1:ix2, iy1:iy2], 2),1)
            else:
                selectedData = Numeric.sum(Numeric.sum(self.stack.data[ix1:ix2,:, iy1:iy2], 2),0)
            dataObject = DataObject.DataObject()
            dataObject.info = {"McaCalib": self.stack.info['McaCalib'],
                               "selectiontype":"1D",
                               "SourceName":"EDF Stack",
                               "Key":"Selection"}
            dataObject.x = [Numeric.arange(len(selectedData)).astype(Numeric.Float)
                            + self.stack.info['Channel0']]
            dataObject.y = [selectedData]
            self.sendMcaSelection(dataObject,
                                  key = "Selection",
                                  legend="EDF Stack Selection",
                                  action = "ADD")

        elif ddict['event'] == "MouseAt":
            return
            #if follow mouse is not activated
            #it only enters here when the mouse is pressed.
            #Therefore is perfect for "brush" selections.
            print ddict
            self.__brushWidth = 1
            width = self.__brushWidth   #in (row, column) units
            r = self.__stackImageData.shape[0]
            c = self.__stackImageData.shape[1]
            i1 = max((int(ddict['x'])-width+1), 0)
            i2 = min((int(ddict['x'])+width), r)
            j1 = max(int(ddict['y'])-width+1, 0)
            j2 = min(int(ddict['y'])+width, c)
            #if self.__adding:
            if 1:
                self.__stackPixmap[j1:j2,i1:i2,:] = 0
                self.__selectionMask[i1:i2, j1:j2] = 1
            else:
                self.__stackPixmap[j1:j2,i1:i2,:] = self.__stackPixmap0[j1:j2,i1:i2,:]
                self.__selectionMask[i1:i2, j1:j2] = 0
            self.plotROIImage(update = False)
            self.plotStackImage(update = False)

    def setStack(self, stack):
        #stack.data is an XYZ array
        self.stack = stack
        
        #for the time being I deduce the MCA
        shape = self.stack.data.shape

        #guess the MCA
        imax = 0
        for i in range(len(shape)):
            if shape[i] > shape[imax]:
                imax = i
        self.mcaIndex = imax
                
        #original image
        self.__stackImageData = Numeric.sum(stack.data, self.mcaIndex)

        #original ICR mca
        if self.mcaIndex == 0:
            mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),1)
        else:
            mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),0)

        calib = self.stack.info['McaCalib']
        dataObject = DataObject.DataObject()
        dataObject.info = {"McaCalib": calib,
                           "selectiontype":"1D",
                           "SourceName":"EDF Stack",
                           "Key":"SUM"}
        dataObject.x = [Numeric.arange(len(mcaData0)).astype(Numeric.Float)
                        + self.stack.info['Channel0']]
        dataObject.y = [mcaData0]

        #store the original spectrum
        self.__mcaData0 = dataObject
        
        #add the original image
        self.__addOriginalImage()

        #add the ICR ROI Image
        #self.updateRoiImage(roidict=None)

        #add the mca
        self.sendMcaSelection(dataObject, action = "ADD")

    def __addOriginalImage(self):
        #init the original image
        self.stackGraphWidget.graph.setTitle("EDF Stack")
        self.stackGraphWidget.graph.y1Label("File")
        if self.mcaIndex == 0:
            self.stackGraphWidget.graph.x1Label('Column')
        else:
            self.stackGraphWidget.graph.x1Label('Row')

        [xmax, ymax] = self.__stackImageData.shape
        if self._y1AxisInverted:
            self.stackGraphWidget.graph.zoomReset()
            self.stackGraphWidget.graph.setY1AxisInverted(True)
            if 0:   #This is not needed because there are no curves in the graph
                self.stackGraphWidget.graph.setY1AxisLimits(0, ymax, replot=False)
                self.stackGraphWidget.graph.setX1AxisLimits(0, xmax, replot=False)
                self.stackGraphWidget.graph.replot() #I need it to update the canvas
            self.plotStackImage(update=True)
        else:
            self.stackGraphWidget.graph.zoomReset()
            self.stackGraphWidget.graph.setY1AxisInverted(False)
            if 0:#This is not needed because there are no curves in the graph
                self.stackGraphWidget.graph.setY1AxisLimits(0, ymax, replot=False)
                self.stackGraphWidget.graph.setX1AxisLimits(0, xmax, replot=False)
                self.stackGraphWidget.graph.replot() #I need it to update the canvas
            self.plotStackImage(update=True)

        self.__selectionMask = Numeric.zeros((xmax, ymax), Numeric.UInt8)

        #init the ROI
        self.roiGraphWidget.graph.setTitle("ICR ROI")
        self.roiGraphWidget.graph.y1Label(self.stackGraphWidget.graph.y1Label())
        self.roiGraphWidget.graph.x1Label(self.stackGraphWidget.graph.x1Label())
        self.roiGraphWidget.graph.setY1AxisInverted(self.stackGraphWidget.graph.isY1AxisInverted())
        if 0:#This is not needed because there are no curves in the graph
            self.roiGraphWidget.graph.setX1AxisLimits(0,
                                            self.__stackImageData.shape[0])
            self.roiGraphWidget.graph.setY1AxisLimits(0,
                                            self.__stackImageData.shape[1])
            self.roiGraphWidget.graph.replot()
        self.__ROIImageData = 1 * self.__stackImageData
        self.plotROIImage(update = True)

    def sendMcaSelection(self, mcaObject, key = None, legend = None, action = None):
        if action is None:action = "ADD"
        if key is None: key = "SUM"
        if legend is None: legend = "EDF Stack SUM"
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

    def _mcaWidgetSignal(self, ddict):
        if ddict['event'] == "ROISignal":
            self.roiGraphWidget.graph.setTitle("%s" % ddict["name"])
            if (ddict["name"] == "ICR"):                
                i1 = 0
                i2 = self.stack.data.shape[self.mcaIndex]
            elif (ddict["type"]).upper() != "CHANNEL":
                #energy roi
                xw =  ddict['calibration'][0] + \
                      ddict['calibration'][1] * self.__mcaData0.x[0] + \
                      ddict['calibration'][2] * self.__mcaData0.x[0] * \
                                                self.__mcaData0.x[0]
                i1 = Numeric.nonzero(ddict['from'] <= xw)
                if len(i1):
                    i1 = min(i1)
                else:
                    return
                i2 = Numeric.nonzero(xw <= ddict['to'])
                if len(i2):
                    i2 = max(i2) + 1
                else:
                    return
            else:
                i1 = max(int(ddict["from"]), 0)
                i2 = min(int(ddict["to"])+1, self.stack.data.shape[self.mcaIndex])
            if self.mcaIndex == 0:
                self.__ROIImageData = Numeric.sum(self.stack.data[i1:i2,0,:],0)
            else:
                self.__ROIImageData = Numeric.sum(self.stack.data[:,i1:i2,:],1)
            self.plotROIImage(update=True)

    def plotROIImage(self, update = True):
        if self.__ROIImageData is None:
            self.roiGraphWidget.graph.clear()
            return
        if update:
            self.getROIPixmapFromData()
            self.__ROIPixmap0 = 1 * self.__ROIPixmap
        if not self.roiGraphWidget.graph.yAutoScale:
            ylimits = self.roiGraphWidget.graph.getY1AxisLimits()
        if not self.roiGraphWidget.graph.xAutoScale:
            xlimits = self.roiGraphWidget.graph.getX1AxisLimits()
        self.roiGraphWidget.graph.pixmapPlot(self.__ROIPixmap.tostring(),
            (self.__ROIImageData.shape[0], self.__ROIImageData.shape[1]),
                                        xmirror = 0,
                                        ymirror = not self._y1AxisInverted)
        if not self.roiGraphWidget.graph.yAutoScale:
            self.roiGraphWidget.graph.setY1AxisLimits(ylimits[0], ylimits[1], replot=False)
        if not self.roiGraphWidget.graph.xAutoScale:
            self.roiGraphWidget.graph.setX1AxisLimits(xlimits[0], xlimits[1], replot=False)
        self.roiGraphWidget.graph.replot()

    def getROIPixmapFromData(self):
        #It does not look nice, but I avoid copying data
        colormap = self.__ROIColormap
        if colormap is None:
            (self.__ROIPixmap,size,minmax)= spslut.transform(\
                                Numeric.transpose(self.__ROIImageData),
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                spslut.TEMP,
                                1,
                                (0,1))
        else:
            (self.__ROIPixmap,size,minmax)= spslut.transform(\
                                Numeric.transpose(self.__ROIImageData),
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]))
        #I hope to find the time to write a new spslut giving back arrays ..
        self.__ROIPixmap = Numeric.array(self.__ROIPixmap).\
                                        astype(Numeric.UInt8)
        self.__ROIPixmap.shape = [self.__ROIImageData.shape[1],
                                    self.__ROIImageData.shape[0],
                                    4]

    def getStackPixmapFromData(self):
        colormap = self.__stackColormap
        if colormap is None:
            (self.__stackPixmap,size,minmax)= spslut.transform(\
                                Numeric.transpose(self.__stackImageData),
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                spslut.TEMP,
                                1,
                                (0,1))
        else:
            (self.__stackPixmap,size,minmax)= spslut.transform(\
                                Numeric.transpose(self.__stackImageData),
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]))
            
        #I hope to find the time to write a new spslut giving back arrays ..
        self.__stackPixmap = Numeric.array(self.__stackPixmap).\
                                        astype(Numeric.UInt8)
        self.__stackPixmap.shape = [self.__stackImageData.shape[1],
                                    self.__stackImageData.shape[0],
                                    4]

    def plotStackImage(self, update = True):
        if self.__stackImageData is None:
            self.stackGraphWidget.graph.clear()
            return
        if update:
            self.getStackPixmapFromData()
            self.__stackPixmap0 = 1 * self.__stackPixmap
        if not self.stackGraphWidget.graph.yAutoScale:
            ylimits = self.stackGraphWidget.graph.getY1AxisLimits()
        if not self.stackGraphWidget.graph.xAutoScale:
            xlimits = self.stackGraphWidget.graph.getX1AxisLimits()
        self.stackGraphWidget.graph.pixmapPlot(self.__stackPixmap.tostring(),
            (self.__stackImageData.shape[0], self.__stackImageData.shape[1]),
                    xmirror = 0,
                    ymirror = not self._y1AxisInverted)            
        if not self.stackGraphWidget.graph.yAutoScale:
            self.stackGraphWidget.graph.setY1AxisLimits(ylimits[0], ylimits[1], replot=False)
        if not self.stackGraphWidget.graph.xAutoScale:
            self.stackGraphWidget.graph.setX1AxisLimits(xlimits[0], xlimits[1], replot=False)        
        self.stackGraphWidget.graph.replot()

    def _hFlipIconSignal(self):
        if QWTVERSION4:
            qt.QMessageBox.information(self, "Flip Image", "Not available under PyQwt4")
            return
        if not self.stackGraphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set stack Y Axis to AutoScale first")
            return
        if not self.stackGraphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set stack X Axis to AutoScale first")
            return
        if not self.roiGraphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set ROI image Y Axis to AutoScale first")
            return
        if not self.roiGraphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set ROI image X Axis to AutoScale first")
            return

        if self._y1AxisInverted:
            self._y1AxisInverted = False
        else:
            self._y1AxisInverted = True
        self.stackGraphWidget.graph.zoomReset()
        self.roiGraphWidget.graph.zoomReset()
        self.stackGraphWidget.graph.setY1AxisInverted(self._y1AxisInverted)
        self.roiGraphWidget.graph.setY1AxisInverted(self._y1AxisInverted)
        self.plotStackImage(True)
        self.plotROIImage(True)

    def selectStackColormap(self):
        if self.__stackImageData is None:return
        if self.__stackColormapDialog is None:
            self.__initStackColormapDialog()
        if self.__stackColormapDialog.isHidden():
            self.__stackColormapDialog.show()
        if QTVERSION < '4.0.0':self.__stackColormapDialog.raiseW()
        else:  self.__stackColormapDialog.raise_()          
        self.__stackColormapDialog.show()


    def __initStackColormapDialog(self):
        a = Numeric.ravel(self.__stackImageData)
        minData = min(a)
        maxData = max(a)
        self.__stackColormapDialog = ColormapDialog.ColormapDialog()
        self.__stackColormapDialog.colormapIndex  = self.__stackColormapDialog.colormapList.index("Temperature")
        self.__stackColormapDialog.colormapString = "Temperature"
        if QTVERSION < '4.0.0':
            self.__stackColormapDialog.setCaption("Stack Colormap Dialog")
            self.connect(self.__stackColormapDialog,
                         qt.PYSIGNAL("ColormapChanged"),
                         self.updateStackColormap)
        else:
            self.__stackColormapDialog.setWindowTitle("Stack Colormap Dialog")
            self.connect(self.__stackColormapDialog,
                         qt.SIGNAL("ColormapChanged"),
                         self.updateStackColormap)
        self.__stackColormapDialog.setDataMinMax(minData, maxData)
        self.__stackColormapDialog.setAutoscale(1)
        self.__stackColormapDialog.setColormap(self.__stackColormapDialog.colormapIndex)
        self.__stackColormap = (self.__stackColormapDialog.colormapIndex,
                              self.__stackColormapDialog.autoscale,
                              self.__stackColormapDialog.minValue, 
                              self.__stackColormapDialog.maxValue,
                              minData, maxData)
        self.__stackColormapDialog._update()

    def updateStackColormap(self, *var):
        if len(var) > 5:
            self.__stackColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        else:
            self.__stackColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        self.plotStackImage(True)

    def selectROIColormap(self):
        if self.__ROIImageData is None:return
        if self.__ROIColormapDialog is None:
            self.__initROIColormapDialog()
        if self.__ROIColormapDialog.isHidden():
            self.__ROIColormapDialog.show()
        if QTVERSION < '4.0.0':self.__ROIColormapDialog.raiseW()
        else:  self.__ROIColormapDialog.raise_()          
        self.__ROIColormapDialog.show()


    def __initROIColormapDialog(self):
        a = Numeric.ravel(self.__ROIImageData)
        minData = min(a)
        maxData = max(a)
        self.__ROIColormapDialog = ColormapDialog.ColormapDialog()
        self.__ROIColormapDialog.colormapIndex  = self.__ROIColormapDialog.colormapList.index("Temperature")
        self.__ROIColormapDialog.colormapString = "Temperature"
        if QTVERSION < '4.0.0':
            self.__ROIColormapDialog.setCaption("ROI Colormap Dialog")
            self.connect(self.__ROIColormapDialog,
                         qt.PYSIGNAL("ColormapChanged"),
                         self.updateROIColormap)
        else:
            self.__ROIColormapDialog.setWindowTitle("ROI Colormap Dialog")
            self.connect(self.__ROIColormapDialog,
                         qt.SIGNAL("ColormapChanged"),
                         self.updateROIColormap)
        self.__ROIColormapDialog.setDataMinMax(minData, maxData)
        self.__ROIColormapDialog.setAutoscale(1)
        self.__ROIColormapDialog.setColormap(self.__ROIColormapDialog.colormapIndex)
        self.__ROIColormap = (self.__ROIColormapDialog.colormapIndex,
                              self.__ROIColormapDialog.autoscale,
                              self.__ROIColormapDialog.minValue, 
                              self.__ROIColormapDialog.maxValue,
                              minData, maxData)
        self.__ROIColormapDialog._update()

    def updateROIColormap(self, *var):
        if len(var) > 5:
            self.__ROIColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        else:
            self.__ROIColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        self.plotROIImage(True)

    def _addImageClicked(self):
        self.rgbWidget.addImage(self.__ROIImageData,
                                str(self.roiGraphWidget.graph.title().text()))

        if self.rgbWidget.isHidden():
            self.rgbWidget.show()
        self.rgbWidget.raise_()

    def _removeImageClicked(self):
        self.rgbWidget.removeImage(str(self.roiGraphWidget.graph.title().text()))

    def _replaceImageClicked(self):
        self.rgbWidget.reset()
        self.rgbWidget.addImage(self.__ROIImageData,
                                str(self.roiGraphWidget.graph.title().text()))
        if self.rgbWidget.isHidden():
            self.rgbWidget.show()
        self.rgbWidget.raise_()

    def _addMcaClicked(self):
        #original ICR mca
        """
        if self.mcaIndex == 0:
            mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),1)
        else:
            mcaData0 = Numeric.sum(Numeric.sum(stack.data, 2),0)
        """
        dataObject = self.__mcaData0

        """
        calib = self.stack.info['McaCalib']
        dataObject = DataObject.DataObject()
        dataObject.info = {"McaCalib": calib,
                           "selectiontype":"1D",
                           "SourceName":"EDF Stack",
                           "Key":"SUM"}
        dataObject.x = [Numeric.arange(len(mcaData0)).astype(Numeric.Float)
                        + self.stack.info['Channel0']]
        dataObject.y = [mcaData0]
        """
        #add the mca
        self.sendMcaSelection(dataObject, action = "ADD")
    
    def _removeMcaClicked(self):
        #remove the mca
        dataObject = self.__mcaData0
        self.sendMcaSelection(dataObject, action = "REMOVE")
    
    def _replaceMcaClicked(self):
        #replace the mca
        dataObject = self.__mcaData0
        self.sendMcaSelection(dataObject, action = "REPLACE")
        
    def closeEvent(self, event):
        ddict = {}
        ddict['event'] = "StackWidgetClosed"
        ddict['id']    = id(self)
        self.emit(qt.SIGNAL("StackWidgetSignal"),ddict)

if __name__ == "__main__":
    import getopt, os
    options = ''
    longoptions = []
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except getopt.error,msg:
        print msg
        sys.exit(1)
    #import time
    #t0= time.time()
    stack = EDFStack.EDFStack()
    if len(args):
        stack.setFileList(args)
    else:
        if os.path.exists(".\COTTE\ch09\ch09__mca_0005_0000_0070.edf"):
            stack.loadIndexedStack(".\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
        elif os.path.exists("Z:\COTTE\ch09\ch09__mca_0005_0000_0070.edf"):
            stack.loadIndexedStack("Z:\COTTE\ch09\ch09__mca_0005_0000_0070.edf")
        else:
            print "Usage: "
            print "python QEDFStackWidget.py SET_OF_EDF_FILES"
            sys.exit(1)
    shape = stack.data.shape
    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))

    w = QEDFStackWidget()
    w.show()
    #print "reading elapsed = ", time.time() - t0
    w.setStack(stack)
    if qt.qVersion() < '4.0.0':
        app.exec_loop()
    else:
        sys.exit(app.exec_())
