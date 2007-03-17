#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import RGBCorrelatorGraph
import ColormapDialog
import Numeric
import spslut
qt = RGBCorrelatorGraph.qt
IconDict = RGBCorrelatorGraph.IconDict
QTVERSION   = qt.qVersion()
QWTVERSION4 = RGBCorrelatorGraph.QWTVERSION4
COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]

DEBUG = 0

class RGBImageCalculator(qt.QWidget):
    def __init__(self, parent = None, math = True, replace = False):
        qt.QWidget.__init__(self, parent)
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.setWindowTitle("PyMCA - RGB Image Calculator")

        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.imageList   = None
        self.imageDict   = None
        self._imageData = None
        self.__imagePixmap   = None
        self.__imageColormap = None
        self.__imageColormapDialog = None
        self._y1AxisInverted = False
        self._build(math = math, replace = replace)

    def _buildMath(self):
        self.mathBox = qt.QWidget(self)
        self.mathBox.mainLayout = qt.QHBoxLayout(self.mathBox)

        self.mathLabel = qt.QLabel(self.mathBox)
        self.mathLabel.setText("New image = ")

        self.mathExpression = qt.QLineEdit(self.mathBox)
        text  = ""
        text += "Enter a mathematical expression in which your images\n"
        text += "have to be identified by a number between curly brackets.\n"
        text += "In order to normalize image 1 by image 2:  {1}/{2}\n"
        text += "The numbers go from 1 to the number of rows in the table.\n"
        text += "If you can suggest useful correlation functions please,\n"
        text += "do not hesitate to contact us in order to implement them."
        
        self.mathExpression.setToolTip(text)

        self.mathAction = qt.QToolButton(self.mathBox) 
        self.mathAction.setText("CALCULATE")

        self.mathBox.mainLayout.addWidget(self.mathLabel)
        self.mathBox.mainLayout.addWidget(self.mathExpression)
        self.mathBox.mainLayout.addWidget(self.mathAction)
        self.mainLayout.addWidget(self.mathBox)
        self.connect(self.mathAction, qt.SIGNAL("clicked()"), 
                    self._calculateClicked)

    def _build(self, math = True, replace = False):
        if math: self._buildMath()
        
        self.graphWidget = RGBCorrelatorGraph.RGBCorrelatorGraph(self,
                                                        colormap=True)

        self.nameBox = qt.QWidget(self)
        self.nameBox.mainLayout = qt.QHBoxLayout(self.nameBox)

        self.nameLabel = qt.QLabel(self.nameBox)
        
        self.nameLabel.setText("Image Name = ")
        #self.nameLabel.setText(qt.QString(qt.QChar(0x3A3)))

        self.name = qt.QLineEdit(self.nameBox)
        self.nameBox.mainLayout.addWidget(self.nameLabel)
        self.nameBox.mainLayout.addWidget(self.name)

        # The IMAGE selection
        #self.imageButtonBox = qt.QWidget(self)
        self.imageButtonBox = self.nameBox
        buttonBox = self.imageButtonBox
        #self.imageButtonBoxLayout = qt.QHBoxLayout(buttonBox)
        self.imageButtonBoxLayout = self.nameBox.mainLayout
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
        if replace:
            self.replaceImageButton = qt.QPushButton(buttonBox)
            self.replaceImageButton.setIcon(icon)
            self.replaceImageButton.setText("REPLACE IMAGE")
            self.imageButtonBoxLayout.addWidget(self.replaceImageButton)
        
        #self.mainLayout.addWidget(self.nameBox)
        self.mainLayout.addWidget(self.graphWidget)
        self.mainLayout.addWidget(buttonBox)
        self.connect(self.addImageButton, qt.SIGNAL("clicked()"), 
                    self._addImageClicked)
        self.connect(self.removeImageButton, qt.SIGNAL("clicked()"), 
                    self._removeImageClicked)
        if replace:
            self.connect(self.replaceImageButton, qt.SIGNAL("clicked()"), 
                         self._replaceImageClicked)
        self.connect(self.graphWidget.colormapToolButton,
             qt.SIGNAL("clicked()"),
             self.selectColormap)

        self.connect(self.graphWidget.hFlipToolButton,
                 qt.SIGNAL("clicked()"),
                 self._hFlipIconSignal)

        self.graphWidget.graph.canvas().setMouseTracking(1)
        self.graphWidget.showInfo()
        self.connect(self.graphWidget.graph,
                     qt.SIGNAL("QtBlissGraphSignal"),
                     self._graphSignal)



    def plotImage(self, update = True):
        if DEBUG:print"plotImage", update
        if self._imageData is None:
            self.graphWidget.graph.clear()
        if update:
            self.getPixmapFromData()
        if not self.graphWidget.graph.yAutoScale:
            ylimits = self.graphWidget.graph.getY1AxisLimits()
        if not self.graphWidget.graph.xAutoScale:
            xlimits = self.graphWidget.graph.getX1AxisLimits()
        if 1:  
            self.graphWidget.graph.pixmapPlot(self.__imagePixmap,
                    (self._imageData.shape[1], self._imageData.shape[0]),
                                        xmirror = 0,
                                        ymirror = not self._y1AxisInverted)
        else:            
            self.graphWidget.graph.imagePlot(self._imageData,
                                        colormap = self.__imageColormap,
                                        xmirror = 0,
                                        ymirror = not self._y1AxisInverted)
        if not self.graphWidget.graph.yAutoScale:
            self.graphWidget.graph.setY1AxisLimits(ylimits[0], ylimits[1], replot=False)
        if not self.graphWidget.graph.xAutoScale:
            self.graphWidget.graph.setX1AxisLimits(xlimits[0], xlimits[1], replot=False)
        self.graphWidget.graph.replot()

    def getPixmapFromData(self):
        colormap = self.__imageColormap
        if colormap is None:
            (self.__imagePixmap,size,minmax)= spslut.transform(\
                                self._imageData,
                                (1,0),
                                (spslut.LINEAR,3.0),
                                "BGRX",
                                spslut.TEMP,
                                1,
                                (0,1))
        else:
            if len(colormap) < 7: colormap.append(spslut.LINEAR)
            (self.__imagePixmap,size,minmax)= spslut.transform(\
                                self._imageData,
                                (1,0),
                                (colormap[6],3.0),
                                "BGRX",
                                COLORMAPLIST[int(str(colormap[0]))],
                                colormap[1],
                                (colormap[2],colormap[3]))
        
    def _calculateClicked(self):
        if DEBUG: print "Calculate clicked"
        text = str(self.mathExpression.text())
        if not len(text):
            qt.QMessageBox.critical(self, "Calculation Error",
                                    "Empty expression")
            return 1

        expression = text * 1
        name       = text * 1
        i = 1
        for label in self.imageList:
            item = "{%d}" % i
            expression = expression.replace(item,
                            "self.imageDict['%s']['image']" % label)
            name = name.replace(item,label)
            i = i + 1
        try:
            self._imageData = 1 * eval(expression)
        except:
            error = sys.exc_info()
            text = "Failed to evaluate expression:\n"
            text += "%s\n" % expression
            text += "%s" % error[1]
            qt.QMessageBox.critical(self,"%s" % error[0], text)
            return 1
        self.plotImage()
        self.name.setText("(%s)" % name)
            
    def _addImageClicked(self):
        if DEBUG: print "Add image clicked"
        if self._imageData is None:return
        if self._imageData == []:return
        text = str(self.name.text())
        if not len(text):
            qt.QMessageBox.critical(self, "Name Error",
                                    "Please give a name to the image")
            return 1
        ddict = {}
        ddict['label'] = text
        ddict['image']  = self._imageData
        self.emit(qt.SIGNAL("addImageClicked"),
                  ddict)

    def _removeImageClicked(self):
        if DEBUG: print "remove image clicked"
        text = str(self.name.text())
        if not len(text):
            qt.QMessageBox.critical(self, "Name Error",
                                    "Please enter the image name")
            return 1
        self.emit(qt.SIGNAL("removeImageClicked"),
                  text)

    def _replaceImageClicked(self):
        if DEBUG: print "remove image clicked"
        text = str(self.name.text())
        if not len(text):
            qt.QMessageBox.critical(self, "Name Error",
                                    "Please enter the image name")
            return 1
        ddict = {}
        ddict['label'] = text
        ddict['image']  = self._imageData
        self.emit(qt.SIGNAL("replaceImageClicked"),
                  ddict)

    def _hFlipIconSignal(self):
        if QWTVERSION4:
            qt.QMessageBox.information(self,
                                       "Flip Image",
                                       "Not available under PyQwt4")
            return
        if not self.graphWidget.graph.yAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set image Y Axis to AutoScale first")
            return 1
        if not self.graphWidget.graph.xAutoScale:
            qt.QMessageBox.information(self, "Open",
                    "Please set image X Axis to AutoScale first")
            return 1

        if self._y1AxisInverted:
            self._y1AxisInverted = False
        else:
            self._y1AxisInverted = True
        self.graphWidget.graph.zoomReset()
        self.graphWidget.graph.zoomReset()
        self.graphWidget.graph.setY1AxisInverted(self._y1AxisInverted)
        self.plotImage(True)

    def selectColormap(self):
        if self._imageData is None:return
        if self.__imageColormapDialog is None:
            self.__initColormapDialog()
        if self.__imageColormapDialog.isHidden():
            self.__imageColormapDialog.show()
        self.__imageColormapDialog.raise_()          
        self.__imageColormapDialog.show()

    def __initColormapDialog(self):
        a = Numeric.ravel(self._imageData)
        minData = min(a)
        maxData = max(a)
        self.__imageColormapDialog = ColormapDialog.ColormapDialog(slider=True)
        self.__imageColormapDialog.colormapIndex  = self.__imageColormapDialog.colormapList.index("Temperature")
        self.__imageColormapDialog.colormapString = "Temperature"
        self.__imageColormapDialog.setWindowTitle("Stack Colormap Dialog")
        self.connect(self.__imageColormapDialog,
                     qt.SIGNAL("ColormapChanged"),
                     self.updateImageColormap)
        self.__imageColormapDialog.setDataMinMax(minData, maxData)
        self.__imageColormapDialog.setAutoscale(1)
        self.__imageColormapDialog.setColormap(self.__imageColormapDialog.colormapIndex)
        self.__imageColormap = (self.__imageColormapDialog.colormapIndex,
                              self.__imageColormapDialog.autoscale,
                              self.__imageColormapDialog.minValue, 
                              self.__imageColormapDialog.maxValue,
                              minData, maxData)
        self.__imageColormapDialog._update()

    def updateImageColormap(self, *var):
        if DEBUG:print "updateImageColormap",var 
        if len(var) > 6:
            self.__imageColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5],
                             var[6]]
        elif len(var) > 5:
            self.__imageColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        else:
            self.__imageColormap = [var[0],
                             var[1],
                             var[2],
                             var[3],
                             var[4],
                             var[5]]
        self.plotImage(True)

    def _graphSignal(self, ddict):
        if ddict['event'] == "MouseAt":
            if self._imageData is None:
                self.graphWidget.setInfoText("    X = ???? Y = ???? Z =????")
                return
            x = round(ddict['y'])
            if x < 0: x = 0
            y = round(ddict['x'])
            if y < 0: y = 0
            limits = self._imageData.shape
            x = min(int(x), limits[0]-1)
            y = min(int(y), limits[1]-1)
            z = self._imageData[x, y]
            self.graphWidget.setInfoText("    X = %d Y = %d Z = %.7g" %\
                                               (y, x, z))
    def closeEvent(self, event):
        if self.__imageColormapDialog is not None:
            self.__imageColormapDialog.close()
        qt.QWidget.closeEvent(self, event)

def test():
    app = qt.QApplication(sys.argv)
    w = RGBImageCalculator()

    array1 = Numeric.arange(10000)
    array2 = Numeric.transpose(array1)
    array3 = array1 * 1
    array1.shape = [100,100]
    array2.shape = [100,100]
    array3.shape = [100,100]
    imageList = ["array1", "array2","array3"]
    imageDict = {"array1":{'image':array1},
                 "array2":{'image':array2},
                 "array3":{'image':array3}}
    w.imageList = imageList
    w.imageDict = imageDict 
    w.show()
    app.exec_()

    


if __name__ == "__main__":
    test()
