#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
import sys
import numpy
from PyMca import MaskImageWidget
from PyMca import ColormapDialog
from PyMca import spslut
try:
    from PyMca import QPyMcaMatplotlibSave
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

qt = MaskImageWidget.qt
IconDict = MaskImageWidget.IconDict
QTVERSION   = qt.qVersion()
COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]


DEBUG = 0

class RGBImageCalculator(qt.QWidget):
    def __init__(self, parent = None, math = True, replace = False, scanwindow=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.setWindowTitle("PyMCA - RGB Image Calculator")

        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.imageList   = None
        self.imageDict   = None
        self._imageData = None
        self._xScale = None
        self._yScale = None
        self.__imagePixmap   = None
        self.__imageColormap = None
        self.__imageColormapDialog = None
        self.setDefaultColormap(2, logflag=False)
        self._y1AxisInverted = False
        self._matplotlibSaveImage = None
        self._build(math = math, replace = replace, scanwindow=scanwindow)
        

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

    def _build(self, math = True, replace = False, scanwindow=False):
        if math:
            self._buildMath()

        self.graphWidget = MaskImageWidget.MaskImageWidget(self,
                                                           colormap=True,
                                                           standalonesave=True,
                                                           imageicons=False,
                                                           profileselection=True,
                                                           selection=False,
                                                           scanwindow=scanwindow)
        
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
            
        #it consumes too much CPU, therefore only on click
        #self.graphWidget.graph.canvas().setMouseTracking(1)
        self.graphWidget.graphWidget.showInfo()
        self.connect(self.graphWidget.graphWidget.graph,
                     qt.SIGNAL("QtBlissGraphSignal"),
                     self._graphSignal)

    def plotImage(self, update=True):
        self.graphWidget.setImageData(self._imageData,
                                      xScale=self._xScale,
                                      yScale=self._yScale)
        return self.graphWidget.plotImage(update=update)

    def _calculateClicked(self):
        if DEBUG:
            print("Calculate clicked")
        text = "%s" % self.mathExpression.text()
        if not len(text):
            qt.QMessageBox.critical(self, "Calculation Error",
                                    "Empty expression")
            return 1

        expression = text * 1
        name       = text * 1
        i = 1
        for label in self.imageList:
            item = "{%d}" % i
            replacingChain ="self.imageDict[self.imageList[%d]]['image']" % (i-1)
            expression = expression.replace(item, replacingChain)
            if sys.version < '3.0':
                try:
                    tmpLabel = label.decode()
                except UnicodeDecodeError:
                    try:
                        tmpLabel = label.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            tmpLabel = label.decode('latin-1')
                        except UnicodeDecodeError:
                            tmpLabel = "image_%0d" % i
            else:
                tmpLabel = label
            name = name.replace(item, tmpLabel)
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
        self.setName("(%s)" % name)

    def setName(self, name):
        self.name.setText(name)
        self.graphWidget.graph.setTitle("%s" % name)
                    
    def _addImageClicked(self):
        if DEBUG:
            print("Add image clicked")
        if self._imageData is None:
            return
        if self._imageData == []:
            return
        text = "%s" % self.name.text()
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
        if DEBUG:
            print("remove image clicked")
        text = "%s" % self.name.text()
        if not len(text):
            qt.QMessageBox.critical(self, "Name Error",
                                    "Please enter the image name")
            return 1
        self.emit(qt.SIGNAL("removeImageClicked"),
                  text)

    def _replaceImageClicked(self):
        if DEBUG:
            print("replace image clicked")
        text = "%s" % self.name.text()
        if not len(text):
            qt.QMessageBox.critical(self, "Name Error",
                                    "Please enter the image name")
            return 1
        ddict = {}
        ddict['label'] = text
        ddict['image']  = self._imageData
        self.emit(qt.SIGNAL("replaceImageClicked"),
                  ddict)


    def setDefaultColormap(self, colormapindex, logflag=False):
        self.__defaultColormap = COLORMAPLIST[min(colormapindex, len(COLORMAPLIST)-1)]
        if logflag:
            self.__defaultColormapType = spslut.LOG
        else:
            self.__defaultColormapType = spslut.LINEAR

    def _graphSignal(self, ddict):
        if ddict['event'] == "MouseAt":
            if self._imageData is None:
                self.graphWidget.setInfoText("    X = ???? Y = ???? Z =????")
                return
            x = int(ddict['y'])
            if x < 0: x = 0
            y = int(ddict['x'])
            if y < 0: y = 0
            limits = self._imageData.shape
            x = min(int(x), limits[0]-1)
            y = min(int(y), limits[1]-1)
            z = self._imageData[x, y]
            self.graphWidget.graphWidget.setInfoText("    X = %d Y = %d Z = %.7g" %\
                                               (y, x, z))
    def closeEvent(self, event):
        if self.__imageColormapDialog is not None:
            self.__imageColormapDialog.close()
        if self._matplotlibSaveImage is not None:
            self._matplotlibSaveImage.close()
        qt.QWidget.closeEvent(self, event)

def test():
    app = qt.QApplication(sys.argv)
    w = RGBImageCalculator()

    array1 = numpy.arange(10000)
    array2 = numpy.transpose(array1)
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
