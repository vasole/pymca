#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import numpy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import ExternalImagesWindow
MaskImageWidget = ExternalImagesWindow.MaskImageWidget
from PyMca5.PyMcaMath.PyMcaSciPy.signal import median
medfilt2d = median.medfilt2d
from PyMca5.PyMcaGui import IconDict

class MedianParameters(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.label = qt.QLabel(self)
        self.label.setText("Median filter width: ")
        self.widthSpin = qt.QSpinBox(self)
        self.widthSpin.setMinimum(1)
        self.widthSpin.setMaximum(99)
        self.widthSpin.setValue(1)
        self.widthSpin.setSingleStep(2)
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.widthSpin)

class StackROIWindow(ExternalImagesWindow.ExternalImagesWindow):
    def __init__(self, *var, **kw):
        original = kw['standalonesave']
        kw['standalonesave'] = False
        ExternalImagesWindow.ExternalImagesWindow.__init__(self, *var, **kw)
        standalonesave = original
        if standalonesave:
            MaskImageWidget.MaskImageWidget.buildStandaloneSaveMenu(self)
        self.backgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))
        infotext  = 'Toggle background image subtraction from current image\n'
        infotext += 'No action if no background image available.'
        self.backgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))
        self.backgroundButton = self.graphWidget._addToolButton(\
                                    self.backgroundIcon,
                                    self.subtractBackground,
                                    infotext,
                                    toggle = True,
                                    state = False,
                                    position = 6)
        self.buildAndConnectImageButtonBox()
        self._toggleSelectionMode()
        self._medianParameters = {'use':True,
                                  'row_width':1,
                                  'column_width':1}
        self._medianParametersWidget = MedianParameters(self)
        self._medianParametersWidget.widthSpin.setValue(1)
        self.layout().addWidget(self._medianParametersWidget)
        self._medianParametersWidget.widthSpin.valueChanged[int].connect( \
                     self.setKernelWidth)

    def setKernelWidth(self, value):
        kernelSize = numpy.asarray(value)
        if len(kernelSize.shape) == 0:
            kernelSize = [kernelSize.item()] * 2
        self._medianParameters['row_width'] = kernelSize[0]
        self._medianParameters['column_width'] = kernelSize[1]
        self._medianParametersWidget.widthSpin.setValue(int(kernelSize[0]))
        current = self.slider.value()
        self.showImage(current, moveslider=False)

    def subtractBackground(self):
        current = self.slider.value()
        self.showImage(current, moveslider=False)

    def showImage(self, index=0, moveslider=True):
        if self.imageList is None:
            return
        if len(self.imageList) == 0:
            return
        backgroundIndex = None
        if self.backgroundButton.isChecked():
            if self.imageNames is not None:
                i = -1
                for imageName in self.imageNames:
                    i += 1
                    if imageName.lower().endswith('background'):
                        backgroundIndex = i
                        break
        mfText = self._medianTitle()
        if backgroundIndex is None:
            if self.imageNames is None:
                self.graphWidget.graph.setGraphTitle(mfText+"Image %d" % index)
            else:
                self.graphWidget.graph.setGraphTitle(mfText+self.imageNames[index])
            self.setImageData(self.imageList[index])
        else:
            # TODO: Should the channel at max. and channel at min. be
            # recalculated?
            if self.imageNames is None:
                self.graphWidget.graph.setGraphTitle(mfText+"Image %d Net" % index)
            else:
                self.graphWidget.graph.setGraphTitle(mfText+self.imageNames[index]+ " Net")
            self.setImageData(self.imageList[index]-\
                              self.imageList[backgroundIndex])
        if moveslider:
            self.slider.setValue(index)

    def _medianTitle(self):
        a = self._medianParameters['row_width']
        b = self._medianParameters['column_width']
        if max(a, b) > 1:
            return "MF(%d,%d) " % (a, b)
        else:
            return ""

    def setImageData(self, data, **kw):
        if self._medianParameters['use']:
            if max(self._medianParameters['row_width'],
                   self._medianParameters['column_width']) > 1:
                data = medfilt2d(data,[self._medianParameters['row_width'],
                                   self._medianParameters['column_width']])
        ExternalImagesWindow.ExternalImagesWindow.setImageData(self, data, **kw)


    def saveImageList(self, filename=None, imagelist=None, labels=None):
        if imagelist is None:
            #only the seen image
            return MaskImageWidget.MaskImageWidget.saveImageList(self,
                                            filename=filename)
        return ExternalImagesWindow.ExternalImagesWindow.saveImageList(\
            filename=filename, imagelist=imagelist, labels=labels)
