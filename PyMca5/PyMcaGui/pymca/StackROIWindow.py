#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Software Group"
import numpy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5 import ExternalImagesWindow
MaskImageWidget = ExternalImagesWindow.MaskImageWidget
try:
    from PyMca5.PyMcaSciPy.signal import median
except:
    print("StackEOIWindow importing PyMcaSciPy.signal directly")
    from PyMcaSciPy.signal import median
medfilt2d = median.medfilt2d
from PyMca5.PyMca_Icons import IconDict

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
        self.connect(self._medianParametersWidget.widthSpin,
                     qt.SIGNAL('valueChanged(int)'),
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
            self.setImageData(self.imageList[index])
            if self.imageNames is None:
                self.graphWidget.graph.setGraphTitle(mfText+"Image %d" % index)
            else:
                self.graphWidget.graph.setGraphTitle(mfText+self.imageNames[index])
        else:
            # TODO: Should the channel at max. and channel at min. be
            # recalculated?
            self.setImageData(self.imageList[index]-\
                              self.imageList[backgroundIndex])
            if self.imageNames is None:
                self.graphWidget.graph.setGraphTitle(mfText+"Image %d Net" % index)
            else:
                self.graphWidget.graph.setGraphTitle(mfText+self.imageNames[index]+ " Net")
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
