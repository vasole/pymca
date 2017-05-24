# /*#########################################################################
# Copyright (C) 2004-2017 V.A. Sole, European Synchrotron Radiation Facility
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
# ###########################################################################*/
__authors__ = ["P. Knobel"]
__license__ = "MIT"


from PyMca5.PyMcaMath.PyMcaSciPy.signal.median import medfilt2d

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import IconDict

from PyMca5.PyMcaGui.plotting import SilxMaskImageWidget

from silx.gui.plot.Profile import ProfileToolBar

import numpy
import copy


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


class SilxStackRoiWindow(SilxMaskImageWidget.SilxMaskImageWidget):
    """

    """
    def __init__(self, parent=None):
        SilxMaskImageWidget.SilxMaskImageWidget.__init__(self, parent=parent)

        self.profile = ProfileToolBar(plot=self.plot)
        self.addToolBar(self.profile)

        self.backgroundIcon = qt.QIcon(qt.QPixmap(IconDict["subtract"]))
        self.backgroundButton = qt.QToolButton(self)
        self.backgroundButton.setCheckable(True)

        self.backgroundButton.setIcon(self.backgroundIcon)
        self.backgroundButton.setToolTip(
            'Toggle background image subtraction from current image\n' +
            'No action if no background image available.')
        self.backgroundButton.clicked.connect(self.subtractBackground)

        toolbar = qt.QToolBar("Background", parent=None)
        toolbar.addWidget(self.backgroundButton)
        self.addToolBar(toolbar)

        self._medianParameters = {'row_width': 1,
                                  'column_width': 1}
        self._medianParametersWidget = MedianParameters(self)
        self._medianParametersWidget.widthSpin.setValue(1)
        self.centralWidget().layout().addWidget(self._medianParametersWidget)
        self._medianParametersWidget.widthSpin.valueChanged[int].connect(
                     self.setKernelWidth)

    def setKernelWidth(self, value):
        kernelSize = numpy.asarray(value)
        if len(kernelSize.shape) == 0:
            kernelSize = [kernelSize.item()] * 2
        self._medianParameters['row_width'] = kernelSize[0]
        self._medianParameters['column_width'] = kernelSize[1]
        self._medianParametersWidget.widthSpin.setValue(int(kernelSize[0]))
        current = self.slider.value()
        self.showImage(current)

    def subtractBackground(self):
        current = self.getCurrentIndex()
        self.showImage(current)

    def showImage(self, index=0):
        if not self._images:
            return
        assert index < len(self._images)
        bg_index = None
        if self.backgroundButton.isChecked():
            for i, imageName in enumerate(self._labels):
                if imageName.lower().endswith('background'):
                    bg_index = i
                    break

        mf_text = ""
        a = self._medianParameters['row_width']
        b = self._medianParameters['column_width']
        if max(a, b) > 1:
            mf_text = "MF(%d,%d) " % (a, b)

        imdata = self._getMedianData(self._images[index])
        if bg_index is None:
            self.plot.setGraphTitle(mf_text + self._labels[index])
        else:
            # TODO: Should the channel at max. and channel at min. be
            # recalculated?
            self.plot.setGraphTitle(mf_text + self._labels[index] + " Net")
            imdata -= self._images[bg_index]

        # FIXME: we use z=0 because the silx mask is always on z=1
        self.plot.addImage(imdata,
                           legend="current",
                           origin=self._origin,
                           scale=self._scale,
                           replace=False,
                           z=0)      # TODO: z=1
        self.plot.setActiveImage("current")
        self.slider.setValue(index)

    def _getMedianData(self, data):
        data = copy.copy(data)
        if max(self._medianParameters['row_width'],
               self._medianParameters['column_width']) > 1:
            data = medfilt2d(data, [self._medianParameters['row_width'],
                                    self._medianParameters['column_width']])
        return data
