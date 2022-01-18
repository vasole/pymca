#/*##########################################################################
# Copyright (C) 2004-2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()

_logger = logging.getLogger(__name__)


class DoubleSlider(qt.QWidget):
    sigDoubleSliderValueChanged = qt.pyqtSignal(object)

    def __init__(self, parent = None, scale = False):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(6, 6, 6, 6)
        self.mainLayout.setSpacing(1)
        orientation = qt.Qt.Horizontal

        self.minSlider = MySlider(self, orientation)
        self.minSlider.setRange(0, 100.)
        self.minSlider.setValue(0)
        self.maxSlider = MySlider(self, orientation)
        self.maxSlider.setRange(0, 100.0)
        self.maxSlider.setValue(100.)
        self.mainLayout.addWidget(self.maxSlider)
        self.mainLayout.addWidget(self.minSlider)
        self.minSlider.sigValueChanged.connect(self._sliderChanged)
        self.maxSlider.sigValueChanged.connect(self._sliderChanged)

    def __getDict(self):
        ddict = {}
        ddict['event'] = "doubleSliderValueChanged"
        m   = self.minSlider.value()
        M   = self.maxSlider.value()
        if m > M:
            ddict['max'] = m
            ddict['min'] = M
        else:
            ddict['min'] = m
            ddict['max'] = M
        return ddict

    def _sliderChanged(self, value):
        _logger.debug("DoubleSlider._sliderChanged()")
        ddict = self.__getDict()
        self.sigDoubleSliderValueChanged.emit(ddict)

    def setMinMax(self, m, M):
        self.minSlider.setValue(m)
        self.maxSlider.setValue(M)

    def getMinMax(self):
        m = self.minSlider.value()
        M = self.maxSlider.value()
        if m > M:
            return M, m
        else:
            return m, M


class MySlider(qt.QWidget):
    sigValueChanged = qt.pyqtSignal(float)

    def __init__(self, parent = None, orientation=qt.Qt.Horizontal):
        qt.QWidget.__init__(self, parent)
        if orientation == qt.Qt.Horizontal:
            alignment = qt.Qt.AlignHCenter | qt.Qt.AlignTop
            layout = qt.QHBoxLayout(self)
        else:
            alignment = qt.Qt.AlignVCenter | qt.Qt.AlignLeft
            layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.slider = qt.QSlider(self)
        self.slider.setOrientation(orientation)
        self.label  = qt.QLabel("0", self)
        self.label.setAlignment(alignment)
        self.label.setFixedWidth(self.label.fontMetrics().maxWidth()*len('100.99'))

        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.slider.valueChanged.connect(self.setNum)

    def setNum(self, value):
        value = value / 100.
        self.label.setText('%.2f' % value)
        self.sigValueChanged.emit(value)

    def setRange(self, minValue, maxValue):
        self.slider.setRange(int(minValue * 100), int(maxValue * 100))

    def setValue(self, value):
        self.slider.setValue(int(value * 100))

    def value(self):
        return self.slider.value()/100.

def test():
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    w = DoubleSlider()
    w.show()
    app.exec()

if __name__ == "__main__":
    test()

