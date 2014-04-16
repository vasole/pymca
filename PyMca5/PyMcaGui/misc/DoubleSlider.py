#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()

DEBUG = 0
    
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
        if DEBUG:
            print("DoubleSlider._sliderChanged()")
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
        self.label.setFixedWidth(self.label.fontMetrics().width('100.99'))

        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.slider.valueChanged.connect(self.setNum)

    def setNum(self, value):
        value = value / 100.
        self.label.setText('%.2f' % value)
        self.sigValueChanged.emit(value)

    def setRange(self, minValue, maxValue):
        self.slider.setRange(minValue * 100, int(maxValue * 100))

    def setValue(self, value):
        self.slider.setValue(value * 100)

    def value(self):
        return self.slider.value()/100.

def test():
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)    
    w = DoubleSlider()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
 
