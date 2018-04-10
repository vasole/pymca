#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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

from . import Object3DQt as qt
QTVERSION = qt.qVersion()

DEBUG = 0

class Object3DSlider(qt.QWidget):
    valueChanged = qt.pyqtSignal(float)

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
        self.__factor = 100.

    def setNum(self, value):
        if self.__factor != 0.0:
            value = value / self.__factor
        self.label.setText('%.2f' % value)
        self.valueChanged.emit(value)

    def setRange(self, minValue, maxValue, increment=None):
        if increment is None:
            self.__factor = 201.
        elif increment == 0.0:
            self.__factor = 1.0
        else:
            self.__factor = (maxValue - minValue) / float(increment)

        self.slider.setRange(int(minValue * self.__factor),
                             int(maxValue * self.__factor))

    def setValue(self, value):
        self.slider.setValue(value * self.__factor)

    def value(self):
        if self.__factor != 0.0:
            return self.slider.value()/self.__factor
        else:
            return float(self.slider.value())

    def minValue(self):
        if self.__factor != 0.0:
            return self.slider.minimum() / self.__factor
        else:
            return float(self.slider.minimum())

    def maxValue(self):
        if self.__factor != 0.0:
            return self.slider.maximum() / self.__factor
        else:
            return float(self.slider.maximum())

    def step(self):
        if self.__factor != 0.0:
            return self.slider.singleStep() / self.__factor
        else:
            return float(self.slider.singleStep())

    def singleStep(self):
        return self.step()

def test():
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)

    w = Object3DSlider()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()

