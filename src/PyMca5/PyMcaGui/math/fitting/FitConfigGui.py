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
from PyMca5.PyMcaGui import PyMcaQt as qt

QTVERSION = qt.qVersion()
def uic_load_pixmap_FitConfigGui(name):
    pix = qt.QPixmap()
    m = qt.QMimeSourceFactory.defaultFactory().data(name)

    if m:
        qt.QImageDrag.decode(m,pix)

    return pix


class FitConfigGui(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        qt.QWidget.__init__(self,parent)

        self.setWindowTitle(str("FitConfigGUI"))

        FitConfigGUILayout = qt.QHBoxLayout(self)
        FitConfigGUILayout.setContentsMargins(11, 11, 11, 11)
        FitConfigGUILayout.setSpacing(6)

        Layout9 = qt.QHBoxLayout(None)
        Layout9.setContentsMargins(0, 0, 0, 0)
        Layout9.setSpacing(6)

        Layout2 = qt.QGridLayout(None)
        Layout2.setContentsMargins(0, 0, 0, 0)
        Layout2.setSpacing(6)

        self.BkgComBox = qt.QComboBox(self)
        self.BkgComBox.addItem(str("Add Background"))

        Layout2.addWidget(self.BkgComBox,1,1)

        self.BkgLabel = qt.QLabel(self)
        self.BkgLabel.setText(str("Background"))

        Layout2.addWidget(self.BkgLabel,1,0)

        self.FunComBox = qt.QComboBox(self)
        self.FunComBox.addItem(str("Add Function(s)"))

        Layout2.addWidget(self.FunComBox,0,1)

        self.FunLabel = qt.QLabel(self)
        self.FunLabel.setText(str("Function"))

        Layout2.addWidget(self.FunLabel,0,0)
        Layout9.addLayout(Layout2)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer)

        Layout6 = qt.QGridLayout(None)
        Layout6.setContentsMargins(0, 0, 0, 0)
        Layout6.setSpacing(6)

        self.WeightCheckBox = qt.QCheckBox(self)
        self.WeightCheckBox.setText(str("Weight"))

        Layout6.addWidget(self.WeightCheckBox,0,0)

        self.MCACheckBox = qt.QCheckBox(self)
        self.MCACheckBox.setText(str("MCA Mode"))

        Layout6.addWidget(self.MCACheckBox,1,0)
        Layout9.addLayout(Layout6)

        Layout6_2 = qt.QGridLayout(None)
        Layout6_2.setContentsMargins(0, 0, 0, 0)
        Layout6_2.setSpacing(6)

        self.AutoFWHMCheckBox = qt.QCheckBox(self)
        self.AutoFWHMCheckBox.setText(str("Auto FWHM"))

        Layout6_2.addWidget(self.AutoFWHMCheckBox,0,0)

        self.AutoScalingCheckBox = qt.QCheckBox(self)
        self.AutoScalingCheckBox.setText(str("Auto Scaling"))

        Layout6_2.addWidget(self.AutoScalingCheckBox,1,0)
        Layout9.addLayout(Layout6_2)
        spacer_2 = qt.QSpacerItem(20,20,qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer_2)

        Layout5 = qt.QGridLayout(None)
        Layout5.setContentsMargins(0, 0, 0, 0)
        Layout5.setSpacing(6)

        self.PrintPushButton = qt.QPushButton(self)
        self.PrintPushButton.setText(str("Print"))

        Layout5.addWidget(self.PrintPushButton,1,0)

        self.ConfigureButton = qt.QPushButton(self)
        self.ConfigureButton.setText(str("Configure"))

        Layout5.addWidget(self.ConfigureButton,0,0)
        Layout9.addLayout(Layout5)
        FitConfigGUILayout.addLayout(Layout9)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = FitConfigGui()
    w.show()
    app.exec()
