#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
from PyMca5.PyMcaGui import XASNormalizationWindow
IconDict = PyMca_Icons.IconDict

class Normalization(qt.QGroupBox):
    def __init__(self, parent=None):
        super(Normalization, self).__init__(parent)
        self.setTitle("Normalization")
        self.build()

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        # the setup button
        self.setupButton = qt.QPushButton(self)
        self.setupButton.setText("SETUP")
        self.setupButton.setAutoDefault(False)

        # the E0 value
        self.e0CheckBox = qt.QCheckBox(self)
        self.e0CheckBox.setText("Auto E0:")
        self.e0CheckBox.setChecked(True)
        self.e0SpinBox = qt.QDoubleSpinBox(self)
        self.e0SpinBox.setDecimals(2)
        self.e0SpinBox.setSingleStep(0.2)
        self.e0SpinBox.setEnabled(False)

        # the jump
        jumpLabel = qt.QLabel(self)
        jumpLabel.setText("Jump:")
        self.jumpLine = qt.QLineEdit(self)
        self.jumpLine.setEnabled(False)

        # the pre-edge
        preLabel = qt.QLabel(self)
        preLabel.setText("Pre-Edge")
        self.preEdgeSelector = XASNormalizationWindow.PolynomSelector(self)

        # the post-edge
        postLabel = qt.QLabel(self)
        postLabel.setText("Post-Edge")
        self.postEdgeSelector = XASNormalizationWindow.PolynomSelector(self)

        # arrange everything
        self.mainLayout.addWidget(self.setupButton, 0, 0, 1, 2)
        self.mainLayout.addWidget(self.e0CheckBox, 1, 0)
        self.mainLayout.addWidget(self.e0SpinBox, 1, 1)
        self.mainLayout.addWidget(jumpLabel, 2, 0)
        self.mainLayout.addWidget(self.jumpLine, 2, 1)
        self.mainLayout.addWidget(preLabel, 3, 0)
        self.mainLayout.addWidget(self.preEdgeSelector, 3, 1)
        self.mainLayout.addWidget(postLabel, 4, 0)
        self.mainLayout.addWidget(self.postEdgeSelector, 4, 1)

        # connect
        self.setupButton.clicked.connect(self._setupClicked)
        self.e0CheckBox.toggled.connect(self._e0Toggled)
        self.e0SpinBox.valueChanged[float].connect(self._e0Changed)
        self.preEdgeSelector.activated[int].connect(self._preEdgeChanged)
        self.postEdgeSelector.activated[int].connect(self._postEdgeChanged)

    def _setupClicked(self):
        print("SETUP CLICKED")

    def _e0Toggled(self, state):
        print("CURRENT STATE = ", state)
        print("STATE from call = ", self.e0CheckBox.isChecked())
        if state:
            print("E0 to be calculated")
            self.e0SpinBox.setEnabled(False)
        else:
            self.e0SpinBox.setEnabled(True)

    def _e0Changed(self, value):
        print("E0 VALUE", value)

    def _preEdgeChanged(self, value):
        print("Current pre-edge value = ", value)

    def _postEdgeChanged(self, value):
        print("Current post-edge value = ", value)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = Normalization()
    w.show()
    app.exec_()
