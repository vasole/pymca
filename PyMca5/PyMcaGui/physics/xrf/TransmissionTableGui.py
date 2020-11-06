#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
import copy
import logging
import numpy
import traceback
from PyMca5.PyMcaGui import PyMcaQt as qt
from .TransmissionTableEditor import TransmissionTableEditor

_logger = logging.getLogger(__name__)

class TransmissionTableGui(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        layout = qt.QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        self.groupBoxList = []
        for i in range(2):
            groupBox = qt.QGroupBox(self)
            groupBox.setAlignment(qt.Qt.AlignHCenter)
            groupBox.setFlat(False)
            groupBox.setCheckable(False)
            groupBox.setTitle("Attenuation Table %d" % i)
            groupBoxLayout = qt.QVBoxLayout(groupBox)
            groupBox.transmissionTable = TransmissionTableEditor(groupBox)
            groupBoxLayout.addWidget(groupBox.transmissionTable)
            groupBoxLayout.addWidget(qt.VerticalSpacer(groupBox))
            layout.addWidget(groupBox, 0, i)
            self.groupBoxList.append(groupBox)

if __name__ == "__main__":
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)
    demo = TransmissionTableGui()
    demo.show()
    ret  = app.exec_()
    app = None
