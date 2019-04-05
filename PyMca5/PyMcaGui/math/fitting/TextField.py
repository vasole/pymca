#/*##########################################################################
# Copyright (C) 2004-2019 V.A. Sole, European Synchrotron Radiation Facility
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
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt

QTVERSION = qt.qVersion()
_logger = logging.getLogger(__name__)


class TextField(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        qt.QWidget.__init__(self,parent)
        self.resize(373,44)
        try:
            self.setSizePolicy(qt.QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
        except:
            _logger.warning("TextField Bad Size policy")

        TextFieldLayout = qt.QHBoxLayout(self)
        Layout2 = qt.QHBoxLayout(None)
        Layout2.setContentsMargins(0, 0, 0, 0)
        Layout2.setSpacing(6)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.TextLabel = qt.QLabel(self)
        try:
            self.TextLabel.setSizePolicy(qt.QSizePolicy(7,1,0,0,self.TextLabel.sizePolicy().hasHeightForWidth()))
        except:
            _logger.warning("TextField Bad Size policy")

        self.TextLabel.setText(str("TextLabel"))
        Layout2.addWidget(self.TextLabel)
        spacer_2 = qt.QSpacerItem(20,20,
                                  qt.QSizePolicy.Expanding,qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer_2)
        TextFieldLayout.addLayout(Layout2)
