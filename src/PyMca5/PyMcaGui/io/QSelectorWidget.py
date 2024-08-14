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

import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()

_logger = logging.getLogger(__name__)


class QSelectorWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self._build()
        self._buildActions()

    def _build(self):
        """
        Method to be overwritten to build the main widget
        """
        _logger.debug("_build():Method to be overwritten")
        pass

    def _buildActions(self):
        self.buttonBox = qt.QWidget(self)
        buttonBox = self.buttonBox
        self.buttonBoxLayout = qt.QHBoxLayout(buttonBox)

        self.addButton = qt.QPushButton(buttonBox)
        self.addButton.setText("ADD")
        self.removeButton = qt.QPushButton(buttonBox)
        self.removeButton.setText("REMOVE")
        self.replaceButton = qt.QPushButton(buttonBox)
        self.replaceButton.setText("REPLACE")

        self.buttonBoxLayout.addWidget(self.addButton)
        self.buttonBoxLayout.addWidget(self.removeButton)
        self.buttonBoxLayout.addWidget(self.replaceButton)

        self.mainLayout.addWidget(buttonBox)

        self.addButton.clicked.connect(self._addClickedSlot)
        self.removeButton.clicked.connect(self._removeClicked)
        self.replaceButton.clicked.connect(self._replaceClicked)

    def _addClickedSlot(self):
        self._addClicked()

    def _addClicked(self):
        _logger.debug("_addClicked()")

    def _removeClicked(self):
        _logger.debug("_removeClicked()")

    def _replaceClicked(self):
        _logger.debug("_replaceClicked()")


def test():
    app = qt.QApplication([])
    w = QSelectorWidget()
    w.show()
    app.lastWindowClosed.connect(app.quit)
    app.exec()

if __name__ == "__main__":
    test()
