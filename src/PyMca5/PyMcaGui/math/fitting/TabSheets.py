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
import sys
from PyMca5.PyMcaGui import PyMcaQt as qt
QTVERSION = qt.qVersion()


class TabSheets(qt.QDialog):
    def __init__(self,parent = None,name = None,modal = 0,fl = 0, nohelp =1,nodefaults=1):
        qt.QDialog.__init__(self,parent)
        self.setWindowTitle(str("TabSheets"))
        self.setModal(modal)

        TabSheetsLayout = qt.QVBoxLayout(self)
        TabSheetsLayout.setContentsMargins(11, 11, 11, 11)
        TabSheetsLayout.setSpacing(6)

        self.tabWidget = qt.QTabWidget(self)

        self.Widget8 = qt.QWidget(self.tabWidget)
        self.Widget9 = qt.QWidget(self.tabWidget)
        self.tabWidget.addTab(self.Widget8,str("Tab"))
        self.tabWidget.addTab(self.Widget9,str("Tab"))

        TabSheetsLayout.addWidget(self.tabWidget)

        Layout2 = qt.QHBoxLayout(None)
        Layout2.setContentsMargins(0, 0, 0, 0)
        Layout2.setSpacing(6)

        if not nohelp:
            self.buttonHelp = qt.QPushButton(self)
            self.buttonHelp.setText(str("Help"))
            Layout2.addWidget(self.buttonHelp)

        if not nodefaults:
            self.buttonDefaults = qt.QPushButton(self)
            self.buttonDefaults.setText(str("Defaults"))
            Layout2.addWidget(self.buttonDefaults)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.buttonOk = qt.QPushButton(self)
        self.buttonOk.setText(str("OK"))
        Layout2.addWidget(self.buttonOk)

        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText(str("Cancel"))
        Layout2.addWidget(self.buttonCancel)
        TabSheetsLayout.addLayout(Layout2)

        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)
