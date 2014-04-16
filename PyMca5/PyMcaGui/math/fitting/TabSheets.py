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

        self.buttonOk.clicked[()].connect(self.accept)
        self.buttonCancel.clicked[()].connect(self.reject)
