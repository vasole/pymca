#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
import sys
from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()

def uic_load_pixmap_FitActionsGUI(name):
    pix = qt.QPixmap()
    if QTVERSION < '4.0.0':
        m = qt.QMimeSourceFactory.defaultFactory().data(name)

        if m:
            qt.QImageDrag.decode(m,pix)

    return pix


class TabSheets(qt.QDialog):
    def __init__(self,parent = None,name = None,modal = 0,fl = 0, nohelp =1,nodefaults=1):
        if QTVERSION < '4.0.0':
            qt.QDialog.__init__(self,parent,name,modal,fl)
            if name == None:
                self.setName("TabSheets")

            self.setCaption(str("TabSheets"))
            self.setSizeGripEnabled(1)
        else:
            qt.QDialog.__init__(self,parent)
            self.setWindowTitle(str("TabSheets"))
            self.setModal(modal)
            #,fl)

        if QTVERSION < '4.0.0':
            TabSheetsLayout = qt.QVBoxLayout(self,11,6,"TabSheetsLayout")
        else:
            TabSheetsLayout = qt.QVBoxLayout(self)
            TabSheetsLayout.setMargin(11)
            TabSheetsLayout.setSpacing(6)
            
        self.tabWidget = qt.QTabWidget(self)

        self.Widget8 = qt.QWidget(self.tabWidget)
        self.Widget9 = qt.QWidget(self.tabWidget)
        if QTVERSION < '4.0.0':
            self.tabWidget.insertTab(self.Widget8,str("Tab"))
            self.tabWidget.insertTab(self.Widget9,str("Tab"))
        else:
            self.tabWidget.addTab(self.Widget8,str("Tab"))
            self.tabWidget.addTab(self.Widget9,str("Tab"))

        TabSheetsLayout.addWidget(self.tabWidget)

        if QTVERSION < '4.0.0':
            Layout2 = qt.QHBoxLayout(None,0,6,"Layout2")
        else:
            Layout2 = qt.QHBoxLayout(None)
            Layout2.setMargin(0)
            Layout2.setSpacing(6)

        if not nohelp:
            self.buttonHelp = qt.QPushButton(self)
            self.buttonHelp.setText(str("Help"))
            if QTVERSION < '4.0.0':
                self.buttonHelp.setAccel(4144)
                self.buttonHelp.setAutoDefault(1)
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
        if QTVERSION < '4.0.0':
            self.buttonOk.setAccel(0)
            self.buttonOk.setAutoDefault(1)
            self.buttonOk.setDefault(1)
        Layout2.addWidget(self.buttonOk)

        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText(str("Cancel"))    
        if QTVERSION < '4.0.0':
            self.buttonCancel.setAccel(0)
            self.buttonCancel.setAutoDefault(1)
        Layout2.addWidget(self.buttonCancel)
        TabSheetsLayout.addLayout(Layout2)

        self.connect(self.buttonOk, qt.SIGNAL("clicked()"),
                     self, qt.SLOT("accept()"))
        self.connect(self.buttonCancel,qt.SIGNAL("clicked()"),
                     self, qt.SLOT("reject()"))
