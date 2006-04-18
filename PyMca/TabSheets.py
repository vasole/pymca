#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
# Form implementation generated from reading ui file 'TabSheets.ui'
#
# Created: Thu Oct 17 15:59:53 2002
#      by: The PyQt User Interface Compiler (pyuic)
#
# WARNING! All changes made in this file will be lost!


from qt import *

def uic_load_pixmap_TabSheets(name):
    pix = QPixmap()
    m = QMimeSourceFactory.defaultFactory().data(name)

    if m:
        QImageDrag.decode(m,pix)

    return pix


class TabSheets(QDialog):
    def __init__(self,parent = None,name = None,modal = 0,fl = 0, nohelp =1,nodefaults=1):
        QDialog.__init__(self,parent,name,modal,fl)

        if name == None:
            self.setName("TabSheets")

        #self.resize(528,368)
        self.setCaption(str("TabSheets"))
        self.setSizeGripEnabled(1)

        TabSheetsLayout = QVBoxLayout(self,11,6,"TabSheetsLayout")

        self.tabWidget = QTabWidget(self,"tabWidget")

        self.Widget8 = QWidget(self.tabWidget,"Widget8")
        self.tabWidget.insertTab(self.Widget8,str("Tab"))

        self.Widget9 = QWidget(self.tabWidget,"Widget9")
        self.tabWidget.insertTab(self.Widget9,str("Tab"))
        TabSheetsLayout.addWidget(self.tabWidget)

        Layout2 = QHBoxLayout(None,0,6,"Layout2")

        if not nohelp:
            self.buttonHelp = QPushButton(self,"buttonHelp")
            self.buttonHelp.setText(str("Help"))
            self.buttonHelp.setAccel(4144)
            self.buttonHelp.setAutoDefault(1)
            Layout2.addWidget(self.buttonHelp)

        if not nodefaults:
            self.buttonDefaults = QPushButton(self,"buttonDefaults")
            self.buttonDefaults.setText(str("Defaults"))
            Layout2.addWidget(self.buttonDefaults)
        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.buttonOk = QPushButton(self,"buttonOk")
        self.buttonOk.setText(str("OK"))
        self.buttonOk.setAccel(0)
        self.buttonOk.setAutoDefault(1)
        self.buttonOk.setDefault(1)
        Layout2.addWidget(self.buttonOk)

        self.buttonCancel = QPushButton(self,"buttonCancel")
        self.buttonCancel.setText(str("Cancel"))
        self.buttonCancel.setAccel(0)
        self.buttonCancel.setAutoDefault(1)
        Layout2.addWidget(self.buttonCancel)
        TabSheetsLayout.addLayout(Layout2)

        self.connect(self.buttonOk,SIGNAL("clicked()"),self,SLOT("accept()"))
        self.connect(self.buttonCancel,SIGNAL("clicked()"),self,SLOT("reject()"))
