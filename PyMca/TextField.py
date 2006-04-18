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
# Form implementation generated from reading ui file 'TextField.ui'
#
# Created: Thu Oct 17 14:18:49 2002
#      by: The PyQt User Interface Compiler (pyuic)
#
# WARNING! All changes made in this file will be lost!


from qt import *

def uic_load_pixmap_TextField(name):
    pix = QPixmap()
    m = QMimeSourceFactory.defaultFactory().data(name)

    if m:
        QImageDrag.decode(m,pix)

    return pix


class TextField(QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        QWidget.__init__(self,parent,name,fl)

        if name == None:
            self.setName("TextField")

        self.resize(373,44)
        self.setSizePolicy(QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
        self.setCaption(str("TextField"))

        TextFieldLayout = QHBoxLayout(self,11,6,"TextFieldLayout")

        Layout2 = QHBoxLayout(None,0,6,"Layout2")
        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.TextLabel = QLabel(self,"TextLabel")
        self.TextLabel.setSizePolicy(QSizePolicy(7,1,0,0,self.TextLabel.sizePolicy().hasHeightForWidth()))
        self.TextLabel.setText(str("TextLabel"))
        Layout2.addWidget(self.TextLabel)
        spacer_2 = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout2.addItem(spacer_2)
        TextFieldLayout.addLayout(Layout2)
