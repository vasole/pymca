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
# Form implementation generated from reading ui file 'EntryField.ui'
#
# Created: Thu Oct 17 14:32:48 2002
#      by: The PyQt User Interface Compiler (pyuic)
#
# WARNING! All changes made in this file will be lost!


from qt import *

def uic_load_pixmap_EntryField(name):
    pix = QPixmap()
    m = QMimeSourceFactory.defaultFactory().data(name)

    if m:
        QImageDrag.decode(m,pix)

    return pix


class EntryField(QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        QWidget.__init__(self,parent,name,fl)

        if name == None:
            self.setName("EntryField")

        #self.resize(317,65)
        #self.setSizePolicy(QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
        self.setCaption(str("EntryField"))

        #EntryFieldLayout = QVBoxLayout(self,11,6,"EntryFieldLayout")
        #spacer = QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding)
        #EntryFieldLayout.addItem(spacer)

        Layout1 = QHBoxLayout(self)
        Layout1.setAutoAdd(1)

        self.TextLabel = QLabel(self,"TextLabel")
        #self.TextLabel.setSizePolicy(QSizePolicy(1,1,0,0,self.TextLabel.sizePolicy().hasHeightForWidth()))
        self.TextLabel.setText(str("TextLabel"))
        #Layout1.addWidget(self.TextLabel)

        self.Entry = QLineEdit(self,"Entry")
        #self.Entry.setSizePolicy(QSizePolicy(7,1,0,0,self.Entry.sizePolicy().hasHeightForWidth()))
        #Layout1.addWidget(self.Entry)
        #EntryFieldLayout.addLayout(Layout1)
        #spacer_2 = QSpacerItem(20,20,QSizePolicy.Minimum,QSizePolicy.Expanding)
        #EntryFieldLayout.addItem(spacer_2)
