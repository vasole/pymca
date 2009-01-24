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
import PyMcaQt as qt

QTVERSION = qt.qVersion()
DEBUG = 0

def uic_load_pixmap_FitActionsGUI(name):
    pix = qt.QPixmap()
    if QTVERSION < '4.0.0':
        m = qt.QMimeSourceFactory.defaultFactory().data(name)

        if m:
            qt.QImageDrag.decode(m,pix)

    return pix

class TextField(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("TextField")

            self.setCaption(str("TextField"))
        else:
            qt.QWidget.__init__(self,parent)
        self.resize(373,44)
        try:
            self.setSizePolicy(qt.QSizePolicy(1,1,0,0,self.sizePolicy().hasHeightForWidth()))
        except:
            if DEBUG:print "TextField Bad Size policy"

        if QTVERSION < '4.0.0':
            TextFieldLayout = qt.QHBoxLayout(self,11,6,"TextFieldLayout")
            Layout2 = qt.QHBoxLayout(None,0,6,"Layout2")
        else:
            TextFieldLayout = qt.QHBoxLayout(self)
            Layout2 = qt.QHBoxLayout(None)
            Layout2.setMargin(0)
            Layout2.setSpacing(6)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer)

        self.TextLabel = qt.QLabel(self)
        try:
            self.TextLabel.setSizePolicy(qt.QSizePolicy(7,1,0,0,self.TextLabel.sizePolicy().hasHeightForWidth()))
        except:
            if DEBUG:print "TextField Bad Size policy"
            
        self.TextLabel.setText(str("TextLabel"))
        Layout2.addWidget(self.TextLabel)
        spacer_2 = qt.QSpacerItem(20,20,
                                  qt.QSizePolicy.Expanding,qt.QSizePolicy.Minimum)
        Layout2.addItem(spacer_2)
        TextFieldLayout.addLayout(Layout2)
