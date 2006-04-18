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
# Form implementation generated from reading ui file 'FitActionsGUI.ui'
#
# Created: Tue Oct 15 09:46:36 2002
#      by: The PyQt User Interface Compiler (pyuic)
#
# WARNING! All changes made in this file will be lost!


from qt import *

def uic_load_pixmap_FitActionsGUI(name):
    pix = QPixmap()
    m = QMimeSourceFactory.defaultFactory().data(name)

    if m:
        QImageDrag.decode(m,pix)

    return pix


class FitActionsGUI(QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        QWidget.__init__(self,parent,name,fl)

        if name == None:
            self.setName("FitActionsGUI")

        self.resize(234,53)
        self.setCaption(str("FitActionsGUI"))

        FitActionsGUILayout = QGridLayout(self,1,1,11,6,"FitActionsGUILayout")

        Layout9 = QHBoxLayout(None,0,6,"Layout9")

        self.EstimateButton = QPushButton(self,"EstimateButton")
        self.EstimateButton.setText(str("Estimate"))
        Layout9.addWidget(self.EstimateButton)
        spacer = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout9.addItem(spacer)

        self.StartfitButton = QPushButton(self,"StartfitButton")
        self.StartfitButton.setText(str("Start Fit"))
        Layout9.addWidget(self.StartfitButton)
        spacer_2 = QSpacerItem(20,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        Layout9.addItem(spacer_2)

        self.DismissButton = QPushButton(self,"DismissButton")
        self.DismissButton.setText(str("Dismiss"))
        Layout9.addWidget(self.DismissButton)

        FitActionsGUILayout.addLayout(Layout9,0,0)
