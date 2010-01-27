#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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


def uic_load_pixmap_FitActionsGUI(name):
    pix = qt.QPixmap()
    if QTVERSION < '4.0.0':
        m = qt.QMimeSourceFactory.defaultFactory().data(name)

        if m:
            qt.QImageDrag.decode(m,pix)

    return pix


class FitActionsGUI(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
            if name == None:
                self.setName("FitActionsGUI")
            self.setCaption(str("FitActionsGUI"))
        else:
            qt.QWidget.__init__(self,parent)

        self.resize(234,53)

        if QTVERSION < '4.0.0':
            FitActionsGUILayout = qt.QGridLayout(self,1,1,11,6,"FitActionsGUILayout")
            Layout9 = qt.QHBoxLayout(None,0,6,"Layout9")
            
        else:
            FitActionsGUILayout = qt.QGridLayout(self)
            FitActionsGUILayout.setMargin(11)
            FitActionsGUILayout.setSpacing(6)
            Layout9 = qt.QHBoxLayout(None)
            Layout9.setMargin(0)
            Layout9.setSpacing(6)

        self.EstimateButton = qt.QPushButton(self)
        self.EstimateButton.setText(str("Estimate"))
        Layout9.addWidget(self.EstimateButton)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer)

        self.StartfitButton = qt.QPushButton(self)
        self.StartfitButton.setText(str("Start Fit"))
        Layout9.addWidget(self.StartfitButton)
        spacer_2 = qt.QSpacerItem(20,20,
                                  qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer_2)

        self.DismissButton = qt.QPushButton(self)
        self.DismissButton.setText(str("Dismiss"))
        Layout9.addWidget(self.DismissButton)

        FitActionsGUILayout.addLayout(Layout9,0,0)
        
if __name__ == "__main__":
    app = qt.QApplication([])
    w = FitActionsGUI()
    w.show()
    app.exec_()
