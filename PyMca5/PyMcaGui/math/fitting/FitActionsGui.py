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
from PyMca5.PyMcaGui import PyMcaQt as qt

QTVERSION = qt.qVersion()


def uic_load_pixmap_FitActionsGui(name):
    pix = qt.QPixmap()
    if QTVERSION < '4.0.0':
        m = qt.QMimeSourceFactory.defaultFactory().data(name)

        if m:
            qt.QImageDrag.decode(m,pix)

    return pix


class FitActionsGui(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        qt.QWidget.__init__(self,parent)

        self.resize(234,53)

        FitActionsGUILayout = qt.QGridLayout(self)
        FitActionsGUILayout.setContentsMargins(11, 11, 11, 11)
        FitActionsGUILayout.setSpacing(6)
        Layout9 = qt.QHBoxLayout(None)
        Layout9.setContentsMargins(0, 0, 0, 0)
        Layout9.setSpacing(6)

        self.EstimateButton = qt.QPushButton(self)
        self.EstimateButton.setText("Estimate")
        Layout9.addWidget(self.EstimateButton)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer)

        self.StartfitButton = qt.QPushButton(self)
        self.StartfitButton.setText("Start Fit")
        Layout9.addWidget(self.StartfitButton)
        spacer_2 = qt.QSpacerItem(20,20,
                                  qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer_2)

        self.DismissButton = qt.QPushButton(self)
        self.DismissButton.setText("Dismiss")
        Layout9.addWidget(self.DismissButton)

        FitActionsGUILayout.addLayout(Layout9,0,0)
        
if __name__ == "__main__":
    app = qt.QApplication([])
    w = FitActionsGui()
    w.show()
    app.exec_()
