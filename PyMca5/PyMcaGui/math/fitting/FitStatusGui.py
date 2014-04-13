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

class FitStatusGui(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        qt.QWidget.__init__(self,parent)

        self.resize(535,47)

        FitStatusGUILayout = qt.QHBoxLayout(self)
        FitStatusGUILayout.setContentsMargins(11, 11, 11, 11)
        FitStatusGUILayout.setSpacing(6)

        self.StatusLabel = qt.QLabel(self)
        self.StatusLabel.setText("Status:")
        FitStatusGUILayout.addWidget(self.StatusLabel)

        self.StatusLine = qt.QLineEdit(self)
        self.StatusLine.setText("Ready")
        self.StatusLine.setReadOnly(1)
        FitStatusGUILayout.addWidget(self.StatusLine)

        self.ChisqLabel = qt.QLabel(self)
        self.ChisqLabel.setText("Chisq:")
        FitStatusGUILayout.addWidget(self.ChisqLabel)

        self.ChisqLine = qt.QLineEdit(self)
        #self.ChisqLine.setSizePolicy(QSizePolicy(1,0,0,0,self.ChisqLine.sizePolicy().hasHeightForWidth()))
        self.ChisqLine.setMaximumSize(qt.QSize(16000,32767))
        self.ChisqLine.setText("")
        self.ChisqLine.setReadOnly(1)
        FitStatusGUILayout.addWidget(self.ChisqLine)
