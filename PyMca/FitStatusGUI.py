#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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

class FitStatusGUI(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("FitStatusGUI")

            self.setCaption("FitStatusGUI")
        else:
            qt.QWidget.__init__(self,parent)

        self.resize(535,47)

        if QTVERSION < '4.0.0':
            FitStatusGUILayout = qt.QHBoxLayout(self,11,6,"FitStatusGUILayout")
        else:
            FitStatusGUILayout = qt.QHBoxLayout(self)
            FitStatusGUILayout.setMargin(11)
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
