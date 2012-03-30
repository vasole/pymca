#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
from PyMca import PyMcaQt as qt

QTVERSION = qt.qVersion()

def uic_load_pixmap_FitActionsGUI(name):
    pix = qt.QPixmap()
    if QTVERSION < '4.0.0':
        m = qt.QMimeSourceFactory.defaultFactory().data(name)
        if m:
            qt.QImageDrag.decode(m,pix)

    return pix

class CheckField(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("CheckField")

            self.setCaption("CheckField")
        else:
            qt.QWidget.__init__(self,parent)
        self.resize(321,45)


        if QTVERSION < '4.0.0':
            CheckFieldLayout = qt.QHBoxLayout(self, 11, 6,
                                              "CheckFieldLayout")
        else:
            CheckFieldLayout = qt.QHBoxLayout(self)
            CheckFieldLayout.setMargin(11)
            CheckFieldLayout.setSpacing(6)


        self.CheckBox = qt.QCheckBox(self)
        self.CheckBox.setText("CheckBox")
        CheckFieldLayout.addWidget(self.CheckBox)
