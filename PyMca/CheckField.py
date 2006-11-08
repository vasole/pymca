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
# Form implementation generated from reading ui file 'CheckField.ui'
#
# Created: Thu Oct 17 14:18:49 2002
#      by: The PyQt User Interface Compiler (pyuic)
#
# WARNING! All changes made in this file will be lost!


import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
    except:
        import qt
else:
    import qt

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

            self.setCaption(str("CheckField"))
        else:
            qt.QWidget.__init__(self,parent)
        self.resize(321,45)


        if QTVERSION < '4.0.0':
            CheckFieldLayout = qt.QHBoxLayout(self,11,6,"CheckFieldLayout")
        else:
            CheckFieldLayout = qt.QHBoxLayout(self)
            CheckFieldLayout.setMargin(11)
            CheckFieldLayout.setSpacing(6)


        self.CheckBox = qt.QCheckBox(self)
        self.CheckBox.setText(str("CheckBox"))
        CheckFieldLayout.addWidget(self.CheckBox)
