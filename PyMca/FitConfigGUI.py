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
# Form implementation generated from reading ui file 'FitConfigGUI.ui'
#
# Created: Tue Oct 15 18:25:44 2002
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
def uic_load_pixmap_FitConfigGUI(name):
    pix = qt.QPixmap()
    m = qt.QMimeSourceFactory.defaultFactory().data(name)

    if m:
        qt.QImageDrag.decode(m,pix)

    return pix


class FitConfigGUI(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("FitConfigGUI")

            self.setCaption(str("FitConfigGUI"))

            FitConfigGUILayout = qt.QHBoxLayout(self,11,6,"FitConfigGUILayout")
    
            Layout9 = qt.QHBoxLayout(None,0,6,"Layout9")
    
            Layout2 = qt.QGridLayout(None,1,1,0,6,"Layout2")
        else:
            qt.QWidget.__init__(self,parent)

            self.setWindowTitle(str("FitConfigGUI"))

            FitConfigGUILayout = qt.QHBoxLayout(self)
            FitConfigGUILayout.setMargin(11)
            FitConfigGUILayout.setSpacing(6)
    
            Layout9 = qt.QHBoxLayout(None)
            Layout9.setMargin(0)
            Layout9.setSpacing(6)
    
            Layout2 = qt.QGridLayout(None)
            Layout2.setMargin(0)
            Layout2.setSpacing(6)

        if QTVERSION < '4.0.0':
            self.BkgComBox = qt.QComboBox(0,self,"BkgComBox")
            self.BkgComBox.insertItem(str("Add Background"))
        else:
            self.BkgComBox = qt.QComboBox(self)
            self.BkgComBox.addItem(str("Add Background"))

        Layout2.addWidget(self.BkgComBox,1,1)

        self.BkgLabel = qt.QLabel(self)
        self.BkgLabel.setText(str("Background"))

        Layout2.addWidget(self.BkgLabel,1,0)

        if QTVERSION < '4.0.0':
            self.FunComBox = qt.QComboBox(0,self,"FunComBox")
            self.FunComBox.insertItem(str("Add Function(s)"))
        else:
            self.FunComBox = qt.QComboBox(self)
            self.FunComBox.addItem(str("Add Function(s)"))

        Layout2.addWidget(self.FunComBox,0,1)

        self.FunLabel = qt.QLabel(self)
        self.FunLabel.setText(str("Function"))

        Layout2.addWidget(self.FunLabel,0,0)
        Layout9.addLayout(Layout2)
        spacer = qt.QSpacerItem(20,20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer)

        if QTVERSION < '4.0.0':
            Layout6 = qt.QGridLayout(None,1,1,0,6,"Layout6")
        else:
            Layout6 = qt.QGridLayout(None)
            Layout6.setMargin(0)
            Layout6.setSpacing(6)

        self.WeightCheckBox = qt.QCheckBox(self)
        self.WeightCheckBox.setText(str("Weight"))

        Layout6.addWidget(self.WeightCheckBox,0,0)

        self.MCACheckBox = qt.QCheckBox(self)
        self.MCACheckBox.setText(str("MCA Mode"))

        Layout6.addWidget(self.MCACheckBox,1,0)
        Layout9.addLayout(Layout6)

        if QTVERSION < '4.0.0':
            Layout6_2 = qt.QGridLayout(None,1,1,0,6,"Layout6_2")
        else:
            Layout6_2 = qt.QGridLayout(None)
            Layout6_2.setMargin(0)
            Layout6_2.setSpacing(6)
            
        self.AutoFWHMCheckBox = qt.QCheckBox(self)
        self.AutoFWHMCheckBox.setText(str("Auto FWHM"))

        Layout6_2.addWidget(self.AutoFWHMCheckBox,0,0)

        self.AutoScalingCheckBox = qt.QCheckBox(self)
        self.AutoScalingCheckBox.setText(str("Auto Scaling"))

        Layout6_2.addWidget(self.AutoScalingCheckBox,1,0)
        Layout9.addLayout(Layout6_2)
        spacer_2 = qt.QSpacerItem(20,20,qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        Layout9.addItem(spacer_2)

        if QTVERSION < '4.0.0':
            Layout5 = qt.QGridLayout(None,1,1,0,6,"Layout5")
        else:
            Layout5 = qt.QGridLayout(None)
            Layout5.setMargin(0)
            Layout5.setSpacing(6)

        self.PrintPushButton = qt.QPushButton(self)
        self.PrintPushButton.setText(str("Print"))

        Layout5.addWidget(self.PrintPushButton,1,0)

        self.ConfigureButton = qt.QPushButton(self)
        self.ConfigureButton.setText(str("Configure"))

        Layout5.addWidget(self.ConfigureButton,0,0)
        Layout9.addLayout(Layout5)
        FitConfigGUILayout.addLayout(Layout9)
