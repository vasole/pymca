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
import os
import SimpleFitModule
import Parameters
qt = Parameters.qt
if qt.qVersion() < '4.0.0':
    raise ImportError, "This module requires PyQt4"

DEBUG = 0

class TopWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)


class SimpleFitGUI(qt.QWidget):
    def __init__(self, parent=None, fit=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("SimpleFitGUI")
        if fit is None:
            self.fitModule = SimpleFitModule.SimpleFit()
        else:
            self.fitModule = fit
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        self.topWidget  = TopWidget(self)
        self.parametersTable = Parameters.Parameters(self)

        #build the actions widget
        self.fitActions = qt.QWidget(self)
        self.fitActions.mainLayout = qt.QHBoxLayout(self.fitActions)
        self.fitActions.mainLayout.setMargin(2)
        self.fitActions.mainLayout.setSpacing(2)
        self.fitActions.estimateButton = qt.QPushButton(self.fitActions)
        self.fitActions.estimateButton.setText("Estimate")
        self.fitActions.startFitButton = qt.QPushButton(self.fitActions)
        self.fitActions.startFitButton.setText("Start Fit")
        self.fitActions.dismissButton = qt.QPushButton(self.fitActions)
        self.fitActions.dismissButton.setText("Dismiss")
        self.fitActions.mainLayout.addWidget(self.fitActions.estimateButton)
        self.fitActions.mainLayout.addWidget(self.fitActions.startFitButton)
        self.fitActions.mainLayout.addWidget(self.fitActions.dismissButton)
        """
        self.connect(self.fitActions.estimateButton,
                    qt.SIGNAL("clicked()"),self.estimate)
        self.connect(self.fitActions.startFitButton,
                                qt.SIGNAL("clicked()"),self.startFit)
        self.connect(self.fitActions.dismissButton,
                                qt.SIGNAL("clicked()"),self.dismiss)        
        """
        self.mainLayout.addWidget(self.topWidget)
        self.mainLayout.addWidget(self.parametersTable)
        self.mainLayout.addWidget(self.fitActions)

    def estimate(self):
        print "estimate called"

    def startFit(self):
        print "start fit called"

    def dismiss(self):
        self.close()

def test():
    w = SimpleFitGUI()
    w.show()
    return w

if __name__=="__main__":
    DEBUG = 1
    app = qt.QApplication([])
    w = test()
    app.exec_()
