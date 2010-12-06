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

DEBUG = 0

class QSelectorWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self._build()
        self._buildActions()

    def _build(self):
        """
        Method to be overwritten to build the main widget
        """
        if DEBUG:
            print("_build():Method to be overwritten")
        pass

    def _buildActions(self):
        self.buttonBox = qt.QWidget(self)
        buttonBox = self.buttonBox
        self.buttonBoxLayout = qt.QHBoxLayout(buttonBox)
        
        self.addButton = qt.QPushButton(buttonBox)
        self.addButton.setText("ADD")
        self.removeButton = qt.QPushButton(buttonBox)
        self.removeButton.setText("REMOVE")
        self.replaceButton = qt.QPushButton(buttonBox)
        self.replaceButton.setText("REPLACE")
        
        self.buttonBoxLayout.addWidget(self.addButton)
        self.buttonBoxLayout.addWidget(self.removeButton)
        self.buttonBoxLayout.addWidget(self.replaceButton)
        
        self.mainLayout.addWidget(buttonBox)
        
        self.connect(self.addButton, qt.SIGNAL("clicked()"), 
                    self._addClicked)

        self.connect(self.removeButton, qt.SIGNAL("clicked()"), 
                    self._removeClicked)

        self.connect(self.replaceButton, qt.SIGNAL("clicked()"), 
                    self._replaceClicked)

    def _addClicked(self):
        if DEBUG:
            print("_addClicked()")
    
    def _removeClicked(self):
        if DEBUG:
            print("_removeClicked()")   

    def _replaceClicked(self):
        if DEBUG: print(
            "_replaceClicked()")

            
def test():
    app = qt.QApplication([])
    w = QSelectorWidget()
    w.show()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()
        
if __name__ == "__main__":
    test()
