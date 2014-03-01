#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
import sys
from PyMca.widgets import ScanWindow
qt = ScanWindow.qt
DEBUG = 0

class ProfileScanWidget(ScanWindow.ScanWindow):
    def __init__(self, parent=None, actions=False, **kw):
        ScanWindow.ScanWindow.__init__(self, parent, **kw)
        if actions:
            self._buildActionsBox()

    def _buildActionsBox(self):
        self.labelBox = qt.QWidget(self)
        self.labelBox.mainLayout = qt.QHBoxLayout(self.labelBox)

        self.labelLabel = qt.QLabel(self.labelBox)
        
        self.labelLabel.setText("Selection Label = ")
        self.label = qt.QLineEdit(self.labelBox)
        self.labelBox.mainLayout.addWidget(self.labelLabel)
        self.labelBox.mainLayout.addWidget(self.label)

        self.buttonBox = self.labelBox
        buttonBox = self.buttonBox
        self.buttonBoxLayout = self.labelBox.mainLayout
        self.buttonBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonBoxLayout.setSpacing(0)
        self.addButton = qt.QPushButton(buttonBox)
        self.addButton.setText("ADD")
        self.addButton.setToolTip("Add curves to destination widget")
        self.removeButton = qt.QPushButton(buttonBox)
        self.removeButton.setText("REMOVE")
        self.removeButton.setToolTip("Remove curves from destination widget")
        self.buttonBoxLayout.addWidget(self.addButton)
        self.buttonBoxLayout.addWidget(self.removeButton)
        self.replaceButton = qt.QPushButton(buttonBox)
        self.replaceButton.setText("REPLACE")
        self.replaceButton.setToolTip("Replace curves in destination widget")
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
            print("ADD clicked")
        self._emitActionSignal(action='ADD')

    def _removeClicked(self):
        if DEBUG:
            print("REMOVE clicked")
        self._emitActionSignal(action='REMOVE')

    def _replaceClicked(self):
        if DEBUG:
            print("REPLACE clicked")
        self._emitActionSignal(action='REPLACE')

    def _emitActionSignal(self, action='ADD'):
        if action not in ['ADD', 'REMOVE', 'REPLACE']:
            print("Unrecognized action %s" % action)

        curveList = self.getAllCurves()
        if curveList in [None, []]:
            return
        text = self.label.text()
        if sys.version < '3.0':
            text = str(text)

        ddict = {}
        ddict['event']   = action
        ddict['action']  = action
        ddict['label']   = text
        ddict['curves']  = curveList
        if action == 'ADD':
            self.emit(qt.SIGNAL("addClicked"), ddict)
        elif action == 'REMOVE':
            self.emit(qt.SIGNAL("removeClicked"), ddict)
        else:
            self.emit(qt.SIGNAL("replaceClicked"), ddict)

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))
    def testSlot(ddict):
        print(ddict)
    w = ProfileScanWidget(actions=True)
    w.addCurve([1, 2, 3, 4], [1, 4, 9, 16], legend='Dummy')
    qt.QObject.connect(w, qt.SIGNAL("addClicked"), testSlot)
    qt.QObject.connect(w, qt.SIGNAL("removeClicked"), testSlot)
    qt.QObject.connect(w, qt.SIGNAL("replaceClicked"), testSlot)
    w.show()
    app.exec_()

if __name__ == "__main__":
    DEBUG = 1
    test()
