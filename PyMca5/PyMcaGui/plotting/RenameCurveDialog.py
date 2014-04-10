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
from .. import PyMcaQt as qt

class RenameCurveDialog(qt.QDialog):
    def __init__(self, parent = None, current="", curves = []):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Rename Curve %s" % current)
        self.curves = curves
        layout = qt.QVBoxLayout(self)
        self.lineEdit = qt.QLineEdit(self)
        self.lineEdit.setText(current)
        self.hbox = qt.QWidget(self)
        self.hboxLayout = qt.QHBoxLayout(self.hbox)
        self.hboxLayout.addWidget(qt.HorizontalSpacer(self.hbox))
        self.okButton    = qt.QPushButton(self.hbox)
        self.okButton.setText('OK')
        self.hboxLayout.addWidget(self.okButton)
        self.cancelButton = qt.QPushButton(self.hbox)
        self.cancelButton.setText('Dismiss')
        self.hboxLayout.addWidget(self.cancelButton)
        self.hboxLayout.addWidget(qt.HorizontalSpacer(self.hbox))
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.hbox)
        self.okButton.clicked[()].connect(self.preAccept)
        self.cancelButton.clicked[()].connect(self.reject)

    def preAccept(self):
        text = str(self.lineEdit.text())
        addedText = "" 
        if len(text):
            if text not in self.curves:
                self.accept()
                return
            else:
                addedText = "Curve already exists."
        text = "Invalid Curve Name"
        msg = qt.QMessageBox(self)
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setWindowTitle(text)
        text += "\n%s" % addedText
        msg.setText(text)
        msg.exec_()

    def getText(self):
        return str(self.lineEdit.text())


if __name__ == "__main__":
   app = qt.QApplication([])
   w=RenameCurveDialog(None, 'curve1', ['curve1', 'curve2', 'curve3'])
   ret = w.exec_()
   if ret == qt.QDialog.Accepted:
       print("newcurve = %s" % str(w.lineEdit.text()))
   else:
       print("keeping old curve")
