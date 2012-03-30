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
import PyQt4.QtGui as QtGui
DEBUG = 0

class HDF5Selection(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.mainLayout = QtGui.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.selectionWidgetsDict = {}
        row = 0
        for key in ['x', 'y', 'm']:
            label = QtGui.QLabel(self)
            label.setText(key+":")
            line  = QtGui.QLineEdit(self)
            line.setReadOnly(True)
            self.mainLayout.addWidget(label, row, 0)
            self.mainLayout.addWidget(line,  row, 1)
            self.selectionWidgetsDict[key] = line
            row += 1

    def setSelection(self, selection):
        if 'cntlist' in selection:
            # "Raw" selection
            cntlist = selection['cntlist']
            for key in ['x', 'y', 'm']:
                if key not in selection:
                    self.selectionWidgetsDict[key].setText("")
                    continue
                n = len(selection[key])
                if not n:
                    self.selectionWidgetsDict[key].setText("")
                    continue
                idx = selection[key][0]
                text = "%s" % cntlist[idx]
                if n > 1:
                    for idx in range(1, n):
                        text += ", %s" % cntlist[selection[key][idx]]
                self.selectionWidgetsDict[key].setText(text)
        else:
            # "Digested" selection
            for key in ['x', 'y', 'm']:
                if key not in selection:
                    self.selectionWidgetsDict[key].setText("")
                    continue
                n = len(selection[key])
                if not n:
                    self.selectionWidgetsDict[key].setText("")
                    continue
                text = "%s" % selection[key][0]
                if n > 1:
                    for idx in range(1, n):
                        text += ", %s" % selection[key][idx]
                self.selectionWidgetsDict[key].setText(text)

    def getSelection(self):
        selection = {}
        for key in ['x', 'y', 'm']:
            selection[key] = []
            text = str(self.selectionWidgetsDict[key].text())
            text = text.replace(" ","")
            if len(text):
                selection[key] = text.split(',')
        return selection

def main():
    app = QtGui.QApplication([])
    tab = HDF5Selection()
    tab.setSelection({'x':[1, 2], 'y':[4], 'cntlist':["dummy", "Cnt0", "Cnt1", "Cnt2", "Cnt3"]})
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main()
    
