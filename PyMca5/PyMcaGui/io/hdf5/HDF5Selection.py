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
try:
    from PyMca import PyMcaQt as qt
    safe_str = qt.safe_str
except ImportError:
    import PyQt4.Qt as qt
    safe_str = str
DEBUG = 0

class HDF5Selection(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.selectionWidgetsDict = {}
        row = 0
        for key in ['x', 'y', 'm']:
            label = qt.QLabel(self)
            label.setText(key+":")
            line  = qt.QLineEdit(self)
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
            text = safe_str(self.selectionWidgetsDict[key].text())
            text = text.replace(" ","")
            if len(text):
                selection[key] = text.split(',')
        return selection

def main():
    app = qt.QApplication([])
    tab = HDF5Selection()
    tab.setSelection({'x':[1, 2], 'y':[4], 'cntlist':["dummy", "Cnt0", "Cnt1", "Cnt2", "Cnt3"]})
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main()
    
