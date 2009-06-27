#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
from PyQt4 import QtCore, QtGui

icon_first = ["22 22 2 1",
". c None",
"# c #000000",
"......................",
"......................",
".#.................##.",
".#...............####.",
".#.............######.",
".#...........########.",
".#.........##########.",
".#.......############.",
".#.....##############.",
".#...################.",
".#.##################.",
".#.##################.",
".#...################.",
".#.....##############.",
".#.......############.",
".#.........##########.",
".#...........########.",
".#.............######.",
".#...............####.",
".#.................##.",
"......................",
"......................"]

icon_previous = ["22 22 2 1",
". c None",
"# c #000000",
"......................",
"......................",
"...................##.",
".................####.",
"...............######.",
".............########.",
"...........##########.",
".........############.",
".......##############.",
".....################.",
"...##################.",
"...##################.",
".....################.",
".......##############.",
".........############.",
"...........##########.",
".............########.",
"...............######.",
".................####.",
"...................##.",
"......................",
"......................"]

icon_next = ["22 22 2 1",
". c None",
"# c #000000",
"......................",
"......................",
".##...................",
".####.................",
".######...............",
".########.............",
".##########...........",
".############.........",
".##############.......",
".################.....",
".##################...",
".##################...",
".################.....",
".##############.......",
".############.........",
".##########...........",
".########.............",
".######...............",
".####.................",
".##...................",
"......................",
"......................"]

icon_last = ["22 22 2 1",
". c None",
"# c #000000",
"......................",
"......................",
".##.................#.",
".####...............#.",
".######.............#.",
".########...........#.",
".##########.........#.",
".############.......#.",
".##############.....#.",
".################...#.",
".##################.#.",
".##################.#.",
".################...#.",
".##############.....#.",
".############.......#.",
".##########.........#.",
".########...........#.",
".######.............#.",
".####...............#.",
".##.................#.",
"......................",
"......................"]

class HorizontalSpacer(QtGui.QWidget):
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                          QtGui.QSizePolicy.Fixed))

class FrameBrowser(QtGui.QWidget):
    def __init__(self, parent=None, n=1):
        QtGui.QWidget.__init__(self, parent)
        self.mainLayout=QtGui.QHBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.firstButton = QtGui.QPushButton(self)
        self.firstButton.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_first)))
        self.previousButton = QtGui.QPushButton(self)
        self.previousButton.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_previous)))
        self.lineEdit = QtGui.QLineEdit(self)
        self.lineEdit.setFixedWidth(self.lineEdit.fontMetrics().width('%05d' % n))        
        validator = QtGui.QIntValidator(1, n, self.lineEdit)
        self.lineEdit.setText("1")
        self._oldIndex = 0
        self.lineEdit.setValidator(validator)
        self.label = QtGui.QLabel(self)
        self.label.setText("of %d" % n)
        self.nextButton = QtGui.QPushButton(self)
        self.nextButton.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_next)))
        self.lastButton = QtGui.QPushButton(self)
        self.lastButton.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_last)))

        self.mainLayout.addWidget(HorizontalSpacer(self))
        self.mainLayout.addWidget(self.firstButton)
        self.mainLayout.addWidget(self.previousButton)
        self.mainLayout.addWidget(self.lineEdit)
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.nextButton)
        self.mainLayout.addWidget(self.lastButton)
        self.mainLayout.addWidget(HorizontalSpacer(self))

        self.connect(self.firstButton,
                     QtCore.SIGNAL("clicked()"),
                     self._firstClicked)

        self.connect(self.previousButton,
                     QtCore.SIGNAL("clicked()"),
                     self._previousClicked)

        self.connect(self.nextButton,
                     QtCore.SIGNAL("clicked()"),
                     self._nextClicked)


        self.connect(self.lastButton,
                     QtCore.SIGNAL("clicked()"),
                     self._lastClicked)

        self.connect(self.lineEdit,
                     QtCore.SIGNAL("editingFinished()"),
                     self._textChangedSlot)

    def _firstClicked(self):
        self.lineEdit.setText("%d" % self.lineEdit.validator().bottom())
        self._textChangedSlot()

    def _previousClicked(self):
        if self._oldIndex >= self.lineEdit.validator().bottom():
            self.lineEdit.setText("%d" % (self._oldIndex)) 
            self._textChangedSlot()

    def _nextClicked(self):
        if self._oldIndex < (self.lineEdit.validator().top()-1):
            self.lineEdit.setText("%d" % (self._oldIndex+2)) 
            self._textChangedSlot()

    def _lastClicked(self):
        self.lineEdit.setText("%d" % self.lineEdit.validator().top())
        self._textChangedSlot()

    def _textChangedSlot(self):
        txt = str(self.lineEdit.text())
        if not len(txt):
            self.lineEdit.setText("%d" % (self._oldIndex+1))
            return
        newValue = int(txt) - 1
        if newValue == self._oldIndex:
            return
        ddict = {}
        ddict["event"] = "indexChanged"
        ddict["old"]   = self._oldIndex + 1
        self._oldIndex = newValue
        ddict["new"]   = self._oldIndex + 1
        self.emit(QtCore.SIGNAL("indexChanged"), ddict)

    def setLimits(self, first, last):
        bottom = min(first, last)
        top = max(first, last)
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" limits = %d, %d"  % (bottom, top))


    def setNFrames(self, nframes):
        bottom = 1
        top = nframes
        self.lineEdit.validator().setTop(top)
        self.lineEdit.validator().setBottom(bottom)
        self._oldIndex = bottom - 1
        self.lineEdit.setText("%d" % (self._oldIndex + 1))
        self.label.setText(" of %d"  % (top))

    def getCurrentIndex(self):
        return self._oldIndex + 1
        
def test(args):
    app=QtGui.QApplication(args)
    w=FrameBrowser(n=10)
    def slot(ddict):
        print ddict
    QtCore.QObject.connect(w,
                       QtCore.SIGNAL("indexChanged"),
                       slot)
    if len(args) > 1:
        w.setLimits(8, 20)
    w.show()
    app.exec_()
                                    

if __name__=="__main__":
    import sys
    test(sys.argv)
    
