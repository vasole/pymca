#!/usr/bin/env python
###########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be used
# as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################
import sys
import os
import PyMcaQt as qt
from PyMca_Icons import IconDict
import PyMcaDirs
import PyMcaFileDialogs
QTVERSION = qt.qVersion()
HDF5SUPPORT = True

class SimpleFitBatchGUI(qt.QWidget):
    def __init__(self,parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("PyMca Simple Fit Batch GUI")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        self._inputDir   = None
        self._outputDir  = None
        self._fileList   = []
        self._build()

    def _build(self):
        self._listLabel   = qt.QLabel(self)
        self._listLabel.setText("Input File list:")
        self._listView   = qt.QTextEdit(self)
        self._listView.setMaximumHeight(30*self._listLabel.sizeHint().height())
        self._listButton = qt.QPushButton(self)
        self._listButton.setText('Browse')
        self._listButton.setAutoDefault(False)
        self.connect(self._listButton,qt.SIGNAL('clicked()'),self.browseList)
        self.mainLayout.addWidget(self._listLabel,  0, 0)
        self.mainLayout.addWidget(self._listView,   0, 1)
        self.mainLayout.addWidget(self._listButton, 0, 2)

    def browseList(self):
        if self._inputDir is None:
            self._inputDir = PyMcaDirs.inputDir
        elif os.path.exists(self._inputDir):
            PyMcaDirs.inputDir = self._inputDir            
        filetypes  = ["Mca Files (*.mca)",
                      "Edf Files (*.edf)"]
        if HDF5SUPPORT:
            filetypes.append("HDF5 Files(*.nxs *.h5 *.hdf)")
        filetypes.append("SPEC Files (*.spec)")
        filetypes.append("SPEC Files (*.dat)")
        filetypes.append("All files (*)")
        message = "Open a set of files"
        mode = "OPEN"
        getfilter = False
        fileList  = PyMcaFileDialogs.getFileList(self,
                                                 filetypelist=filetypes,
                                                 message=message,
                                                 mode=mode,
                                                 getfilter=False,
                                                 single=False)
        if not len(fileList):
            return
        self._inputDir = os.path.dirname(fileList[0])
        if (QTVERSION < '4.2.0') or (not len(self._fileList)):
            self.setFileList(fileList)
            self.raise_()
            return
        msg = qt.QMessageBox()
        msg.setWindowTitle("Append or replace")
        msg.setIcon(qt.QMessageBox.Information)
        msg.setText("Do you want to delete current file list?")
        msg.setStandardButtons(qt.QMessageBox.Yes|qt.QMessageBox.No)
        answer=msg.exec_()
        if answer == qt.QMessageBox.Yes:
            append = False
        else:
            append = True
        self.setFileList(fileList, append=append)
        self.raise_()

    def setFileList(self, filelist, append=False):
        if filelist is None:
            filelist = []
        if not append:
            self._fileList = []
        self._listView.clear()
        text = ""
        for ffile in self._fileList:
            text += ffile + "\n"
        for ffile in filelist:
            if ffile not in self._fileList:
                self._fileList.append(ffile)
                text += ffile + "\n"
        self._listView.insertPlainText(text)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = SimpleFitBatchGUI()
    w.show()
    app.exec_()
