#/*##########################################################################
# Copyright (C) 2004-2011 European Synchrotron Radiation Facility
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
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
from PyMca import HDF5Stack1D
from PyMca import QHDF5StackWizard
DEBUG = 0

class QHDF5Stack1D(HDF5Stack1D.HDF5Stack1D):
    def __init__(self, filelist=None,
                       selection=None,
                       scanlist=None,
                       dtype=None):
        if (filelist is None) or (selection is None):
            wizard = QHDF5StackWizard.QHDF5StackWizard()
            if filelist is not None:
                wizard.setFileList(filelist)
                wizard.setStartId(1)
            ret = wizard.exec_()
            if ret != QtGui.QDialog.Accepted:
                raise ValueError("Incomplete selection")
            filelist, selection, scanlist = wizard.getParameters()
        HDF5Stack1D.HDF5Stack1D.__init__(self, filelist, selection,
                                scanlist=scanlist,
                                dtype=dtype)

    def onBegin(self, nfiles):
        self.bars =QtGui.QWidget()
        self.bars.setWindowTitle("Reading progress")
        self.barsLayout = QtGui.QGridLayout(self.bars)
        self.barsLayout.setMargin(2)
        self.barsLayout.setSpacing(3)
        self.progressBar   = QtGui.QProgressBar(self.bars)
        self.progressLabel = QtGui.QLabel(self.bars)
        self.progressLabel.setText('Mca Progress:')
        self.barsLayout.addWidget(self.progressLabel,0,0)        
        self.barsLayout.addWidget(self.progressBar,0,1)
        self.progressBar.setMaximum(nfiles)
        self.progressBar.setValue(0)
        self.bars.show()

    def onProgress(self,index):
        self.progressBar.setValue(index)

    def onEnd(self):
        self.bars.hide()
        del self.bars
