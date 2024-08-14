#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaIO import HDF5Stack1D
from PyMca5.PyMcaGui.pymca import QHDF5StackWizard

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
            ret = wizard.exec()
            if ret != qt.QDialog.Accepted:
                raise ValueError("Incomplete selection")
            filelist, selection, scanlist = wizard.getParameters()
        HDF5Stack1D.HDF5Stack1D.__init__(self, filelist, selection,
                                scanlist=scanlist,
                                dtype=dtype)

    def onBegin(self, nfiles):
        self.bars =qt.QWidget()
        self.bars.setWindowTitle("Reading progress")
        self.barsLayout = qt.QGridLayout(self.bars)
        self.barsLayout.setContentsMargins(2, 2, 2, 2)
        self.barsLayout.setSpacing(3)
        self.progressBar   = qt.QProgressBar(self.bars)
        self.progressLabel = qt.QLabel(self.bars)
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
