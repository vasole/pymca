#!/usr/bin/env python
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
import sys
import os
import numpy
import copy
import time
from PyMca5.PyMcaGui import PyMcaQt as qt
from . import McaWindow
from PyMca5.PyMcaIO import EDFStack
from PyMca5.PyMcaIO import SpecFileStack
DEBUG = 0

class SimpleThread(qt.QThread):
    def __init__(self, function, *var, **kw):
        if kw is None:kw={}
        qt.QThread.__init__(self)
        self._function = function
        self._var      = var
        self._kw       = kw
        self._result   = None
    
    def run(self):
        if DEBUG:
            self._result = self._function(*self._var, **self._kw )
        else:
            try:
                self._result = self._function(*self._var, **self._kw )
            except:
                self._result = ("Exception",) + sys.exc_info()
                
class QSpecFileStack(SpecFileStack.SpecFileStack):
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

class QStack(EDFStack.EDFStack):
    def onBegin(self, nfiles):
        self.bars =qt.QWidget()
        self.bars.setWindowTitle("Reading progress")
        self.barsLayout = qt.QGridLayout(self.bars)
        self.barsLayout.setContentsMargins(2, 2, 2, 2)
        self.barsLayout.setSpacing(3)
        self.progressBar   = qt.QProgressBar(self.bars)
        self.progressLabel = qt.QLabel(self.bars)
        self.progressLabel.setText('File Progress:')
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
