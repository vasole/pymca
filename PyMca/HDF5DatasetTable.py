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
from PyMca import CloseEventNotifyingWidget
from PyMca import NumpyArrayTableWidget

class HDF5DatasetTable(CloseEventNotifyingWidget.CloseEventNotifyingWidget):
    def __init__(self, parent=None):
        CloseEventNotifyingWidget.CloseEventNotifyingWidget.__init__(self,
                                                                     parent)
        self.mainLayout = QtGui.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.arrayTableWidget =  NumpyArrayTableWidget.NumpyArrayTableWidget(self)
        self.mainLayout.addWidget(self.arrayTableWidget)

    def setDataset(self, dataset):
        self.arrayTableWidget.setArrayData(dataset)

