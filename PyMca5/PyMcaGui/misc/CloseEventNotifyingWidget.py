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
from PyMca5.PyMcaGui import PyMcaQt as qt

class HDFInfoCustomEvent(qt.QEvent):
    def __init__(self, ddict):
        if ddict is None:
            ddict = {}
        self.dict = ddict
        qt.QEvent.__init__(self, qt.QEvent.User)
        
class CloseEventNotifyingWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self._notifyCloseEventToWidget = []

    def notifyCloseEventToWidget(self, widget):
        if widget not in self._notifyCloseEventToWidget:
            self._notifyCloseEventToWidget.append(widget)

    def closeEvent(self, event):
        if len(self._notifyCloseEventToWidget):
            for widget in self._notifyCloseEventToWidget:
                ddict={}
                ddict['event'] = 'closeEventSignal'
                ddict['id']    = id(self)
                newEvent = HDFInfoCustomEvent(ddict)
                qt.QApplication.postEvent(widget,
                                      newEvent)
            self._notifyCloseEventToWidget = []
        return qt.QWidget.closeEvent(self, event)
