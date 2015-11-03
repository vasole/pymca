#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
from . import Object3DQt as qt
from .HorizontalSpacer import HorizontalSpacer
from .VerticalSpacer import VerticalSpacer
import weakref
DEBUG = 0

class Object3DPrivateConfig(qt.QWidget):
    def __init__(self, parent = None, name=""):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("%s private configuration" % name)
        self._configuration = {}
        self._configuration['widget'] = weakref.proxy(self)
        self.callBack = None
        self._name = name
        self.build()

    def __del__(self):
        if DEBUG:
            print("Object3DPrivateConfig %s deleted" % self._name)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)
        self.button = qt.QPushButton(self)
        self.button.setText("Test Signal")
        self.mainLayout.addWidget(VerticalSpacer(self), 0, 0, 1, 3)
        self.mainLayout.addWidget(HorizontalSpacer(self), 1, 0)
        self.mainLayout.addWidget(self.button, 1, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self), 1, 2)
        self.mainLayout.addWidget(VerticalSpacer(self), 2, 0, 1, 3)
        self.button.clicked.connect(self.updateCallBack)

    def setParameters(self, ddict):
        for key in ddict.keys():
            if key in self._configuration:
                if key != 'widget':
                    self._configuration[key].update(ddict[key])
        self._updateWidget()
        return True

    def _updateWidget(self):
        return

    def getParameters(self):
        return self._configuration

    def setCallBack(self, f):
        self.callBack = f

    def updateCallBack(self):
        if self.callBack is not None:
            self.callBack()

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot():
        print("Callback called")
    w = Object3DPrivateConfig()
    w.setCallBack(myslot)
    w.show()
    app.exec_()
