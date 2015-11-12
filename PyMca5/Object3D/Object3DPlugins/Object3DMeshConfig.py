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
try:
    from PyMca5.Object3D import Object3DQt as qt
    from PyMca5.Object3D import Object3DPrivateConfig
    from PyMca5.Object3D import PrivateConfigTools
    from PyMca5.Object3D.HorizontalSpacer import HorizontalSpacer
    from PyMca5.Object3D.VerticalSpacer import VerticalSpacer
except ImportError:
    from Object3D import Object3DQt as qt
    from Object3D import Object3DPrivateConfig
    from Object3D import PrivateConfigTools
    from Object3D.HorizontalSpacer import HorizontalSpacer
    from Object3D.VerticalSpacer import VerticalSpacer
import weakref
DEBUG = 0

class Object3DMeshConfig(Object3DPrivateConfig.Object3DPrivateConfig):
    def __init__(self, parent = None, name=""):
        Object3DPrivateConfig.Object3DPrivateConfig.__init__(self, parent, name)
        self._configuration = {}
        self._configuration['widget'] = weakref.proxy(self)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)

        #info
        self.infoLabel = PrivateConfigTools.InfoLabel(self)

        #color filtering
        self.colorFilter = PrivateConfigTools.ColorFilter(self)
        self.colorFilter.sigColorFilterSignal.connect(self.updateCallBack)

        #value filtering
        self.valueFilter = PrivateConfigTools.ValueFilter(self)

        #isosurfaces
        self.isosurfaces = PrivateConfigTools.Isosurfaces(self)


        #actions
        self.updateButton = qt.QPushButton(self)
        self.updateButton.setText("Update")
        self.updateButton.setAutoDefault(False)

        #self.mainLayout.addWidget(HorizontalSpacer(self), 0, 0)
        self.mainLayout.addWidget(self.infoLabel,   0, 0, 1, 4)
        #self.mainLayout.addWidget(HorizontalSpacer(self), 0, 2)
        self.mainLayout.addWidget(self.colorFilter, 1, 0)
        self.mainLayout.addWidget(self.valueFilter, 1, 1)
        self.mainLayout.addWidget(self.isosurfaces, 1, 2)
        self.mainLayout.addWidget(HorizontalSpacer(self), 2, 0)
        self.mainLayout.addWidget(self.updateButton, 2, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self), 2, 3)
        self.mainLayout.addWidget(VerticalSpacer(self), 3, 0)

        #connect
        self.colorFilter.sigColorFilterSignal.connect(self.updateCallBack)
        self.updateButton.clicked.connect(self.updateCallBack)

    def setParameters(self, ddict):
        #if 'widget' in ddict:
        #    del ddict['widget']
        self._configuration.update(ddict)
        self._updateWidget()
        return True

    def _updateWidget(self):
        self.infoLabel.setParameters(self._configuration)
        self.colorFilter.setParameters(self._configuration)
        self.valueFilter.setParameters(self._configuration)
        self.isosurfaces.setParameters(self._configuration)
        return

    def getParameters(self):
        self._configuration.update(self.infoLabel.getParameters())
        self._configuration.update(self.colorFilter.getParameters())
        self._configuration.update(self.valueFilter.getParameters())
        self._configuration.update(self.isosurfaces.getParameters())
        return self._configuration

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot():
        print("Callback called")
    w = Object3DMeshConfig()
    w.setCallBack(myslot)
    w.show()
    app.exec_()
