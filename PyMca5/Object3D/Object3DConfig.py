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
from . import Object3DQt as qt
from . import Object3DMovement
from . import Object3DProperties
from . import ClippingPlaneConfiguration
from . import Object3DColormap
from .VerticalSpacer import VerticalSpacer

class Object3DConfig(qt.QWidget):
    sigObject3DConfigSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)

        #drawing
        self.mainTab = qt.QTabWidget(self)
        self.tabDrawing = qt.QWidget(self.mainTab)
        self.tabDrawing.mainLayout = qt.QVBoxLayout(self.tabDrawing)
        self.tabDrawing.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.tabDrawing.mainLayout.setSpacing(0)
        self.movementsWidget  = Object3DMovement.Object3DMovement(self.tabDrawing)
        self.scaleWidget      = Object3DProperties.Object3DScale(self.tabDrawing)
        self.propertiesWidget = Object3DProperties.Object3DProperties(self.tabDrawing)
        self.tabDrawing.mainLayout.addWidget(self.movementsWidget)
        self.tabDrawing.mainLayout.addWidget(self.scaleWidget)
        self.tabDrawing.mainLayout.addWidget(self.propertiesWidget)
        self.mainTab.addTab(self.tabDrawing, "DRAWING")

        #clipping
        self.tabClipping = qt.QWidget(self.mainTab)
        self.tabClipping.mainLayout = qt.QVBoxLayout(self.tabClipping)
        self.tabClipping.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.tabClipping.mainLayout.setSpacing(0)
        self.clippingPlaneWidget = ClippingPlaneConfiguration.ClippingPlaneWidget(self.tabClipping)
        self.colormapWidget = Object3DColormap.Object3DColormap(self.tabClipping)
        self.tabClipping.mainLayout.addWidget(self.clippingPlaneWidget)
        self.tabClipping.mainLayout.addWidget(self.colormapWidget)
        self.tabClipping.mainLayout.addWidget(VerticalSpacer())
        self.mainTab.addTab(self.tabClipping, "CLIP and COLOR")

        self.mainLayout.addWidget(self.mainTab)

        self.movementsWidget.sigObject3DMovementSignal.connect(\
                     self._movementsSlot)
        self.scaleWidget.sigObject3DScaleSignal.connect(\
                     self._scaleSlot)
        self.propertiesWidget.sigObject3DPropertiesSignal.connect(\
                     self._propertiesSlot)
        self.clippingPlaneWidget.sigClippingPlaneWidgetSignal.connect(\
                     self._clippingPlaneSlot)
        self.colormapWidget.sigObject3DColormapSignal.connect(\
                     self._colormapSlot)

    def _movementsSlot(self, ddict0):
        ddict = {}
        ddict['common'] = ddict0
        ddict['common'].update(self.propertiesWidget.getParameters())
        self._signal(ddict)

    def _scaleSlot(self, ddict0):
        ddict = {}
        event = ddict0['event']
        ddict['common'] = ddict0
        ddict['common'].update(self.scaleWidget.getParameters())

        #get the movement positions
        movementsDict = self.movementsWidget.getParameters()
        ddict['common'].update(movementsDict)
        if event == "xScaleUpdated":
            if ddict['common']['scale'][0] != 0.0:
                ddict['common']['translation'][0] /= ddict0['magnification']
        if event == "yScaleUpdated":
            if ddict['common']['scale'][1] != 0.0:
                ddict['common']['translation'][1] /= ddict0['magnification']
        if event == "zScaleUpdated":
            if ddict['common']['scale'][2] != 0.0:
                ddict['common']['translation'][2] /= ddict0['magnification']
        self.movementsWidget.setParameters(ddict['common'])
        self._signal(ddict)

    def _propertiesSlot(self, ddict):
        ddict['common'].update(self.movementsWidget.getParameters())
        ddict['common'].update(self.scaleWidget.getParameters())
        self._signal(ddict)

    def _clippingPlaneSlot(self, ddict0):
        ddict = {}
        ddict['common'] = ddict0
        self._signal(ddict)

    def _colormapSlot(self, ddict0):
        ddict = {}
        ddict['common'] = ddict0
        self._signal(ddict)

    def _signal(self, ddict):
        self.sigObject3DConfigSignal.emit(ddict)

    def getConfiguration(self):
        ddict = self.propertiesWidget.getParameters()
        ddict['common'].update(self.movementsWidget.getParameters())
        ddict['common'].update(self.scaleWidget.getParameters())
        ddict['common'].update(self.clippingPlaneWidget.getParameters())
        ddict['common'].update(self.colormapWidget.getParameters())
        return ddict

    def setConfiguration(self, ddict):
        self.movementsWidget.setParameters(ddict['common'])
        self.scaleWidget.setParameters(ddict['common'])
        self.propertiesWidget.setParameters(ddict)
        self.clippingPlaneWidget.setParameters(ddict['common'])
        self.colormapWidget.setParameters(ddict['common'])

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot(ddict):
        print("Signal received")
        print("dict = ", ddict)

    w = Object3DConfig()
    w.sigObject3DConfigSignal.connect(myslot)
    w.show()
    app.exec_()
