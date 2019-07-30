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
from . import Object3DSlider
from .Object3DMovement import Object3DRotationWidget, Object3DTranslationWidget
import numpy
import logging


_logger = logging.getLogger(__name__)


class ClippingPlaneConfiguration(qt.QGroupBox):
    sigClippingPlaneSignal = qt.pyqtSignal(object)

    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle("Clipping Planes: Only points satisfying Ax + By + Cz + D >=0 will be plotted")
        self.__disconnected = False
        self.build()

    def build(self):
        self.l = qt.QGridLayout(self)
        self.l.setContentsMargins(4, 4, 4, 4)
        self.l.setSpacing(2)
        i = 0
        j = 0
        for item in ['Use', 'Ax', 'By', 'Cz', 'D']:
           label = qt.QLabel(self)
           label.setText(item)
           self.l.addWidget(label, i , j)
           j += 1

        self.useList    = []
        self.planeList = []
        self.validatorList = []
        for item in [ 'XY', 'XZ', 'YZ', 'U0']:
            use = qt.QCheckBox(self)
            use.setText(item)
            vector = [use]
            use.clicked.connect(self._signalClickedSlot)
            for k in [0, 1, 2, 3]:
                line = qt.QLineEdit(self)
                line.setFixedWidth(line.fontMetrics().width('########'))
                line.setText('0.0')
                line.setReadOnly(True)
                v = qt.CLocaleQDoubleValidator(line)
                line.setValidator(v)
                self.validatorList.append(v)
                vector.append(line)
            if item != "U0":
                slider = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
                slider.setRange(-100, 100, 0.05)
                slider.setValue(0.0)
                slider.valueChanged.connect(self._sliderSlot)

                vector.append(slider)
            self.planeList.append(vector)

        #XY plane
        #vector and point
        self.planeList[0][3].setText('1.0')
        self.planeList[0][3].setReadOnly(False)
        self.planeList[0][4].setReadOnly(False)

        #XZ plane
        #vector and point
        self.planeList[1][2].setText('1.0')
        self.planeList[1][2].setReadOnly(False)
        self.planeList[1][4].setReadOnly(False)

        #YZ plane
        #vector and point
        self.planeList[2][1].setText('1.0')
        self.planeList[2][1].setReadOnly(False)
        self.planeList[2][4].setReadOnly(False)

        for i in range(3):
            self.planeList[i][4].editingFinished.connect(self._lineSlot)
        for p in self.planeList:
            i += 1
            for j in range(len(p)):
                self.l.addWidget(p[j], i, j)

    def _sliderSlot(self, *var):
        if self.__disconnected: return
        _logger.debug("sliderSlot")
        for i in range(3):
            value = self.planeList[i][5].value()
            self.planeList[i][4].setText("%f" % value)

        self._signal()

    def _lineSlot(self):
        _logger.debug("lineSlot")
        for i in range(3):
            oldValue = self.planeList[i][5].value()
            value = float(str(self.planeList[i][4].text()))
            if (oldValue-value) != 0.0:
                self.planeList[i][5].setValue(value)

    def getParameters(self):
        ddict = {}
        ddict['clippingplanes'] = []
        for plane in self.planeList:
            use = plane[0].isChecked()
            A   = float(str(plane[1].text()))
            B   = float(str(plane[2].text()))
            C   = float(str(plane[3].text()))
            D   = float(str(plane[4].text()))
            ddict['clippingplanes'].append([use, A, B, C, D])
        return ddict

    def setParameters(self, ddict=None):
        if ddict is None:return
        if 'limits' in ddict:
            xmin, ymin, zmin = ddict['limits'][0]
            xmax, ymax, zmax = ddict['limits'][1]
            vxmax = max(xmax, xmin)
            vymax = max(ymax, ymin)
            vzmax = max(zmax, zmin)
            self.planeList[0][5].setRange(-abs(vzmax), abs(vzmax))
            self.planeList[1][5].setRange(-abs(vymax), abs(vymax))
            self.planeList[2][5].setRange(-abs(vxmax), abs(vxmax))
        if 'clippingplanes' in ddict:
            i = 0
            self.__disconnected = True
            for plane in ddict['clippingplanes']:
                self.planeList[i][0].setChecked(plane[0])
                for j in [1, 2, 3, 4]:
                    self.planeList[i][j].setText("%f"% plane[j])
                i += 1
            self.__disconnected = False

    def _signalClickedSlot(self):
        self._signal()

    def _signal(self, event = None):
        if event is None:
            event = "ClippingPlaneUpdated"
        ddict = self.getParameters()
        ddict['event'] = event
        self.sigClippingPlaneSignal.emit(ddict)

class UserClippingPlaneWidget(qt.QWidget):

    sigUserClippingPlaneSignal= qt.pyqtSignal(object)

    def __init__(self, parent = None, vector = None, point = None, rotation=None):
        qt.QWidget.__init__(self, parent)
        self.l = qt.QHBoxLayout(self)
        self.l.setContentsMargins(0, 0, 0, 0)
        self.l.setSpacing(0)
        if vector is None: vector = [0.0, 0.0, 1.0]
        self.vectorWidget = Object3DTranslationWidget(self, vector, labels = ['Axis', 'Value'])
        self.vectorWidget.setTitle("U0 Plane Vector")
        text  = "Normal vector to the U0 Plane\n"
        text += "prior to rotations"
        self.vectorWidget.setToolTip(text)
        self.pointWidget  = Object3DTranslationWidget(self, point,  ['Axis', 'Value'])
        self.pointWidget.setTitle("U0 Plane Point")
        text  = "Coordinates of a U0 Plane point\n"
        text += "in object coordinates"
        self.pointWidget.setToolTip(text)
        self.rotationWidget = Object3DRotationWidget(self, rotation)
        self.rotationWidget.setTitle("U0 Plane Rotation")
        self.l.addWidget(self.vectorWidget)
        self.l.addWidget(self.pointWidget)
        self.l.addWidget(self.rotationWidget)
        self.__disconnected = False
        self.vectorWidget.sigObject3DTranslationSignal.connect(self._slot)
        self.pointWidget.sigObject3DTranslationSignal.connect(self._slot)
        self.rotationWidget.sigObject3DRotationSignal.connect(self._slot)

    def _slot(self, ddict):
        self._emitSignal()

    def getParameters(self):
        ddict={}
        ddict['U0vector']   = self.vectorWidget.getTranslation()
        ddict['U0point']    = self.pointWidget.getTranslation()
        ddict['U0rotation'] = self.rotationWidget.getRotation()
        return ddict

    def setParameters(self, ddict):
        self.__disconnected = True
        if 'U0vector' in ddict:
            self.vectorWidget.setTranslation(ddict['U0vector'])
        if 'U0point' in ddict:
            self.pointWidget.setTranslation(ddict['U0point'])
        if 'U0rotation' in ddict:
            self.rotationWidget.setRotation(ddict['U0rotation'])
        self.__disconnected = False

    def _emitSignal(self, event=None):
        if self.__disconnected: return
        _logger.debug("Emitting UserClippingPlaneSignal")
        if event is None:
            event="U0PlaneUpdated"
        ddict = self.getParameters()
        ddict['event'] = event
        self.sigUserClippingPlaneSignal.emit(ddict)

class ClippingPlaneWidget(qt.QWidget):
    sigClippingPlaneWidgetSignal = qt.pyqtSignal(object)

    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.standardClippingPlane = ClippingPlaneConfiguration(self)
        self.userClippingPlane = UserClippingPlaneWidget(self)
        self.mainLayout.addWidget(self.standardClippingPlane)
        self.mainLayout.addWidget(self.userClippingPlane)

        self.standardClippingPlane.sigClippingPlaneSignal.connect(\
                            self._emitSignal)

        self.userClippingPlane.sigUserClippingPlaneSignal.connect(\
                     self._userPlaneSlot)

    def _userPlaneSlot(self, ddict):
        x, y, z = ddict['U0vector']
        if numpy.sqrt(x*x + y*y + z*z) == 0.0:
            if 0:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText('Invalid Normal Vector. Module = 0')
                msg.exec_()
                return
        equation = self.calculatePlaneEquation(ddict['U0vector'],
                                               ddict['U0point'],
                                               ddict['U0rotation'])

        ddict = self.standardClippingPlane.getParameters()
        ddict['clippingplanes'][-1][1] = equation [0]
        ddict['clippingplanes'][-1][2] = equation [1]
        ddict['clippingplanes'][-1][3] = equation [2]
        ddict['clippingplanes'][-1][4] = equation [3]
        self.standardClippingPlane.setParameters(ddict)
        self._emitSignal()

    def calculatePlaneEquation(self, vector, point, rotation):
        if (rotation[0] != 0.0) or (rotation[1] != 0.0) or (rotation[2] != 0):
            #RotX
            angle = rotation[0]*numpy.pi/180.
            cs = numpy.cos(angle)
            sn = numpy.sin(angle)
            rotX = numpy.zeros((3,3), numpy.float64)
            rotX[0,0] =  1
            rotX[1,1] =  1
            rotX[2,2] =  1
            rotX[1,1] =  cs; rotX[1,2] = sn
            rotX[2,1] = -sn; rotX[2,2] = cs

            #RotY
            angle = rotation[1]*numpy.pi/180.
            cs = numpy.cos(angle)
            sn = numpy.sin(angle)
            rotY = numpy.zeros((3,3), numpy.float64)
            rotY[0,0] =  1
            rotY[1,1] =  1
            rotY[2,2] =  1
            rotY[0,0] =  cs; rotY[0,2] = -sn   #inverted respect to the others
            rotY[2,0] =  sn; rotY[2,2] =  cs

            #RotZ
            angle = rotation[2]*numpy.pi/180.
            cs = numpy.cos(angle)
            sn = numpy.sin(angle)
            rotZ = numpy.zeros((3,3), numpy.float64)
            rotZ[0,0] =  1
            rotZ[1,1] =  1
            rotZ[2,2] =  1
            rotZ[0,0] =  cs; rotZ[0,1] = sn
            rotZ[1,0] = -sn; rotZ[1,1] = cs

            #The final matrix
            rotMatrix = numpy.dot(rotZ,numpy.dot(rotY, rotX))
        else:
            rotMatrix = numpy.zeros((3,3), numpy.float64)
            rotMatrix[0,0] =  1
            rotMatrix[1,1] =  1
            rotMatrix[2,2] =  1

        #ABC
        A, B, C = numpy.dot(rotMatrix, numpy.array(vector)).tolist()

        #calculate D
        D = -(A * point[0] +\
              B * point[1] +\
              C * point[2] )

        return [A, B, C, D]

    def _emitSignal(self, event = None):
        _logger.debug("Emitting ClippingPlaneWidgetSignal")
        if event is None:
            event = 'ClippingPlaneWidgetUpdated'
        ddict = self.standardClippingPlane.getParameters()
        ddict.update(self.userClippingPlane.getParameters())
        ddict['event'] = event
        self.sigClippingPlaneWidgetSignal.emit(ddict)

    def setParameters(self, ddict):
        self.standardClippingPlane.setParameters(ddict)
        self.userClippingPlane.setParameters(ddict)

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot(ddict):
        print("Signal received")
        print("ddict      = %s" % ddict)
    if 0:
        w = ClippingPlaneConfiguration()
        w.sigClippingPlaneSignal.connect(myslot)
    elif 1:
        w = ClippingPlaneWidget()
        w.sigClippingPlaneWidgetSignal.connect(myslot)
    w.show()
    app.exec_()
