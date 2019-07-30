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
from . import Object3DQt   as qt
from . import Object3DSlider
from .VerticalSpacer import VerticalSpacer

DEBUG = 0
DRAW_MODES = ['NONE',
              'POINT',
              'WIRE',
              'SURFACE']


class Object3DDrawingModeWidget(qt.QGroupBox):
    sigObject3DDrawingModeSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Drawing Mode')
        self.build()
        self.setDrawingMode(1)

    def build(self):
        self.l = qt.QVBoxLayout(self)
        self.l.setContentsMargins(0, 0, 0, 0)
        self.l.setSpacing(4)
        self.buttonGroup = qt.QButtonGroup(self)
        j = 0
        for mode in DRAW_MODES:
            rButton = qt.QRadioButton(self)
            rButton.setText(mode)
            self.l.addWidget(rButton)
            self.l.setAlignment(rButton, qt.Qt.AlignLeft)
            self.buttonGroup.addButton(rButton)
            self.buttonGroup.setId(rButton, j)
            j += 1
            self.buttonGroup.buttonPressed.connect(self._slot)

    def _slot(self, button):
        button.setChecked(True)
        self._signal()

    def _signal(self, event = None):
        if DEBUG:
            print("emit Object3DDrawingModeSignal")
        if event is None:
            event = 'DrawModeUpdated'
        ddict = self.getParameters()
        ddict['event']  = event
        self.sigObject3DDrawingModeSignal.emit(ddict)

    def getParameters(self):
        mode = self.getDrawingMode()
        ddict = {}
        ddict['mode']   = mode
        ddict['label']  = str(self.buttonGroup.button(mode).text())
        return ddict

    def setParameters(self, ddict = None):
        if DEBUG:
            print("setParameters")
        if ddict is None:
            ddict = {}
        mode = ddict.get('mode', 1)
        self.setDrawingMode(mode)

    def setDrawingMode(self, mode):
        if type(mode) == type(" "):
            if mode.upper() in DRAW_MODES:
                i = DRAW_MODES.index(mode)
            else:
                raise ValueError("Unknown drawing mode: %s "  % mode)
        else:
            i = mode
        self.buttonGroup.button(i).setChecked(True)

    def getDrawingMode(self):
        mode = 0
        n = self.buttonGroup.checkedId()
        if n >= 0:
            mode = n
        else:
            print("WARNING: getAnchor -> Unselected button")
        return mode

    def setSupportedModes(self, modes):
        current = self.getDrawingMode()
        for i in modes:
            if i < len(DRAW_MODES):
                self.buttonGroup.button(i).setEnabled(True)

        # always possible to draw nothing
        self.buttonGroup.button(i).setEnabled(True)

        if not self.buttonGroup.button(current).isEnabled():
            self.buttonGroup.button(0).setChecked(True)
            self._signal()

class Object3DAspect(qt.QGroupBox):
    sigObject3DAspectSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Aspect')
        self.build()

    def build(self):
        self.l = qt.QGridLayout(self)

        i = 0
        # point size
        label = qt.QLabel('Point size')
        self.pointSize = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.pointSize.setRange(0.0, 1.0, 1.0)
        self.pointSize.setValue(1.0)
        self.l.addWidget(label, i, 0)
        self.l.addWidget(self.pointSize, i, 1)
        self.pointSize.valueChanged.connect(self._slot)

        # line width
        i += 1
        label = qt.QLabel('Line width')
        self.lineWidth = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.lineWidth.setRange(0.0, 1.0, 1.0)
        self.lineWidth.setValue(1.0)
        self.l.addWidget(label, i, 0)
        self.l.addWidget(self.lineWidth, i, 1)
        self.lineWidth.valueChanged.connect(self._slot)

        # transparency
        i += 1
        label = qt.QLabel('Transparency')
        self.transparency = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.transparency.setRange(0.0, 1.0, 0.01)
        self.transparency.setValue(0.0)
        self.l.addWidget(label, i, 0)
        self.l.addWidget(self.transparency, i, 1)
        self.transparency.valueChanged.connect(self._slot)

        # bounding box
        self.boundingBoxCheckBox = qt.QCheckBox(self)
        self.boundingBoxCheckBox.setText("Show bounding box")
        self.boundingBoxCheckBox.stateChanged[int].connect(self._signal)

        i = 0
        j = 2
        self.l.addWidget(self.boundingBoxCheckBox, i, j)
        self.showLimitsCheckBoxes = []
        for t in ['X', 'Y', 'Z']:
            i += 1
            checkBox = qt.QCheckBox(self)
            checkBox.setText('Show bbox %s limit' % t)
            self.l.addWidget(checkBox, i, j)
            checkBox.stateChanged[int].connect(self._slot)
            self.showLimitsCheckBoxes.append(checkBox)


    def _slot(self, *var):
        self._signal()

    def getParameters(self):
        pointSize = self.pointSize.value()
        lineWidth = self.lineWidth.value()
        transparency = self.transparency.value()
        if self.boundingBoxCheckBox.isChecked():
            showBBox = 1
        else:
            showBBox = 0
        showLimits   = [0, 0, 0]
        for i in range(3):
            if self.showLimitsCheckBoxes[i].isChecked():
                showLimits[i] = 1
        ddict = {}
        ddict['pointsize'] = pointSize
        ddict['pointsizecapabilities'] = [self.pointSize.minValue(),
                                          self.pointSize.maxValue(),
                                          self.pointSize.step()]
        ddict['linewidth'] = lineWidth
        ddict['linewidthcapabilities'] = [self.lineWidth.minValue(),
                                          self.lineWidth.maxValue(),
                                          self.lineWidth.step()]
        ddict['transparency'] = transparency
        ddict['bboxflag' ]  = showBBox
        ddict['showlimits'] = showLimits
        return ddict

    def setParameters(self, ddict = None):
        if DEBUG:
            print("setParameters")
        if ddict is None:
            ddict = {}
        pointSize = ddict.get('pointsize', 1.0)
        pointSizeCapabilities = ddict.get('pointsizecapabilities',
                                          [1.0, 1.0, 1.0])
        lineWidth = ddict.get('linewidth', 1.0)
        lineWidthCapabilities = ddict.get('linewidthcapabilities',
                                          [1.0, 1.0, 1.0])
        transparency = ddict.get('transparency', 0.0)
        showBBox =  ddict.get('bboxflag', 1)
        showLimits = ddict.get('showlimits', [1, 1, 1])

        self.pointSize.setRange(pointSizeCapabilities[0],
                                pointSizeCapabilities[1],
                                pointSizeCapabilities[2])
        self.pointSize.setValue(pointSize)

        self.lineWidth.setRange(lineWidthCapabilities[0],
                                lineWidthCapabilities[1],
                                lineWidthCapabilities[2])
        self.lineWidth.setValue(lineWidth)
        if lineWidth > lineWidthCapabilities[1]:
            lineWidth = lineWidthCapabilities[1]
        self.transparency.setValue(transparency)


        self.boundingBoxCheckBox.setChecked(showBBox)

        for i in [0, 1, 2]:
            self.showLimitsCheckBoxes[i].setChecked(showLimits[i])

    def _signal(self, event = None):
        if DEBUG:
            print("emitting Object3DAspectSignal")
        if event is None:
            event = "AspectUpdated"
        ddict = self.getParameters()
        ddict['event'] = event
        self.sigObject3DAspectSignal.emit(ddict)

class Object3DScale(qt.QGroupBox):
    sigObject3DScaleSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Object Scaling')
        self.l = qt.QGridLayout(self)
        self.__disconnect = False
        self.__oldScale = [1.0, 1.0, 1.0]

        self.lineEditList = []
        self.validatorList = []
        i = 0
        self._lineSlotList =[self._xLineSlot,
                             self._yLineSlot,
                             self._zLineSlot]
        for axis in ['x', 'y', 'z']:
            label = qt.QLabel("%s Scale" % axis)
            lineEdit = qt.QLineEdit(self)
            v = qt.CLocaleQDoubleValidator(lineEdit)
            lineEdit.setValidator(v)

            self.validatorList.append(v)
            self.l.addWidget(label, i, 0)
            self.l.addWidget(lineEdit, i, 1)
            self.lineEditList.append(lineEdit)
            lineEdit.setText('1.0')
            lineEdit.setFixedWidth(lineEdit.fontMetrics().width('######.#####'))
            lineEdit.editingFinished.connect(self._lineSlotList[i])
            i+= 1

        # xScaling
        i = 0
        self.xScaleSlider = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.xScaleSlider.setRange(-10.0, 10.0, 0.001)
        self.xScaleSlider.setValue(1.0)
        self.l.addWidget(self.xScaleSlider, i, 2)
        self.xScaleSlider.valueChanged.connect(self._xSliderSlot)

        # yScaling
        i += 1
        self.yScaleSlider = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.yScaleSlider.setRange(-100.0, 100.0, 0.01)
        self.yScaleSlider.setValue(1.0)
        self.l.addWidget(self.yScaleSlider, i, 2)
        self.yScaleSlider.valueChanged.connect(self._ySliderSlot)

        # zScaling
        i += 1
        self.zScaleSlider = Object3DSlider.Object3DSlider(self, qt.Qt.Horizontal)
        self.zScaleSlider.setRange(-100.0, 100.0, 0.01)
        self.zScaleSlider.setValue(1.0)
        self.l.addWidget(self.zScaleSlider, i, 2)
        self.zScaleSlider.valueChanged.connect(self._zSliderSlot)

    def _xSliderSlot(self, *var):
        if not self.__disconnect:
            scale = [self.xScaleSlider.value(),
                     self.yScaleSlider.value(),
                     self.zScaleSlider.value()]

            self.__disconnect = True
            for i in [0, 1, 2]:
                if scale[i] != float(str(self.lineEditList[i].text())):
                    self.lineEditList[i].setText("%.7g" % scale[i])
            self.__disconnect = False
            if (self.__oldScale[0] != scale[0]) or \
               (self.__oldScale[1] != scale[1]) or \
               (self.__oldScale[2] != scale[2]) :
                self.__oldScale = scale
                self._signal("xScaleUpdated")

    def _ySliderSlot(self, *var):
        if not self.__disconnect:
            scale = [self.xScaleSlider.value(),
                     self.yScaleSlider.value(),
                     self.zScaleSlider.value()]

            self.__disconnect = True
            for i in [0, 1, 2]:
                if scale[i] != float(str(self.lineEditList[i].text())):
                    self.lineEditList[i].setText("%.7g" % scale[i])
            self.__disconnect = False
            if (self.__oldScale[0] != scale[0]) or \
               (self.__oldScale[1] != scale[1]) or \
               (self.__oldScale[2] != scale[2]) :
                self.__oldScale = scale
                self._signal("yScaleUpdated")

    def _zSliderSlot(self, *var):
        if not self.__disconnect:
            scale = [self.xScaleSlider.value(),
                     self.yScaleSlider.value(),
                     self.zScaleSlider.value()]

            self.__disconnect = True
            for i in [0, 1, 2]:
                if scale[i] != float(str(self.lineEditList[i].text())):
                    self.lineEditList[i].setText("%.7g" % scale[i])
            self.__disconnect = False
            if (self.__oldScale[0] != scale[0]) or \
               (self.__oldScale[1] != scale[1]) or \
               (self.__oldScale[2] != scale[2]) :
                self.__oldScale = scale
                self._signal("zScaleUpdated")

    def _xLineSlot(self):
        if not self.__disconnect:
            self.__disconnect = True
            scale = [1, 1, 1]
            for i in [0, 1 , 2]:
                scale[i] = float(str(self.lineEditList[i].text()))
            self.xScaleSlider.setValue(scale[0])
            self.yScaleSlider.setValue(scale[1])
            self.zScaleSlider.setValue(scale[2])
            self.__disconnect = False
            self._signal("xScaleUpdated")

    def _yLineSlot(self):
        if not self.__disconnect:
            self.__disconnect = True
            scale = [1, 1, 1]
            for i in [0, 1 , 2]:
                scale[i] = float(str(self.lineEditList[i].text()))
            self.xScaleSlider.setValue(scale[0])
            self.yScaleSlider.setValue(scale[1])
            self.zScaleSlider.setValue(scale[2])
            self.__disconnect = False
            self._signal("yScaleUpdated")

    def _zLineSlot(self):
        if not self.__disconnect:
            self.__disconnect = True
            scale = [1, 1, 1]
            for i in [0, 1 , 2]:
                scale[i] = float(str(self.lineEditList[i].text()))
            self.xScaleSlider.setValue(scale[0])
            self.yScaleSlider.setValue(scale[1])
            self.zScaleSlider.setValue(scale[2])
            self.__disconnect = False
            self._signal("zScaleUpdated")

    def _signal(self, event = None):
        if DEBUG:
            print("emitting Object3DScaleSignal")
        if self.__disconnect: return
        if event is None:
            event = "ScaleUpdated"
        oldScale = self._lastParameters * 1
        ddict = self.getParameters()
        scale = ddict['scale']
        emit = False
        for i in range(3):
            if abs((scale[i]-oldScale[i])) > 1.0e-10:
                emit = True
                ddict['magnification'] = scale[i]/oldScale[i]
                break
        if not emit:
            return
        ddict['event'] = event
        self.sigObject3DScaleSignal.emit(ddict)

    def getParameters(self):
        scale = [1.0, 1.0, 1.0]
        for i in [0, 1 , 2]:
            scale[i] = float(str(self.lineEditList[i].text()))

        ddict = {}
        ddict['scale'] = scale
        self._lastParameters = scale
        return ddict

    def setParameters(self, ddict = None):
        if DEBUG:
            print("setParameters", ddict)
        if ddict is None:ddict = {}
        scale = ddict.get('scale', [1.0, 1.0, 1.0])

        self.xScaleSlider.setValue(scale[0])
        self.yScaleSlider.setValue(scale[1])
        self.zScaleSlider.setValue(scale[2])

        for i in [0, 1, 2]:
            self.lineEditList[i].setText("%.7g" % scale[i])
        self._lastParameters = scale

class Object3DPrivateInterface(qt.QGroupBox):
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Private Configuration')
        self.mainLayout = qt.QVBoxLayout(self)
        self.button = qt.QPushButton(self)
        self.button.setText("More")
        self.mainLayout.addWidget(self.button)
        self.mainLayout.addWidget(VerticalSpacer(self))

class Object3DProperties(qt.QWidget):
    sigObject3DPropertiesSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.l = qt.QHBoxLayout(self)
        self.drawingModeWidget = Object3DDrawingModeWidget(self)
        self.aspectWidget = Object3DAspect(self)
        self.privateInterfaceWidget = Object3DPrivateInterface(self)
        self.privateWidget = None
        self.l.addWidget(self.drawingModeWidget)
        self.l.addWidget(self.aspectWidget)
        self.l.addWidget(self.privateInterfaceWidget)

        self.drawingModeWidget.sigObject3DDrawingModeSignal.connect(\
                       self._slot)

        self.aspectWidget.sigObject3DAspectSignal.connect(\
                       self._slot)

        self.privateInterfaceWidget.button.clicked.connect(\
                            self.showPrivateInterface)


    def _slot(self, ddict):
        self._signal(event = ddict['event'])

    def _signal(self, event=None):
        if DEBUG:
            print("emit Object3DPropertiesSignal")
        ddict = self.getParameters()
        if event is None:
            ddict['event'] = "PropertiesUpdated"
        else:
            ddict['event'] = event
        self.sigObject3DPropertiesSignal.emit(ddict)

    def _privateCallBack(self):
        ddict = self.getParameters()
        ddict['event'] = "PropertiesUpdated"
        self.sigObject3DPropertiesSignal.emit(ddict)

    def showPrivateInterface(self):
        if self.privateWidget is not None:
            self.privateWidget.show()
        else:
            text = "Selected object does not have a particular"
            text += " configuration interface"
            qt.QMessageBox.information(self,
                "No private configuration", text)

    def getParameters(self):
        ddict ={}
        ddict['common'] = self.drawingModeWidget.getParameters()
        ddict['common'].update(self.aspectWidget.getParameters())
        if self.privateWidget is None:
            ddict['private'] = {'widget':None}
        else:
            ddict['private'] = self.privateWidget.getParameters()
        return ddict

    def setParameters(self, ddict):
        if DEBUG:
            print("setParameters", ddict)
        self.drawingModeWidget.setParameters(ddict['common'])
        self.aspectWidget.setParameters(ddict['common'])
        widget = ddict['private'].get('widget', None)
        if self.privateWidget is None:
            pass
        else:
            try:
                if self.privateWidget != widget:
                    self.privateWidget.close()
            except ReferenceError:
                if DEBUG:
                    print("Reference error")
                pass
        self.privateWidget = widget
        if widget is not None:
            self.privateWidget.setCallBack(self._privateCallBack)
            self.privateWidget.setParameters(ddict['private'])

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot(ddict):
        print("Signal received")
        print("ddict = ", ddict)

    if 1:
        w = Object3DProperties()
        w.sigObject3DPropertiesSignal.connect(myslot)
    elif 0:
        w = Object3DDrawingModeWidget()
        w.sigObject3DDrawingModeSignal.connect(myslot)
    elif 0:
        w = Object3DAspect()
        w.sigObject3DAspectSignal.connect(myslot)
    elif 1:
        w = Object3DScale()
        w.sigObject3DScaleSignal.connect(myslot)
    w.show()
    app.exec_()
