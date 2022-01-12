#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict

_logger = logging.getLogger(__name__)


def myFloat(x):
    try:
        return float(x)
    except ValueError:
        if ',' in x:
            try:
                return float(x.replace(',','.'))
            except:
                return float(x)
        elif '.' in x:
            try:
                return float(x.replace('.',','))
            except:
                return float(x)
        else:
            raise

class XASPostEdgeParameters(qt.QGroupBox):
    sigPostEdgeParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, color=None):
        super(XASPostEdgeParameters, self).__init__(parent)
        self.setTitle("EXAFS")
        self.build()
        if color is not None:
            self.setTitleColor(color)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        # k Min
        kMinLabel = qt.QLabel(self)
        kMinLabel.setText("K Min:")
        self.kMinBox = qt.QDoubleSpinBox(self)
        self.kMinBox.setDecimals(2)
        self.kMinBox.setMinimum(0.0)
        self.kMinBox.setValue(2.0)
        self.kMinBox.setSingleStep(0.1)
        self.kMinBox.setEnabled(True)

        # k Max
        kMaxLabel = qt.QLabel(self)
        kMaxLabel.setText("K Max:")
        self.kMaxBox = qt.QDoubleSpinBox(self)
        self.kMaxBox.setDecimals(2)
        self.kMaxBox.setMaximum(25.0)
        self.kMaxBox.setValue(20.0)
        self.kMaxBox.setSingleStep(0.1)
        self.kMaxBox.setEnabled(True)

        # k Weight
        kWeightLabel = qt.QLabel(self)
        kWeightLabel.setText("K Weight:")
        self.kWeightBox = qt.QSpinBox(self)
        self.kWeightBox.setMinimum(0)
        self.kWeightBox.setMaximum(3)
        self.kWeightBox.setValue(0)
        self.kWeightBox.setSingleStep(1)
        self.kWeightBox.setEnabled(True)

        # knots
        knotsLabel = qt.QLabel(self)
        knotsLabel.setText("Knots:")
        self.knotsBox = qt.QSpinBox(self)
        self.knotsBox.setMinimum(0)
        self.knotsBox.setMaximum(5)
        self.knotsBox.setValue(3)
        self.knotsBox.setEnabled(True)

        # table of knots
        positionLabel = qt.QLabel(self)
        positionLabel.setText("Region Start")
        degreeLabel = qt.QLabel(self)
        degreeLabel.setText("Degree")
        self._knotPositions = []
        self._knotDegrees = []
        for i in range(self.knotsBox.maximum()+1):
            position = qt.QDoubleSpinBox(self)
            position.setSingleStep(0.1)
            degree = qt.QSpinBox(self)
            degree.setMinimum(1)
            degree.setMaximum(3)
            degree.setValue(3)
            self._knotPositions.append(position)
            self._knotDegrees.append(degree)
            if i > self.knotsBox.value():
                position.setEnabled(False)
                degree.setEnabled(False)
            if i == 0:
                position.setEnabled(False)

        # arrange everything
        self.mainLayout.addWidget(kMinLabel, 0, 0)
        self.mainLayout.addWidget(self.kMinBox, 0, 1)
        self.mainLayout.addWidget(kMaxLabel, 1, 0)
        self.mainLayout.addWidget(self.kMaxBox, 1, 1)
        self.mainLayout.addWidget(kWeightLabel, 2, 0)
        self.mainLayout.addWidget(self.kWeightBox, 2, 1)
        self.mainLayout.addWidget(knotsLabel, 3, 0)
        self.mainLayout.addWidget(self.knotsBox, 3, 1)
        self.mainLayout.addWidget(positionLabel, 4, 0)
        self.mainLayout.addWidget(degreeLabel, 4, 1)

        lastRow = 4
        for i in range(self.knotsBox.maximum()+1):
            lastRow += 1
            self.mainLayout.addWidget(self._knotPositions[i], lastRow, 0)
            self.mainLayout.addWidget(self._knotDegrees[i], lastRow, 1)

        # initialize values
        self._fillKnots()

        # connect
        self.kMinBox.valueChanged[float].connect(self._kMinChanged)
        self.kMaxBox.valueChanged[float].connect(self._kMaxChanged)
        self.kWeightBox.valueChanged[int].connect(self._kWeightChanged)
        self.knotsBox.valueChanged[int].connect(self._knotNumberChanged)
        for i in range(self.knotsBox.maximum() + 1):
            self._knotPositions[i].valueChanged[float].connect(self._knotChanged)
            self._knotDegrees[i].valueChanged[int].connect(self._degreeChanged)
        self.__connected = True

    def _knotNumberChanged(self, value):
        _logger.debug("Current number of knots = %s" % value)
        oldValue = self.__connected
        self.__connected = False
        try:
            for i in range(self.knotsBox.maximum()+1):
                if i < value+1:
                    enabled = True
                else:
                    enabled = False
                self._knotPositions[i].setEnabled(enabled)
                self._knotDegrees[i].setEnabled(enabled)
            self._knotPositions[0].setEnabled(False)
            self._fillKnots()
        finally:
            self.__connected = oldValue
        if self.__connected:
            self.emitSignal("KnotNumberChanged")

    def _kMinChanged(self, value):
        _logger.debug("Current kMin Value = %s" % value)
        oldValue = self.__connected
        self.__connected = False
        try:
            self._fillKnots()
        finally:
            self.__connected = oldValue
        if self.__connected:
            self.emitSignal("KMinChanged")

    def _kMaxChanged(self, value):
        _logger.debug("Current kMax Value = %s" % value)
        if value <= self.kMinBox.value():
            # I should check if we have the focus prior to
            # raise any error.
            # This situation happens during manual editing
            return
        oldValue = self.__connected
        self.__connected = False
        try:
            self._fillKnots()
        finally:
            self.__connected = oldValue
        if self.__connected:
            self.emitSignal("KMaxChanged")

    def _kWeightChanged(self, value):
        _logger.debug("Current kWeight Value = %s" % value)
        if self.__connected:
            self.emitSignal("KWeightChanged")

    def _knotChanged(self, value):
        _logger.debug("One knot has been changed = %s" % value)
        # adjust limits
        oldValue = self.__connected
        self.__connected = False
        try:
            kMax = self.kMaxBox.value()
            for i in range(self.knotsBox.maximum()+1):
                if self._knotPositions[i].isEnabled():
                    # the first one is never enabled
                    singleStep = self._knotPositions[i].singleStep()
                    minimum = self._knotPositions[i-1].value() + singleStep
                    self._knotPositions[i].setMinimum(minimum)
                    if i < self.knotsBox.maximum():
                        maximum = self._knotPositions[i+1].value() - singleStep
                    else:
                        maximum = kMax
                    self._knotPositions[i].setMaximum(maximum)
                else:
                    self._knotPositions[i].setMaximum(kMax)
        finally:
            self.__connected = oldValue
        if self.__connected:
            self.emitSignal("KnotPositionChanged")

    def _degreeChanged(self, value):
        _logger.debug("One knot polynomial degree changed: %s" % value)
        if self.__connected:
            self.emitSignal("KnotOrderChanged")

    def getParameters(self):
        ddict = {}
        ddict["KMin"] = self.kMinBox.value()
        ddict["KMax"] = self.kMaxBox.value()
        ddict["KWeight"] = self.kWeightBox.value()
        ddict["Knots"] = {}
        ddict["Knots"]["Number"] = self.knotsBox.value()
        ddict["Knots"]["Values"] = []
        ddict["Knots"]["Orders"] = []
        for i in range(ddict["Knots"]["Number"]+1):
            txt = str(self._knotPositions[i].text())
            if i == 0:
                pass
                #ddict["Knots"]["Values"].append(ddict["KMin"])
            else:
                ddict["Knots"]["Values"].append(myFloat(txt))
            ddict["Knots"]["Orders"].append(self._knotDegrees[i].value())
        return ddict

    def setParameters(self, ddict, signal=True):
        _logger.debug("setParameters called: ddict %s, signal %s" % (ddict, signal))
        if "EXAFS" in ddict:
            ddict = ddict["EXAFS"]
        elif "PostEdge" in ddict:
            ddict = ddict["PostEdge"]
        try:
            self.__connected = False
            kMin = ddict.get("KMin", self.kMinBox.value())
            if kMin is not None:
                self.kMinBox.setValue(kMin)
            kMax = ddict.get("KMax", self.kMaxBox.value())
            if kMax is not None:
                self.kMaxBox.setValue(kMax)
            kWeight = ddict.get("KWeight", self.kWeightBox.value())
            if kWeight is not None:
                self.kWeightBox.setValue(kWeight)
            nKnots = self.knotsBox.value()
            if "Knots" in ddict:
                self.knotsBox.setValue(ddict["Knots"].get("Number", nKnots))
            nKnots = self.knotsBox.value()
            positions, orders = self._getDefaultKnots(knots=nKnots)
            n = len(positions)
            for i in range(self.knotsBox.maximum()+1):
                if i < n:
                    enabled = True
                else:
                    enabled = False
                self._knotPositions[i].setEnabled(enabled)
                self._knotDegrees[i].setEnabled(enabled)
            self._knotPositions[0].setEnabled(False)
            if "Knots" in ddict:
                newPositions = ddict["Knots"].get("Values", positions)
                if newPositions is not None:
                    if hasattr(newPositions, "__len__"):
                        positions = newPositions
                    else:
                        positions = [newPositions]
                orders = ddict["Knots"].get("Orders", orders)
                if len(positions) ==  (len(orders) - 1):
                    positions = [self.kMinBox.value()] + list(positions)
            self._fillKnots(positions, orders)
        finally:
            self.__connected = True
        if signal:
            # any signal do the job
            self.emitSignal("KMaxChanged")

    def _fillKnots(self, positions=None, orders=None):
        if (positions is None) and (orders is None):
            positions, orders = self._getDefaultKnots()
        kMin = self.kMinBox.value()
        kMax = self.kMaxBox.value()
        for i in range(self.knotsBox.maximum() + 1):
            self._knotPositions[i].setMinimum(kMin)
            self._knotPositions[i].setMaximum(kMax)
        n = len(positions)
        for i in range(n):
            self._knotPositions[i].setValue(positions[i])
            self._knotDegrees[i].setValue(orders[i])
        for i in range(n, self.knotsBox.maximum()+1):
            self._knotPositions[i].setValue(kMax)

    def _getDefaultKnots(self, kMin=None, kMax=None, knots=None):
        if kMin is None:
            kMin = self.kMinBox.value()
        if kMax is None:
            kMax = self.kMaxBox.value()
        if knots is None:
            knots = self.knotsBox.value()
        positions = [kMin]
        degrees = [3]
        delta = (kMax - kMin) / (knots + 1.0)
        for i in range(knots):
            positions.append(kMin + (i + 1) * delta)
            # here I could do something as function of k
            degrees.append(3)
        return positions, degrees

    def emitSignal(self, event):
        ddict = self.getParameters()
        ddict["event"] = event
        self.sigPostEdgeParametersSignal.emit(ddict)

    def setMaximumK(self, value):
        self.kMaxBox.setMaximum(value)
        current = self.kMaxBox.value()
        if current > (value+0.01):
            self.kMaxBox.setValue(value)

    def setTitleColor(self, color):
        #self.setStyleSheet("QGroupBox {font-weight: bold; color: red;}")
        self.setStyleSheet("QGroupBox {color: %s;}" % color)

if __name__ == "__main__":
    _logger.setLevel(logging.DEBUG)
    app = qt.QApplication([])
    def testSlot(ddict):
        print("Emitted signal = ", ddict)
    w = XASPostEdgeParameters()
    w.sigPostEdgeParametersSignal.connect(testSlot)
    w.show()
    app.exec()
