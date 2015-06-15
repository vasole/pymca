#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict

DEBUG = 0

class PostEdgeParameters(qt.QGroupBox):
    sigPostEdgeParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        super(PostEdgeParameters, self).__init__(parent)
        self.setTitle("EXAFS")
        self.build()

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
        positionLabel.setText("Position")
        degreeLabel = qt.QLabel(self)
        degreeLabel.setText("Degree")
        self._knotPositions = []
        self._knotDegrees = []
        for i in range(self.knotsBox.maximum()):
            position = qt.QLineEdit(self)
            position._validator = qt.QDoubleValidator(position)
            position.setValidator(position._validator)
            degree = qt.QSpinBox(self)
            degree.setMinimum(1)
            degree.setMaximum(3)
            degree.setValue(3)
            self._knotPositions.append(position)
            self._knotDegrees.append(degree)
            if i >= self.knotsBox.value():
                position.setEnabled(False)
                degree.setEnabled(False)

        # arrange everything
        self.mainLayout.addWidget(kMinLabel, 0, 0)
        self.mainLayout.addWidget(self.kMinBox, 0, 1)
        self.mainLayout.addWidget(kMaxLabel, 1, 0)
        self.mainLayout.addWidget(self.kMaxBox, 1, 1)
        self.mainLayout.addWidget(knotsLabel, 2, 0)
        self.mainLayout.addWidget(self.knotsBox, 2, 1)
        self.mainLayout.addWidget(positionLabel, 3, 0)
        self.mainLayout.addWidget(degreeLabel, 3, 1)

        lastRow = 3
        for i in range(self.knotsBox.maximum()):
            lastRow += 1
            self.mainLayout.addWidget(self._knotPositions[i], lastRow, 0)
            self.mainLayout.addWidget(self._knotDegrees[i], lastRow, 1)

        # connect
        self.kMinBox.valueChanged[float].connect(self._kMinChanged)
        self.kMaxBox.valueChanged[float].connect(self._kMaxChanged)
        self.knotsBox.valueChanged[int].connect(self._knotNumberChanged)
        for i in range(self.knotsBox.maximum()):
            self._knotPositions[i].editingFinished.connect(self._knotEdited)
            self._knotDegrees[i].valueChanged[int].connect(self._degreeChanged)

    def _knotNumberChanged(self, value):
        if DEBUG:
            print("Current number of knots = ", value)
        for i in range(self.knotsBox.maximum()):
            if i < value:
                enabled = True
            else:
                enabled = False
            self._knotPositions[i].setEnabled(enabled)
            self._knotDegrees[i].setEnabled(enabled)
        self._fillKnots()
        self.emitSignal("KnotNumberChanged")

    def _kMinChanged(self, value):
        if DEBUG:
            print("Current kMin Value =", value)
        self._fillKnots()
        self.emitSignal("KMinChanged")

    def _kMaxChanged(self, value):
        if DEBUG:
            print("Current kMax Value =", value)
        self._fillKnots()
        self.emitSignal("KMaxChanged")

    def _knotEdited(self):
        if DEBUG:
            print("One knot has been edited")
        self.emitSignal("KnotPositionChanged")

    def _degreeChanged(self, value):
        if DEBUG:
            print("One knot polynomial degree changed", value)
        self.emitSignal("KnotOrderChanged")

    def getParameters(self):
        ddict = {}
        ddict["KMin"] = self.kMinBox.value()
        ddict["KMax"] = self.kMaxBox.value()
        ddict["Knots"] = {}
        ddict["Knots"]["Number"] = self.knotsBox.value()
        ddict["Knots"]["Values"] = []
        ddict["Knots"]["Orders"] = []
        for i in range(ddict["Knots"]["Number"]):
            txt = str(self._knotPositions[i].text())
            ddict["Knots"]["Values"].append(float(txt))
            ddict["Knots"]["Orders"].append(self._knotDegrees[i].value())
        return ddict

    def setParameters(self, ddict):
        if "EXAFS" in ddict:
            ddict = ddict["EXAFS"]
        elif "PostEdge" in ddict:
            ddict = ddict["PostEdge"]
        self.kMinBox.setValue(ddict.get("KMin", self.kMinBox.value()))
        self.kMaxBox.setValue(ddict.get("KMax", self.kMaxBox.value()))
        nKnots = self.knotsBox.value()
        if "Knots" in ddict:
            self.knotsBox.setValue(ddict["Knots"].get("Number", nKnots))
        nKnots = self.knotsBox.value()
        positions, orders = self._getDefaultKnotValues(knots=nKnots)
        n = len(positions)
        for i in range(self.knotsBox.maximum()):
            if i < n:
                enabled = True
            else:
                enabled = False
            self._knotPositions[i].setEnabled(enabled)
            self._knotDegrees[i].setEnabled(enabled)
        if "Knots" in ddict:
            positions = ddict["Knots"].get("Values", positions)
            orders = ddict["Knots"].get("Orders", orders)
        self._fillKnots(positions, orders)

    def _fillKnots(self, positions=None, orders=None):
        if (positions is None) and (orders is None):
            positions, orders = self._getDefaultKnots()
        n = len(positions)
        for i in range(n):
            self._knotPositions[i].setText("%.3f" % positions[i])
            self._knotDegrees[i].setValue(orders[i])
        for i in range(n, self.knotsBox.maximum()):
            self._knotPositions[i].setText("")

    def _getDefaultKnots(self, kMin=None, kMax=None, knots=None):
        if kMin is None:
            kMin = self.kMinBox.value()
        if kMax is None:
            kMax = self.kMaxBox.value()
        if knots is None:
            knots = self.knotsBox.value()
        positions = []
        degrees = []
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

if __name__ == "__main__":
    DEBUG = 1
    app = qt.QApplication([])
    def testSlot(ddict):
        print("Emitted signal = ", ddict)
    w = PostEdgeParameters()
    w.sigPostEdgeParametersSignal.connect(testSlot)
    w.show()
    app.exec_()
