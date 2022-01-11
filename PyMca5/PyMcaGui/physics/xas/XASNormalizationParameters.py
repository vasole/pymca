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
import os
import sys
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
from PyMca5.PyMcaGui import XASNormalizationWindow
from PyMca5.PyMca import XASNormalization
IconDict = PyMca_Icons.IconDict

_logger = logging.getLogger(__name__)


class XASNormalizationParameters(qt.QGroupBox):
    sigNormalizationParametersSignal = qt.pyqtSignal(object)
    def __init__(self, parent=None, color=None):
        super(XASNormalizationParameters, self).__init__(parent)
        self.setTitle("Normalization")
        self._dialog = None
        self._energy = None
        self._mu = None
        self.__connected = True
        self.build()
        if color is not None:
            self.setTitleColor(color)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        # the normalization method
        normalizationLabel = qt.QLabel(self)
        normalizationLabel.setText("Method:")
        self.normalizationOptions = ["Constant", "Flattened"]
        self.normalizationSelector = qt.QComboBox(self)
        for option in self.normalizationOptions:
            self.normalizationSelector.addItem(option)
        self.normalizationSelector.setCurrentIndex(1)

        # the E0 value
        self.e0CheckBox = qt.QCheckBox(self)
        self.e0CheckBox.setText("Auto E0:")
        self.e0CheckBox.setChecked(True)
        self.e0SpinBox = qt.QDoubleSpinBox(self)
        self.e0SpinBox.setMinimum(200.)
        self.e0SpinBox.setMaximum(200000.)
        self.e0SpinBox.setDecimals(2)
        self.e0SpinBox.setSingleStep(0.2)
        self.e0SpinBox.setEnabled(False)

        # the jump
        jumpLabel = qt.QLabel(self)
        jumpLabel.setText("Jump:")
        self.jumpLine = qt.QLineEdit(self)
        self.jumpLine.setEnabled(False)

        # the pre-edge
        preEdgeLabel = qt.QLabel(self)
        preEdgeLabel.setText("Pre-Edge")
        self.preEdgeSelector = XASNormalizationWindow.PolynomSelector(self)
        self.preEdgeSelector.setCurrentIndex(3)

        # pre-edge regions
        preEdgeStartLabel = qt.QLabel(self)
        preEdgeStartLabel.setText("Begin:")
        self.preEdgeStartBox = qt.QDoubleSpinBox(self)
        self.preEdgeStartBox.setDecimals(2)
        self.preEdgeStartBox.setMinimum(-2000.0)
        self.preEdgeStartBox.setMaximum(-5.0)
        self.preEdgeStartBox.setValue(-100)
        self.preEdgeStartBox.setSingleStep(5.0)
        self.preEdgeStartBox.setEnabled(True)

        preEdgeEndLabel = qt.QLabel(self)
        preEdgeEndLabel.setText("End:")
        self.preEdgeEndBox = qt.QDoubleSpinBox(self)
        self.preEdgeEndBox.setDecimals(2)
        self.preEdgeEndBox.setMinimum(-200.0)
        self.preEdgeEndBox.setMaximum(-1.0)
        self.preEdgeEndBox.setValue(-40)
        self.preEdgeEndBox.setSingleStep(5.0)
        self.preEdgeEndBox.setEnabled(True)

        # the post-edge
        postEdgeLabel = qt.QLabel(self)
        postEdgeLabel.setText("Post-Edge")
        self.postEdgeSelector = XASNormalizationWindow.PolynomSelector(self)
        self.postEdgeSelector.setCurrentIndex(3)

        # post-edge regions
        postEdgeStartLabel = qt.QLabel(self)
        postEdgeStartLabel.setText("Begin:")
        self.postEdgeStartBox = qt.QDoubleSpinBox(self)
        self.postEdgeStartBox.setDecimals(2)
        self.postEdgeStartBox.setMinimum(1.0)
        self.postEdgeStartBox.setMaximum(3000.0)
        self.postEdgeStartBox.setValue(10)
        self.postEdgeStartBox.setSingleStep(5.0)
        self.postEdgeStartBox.setEnabled(True)

        postEdgeEndLabel = qt.QLabel(self)
        postEdgeEndLabel.setText("End:")
        self.postEdgeEndBox = qt.QDoubleSpinBox(self)
        self.postEdgeEndBox.setDecimals(2)
        self.postEdgeEndBox.setMinimum(10.0)
        self.postEdgeEndBox.setMaximum(2000.0)
        self.postEdgeEndBox.setValue(300)
        self.postEdgeEndBox.setSingleStep(5.0)
        self.postEdgeEndBox.setEnabled(True)

        # arrange everything
        self.mainLayout.addWidget(normalizationLabel, 0, 0)
        self.mainLayout.addWidget(self.normalizationSelector, 0, 1)
        self.mainLayout.addWidget(self.e0CheckBox, 1, 0)
        self.mainLayout.addWidget(self.e0SpinBox, 1, 1)
        self.mainLayout.addWidget(jumpLabel, 2, 0)
        self.mainLayout.addWidget(self.jumpLine, 2, 1)

        self.mainLayout.addWidget(preEdgeLabel, 3, 0)
        self.mainLayout.addWidget(self.preEdgeSelector, 3, 1)
        self.mainLayout.addWidget(preEdgeStartLabel, 4, 0)
        self.mainLayout.addWidget(self.preEdgeStartBox, 4, 1)
        self.mainLayout.addWidget(preEdgeEndLabel, 5, 0)
        self.mainLayout.addWidget(self.preEdgeEndBox, 5, 1)

        self.mainLayout.addWidget(postEdgeLabel, 6, 0)
        self.mainLayout.addWidget(self.postEdgeSelector, 6, 1)
        self.mainLayout.addWidget(postEdgeStartLabel, 7, 0)
        self.mainLayout.addWidget(self.postEdgeStartBox, 7, 1)
        self.mainLayout.addWidget(postEdgeEndLabel, 8, 0)
        self.mainLayout.addWidget(self.postEdgeEndBox, 8, 1)

        # connect
        self.normalizationSelector.activated[int].connect(self._normalizationChanged)
        self.e0CheckBox.toggled.connect(self._e0Toggled)
        self.e0SpinBox.valueChanged[float].connect(self._e0Changed)
        self.preEdgeSelector.activated[int].connect(self._preEdgeChanged)
        self.preEdgeStartBox.valueChanged[float].connect(self._preEdgeStartChanged)
        self.preEdgeEndBox.valueChanged[float].connect(self._preEdgeEndChanged)
        self.postEdgeSelector.activated[int].connect(self._postEdgeChanged)
        self.postEdgeStartBox.valueChanged[float].connect(self._postEdgeStartChanged)
        self.postEdgeEndBox.valueChanged[float].connect(self._postEdgeEndChanged)

    def _normalizationChanged(self, value):
        _logger.debug("_normalizationChanged, %s " % value)
        if self.__connected:
            self._emitSignal("JumpNormalizationChanged")

    def setSpectrum(self, energy, mu):
        # try to detect keV
        if abs(energy[-1]-energy[0]) < 10:
            self._energy = energy * 1000.
        else:
            self._energy = energy * 1.0
        self._mu = mu
        try:
            self.__connected = False
            self._update()
        finally:
            self.__connected = True
        self._emitSignal("SpectrumChanged")

    def _calculateE0(self):
        return XASNormalization.getE0SavitzkyGolay(self._energy, self._mu,
                                                   points=5, full=False)
    def _e0Toggled(self, state):
        if state:
            self.e0SpinBox.setEnabled(False)
            if self._mu is not None:
                e0 = self._calculateE0()
                self.e0SpinBox.setValue(e0)
        else:
            self.e0SpinBox.setEnabled(True)

    def _e0Changed(self, value):
        _logger.debug("E0 CHANGED, %s" % value)
        if self.__connected:
            try:
                self.__connected = False
                self._update()
            finally:
                self.__connected = True
            self._emitSignal("E0Changed")

    def _preEdgeChanged(self, value):
        _logger.debug("Current pre-edge value = %s" % value)
        if self.__connected:
            self._emitSignal("PreEdgeChanged")

    def _preEdgeStartChanged(self, value):
        _logger.debug("pre start changed: %s" % value)
        if self.__connected:
            try:
                self.__connected = False
                self._update()
            finally:
                self.__connected = True
            self._emitSignal("PreEdgeChanged")

    def _preEdgeEndChanged(self, value):
        _logger.debug("pre end changed: %s" % value)
        if self.__connected:
            try:
                self.__connected = False
                self._update()
            finally:
                self.__connected = True
            self._emitSignal("PreEdgeChanged")

    def _postEdgeChanged(self, value):
        _logger.debug("post-edge changed: %s" % value)
        if self.__connected:
            self._emitSignal("PostEdgeChanged")

    def _postEdgeStartChanged(self, value):
        _logger.debug("post-edge start changed: %s" % value)
        if self.__connected:
            try:
                self.__connected = False
                self._update()
            finally:
                self.__connected = True
            self._emitSignal("PostEdgeChanged")

    def _postEdgeEndChanged(self, value):
        _logger.debug("post-edge changed: %s" % value)
        if self.__connected:
            try:
                self.__connected = False
                self._update()
            finally:
                self.__connected = True
            self._emitSignal("PostEdgeChanged")

    def _update(self):
        if self._energy is None:
            return
        eMin = self._energy.min()
        eMax = self._energy.max()
        current = self.getParameters()

        if current["E0Method"].lower().startswith("auto") or \
           current["E0Value"] < self._energy.min()  or \
           current["E0Value"] > self._energy.max():
            energy = self._calculateE0()
            current["E0Value"] = energy

        # energy
        e0 = current["E0Value"]
        self.e0SpinBox.setValue(e0)

        # pre-edge
        start = e0 + current["PreEdge"]["Regions"][0]
        end = e0 + current["PreEdge"]["Regions"][-1]

        if start > end:
            start, end = end, start
        start = max(start, eMin)
        if end <= start:
            end = 0.5 * (start + energy)

        self.preEdgeStartBox.setValue(start - e0)
        self.preEdgeEndBox.setValue(end - e0)

        # post-edge
        start = e0 + current["PostEdge"]["Regions"][0]
        end = e0 + current["PostEdge"]["Regions"][-1]

        if start > end:
            start, end = end, start
        end = min(end, eMax)
        if end <= start:
            start = 0.5 * (end + energy)

        self.postEdgeStartBox.setValue(start - e0)
        self.postEdgeEndBox.setValue(end - e0)

    def getParameters(self):
        ddict = {}
        # normalization method
        ddict["JumpNormalizationMethod"] = str(self.normalizationSelector.currentText())

        # default values not yet handled by the interface
        ddict["E0MinValue"] = None
        ddict["E0MaxValue"] = None

        if self.e0CheckBox.isChecked():
            ddict["E0Method"] = "Auto - 5pt SG"
        else:
            ddict["E0Method"] = "Manual"
        ddict["E0Value"] = self.e0SpinBox.value()

        # pre-edge
        ddict["PreEdge"] = {}
        ddict["PreEdge"] ["Method"] = "Polynomial"
        ddict["PreEdge"] ["Polynomial"] = str(self.preEdgeSelector.currentText())
        # Regions is a single list with 2 * n values delimiting n regions.
        ddict["PreEdge"] ["Regions"] = [self.preEdgeStartBox.value(),
                                        self.preEdgeEndBox.value()]
        ddict["PostEdge"] = {}
        ddict["PostEdge"] ["Method"] = "Polynomial"
        ddict["PostEdge"] ["Polynomial"] = str(self.postEdgeSelector.currentText())
        ddict["PostEdge"] ["Regions"] = [self.postEdgeStartBox.value(),
                                         self.postEdgeEndBox.value()]
        return ddict

    def setParameters(self, ddict, signal=True):
        _logger.debug("setParameters called, %s %s" % (ddict, signal))
        if "Normalization" in ddict:
            ddict = ddict["Normalization"]
        try:
            self.__connected = False
            if "JumpNormalizationMethod" in ddict:
                option = ddict["JumpNormalizationMethod"]
                if type(ddict["JumpNormalizationMethod"]) == type(1):
                    self.normalizationSelector.setCurrentIndex(option)
                else:
                    selectorOptions = []
                    for i in range(self.normalizationSelector.count()):
                        selectorOptions.append(str(self.normalizationSelector.itemText(i)))
                    for i in range(len(selectorOptions)):
                        if selectorOptions[i].lower().startswith(str(option).lower()):
                            self.normalizationSelector.setCurrentIndex(i)
                            break
            if ddict["E0Value"] is None:
                self.e0CheckBox.setChecked(True)
            else:
                self.e0SpinBox.setValue(ddict["E0Value"])
                if ddict["E0Method"].lower().startswith("manual"):
                    self.e0CheckBox.setChecked(False)
                else:
                    self.e0CheckBox.setChecked(True)
            selectorOptions = self.preEdgeSelector.getOptions()
            i = 0
            for option in selectorOptions:
                if str(option) == str(ddict["PreEdge"] ["Polynomial"]):
                    self.preEdgeSelector.setCurrentIndex(i)
                    break
                i += 1
            selectorOptions = self.postEdgeSelector.getOptions()
            i = 0
            for option in selectorOptions:
                if str(option) == str(ddict["PostEdge"] ["Polynomial"]):
                    self.postEdgeSelector.setCurrentIndex(i)
                    break
                i += 1
            self.preEdgeStartBox.setValue(ddict["PreEdge"]["Regions"][0])
            self.preEdgeEndBox.setValue(ddict["PreEdge"]["Regions"][-1])
            self.postEdgeStartBox.setValue(ddict["PostEdge"]["Regions"][0])
            self.postEdgeEndBox.setValue(ddict["PostEdge"]["Regions"][-1])
            self._update()
        finally:
            self.__connected = True
        if signal:
            # E0Changed or SpectrumUpdated
            self._emitSignal("E0Changed")

    def _emitSignal(self, event):
        ddict = self.getParameters()
        ddict["event"] = event
        self.jumpLine.setText("")
        self.sigNormalizationParametersSignal.emit(ddict)

    def setJump(self, value):
        self.jumpLine.setText("%f" % value)

    def setTitleColor(self, color):
        #self.setStyleSheet("QGroupBox {font-weight: bold; color: red;}")
        self.setStyleSheet("QGroupBox {color: %s;}" % color)

if __name__ == "__main__":
    _logger.setLevel(logging.DEBUG)
    app = qt.QApplication([])
    def mySlot(ddict):
        print("Signal received: ", ddict)
    w = XASNormalizationParameters()
    w.show()
    w.sigNormalizationParametersSignal.connect(mySlot)
    from PyMca5.PyMcaIO import specfilewrapper as specfile
    from PyMca5.PyMcaDataDir import PYMCA_DATA_DIR
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        fileName = os.path.join(PYMCA_DATA_DIR, "EXAFS_Cu.dat")
    data = specfile.Specfile(fileName)[0].data()[-2:, :]
    energy = data[0, :]
    mu = data[1, :]
    w.setSpectrum(energy, mu)
    w.setTitleColor("blue")
    app.exec()
