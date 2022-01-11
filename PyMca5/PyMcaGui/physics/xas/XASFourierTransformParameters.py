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


class XASFourierTransformParameters(qt.QGroupBox):
    sigFTParametersSignal = qt.pyqtSignal(object)
    def __init__(self, parent=None):
        super(XASFourierTransformParameters, self).__init__(parent)
        self.setTitle("Fourier Transform")
        self.__connected = True
        self.build()
        config = {}
        config["FT"] = {}
        ddict = config["FT"]
        ddict["Window"] = "Gaussian"
        ddict["WindowList"] = ["Gaussian", "Hanning", "Box", "Parzen",
                               "Welch", "Hamming", "Tukey", "Papul", "Kaiser"]
        ddict["WindowApodization"] = 0.02
        ddict["WindowRange"] = None
        ddict["KStep"] = 0.04
        ddict["Points"] = 2048
        ddict["Range"] = [0.0, 7.0]
        self.setParameters(config)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        # window selector
        windowLabel = qt.QLabel(self)
        windowLabel.setText("Window:")
        windowOptions = ["Gaussian", "Hanning", "Box", "Parzen",
                         "Welch", "Hamming", "Tukey", "Papul", "Kaiser"]
        self.windowSelector = qt.QComboBox(self)
        for option in windowOptions:
            self.windowSelector.addItem(option)
        self.windowSelector.setCurrentIndex(0)

        # the apodization value
        apodizationLabel = qt.QLabel(self)
        apodizationLabel.setText("Apodization:")
        self.apodizationBox = qt.QDoubleSpinBox(self)
        self.apodizationBox.setMinimum(0.001)
        self.apodizationBox.setMaximum(4.)
        self.apodizationBox.setDecimals(3)
        self.apodizationBox.setSingleStep(0.01)
        self.apodizationBox.setValue(0.02)
        self.apodizationBox.setEnabled(False)

        # the window range
        # k Min
        kMinLabel = qt.QLabel(self)
        kMinLabel.setText("Window K Min:")
        self.kMinBox = qt.QDoubleSpinBox(self)
        self.kMinBox.setDecimals(2)
        self.kMinBox.setMinimum(0.0)
        self.kMinBox.setValue(2.0)
        self.kMinBox.setSingleStep(0.1)
        self.kMinBox.setEnabled(True)

        # k Max
        kMaxLabel = qt.QLabel(self)
        kMaxLabel.setText("Window K Max:")
        self.kMaxBox = qt.QDoubleSpinBox(self)
        self.kMaxBox.setDecimals(2)
        self.kMaxBox.setMaximum(25.0)
        self.kMaxBox.setValue(20.0)
        self.kMaxBox.setSingleStep(0.1)
        self.kMaxBox.setEnabled(True)

        # k Step
        kStepLabel = qt.QLabel(self)
        kStepLabel.setText("Window K Step:")
        self.kStepBox = qt.QDoubleSpinBox(self)
        self.kStepBox.setDecimals(2)
        self.kStepBox.setMinimum(0.01)
        self.kStepBox.setMaximum(0.5)
        self.kStepBox.setValue(0.02)
        self.kStepBox.setSingleStep(0.01)
        self.kStepBox.setEnabled(True)

        # the FT Range
        # R Max
        rMaxLabel = qt.QLabel(self)
        rMaxLabel.setText("FT Max. R:")
        self.rMaxBox = qt.QDoubleSpinBox(self)
        self.rMaxBox.setDecimals(2)
        self.rMaxBox.setMaximum(10.0)
        self.rMaxBox.setValue(6.0)
        self.rMaxBox.setSingleStep(0.5)
        self.rMaxBox.setEnabled(True)

        # the FT number of points
        pointsLabel = qt.QLabel(self)
        pointsLabel.setText("Points:")
        pointsOptions = ["512", "1024", "2048", "4096"]
        self.pointsSelector = qt.QComboBox(self)
        for option in pointsOptions:
            self.pointsSelector.addItem(option)
        self.pointsSelector.setCurrentIndex(2)

        # arrange everything
        self.mainLayout.addWidget(windowLabel, 0, 0)
        self.mainLayout.addWidget(self.windowSelector, 0, 1)
        self.mainLayout.addWidget(apodizationLabel, 1, 0)
        self.mainLayout.addWidget(self.apodizationBox, 1, 1)
        self.mainLayout.addWidget(kMinLabel, 2, 0)
        self.mainLayout.addWidget(self.kMinBox, 2, 1)
        self.mainLayout.addWidget(kMaxLabel, 3, 0)
        self.mainLayout.addWidget(self.kMaxBox, 3, 1)
        self.mainLayout.addWidget(kStepLabel, 4, 0)
        self.mainLayout.addWidget(self.kStepBox, 4, 1)
        self.mainLayout.addWidget(rMaxLabel, 5, 0)
        self.mainLayout.addWidget(self.rMaxBox, 5, 1)
        self.mainLayout.addWidget(pointsLabel, 6, 0)
        self.mainLayout.addWidget(self.pointsSelector, 6, 1)


        # connect
        #self.setupButton.clicked.connect(self._setupClicked)
        self.windowSelector.activated[int].connect(self._windowChanged)
        self.apodizationBox.valueChanged[float].connect(self._apodizationChanged)
        self.kMinBox.valueChanged[float].connect(self._kMinChanged)
        self.kMaxBox.valueChanged[float].connect(self._kMaxChanged)
        self.kStepBox.valueChanged[float].connect(self._kStepChanged)
        self.rMaxBox.valueChanged[float].connect(self._rMaxChanged)
        self.pointsSelector.activated[int].connect(self._pointsChanged)

    def _windowChanged(self, value):
        _logger.debug("_windowChanged %s" % value)
        current = str(self.windowSelector.currentText())
        if current.lower() in ["gaussian", "gauss", "tukey", "papul"]:
            self.apodizationBox.setEnabled(False)
        if current.lower() in ["kaiser"]:
            self.apodizationBox.setEnabled(True)
        else:
            self.apodizationBox.setEnabled(True)
        if self.__connected:
            self.emitSignal("FTWindowChanged")

    def _apodizationChanged(self, value):
        _logger.debug("_apodizationChanged %s" % value)
        if self.__connected:
            self.emitSignal("FTApodizationChanged")

    def _kMinChanged(self, value):
        _logger.debug("Current kMin Value = %s" % value)
        if self.__connected:
            self.emitSignal("FTKMinChanged")

    def _kMaxChanged(self, value):
        _logger.debug("Current kMax Value = %s" % value)
        if self.__connected:
            if value > self.kMinBox.value():
                self.emitSignal("FTKMaxChanged")
            else:
                # I should check if we have the focus prior to
                # raise any error.
                # This situation happens during manual editing
                pass

    def _kStepChanged(self, value):
        _logger.debug("Current kStep value = %s" % value)
        if self.__connected:
            self.emitSignal("FTKStepChanged")

    def _rMaxChanged(self, value):
        _logger.debug("Current rMax Value = %s", value)
        if self.__connected:
            self.emitSignal("FTRMaxChanged")

    def _pointsChanged(self, value):
        _logger.debug("_pointsChanged %s" % value)
        if self.__connected:
            self.emitSignal("FTPointsChanged")

    def getParameters(self):
        ddict = {}
        # window
        ddict["Window"] = str(self.windowSelector.currentText())
        ddict["WindowList"] = []
        for i in range(self.windowSelector.count()):
            ddict["WindowList"].append(str(self.windowSelector.itemText(i)))
        ddict["WindowApodization"] = self.apodizationBox.value()
        ddict["WindowRange"] = [self.kMinBox.value(),
                                self.kMaxBox.value()]
        ddict["KStep"] = self.kStepBox.value()
        ddict["Points"] = int(str(self.pointsSelector.currentText()))
        ddict["Range"] = [0.0, self.rMaxBox.value()]
        return ddict

    def setParameters(self, ddict, signal=True):
        _logger.debug("setParameters called, ddict %s, signal %s" % (ddict, signal))
        if "FT" in ddict:
            ddict = ddict["FT"]
        try:
            self.__connected = False
            if "Window" in ddict:
                option = ddict["Window"]
                if type(ddict["Window"]) == type(1):
                    self.windowSelector.setCurrentIndex(option)
                else:
                    selectorOptions = []
                    for i in range(self.windowSelector.count()):
                        selectorOptions.append(str(self.windowSelector.itemText(i)))
                    for i in range(len(selectorOptions)):
                        if selectorOptions[i].lower().startswith(str(option).lower()):
                            self.windowSelector.setCurrentIndex(i)
                            break
            if ddict["WindowRange"] not in [None, "None", "none"]:
                self.kMinBox.setValue(ddict["WindowRange"][0])
                self.kMaxBox.setValue(ddict["WindowRange"][-1])
            self.kStepBox.setValue(ddict["KStep"])
            self.rMaxBox.setValue(ddict["Range"][-1])
            v = 0
            for i in range(self.pointsSelector.count()):
                if int(str(self.pointsSelector.itemText(i))) < int(ddict["Points"]):
                    v += 1
                else:
                    break
            self.pointsSelector.setCurrentIndex(v)
        finally:
            self.__connected = True
        if signal:
            self.emitSignal("FTWindowChanged")

    def emitSignal(self, event):
        ddict = self.getParameters()
        ddict["event"] = event
        self.sigFTParametersSignal.emit(ddict)

    def setKRange(self, kRange):
        if kRange[0] > kRange[1]:
            # do nothing (it happens on editing)
            return
        if self.kMinBox.minimum() > kRange[0]: 
            self.kMinBox.setMinimum(kRange[0])
        if self.kMaxBox.maximum() < kRange[1]:
            self.kMaxBox.setMaximum(kRange[1])
        #kMin = self.kMinBox.value()
        #kMax = self.kMaxBox.value()
        #if kRange[1] > kMin:
        #    self.kMaxBox.setMaximum(kRange[1])
        #current = self.kMaxBox.value()
        #if current > (kRange[1]+0.01):
        #    self.kMaxBox.setValue(value)

    def setTitleColor(self, color):
        #self.setStyleSheet("QGroupBox {font-weight: bold; color: red;}")
        self.setStyleSheet("QGroupBox {color: %s;}" % color)

if __name__ == "__main__":
    _logger.setLevel(logging.DEBUG)
    app = qt.QApplication([])
    def mySlot(ddict):
        print("Signal received: ", ddict)
    w = XASFourierTransformParameters()
    w.show()
    w.sigFTParametersSignal.connect(mySlot)
    app.exec()
