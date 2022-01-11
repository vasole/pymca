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
import sys
import logging
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGui import XASNormalizationParameters
from PyMca5.PyMcaGui import XASPostEdgeParameters
from PyMca5.PyMcaGui import XASFourierTransformParameters
from PyMca5.PyMcaGui import PyMcaFileDialogs
from PyMca5.PyMcaIO import ConfigDict

_logger = logging.getLogger(__name__)


class XASParameters(qt.QWidget):
    sigXASParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, color=None):
        super(XASParameters, self).__init__(parent)
        self.setWindowTitle("XAS Parameters")
        self.build()
        if color is not None:
            self.setTitleColor(color)

    def build(self):
        # perhaps the layout will change to a QGridLayout 
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.normalizationWidget = \
                XASNormalizationParameters.XASNormalizationParameters(self)
        self.postEdgeWidget = \
                XASPostEdgeParameters.XASPostEdgeParameters(self)
        self.fourierTransformWidget = \
                XASFourierTransformParameters.XASFourierTransformParameters(self)
        self.mainLayout.addWidget(self.normalizationWidget)
        self.mainLayout.addWidget(self.postEdgeWidget)
        self.mainLayout.addWidget(self.fourierTransformWidget)
        self.mainLayout.addWidget(qt.VerticalSpacer(self))

        container = qt.QWidget(self)
        container.mainLayout = qt.QHBoxLayout(container)
        container.mainLayout.setContentsMargins(0, 0, 0, 0)
        container.mainLayout.setSpacing(2)
        self.loadButton = qt.QPushButton(container)
        self.loadButton.setText("Load")
        self.loadButton.setAutoDefault(False)        
        self.saveButton = qt.QPushButton(container)
        self.saveButton.setText("Save")
        self.saveButton.setAutoDefault(False)
        container.mainLayout.addWidget(self.loadButton)
        container.mainLayout.addWidget(self.saveButton)
        self.mainLayout.addWidget(container)

        # add function
        self.setJump = self.normalizationWidget.setJump
        self.setMaximumK = self.postEdgeWidget.setMaximumK

        # connect
        self.normalizationWidget.sigNormalizationParametersSignal.connect( \
            self._normalizationSlot)
        self.postEdgeWidget.sigPostEdgeParametersSignal.connect( \
            self._postEdgeParameterSlot)
        self.fourierTransformWidget.sigFTParametersSignal.connect( \
            self._fourierTransformParameterSlot)
        self.loadButton.clicked.connect(self._loadClicked)
        self.saveButton.clicked.connect(self._saveClicked)

    def setMaximumK(self, value):
        self.postEdgeWidget.setMaximumK(value)
        self.fourierTransformWidget.setMaximumK(value)

    def emitSignal(self, event):
        ddict = self.getParameters()
        ddict["event"] = event
        self.sigPostEdgeParametersSignal.emit(ddict)

    def getParameters(self):
        ddict = {}
        ddict["Version"] = 1.0
        ddict["Normalization"] = self.getNormalizationParameters()
        ddict["EXAFS"] = self.getPostEdgeParameters()
        ddict["FT"] = self.getFTParameters()
        return ddict

    def getNormalizationParameters(self):
        return self.normalizationWidget.getParameters()

    def getPostEdgeParameters(self):
        return self.postEdgeWidget.getParameters()

    def getFTParameters(self):
        return self.fourierTransformWidget.getParameters()

    def setParameters(self, ddict):
        if "Normalization" in ddict:
            self.setNormalizationParameters(ddict["Normalization"])
        if "EXAFS" in ddict:
            self.setPostEdgeParameters(ddict["EXAFS"])
        if "FT" in ddict:
            self.setFTParameters(ddict["FT"])

    def setNormalizationParameters(self, ddict):
        self.normalizationWidget.setParameters(ddict)

    def setPostEdgeParameters(self, ddict):
        self.postEdgeWidget.setParameters(ddict)

    def setFTParameters(self, ddict):
        self.fourierTransformWidget.setParameters(ddict)
        #self._FTParameters = ddict

    def _normalizationSlot(self, ddict):
        # Should I change the event to "NormalizationChanged"?
        self._emitSignal(ddict["event"])

    def _postEdgeParameterSlot(self, ddict):
        _logger.debug("_postEdgeParameterSlot: %s" % ddict)
        # Should I change the event to "EXAFSChanged"?
        self.fourierTransformWidget.setKRange([ddict["KMin"], ddict["KMax"]])
        self._emitSignal(ddict["event"])

    def _fourierTransformParameterSlot(self, ddict):
        # Should I change the event to "FTChanged"?
        self._emitSignal(ddict["event"])

    def _emitSignal(self, event):
        ddict = self.getParameters()
        ddict["event"] = event
        self.sigXASParametersSignal.emit(ddict)

    def setSpectrum(self, energy, mu):
        return self.normalizationWidget.setSpectrum(energy, mu)

    def setJump(self, value):
        return self.normalizationWidget.setJump(value)

    def _loadClicked(self):
        return self.loadParameters()

    def loadParameters(self, fname=None):
        if fname is None:
            fname = PyMcaFileDialogs.getFileList(self,
                                         filetypelist=["Configuration (*.ini)",
                                                       "Configuration (*.cfg)",
                                                       "All files (*)"],
                                         message="Please set input file name",
                                         mode="OPEN",
                                         getfilter=False,
                                         single=True)
            if len(fname):
                fname = fname[0]
            else:
                return
        d = ConfigDict.ConfigDict()
        d.read(fname)
        self.setParameters(d["XASParameters"])

    def _saveClicked(self):
        return self.saveParameters()

    def saveParameters(self, fname=None):
        if fname is None:
            fname = PyMcaFileDialogs.getFileList(self,
                                         filetypelist=["Configuration (*.ini)",
                                                       "Configuration (*.cfg)"],
                                         message="Please enter output file name",
                                         mode="SAVE",
                                         getfilter=False,
                                         single=True)
            if len(fname):
                fname = fname[0]
            else:
                return
        ddict = ConfigDict.ConfigDict()
        ddict["XASParameters"] = self.getParameters()
        ddict.write(fname)

    def setTitleColor(self, color):
        try:
            self.normalizationWidget.setTitleColor(color)
            self.postEdgeWidget.setTitleColor(color)
            self.fourierTransformWidget.setTitleColor(color)
        except:
            _logger.error("Error setting title color: %s" % sys.exc_info())

if __name__ == "__main__":
    _logger.setLevel(logging.DEBUG)
    app = qt.QApplication([])
    def testSlot(ddict):
        print("Emitted signal = ", ddict)
    w = XASParameters()
    w.sigXASParametersSignal.connect(testSlot)
    w.show()
    try:
        import os
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
    except:
        print("ERROR: ", sys.exc_info())
    app.exec()
