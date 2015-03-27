#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
import numpy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGui import PyMcaFileDialogs

DEBUG = 0

class FastXRFLinearFitWindow(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("FastXRFLinearFitWindow")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        # configuration file
        configLabel = qt.QLabel(self)
        configLabel.setText("Fit Configuration File:")
        self._configLine = qt.QLineEdit(self)
        self._configLine.setReadOnly(True)

        self._configButton = qt.QPushButton(self)
        self._configButton.setText("Browse")
        self._configButton.setAutoDefault(False)
        self._configButton.clicked.connect(self.browseConfigurationFile)

        # output directory
        outLabel   = qt.QLabel(self)
        outLabel.setText("Output dir:")
        self._outLine = qt.QLineEdit(self)
        self._outLine.setReadOnly(True)

        self._outButton = qt.QPushButton(self)
        self._outButton.setText('Browse')
        self._outButton.setAutoDefault(False)
        self._outButton.clicked.connect(self.browseOutputDir)

        # output file name
        fileLabel   = qt.QLabel(self)
        fileLabel.setText("Output file root:")
        self._fileLine = qt.QLineEdit(self)
        self._fileLine.setReadOnly(False)
        self._fileLine.setText("images")


        boxLabel   = qt.QLabel(self)
        boxLabel.setText("Misc. flags:")

        self._boxContainer = qt.QWidget(self) 
        self._boxContainerLayout = qt.QHBoxLayout(self._boxContainer)
        self._boxContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._boxContainerLayout.setSpacing(0)
        # concentrations
        self._concentrationsBox = qt.QCheckBox(self._boxContainer)
        self._concentrationsBox.setText("calculate concentrations")
        self._concentrationsBox.setChecked(False)
        self._concentrationsBox.setEnabled(True)

        # repeat fit on negative contributions
        self._fitAgainBox = qt.QCheckBox(self._boxContainer)
        self._fitAgainBox.setText("Repeat fit on negative contributions")
        self._fitAgainBox.setChecked(True)
        self._fitAgainBox.setEnabled(True)
        text  = "Fit of pixels with negative peak area\n"
        text += "contributions will be repeated.\n"
        text += "This can seriously slow down the process\n"
        text += "if your sample model is far from the truth."
        self._fitAgainBox.setToolTip(text)

        # generate tiff files
        self._tiffBox = qt.QCheckBox(self._boxContainer)
        self._tiffBox.setText("generate TIFF files")
        self._tiffBox.setChecked(False)
        self._tiffBox.setEnabled(True)

        self._boxContainerLayout.addWidget(self._concentrationsBox)
        self._boxContainerLayout.addWidget(self._fitAgainBox)
        self._boxContainerLayout.addWidget(self._tiffBox)
        
        # weight method
        self._weightWidget = qt.QWidget(self)
        self._weightWidget.mainLayout = qt.QHBoxLayout(self._weightWidget)
        self._weightWidget.mainLayout.setContentsMargins(0, 0, 0, 0)
        self._weightWidget.mainLayout.setSpacing(0)
        self._weightButtonGroup = qt.QButtonGroup(self._weightWidget)
        i = 0
        weightLabel = qt.QLabel(self)
        weightLabel.setText("Weight policy: ")
        for txt in ["No Weight (Fastest)", "Average Weight (Fast)", "Individual Weights (slow)"]:
            button = qt.QRadioButton(self._weightWidget)
            button.setText(txt)
            self._weightButtonGroup.addButton(button)
            self._weightButtonGroup.setId(button, i)
            self._weightWidget.mainLayout.addWidget(button)
            i += 1
        self._weightButtonGroup.buttons()[0].setChecked(True)
        #self._weightWidget.mainLayout.addWidget(qt.HorizontalSpacer(self._weightWidget))

        self.mainLayout.addWidget(configLabel, 0, 0)
        self.mainLayout.addWidget(self._configLine, 0, 1)
        self.mainLayout.addWidget(self._configButton, 0, 2)
        self.mainLayout.addWidget(outLabel, 1, 0)
        self.mainLayout.addWidget(self._outLine, 1, 1)
        self.mainLayout.addWidget(self._outButton, 1, 2)
        self.mainLayout.addWidget(fileLabel, 2, 0)
        self.mainLayout.addWidget(self._fileLine, 2, 1)
        self.mainLayout.addWidget(weightLabel, 3, 0)
        self.mainLayout.addWidget(self._weightWidget, 3, 1, 1, 1)
        self.mainLayout.addWidget(boxLabel, 4, 0)
        self.mainLayout.addWidget(self._boxContainer, 4, 1, 1, 1)

    def sizeHint(self):
        return qt.QSize(int(1.8 * qt.QWidget.sizeHint(self).width()),
                        qt.QWidget.sizeHint(self).height())

    def browseConfigurationFile(self):
        f = PyMcaFileDialogs.getFileList(parent=self,
                                     filetypelist=["Configuration files (*.cfg)"],
                                     message="Open a fit configuration file",
                                     mode="OPEN",
                                     single=True)
        if len(f):
            self._configLine.setText(f[0])

    def browseOutputDir(self):
        f = PyMcaFileDialogs.getExistingDirectory(parent=self,
                                     message="Please select output directory",
                                     mode="OPEN")
        if len(f):
            self._outLine.setText(f)

    def getParameters(self):
        ddict = {}
        ddict['configuration'] = qt.safe_str(self._configLine.text())
        ddict['output_dir'] = qt.safe_str(self._outLine.text())
        ddict['file_root'] = qt.safe_str(self._fileLine.text())
        if self._concentrationsBox.isChecked():
            ddict['concentrations'] = 1
        else:
            ddict['concentrations'] = 0
        ddict['weight_policy'] = self._weightButtonGroup.checkedId()
        if self._fitAgainBox.isChecked():
            ddict['refit'] = 1
        else:
            ddict['refit'] = 0
        if self._tiffBox.isChecked():
            ddict['tiff'] = 1
        else:
            ddict['tiff'] = 0
        return ddict

class FastXRFLinearFitDialog(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Fast XRF Linear Fit Dialog")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.parametersWidget = FastXRFLinearFitWindow(self)

        self.rejectButton= qt.QPushButton(self)
        self.rejectButton.setAutoDefault(False)
        self.rejectButton.setText("Cancel")

        self.acceptButton= qt.QPushButton(self)
        self.acceptButton.setAutoDefault(False)
        self.acceptButton.setText("OK")

        self.rejectButton.clicked.connect(self.reject)
        self.acceptButton.clicked.connect(self.accept)

        self.mainLayout.addWidget(self.parametersWidget, 0, 0, 5, 4)
        self.mainLayout.addWidget(self.rejectButton, 6, 1)
        self.mainLayout.addWidget(self.acceptButton, 6, 2)

    def getParameters(self):
        return self.parametersWidget.getParameters()

if __name__ == "__main__":
    app = qt.QApplication([])
    w = FastXRFLinearFitDialog()
    ret = w.exec_()
    if ret:
        print(w.getParameters())
    #app.exec_()
