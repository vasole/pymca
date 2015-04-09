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

class StackROIBatchWindow(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("StackROIBatchWindow")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        # configuration file
        configLabel = qt.QLabel(self)
        configLabel.setText("ROI Configuration File:")
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
        self._fileLine.setText("ROI_images")


        boxLabel   = qt.QLabel(self)
        boxLabel.setText("Options:")

        self._boxContainer = qt.QWidget(self) 
        self._boxContainerLayout = qt.QHBoxLayout(self._boxContainer)
        self._boxContainerLayout.setContentsMargins(0, 0, 0, 0)
        self._boxContainerLayout.setSpacing(0)
        # Net ROI
        self._netBox = qt.QCheckBox(self._boxContainer)
        self._netBox.setText("Calculate Net ROI")
        self._netBox.setChecked(True)
        self._netBox.setEnabled(False)

        # xAtMax
        self._xAtMaxBox = qt.QCheckBox(self._boxContainer)
        self._xAtMaxBox.setText("Image X at Min/Max. Y")
        self._xAtMaxBox.setChecked(False)
        self._xAtMaxBox.setEnabled(True)
        text  = "Calculate the channel of the maximum and\n"
        text += "the minimum value in the region.\n"
        self._xAtMaxBox.setToolTip(text)

        # generate tiff files
        self._tiffBox = qt.QCheckBox(self._boxContainer)
        self._tiffBox.setText("generate TIFF files")
        self._tiffBox.setChecked(False)
        self._tiffBox.setEnabled(True)

        self._boxContainerLayout.addWidget(self._netBox)
        self._boxContainerLayout.addWidget(self._xAtMaxBox)
        self._boxContainerLayout.addWidget(self._tiffBox)
        
        self.mainLayout.addWidget(configLabel, 0, 0)
        self.mainLayout.addWidget(self._configLine, 0, 1)
        self.mainLayout.addWidget(self._configButton, 0, 2)
        self.mainLayout.addWidget(outLabel, 1, 0)
        self.mainLayout.addWidget(self._outLine, 1, 1)
        self.mainLayout.addWidget(self._outButton, 1, 2)
        self.mainLayout.addWidget(fileLabel, 2, 0)
        self.mainLayout.addWidget(self._fileLine, 2, 1)
        self.mainLayout.addWidget(boxLabel, 3, 0)
        self.mainLayout.addWidget(self._boxContainer, 3, 1, 1, 1)

    def sizeHint(self):
        return qt.QSize(int(1.8 * qt.QWidget.sizeHint(self).width()),
                        qt.QWidget.sizeHint(self).height())

    def browseConfigurationFile(self):
        f = PyMcaFileDialogs.getFileList(parent=self,
                                     filetypelist=["Configuration files (*.ini)"],
                                     message="Open a ROI configuration file",
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
        if self._netBox.isChecked():
            ddict['net'] = 1
        else:
            ddict['net'] = 0
        if self._xAtMaxBox.isChecked():
            ddict['xAtMinMax'] = 1
        else:
            ddict['xAtMinMax'] = 0
        if self._tiffBox.isChecked():
            ddict['tiff'] = 1
        else:
            ddict['tiff'] = 0
        return ddict

class StackROIBatchDialog(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Stack ROI Batch Dialog")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.parametersWidget = StackROIBatchWindow(self)

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
    w = StackROIBatchDialog()
    ret = w.exec_()
    if ret:
        print(w.getParameters())
    #app.exec_()
