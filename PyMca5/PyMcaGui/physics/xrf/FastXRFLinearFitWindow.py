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

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGui import PyMcaFileDialogs


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
        outdirLabel = qt.QLabel(self)
        outdirLabel.setText("Output dir:")
        self._outdirLine = qt.QLineEdit(self)
        self._outdirLine.setReadOnly(True)

        self._outdirButton = qt.QPushButton(self)
        self._outdirButton.setText('Browse')
        self._outdirButton.setAutoDefault(False)
        self._outdirButton.clicked.connect(self.browseOutputDir)

        # output root
        outrootLabel = qt.QLabel(self)
        outrootLabel.setText("Output root:")
        self._outrootLine = qt.QLineEdit(self)
        self._outrootLine.setReadOnly(False)
        self._outrootLine.setText("IMAGES")

        # output entry
        outentryLabel = qt.QLabel(self)
        outentryLabel.setText("Output entry:")
        self._outentryLine = qt.QLineEdit(self)
        self._outentryLine.setReadOnly(False)
        self._outentryLine.setText("images")

        # output process
        outnameLabel = qt.QLabel(self)
        outnameLabel.setText("Output name:")
        self._outnameLine = qt.QLineEdit(self)
        self._outnameLine.setReadOnly(False)
        self._outnameLine.setText("")

        # fit options
        boxLabel1 = qt.QLabel(self)
        boxLabel1.setText("Fit options:")
        self._boxContainer1 = qt.QWidget(self) 
        self._boxContainerLayout1 = qt.QHBoxLayout(self._boxContainer1)
        self._boxContainerLayout1.setContentsMargins(0, 0, 0, 0)
        self._boxContainerLayout1.setSpacing(0)
        
        # save options
        boxLabel2 = qt.QLabel(self)
        boxLabel2.setText("Save options:")
        self._boxContainer2 = qt.QWidget(self) 
        self._boxContainerLayout2 = qt.QHBoxLayout(self._boxContainer2)
        self._boxContainerLayout2.setContentsMargins(0, 0, 0, 0)
        self._boxContainerLayout2.setSpacing(0)
        
        # concentrations
        self._concentrationsBox = qt.QCheckBox(self._boxContainer1)
        self._concentrationsBox.setText("calculate concentrations")
        self._concentrationsBox.setChecked(False)
        self._concentrationsBox.setEnabled(True)

        # repeat fit on negative contributions
        self._fitAgainBox = qt.QCheckBox(self._boxContainer1)
        self._fitAgainBox.setText("Repeat fit on negative contributions")
        self._fitAgainBox.setChecked(True)
        self._fitAgainBox.setEnabled(True)
        text  = "Fit of pixels with negative peak area\n"
        text += "contributions will be repeated.\n"
        text += "This can seriously slow down the process\n"
        text += "if your sample model is far from the truth."
        self._fitAgainBox.setToolTip(text)

        # generate tiff files
        self._tiffBox = qt.QCheckBox(self._boxContainer2)
        self._tiffBox.setText("TIFF")
        self._tiffBox.setChecked(False)
        self._tiffBox.setEnabled(True)
        
        # generate csv file
        self._csvBox = qt.QCheckBox(self._boxContainer2)
        self._csvBox.setText("CSV")
        self._csvBox.setChecked(False)
        self._csvBox.setEnabled(True)
        
        # generate edf file
        self._edfBox = qt.QCheckBox(self._boxContainer2)
        self._edfBox.setText("EDF")
        self._edfBox.setChecked(False)
        self._edfBox.setEnabled(True)
        
        # generate hdf5 file
        self._h5Box = qt.QCheckBox(self._boxContainer2)
        self._h5Box.setText("HDF5")
        self._h5Box.setChecked(True)
        self._h5Box.setEnabled(True)

        self._boxContainerLayout1.addWidget(self._concentrationsBox)
        self._boxContainerLayout1.addWidget(self._fitAgainBox)
        self._boxContainerLayout2.addWidget(self._h5Box)
        self._boxContainerLayout2.addWidget(self._edfBox)
        self._boxContainerLayout2.addWidget(self._csvBox)
        self._boxContainerLayout2.addWidget(self._tiffBox)
        
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

        i = 0
        self.mainLayout.addWidget(configLabel, i, 0)
        self.mainLayout.addWidget(self._configLine, i, 1)
        self.mainLayout.addWidget(self._configButton, i, 2)
        i += 1
        self.mainLayout.addWidget(outdirLabel, i, 0)
        self.mainLayout.addWidget(self._outdirLine, i, 1)
        self.mainLayout.addWidget(self._outdirButton, i, 2)
        i += 1
        self.mainLayout.addWidget(outrootLabel, i, 0)
        self.mainLayout.addWidget(self._outrootLine, i, 1)
        i += 1
        self.mainLayout.addWidget(outentryLabel, i, 0)
        self.mainLayout.addWidget(self._outentryLine, i, 1)
        i += 1
        self.mainLayout.addWidget(outnameLabel, i, 0)
        self.mainLayout.addWidget(self._outnameLine, i, 1)
        i += 1
        self.mainLayout.addWidget(weightLabel, i, 0)
        self.mainLayout.addWidget(self._weightWidget, i, 1, 1, 1)
        i += 1
        self.mainLayout.addWidget(boxLabel1, i, 0)
        self.mainLayout.addWidget(self._boxContainer1, i, 1, 1, 1)
        i += 1
        self.mainLayout.addWidget(boxLabel2, i, 0)
        self.mainLayout.addWidget(self._boxContainer2, i, 1, 1, 1)
        
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
            self._outdirLine.setText(f)

    def getParameters(self):
        ddict = {}
        ddict['configuration'] = qt.safe_str(self._configLine.text())
        ddict['output_dir'] = qt.safe_str(self._outdirLine.text())
        ddict['output_root'] = qt.safe_str(self._outrootLine.text())
        ddict['file_entry'] = qt.safe_str(self._outentryLine.text())
        ddict['file_name'] = qt.safe_str(self._outnameLine.text())
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
        if self._csvBox.isChecked():
            ddict['csv'] = 1
        else:
            ddict['csv'] = 0
        if self._edfBox.isChecked():
            ddict['edf'] = 1
        else:
            ddict['edf'] = 0
        if self._h5Box.isChecked():
            ddict['h5'] = 1
        else:
            ddict['h5'] = 0
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
