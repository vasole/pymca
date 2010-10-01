#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
import PyMcaDirs
import SimpleFitGUI
qt = SimpleFitGUI.qt
from PyMca_Icons import IconDict
import StackSimpleFit
import ArraySave

class CalculationThread(qt.QThread):
    def __init__(self, parent=None, calculation_method=None):
        qt.QThread.__init__(self, parent)
        self.calculation_method = calculation_method

    def run(self):
        self.calculation_method()

class OutputParameters(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        self.outputDirLabel = qt.QLabel(self)
        self.outputDirLabel.setText("Output directory")
        self.outputDirLine  = qt.QLineEdit(self)
        self.outputDirLine.setReadOnly(True)
        self.outputDirButton = qt.QPushButton(self)
        self.outputDirButton.setText("Browse")
        
        self.outputFileLabel = qt.QLabel(self)
        self.outputFileLabel.setText("Output file root")
        self.outputFileLine  = qt.QLineEdit(self)
        self.outputFileLine.setReadOnly(True)

        self.outputDir = PyMcaDirs.outputDir
        self.outputFile = "StackSimpleFitOutput"
        self.setOutputDirectory(self.outputDir)
        self.setOutputFileBaseName(self.outputFile)

        self.mainLayout.addWidget(self.outputDirLabel,  0, 0)
        self.mainLayout.addWidget(self.outputDirLine,   0, 1)
        self.mainLayout.addWidget(self.outputDirButton, 0, 2)
        self.mainLayout.addWidget(self.outputFileLabel,  1, 0)
        self.mainLayout.addWidget(self.outputFileLine,   1, 1)

        self.connect(self.outputDirButton,
                     qt.SIGNAL('clicked()'),
                     self.browseDirectory)

    def getOutputDirectory(self):
        return str(self.outputDirLine.text())

    def getOutputFileBaseName(self):
        return str(self.outputFileLine.text())

    def setOutputDirectory(self, txt):
        if os.path.exists(txt):
            self.outputDirLine.setText(txt)
            self.outputDir =  txt
            PyMcaDirs.outputDir = txt
        else:
            raise IOError("Directory does not exists")
            

    def setOutputFileBaseName(self, txt):
        if len(txt):
            self.outputFileLine.setText(txt)
            self.outputFile = txt
        
    def browseDirectory(self):
        wdir = self.outputDir
        outputDir = qt.QFileDialog.getExistingDirectory(self,
                                                        "Please select output directory",
                                                        wdir)
        if len(outputDir):
            self.setOutputDirectory(str(outputDir))

class StackSimpleFitWindow(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Stack Fit Window')
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)

        self.fitSetupWindow = SimpleFitGUI.SimpleFitGUI(self)
        self.fitSetupWindow.fitActions.dismissButton.hide()
        self.mainLayout.addWidget(self.fitSetupWindow)
        self.fitInstance = self.fitSetupWindow.fitModule
        self.stackFitInstance = StackSimpleFit.StackSimpleFit(fit=self.fitInstance)
        
        self.importFunctions = self.fitSetupWindow.importFunctions

        self.outputParameters = OutputParameters(self)
        self.mainLayout.addWidget(self.outputParameters)
        self.startButton = qt.QPushButton(self)
        self.startButton.setText("FitStack")
        self.mainLayout.addWidget(self.startButton)
        self.connect(self.startButton,
                     qt.SIGNAL('clicked()'),
                     self.startStackFit)

        #progress handling
        self.stackFitInstance.setProgressCallback(self.progressBarUpdate)
        self.thread = CalculationThread(self, calculation_method=self.stackFitInstance.processStack)
        self.progressBar = qt.QProgressBar(self)
        self.mainLayout.addWidget(self.progressBar)
        self.connect(self.thread, qt.SIGNAL('finished()'), self.threadFinished)

    #def setSpectrum(self, x, y, sigma=None, xmin=None, xmax=None):
    def setSpectrum(self, *var, **kw):
        self.fitSetupWindow.setData(*var, **kw)

    def setData(self, x, stack, data_index=-1):
        self.stack_x = x
        self.stack_y = stack
        if hasattr(stack, "data") and\
           hasattr(stack, "info"):
            data = stack.data
        else:
            data = stack
        if data_index < 0:
            data_index = range(len(data.shape))[data_index]
        self.data_index = data_index

    def startStackFit(self):
        xmin = self.fitInstance._fitConfiguration['fit']['xmin']
        xmax = self.fitInstance._fitConfiguration['fit']['xmax']
        self.stackFitInstance.setOutputDirectory(self.outputParameters.getOutputDirectory())
        self.stackFitInstance.setOutputFileBaseName(self.outputParameters.getOutputFileBaseName())
        self.stackFitInstance.setData(self.stack_x, self.stack_y,
                                     sigma=None, xmin=xmin, xmax=xmax)
        self.stackFitInstance.setDataIndex(self.data_index)
        #check filenames
        fileNames = self.stackFitInstance.getOutputFileNames()
        deleteFiles = None
        for key in fileNames.keys():
            fileName = fileNames[key]
            if os.path.exists(fileName):
                msg = qt.QMessageBox()
                msg.setWindowTitle("Output file(s) exists")
                msg.setIcon(qt.QMessageBox.Information)
                msg.setText("Do you want to delete current output files?")
                msg.setStandardButtons(qt.QMessageBox.Yes|qt.QMessageBox.No)
                answer=msg.exec_()
                if answer == qt.QMessageBox.Yes:
                    deleteFiles = True
                else:
                    deleteFiles = False
                break

        if deleteFiles == False:
            #nothing to be done (yet)
            return
        
        if deleteFiles:
            try:
                for key in fileNames.keys():
                    fileName = fileNames[key]
                    if os.path.exists(fileName):
                        os.remove(fileName)
            except:
                qt.QMessageBox.critical(self, "Delete Error",
                    "ERROR while deleting file:\n%s"% fileName, 
                    qt.QMessageBox.Ok,
                    qt.QMessageBox.NoButton,
                    qt.QMessageBox.NoButton)
                return
        self.thread.start()
        self.setEnabled(False)

    def progressBarUpdate(self, idx, total):
        self.progressBar.show()
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(idx)

    def threadFinished(self):
        self.setEnabled(True)

if __name__ == "__main__":
    import numpy
    import SpecfitFuns
    import DefaultFitFunctions as Functions
    x = numpy.arange(1000.)
    data = numpy.zeros((50, 1000), numpy.float)

    #the peaks to be fitted
    p0 = [100., 300., 50.,
          200., 500., 30.,
          300., 800., 65]

    #generate the data to be fitted
    for i in range(data.shape[0]):
        nPeaks = 3 - i % 3
        data[i,:] = SpecfitFuns.gauss(p0[:3*nPeaks],x)
    #the spectrum for setup
    y = data.sum(axis=0)
    oldShape = data.shape
    data.shape = 1,oldShape[0], oldShape[1]
    app = qt.QApplication([])
    w = StackSimpleFitWindow()
    w.setSpectrum(x, y)
    w.setData(x, data)
    #w.importFunctions(Functions.__file__)
    #w.fitModule.setFitFunction('Gaussians')
    w.show()
    app.exec_()
