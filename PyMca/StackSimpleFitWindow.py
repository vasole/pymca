#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
import traceback
import time
from PyMca import PyMcaQt as qt
from PyMca import PyMcaDirs
from PyMca import SimpleFitGUI
from PyMca.PyMca_Icons import IconDict
from PyMca import StackSimpleFit
from PyMca.PyMcaIO import ArraySave
from PyMca import CalculationThread
safe_str = qt.safe_str

class OutputParameters(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(2, 2, 2, 2)
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
        return safe_str(self.outputDirLine.text())

    def getOutputFileBaseName(self):
        return safe_str(self.outputFileLine.text())

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
            self.setOutputDirectory(safe_str(outputDir))

class StackSimpleFitWindow(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Stack Fit Window')
        self.setWindowIcon(qt.QIcon(qt.QPixmap(IconDict['gioconda16'])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(2, 2, 2, 2)
        self.mainLayout.setSpacing(2)

        self.fitSetupWindow = SimpleFitGUI.SimpleFitGUI(self)
        self.fitSetupWindow.fitActions.dismissButton.hide()
        self.mainLayout.addWidget(self.fitSetupWindow)
        self.fitInstance = self.fitSetupWindow.fitModule
        self.stackFitInstance = StackSimpleFit.StackSimpleFit(fit=self.fitInstance)
        self.__mask = None

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
        self._total = 100
        self._index = 0
        self.stackFitInstance.setProgressCallback(self.progressBarUpdate)
        self.progressBar = qt.QProgressBar(self)
        self.mainLayout.addWidget(self.progressBar)

    #def setSpectrum(self, x, y, sigma=None, xmin=None, xmax=None):
    def setSpectrum(self, *var, **kw):
        self.fitSetupWindow.setData(*var, **kw)

    def setData(self, x, stack, data_index=-1, mask=None):
        self.stack_x = x
        self.stack_y = stack
        self.__mask = mask
        if hasattr(stack, "data") and\
           hasattr(stack, "info"):
            data = stack.data
        else:
            data = stack
        if data_index < 0:
            data_index = range(len(data.shape))[data_index]
        self.data_index = data_index

    def setMask(self, mask):
        self.__mask = mask

    def processStack(self):
        self.stackFitInstance.processStack(mask=self.__mask)

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
        try:
            self._startWork()
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Stack Fitting Error")
            msg.setText("Error has occured while processing the data")
            msg.setInformativeText(safe_str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
        finally:
            self.progressBar.hide()
            self.setEnabled(True)

    def _startWork(self):
        self.setEnabled(False)
        self.progressBar.show()
        thread = CalculationThread.CalculationThread(parent=self,
                                calculation_method=self.processStack)
        thread.start()
        self._total = 100
        self._index = 0
        while thread.isRunning():
            time.sleep(2)
            qt.qApp.processEvents()
            self.progressBar.setMaximum(self._total)
            self.progressBar.setValue(self._index)
        self.progressBar.hide()
        self.setEnabled(True)
        if thread.result is not None:
            if len(thread.result):
                raise Exception(*thread.result[1:])

    def progressBarUpdate(self, idx, total):
        self._index = int(idx)
        self._total = int(total)
        if idx % 10 == 0:
            print("Fited %d of %d" % (idx, total))

    def threadFinished(self):
        self.setEnabled(True)

if __name__ == "__main__":
    import numpy
    from PyMca import SpecfitFuns
    from PyMca import SimpleFitUserEstimatedFunctions as Functions
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
