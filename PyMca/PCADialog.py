#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
import sys
import time
import numpy
from PyMca import PyMcaQt as qt
PCA = False
MDP = False
if qt.qVersion() > '4.0.0':
    try:
        import PyQt4.Qwt5 as Qwt
        from PyMca import PCAWindow
        PCA = True
        MDP = PCAWindow.MDP
    except ImportError:
        pass
DEBUG = 0

class SimpleThread(qt.QThread):
    def __init__(self, function, *var, **kw):
        if kw is None:kw={}
        qt.QThread.__init__(self)
        self._function = function
        self._var      = var
        self._kw       = kw
        self._result   = None
    
    def run(self):
        if DEBUG:
            self._result = self._function(*self._var, **self._kw )
        else:
            try:
                self._result = self._function(*self._var, **self._kw )
            except:
                self._result = ("Exception",) + sys.exc_info()

class PCADialog(qt.QDialog):
    def __init__(self, parent=None, rgbwidget=None, selection=False):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("PCA calculation dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.calculateButton = qt.QPushButton(self)
        self.calculateButton.setAutoDefault(False)
        self.calculateButton.setText("Perform PCA")
        self.showLastButton = qt.QPushButton(self)
        self.showLastButton.setAutoDefault(False)
        self.showLastButton.setText("Last Results")
        self.mainLayout.addWidget(self.calculateButton)
        self.mainLayout.addWidget(self.showLastButton)
        self._data = None
        
        self.pcaWindow = PCAWindow.PCAWindow(parent = None,
                                            rgbwidget=rgbwidget,
                                            #selection=True,
                                            selection=selection,
                                            colormap=True,
                                            #imageicons=True,
                                            imageicons=selection,
                                            standalonesave=True)
        self.pcaWindow.setDefaultColormap(0, logflag=False)
        self.pcaParametersDialog = PCAWindow.PCAParametersDialog(self)
        self.pcaParametersDialog.nPC.setMaximum(11)
        self.pcaParametersDialog.nPC.setValue(10)
        self.pcaParametersDialog.hide()
        self.pcaParametersDialogInitialized = False
        self.pcaWindow.hide()

        #connections
        self.connect(self.calculateButton,
                     qt.SIGNAL("clicked()"),
                     self._calculateSlot)

        self.connect(self.showLastButton,
                     qt.SIGNAL("clicked()"),
                     self._showLastSlot)

    def sizeHint(self):
        return qt.QSize(int(4*qt.QDialog.sizeHint(self).width()),
                        qt.QDialog.sizeHint(self).height())
        
    def _calculateSlot(self):
        if self._data is None:
            msg = qt.QMessageBox(self)
            msg.setWindowTitle("No data")
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("No data to perform calculation")
            msg.exec_()
            return

        if not self.pcaParametersDialogInitialized:
            self.pcaParametersDialog.nPC.setMaximum(self._spectrumLength)
            self.pcaParametersDialog.nPC.setValue(min(10, self._spectrumLength))
            ddict = {'options':self._binningOptions, 'binning': 1, 'method': 0}
            self.pcaParametersDialog.setParameters(ddict)
            self.pcaParametersDialogInitialized = True
        ret = self.pcaParametersDialog.exec_()
        if ret:
            if DEBUG:
                t0 = time.time()
            pcaParameters = self.pcaParametersDialog.getParameters()
            self.pcaParametersDialog.close()
            function = pcaParameters['function']
            binning = pcaParameters['binning']
            npc = pcaParameters['npc']
            mask = pcaParameters['mask']
            kw = pcaParameters.get('kw', {})
            data = self._data
            old_shape = self._data.shape
            if mask is not None:
                if mask.sum() < npc:
                    msg = qt.QMessageBox(self)
                    msg.setWindowTitle("Not enough data")
                    msg.setIcon(qt.QMessageBox.Information)
                    msg.setText("Number of components too high")
                    msg.exec_()
                    return
            if DEBUG:
                images, eigenvalues, eigenvectors = function(data,
                                                             npc,
                                                             binning=binning,
                                                             mask=mask,
                                                             **kw)
            else:
                try:
                    threadResult = self._submitThread(function,
                                                         data,
                                                         npc,
                                                         binning=binning,
                                                         mask=mask,
                                                         **kw)
                    if type(threadResult) == type((1,)):
                        if len(threadResult):
                            if threadResult[0] == "Exception":
                                raise Exception(threadResult[1],threadResult[2])
                    images, eigenvalues, eigenvectors = threadResult
                except:
                    if isinstance(data, numpy.ndarray):
                        self._data.shape = old_shape
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Critical)
                    msg.setText("%s" % sys.exc_info()[1])
                    msg.exec_()
                    return
            if isinstance(self._data, numpy.ndarray):
                self._data.shape = old_shape
            if DEBUG:
                print("PCA Elapsed = ", time.time() - t0)
            methodlabel = pcaParameters.get('methodlabel', "")
            imagenames=None
            vectornames=None
            if " ICA " in methodlabel:
                nimages = images.shape[0]
                imagenames = []
                vectornames = []
                itmp = nimages/2
                for i in range(itmp):
                    imagenames.append("ICAimage %02d" % i)
                    vectornames.append("ICAvector %02d" % i)
                for i in range(itmp):
                    imagenames.append("Eigenimage %02d" % i)
                    vectornames.append("Eigenvector %02d" % i)
            self.pcaWindow.setPCAData(images,
                                      eigenvalues,
                                      eigenvectors,
                                      imagenames=imagenames,
                                      vectornames=vectornames)
            self.pcaWindow.show()
            self.pcaWindow.raise_()


    def _showLastSlot(self):
        self.pcaWindow.show()
        self.pcaWindow.raise_()

    def setData(self, data=None, spectrumindex=-1):
        if type(data) == type([]):
            #assume is an image list
            if data[0].dtype not in [numpy.float32, numpy.float64]:
                dtype = numpy.float64
            else:
                dtype = data[0].dtype
            self._spectrumLength = len(data)
            self._shape = data[0].shape
            n = 1
            for shape in self._shape:
                n *= shape
            self._binningOptions = [1]
            if len(self._shape) == 1:
                self._data = numpy.zeros((self._shape[0],
                                          self._spectrumLength), dtype)
                for i in range(self._spectrumLength):
                    self._data[:, i] = data[i][:]
            elif len(self._shape) == 2:
                self._data = numpy.zeros((self._shape[0],
                                          self._shape[1],
                                          self._spectrumLength), dtype)
                for i in range(self._spectrumLength):
                    self._data[:, :, i] = data[i][:,:]
        else:
            self._shape = data.shape
            self._data = data
            self._spectrumLength = self._shape[spectrumindex]
            self._binningOptions=[1]
            for number in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19]:
                if (self._spectrumLength % number) == 0:
                    self._binningOptions.append(number)
        if self.pcaParametersDialog is not None:
            value = self.pcaParametersDialog.nPC.value()
            self.pcaParametersDialog.nPC.setMaximum(self._spectrumLength)
            self.pcaParametersDialog.nPC.setValue(min(value, self._spectrumLength))


    def setSpectrum(self, x, y, legend=None):
        return self.pcaParametersDialog.setSpectrum(x, y, legend=legend)

    def _submitThread(self, function, *var, **kw):
        message = "Please Wait: PCA Going On"
        sthread = SimpleThread(function, *var, **kw)
        return self._startThread(sthread, message)

    def _startThread(self, sthread, message):
        sthread.start()
        if 0:
            msg = qt.QDialog(self, qt.Qt.FramelessWindowHint)
            msg.setModal(0)
        else:
            msg = qt.QDialog(self, qt.Qt.FramelessWindowHint)
            msg.setModal(1)
        msg.setWindowTitle("Please Wait")
        layout = qt.QHBoxLayout(msg)
        layout.setMargin(0)
        layout.setSpacing(0)
        l1 = qt.QLabel(msg)
        l1.setFixedWidth(l1.fontMetrics().width('##'))
        l2 = qt.QLabel(msg)
        l2.setText("%s" % message)
        l3 = qt.QLabel(msg)
        l3.setFixedWidth(l3.fontMetrics().width('##'))
        layout.addWidget(l1)
        layout.addWidget(l2)
        layout.addWidget(l3)
        msg.show()
        qt.qApp.processEvents()
        t0 = time.time()
        i = 0
        ticks = ['-','\\', "|", "/","-","\\",'|','/']
        while (sthread.isRunning()):
            i = (i+1) % 8
            l1.setText(ticks[i])
            l3.setText(" "+ticks[i])
            qt.qApp.processEvents()
            time.sleep(2)
        msg.close()
        result = sthread._result
        del sthread
        self.raise_()
        return result

if __name__ == "__main__":
    DEBUG = 1
    import os
    from PyMca import EdfFile
    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    d = PCADialog()
    if len(sys.argv) < 2:
        fileList = ["D:\DATA\ICA\mix1.edf",
                    "D:\DATA\ICA\mix2.edf",
                    "D:\DATA\ICA\mix3.edf"]
    else:
        fileList = []
        for i in range(1, len(sys.argv)):
            fileList.append(sys.argv[i])
    imageList = []
    for fname in fileList:
        print(fname)
        if not os.path.exists(fname):
            print("File name %s does not exists" % fname)
            break
        edf = EdfFile.EdfFile(fname)
        data = edf.GetData(0)
        edf = None
        imageList.append(data)
    if len(imageList):
        d.setData(imageList)
    d.show()
    app.exec_()
