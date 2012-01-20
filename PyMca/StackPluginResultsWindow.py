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
__author__ = "V.A. Sole - ESRF BLISS Group"
import PyMcaQt as qt
HorizontalSpacer = qt.HorizontalSpacer
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
from PyMca_Icons import IconDict
import MaskImageWidget
import ScanWindow
import sys
import numpy
MATPLOTLIB = MaskImageWidget.MATPLOTLIB
QTVERSION = MaskImageWidget.QTVERSION


class StackPluginResultsWindow(MaskImageWidget.MaskImageWidget):
    def __init__(self, *var, **kw):
        ddict = {}
        ddict['usetab'] = kw.get("usetab",True)
        ddict.update(kw)
        ddict['standalonesave'] = False
        MaskImageWidget.MaskImageWidget.__init__(self, *var, **ddict) 
        self.slider = qt.QSlider(self)
        self.slider.setOrientation(qt.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        if ddict['usetab']:
            # The 1D graph
            self.spectrumGraph = ScanWindow.ScanWindow(self)
            self.mainTab.addTab(self.spectrumGraph, "VECTORS")
        
        self.mainLayout.addWidget(self.slider)
        self.connect(self.slider,
                     qt.SIGNAL("valueChanged(int)"),
                     self._showImage)

        self.imageList = None
        self.spectrumList = None
        self.spectrumNames = None
        self.spectrumGraphTitles = None
        standalonesave = kw.get("standalonesave", True)
        if standalonesave:
            self.connect(self.graphWidget.saveToolButton,
                         qt.SIGNAL("clicked()"), 
                         self._saveToolButtonSignal)
            self._saveMenu = qt.QMenu()
            self._saveMenu.addAction(QString("Image Data"),
                                     self.saveImageList)
            self._saveMenu.addAction(QString("Standard Graphics"),
                                     self.graphWidget._saveIconSignal)
            if QTVERSION > '4.0.0':
                if MATPLOTLIB:
                    self._saveMenu.addAction(QString("Matplotlib") ,
                                     self._saveMatplotlibImage)
        self.multiplyIcon = qt.QIcon(qt.QPixmap(IconDict["swapsign"]))
        infotext = "Multiply image by -1"
        self.multiplyButton = self.graphWidget._addToolButton(\
                                        self.multiplyIcon,
                                        self._multiplyIconChecked,
                                        infotext,
                                        toggle = False,
                                        position = 12)

    def sizeHint(self):
        return qt.QSize(400, 400)

    def _multiplyIconChecked(self):
        if self.imageList is None:
            return
        index = self.slider.value()
        self.imageList[index] *= -1
        if self.spectrumList is not None:
            self.spectrumList[index] *= -1

        self._showImage(index)

    def _showImage(self, index):
        if len(self.imageList):
            self.showImage(index, moveslider=False)
        if self.spectrumList is not None:
            legend = self.spectrumNames[index]
            x = self.xValues[index]
            y = self.spectrumList[index]
            self.spectrumGraph.newCurve(x, y, legend, replace=True)
            if self.spectrumGraphTitles is not None:
                self.spectrumGraph.graph.setTitle(self.spectrumGraphTitles[index])
                
            
    def showImage(self, index=0, moveslider=True):
        if self.imageList is None:
            return
        if len(self.imageList) == 0:
            return
        self.setImageData(self.imageList[index])
        self.graphWidget.graph.setTitle(self.imageNames[index])
        if moveslider:
            self.slider.setValue(index)

    def setStackPluginResults(self, images, spectra=None,
                   image_names = None, spectra_names = None,
                   xvalues=None, spectra_titles=None):
        self.spectrumList = spectra
        if type(images) == type([]):
            self.imageList = images
            if image_names is None:
                self.imageNames = []
                for i in range(nimages):
                    self.imageNames.append("Image %02d" % i)
            else:
                self.imageNames = image_names
        elif len(images.shape) == 3:
            nimages = images.shape[0]
            self.imageList = [0] * nimages
            for i in range(nimages):
                self.imageList[i] = images[i,:]
                if 0:
                    #leave the data as they originally come
                    if self.imageList[i].max() < 0:
                        self.imageList[i] *= -1
                        if self.spectrumList is not None:
                            self.spectrumList [i] *= -1
            if image_names is None:
                self.imageNames = []
                for i in range(nimages):
                    self.imageNames.append("Image %02d" % i)
            else:
                self.imageNames = image_names
                
        if self.imageList is not None:
            self.slider.setMaximum(len(self.imageList)-1)
            self.showImage(0)
        else:
            self.slider.setMaximum(0)

        if self.spectrumList is not None:
            if spectra_names is None:
                self.spectrumNames = []
                for i in range(nimages):
                    self.spectrumNames.append("Spectrum %02d" % i)
            else:
                self.spectrumNames = spectra_names
            if xvalues is None:
                self.xValues = []
                for i in range(nimages):
                    self.xValues.append(numpy.arange(len(self.spectrumList[0])))
            else:
                self.xValues = xValues
            self.spectrumGraphTitles = spectra_titles
            legend = self.spectrumNames[0]
            x = self.xValues[0]
            y = self.spectrumList[0]
            self.spectrumGraph.newCurve(x, y, legend, replace=True)
            if self.spectrumGraphTitles is not None:
                self.spectrumGraph.graph.setTitle(self.spectrumGraphTitles[0])
            
        self.slider.setValue(0)


    def saveImageList(self, filename=None, imagelist=None, labels=None):
        if self.imageList is None:
            return
        labels = []
        for i in range(len(self.imageList)):
            labels.append(self.imageNames[i].replace(" ","_"))
        return MaskImageWidget.MaskImageWidget.saveImageList(self,
                                                             imagelist=self.imageList,
                                                             labels=labels)
    def setImageList(self, imagelist):
        self.imageList = imagelist
        self.spectrumList = None
        if imagelist is not None:
            self.slider.setMaximum(len(self.imageList)-1)
            self.showImage(0)
            

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

    container = StackPluginResultsWindow()
    data = numpy.arange(20000)
    data.shape = 2, 100, 100
    data[1, 0:100,0:50] = 100
    container.setStackPluginResults(data, spectra=[numpy.arange(100.), numpy.arange(100.)+10],
                                image_names=["I1", "I2"], spectra_names=["V1", "V2"])
    container.show()
    def theSlot(ddict):
        print(ddict['event'])

    if QTVERSION < '4.0.0':
        qt.QObject.connect(container,
                       qt.PYSIGNAL("MaskImageWidgetSignal"),
                       updateMask)
        app.setMainWidget(container)
        app.exec_loop()
    else:
        qt.QObject.connect(container,
                           qt.SIGNAL("MaskImageWidgetSignal"),
                           theSlot)
        app.exec_()

if __name__ == "__main__":
    import numpy
    test()
        
