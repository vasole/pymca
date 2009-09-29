#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
from PyMca_Icons import IconDict
import MaskImageWidget
import ScanWindow
import sys
import SNIPModule

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))
class SNIPParametersWidget(qt.QWidget):
    def __init__(self, parent = None, length=2000):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)

        i = 0
        self.parametersDict = {'xmin':0,
                               'xmax':length,
                               'width':min(30, length/10),
                               'smoothing':1}                           
        for text in ["SNIP background width (2 to 3 times fwhm) :",
                     "Minimum channel considered:",
                     "Maximum channel considered:",
                     "Pleliminar smoothing level:"]:
            label = qt.QLabel(self)
            label.setText(text)
            self.mainLayout.addWidget(label, i, 0)        
            #self.mainLayout.addWidget(HorizontalSpacer(self), i, 1)
            i +=1 

        i = 0
        self.widgetDict = {}
        for key in ['width', 'xmin', 'xmax', 'smoothing']:
            self.widgetDict[key] = qt.QSpinBox(self)
            self.widgetDict[key].setMinimum(0)
            self.widgetDict[key].setMaximum(self.parametersDict['xmax'])
            self.widgetDict[key].setValue(self.parametersDict[key])
            self.connect(self.widgetDict[key],
                     qt.SIGNAL("valueChanged(int)"),
                     self._updateParameters)
            self.mainLayout.addWidget(self.widgetDict[key], i, 1)
            i += 1
        self.widgetDict['smoothing'].setMaximum(100)

    def _updateParameters(self, val):
        for key in ['width', 'xmin', 'xmax', 'smoothing']:
            self.parametersDict[key] = self.widgetDict[key].value()
        ddict = {}
        ddict['event']='SNIPParametersChanged'
        ddict.update(self.parametersDict)
        self.emit(qt.SIGNAL('SNIPParametersSignal'), ddict)
                  
    def getParameters(self):
        return self.parametersDict


class SNIPWindow(qt.QWidget):
    def __init__(self, parent, spectrum):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("SNIP Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        self.parametersWidget = SNIPParametersWidget(self, length=len(spectrum))
        self.graph = ScanWindow.ScanWindow(self)
        self.graph.newCurve(range(len(spectrum)),
                            spectrum, "Spectrum", replace=True)
        self.mainLayout.addWidget(self.parametersWidget)
        self.mainLayout.addWidget(self.graph)
        self.spectrum = spectrum
        self.getParameters = self.parametersWidget.getParameters
        self.connect(self.parametersWidget,
                     qt.SIGNAL('SNIPParametersSignal'),
                     self.updateGraph)
        self.updateGraph(self.getParameters())

    def updateGraph(self, ddict):
        width = ddict['width']
        chmin = ddict['xmin']
        chmax = ddict['xmax']
        smoothing = ddict['smoothing']
        self.background = SNIPModule.getSpectrumBackground(self.spectrum, width,
                                                   chmin=chmin,
                                                   chmax=chmax,
                                                   smoothing=smoothing)
        self.graph.newCurve(range(len(self.spectrum)),
                            self.background, "Background", replace=False)


class SNIPDialog(qt.QDialog):
    def __init__(self, parent, spectrum):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("SNIP Configuration Dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(10)
        self.mainLayout.setSpacing(2)
        self.parametersWidget = SNIPWindow(self, spectrum)
        self.mainLayout.addWidget(self.parametersWidget)

        hbox = qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.setMargin(0)
        hboxLayout.setSpacing(0)
        self.okButton = qt.QPushButton(hbox)
        self.okButton.setText("OK")
        self.okButton.setAutoDefault(False)   
        self.dismissButton = qt.QPushButton(hbox)
        self.dismissButton.setText("Cancel")
        self.dismissButton.setAutoDefault(False)
        hboxLayout.addWidget(self.okButton)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        hboxLayout.addWidget(self.dismissButton)
        self.mainLayout.addWidget(hbox)
        self.connect(self.dismissButton, qt.SIGNAL("clicked()"), self.reject)
        self.connect(self.okButton, qt.SIGNAL("clicked()"), self.accept)

    def getParameters(self):
        parametersDict = self.parametersWidget.getParameters()
        parametersDict['function'] = SNIPModule.subtractBackgroundFromStack
        parametersDict['arguments'] = (parametersDict['width'],
                                       parametersDict['xmin'],
                                       parametersDict['xmax'],
                                       parametersDict['smoothing'])
        return parametersDict                                       
                 
if __name__ == "__main__":
    import numpy
    app = qt.QApplication([])
    noise = numpy.random.randn(1000.) 
    y=numpy.arange(1000.)
    w = SNIPDialog(None, y+numpy.sqrt(y)* noise)
    w.show()
    w.exec_()
