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
import PyMcaQt as qt
from PyMca_Icons import IconDict
import MaskImageWidget
#RGBCorrelatorGraph
import ScanWindow
import sys
import SGModule

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))
class SGParametersWidget(qt.QWidget):
    def __init__(self, parent = None, length=2000):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)

        i = 0
        self._keyList = ['points', 'degree', 'order']
        self.parametersDict = {'points':3,
                               'degree':1,
                               'order':0}
        for text in ["Savitzky-Golay filter width:",
                     "Interpolating polynomial degree:",
                     "Derivative order (0=smoothing):"]:
            label = qt.QLabel(self)
            label.setText(text)
            self.mainLayout.addWidget(label, i, 0)        
            #self.mainLayout.addWidget(HorizontalSpacer(self), i, 1)
            i +=1 

        i = 0
        self.widgetDict = {}
        for key in self._keyList:
            self.widgetDict[key] = qt.QSpinBox(self)
            self.widgetDict[key].setMinimum(1)
            self.widgetDict[key].setMaximum(100)
            self.widgetDict[key].setValue(self.parametersDict[key])
            self.connect(self.widgetDict[key],
                     qt.SIGNAL("valueChanged(int)"),
                     self._updateParameters)
            self.mainLayout.addWidget(self.widgetDict[key], i, 1)
            i += 1
        self.widgetDict['order'].setMinimum(0)
        self.widgetDict['order'].setValue(0)
        self.widgetDict['order'].setMaximum(4)

    def _updateParameters(self, val):
        for key in self._keyList:
            self.parametersDict[key] = self.widgetDict[key].value()
        self.widgetDict['order'].setMaximum(self.parametersDict['degree'])
        if self.parametersDict['order'] > self.parametersDict['degree']:
            self.parametersDict['order']=self.parametersDict['degree']
            self.widgetDict['order'].setValue(self.parametersDict['order'])
        ddict = {}
        ddict['event']='SGParametersChanged'
        ddict.update(self.parametersDict)
        self.emit(qt.SIGNAL('SGParametersSignal'), ddict)
                  
    def getParameters(self):
        return self.parametersDict

class SGWindow(qt.QWidget):
    def __init__(self, parent, data, image=None, x=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("Savitzky-Golay Filter Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        spectrum = data
        if x is None:
            self.xValues = range(len(spectrum))
        else:
            self.xValues = x
        self.image = None
        self.spectrum = spectrum
        self.parametersWidget = SGParametersWidget(self, length=len(spectrum))
        self.graph = ScanWindow.ScanWindow(self)
        self.graph.newCurve(self.xValues,
                        spectrum, "Spectrum", replace=True)
        self.mainLayout.addWidget(self.parametersWidget)
        self.mainLayout.addWidget(self.graph)
        self.getParameters = self.parametersWidget.getParameters
        self.connect(self.parametersWidget,
                     qt.SIGNAL('SGParametersSignal'),
                     self.updateGraph)
        self.updateGraph(self.getParameters())

    def updateGraph(self, ddict):
        points = ddict['points']
        degree = ddict['degree']
        order  = ddict['order']
        self.background = SGModule.getSavitzkyGolay(self.spectrum,
                                                    points,
                                                    degree=degree,
                                                    order=order)
        if order > 0:
            maptoy2 = True
        else:
            maptoy2 = False
        self.graph.newCurve(self.xValues,
                    self.background, "Filtered Spectrum",
                    replace=False,
                    maptoy2=maptoy2)
    
        #Force information update
        legend = self.graph.getActiveCurve(just_legend=True)
        if legend.startswith('Filtered'):
            self.graph.setActiveCurve(legend)

class SGDialog(qt.QDialog):
    def __init__(self, parent, data, x=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Savitzky-Golay Configuration Dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(10)
        self.mainLayout.setSpacing(2)
        self.__image = False
        if len(data.shape) == 2:
            spectrum = data.ravel()
        else:
            spectrum = data
        self.parametersWidget = SGWindow(self, spectrum, image=False, x=x)
        self.graph = self.parametersWidget.graph
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
        parametersDict['function'] = SGModule.replaceStackWithSavitzkyGolay
        parametersDict['arguments'] = [parametersDict['points'],
                                       parametersDict['degree'],
                                       parametersDict['order']]
        return parametersDict                                       
                 
if __name__ == "__main__":
    import numpy
    app = qt.QApplication([])
    if 1:
        noise = numpy.random.randn(1000.) 
        y=numpy.arange(1000.)
        w = SGDialog(None, y+numpy.sqrt(y)* noise)
    w.show()
    ret=w.exec_()
    if ret:
        print w.getParameters()
