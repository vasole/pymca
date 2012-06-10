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
import numpy
import traceback
import copy
from PyMca import PyMcaQt as qt
from PyMca.PyMca_Icons import IconDict
from PyMca import ScanWindow
try:
    from PyMca import XASNormalization
except:
    print("WARNING: XASNormalizationWindow performing local import")
    from . import XASNormalization

class PolynomSelector(qt.QComboBox):
    def __init__(self, parent=None, options=None):
        qt.QComboBox.__init__(self, parent)
        self.setEditable(0)
        if options is not None:
            self.setOptions(options)

    def setOptions(self, options):
        for item in options:
            self.addItem(item)

class XASNormalizationParametersWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self.__parametersDict = self._getDefaultParameters()
        self.__defaultEdgeEnergy = None
        self._polynomOptions = ['Modif. Victoreen',
                                'Victoreen',
                                'Constant',
                                'Linear',
                                'Parabolic',
                                'Cubic']

        i = 0
        edgeGroupBox = qt.QGroupBox(self)
        edgeGroupBoxLayout = qt.QGridLayout(edgeGroupBox)
        edgeGroupBox.setAlignment(qt.Qt.AlignHCenter)
        edgeGroupBox.setTitle('Edge Position')
        autoButton = qt.QRadioButton(edgeGroupBox)
        autoButton.setText('Auto')
        autoButton.setChecked(True)
        userButton = qt.QRadioButton(edgeGroupBox)
        userButton.setText('User')
        buttonGroup = qt.QButtonGroup(edgeGroupBox)
        buttonGroup.addButton(autoButton, 0)
        buttonGroup.addButton(userButton, 1)
        buttonGroup.setExclusive(True)
        userEnergy = qt.QLineEdit(edgeGroupBox)
        userEnergy.setEnabled(False)
        validator = qt.QDoubleValidator(userEnergy)
        userEnergy.setValidator(validator)
        edgeGroupBoxLayout.addWidget(autoButton, 0, 0)
        edgeGroupBoxLayout.addWidget(userButton, 1, 0)
        edgeGroupBoxLayout.addWidget(userEnergy, 1, 1)
        self.mainLayout.addWidget(edgeGroupBox, 0, 0, 2, 2)

        #create handles to the relevant widgets
        self.autoEdgeButton = autoButton
        self.userEdgeButton = userButton
        self.userEdgeEnergy = userEnergy

        # connect the signals
        self.connect(buttonGroup,
                     qt.SIGNAL('buttonClicked(int)'),
                     self._buttonClicked)

        self.connect(self.userEdgeEnergy,
                     qt.SIGNAL('editingFinished()'),
                     self._userEdgeEnergyEditingFinished)

        regionsGroupBox = qt.QGroupBox(self)
        regionsGroupBoxLayout = qt.QGridLayout(regionsGroupBox)
        regionsGroupBox.setAlignment(qt.Qt.AlignHCenter)
        regionsGroupBox.setTitle('Regions')

        i = 1
        for text in ["Pre-edge Polynom:",
                     "Post-edge Polynom:"]:
            label = qt.QLabel(regionsGroupBox)
            label.setText(text)
            regionsGroupBoxLayout.addWidget(label, i, 0)
            #self.mainLayout.addWidget(qt.HorizontalSpacer(self), i, 1)
            i +=1 

        i = 1
        self.widgetDict = {}
        for key in ['pre_edge',
                    'post_edge']:
            self.widgetDict[key] = {}
            c = 1
            w = PolynomSelector(regionsGroupBox, options=self._polynomOptions)
            self.connect(w,
                         qt.SIGNAL('activated(int)'),
                         self._regionParameterChanged)
            regionsGroupBoxLayout.addWidget(w, i, c)
            c += 1
            self.widgetDict[key]['polynomial'] = w
            for text in ['delta xmin', 'delta xmax']:
                label = qt.QLabel(regionsGroupBox)
                label.setText(text)
                self.widgetDict[key][text] = qt.QLineEdit(regionsGroupBox)
                self.connect(self.widgetDict[key][text],
                             qt.SIGNAL('editingFinished()'),
                             self._regionParameterChanged)
                validator = qt.QDoubleValidator(self.widgetDict[key][text])
                self.widgetDict[key][text].setValidator(validator)
                regionsGroupBoxLayout.addWidget(label, i, c)
                regionsGroupBoxLayout.addWidget(self.widgetDict[key][text], i, c + 1)
                c += 2
            i += 1
        self.mainLayout.addWidget(regionsGroupBox, 0, 2)
        self._updateParameters()
        
    def _getDefaultParameters(self):
        ddict = {}
        ddict['auto_edge'] = 1
        #give a dummy value
        ddict['edge_energy'] = 0.0
        ddict['pre_edge'] = {}
        ddict['pre_edge']['regions'] = [[-200., -50.]]
        ddict['pre_edge']['polynomial'] = 'Constant'
        ddict['post_edge'] = {}
        ddict['post_edge']['regions'] = [[20., 500.]]
        ddict['post_edge']['polynomial'] = 'Linear'
        return ddict

    def _buttonClicked(self, intValue):
        event = None
        ddict={}
        if intValue == 0:
            event = "AutoEdgeEnergyClicked"
            ddict['auto_edge'] = 1
        else:
            ddict['auto_edge'] = 0
        self.setParameters(ddict, signal=True, event=event)

    def _userEdgeEnergyEditingFinished(self):
        ddict={}
        ddict['edge_energy'] = float(self.userEdgeEnergy.text())
        self.setParameters(ddict, signal=True)

    def _regionParameterChanged(self, dummy=None, signal=True):
        ddict = {}
        for key in ['pre_edge', 'post_edge']:
            ddict[key] = {}
            ddict[key]['polynomial'] = self.widgetDict[key]['polynomial'].currentIndex() - 2
            delta_xmin = float(self.widgetDict[key]['delta xmin'].text())
            delta_xmax = float(self.widgetDict[key]['delta xmax'].text())
            if delta_xmin > delta_xmax:
                ddict[key]['regions'] = [[delta_xmax, delta_xmin]]
                self.widgetDict[key]['delta xmin'].setText("%f" % delta_xmax)
                self.widgetDict[key]['delta xmax'].setText("%f" % delta_xmin)
            else:
                ddict[key]['regions'] = [[delta_xmin, delta_xmax]]
        self.setParameters(ddict, signal=True)

    def setParameters(self, ddict, signal=False, event=None):
        for key in ddict:
            if key in ['pre_edge',
                       'post_edge']:
                self.__parametersDict[key].update(ddict[key])
            elif key in self.__parametersDict:
                self.__parametersDict[key] = ddict[key]
        self._updateParameters(signal=signal, event=event)

    def _updateParameters(self, signal=True, event=None):
        for key in ['pre_edge',
                    'post_edge']:
            idx = self.__parametersDict[key]['polynomial']
            if type(idx) == type(1):
                # polynomial order
                self.widgetDict[key]['polynomial'].setCurrentIndex(idx + 2)
            else:
                # string
                self.widgetDict[key]['polynomial'].setCurrentIndex(\
                    self._polynomOptions.index(idx))
            i = 0
            for text in ['delta xmin', 'delta xmax']:
                # only the first region of each shown
                self.widgetDict[key][text].setText("%f" %\
                            self.__parametersDict[key]['regions'][0][i])
                i += 1
        self.userEdgeEnergy.setText("%f" % self.__parametersDict['edge_energy'])
        if self.__parametersDict['auto_edge']:
            self.autoEdgeButton.setChecked(True)
            self.userEdgeButton.setChecked(False)
            self.userEdgeEnergy.setEnabled(False)
        else:
            self.autoEdgeButton.setChecked(False)
            self.userEdgeButton.setChecked(True)
            self.userEdgeEnergy.setEnabled(True)
        if signal:
            ddict = self.getParameters()
            if event is None:
                ddict['event']='XASNormalizationParametersChanged'
            else:
                ddict['event'] = event
            self.emit(qt.SIGNAL('XASNormalizationParametersSignal'), ddict)
                  
    def getParameters(self):
        # make sure a copy is given back
        return copy.deepcopy(self.__parametersDict)

    def setEdgeEnergy(self, energy):
        self.userEdgeEnergy.setText("%f" % energy)
        self.__parametersDict['edge_energy'] = energy

class XASNormalizationWindow(qt.QWidget):
    def __init__(self, parent, spectrum, energy=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("XAS Normalization Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(2)
        if energy is None:
            self.energy = range(len(spectrum))
        else:
            self.energy = energy
        self.spectrum = spectrum
        self.parametersWidget = XASNormalizationParametersWidget(self)
        self.graph = ScanWindow.ScanWindow(self)
        self.__lastDict = {}
        self.__markerHandling = False
        self.__preEdgeMarkers = [None, None]
        self.__postEdgeMarkers = [None, None]
        self.__edgeMarker = None
        if hasattr(self.graph, "scanWindowInfoWidget"):
            self.graph.scanWindowInfoWidget.hide()
            if hasattr(self.graph, "graph"):
                if hasattr(self.graph.graph, "insertX1Marker"):
                    self.__markerHandling = True
                    self.connect(self.graph.graph,
                                 qt.SIGNAL("QtBlissGraphSignal"),
                                 self._handleGraphSignal)
        self.graph.addCurve(self.energy,
                            spectrum, legend="Spectrum", replace=True)
        self.mainLayout.addWidget(self.parametersWidget)
        self.mainLayout.addWidget(self.graph)
        # initialize variables
        edgeEnergy, sortedX, sortedY, xPrime, yPrime =\
                XASNormalization.estimateXANESEdge(spectrum, energy=self.energy, full=True)
        self._xPrime = xPrime
        self._yPrime = yPrime
        self.parametersWidget.setEdgeEnergy(edgeEnergy)
        self.getParameters = self.parametersWidget.getParameters
        self.setParameters = self.parametersWidget.setParameters
        self.connect(self.parametersWidget,
                     qt.SIGNAL('XASNormalizationParametersSignal'),
                     self.updateGraph)
        self.updateGraph(self.getParameters())


    def setData(self, spectrum, energy=None):
        self.spectrum = spectrum
        if energy is None:
            self.energy = range(len(spectrum))
        else:
            self.energy = energy
        if self.__markerHandling:
            self.graph.graph.clearMarkers()
            self.__preEdgeMarkers = [None, None]
            self.__postEdgeMarkers = [None, None]
            self.__edgeMarker = None
        self.graph.addCurve(self.energy,
                            self.spectrum,
                            legend="Spectrum",
                            replot=True,
                            replace=True)
        edgeEnergy = XASNormalization.estimateXANESEdge(self.spectrum,
                                                        energy=self.energy,
                                                        full=False)
        self.parametersWidget.setEdgeEnergy(edgeEnergy)
        self.updateGraph(self.getParameters())

    def updateGraph(self, ddict):
        self.__lastDict = ddict
        edgeEnergy = ddict['edge_energy']
        preRegions = ddict['pre_edge']['regions']
        postRegions = ddict['post_edge']['regions']
        event = ddict.get('event', None)
        if event == "AutoEdgeEnergyClicked":
            try:
                # recalculate edge energy following region limits
                xmin = edgeEnergy + preRegions[0][0]
                xmax = edgeEnergy + postRegions[0][1]
                idx = numpy.nonzero((self.energy >= xmin) &\
                                    (self.energy <= xmax))[0]
                x = numpy.take(self.energy, idx)
                y = numpy.take(self.spectrum, idx)
                edgeEnergy = XASNormalization.estimateXANESEdge(y, energy=x, full=False)
                self.parametersWidget.setEdgeEnergy(edgeEnergy)
                self.__lastDict['edge_energy'] = edgeEnergy
            except:
                pass
        parameters = {}
        parameters['pre_edge_order'] = ddict['pre_edge']['polynomial']
        parameters['post_edge_order'] = ddict['post_edge']['polynomial']
        algorithm = 'polynomial'
        if self.__markerHandling:
            self.updateMarkers(edgeEnergy,
                               preRegions,
                               postRegions,
                               edge_auto=ddict['auto_edge'])
        else:
            self.graph.addCurve([edgeEnergy, edgeEnergy],
                            [self.spectrum.min(), self.spectrum.max()],
                            legend="Edge Position",
                            replace=False)
        try:
            normalizationResult = XASNormalization.XASNormalization(self.spectrum,
                                                            self.energy,
                                                            edge=edgeEnergy,
                                                            pre_edge_regions=preRegions,
                                                            post_edge_regions=postRegions,
                                                            algorithm=algorithm,
                                                            algorithm_parameters=parameters)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setWindowTitle("Normalization Error")
            msg.setText("An error has occured while normalizing the data")
            msg.setInformativeText(str(sys.exc_info()[1]))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            return
            
        nEnergy, nSpectrum, usedEdge, jump = normalizationResult[0:4]
        preEdgeFunction, preEdgeParameters = normalizationResult[4:6]
        postEdgeFunction, postEdgeParameters = normalizationResult[6:8]
        idx = self.energy > (usedEdge + preRegions[0][0])
        x = self.energy[idx]
        yPre = preEdgeFunction(preEdgeParameters, x)
        yPost = postEdgeFunction(postEdgeParameters, x)
        self.graph.addCurve(x,
                            yPre,
                            legend="Pre-edge Polynomial",
                            replace=False)
        self.graph.addCurve(x,
                            yPost+yPre,
                            legend="Post-edge Polynomial",
                            replace=False,
                            replot=True)

    def updateMarkers(self, edgeEnergy, preEdgeRegions, postEdgeRegions, edge_auto=True):
        if not self.__markerHandling:
            return
        if self.__edgeMarker is None:
            self.__edgeMarker = self.graph.graph.insertX1Marker(edgeEnergy,
                                    0.5 * (self.spectrum.max() + self.spectrum.min()),
                                    label='EDGE')
            self.graph.graph.setmarkercolor(self.__edgeMarker, 'pink')
            self.graph.graph.enablemarkermode()
        else:
            self.graph.graph.setMarkerXPos(self.__edgeMarker, edgeEnergy)
        if edge_auto:
            self.graph.graph.setmarkerfollowmouse(self.__edgeMarker, 0)
        else:
            self.graph.graph.setmarkerfollowmouse(self.__edgeMarker, 1)            
        for i in range(2):
            x = preEdgeRegions[0][i] + edgeEnergy
            if self.__preEdgeMarkers[i] is None:
                if i == 0:
                    label = 'MIN'
                else:
                    label = 'MAX'
                self.__preEdgeMarkers[i] = self.graph.graph.insertX1Marker(x,
                                    0.5 * (self.spectrum.max() + self.spectrum.min()),
                                    label=label)
                self.graph.graph.setmarkercolor(self.__preEdgeMarkers[i], 'blue')
                self.graph.graph.setmarkerfollowmouse(self.__preEdgeMarkers[i], 1)
            else:
                self.graph.graph.setMarkerXPos(self.__preEdgeMarkers[i], x)
        for i in range(2):
            x = postEdgeRegions[0][i] + edgeEnergy
            if self.__postEdgeMarkers[i] is None:
                if i == 0:
                    label = 'MIN'
                else:
                    label = 'MAX'
                self.__postEdgeMarkers[i] = self.graph.graph.insertX1Marker(x,
                                    0.5 * (self.spectrum.max() + self.spectrum.min()),
                                    label=label)
                self.graph.graph.setmarkercolor(self.__postEdgeMarkers[i], 'blue')
                self.graph.graph.setmarkerfollowmouse(self.__postEdgeMarkers[i], 1)
            else:
                self.graph.graph.setMarkerXPos(self.__postEdgeMarkers[i], x)

    def _handleGraphSignal(self, ddict):
        if ddict['event'] != 'markerMoved':
            return
        marker = ddict['marker']
        edgeEnergy =  self.__lastDict['edge_energy']
        x = ddict['x']
        ddict ={}
        if marker == self.__edgeMarker:
            ddict['edge_energy'] = x
        elif marker == self.__preEdgeMarkers[0]:
            ddict['pre_edge'] ={}
            xmin = x - edgeEnergy
            xmax = self.__lastDict['pre_edge']['regions'][0][1]
            if xmin > xmax:
                ddict['pre_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['pre_edge']['regions'] = [[xmin, xmax]]
        elif marker == self.__preEdgeMarkers[1]:
            ddict['pre_edge'] ={}
            xmin = self.__lastDict['pre_edge']['regions'][0][0]
            xmax = x - edgeEnergy
            if xmin > xmax:
                ddict['pre_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['pre_edge']['regions'] = [[xmin, xmax]]
        elif marker == self.__postEdgeMarkers[0]:
            ddict['post_edge'] ={}
            xmin = x - edgeEnergy
            xmax = self.__lastDict['post_edge']['regions'][0][1]
            if xmin > xmax:
                ddict['post_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['post_edge']['regions'] = [[xmin, xmax]]
        elif marker == self.__postEdgeMarkers[1]:
            ddict['post_edge'] ={}
            xmin = self.__lastDict['post_edge']['regions'][0][0]
            xmax = x - edgeEnergy
            if xmin > xmax:
                ddict['post_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['post_edge']['regions'] = [[xmin, xmax]]
        else:
            print("Unhandled markerMoved Signal")
            return
        self.setParameters(ddict, signal=True)

class XASNormalizationDialog(qt.QDialog):
    def __init__(self, parent, data, energy=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("XAS Normalization Dialog")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(10)
        self.mainLayout.setSpacing(2)
        self.__image = False
        if len(data.shape) == 2:
            spectrum = data.ravel()
        else:
            spectrum = data
        self.parametersWidget =XASNormalizationWindow(self, spectrum, energy=energy)
        self.graph = self.parametersWidget.graph
        self.setData = self.parametersWidget.setData
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
        hboxLayout.addWidget(qt.HorizontalSpacer(hbox))
        hboxLayout.addWidget(self.dismissButton)
        self.mainLayout.addWidget(hbox)
        self.connect(self.dismissButton, qt.SIGNAL("clicked()"), self.reject)
        self.connect(self.okButton, qt.SIGNAL("clicked()"), self.accept)

    def getParameters(self):
        parametersDict = self.parametersWidget.getParameters()
        return parametersDict                                       

    def setParameters(self, ddict):
        return self.parametersWidget.setParameters(ddict)

                 
if __name__ == "__main__":
    app = qt.QApplication([])
    if len(sys.argv) > 1:
        from PyMca import specfilewrapper as specfile
        sf = specfile.Specfile(sys.argv[1])
        scan = sf[0]
        data = scan.data()
        energy = data[0, :]
        spectrum = data[1, :]
        w = XASNormalizationDialog(None, spectrum, energy=energy)
    else:
        from PyMca import SpecfitFuns
        noise = numpy.random.randn(1500.) 
        x = 8000. + numpy.arange(1500.)
        y = SpecfitFuns.upstep([100, 8500., 50], x)
        w = XASNormalizationDialog(None, y + numpy.sqrt(y)* noise, energy=x)
    ret=w.exec_()
    if ret:
        print(w.getParameters())

