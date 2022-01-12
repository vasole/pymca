#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V. Armando Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import numpy
import traceback
import copy
from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMca_Icons
IconDict = PyMca_Icons.IconDict
from PyMca5.PyMcaGraph.backends.MatplotlibBackend \
     import MatplotlibBackend as backend
from PyMca5.PyMcaGui import PlotWindow
from PyMca5.PyMcaPhysics import XASNormalization

POLYNOM_OPTIONS = ['Modif. Victoreen',
                   'Victoreen',
                   'Constant',
                   'Linear',
                   'Parabolic',
                   'Cubic']

class PolynomSelector(qt.QComboBox):
    def __init__(self, parent=None, options=None):
        qt.QComboBox.__init__(self, parent)
        self.setEditable(0)
        if options is not None:
            self.setOptions(options)
        else:
            self.setOptions(POLYNOM_OPTIONS)

    def setOptions(self, options):
        for item in options:
            self.addItem(item)

    def getOptions(self):
        return POLYNOM_OPTIONS * 1

class XASNormalizationParametersWidget(qt.QWidget):
    sigXASNormalizationParametersSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self.__parametersDict = self._getDefaultParameters()
        self.__defaultEdgeEnergy = None
        self._polynomOptions = POLYNOM_OPTIONS
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
        validator = qt.CLocaleQDoubleValidator(userEnergy)
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
        if hasattr(buttonGroup, "idClicked"):
            buttonGroup.idClicked[int].connect(self._buttonClicked)
        else:
            # deprecated
            buttonGroup.buttonClicked[int].connect(self._buttonClicked)

        self.userEdgeEnergy.editingFinished.connect(self._userEdgeEnergyEditingFinished)

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
            w.activated[int].connect(self._regionParameterChanged)
            regionsGroupBoxLayout.addWidget(w, i, c)
            c += 1
            self.widgetDict[key]['polynomial'] = w
            for text in ['delta xmin', 'delta xmax']:
                label = qt.QLabel(regionsGroupBox)
                label.setText(text)
                self.widgetDict[key][text] = qt.QLineEdit(regionsGroupBox)
                self.widgetDict[key][text].editingFinished.connect( \
                             self._regionParameterChanged)
                validator = qt.CLocaleQDoubleValidator(self.widgetDict[key][text])
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
        ddict['pre_edge']['regions'] = [[-100., -40.]]
        ddict['pre_edge']['polynomial'] = 'Constant'
        ddict['post_edge'] = {}
        ddict['post_edge']['regions'] = [[20., 300.]]
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
            self.sigXASNormalizationParametersSignal.emit(ddict)

    def getParameters(self):
        # make sure a copy is given back
        return copy.deepcopy(self.__parametersDict)

    def setEdgeEnergy(self, energy, emin=None, emax=None):
        self.userEdgeEnergy.setText("%f" % energy)
        signal = True
        if self.__parametersDict['edge_energy'] == energy:
            signal = False
        ddict ={'edge_energy':energy}
        if emin is not None:
            for region in self.__parametersDict['pre_edge']['regions']:
                if (region[0] + energy) < emin:
                    signal = True
                    ddict['pre_edge'] = {}
                    xmin = emin - energy
                    xmax = 0.5 * xmin
                    ddict['pre_edge']['regions'] = [[xmin, xmax]]
                    break
        if emax is not None:
            for region in self.__parametersDict['post_edge']['regions']:
                if (region[1] + energy) > emax:
                    signal=True
                    ddict['post_edge'] = {}
                    xmax = emax - energy
                    xmin = 0.1 * xmax
                    ddict['post_edge']['regions'] = [[xmin, xmax]]
                    break
        self.setParameters(ddict, signal=signal)

class XASNormalizationWindow(qt.QWidget):
    def __init__(self, parent, spectrum, energy=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("XAS Normalization Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        if energy is None:
            self.energy = numpy.arange(len(spectrum)).astype(numpy.float64)
        else:
            self.energy = energy
        self.spectrum = numpy.array(spectrum, dtype=numpy.float64, copy=False) 
        self.parametersWidget = XASNormalizationParametersWidget(self)
        self.graph = PlotWindow.PlotWindow(self, backend=backend,
                                           plugins=False, newplot=False)
        self.__lastDict = {}
        self.graph.sigPlotSignal.connect(self._handleGraphSignal)
        self.graph.addCurve(self.energy,
                            spectrum, legend="Spectrum", replace=True)
        self.mainLayout.addWidget(self.parametersWidget)
        self.mainLayout.addWidget(self.graph)
        # initialize variables
        edgeEnergy, sortedX, sortedY, xPrime, yPrime =\
                XASNormalization.estimateXANESEdge(spectrum, energy=self.energy, full=True)
        self._xPrime = xPrime
        self._yPrime = yPrime
        self.parametersWidget.setEdgeEnergy(edgeEnergy,
                                            emin=self.energy.min(),
                                            emax=self.energy.max())
        self.getParameters = self.parametersWidget.getParameters
        self.setParameters = self.parametersWidget.setParameters
        self.parametersWidget.sigXASNormalizationParametersSignal.connect( \
                     self.updateGraph)
        self.updateGraph(self.getParameters())

    def setData(self, spectrum, energy=None):
        self.spectrum = spectrum
        if energy is None:
            self.energy = numpy.arange(len(spectrum)).astype(numpy.float64)
        else:
            self.energy = energy
        self.graph.clearMarkers()
        self.graph.addCurve(self.energy,
                            self.spectrum,
                            legend="Spectrum",
                            replot=True,
                            replace=True)
        edgeEnergy = XASNormalization.estimateXANESEdge(self.spectrum,
                                                        energy=self.energy,
                                                        full=False)
        self.parametersWidget.setEdgeEnergy(edgeEnergy,
                                            emin=self.energy.min(),
                                            emax=self.energy.max())
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
                edgeEnergy = XASNormalization.estimateXANESEdge(y,
                                                                energy=x,
                                                                full=False)
                self.parametersWidget.setEdgeEnergy(edgeEnergy,
                                            emin=self.energy.min(),
                                            emax=self.energy.max())
                self.__lastDict['edge_energy'] = edgeEnergy
            except:
                pass
        parameters = {}
        parameters['pre_edge_order'] = ddict['pre_edge']['polynomial']
        parameters['post_edge_order'] = ddict['post_edge']['polynomial']
        algorithm = 'polynomial'
        self.updateMarkers(edgeEnergy,
                           preRegions,
                           postRegions,
                           edge_auto=ddict['auto_edge'])
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
            msg.exec()
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
        if edge_auto:
            draggable = False
        else:
            draggable = True
        #self.graph.clearMarkers()
        self.graph.insertXMarker(edgeEnergy,
                                'EDGE',
                                 text='EDGE',
                                 color='pink',
                                 draggable=draggable,
                                 replot=False)
        for i in range(2):
            x = preEdgeRegions[0][i] + edgeEnergy
            if i == 0:
                label = 'MIN'
            else:
                label = 'MAX'
            self.graph.insertXMarker(x,
                                'Pre-'+ label,
                                text=label,
                                color='blue',
                                draggable=True,
                                replot=False)
        for i in range(2):
            x = postEdgeRegions[0][i] + edgeEnergy
            if i == 0:
                label = 'MIN'
                replot=False
            else:
                label = 'MAX'
                replot=True
            self.graph.insertXMarker(x,
                                'Post-'+ label,
                                text=label,
                                color='blue',
                                draggable=True,
                                replot=replot)

    def _handleGraphSignal(self, ddict):
        #print("ddict = ", ddict)
        if ddict['event'] != 'markerMoved':
            return
        marker = ddict['label']
        edgeEnergy =  self.__lastDict['edge_energy']
        x = ddict['x']
        if marker == "EDGE":
            self.parametersWidget.setEdgeEnergy(x,
                                    emin=self.energy.min(),
                                    emax=self.energy.max())
            return

        ddict ={}
        if marker == "Pre-MIN":
            ddict['pre_edge'] ={}
            xmin = x - edgeEnergy
            xmax = self.__lastDict['pre_edge']['regions'][0][1]
            if xmin > xmax:
                ddict['pre_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['pre_edge']['regions'] = [[xmin, xmax]]
        elif marker == "Pre-MAX":
            ddict['pre_edge'] ={}
            xmin = self.__lastDict['pre_edge']['regions'][0][0]
            xmax = x - edgeEnergy
            if xmin > xmax:
                ddict['pre_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['pre_edge']['regions'] = [[xmin, xmax]]
        elif marker == "Post-MIN":
            ddict['post_edge'] ={}
            xmin = x - edgeEnergy
            xmax = self.__lastDict['post_edge']['regions'][0][1]
            if xmin > xmax:
                ddict['post_edge']['regions'] = [[xmax, xmin]]
            else:
                ddict['post_edge']['regions'] = [[xmin, xmax]]
        elif marker == "Post-MAX":
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
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
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
        hboxLayout.setContentsMargins(0, 0, 0, 0)
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
        self.dismissButton.clicked.connect(self.reject)
        self.okButton.clicked.connect(self.accept)

    def getParameters(self):
        parametersDict = self.parametersWidget.getParameters()
        return parametersDict

    def setParameters(self, ddict):
        return self.parametersWidget.setParameters(ddict)

    def setSpectrum(self, energy, mu):
        self.parametersWidget.setData(mu, energy=energy)

if __name__ == "__main__":
    app = qt.QApplication([])
    if len(sys.argv) > 1:
        from PyMca5.PyMcaIO import specfilewrapper as specfile
        sf = specfile.Specfile(sys.argv[1])
        scan = sf[0]
        data = scan.data()
        energy = data[0, :]
        spectrum = data[1, :]
        w = XASNormalizationDialog(None, spectrum, energy=energy)
    else:
        from PyMca5.PyMcaMath.fitting import SpecfitFuns
        noise = numpy.random.randn(1500).astype(numpy.float64)
        x = 8000. + numpy.arange(1500).astype(numpy.float64)
        y = SpecfitFuns.upstep([100, 8500., 50], x)
        w = XASNormalizationDialog(None, y + numpy.sqrt(y)* noise, energy=x)
    ret=w.exec()
    if ret:
        print(w.getParameters())

