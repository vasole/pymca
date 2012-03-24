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
from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()
if QTVERSION < '4.0.0':
    raise ImportError("This module requires Qt4")

HorizontalSpacer = qt.HorizontalSpacer
VerticalSpacer   = qt.VerticalSpacer

class FitFunctionDefinition(qt.QGroupBox):
    def __init__(self, parent=None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle("Function Definition")
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)

        row = 0

        #actual fit function
        self.fitFunctionCheckBox = qt.QCheckBox(self)
        self.fitFunctionCheckBox.setText("Fit Function to be used")
        self.fitFunctionCombo    = qt.QComboBox(self)
        self.fitFunctionCombo.addItem(str("None"))
        self.connect(self.fitFunctionCombo,
                     qt.SIGNAL("activated(int)"),
                     self._fitFunctionComboActivated)
        self.fitFunctionSetupButton = qt.QPushButton(self)
        self.fitFunctionSetupButton.setText('SETUP')
        self.fitFunctionSetupButton.setAutoDefault(False)
        self.fitFunctionSetupButton.hide()

        self.mainLayout.addWidget(self.fitFunctionCheckBox,    row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self),   row, 1)
        self.mainLayout.addWidget(self.fitFunctionSetupButton, row, 2)
        self.mainLayout.addWidget(self.fitFunctionCombo,       row, 3)
        row += 1        

        #background
        self.backgroundCheckBox = qt.QCheckBox(self)
        self.backgroundCheckBox.setText("Background function")
        self.backgroundCombo    = qt.QComboBox(self)
        self.backgroundCombo.addItem(str("None"))
        self.connect(self.backgroundCombo,
                     qt.SIGNAL("activated(int)"),
                     self._backgroundComboActivated)

        self.backgroundSetupButton = qt.QPushButton(self)
        self.backgroundSetupButton.setText('SETUP')
        self.backgroundSetupButton.setAutoDefault(False)
        self.backgroundSetupButton.hide()

        self.mainLayout.addWidget(self.backgroundCheckBox,    row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self),     row, 1)
        self.mainLayout.addWidget(self.backgroundSetupButton, row, 2)
        self.mainLayout.addWidget(self.backgroundCombo,       row, 3)
        row += 1


        #stripping
        self.stripCheckBox = qt.QCheckBox(self)
        self.stripCheckBox.setText("Non-analytical (or estimation) background algorithm")
        self.stripCombo    = qt.QComboBox(self)
        self.stripCombo.addItem(str("Strip"))
        self.stripCombo.addItem(str("SNIP"))
        self.stripSetupButton = qt.QPushButton(self)
        self.stripSetupButton.setText('SETUP')
        self.stripSetupButton.setAutoDefault(False)
        self.connect(self.stripCombo,
                     qt.SIGNAL("activated(int)"),
                     self._stripComboActivated)

        self.mainLayout.addWidget(self.stripCheckBox,       row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self),   row, 1)
        self.mainLayout.addWidget(self.stripSetupButton,    row, 2)
        self.mainLayout.addWidget(self.stripCombo,          row, 3)
        row += 1

        self.snipWidthLabel = qt.QLabel(self)
        self.snipWidthLabel.setText(str("SNIP Background Width"))
        self.snipWidthSpin = qt.QSpinBox(self)
        self.snipWidthSpin.setMinimum(1)
        self.snipWidthSpin.setMaximum(300)
        self.snipWidthSpin.setValue(10)
        self.mainLayout.addWidget(self.snipWidthLabel,     row, 0)
        self.mainLayout.addWidget(self.snipWidthSpin,      row, 3)
        row += 1

        self.stripWidthLabel = qt.QLabel(self)
        self.stripWidthLabel.setText(str("Strip Background Width"))
        self.stripWidthSpin = qt.QSpinBox(self)
        self.stripWidthSpin.setMinimum(1)
        self.stripWidthSpin.setMaximum(100)
        self.stripWidthSpin.setValue(4)
        self.mainLayout.addWidget(self.stripWidthLabel,     row, 0)
        self.mainLayout.addWidget(self.stripWidthSpin,      row, 3)
        row += 1

        self.stripIterLabel = qt.QLabel(self)
        self.stripIterLabel.setText(str("Strip Background Iterations"))
        self.stripIterSpin = qt.QSpinBox(self)
        self.stripIterSpin.setMinimum(0)
        self.stripIterSpin.setMaximum(100000)
        self.stripIterSpin.setValue(5000)
        self.mainLayout.addWidget(self.stripIterLabel,     row, 0)
        self.mainLayout.addWidget(self.stripIterSpin,      row, 3)
        row += 1

        self.stripFilterLabel = qt.QLabel(self)
        text = str("Strip Background Smoothing Width (Savitsky-Golay)")
        self.stripFilterLabel.setText(text)
        self.stripFilterSpin = qt.QSpinBox(self)
        self.stripFilterSpin.setMinimum(0)
        self.stripFilterSpin.setMaximum(40)
        self.stripFilterSpin.setSingleStep(2)
        self.mainLayout.addWidget(self.stripFilterLabel,     row, 0)
        self.mainLayout.addWidget(self.stripFilterSpin,      row, 3)
        row += 1

        #anchors
        self.anchorsContainer = qt.QWidget(self)
        anchorsContainerLayout = qt.QHBoxLayout(self.anchorsContainer)
        anchorsContainerLayout.setMargin(2)
        anchorsContainerLayout.setSpacing(2)
        self.stripAnchorsCheckBox = qt.QCheckBox(self.anchorsContainer)
        self.stripAnchorsCheckBox.setText(str("Strip Background use Anchors"))
        anchorsContainerLayout.addWidget(self.stripAnchorsCheckBox)
        self.stripAnchorsList = []
        for i in range(4):
            anchor = qt.QLineEdit(self.anchorsContainer)
            anchor._v = qt.QDoubleValidator(anchor)
            anchor.setValidator(anchor._v)
            anchor.setText("0.0")
            anchorsContainerLayout.addWidget(anchor)
            self.stripAnchorsList.append(anchor)
        self.mainLayout.addWidget(self.anchorsContainer,      row, 0, 1, 4)
        row += 1


        #signals
        self.connect(self.fitFunctionSetupButton,
                     qt.SIGNAL('clicked()'),
                     self.setupFitFunction)

        self.connect(self.backgroundSetupButton,
                     qt.SIGNAL('clicked()'),
                     self.setupBackground)

        self.connect(self.stripSetupButton,
                     qt.SIGNAL('clicked()'),
                     self.setupStrip)
        
    def _stripComboActivated(self, iValue):
        if iValue == 1:
            self.setSNIP(True)
        else:
            self.setSNIP(False)

    def _fitFunctionComboActivated(self, iValue):
        ddict = {}
        ddict['event'] = "fitFunctionChanged"
        ddict['fit_function'] = str(self.fitFunctionCombo.currentText())
        self.emit(qt.SIGNAL("FitFunctionDefinitionSignal"), ddict)

    def _backgroundComboActivated(self, iValue):
        ddict = {}
        ddict['event'] = "backgroundFunctionChanged"
        ddict['background_function'] = str(self.backgroundCombo.currentText())
        self.emit(qt.SIGNAL("FitFunctionDefinitionSignal"), ddict)

    def setSNIP(self, bValue):
        if bValue:
            self.snipWidthSpin.setEnabled(True)
            self.stripWidthSpin.setEnabled(False)
            self.stripIterSpin.setEnabled(False)
            self.stripCombo.setCurrentIndex(1)
        else:
            self.snipWidthSpin.setEnabled(False)
            self.stripWidthSpin.setEnabled(True)
            self.stripIterSpin.setEnabled(True)
            self.stripCombo.setCurrentIndex(0)

    def setFunctions(self, functionList):
        currentFunction = str(self.fitFunctionCombo.currentText())
        currentBackground = str(self.backgroundCombo.currentText())
        self.fitFunctionCombo.clear()
        self.backgroundCombo.clear()
        self.fitFunctionCombo.addItem('None')
        self.backgroundCombo.addItem('None')
        for key in functionList:
            self.fitFunctionCombo.addItem(str(key))
            self.backgroundCombo.addItem(str(key))

        #restore previous values
        idx = self.fitFunctionCombo.findText(currentFunction)
        self.fitFunctionCombo.setCurrentIndex(idx)
        idx = self.backgroundCombo.findText(currentBackground)
        self.backgroundCombo.setCurrentIndex(idx)

    def getFunctions(self):
        functionList = []
        n = self.fitFunctionCombo.count()
        for i in range(n):
            if i == 0:
                continue
            functionList.append(str(self.fitFunctionCombo.itemText(i)))
        return functionList

    def setupFitFunction(self):
        print("FUNCTION SETUP CALLED")

    def setupBackground(self):
        print("Background SETUP CALLED")

    def setupStrip(self):
        ddict = {}
        ddict['event'] = "stripSetupCalled"
        ddict['strip_function'] = str(self.stripCombo.currentText())
        ddict['stripalgorithm'] = self.stripCombo.currentIndex()
        self.emit(qt.SIGNAL("FitFunctionDefinitionSignal"), ddict)

class FitControl(qt.QGroupBox):
    def __init__(self, parent=None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle("Fit Control")
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)

        row =0
        #linear fit
        self.fitAlgorithmLabel = qt.QLabel(self)
        self.fitAlgorithmLabel.setText("Fit algorithm")
        self.fitAlgorithmCombo = qt.QComboBox(self)
        self.fitAlgorithmCombo.addItem(str("Levenberg-Marquardt"))
        self.fitAlgorithmCombo.addItem(str("Linear Fit"))

        self.mainLayout.addWidget(self.fitAlgorithmLabel, row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 1)
        self.mainLayout.addWidget(self.fitAlgorithmCombo, row, 3)
        row += 1

        #weighting
        self.weightLabel = qt.QLabel(self)
        self.weightLabel.setText("Statistical weighting of data")
        self.weightCombo = qt.QComboBox(self)
        self.weightCombo.addItem(str("NO Weight"))
        self.weightCombo.addItem(str("Poisson (1/Y)"))

        self.mainLayout.addWidget(self.weightLabel,       row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 1)
        self.mainLayout.addWidget(self.weightCombo,       row, 3)
        row += 1

        
        #function estimation policy
        self.functionEstimationLabel    = qt.QLabel(self)
        self.functionEstimationLabel.setText("Function estimation policy")
        self.functionEstimationCombo    = qt.QComboBox(self)
        self.functionEstimationCombo.addItem(str("Use configuration"))
        self.functionEstimationCombo.addItem(str("Estimate once"))
        self.functionEstimationCombo.addItem(str("Estimate always"))
        self.functionEstimationCombo.setCurrentIndex(2)

        self.mainLayout.addWidget(self.functionEstimationLabel, row, 0)
        self.mainLayout.addWidget(self.functionEstimationCombo, row, 3)
        row += 1

        #background estimation policy
        self.backgroundEstimationLabel    = qt.QLabel(self)
        text = "Background estimation policy"
        self.backgroundEstimationLabel.setText(text)
        self.backgroundEstimationCombo    = qt.QComboBox(self)
        self.backgroundEstimationCombo.addItem(str("Use configuration"))
        self.backgroundEstimationCombo.addItem(str("Estimate once"))
        self.backgroundEstimationCombo.addItem(str("Estimate always"))
        self.backgroundEstimationCombo.setCurrentIndex(2)

        self.mainLayout.addWidget(self.backgroundEstimationLabel, row, 0)
        self.mainLayout.addWidget(self.backgroundEstimationCombo, row, 3)
        row += 1

        #number of iterations
        self.iterLabel = qt.QLabel(self)
        self.iterLabel.setText(str("Maximum number of fit iterations"))
        self.iterSpin = qt.QSpinBox(self)
        self.iterSpin.setMinimum(1)
        self.iterSpin.setMaximum(10000)
        self.iterSpin.setValue(10)
        self.mainLayout.addWidget(self.iterLabel,         row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 1)
        self.mainLayout.addWidget(self.iterSpin,          row, 3)
        row += 1

        #chi square handling
        self.chi2Label = qt.QLabel(self)
        self.chi2Label.setText(str("Minimum chi^2 difference (%)"))
        if 0:
            self.chi2Value = qt.QLineEdit(self)
            self.chi2Value._v = qt.QDoubleValidator(self.chi2Value)
            self.chi2Value.setValidator(self.chi2Value._v)
            self.chi2Value.setText(str("0.001"))
        else:
            self.chi2Value = qt.QDoubleSpinBox(self)
            self.chi2Value.setDecimals(4)
            self.chi2Value.setMinimum(0.0001)
            self.chi2Value.setMaximum(100.)
            self.chi2Value.setSingleStep(0.0001)
            self.chi2Value.setValue(0.001)

        self.mainLayout.addWidget(self.chi2Label,         row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 1)
        self.mainLayout.addWidget(self.chi2Value,         row, 3)
        row +=1 

        #fitting region
        self.regionTopLine = qt.QFrame(self)
        self.regionTopLine.setFrameShape(qt.QFrame.HLine)
        self.regionTopLine.setFrameShadow(qt.QFrame.Sunken)
        self.regionTopLine.setFrameShape(qt.QFrame.HLine)


        self.regionCheckBox = qt.QCheckBox(self)
        self.regionCheckBox.setText(str("Limit fitting region to :"))
        
        self.firstLabel = qt.QLabel(self)
        firstLabel_font = qt.QFont(self.firstLabel.font())
        firstLabel_font.setItalic(1)
        self.firstLabel.setFont(firstLabel_font)
        self.firstLabel.setText(str("First X Value "))
        self.firstLabel.setAlignment(qt.Qt.AlignVCenter | qt.Qt.AlignRight)
        self.firstValue = qt.QLineEdit(self)
        self.firstValue._v = qt.QDoubleValidator(self.firstValue)
        self.firstValue.setValidator(self.firstValue._v)
        self.firstValue.setText(str("0."))

        self.lastLabel = qt.QLabel(self)
        lastLabel_font = qt.QFont(self.lastLabel.font())
        lastLabel_font.setItalic(1)
        self.lastLabel.setFont(lastLabel_font)
        self.lastLabel.setText(str("Last X Value "))
        self.lastLabel.setAlignment(qt.Qt.AlignVCenter | qt.Qt.AlignRight)
        self.lastValue = qt.QLineEdit(self)
        self.lastValue._v = qt.QDoubleValidator(self.lastValue)
        self.lastValue.setValidator(self.lastValue._v)
        self.lastValue.setText(str("1000."))


        self.regionBottomLine = qt.QFrame(self)
        self.regionBottomLine.setFrameShape(qt.QFrame.HLine)
        self.regionBottomLine.setFrameShadow(qt.QFrame.Sunken)
        self.regionBottomLine.setFrameShape(qt.QFrame.HLine)

        self.mainLayout.addWidget(self.regionTopLine,   row, 0, 1, 4)
        row += 1 
        self.mainLayout.addWidget(self.regionCheckBox,row, 0)
        self.mainLayout.addWidget(self.firstLabel,    row, 1)
        self.mainLayout.addWidget(self.firstValue,    row, 3)
        row += 1
        self.mainLayout.addWidget(self.lastLabel,     row, 1)
        self.mainLayout.addWidget(self.lastValue,     row, 3)
        row += 1
        self.mainLayout.addWidget(self.regionBottomLine, row, 0, 1, 4)
        row += 1

class SimpleFitControlWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt. QVBoxLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        self.functionDefinitionWidget = FitFunctionDefinition(self)
        self.fitControlWidget = FitControl(self)
        self.mainLayout.addWidget(self.functionDefinitionWidget)
        self.mainLayout.addWidget(self.fitControlWidget)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.connect(self.functionDefinitionWidget,
                     qt.SIGNAL("FitFunctionDefinitionSignal"),
                     self._functionDefinitionSlot)

    def _functionDefinitionSlot(self, ddict):
        self.emit(qt.SIGNAL("FitControlSignal"), ddict)

    def setConfiguration(self, ddict0):
        if "fit" in ddict0:
            ddict = ddict0["fit"]
        else:
            ddict = ddict0
        workingKeys = []
        originalKeys = list(ddict.keys())
        for key in originalKeys:
            workingKeys.append(key.lower())
            
        #get current configuration
        current = self.getConfiguration()
        for key in list(current.keys()):
            #avoid case sensitivity problems
            lowerCaseKey = key.lower()
            if lowerCaseKey in workingKeys:
                idx = workingKeys.index(lowerCaseKey)
                current[key] = ddict[originalKeys[idx]]

        self._setConfiguration(current)

    def _setConfiguration(self, ddict):
        #all the keys will be present
        w = self.functionDefinitionWidget
        if 'functions' in ddict:
            w.setFunctions(ddict['functions'])
        if ddict['fit_function'] in [None, "None", "NONE"]:
            idx = 0
        else:
            idx = w.fitFunctionCombo.findText(ddict['fit_function'])
        w.fitFunctionCombo.setCurrentIndex(idx)
        if ddict['background_function'] in [None, "None", "NONE"]:
            idx = 0
        else:
            idx = w.backgroundCombo.findText(ddict['background_function'])
        w.backgroundCombo.setCurrentIndex(idx)
        if ddict['function_flag']:
            w.fitFunctionCheckBox.setChecked(True)
        else:
            w.fitFunctionCheckBox.setChecked(False)
        if ddict['strip_flag']:
            w.stripCheckBox.setChecked(True)
        else:
            w.stripCheckBox.setChecked(False)
        if ddict['background_flag']:
            w.backgroundCheckBox.setChecked(True)
        else:
            w.backgroundCheckBox.setChecked(False)
        if ddict['stripalgorithm'] in [0, 1, "0", "1"]:
            idx = int(ddict['stripalgorithm'])
        else:
            idx = w.stripCombo.findText(ddict['strip_function'])            
        w.stripCombo.setCurrentIndex(idx)
        w.setSNIP(idx)

        w.snipWidthSpin.setValue(int(ddict["snipwidth"]))
        w.stripWidthSpin.setValue(int(ddict["stripwidth"]))
        w.stripFilterSpin.setValue(int(ddict["stripfilterwidth"]))

        w.stripIterSpin.setValue(int(ddict['stripiterations']))             
        ddict['stripconstant'] = 1.0
        w.stripFilterSpin.setValue(int(ddict['stripfilterwidth']))

        if int(ddict["stripanchorsflag"]):
            w.stripAnchorsCheckBox.setChecked(True)
        else:
            w.stripAnchorsCheckBox.setChecked(False)
        anchorslist = ddict.get("stripanchorslist", [0, 0, 0, 0])
        anchorslist = ddict["stripanchorslist"]
        for lineEdit in w.stripAnchorsList:
            lineEdit.setText("0.0")

        i = 0
        for value in anchorslist:
            w.stripAnchorsList[i].setText("%g" % float(value))
            i += 1
        
        w = self.fitControlWidget
        idx = w.fitAlgorithmCombo.findText(ddict['fit_algorithm'])
        w.fitAlgorithmCombo.setCurrentIndex(idx)
        
        idx = w.weightCombo.findText(ddict['weight'])
        w.weightCombo.setCurrentIndex(idx)
        text = ddict['function_estimation_policy']
        idx = w.functionEstimationCombo.findText(text)
        w.functionEstimationCombo.setCurrentIndex(idx)

        text = ddict['background_estimation_policy']
        idx = w.backgroundEstimationCombo.findText(text)
        w.backgroundEstimationCombo.setCurrentIndex(idx)
        
        w.iterSpin.setValue(int(ddict['maximum_fit_iterations']))
        w.chi2Value.setValue(float(ddict['minimum_delta_chi']))
        if ddict['use_limits']:
            w.regionCheckBox.setChecked(True)
        else:
            w.regionCheckBox.setChecked(False)
        w.firstValue.setText("%g" % float(ddict['xmin']))
        w.lastValue.setText("%g" % float(ddict['xmax']))
        return

    def getConfiguration(self):
        ddict = {}
        w = self.functionDefinitionWidget
        ddict['functions'] =  w.getFunctions()
        ddict['fit_function'] = str(w.fitFunctionCombo.currentText())
        ddict['strip_function'] = str(w.stripCombo.currentText())
        ddict['stripalgorithm'] = w.stripCombo.currentIndex()
        ddict['background_function'] = str(w.backgroundCombo.currentText())
        if w.fitFunctionCheckBox.isChecked():
            ddict['function_flag']  = 1
        else:
            ddict['function_flag']  = 0
        if w.backgroundCheckBox.isChecked():
            ddict['background_flag']  = 1
        else:
            ddict['background_flag']  = 0
        if w.stripCheckBox.isChecked():
            ddict['strip_flag']  = 1
        else:
            ddict['strip_flag']  = 0

        ddict['snipwidth']  = w.snipWidthSpin.value()
        ddict['stripwidth'] = w.stripWidthSpin.value()
        ddict['stripiterations'] = w.stripIterSpin.value()            
        ddict['stripconstant'] = 1.0
        ddict['stripfilterwidth'] = w.stripFilterSpin.value()

        if w.stripAnchorsCheckBox.isChecked():
            ddict['stripanchorsflag'] = 1
        else:
            ddict['stripanchorsflag'] = 0

        ddict["stripanchorslist"] = []
        for lineEdit in w.stripAnchorsList:
            text = str(lineEdit.text())
            if not len(text):
                text = 0.0
            ddict["stripanchorslist"].append(float(text))

        w = self.fitControlWidget
        ddict['fit_algorithm'] = str(w.fitAlgorithmCombo.currentText())
        ddict['weight']     = str(w.weightCombo.currentText())
        text = str(w.functionEstimationCombo.currentText())
        ddict['function_estimation_policy'] = text
        text = str(w.backgroundEstimationCombo.currentText())
        ddict['background_estimation_policy'] = text
        ddict['maximum_fit_iterations'] = w.iterSpin.value()
        ddict['minimum_delta_chi']  = w.chi2Value.value()
        if w.regionCheckBox.isChecked():
            ddict['use_limits']  = 1
        else:
            ddict['use_limits']  = 0
        ddict['xmin']       = float(str(w.firstValue.text()))
        ddict['xmax']       = float(str(w.lastValue.text()))        
        return ddict

def test():
    app = qt.QApplication(sys.argv)
    app.connect(app, qt.SIGNAL("lastWindowClosed()"), app.quit)
    wid = SimpleFitControlWidget()
    ddict = {}
    ddict['stripwidth'] = 4
    ddict['stripiterations'] = 4000
    ddict['stripconstant'] = 1.0
    ddict['stripfilterwidth'] = 3
    ddict['stripanchorsflag'] = 1
    ddict['stripanchorslist'] = [0, 1, 2, 3]
    ddict['use_limits'] = 1
    ddict['xmin'] = 1
    ddict['xmax'] = 1024
    wid.setConfiguration(ddict)
    wid.show()
    app.exec_()

if __name__=="__main__":
    test()
