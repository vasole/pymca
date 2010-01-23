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
import sys
import PyMcaQt as qt
QTVERSION = qt.qVersion()
if QTVERSION < '4.0.0':
    raise ImportError, "This module requieres Qt4"

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))

class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                                          qt.QSizePolicy.Expanding))

class FitFunctionDefinition(qt.QGroupBox):
    def __init__(self, parent=None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle("Function Definition")
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)

        #actual fit function
        self.functionCheckBox = qt.QCheckBox(self)
        self.functionLabel    = qt.QLabel(self)
        self.functionLabel.setText("Fit Function to be used")
        self.functionCombo    = qt.QComboBox(self)
        self.functionCombo.addItem(str("Add Function(s)"))
        self.functionSetupButton = qt.QPushButton(self)
        self.functionSetupButton.setText('SETUP')
        self.functionSetupButton.setAutoDefault(False)

        row = 0
        self.mainLayout.addWidget(self.functionCheckBox,    row, 0)
        self.mainLayout.addWidget(self.functionLabel,       row, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self),   row, 2)
        self.mainLayout.addWidget(self.functionCombo,       row, 3)
        self.mainLayout.addWidget(self.functionSetupButton, row, 4)


        #stripping
        self.stripCheckBox = qt.QCheckBox(self)
        self.stripLabel    = qt.QLabel(self)
        self.stripLabel.setText("Non-analytical (or estimation) background algorithm")
        self.stripCombo    = qt.QComboBox(self)
        self.stripCombo.addItem(str("Strip"))
        self.stripCombo.addItem(str("SNIP"))
        self.stripSetupButton = qt.QPushButton(self)
        self.stripSetupButton.setText('SETUP')
        self.stripSetupButton.setAutoDefault(False)

        row = 1
        self.mainLayout.addWidget(self.stripCheckBox,       row, 0)
        self.mainLayout.addWidget(self.stripLabel,          row, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self),   row, 2)
        self.mainLayout.addWidget(self.stripCombo,          row, 3)
        self.mainLayout.addWidget(self.stripSetupButton,    row, 4)

        #background
        self.backgroundCheckBox = qt.QCheckBox(self)
        self.backgroundLabel    = qt.QLabel(self)
        self.backgroundLabel.setText("Background function")
        self.backgroundCombo    = qt.QComboBox(self)
        self.backgroundCombo.addItem(str("Add Function(s)"))
        self.backgroundSetupButton = qt.QPushButton(self)
        self.backgroundSetupButton.setText('SETUP')
        self.backgroundSetupButton.setAutoDefault(False)

        row = 2
        self.mainLayout.addWidget(self.backgroundCheckBox,    row, 0)
        self.mainLayout.addWidget(self.backgroundLabel,       row, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self),     row, 2)
        self.mainLayout.addWidget(self.backgroundCombo,       row, 3)
        self.mainLayout.addWidget(self.backgroundSetupButton, row, 4)

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
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 2)
        self.mainLayout.addWidget(self.fitAlgorithmCombo, row, 4)
        row += 1

        #weighting
        self.weightLabel = qt.QLabel(self)
        self.weightLabel.setText("Statistical weighting of data")
        self.weightCombo = qt.QComboBox(self)
        self.weightCombo.addItem(str("NO Weight"))
        self.weightCombo.addItem(str("Poisson (1/Y)"))

        self.mainLayout.addWidget(self.weightLabel,       row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 2)
        self.mainLayout.addWidget(self.weightCombo,       row, 4)
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
        self.mainLayout.addWidget(self.functionEstimationCombo, row, 4)
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
        self.mainLayout.addWidget(self.backgroundEstimationCombo, row, 4)
        row += 1

        #number of iterations
        self.iterLabel = qt.QLabel(self)
        self.iterLabel.setText(str("Maximum number of fit iterations"))
        self.iterSpin = qt.QSpinBox(self)
        self.iterSpin.setMinimum(1)
        self.iterSpin.setMaximum(10000)
        self.iterSpin.setValue(10)
        self.mainLayout.addWidget(self.iterLabel,         row, 0)
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 2)
        self.mainLayout.addWidget(self.iterSpin,          row, 4)
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
        self.mainLayout.addWidget(HorizontalSpacer(self), row, 2)
        self.mainLayout.addWidget(self.chi2Value,         row, 4)
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

        self.mainLayout.addWidget(self.regionTopLine,   row, 0, 1, 5)
        row += 1 
        self.mainLayout.addWidget(self.regionCheckBox,row, 0)
        self.mainLayout.addWidget(self.firstLabel,    row, 3)
        self.mainLayout.addWidget(self.firstValue,    row, 4)
        row += 1
        self.mainLayout.addWidget(self.lastLabel,     row, 3)
        self.mainLayout.addWidget(self.lastValue,     row, 4)
        row += 1
        self.mainLayout.addWidget(self.regionBottomLine, row, 0, 1, 5)
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

    def setConfiguration(self, ddict0):
        if ddict0.has_key("fit"):
            ddict = ddict0["fit"]
        else:
            ddict = ddict0

        workingKeys = []
        originalKeys = ddict.keys() 
        for key in originalKeys:
            workingKeys.append(key.lower())
            
        #get current configuration
        current = self.getConfiguration()
        for key in current.keys():
            #avoid case sensitivity problems
            lowerCaseKey = key.lower()
            if lowerCaseKey in workingKeys:
                idx = workingKeys.index(lowerCaseKey)
                current[key] = ddict[originalKeys[idx]]

        self._setConfiguration(current)

    def _setConfiguration(self, ddict):
        #all the keys will be present
        w = self.functionDefinitionWidget
        idx = w.functionCombo.findText(ddict['fit_function'])
        w.functionCombo.setCurrentIndex(idx)
        idx = w.stripCombo.findText(ddict['strip_function'])
        w.stripCombo.setCurrentIndex(idx)
        idx = w.backgroundCombo.findText(ddict['background_function'])
        w.backgroundCombo.setCurrentIndex(idx)
        if ddict['function_flag']:
            w.functionCheckBox.setChecked(True)
        else:
            w.functionCheckBox.setChecked(False)
        if ddict['strip_flag']:
            w.stripCheckBox.setChecked(True)
        else:
            w.stripCheckBox.setChecked(False)
        if ddict['background_flag']:
            w.backgroundCheckBox.setChecked(True)
        else:
            w.backgroundCheckBox.setChecked(False)
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
        ddict['fit_function'] = str(w.functionCombo.currentText())
        ddict['strip_function'] = str(w.stripCombo.currentText())
        ddict['background_function'] = str(w.backgroundCombo.currentText())
        if w.functionCheckBox.isChecked():
            ddict['function_flag']  = 1
        else:
            ddict['function_flag']  = 0
        if w.stripCheckBox.isChecked():
            ddict['strip_flag']  = 1
        else:
            ddict['strip_flag']  = 0
        if w.backgroundCheckBox.isChecked():
            ddict['background_flag']  = 1
        else:
            ddict['background_flag']  = 0
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
    ddict['use_limits'] = 1
    ddict['xmin'] = 1
    ddict['xmax'] = 1024
    wid.setConfiguration(ddict)
    wid.show()
    print wid.getConfiguration()
    app.exec_()

if __name__=="__main__":
    test()
