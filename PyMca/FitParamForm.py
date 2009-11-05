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
__revision__ = "$Revision: 1.18 $"
import sys
from PyMcaQt import *

QTVERSION = qVersion()
if QTVERSION < '4.0.0':
    Q3SpinBox = QSpinBox
else:
    QLabel.AlignRight = Qt.AlignRight
    QLabel.AlignCenter = Qt.AlignCenter
    QLabel.AlignVCenter= Qt.AlignVCenter
    class Q3SpinBox(QSpinBox):
        def setMinValue(self, v):
            self.setMinimum(v)

        def setMaxValue(self, v):
            self.setMaximum(v)

        def setLineStep(self, v):
            pass
        
    class Q3GridLayout(QGridLayout):
        def addMultiCellWidget(self, w, r0, r1, c0, c1, *var):
            self.addWidget(w, r0, c0, 1 + r1 - r0, 1 + c1 - c0)
        
class HorizontalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)

        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

class VerticalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Expanding))
        
class FitParamForm(QWidget):
    def __init__(self,parent = None,name = None,fl = 0):
        if qVersion() < '4.0.0':
            QWidget.__init__(self,parent,name,fl)

            if name == None:
                self.setName("FitParamForm")
            self.setCaption(str("FIT Parameters"))
        else:
            QWidget.__init__(self,parent)

        if qVersion() < '4.0.0':
            FitParamFormLayout = QVBoxLayout(self,11,6,"FitParamFormLayout")
            self.mainTab = QTabWidget(self,"mainTab")
            self.tabFit = QWidget(self.mainTab,"tabFit")
            tabFitLayout = QVBoxLayout(self.tabFit,11,6,"tabFitLayout")

            layout5 = QGridLayout(None,1,1,12,6,"layout5")
        else:
            FitParamFormLayout = QVBoxLayout(self)
            FitParamFormLayout.setMargin(11)
            FitParamFormLayout.setSpacing(6)
            self.mainTab = QTabWidget(self)
            self.tabFit = QWidget()
            tabFitLayout = QVBoxLayout(self.tabFit)
            tabFitLayout.setMargin(11)
            tabFitLayout.setSpacing(6)
            layout5 = Q3GridLayout(None)
            #,1,1,
            layout5.setMargin(11)
            layout5.setSpacing(6)

        if qVersion() < '4.0.0':
            self.functionCombo = QComboBox(0,self.tabFit)
        else:
            self.functionCombo = QComboBox(self.tabFit)
            self.functionCombo.insertItem = self.functionCombo.addItem

        self.functionLabel = QLabel(self.tabFit)
        self.functionLabel.setText("Fit Function")
        self.functionCombo.insertItem(str("Mca Hypermet"))
        self.functionCombo.insertItem(str("Mca Pseudo-Voigt"))


        self.snipWidthLabel = QLabel(self.tabFit)
        self.snipWidthLabel.setText(str("SNIP Background Width"))

        self.stripWidthLabel = QLabel(self.tabFit)
        self.stripWidthLabel.setText(str("Strip Background Width"))
        self.stripIterValue = QLineEdit(self.tabFit)


        self.chi2Label = QLabel(self.tabFit)
        self.chi2Label.setText(str("Minimum chi^2 difference (%)"))

        self.chi2Value = QLineEdit(self.tabFit)

        self.linearFitFlagCheck = QCheckBox(self.tabFit)
        self.linearFitFlagCheck.setText(str("Perform a Linear Fit Fixing non-linear Parameters to Initial Values"))

        self.mainTab.addTab(self.tabFit,str("FIT"))

        self.lastLabel = QLabel(self.tabFit)
        lastLabel_font = QFont(self.lastLabel.font())
        lastLabel_font.setItalic(1)
        self.lastLabel.setFont(lastLabel_font)
        self.lastLabel.setText(str("Last channel :"))
        self.lastLabel.setAlignment(QLabel.AlignVCenter | QLabel.AlignRight)

        self.regionCheck = QCheckBox(self.tabFit)
        self.regionCheck.setText(str("Limit fitting region to :"))

        self.topLine = QFrame(self.tabFit)
        self.topLine.setFrameShape(QFrame.HLine)
        self.topLine.setFrameShadow(QFrame.Sunken)
        self.topLine.setFrameShape(QFrame.HLine)


        ##########
        self.weightLabel = QLabel(self.tabFit)
        self.weightLabel.setText("Statistical weighting of data")
        if qVersion() < '4.0.0':
            self.weightCombo = QComboBox(0,self.tabFit)
        else:
            self.weightCombo = QComboBox(self.tabFit)
            self.weightCombo.insertItem = self.weightCombo.addItem
        
        self.weightCombo.insertItem(str("NO Weight"))
        self.weightCombo.insertItem(str("Poisson (1/Y)"))
        #self.weightCombo.insertItem(str("Poisson (1/Y2)"))


        ##########
        self.iterLabel = QLabel(self.tabFit)
        self.iterLabel.setText(str("Number of fit iterations"))


        if qVersion() < '4.0.0':
            self.contCombo = QComboBox(0,self.tabFit)
        else:
            self.contCombo = QComboBox(self.tabFit)
            self.contCombo.insertItem = self.contCombo.addItem
        
        self.contCombo.insertItem(str("NO Continuum"))
        self.contCombo.insertItem(str("Constant"))
        self.contCombo.insertItem(str("Linear"))
        self.contCombo.insertItem(str("Parabolic"))
        self.contCombo.insertItem(str("Linear Polynomial"))
        self.contCombo.insertItem(str("Exp. Polynomial"))

        if qVersion() < '4.0.0':
            self.stripCombo = QComboBox(0,self.tabFit)
        else:
            self.stripCombo = QComboBox(self.tabFit)
            self.stripCombo.insertItem = self.stripCombo.addItem
        
        self.stripComboLabel = QLabel(self.tabFit)
        self.stripComboLabel.setText("Non-analytical (or estimation) background algorithm")
        self.stripCombo.insertItem(str("Strip"))
        self.stripCombo.insertItem(str("SNIP"))
        self.connect(self.stripCombo,
                     SIGNAL("activated(int)"),
                     self._stripComboActivated)

        self.snipWidthSpin = Q3SpinBox(self.tabFit)
        self.snipWidthSpin.setMaxValue(300)
        self.snipWidthSpin.setMinValue(0)

        self.stripWidthSpin = Q3SpinBox(self.tabFit)
        self.stripWidthSpin.setMaxValue(100)
        self.stripWidthSpin.setMinValue(1)


        self.orderSpin = Q3SpinBox(self.tabFit)
        self.orderSpin.setMaxValue(10)
        self.orderSpin.setMinValue(1)

        maxnchannel  = 16384*4

        self.maxSpin = Q3SpinBox(self.tabFit)
        self.maxSpin.setMaxValue(maxnchannel)
        self.maxSpin.setLineStep(128)

        self.minSpin = Q3SpinBox(self.tabFit)
        self.minSpin.setMaxValue(maxnchannel)
        self.minSpin.setLineStep(128)

        self.stripIterLabel = QLabel(self.tabFit)
        self.stripIterLabel.setText(str("Strip Background Iterations"))

        self.iterSpin = Q3SpinBox(self.tabFit)
        self.iterSpin.setMinValue(1)

        self.stripFilterLabel = QLabel(self.tabFit)
        self.stripFilterLabel.setText(str("Strip Background Smoothing Width (Savitsky-Golay)"))

        self.stripFilterSpin = Q3SpinBox(self.tabFit)
        self.stripFilterSpin.setMinValue(1)
        self.stripFilterSpin.setMaxValue(40)
        self.stripFilterSpin.setLineStep(2)

        ########
        self.anchorsContainer = QWidget(self.tabFit)
        anchorsContainerLayout = QHBoxLayout(self.anchorsContainer)
        anchorsContainerLayout.setMargin(0)
        anchorsContainerLayout.setSpacing(2)
        self.stripAnchorsFlagCheck = QCheckBox(self.anchorsContainer)
        self.stripAnchorsFlagCheck.setText(str("Strip Background use Anchors"))
        anchorsContainerLayout.addWidget(self.stripAnchorsFlagCheck)

        self.stripAnchorsList = []
        for i in range(4):
            anchorSpin = Q3SpinBox(self.anchorsContainer)
            anchorSpin.setMinValue(0)
            anchorSpin.setMaxValue(maxnchannel)
            anchorsContainerLayout.addWidget(anchorSpin)
            self.stripAnchorsList.append(anchorSpin)
        #######

        self.firstLabel = QLabel(self.tabFit)
        firstLabel_font = QFont(self.firstLabel.font())
        firstLabel_font.setItalic(1)
        self.firstLabel.setFont(firstLabel_font)
        self.firstLabel.setText(str("First channel :"))
        if qVersion() < '4.0.0':
            self.firstLabel.setAlignment(QLabel.AlignVCenter | QLabel.AlignRight)
        else:
            self.firstLabel.setAlignment(Qt.AlignVCenter | Qt.AlignRight)


        self.typeLabel = QLabel(self.tabFit)
        self.typeLabel.setText(str("Continuum type"))

        self.orderLabel = QLabel(self.tabFit)
        self.orderLabel.setText(str("Polynomial order"))

        self.bottomLine = QFrame(self.tabFit)
        self.bottomLine.setFrameShape(QFrame.HLine)
        self.bottomLine.setFrameShadow(QFrame.Sunken)
        self.bottomLine.setFrameShape(QFrame.HLine)

        layout5.addMultiCellWidget(self.functionLabel,0,0,0,1)
        layout5.addMultiCellWidget(self.functionCombo,0,0,3,4)


        layout5.addMultiCellWidget(self.typeLabel,1,1,0,1)
        layout5.addMultiCellWidget(self.contCombo,1,1,3,4)

        layout5.addMultiCellWidget(self.orderLabel,2,2,0,1)
        layout5.addMultiCellWidget(self.orderSpin,2,2,3,4)


        layout5.addMultiCellWidget(self.stripComboLabel, 3, 3, 0, 1)
        if QTVERSION > '4.0.0':
            self.stripSetupButton = QPushButton(self.tabFit)
            self.stripSetupButton.setText('SETUP')
            self.stripSetupButton.setAutoDefault(False)
            layout5.addWidget(self.stripCombo, 3, 3)
            layout5.addWidget(self.stripSetupButton, 3, 4)
        else:
            layout5.addMultiCellWidget(self.stripCombo, 3, 3, 3, 4)

        layout5.addMultiCellWidget(self.snipWidthLabel,4,4,0,1)
        layout5.addMultiCellWidget(self.snipWidthSpin,4,4,3,4)

        layout5.addMultiCellWidget(self.stripWidthLabel,5,5,0,1)
        layout5.addMultiCellWidget(self.stripWidthSpin,5,5,3,4)

        layout5.addMultiCellWidget(self.stripIterLabel,6,6,0,1)
        layout5.addMultiCellWidget(self.stripIterValue,6,6,3,4)
        
        layout5.addMultiCellWidget(self.stripFilterLabel,7,7,0,1)
        layout5.addMultiCellWidget(self.stripFilterSpin,7,7,3,4)

        layout5.addMultiCellWidget(self.anchorsContainer,8,8,0,4)
        
        layout5.addWidget(self.weightLabel,9,0)
        layout5.addMultiCellWidget(self.weightCombo,9,9,3,4)

        layout5.addWidget(self.iterLabel,10,0)
        if qVersion() < '4.0.0':
            spacer = QSpacerItem(185,16,QSizePolicy.Expanding,QSizePolicy.Minimum)
            layout5.addMultiCell(spacer,10,10,1,2)
        else:
            layout5.addWidget(HorizontalSpacer(self.tabFit),10,1)
        layout5.addMultiCellWidget(self.iterSpin,10,10,3,4)

        layout5.addWidget(self.chi2Label, 11, 0)
        layout5.addMultiCellWidget(self.chi2Value, 11, 11,3,4)

        layout5.addMultiCellWidget(self.linearFitFlagCheck, 12, 12, 0, 4)

        layout5.addMultiCellWidget(self.topLine, 13, 14,0,4)

        layout5.addMultiCellWidget(self.minSpin,14, 15,4,4)

        layout5.addWidget(self.regionCheck,15,0)
        layout5.addMultiCellWidget(self.firstLabel,15, 15,2,3)

        layout5.addMultiCellWidget(self.lastLabel,16,16,2,3)
        layout5.addWidget(self.maxSpin,16,4)
        layout5.addMultiCellWidget(self.bottomLine,17,18,0,4)

        tabFitLayout.addLayout(layout5)

        includeWidget = QWidget(self.tabFit)
        
        if qVersion() < '4.0.0':
            includeLayout = QGridLayout(includeWidget,1,1,0,3,"includeLayout")
        else:
            includeLayout = Q3GridLayout(includeWidget)
            includeLayout.setMargin(0)
            includeLayout.setSpacing(3)

        self.stepCheck = QCheckBox(includeWidget)
        self.stepCheck.setText(str("Step tail"))

        includeLayout.addWidget(self.stepCheck,2,2)

        self.escapeCheck = QCheckBox(includeWidget)
        self.escapeCheck.setText(str("Escape peaks"))

        includeLayout.addWidget(self.escapeCheck,1,1)

        self.includeLabel = QLabel(includeWidget)
        includeLabel_font = QFont(self.includeLabel.font())
        includeLabel_font.setBold(1)
        self.includeLabel.setFont(includeLabel_font)
        self.includeLabel.setText(str("Include:"))

        includeLayout.addWidget(self.includeLabel,0,0)

        self.sumCheck = QCheckBox(includeWidget)
        self.sumCheck.setText(str("Pile-up peaks"))

        includeLayout.addWidget(self.sumCheck,1,2)

        self.scatterCheck = QCheckBox(includeWidget)
        self.scatterCheck.setText(str("Scattering peaks"))

        includeLayout.addWidget(self.scatterCheck,1,3)

        self.stripCheck = QCheckBox(includeWidget)
        self.stripCheck.setText(str("Stripping"))

        includeLayout.addWidget(self.stripCheck,1,0)

        self.longCheck = QCheckBox(includeWidget)
        self.longCheck.setText(str("Long tail"))

        includeLayout.addWidget(self.longCheck,2,1)

        self.shortCheck = QCheckBox(includeWidget)
        self.shortCheck.setText(str("Short tail"))

        includeLayout.addWidget(self.shortCheck,2,0)
        #tabFitLayout.addLayout(includeLayout)
        layout5.addMultiCellWidget(includeWidget,18,19,0,4)

        spacer_2 = QSpacerItem(20,40,QSizePolicy.Minimum,QSizePolicy.Expanding)
        tabFitLayout.addItem(spacer_2)

        if qVersion() < '4.0.0':
            #self.mainTab.insertTab(self.tabFit,str("FIT"))
            self.tabDetector = QWidget(self.mainTab,"tabDetector")
            tabDetectorLayout = QVBoxLayout(self.tabDetector,11,6,"tabDetectorLayout")

            detLayout = QGridLayout(None,1,1,0,2,"detLayout")
            self.elementCombo = QComboBox(0,self.tabDetector,"elementCombo")
        else:
            #self.mainTab.addTab(self.tabFit,str("FIT"))
            self.tabDetector = QWidget()
            tabDetectorLayout = QVBoxLayout(self.tabDetector)
            tabDetectorLayout.setMargin(11)
            tabDetectorLayout.setSpacing(6)

            detLayout = Q3GridLayout(None)
            detLayout.setMargin(0)
            detLayout.setSpacing(2)
            self.elementCombo = QComboBox(self.tabDetector)

        if qVersion() < '4.0.0':
            self.elementCombo.insertItem(str("Si"))
            self.elementCombo.insertItem(str("Ge"))
            self.elementCombo.insertItem(str("Cd1Te1"))
            self.elementCombo.insertItem(str("Hg1I2"))
            self.elementCombo.insertItem(str("Ga1As1"))
        else:
            self.elementCombo.insertItem(0, str("Si"))
            self.elementCombo.insertItem(1, str("Ge"))
            self.elementCombo.insertItem(2, str("Cd1Te1"))
            self.elementCombo.insertItem(3, str("Hg1I2"))
            self.elementCombo.insertItem(4, str("Ga1As1"))
        self.elementCombo.setEnabled(1)
        self.elementCombo.setDuplicatesEnabled(0)

        detLayout.addWidget(self.elementCombo,0,3)

        self.elementLabel = QLabel(self.tabDetector)
        self.elementLabel.setText(str("Detector Composition"))

        detLayout.addWidget(self.elementLabel,0,0)
        self.escapeLabel = QLabel(self.tabDetector)
        self.escapeLabel.setText(str("Maximum Number of Escape energies"))
        detLayout.addMultiCellWidget(self.escapeLabel,3,4,0,0)

        #self.intensityValue0 = QLineEdit(self.tabDetector,"intensityValue0")
        #self.intensityValue0.setText(str("1.0"))
        #self.intensityValue0.setReadOnly(1)
        self.nEscapeThreshold = Q3SpinBox(self.tabDetector)
        self.nEscapeThreshold.setMaxValue(20)
        self.nEscapeThreshold.setMinValue(1)
        self.nEscapeThreshold.setValue(4)
        #detLayout.addWidget(self.intensityValue0,3,3)
        detLayout.addWidget(self.nEscapeThreshold,3,3)
        spacer_4 = QSpacerItem(89,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        detLayout.addItem(spacer_4,3,1)
        tabDetectorLayout.addLayout(detLayout)

        self.calibLine = QFrame(self.tabDetector)
        self.calibLine.setFrameShape(QFrame.HLine)
        self.calibLine.setFrameShadow(QFrame.Sunken)
        self.calibLine.setFrameShape(QFrame.HLine)
        tabDetectorLayout.addWidget(self.calibLine)

        if qVersion() < '4.0.0':
            layout5_2 = QGridLayout(None,1,1,11,2,"layout5_2")
        else:
            layout5_2 = Q3GridLayout(None)
            layout5_2.setMargin(11)
            layout5_2.setSpacing(2)

        self.zeroError = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.zeroError,1,5)

        self.sumfacSepLabel = QLabel(self.tabDetector)
        sumfacSepLabel_font = QFont(self.sumfacSepLabel.font())
        sumfacSepLabel_font.setBold(1)
        self.sumfacSepLabel.setFont(sumfacSepLabel_font)
        self.sumfacSepLabel.setText(str("+/-"))

        layout5_2.addWidget(self.sumfacSepLabel,5,4)

        self.noiseLabel = QLabel(self.tabDetector)
        self.noiseLabel.setText(str("Detector noise (keV)"))

        layout5_2.addWidget(self.noiseLabel,3,0)

        self.gainCheck = QCheckBox(self.tabDetector)
        self.gainCheck.setText(str(""))

        layout5_2.addWidget(self.gainCheck,2,2)

        self.gainLabel = QLabel(self.tabDetector)
        self.gainLabel.setText(str("Spectrometer gain (keV/ch)"))

        layout5_2.addWidget(self.gainLabel,2,0)

        self.sumfacLabel = QLabel(self.tabDetector)
        self.sumfacLabel.setText(str("Pile-up Factor"))

        layout5_2.addWidget(self.sumfacLabel,5,0)

        self.noiseError = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.noiseError,3,5)

        self.zeroValue = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.zeroValue,1,3)

        self.fanoSepLabel = QLabel(self.tabDetector)
        fanoSepLabel_font = QFont(self.fanoSepLabel.font())
        fanoSepLabel_font.setBold(1)
        self.fanoSepLabel.setFont(fanoSepLabel_font)
        self.fanoSepLabel.setText(str("+/-"))

        layout5_2.addWidget(self.fanoSepLabel,4,4)

        self.fanoError = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.fanoError,4,5)

        self.zeroSepLabel = QLabel(self.tabDetector)
        zeroSepLabel_font = QFont(self.zeroSepLabel.font())
        zeroSepLabel_font.setBold(1)
        self.zeroSepLabel.setFont(zeroSepLabel_font)
        self.zeroSepLabel.setText(str("+/-"))

        layout5_2.addWidget(self.zeroSepLabel,1,4)

        self.valueLabel = QLabel(self.tabDetector)
        valueLabel_font = QFont(self.valueLabel.font())
        valueLabel_font.setItalic(1)
        self.valueLabel.setFont(valueLabel_font)
        self.valueLabel.setText(str("Value"))
        if qVersion() < '4.0.0':
            self.valueLabel.setAlignment(QLabel.AlignCenter)
        else:
            self.valueLabel.setAlignment(Qt.AlignCenter)

        layout5_2.addWidget(self.valueLabel,0,3)
        if qVersion() < '4.0.0':
            spacer_5 = QSpacerItem(44,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
            layout5_2.addItem(spacer_5,1,1)
        else:
            layout5_2.addWidget(HorizontalSpacer(self.tabDetector),1,1)

        self.noiseValue = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.noiseValue,3,3)

        self.fanoValue = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.fanoValue,4,3)

        self.zeroLabel = QLabel(self.tabDetector)
        self.zeroLabel.setText(str("Spectrometer zero (keV)"))

        layout5_2.addWidget(self.zeroLabel,1,0)

        self.sumfacError = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.sumfacError,5,5)

        self.noiseSepLabel = QLabel(self.tabDetector)
        noiseSepLabel_font = QFont(self.noiseSepLabel.font())
        noiseSepLabel_font.setBold(1)
        self.noiseSepLabel.setFont(noiseSepLabel_font)
        self.noiseSepLabel.setText(str("+/-"))

        layout5_2.addWidget(self.noiseSepLabel,3,4)

        self.sumfacCheck = QCheckBox(self.tabDetector)
        self.sumfacCheck.setText(str(""))

        layout5_2.addWidget(self.sumfacCheck,5,2)

        self.noiseCheck = QCheckBox(self.tabDetector)
        self.noiseCheck.setText(str(""))

        layout5_2.addWidget(self.noiseCheck,3,2)

        self.errorLabel = QLabel(self.tabDetector)
        errorLabel_font = QFont(self.errorLabel.font())
        errorLabel_font.setItalic(1)
        self.errorLabel.setFont(errorLabel_font)
        self.errorLabel.setText(str("Delta "))
        self.errorLabel.setAlignment(QLabel.AlignCenter)

        layout5_2.addWidget(self.errorLabel,0,5)

        self.fixedLabel = QLabel(self.tabDetector)
        fixedLabel_font = QFont(self.fixedLabel.font())
        fixedLabel_font.setItalic(1)
        self.fixedLabel.setFont(fixedLabel_font)
        self.fixedLabel.setText(str("Fixed "))
        if qVersion() < '4.0.0':
            self.fixedLabel.setAlignment(QLabel.AlignVCenter)
        else:
            self.fixedLabel.setAlignment(Qt.AlignVCenter)

        layout5_2.addWidget(self.fixedLabel,0,2)

        self.zeroCheck = QCheckBox(self.tabDetector)
        self.zeroCheck.setText(str(""))

        layout5_2.addWidget(self.zeroCheck,1,2)

        self.sumfacValue = QLineEdit(self.tabDetector,)

        layout5_2.addWidget(self.sumfacValue,5,3)

        self.fanoLabel = QLabel(self.tabDetector)
        self.fanoLabel.setText(str("Fano factor (Si ~ 0.12, Ge ~ 0.1)"))

        layout5_2.addWidget(self.fanoLabel,4,0)

        self.gainValue = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.gainValue,2,3)

        self.gainSepLabel = QLabel(self.tabDetector)
        gainSepLabel_font = QFont(self.gainSepLabel.font())
        gainSepLabel_font.setBold(1)
        self.gainSepLabel.setFont(gainSepLabel_font)
        self.gainSepLabel.setText(str("+/-"))

        layout5_2.addWidget(self.gainSepLabel,2,4)

        self.fanoCheck = QCheckBox(self.tabDetector)
        self.fanoCheck.setText(str(""))

        layout5_2.addWidget(self.fanoCheck,4,2)

        self.gainError = QLineEdit(self.tabDetector)

        layout5_2.addWidget(self.gainError,2,5)
        tabDetectorLayout.addLayout(layout5_2)
        spacer_6 = QSpacerItem(20,2,QSizePolicy.Minimum,QSizePolicy.Expanding)
        tabDetectorLayout.addItem(spacer_6)
        if qVersion() < '4.0.0':
            self.mainTab.insertTab(self.tabDetector,str("DETECTOR"))
            self.TabBeam = QWidget(self.mainTab,"TabBeam")
            self.mainTab.insertTab(self.TabBeam,str("BEAM"))

            self.TabPeaks = QWidget(self.mainTab,"TabPeaks")
            self.mainTab.insertTab(self.TabPeaks,str("PEAKS"))
        else:
            self.mainTab.addTab(self.tabDetector,str("DETECTOR"))
            self.TabBeam = QWidget()
            self.mainTab.addTab(self.TabBeam,str("BEAM"))

            self.TabPeaks = QWidget()
            self.mainTab.addTab(self.TabPeaks,str("PEAKS"))

        if qVersion() < '4.0.0':
            self.tabPeakShape = QWidget(self.mainTab)
            tabPeakShapeLayout = QGridLayout(self.tabPeakShape,1,1,11,2,
                                             "tabPeakShapeLayout")
        else:
            self.tabPeakShape = QWidget()
            tabPeakShapeLayout = Q3GridLayout(self.tabPeakShape)
            tabPeakShapeLayout.setMargin(11)
            tabPeakShapeLayout.setSpacing(2)
            
        spacer_7 = QSpacerItem(20,90,QSizePolicy.Minimum,QSizePolicy.Expanding)
        tabPeakShapeLayout.addItem(spacer_7,8,0)

        self.staLabel = QLabel(self.tabPeakShape)
        self.staLabel.setText(str("Short Tail Area"))

        tabPeakShapeLayout.addWidget(self.staLabel,2,0)
        spacer_8 = QSpacerItem(59,20,QSizePolicy.Expanding,QSizePolicy.Minimum)
        tabPeakShapeLayout.addItem(spacer_8,1,1)

        self.fixedLabel_2 = QLabel(self.tabPeakShape)
        fixedLabel_2_font = QFont(self.fixedLabel_2.font())
        fixedLabel_2_font.setItalic(1)
        self.fixedLabel_2.setFont(fixedLabel_2_font)
        self.fixedLabel_2.setText(str("Fixed"))
        self.fixedLabel_2.setAlignment(QLabel.AlignVCenter)

        tabPeakShapeLayout.addWidget(self.fixedLabel_2, 1, 2)

        self.staCheck = QCheckBox(self.tabPeakShape)
        self.staCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.staCheck,2,2)

        self.valueLabel_2 = QLabel(self.tabPeakShape)
        valueLabel_2_font = QFont(self.valueLabel_2.font())
        valueLabel_2_font.setItalic(1)
        self.valueLabel_2.setFont(valueLabel_2_font)
        self.valueLabel_2.setText(str("Value"))
        self.valueLabel_2.setAlignment(QLabel.AlignCenter)

        tabPeakShapeLayout.addWidget(self.valueLabel_2,1,3)

        self.staValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.staValue,2,3)

        self.staSep = QLabel(self.tabPeakShape)
        staSep_font = QFont(self.staSep.font())
        staSep_font.setBold(1)
        self.staSep.setFont(staSep_font)
        self.staSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.staSep,2,4)

        self.errorLabel_2 = QLabel(self.tabPeakShape)
        errorLabel_2_font = QFont(self.errorLabel_2.font())
        errorLabel_2_font.setItalic(1)
        self.errorLabel_2.setFont(errorLabel_2_font)
        self.errorLabel_2.setText(str("Error"))
        self.errorLabel_2.setAlignment(QLabel.AlignCenter)

        tabPeakShapeLayout.addWidget(self.errorLabel_2,1,5)

        self.staError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.staError,2,5)

        self.stsError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.stsError,3,5)

        self.stsSep = QLabel(self.tabPeakShape)
        stsSep_font = QFont(self.stsSep.font())
        stsSep_font.setBold(1)
        self.stsSep.setFont(stsSep_font)
        self.stsSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.stsSep,3,4)

        self.stsValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.stsValue,3,3)

        self.stsCheck = QCheckBox(self.tabPeakShape)
        self.stsCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.stsCheck,3,2)

        self.stsLabel = QLabel(self.tabPeakShape)
        self.stsLabel.setText(str("Short Tail Slope"))

        tabPeakShapeLayout.addWidget(self.stsLabel,3,0)

        self.ltaLabel = QLabel(self.tabPeakShape)
        self.ltaLabel.setText(str("Long Tail Area"))

        tabPeakShapeLayout.addWidget(self.ltaLabel,4,0)

        self.ltaCheck = QCheckBox(self.tabPeakShape)
        self.ltaCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.ltaCheck,4,2)

        self.ltaValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.ltaValue,4,3)

        self.ltaSep = QLabel(self.tabPeakShape)
        ltaSep_font = QFont(self.ltaSep.font())
        ltaSep_font.setBold(1)
        self.ltaSep.setFont(ltaSep_font)
        self.ltaSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.ltaSep,4,4)

        self.ltaError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.ltaError,4,5)

        self.ltsError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.ltsError,5,5)

        self.ltsSep = QLabel(self.tabPeakShape)
        ltsSep_font = QFont(self.ltsSep.font())
        ltsSep_font.setBold(1)
        self.ltsSep.setFont(ltsSep_font)
        self.ltsSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.ltsSep,5,4)

        self.ltsValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.ltsValue,5,3)

        self.ltsCheck = QCheckBox(self.tabPeakShape)
        self.ltsCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.ltsCheck,5,2)

        self.ltsLabel = QLabel(self.tabPeakShape)
        self.ltsLabel.setText(str("Long Tail Slope"))

        tabPeakShapeLayout.addWidget(self.ltsLabel,5,0)

        # Step Height
        self.shLabel = QLabel(self.tabPeakShape)
        self.shLabel.setText(str("Step Height"))

        tabPeakShapeLayout.addWidget(self.shLabel,6,0)

        self.shCheck = QCheckBox(self.tabPeakShape)
        self.shCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.shCheck,6,2)

        self.shValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.shValue,6,3)

        self.shSep = QLabel(self.tabPeakShape)
        shSep_font = QFont(self.shSep.font())
        shSep_font.setBold(1)
        self.shSep.setFont(shSep_font)
        self.shSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.shSep,6,4)

        self.shError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.shError,6,5)

        # Pseudo-Voigt Eta Factor
        self.etaLabel = QLabel(self.tabPeakShape)
        self.etaLabel.setText(str("Pseudo-Voigt Eta"))

        tabPeakShapeLayout.addWidget(self.etaLabel,7,0)

        self.etaCheck = QCheckBox(self.tabPeakShape)
        self.etaCheck.setText(str(""))

        tabPeakShapeLayout.addWidget(self.etaCheck,7,2)

        self.etaValue = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.etaValue,7,3)

        self.etaSep = QLabel(self.tabPeakShape)
        etaSep_font = QFont(self.etaSep.font())
        etaSep_font.setBold(1)
        self.etaSep.setFont(etaSep_font)
        self.etaSep.setText(str("+/-"))

        tabPeakShapeLayout.addWidget(self.etaSep,7,4)

        self.etaError = QLineEdit(self.tabPeakShape)

        tabPeakShapeLayout.addWidget(self.etaError,7,5)

        
        if qVersion() < '4.0.0':
            self.mainTab.insertTab(self.tabPeakShape,str("PEAK SHAPE"))
        else:
            self.mainTab.addTab(self.tabPeakShape,str("PEAK SHAPE"))

        FitParamFormLayout.addWidget(self.mainTab)

        self.setTabOrder(self.mainTab,self.elementCombo)
        self.setTabOrder(self.zeroCheck,self.zeroValue)
        self.setTabOrder(self.zeroValue,self.zeroError)
        self.setTabOrder(self.zeroError,self.gainCheck)
        self.setTabOrder(self.gainCheck,self.gainValue)
        self.setTabOrder(self.gainValue,self.gainError)
        self.setTabOrder(self.gainError,self.noiseCheck)
        self.setTabOrder(self.noiseCheck,self.noiseValue)
        self.setTabOrder(self.noiseValue,self.noiseError)
        self.setTabOrder(self.noiseError,self.fanoCheck)
        self.setTabOrder(self.fanoCheck,self.fanoValue)
        self.setTabOrder(self.fanoValue,self.fanoError)
        self.setTabOrder(self.fanoError,self.staCheck)
        self.setTabOrder(self.staCheck,self.staValue)
        self.setTabOrder(self.staValue,self.staError)
        self.setTabOrder(self.staError,self.stsCheck)
        self.setTabOrder(self.stsCheck,self.stsValue)
        self.setTabOrder(self.stsValue,self.stsError)
        self.setTabOrder(self.stsError,self.ltaCheck)
        self.setTabOrder(self.ltaCheck,self.ltaValue)
        self.setTabOrder(self.ltaValue,self.ltaError)
        self.setTabOrder(self.ltaError,self.ltsCheck)
        self.setTabOrder(self.ltsCheck,self.ltsValue)
        self.setTabOrder(self.ltsValue,self.ltsError)
        self.setTabOrder(self.ltsError,self.shCheck)
        self.setTabOrder(self.shCheck,self.shValue)
        self.setTabOrder(self.shValue,self.shError)
        self.setTabOrder(self.shError,self.contCombo)
        self.setTabOrder(self.contCombo,self.stripCombo)
        self.setTabOrder(self.stripCombo,self.iterSpin)
        self.setTabOrder(self.iterSpin,self.chi2Value)
        self.setTabOrder(self.chi2Value,self.regionCheck)
        self.setTabOrder(self.regionCheck,self.minSpin)
        self.setTabOrder(self.minSpin,self.maxSpin)
        self.setTabOrder(self.maxSpin,self.stripCheck)
        self.setTabOrder(self.stripCheck,self.escapeCheck)
        self.setTabOrder(self.escapeCheck,self.sumCheck)
        self.setTabOrder(self.sumCheck,self.scatterCheck)
        self.setTabOrder(self.scatterCheck,self.shortCheck)
        self.setTabOrder(self.shortCheck,self.longCheck)
        self.setTabOrder(self.longCheck,self.stepCheck)
        self._stripComboActivated(0)

    def _stripComboActivated(self, iValue):
        if iValue == 1:
            self.setSNIP(True)
        else:
            self.setSNIP(False)

    def setSNIP(self, bValue):
        if bValue:
            self.snipWidthSpin.setEnabled(True)
            self.stripWidthSpin.setEnabled(False)
            #self.stripFilterSpin.setEnabled(False)
            self.stripIterValue.setEnabled(False)
            if QTVERSION < '4.0.0':
                self.stripCombo.setCurrentItem(1)
            else:
                self.stripCombo.setCurrentIndex(1)
        else:
            self.snipWidthSpin.setEnabled(False)
            #self.stripFilterSpin.setEnabled(True)
            self.stripWidthSpin.setEnabled(True)
            self.stripIterValue.setEnabled(True)
            if QTVERSION < '4.0.0':
                self.stripCombo.setCurrentItem(0)
            else:
                self.stripCombo.setCurrentIndex(0)


if __name__ == "__main__":
    a = QApplication(sys.argv)
    QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"))
    w = FitParamForm()
    if qVersion() < '4.0.0':
        a.setMainWidget(w)
        w.show()
        a.exec_loop()
    else:
        w.show()
        a.exec_()
