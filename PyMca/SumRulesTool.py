#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
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
import numpy
from PyMca import PyMcaQt as qt
from PyMca import ScanWindow
from PyMca import specfilewrapper as sf
from PyMca import SimpleFitModule as SFM
from PyMca import SpecfitFunctions
from PyMca import Elements
from PyMca import ConfigDict
from PyMca import PyMcaDirs
from PyMca import QSpecFileWidget
from PyMca import SpecFileDataSource
from PyMca.SpecfitFuns import upstep, downstep
from PyMca.Gefit import LeastSquaresFit as LSF
from os.path import isdir as osPathIsDir
from os.path import basename as osPathBasename

try:
    from PyMca import Plugin1DBase
except ImportError:
    print("WARNING:SumRulesPlugin import from somewhere else")
    from . import Plugin1DBase

DEBUG = 1
NEWLINE = '\n'

class Calculations(object):
    def __init__(self):
        pass
        
    def cumtrapz(self, y, x=None, dx=1.0):
        y = y[:]
        if x is None:
            x = numpy.arange(len(y), dtype=y.dtype) * dx
        else:
            x = x[:]
        
        if not numpy.all(numpy.diff(x) > 0.):
            # assure monotonically increasing x
            idx = numpy.argsort(x)
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
            # Avoid dublicates
            x.ravel()
            idx = numpy.nonzero(numpy.diff(x) > 0)[0]
            x = numpy.take(x, idx)
            y = numpy.take(y, idx)
        
        return numpy.cumsum(.5 * numpy.diff(x) * (y[1:] + y[:-1]))
    
    def magneticMoment(self, p, q, r, n, econf = '3d'):
        '''
        Input
        -----
        
        p : Float
            Integral over the L3 (first) edge of the XMCD
            (difference) signal
        q : Float
            Integral over the L2 (second) edge of the XMCD
            (difference) signal
        r : Float
            Integral over the complete XAS signal
        n : Float
            Electron occupation number of the sample material
        econf : String
            Determines if material is of 3d or 4f type and
            thus the number of electronic states in the outer
            shell
        
        Returns the orbital resp. the spin part of the magnetic moment        
        (c.f. Chen et al., Phys. Rev. Lett., 75(1), 152)
        '''
        mOrbt, mSpin, mRatio = None, None, None
        
        # Determine number of states in outer shell
        if econf not in ['3d','4f']:
            raise ValueError('Element must either be 3d or 4f type!')
        elif econf == '3d':
            nMax = 10.
        else:
            nMax = 14.
        
        # Check if r is non-zero
        if r == 0.:
            raise ZeroDivisionError()
            
        # Calculate Integrals
        if q is not None:
            mOrbt = -4./3. * q * (nMax - n) / r
        if (q is not None) and (p is not None):
            mSpin  = -(6.*p - 4.*q) * (nMax - n) / r
            mRatio = 2*q/(9*p-6*q)
        return mOrbt, mSpin, mRatio

class MarkerSpinBox(qt.QDoubleSpinBox):
    
    valueChangedSignal = qt.pyqtSignal(float)
    intersectionsChangedSignal = qt.pyqtSignal(object)
    
    def __init__(self, window, plotWindow, label='', parent=None):
        qt.QDoubleSpinBox.__init__(self, parent)
        
        # Attributes
        self.label = label
        self.window = window
        self.plotWindow = plotWindow
        #self.graph = graph
        #self.markerID = self.graph.insertX1Marker(0., label=label)
        self.markerID = self.plotWindow.graph.insertX1Marker(0., label=label)
        
        # Initialize
        self.setMinimum(0.)
        self.setMaximum(10000.) # TODO: Change init value
        self.setValue(0.)
        
        # Connects
        #self.connect(self.graph,
        self.connect(self.plotWindow.graph,
             qt.SIGNAL("QtBlissGraphSignal"),
             self._markerMoved)
        self.valueChanged['double'].connect(self._valueChanged)
        self.valueChanged['QString'].connect(self._valueChanged)

    def getIntersections(self):
        dataList = self.plotWindow.getAllCurves()
        #dataDict = self.graph.curves
        resDict  = {}
        pos      = self.value()
        if not isinstance(pos, float):
            return
        #for listIdx, (x, y, legend, info) in enumerate(dataDict.items()):
        for x, y, legend, info in dataList:
            res  = float('NaN')
            if numpy.all(pos < x) or numpy.all(x < pos):
                continue
                #raise ValueError('Marker outside of data range')
            if pos in x:
                idx = numpy.where(x == pos)
                res = y[idx]
            else:
                # Intepolation needed, assume well
                # behaved data (c.f. copy routine)
                lesserIdx  = numpy.nonzero(x < pos)[0][-1]
                greaterIdx = numpy.nonzero(x > pos)[0][0]
                dy = y[lesserIdx] - y[greaterIdx]
                dx = x[lesserIdx] - x[greaterIdx]
                res = dy/dx * (pos - x[lesserIdx]) + y[lesserIdx]
            resDict[legend] = (pos, res)
        return resDict

    def hideMarker(self):
        graph = self.plotWindow.graph
        if self.markerID in graph.markersdict.keys():
            marker = graph.markersdict[self.markerID]['marker']
            marker.hide()
            
    def showMarker(self):
        graph = self.plotWindow.graph
        if self.markerID in graph.markersdict.keys():
            marker = graph.markersdict[self.markerID]['marker']
            marker.show()

    def _setMarkerFollowMouse(self, windowTitle):
        windowTitle = str(windowTitle)
        graph = self.plotWindow.graph
        if self.window == windowTitle:
            graph.setmarkercolor(self.markerID, 'blue')
            graph.setmarkerfollowmouse(self.markerID, True)
            graph.replot()
        else:
            graph.setmarkercolor(self.markerID, 'black')
            graph.setmarkerfollowmouse(self.markerID, False)
            graph.replot()

    def _markerMoved(self, ddict):
        if 'marker' not in ddict:
            return
        else:
            if ddict['marker'] != self.markerID:
                return
        if ddict['event'] == 'markerMoving':
            self.setValue(ddict['x'])
            
    def _valueChanged(self, val):
        try:
            val = float(val)
        except ValueError:
            if DEBUG == 0:
                print('_valueChanged -- Sorry, it ain\'t gonna float: %s'%str(val))
            return
        graph = self.plotWindow.graph
        #self.graph.setMarkerXPos(self.markerID, val)
        graph.setMarkerXPos(self.markerID, val)
        #self.graph.replot()
        graph.replot()
        #self.valueChangedSignal.emit(val)
        #ddict = self.getIntersections()
        #self.intersectionsChangedSignal.emit(ddict)
        
class LineEditDisplay(qt.QLineEdit):
    def __init__(self, controller, ddict={}, unit='', parent=None):
        qt.QLineEdit.__init__(self, parent)
        self.setReadOnly(True)
        self.setAlignment(qt.Qt.AlignRight)
        self.ddict = ddict
        self.unit = unit
        self.setMaximumWidth(120)
        self.controller = controller
        if isinstance(self.controller, qt.QComboBox):
            self.controller.currentIndexChanged['QString'].connect(self.setText)
        elif isinstance(self.controller, qt.QDoubleSpinBox):
            # Update must be triggered otherwise
            #self.controller.valueChanged['QString'].connect(self.setText)
            pass
        else:
            raise ValueError('LineEditDisplay: Controller must be of type QComboBox or QDoubleSpinBox')
        #self.controller.destroyed.connect(self.destroy)
        
    def updateDict(self, ddict):
        # Only relevant if type(controller) == QComboBox
        self.ddict = ddict

    def updateUnit(self, unit):
        self.unit = unit

    def checkController(self):
        if isinstance(self.controller, qt.QComboBox):
            tmp = self.controller.currentText()
        elif isinstance(self.controller, qt.QDoubleSpinBox):
            tmp = self.controller.value()
        self.setText(tmp)
        
    def setText(self, inp):
        inp = str(inp)
        if isinstance(self.controller, qt.QComboBox):
            if inp == '':
                text = ''
            else:
                tmp = self.ddict.get(inp,None)
                if tmp is not None:
                    try:
                        text = '%.2f meV'%(1000. * float(tmp))
                    except ValueError:
                        text = 'NaN' 
                else:
                    text = '---'
        elif isinstance(self.controller, qt.QDoubleSpinBox):
            text = inp + ' ' + self.unit
        qt.QLineEdit.setText(self, text)


class SumRulesWindow(qt.QMainWindow):

    # Curve labeling
    __xasBGmodel = 'xas BG model'
    # Tab names
    __tabElem = 'element'
    __tabBG   = 'background'
    __tabInt  = 'integration'
    # Marker names
    __preMin = 'Pre Min'
    __preMax = 'Pre Max'
    __postMin = 'Post Min'
    __postMax = 'Post Max'
    __intP = 'p'
    __intQ = 'q'
    __intR = 'r'
    
    # Lists
    tabList = [__tabElem,
               __tabBG,
               __tabInt]
    xasMarkerList  = [__preMin,
                      __preMax,
                      __postMin,
                      __postMax]
    xmcdMarkerList = [__intP,
                      __intQ,
                      __intR]
    edgeMarkerList = []
    
    # Elements with 3d final state
    transitionMetals = ['Sc', 'Ti', 'V', 'Cr', 'Mn',\
                         'Fe', 'Co', 'Ni', 'Cu']
    # Elements with 4f final state
    rareEarths = ['La', 'Ce', 'Pr', 'Nd', 'Pm',\
                  'Sm', 'Eu', 'Gd', 'Tb', 'Dy',\
                  'Ho', 'Er', 'Tm', 'Yb']
    elementsDict = {
            ''  : [],
            '3d': transitionMetals,
            '4f': rareEarths
    }
    # Electron final states
    electronConfs = ['3d','4f']
    # Occuring Transitions
    occuringTransitions = ['L3M4', 'L3M5', 'L2M4', 'M5O3','M4O3']
    
    # Signals
    tabChangedSignal = qt.pyqtSignal('QString')
    modelWidthChangedSignal = qt.pyqtSignal('QString')

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Sum Rules Tool')
        self.plotWindow = ScanWindow.ScanWindow(self)
        self.plotWindow.scanWindowInfoWidget.hide()
        self.plotWindow.graph.enablemarkermode()
        
        # Hide unnecessary buttons in the toolbar
        toolbarChildren = self.plotWindow.toolBar
        # QWidget.findChildren(<qt-type>) matches 
        # all child widgets with the specified type
        toolbarButtons  = toolbarChildren.findChildren(qt.QToolButton)
        toolbarButtons[6].hide() # Simple Fit
        toolbarButtons[7].hide() # Average Plotted Curves
        toolbarButtons[8].hide() # Derivative
        toolbarButtons[9].hide() # Smooth
        toolbarButtons[12].hide() # Subtract active curve
        toolbarButtons[13].hide() # Save active curve
        toolbarButtons[14].hide() # Plugins   
        
        self.__savedConf = False
        self.__savedData = False
        
        # Marker Handling
        # spinboxDict connects marker movement to spinbox
        # keys() -> id(MarkerSpinBox)
        # values() -> MarkerSpinBox
        self.spinboxDict = {}
        self.valuesDict = dict(
                    [(item, {}) for item in self.tabList])
        
        # Tab Widget
        tabIdx = 0
        self.tabWidget = qt.QTabWidget()    
        for window in self.tabList:
            if window == self.__tabElem:
                # BEGIN sampleGB
                sampleGB = qt.QGroupBox('Sample definition')
                sampleLayout = qt.QVBoxLayout()
                sampleGB.setLayout(sampleLayout)
                
                # electron shell combo box
                self.elementEConfCB = qt.QComboBox()
                self.elementEConfCB.setMinimumWidth(100)
                self.elementEConfCB.addItems(['']+self.electronConfs)
                self.elementEConfCB.currentIndexChanged['QString'].connect(self.setElectronConf)
                elementEConfLayout = qt.QHBoxLayout()
                elementEConfLayout.setContentsMargins(0,0,0,0)
                elementEConfLayout.addWidget(qt.QLabel('Electron shell'))
                elementEConfLayout.addWidget(qt.HorizontalSpacer())
                elementEConfLayout.addWidget(self.elementEConfCB)
                elementEConfWidget = qt.QWidget()
                elementEConfWidget.setLayout(elementEConfLayout)
                sampleLayout.addWidget(elementEConfWidget)
                # Element selection combo box
                self.elementCB = qt.QComboBox()
                self.elementCB.setMinimumWidth(100)
                self.elementCB.addItems([''])
                self.elementCB.currentIndexChanged['QString'].connect(self.getElementInfo)
                elementLayout = qt.QHBoxLayout()
                elementLayout.setContentsMargins(0,0,0,0)
                elementLayout.addWidget(qt.QLabel('Element'))
                elementLayout.addWidget(qt.HorizontalSpacer())
                elementLayout.addWidget(self.elementCB)
                elementWidget = qt.QWidget()
                elementWidget.setLayout(elementLayout)
                sampleLayout.addWidget(elementWidget)
                # electron occupation number
                self.electronOccupation = qt.QLineEdit('e.g. 3.14')
                self.electronOccupation.setMaximumWidth(120)
                electronOccupationValidator = qt.QDoubleValidator()
                electronOccupationValidator.setBottom(0.)
                electronOccupationValidator.setTop(14.)
                self.electronOccupation.setValidator(electronOccupationValidator)
                electronOccupationLayout = qt.QHBoxLayout()
                electronOccupationLayout.setContentsMargins(0,0,0,0)
                electronOccupationLayout.addWidget(qt.QLabel('Electron Occupation Number'))
                electronOccupationLayout.addWidget(qt.HorizontalSpacer())
                electronOccupationLayout.addWidget(self.electronOccupation)
                electronOccupationWidget = qt.QWidget()
                electronOccupationWidget.setLayout(electronOccupationLayout)
                sampleLayout.addWidget(electronOccupationWidget)
                # END sampleGB
                
                # BEGIN absorptionGB: X-ray absorption edge 
                # selection combo box by transition (L3M1, etc.)
                absorptionGB     = qt.QGroupBox('X-ray absorption edges')
                absorptionLayout = qt.QVBoxLayout()
                absorptionGB.setLayout(absorptionLayout)
                
                self.edge1CB = qt.QComboBox()
                self.edge1CB.setMinimumWidth(100)
                self.edge1CB.addItems([''])
                self.edge1Line = LineEditDisplay(self.edge1CB)
                edge1Layout = qt.QHBoxLayout()
                edge1Layout.setContentsMargins(0,0,0,0)
                edge1Layout.addWidget(qt.QLabel('Edge 1'))
                edge1Layout.addWidget(qt.HorizontalSpacer())
                edge1Layout.addWidget(self.edge1CB)
                edge1Layout.addWidget(self.edge1Line)
                edge1Widget = qt.QWidget()
                edge1Widget.setLayout(edge1Layout)
                absorptionLayout.addWidget(edge1Widget)
                
                self.edge2CB = qt.QComboBox()
                self.edge2CB.setMinimumWidth(100)
                self.edge2CB.addItems([''])
                self.edge2Line = LineEditDisplay(self.edge2CB)
                edge2Layout = qt.QHBoxLayout()
                edge2Layout.setContentsMargins(0,0,0,0)
                edge2Layout.addWidget(qt.QLabel('Edge 2'))
                edge2Layout.addWidget(qt.HorizontalSpacer())
                edge2Layout.addWidget(self.edge2CB)
                edge2Layout.addWidget(self.edge2Line)
                edge2Widget = qt.QWidget()
                edge2Widget.setLayout(edge2Layout)
                absorptionLayout.addWidget(edge2Widget)
                absorptionLayout.setAlignment(qt.Qt.AlignTop)
                # END absorptionGB
                
                # Combine sampleGB & absorptionGB in one Line
                topLineLayout = qt.QHBoxLayout()
                topLineLayout.setContentsMargins(0,0,0,0)
                topLineLayout.addWidget(sampleGB)
                topLineLayout.addWidget(absorptionGB)
                
                topLine = qt.QWidget()
                topLine.setLayout(topLineLayout)
                
                # BEGIN tab layouting
                elementTabLayout = qt.QVBoxLayout()
                elementTabLayout.setContentsMargins(1,1,1,1)
                elementTabLayout.addWidget(topLine)
                elementTabLayout.addWidget(qt.VerticalSpacer())
                elementTabWidget = qt.QWidget()
                elementTabWidget.setLayout(elementTabLayout)
                self.tabWidget.addTab(
                            elementTabWidget,
                            window.upper())
                self.tabWidget.setTabToolTip(tabIdx,
                        'Shortcut: F2\n'
                       +'Define sample element here')
                # END tab layouting
                
                self.valuesDict[self.__tabElem]\
                    ['element'] = self.elementCB
                self.valuesDict[self.__tabElem]\
                    ['electron shell'] = self.elementEConfCB
                self.valuesDict[self.__tabElem]\
                    ['electron occupation'] = self.electronOccupation
                self.valuesDict[self.__tabElem]\
                    ['edge1Transition'] = self.edge1CB
                self.valuesDict[self.__tabElem]\
                    ['edge2Transition'] = self.edge2CB
                self.valuesDict[self.__tabElem]\
                    ['edge1Energy'] = self.edge1Line
                self.valuesDict[self.__tabElem]\
                    ['edge2Energy'] = self.edge2Line
                self.valuesDict[self.__tabElem]['info'] = {}
                
            elif window == self.__tabBG:
                # BEGIN Pre/Post edge group box
                prePostLayout = qt.QGridLayout()
                prePostLayout.setContentsMargins(1,1,1,1)
                for idx, markerLabel in enumerate(self.xasMarkerList):
                    # Estimate intial xpos by estimateInt
                    markerWidget, spinbox = self.addMarker(window=window,
                                                  label=markerLabel,
                                                  xpos=0.,
                                                  unit='[eV]')
                    self.valuesDict[self.__tabBG][markerLabel] = spinbox
                    if idx == 0:
                        posx, posy = 0,0
                        markerWidget.setContentsMargins(0,0,0,-8)
                    elif idx == 1:
                        posx, posy = 1,0
                        markerWidget.setContentsMargins(0,-8,0,0)
                    elif idx == 2:
                        posx, posy = 0,1
                        markerWidget.setContentsMargins(0,0,0,-8)
                    elif idx == 3:
                        posx, posy = 1,1
                        markerWidget.setContentsMargins(0,-8,0,0)
                    else:
                        raise IndexError('Index out of bounds: %d -> %s'\
                                             %(idx, markerLabel))
                    prePostLayout.addWidget(markerWidget, posx, posy)
                prePostGB = qt.QGroupBox('Pre/Post edge')
                prePostGB.setLayout(prePostLayout)
                # END Pre/Post edge group box
                
                # BEGIN Edge group box
                numberOfEdges = 2
                edgeLayout = qt.QVBoxLayout()
                edgeLayout.setContentsMargins(1,1,1,1)
                for idx in range(numberOfEdges):
                    markerLabel = 'Edge %d'%(idx+1)
                    self.edgeMarkerList += [markerLabel]
                    markerWidget, spinbox = self.addMarker(window=window,
                                                  label=markerLabel,
                                                  xpos=0.,
                                                  unit='[eV]')
                    self.valuesDict[self.__tabBG][markerLabel] = spinbox
                    if idx == 0:
                        markerWidget.setContentsMargins(0,0,0,-8)
                    elif idx == (numberOfEdges-1):
                        markerWidget.setContentsMargins(0,-8,0,0)
                    else:
                        markerWidget.setContentsMargins(0,-8,0,-8)
                    edgeLayout.addWidget(markerWidget)
                    markerWidget.setEnabled(False)
                edgeGB = qt.QGroupBox('Edge positions')
                edgeGB.setLayout(edgeLayout)
                # END Edge group box
                
                # BEGIN Background model group box
                stepRatio = qt.QDoubleSpinBox()
                stepRatio.setMaximumWidth(100)
                stepRatio.setAlignment(qt.Qt.AlignRight)
                stepRatio.setMinimum(0.)
                stepRatio.setMaximum(1.)
                stepRatio.setSingleStep(.025)
                stepRatio.setValue(.5)
                stepRatioLayout = qt.QHBoxLayout()
                stepRatioLayout.addWidget(qt.QLabel('Step ratio'))
                stepRatioLayout.addWidget(qt.HorizontalSpacer())
                stepRatioLayout.addWidget(stepRatio)
                stepRatioWidget = qt.QWidget()
                stepRatioWidget.setContentsMargins(0,0,0,-8)
                stepRatioWidget.setLayout(stepRatioLayout)
                
                stepWidth = qt.QDoubleSpinBox()
                stepWidth.setMaximumWidth(100)
                stepWidth.setAlignment(qt.Qt.AlignRight)
                stepWidth.setMinimum(0.)
                stepWidth.setMaximum(1.)
                stepWidth.setSingleStep(.01)
                stepWidth.setValue(.5)
                modelWidthLineEdit = LineEditDisplay(
                                            controller=stepWidth,
                                            unit='eV')
                self.modelWidthChangedSignal.connect(modelWidthLineEdit.setText)
                stepWidthLayout = qt.QHBoxLayout()
                stepWidthLayout.addWidget(qt.QLabel('Step width'))
                stepWidthLayout.addWidget(qt.HorizontalSpacer())
                stepWidthLayout.addWidget(modelWidthLineEdit)
                stepWidthLayout.addWidget(stepWidth)
                stepWidthWidget = qt.QWidget()
                stepWidthWidget.setContentsMargins(0,-8,0,0)
                stepWidthWidget.setLayout(stepWidthLayout)
                
                fitControlLayout = qt.QVBoxLayout()
                fitControlLayout.addWidget(stepRatioWidget)
                fitControlLayout.addWidget(stepWidthWidget)
                fitControlGB = qt.QGroupBox('Background model control')
                fitControlGB.setLayout(fitControlLayout)
                # END Background model group box
                
                # Combine edge position and background model in single line
                sndLine = qt.QWidget()
                sndLineLayout = qt.QHBoxLayout()
                sndLineLayout.setContentsMargins(1,1,1,1)
                sndLine.setLayout(sndLineLayout)
                sndLineLayout.addWidget(edgeGB)
                sndLineLayout.addWidget(fitControlGB)
                
                # Insert into tab
                backgroundTabLayout = qt.QVBoxLayout()
                backgroundTabLayout.setContentsMargins(1,1,1,1)
                backgroundTabLayout.addWidget(prePostGB)
                #backgroundTabLayout.addWidget(edgeGB)    
                #backgroundTabLayout.addWidget(fitControlGB)    
                backgroundTabLayout.addWidget(sndLine)
                backgroundTabLayout.addWidget(qt.VerticalSpacer())
                backgroundWidget = qt.QWidget()
                backgroundWidget.setLayout(backgroundTabLayout)
                self.tabWidget.addTab(
                            backgroundWidget,
                            window.upper())
                self.tabWidget.setTabToolTip(tabIdx,
                        'Shortcut: F3\n'
                       +'Model background here.')
                
                stepRatio.valueChanged['double'].connect(self.estimateBG)
                stepWidth.valueChanged['double'].connect(self.estimateBG)
                
                self.valuesDict[self.__tabBG]\
                        ['Step Ratio'] = stepRatio
                self.valuesDict[self.__tabBG]\
                        ['Step Width'] = stepWidth
                self.valuesDict[self.__tabBG]\
                        ['Model Width'] = modelWidthLineEdit
            
            elif window == self.__tabInt:
                # BEGIN Integral marker groupbox
                pqLayout = qt.QVBoxLayout()
                pqLayout.setContentsMargins(0,-8,0,-8)
                for markerLabel in self.xmcdMarkerList:
                    # TODO: Fix intial xpos
                    markerWidget, spinbox = self.addMarker(window=window,
                                                  label=markerLabel,
                                                  xpos=0.,
                                                  unit='[eV]')
                    self.valuesDict[self.__tabInt][markerLabel] = spinbox
                    if markerLabel == self.xmcdMarkerList[0]:
                        markerWidget.setContentsMargins(0,0,0,-8)
                    elif markerLabel == self.xmcdMarkerList[-1]:
                        # Last widget gets more content margin
                        # at the bottom
                        markerWidget.setContentsMargins(0,-8,0,0)
                    else:
                        markerWidget.setContentsMargins(0,-8,0,-8)
                    integralVal = qt.QLineEdit()
                    integralVal.setReadOnly(True)
                    integralVal.setMaximumWidth(120)
                    #spinbox.valueChanged['QString'].connect(self.getIntegralValue)
                    valLabel = qt.QLabel('Integral Value:')
                    mwLayout = markerWidget.layout()
                    mwLayout.addWidget(valLabel)
                    mwLayout.addWidget(integralVal)
                    pqLayout.addWidget(markerWidget)
                    spinbox.valueChanged.connect(self.calcMagneticMoments)
                    key = 'Integral ' + markerLabel
                    self.valuesDict[self.__tabInt][key] = integralVal
                pqGB = qt.QGroupBox('XAS/XMCD integrals')
                pqGB.setLayout(pqLayout)
                # END Integral marker groupbox
                
                # BEGIN magnetic moments groupbox
                mmLayout = qt.QVBoxLayout()
                mmLayout.setContentsMargins(0,-8,0,-8)

                text = 'Spin Magnetic Moment'
                mmLineLayout = qt.QHBoxLayout()
                self.mmSpin = qt.QLineEdit()
                self.mmSpin.setReadOnly(True)
                self.mmSpin.setMaximumWidth(120)
                mmLineLayout.addWidget(qt.QLabel(text))
                mmLineLayout.addWidget(qt.HorizontalSpacer())
                mmLineLayout.addWidget(qt.QLabel('mS = '))
                mmLineLayout.addWidget(self.mmSpin)
                mmLineWidget = qt.QWidget()
                mmLineWidget.setLayout(mmLineLayout)
                mmLineWidget.setContentsMargins(0,0,0,-8)
                mmLayout.addWidget(mmLineWidget)

                text = 'Orbital Magnetic Moment'
                mmLineLayout = qt.QHBoxLayout()
                self.mmOrbt = qt.QLineEdit()
                self.mmOrbt.setReadOnly(True)
                self.mmOrbt.setMaximumWidth(120)
                mmLineLayout.addWidget(qt.QLabel(text))
                mmLineLayout.addWidget(qt.HorizontalSpacer())
                mmLineLayout.addWidget(qt.QLabel('mO = '))
                mmLineLayout.addWidget(self.mmOrbt)
                mmLineWidget = qt.QWidget()
                mmLineWidget.setLayout(mmLineLayout)
                mmLineWidget.setContentsMargins(0,-8,0,-8)
                mmLayout.addWidget(mmLineWidget)
                
                text = 'Ratio Magnetic Moments'
                mmLineLayout = qt.QHBoxLayout()
                self.mmRatio = qt.QLineEdit()
                self.mmRatio.setReadOnly(True)
                self.mmRatio.setMaximumWidth(120)
                mmLineLayout.addWidget(qt.QLabel(text))
                mmLineLayout.addWidget(qt.HorizontalSpacer())
                mmLineLayout.addWidget(qt.QLabel('mO/mS = '))
                mmLineLayout.addWidget(self.mmRatio)
                mmLineWidget = qt.QWidget()
                mmLineWidget.setLayout(mmLineLayout)
                mmLineWidget.setContentsMargins(0,-8,0,0)
                mmLayout.addWidget(mmLineWidget)
                
                mmGB = qt.QGroupBox('Magnetic moments')
                mmGB.setLayout(mmLayout)
                # END magnetic moments groupbox

                # Combine Integral marker groupbox and
                # magnetic moments groupbox in single line
                topLineLayout = qt.QHBoxLayout()
                topLineLayout.setContentsMargins(0,0,0,0)
                topLineLayout.addWidget(pqGB)
                topLineLayout.addWidget(mmGB)
                topLine = qt.QWidget()
                topLine.setLayout(topLineLayout)
                
                # BEGIN XMCD correction
                self.xmcdDetrend = qt.QCheckBox()
                self.xmcdDetrend.stateChanged['int'].connect(self.triggerDetrend)
                xmcdDetrendLayout = qt.QHBoxLayout()
                #xmcdDetrendLayout.setContentsMargins(0,0,0,1)
                xmcdDetrendLayout.addWidget(qt.QLabel(
                        'Detrend XMCD Signal (Subtracts linear fit of pre-edge Region from the signal)'))
                xmcdDetrendLayout.addWidget(qt.HorizontalSpacer())
                xmcdDetrendLayout.addWidget(self.xmcdDetrend)
                xmcdDetrendWidget = qt.QWidget()
                xmcdDetrendWidget.setLayout(xmcdDetrendLayout)
                
                xmcdDetrendGB = qt.QGroupBox('XMCD Data Preprocessing')
                xmcdDetrendGB.setLayout(xmcdDetrendLayout)
                xmcdDetrendLayout.addWidget(xmcdDetrendWidget)
                # END XMCD correction
                
                xmcdTabLayout = qt.QVBoxLayout()
                xmcdTabLayout.setContentsMargins(2,2,2,2)
                xmcdTabLayout.addWidget(topLine)
                xmcdTabLayout.addWidget(xmcdDetrendGB)
                xmcdTabLayout.addWidget(qt.VerticalSpacer())
                xmcdWidget = qt.QWidget()
                xmcdWidget.setLayout(xmcdTabLayout)
                self.tabWidget.addTab(
                            xmcdWidget,
                            window.upper())
                self.tabWidget.setTabToolTip(tabIdx,
                        'Shortcut: F4\n'
                       +'Assign Markers p, q and r here')
                            
                self.valuesDict[self.__tabInt]\
                    ['Orbital Magnetic Moment'] = self.mmOrbt
                self.valuesDict[self.__tabInt]\
                    ['Spin Magnetic Moment'] = self.mmSpin
                self.valuesDict[self.__tabInt]\
                    ['Ratio Magnetic Moments'] = self.mmRatio
                self.valuesDict[self.__tabInt]\
                    ['XMCD Detrend'] = self.xmcdDetrend
            tabIdx += 1
        # END TabWidget
        # Add to self.valuesDict
        self.tabWidget.currentChanged['int'].connect(
                            self._handleTabChangedSignal)
        
        
        # Estimate button in bottom of plot window layout
        self.buttonEstimate = qt.QPushButton('Estimate')
        self.buttonEstimate.setToolTip(
            'Shortcut: CRTL+E'
           +'Depending on the tab, estimate either the pre/post'
           +'edge regions and edge positions or the positions of'
           +'the p, q and r markers.')
        self.buttonEstimate.setShortcut(qt.Qt.CTRL+qt.Qt.Key_E)
        self.buttonEstimate.clicked.connect(self.estimate)
        self.buttonEstimate.setEnabled(False)
        self.plotWindow.graphBottomLayout.addWidget(qt.HorizontalSpacer())
        self.plotWindow.graphBottomLayout.addWidget(self.buttonEstimate)
        
        self.connect(self.plotWindow.graph,
             qt.SIGNAL("QtBlissGraphSignal"),
             self._handleGraphSignal)
             
        # Layout
        mainWidget = qt.QWidget()
        mainLayout = qt.QVBoxLayout()
        mainLayout.addWidget(self.plotWindow)
        mainLayout.addWidget(self.tabWidget)
        mainLayout.setContentsMargins(1,1,1,1)
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
        
        #
        # Data handling:
        #
        # Each is Tuple (x,y)
        # type(x),type(y) == ndarray
        self.xmcdData     = None # XMCD Spectrum
        self.xasData      = None # XAS Spectrum
        self.xasDataCorr  = None # XAS minus Background model
        self.xasDataBG    = None # XAS Backgrouns
        self.xmcdCorrData = None
        # Integrated spectra: Notice that the shape
        # diminished by one..
        self.xmcdInt     = None
        self.xasInt      = None
        
        #
        # File (name) handling
        #
        self.dataInputFilename = None
        self.confFilename      = None
        self.baseFilename      = None
        
#        tmpDict = {
#                'background' : {
#                    'Pre Min': 658.02,
#                    'Pre Max': 703.75,
#                    'Post Min': 730.5,
#                    'Post Max': 808.7,
#                    'Edge 1': 721.44,
#                    'Edge 2': 708.7,
#                    'Step Ratio': 0.25,
#                    'Step Width': 0.25
#                },
#                'integration': {
#                    'p': 717.3,
#                    'q': 740.,
#                    'r': 732.
#                },
#                'element': {
#                   'electron shell': '3d',
#                    'electron occupation': '6.6',
#                    'element': 'Fe',
#                    'edge1Transition': 'L3M4',
#                    'edge2Transition': 'L2M4'
#                }
#        }
        
#        self.setValuesDict(tmpDict)
        self._createMenuBar()

    def _createMenuBar(self):        
        # Creates empty menu bar, if none existed before
        menu = self.menuBar()
        menu.clear()

        #
        # 'File' Menu
        #
        file = menu.addMenu('&File')

        openAction = qt.QAction('&Open Spec File', self)
        openAction.setShortcut(qt.Qt.CTRL+qt.Qt.Key_O)
        openAction.setStatusTip('Opened file')
        openAction.setToolTip('Opens a data file (*.spec)')
        openAction.triggered.connect(self.loadData)
        
        loadAction = qt.QAction('&Load Configuration', self)
        loadAction.setShortcut(qt.Qt.CTRL+qt.Qt.Key_L)
        loadAction.setStatusTip('Loaded analysis file')
        loadAction.setToolTip('Loads an existing analysis file (*.sra)')
        loadAction.triggered.connect(self.loadConfiguration)
        
        saveConfAction = qt.QAction('&Save Configuration', self)
        saveConfAction.setShortcut(qt.Qt.CTRL+qt.Qt.Key_S)
        saveConfAction.setStatusTip('Saved analysis file')
        saveConfAction.setToolTip('Save analysis in file (*.sra)')
        saveConfAction.triggered.connect(self.saveConfiguration)
        
        saveConfAsAction = qt.QAction('Save &Configuration as', self)
        saveConfAsAction.setShortcut(qt.Qt.SHIFT+qt.Qt.CTRL+qt.Qt.Key_S)
        saveConfAsAction.setStatusTip('Saved analysis file')
        saveConfAsAction.setToolTip('Save analysis in file (*.sra)')
        saveConfAsAction.triggered.connect(self.saveConfigurationAs)
        
        saveDataAction = qt.QAction('Save &Data', self)
        saveDataAction.setShortcut(qt.Qt.CTRL+qt.Qt.Key_D)
        saveDataAction.setStatusTip('Saved analysis file')
        saveDataAction.setToolTip('Save analysis in file (*.sra)')
        saveDataAction.triggered.connect(self.saveData)
        
        saveDataAsAction = qt.QAction('Save D&ata as', self)
        saveDataAsAction.setShortcut(qt.Qt.SHIFT+qt.Qt.CTRL+qt.Qt.Key_D)
        saveDataAsAction.setStatusTip('Saved analysis file')
        saveDataAsAction.setToolTip('Save analysis in file (*.sra)')
        saveDataAsAction.triggered.connect(self.saveDataAs)
        
        # Populate the 'File' menu
        for action in [openAction, 
                       loadAction,
                       'sep',
                       saveConfAction, 
                       saveConfAsAction,
                       'sep',
                       saveDataAction, 
                       saveDataAsAction,
                       'sep']:
            if isinstance(action, qt.QAction):
                file.addAction(action)
            else:
                file.addSeparator()
        file.addAction('E&xit', self.close)

    def triggerDetrend(self, state):
        if (state == qt.Qt.Unchecked) or\
           (state == qt.Qt.PartiallyChecked):
            # Replot original data
            self.xmcdCorrData = None
        else:
            ddict = self.getValuesDict()
            if self.xmcdData is None:
                return
            x, y  = self.xmcdData
            preMin = ddict[self.__tabBG][self.__preMin]
            preMax = ddict[self.__tabBG][self.__preMax]
            
            mask = numpy.nonzero((preMin <= x) & (x <= preMax))[0]
            xFit = x.take(mask)
            yFit = y.take(mask)

            if (len(xFit) == 0) or (len(yFit) == 0):
                return

            # Fit linear model y = a*x + b
            a, b = numpy.polyfit(xFit, yFit, 1)

            if DEBUG == 0:
                print('a = %.5f, b = %.5f'%(a,b))
            
            trend = a*x + b
            self.xmcdCorrData = (x, y-trend)
        if self.getCurrentTab() == self.__tabInt:
            self.plotOnDemand(self.__tabInt)
            self.calcMagneticMoments()

    def calcMagneticMoments(self):
        # 0. Get Marker intersections
        ddict = self.valuesDict
        pqr = []
        mathObj = Calculations()
        for marker in self.xmcdMarkerList:
            # TODO: Find better way to determine curves..
            if marker in [self.__intP, self.__intQ]:
                if self.xmcdCorrData is not None:
                    curve = 'xmcd corr Int Y'
                else:
                    curve = 'xmcd Int Y'
            else:
                curve = 'xas Int Y'
            spinbox = ddict[self.__tabInt][marker]
            integralVals = spinbox.getIntersections()
            x, y = integralVals.get(curve, (float('NaN'),float('NaN')))
            key = 'Integral ' + marker
            lineEdit = ddict[self.__tabInt][key]
            lineEdit.setText(str(y))
            pqr += [y]
        # 1. Display intergral values
        # 2. Calculate the moments
        p, q, r = pqr
        electronOccupation = ddict[self.__tabElem]['electron occupation']
        try:
            n = float(electronOccupation.text())
        except ValueError:
            print('calcMM -- Could not convert electron occupation')
            return
        mmO, mmS, mmR = mathObj.magneticMoment(p,q,r,n)
        # 3. Display moments
        self.mmOrbt.setText(str(mmO))
        self.mmSpin.setText(str(mmS))
        self.mmRatio.setText(str(mmR))

    def getIntegralValues(self, pos):
        dataList = [self.xmcdInt, self.xasInt]
        res      = float('NaN')
        resList  = [res] * len(dataList)
        if not self.xmcdInt:
            if DEBUG == 0:
                print('getIntegralValues -- self.xmcdInt not present')
            return
        if not self.xasInt:
            if DEBUG == 0:
                print('getIntegralValues -- self.xasInt not present')
            return
        for listIdx, data in enumerate(dataList):
            x, y = data
            if numpy.all(pos < x) or numpy.all(x < pos):
                if DEBUG == 0:
                    print('getIntegralValues -- Marker position outside of data range')
                continue
            if pos in x:
                idx = numpy.where(x == pos)
                res = y[idx]
            else:
                # Intepolation needed, assume well
                # behaved data (c.f. copy routine)
                lesserIdx  = numpy.nonzero(x < pos)[0][-1]
                greaterIdx = numpy.nonzero(x > pos)[0][0]
                dy = y[lesserIdx] - y[greaterIdx]
                dx = x[lesserIdx] - x[greaterIdx]
                res = dy/dx * (pos - x[lesserIdx]) + y[lesserIdx]
            resList[listIdx] = res
        #return res
        if DEBUG == 0:
            print('getIntegralValues -- Result:\n\t%s'%str(resList))

    def loadData(self):
        dial = LoadDichorismDataDialog()
        dial.setDirectory(PyMcaDirs.outputDir)
        if dial.exec_():
            dataDict = dial.dataDict
        else:
            return
            
        # Reset calculated data
        self.xasDataCorr  = None
        self.xasDataBG    = None
        self.xmcdCorrData = None
        self.xmcdInt      = None
        self.xasInt       = None

        x = dataDict['x']
        xas = dataDict['xas']
        xmcd = dataDict['xmcd']
        self.dataInputFilename = dataDict['fn']
        self.setRawData(x, xas, 'xas')
        self.setRawData(x, xmcd, 'xmcd')

    def saveDataAs(self):
        self.baseFilename = None
        self.__savedData  = False
        self.saveData()

    def saveData(self):
        # Saves spectral data that is calculated during
        # the evaluation process:
        # First scan: XAS BG-Modell XAS-BG
        # Second scan: XAS/XMCD integrals
        dataList = [self.xasData,
                    self.xasDataCorr,
                    self.xasDataBG,
                    self.xmcdInt,
                    self.xasInt]
        if None in dataList:
            msg = qt.QMessageBox()
            msg.setWindowTitle('Sum Rules Analysis Error')
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText('Analysis incomplete!\nCannot save generated data')
            msg.exec_()
            return False
        
        if self.__savedData and self.baseFilename:
            pass
        else:
            ddict    = self.getValuesDict()
            saveDir  = PyMcaDirs.outputDir
            filters  = 'spec File (*.spec);;All files (*.*)'
            selectedFilter = 'Sum Rules Analysis files (*.spec)'
            
            baseFilename = qt.QFileDialog.getSaveFileName(self,
                                   'Save Sum Rule Analysis Data',
                                   saveDir,
                                   filters,
                                   selectedFilter)
            if len(baseFilename) == 0:
                # Leave self.baseFilename as it is..
                #self.baseFilename = None
                return False
            else:
                self.baseFilename = str(baseFilename)
            if not self.baseFilename.endswith('.spec'):
                # Append extension later
                self.baseFilename += '.spec'

        # Create filenames
        specFilename = self.baseFilename
        baseName = osPathBasename(self.dataInputFilename)
        
        self.__savedData = False
        
        # Acquire filehandle
        try:
            specFilehandle = open(specFilename, 'wb')
        except IOError:
            msg = qt.QMessageBox()
            msg.setWindowTitle('Sum Rules Analysis Error')
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText('Unable to open file \'%s\''%specFilename)
            msg.exec_()
            return False
        
        delim   = ' '

        # 1. Background Modell, XAS, XAS-Background
        # All share the same x-range
        xSpec, yXas   = self.xasData
        xSpec, yBG    = self.xasDataBG
        xSpec, yXasBG = self.xasDataCorr
        if self.xmcdCorrData:
            xInt, yXmcd = self.xmcdCorrData
        else:
            xInt, yXmcd = self.xmcdData
        dataSpec = numpy.vstack((xSpec, yXas, yBG, yXasBG, yXmcd)).T
        
        # 2. Integrals
        # Also share the same x-range
        xInt, yXasInt  = self.xasInt
        xInt, yXmcdInt = self.xmcdInt
        dataInt = numpy.vstack((xInt, yXasInt, yXmcdInt)).T

        outSpec  = '#S 1 XAS data %s'%baseName + NEWLINE
        outSpec += '#N %d'%5 + NEWLINE
        if self.xmcdCorrData:
            outSpec += '#L x  XAS  Background model  XAS corrected  XMCD corrected' + NEWLINE
        else:
            outSpec += '#L x  XAS  Background model  XAS corrected  XMCD' + NEWLINE
        for line in dataSpec:
            tmp = delim.join(['%f'%num for num in line])
            outSpec += (tmp + NEWLINE)

        outInt  = '#S 2 Integral data %s'%baseName + NEWLINE
        outInt += '#N %d'%3 + NEWLINE
        if self.xmcdCorrData:
            outInt += '#L x  XAS Int  XMCD Int' + NEWLINE
        else:
            outInt += '#L x  XAS Int  XMCD Int corrected' + NEWLINE
        for line in dataInt:
            tmp = delim.join(['%f'%num for num in line])
            outInt += (tmp + NEWLINE)
        
        for output in [outSpec, outInt]:
            specFilehandle.write(NEWLINE)
            specFilehandle.write(output)
            specFilehandle.write(NEWLINE)
        specFilehandle.close()
        
        self.__savedData = True
        return True        

    def saveConfigurationAs(self, shortcut=False):
        self.confFilename = None
        self.__savedConf  = False
        self.saveConfiguration()
        
    def saveConfiguration(self):
        ddict    = self.getValuesDict()
        
        if self.__savedConf and self.confFilename:
            filename = self.confFilename
        else:
            saveDir  = PyMcaDirs.outputDir
            filter   = 'Sum Rules Analysis files (*.sra);;All files (*.*)'
            selectedFilter = 'Sum Rules Analysis files (*.sra)'
            
            filename = qt.QFileDialog.getSaveFileName(self,
                                   'Save Sum Rule Analysis Configuration',
                                   saveDir,
                                   filter,
                                   selectedFilter)
            if len(filename) == 0:
                return False
            else:
                filename = str(filename)
            if not filename.endswith('.sra'):
                filename += '.sra'
            self.confFilename = filename

        self.__savedConf = False
        confDict = ConfigDict.ConfigDict(self.getValuesDict())
        try:
            confDict.write(filename)
        except IOError:
            msg = qt.QMessageBox()
            msg.setWindowTitle('Sum Rules Analysis Error')
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText('Unable to write configuration to \'%s\''%filename)
            msg.exec_()
            return False
        self.__savedConf = True
        return True
    
    def loadConfiguration(self):
        confDict = ConfigDict.ConfigDict()
        ddict    = self.getValuesDict()
        loadDir  = PyMcaDirs.outputDir
        filter   = 'Sum Rules Analysis files (*.sra);;All files (*.*)'
        selectedFilter = 'Sum Rules Analysis files (*.sra)'
        
        filename = qt.QFileDialog.getOpenFileName(self,
                               'Load Sum Rule Analysis Configuration',
                               loadDir,
                               filter,
                               selectedFilter)
        if len(filename) == 0:
            return
        else:
            filename = str(filename)
        try:
            confDict.read(filename)
        except IOError:
            msg = qt.QMessageBox()
            msg.setTitle('Sum Rules Analysis Error')
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText('Unable to read configuration file \'%s\''%filename)
            return
        try:
            self.setValuesDict(confDict)
            #keysLoaded = confDict.keys()
            #keysValues = self.valuesDict.keys()
        except KeyError as e:
            if DEBUG:
                print('loadConfiguration -- Key Error in \'%s\''%filename)
                print('\tMessage:', e)
            else:
                msg = qt.QMessageBox()
                msg.setTitle('Sum Rules Analysis Error')
                msg.setIcon(qt.QMessageBox.Warning)
                msg.setText('Malformed configuration file \'%s\''%filename)
            return
        self.__savedConf = True
        

    def close(self):
        if not self.__savedConf:
            msg = qt.QMessageBox()
            msg.setWindowTitle('Sum Rules Tool')
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText('The configuration has changed!\nAre you shure you want to close the window?')
            msg.setStandardButtons(qt.QMessageBox.Cancel | qt.QMessageBox.Discard)
            if msg.exec_() == qt.QMessageBox.Cancel:
                return
        qt.sQMainWindow.close(self)

    def setElectronConf(self, eConf):
        eConf = str(eConf)
        if len(eConf) == 0:
            self.electronOccupation.setDisabled(True)
        else:
            self.electronOccupation.setDisabled(False)
        # updates the element combo box
        self.elementCB.clear()
        elementsList = self.elementsDict[eConf]
        self.elementCB.addItems(['']+elementsList)

    def getElementInfo(self, symbol):
        ddict = {}
        symbol = str(symbol)
        if len(symbol) == 0:
            self.valuesDict[self.__tabElem]['info'] = {}
            return
        try:
            ddict = Elements.Element[symbol]
        except KeyError:
            msg  = ('setElement -- \'%s\' not found in '%symbol)
            msg += 'Elements.Element dictionary'
            print(msg)
        # Update valuesDict
        self.valuesDict[self.__tabElem]['info'] = ddict
        # Update the EdgeCBs
        # Lookup all keys ending in 'xrays'
        keys = [item for item in ddict.keys() if item.endswith('xrays')]
        keys.sort()
        # keys is list of list, flatten it..
        transitions = sum([ddict[key] for key in keys],[])
        # Only take transitions that occur in the experiment
        transitions = [t for t in transitions if t in self.occuringTransitions]
        tmpDict = dict( [(transition, ddict[transition]['energy']) for transition in transitions])
        for cb, ed in [(self.edge1CB, self.edge1Line),
                       (self.edge2CB, self.edge2Line)]:
            curr = cb.currentText()
            cb.clear()
            ed.clear()
            ed.updateDict(tmpDict)
            cb.addItems(['']+transitions)
            # Try to set to old entry
            idx = cb.findText(qt.QString(curr))
            if idx < 0: idx = 0
            cb.setCurrentIndex(idx)

    def getCurrentTab(self):
        idx = self.tabWidget.currentIndex()
        return self.tabList[idx]

    def getValuesDict(self):
        ddict = {}
        for tab, tabDict in self.valuesDict.items():
            if tab not in ddict.keys():
                ddict[tab] = {}
            for key, obj in tabDict.items():
                value = None
                if isinstance(obj, MarkerSpinBox):
                    value = obj.value()
                elif isinstance(obj, qt.QCheckBox):
                    state = obj.checkState()
                    if state == qt.Qt.Checked:
                        value = True
                    else:
                        # Also covers state == qt.Qt.PartiallyChecked
                        value = False
                elif isinstance(obj, qt.QComboBox):
                    tmp = obj.currentText()
                    value = str(tmp)
                elif isinstance(obj, LineEditDisplay) or\
                     isinstance(obj, qt.QLineEdit):
                    tmp = str(obj.text())
                    try:
                        value = float(tmp)
                    except ValueError:
                        value = tmp
                elif isinstance(obj, qt.QDoubleSpinBox):
                    value = obj.value()
                elif isinstance(obj, dict):
                    value = obj
                ddict[tab][key] = value
        return ddict

    def setValuesDict(self, ddict):
        markerList  = (self.xasMarkerList + self.xmcdMarkerList)
        elementList = (self.transitionMetals 
                       + self.rareEarths 
                       + self.electronConfs)
        # Check as early as possible if element symbol is present
        try:
            symbol = ddict[self.__tabElem]['element']
            self.getElementInfo(symbol)
        except KeyError:
            pass
        for tab, tabDict in ddict.items():
            if tab not in self.valuesDict.keys():
                raise KeyError('setValuesDict -- Tab not found')
            for key, value in tabDict.items():
                if not isinstance(key, str):
                    raise KeyError('setValuesDict -- Key is not str instance')
                obj = self.valuesDict[tab][key]
                if isinstance(obj, MarkerSpinBox):
                    try:
                        tmp = float(value)
                        obj.setValue(tmp)
                    except ValueError:
                        xmin, xmax = self.plotWindow.graph.getX1AxisLimits()
                        tmp = xmin + (xmax-xmin)/10.
                        if DEBUG:
                            msg  = 'setValuesDict -- Float conversion failed'
                            msg += ' while setting marker positions. Value:', value
                            print(msg)
                elif isinstance(obj, qt.QCheckBox):
                    if value == 'True':
                        state = qt.Qt.Checked
                    else:
                        state = qt.Qt.Unchecked
                    obj.setCheckState(state)
                elif isinstance(obj, qt.QDoubleSpinBox):
                    try:
                        tmp = float(value)
                        obj.setValue(tmp)
                    except ValueError:
                        if DEBUG:
                            msg  = 'setValuesDict -- Float conversion failed'
                            msg += ' while setting QDoubleSpinBox value. Value:', value
                            print(msg)
                elif isinstance(obj, qt.QComboBox):
                    idx = obj.findText(qt.QString(value))
                    obj.setCurrentIndex(idx)
                elif isinstance(obj, LineEditDisplay):
                    # Must be before isinstance(obj, qt.QLineEdit)
                    # since LineEditDisplay inherits from QLineEdit
                    obj.checkController()
                elif isinstance(obj, qt.QLineEdit):
                    if value:
                        tmp = str(value)
                        obj.setText(tmp) 
                    else:
                        obj.setText('???')
                elif isinstance(obj, dict):
                    obj = value
                else:
                    raise KeyError('setValuesDict -- \'%s\' not found'%key)
        # In case electron shell is set after element..
        try:
            symbol = ddict[self.__tabElem]['element']
            cb = self.valuesDict[self.__tabElem]['element']
            idx = cb.findText(qt.QString(symbol))
            cb.setCurrentIndex(idx)
        except KeyError:
            pass
                
    def setRawData(self, x, y, identifier):
        if identifier not in ['xmcd', 'xas']:
            msg  = 'Identifier must either be \'xmcd\' or \'xas\''
            raise ValueError(msg)
        # Sort energy range
        sortedIdx = x.argsort()
        xSorted = x.take(sortedIdx)[:]
        ySorted = y.take(sortedIdx)[:]
        # Ensure strictly monotonically increasing energy range
        dx = numpy.diff(x)
        if not numpy.all(dx > 0.):
            mask = numpy.nonzero(dx)
            xSorted = numpy.take(xSorted, mask)
            ySorted = numpy.take(ySorted, mask)
        # Add spectrum to plotWindow using the 
        if identifier == 'xmcd':
            self.xmcdData = (xSorted, ySorted)
            #self.plotWindow.graph.mapToY2(intLegend)
        elif identifier == 'xas':
            self.xasData  = (xSorted, ySorted)
        # Trigger replot when data is added
        currIdx = self.tabWidget.currentIndex()
        self._handleTabChangedSignal(currIdx)

    def estimate(self):
        tab = self.getCurrentTab()
        if tab == self.__tabBG:
            self.estimatePrePostEdgePositions()
        elif tab == self.__tabInt:
            self.estimateInt()
        else:
            # Do nothing
            pass
        return

    def estimatePrePostEdgePositions(self):
        if self.xasData is None:
            return

        ddict = self.getValuesDict()
        edgeList = [ddict[self.__tabElem]['edge1Energy'],
                    ddict[self.__tabElem]['edge2Energy']]
        filterEdgeList = lambda x:\
                            float(x.replace('meV',''))\
                            if (len(x)>0 and x!='---')\
                            else 0.0
        # Use list comprehension instead of map(filterEdgeList, edgeList)
        edgeList = [filterEdgeList(edge) for edge in edgeList]
        x, y = self.xasData
        xLimMin, xLimMax = self.plotWindow.getGraphXLimits()
            
        xMin = x[0]
        xMax = x[-1]
        xLen = xMax - xMin
        xMiddle = .5 *(xMax + xMin)
        # Average step length (Watch out for uneven data!)
        xStep = (xMax + xMin) / float(len(x))
        # Look for the index closest to the physical middle
        mask = numpy.nonzero(x <= xMiddle)[0]
        idxMid = mask[-1]

        factor = 10./100.
        edge1, edge2 = edgeList
        maxEdge = max(edgeList)
        minEdge = min(edgeList)
        if minEdge == 0. and maxEdge == 0.:
            # No edge set
            preMax  = xMiddle - factor*xLen
            postMin = xMiddle + factor*xLen
            edge1  = xMiddle
            edge2  = 0.
        elif minEdge == 0.:
            # Single edge set
            preMax  = maxEdge - factor*xLen
            postMin = maxEdge + factor*xLen
        else:
            # Two edges set
            preMax  = minEdge - factor*xLen
            postMin = maxEdge + factor*xLen

        ddict[self.__tabBG][self.__preMin]  = max(xMin,xLimMin+xStep)
        ddict[self.__tabBG][self.__preMax]  = preMax
        ddict[self.__tabBG][self.__postMin] = postMin
        ddict[self.__tabBG][self.__postMax] = min(xMax,xLimMax-xStep)
        ddict[self.__tabBG]['Edge 1'] = edge1
        ddict[self.__tabBG]['Edge 2'] = edge2
        
        self.setValuesDict(ddict)
        self.estimateBG()

    def estimateInt(self):
        if self.xasDataCorr is None or\
           self.xasInt      is None or\
           self.xmcdInt     is None:
            # Nothing to do...
            return
        ddict = self.getValuesDict()
        
        x, y = self.xasData
        xLimMin, xLimMax = self.plotWindow.getGraphXLimits()
            
        xMin = x[0]
        xMax = x[-1]
        xLen = xMax - xMin
        xMiddle = .5 *(xMax + xMin)
        
        factor = 10./100.
        postMin = ddict[self.__tabBG][self.__postMin]
        postMax = ddict[self.__tabBG][self.__postMax]
        edge1 = ddict[self.__tabBG]['Edge 1']
        edge2 = ddict[self.__tabBG]['Edge 2']
        
        # Estimate intP
        if edge1 == 0.:
            intP = edge2 + factor * xLen
        elif edge2 == 0.:
            intP = edge1 + factor * xLen
        else:
            intP = min(edge1, edge2) + factor * xLen
            
        # Estimate intQ
        intQ = postMin + factor * xLen
        
        # Estimate intR
        intR = postMax - factor * xLen
        
        # Also estimate the p, q, r Markers:
        ddict[self.__tabInt][self.__intP] = intP
        ddict[self.__tabInt][self.__intQ] = intQ
        ddict[self.__tabInt][self.__intR] = intR
        
        self.setValuesDict(ddict)

    def estimateBG(self, val=None):
        if self.xasData is None:
            return
        if self.tabWidget.currentIndex() != 1:
            # Only call from tab 1
            return

        # TODO: Remove all background curves

        x, y = self.xasData
        #self.estimatePrePostEdgePositions()
        ddict = self.getValuesDict()
        x01 = ddict[self.__tabBG]['Edge 1']
        x02 = ddict[self.__tabBG]['Edge 2']
        preMin  = ddict[self.__tabBG][self.__preMin]
        preMax  = ddict[self.__tabBG][self.__preMax]
        postMin = ddict[self.__tabBG][self.__postMin]
        postMax = ddict[self.__tabBG][self.__postMax]
        width = ddict[self.__tabBG]['Step Width']
        ratio = ddict[self.__tabBG]['Step Ratio']
        
        if preMin > preMax:
            tmp = preMin
            preMin = preMax
            preMax = tmp
        if postMin > postMax:
            tmp = preMin
            preMin = preMax
            preMax = tmp
        
        idxPre  = numpy.nonzero((preMin <= x) & (x <= preMax))[0]
        idxPost = numpy.nonzero((postMin <= x) & (x <= postMax))[0]
        
        if (len(idxPre) == 0) or (len(idxPost) == 0):
            if DEBUG:
                print('estimateBG -- Somethings wrong with pre/post edge markers')
            return
        
        xPreMax  = x[idxPre.max()]
        xPostMin = x[idxPost.min()]
        gap = abs(xPreMax - xPostMin)
        
        avgPre  = numpy.average(y[idxPre])
        avgPost = numpy.average(y[idxPost])
        bottom  = min(avgPre,avgPost)
        top     = max(avgPre,avgPost)
        if avgPost >= avgPre:
            sign = 1.
            erf  = upstep
        else:
            sign = -1.
            erf  = downstep
        diff = abs(avgPost - avgPre)
        
        ymin = y.min()
        ymax = y.max()
        
        if x01 == 0.:
            par1 = (ratio, x02, width*gap)
            if DEBUG:
                print('estimateBG -- x01 == 0, using par1: %s'%str(par1))
            model = bottom + sign * diff * erf(par1, x)
        elif x02 == 0.:
            par1 = (ratio, x01, width*gap)
            if DEBUG:
                print('estimateBG -- x02 == 0, using par1:'%str(par1))
            model = bottom + sign * diff * erf(par1, x)
        elif x02 < x01:
            par1 = (ratio, x02, width*gap)
            par2 = ((1.-ratio), x01, width*gap)
            if DEBUG:
                print('estimateBG -- x02 < x01, using par1: %s and par2: %s'\
                        %(str(par1),str(par2)))
            model = bottom + sign * diff * (erf(par1, x) + erf(par2, x))
        else:
            par1 = (ratio, x01, width*gap)
            par2 = ((1.-ratio), x02, width*gap)
            if DEBUG:
                print('estimateBG -- x01 < x02, using par1: %s and par2: %s'\
                        %(str(par1),str(par2)))
            model = bottom + sign * diff * (erf(par1, x) + erf(par2, x))
        preModel  = numpy.asarray(len(x)*[avgPre])
        postModel = numpy.asarray(len(x)*[avgPost])
        
        self.xasDataBG = x, model
        
        self.plotWindow.addCurve(x,
                                 model,
                                 self.__xasBGmodel,
                                 {},
                                 replot=False)
        self.plotWindow.addCurve(x,
                                 preModel,
                                 'Pre BG model',
                                 {},
                                 replot=False)
        self.plotWindow.addCurve(x,
                                 postModel,
                                 'Post BG model',
                                 {},
                                 replot=False)
        self.modelWidthChangedSignal.emit('%.3f'%(width*gap))
        self.plotWindow.graph.replot()

    def plotOnDemand(self, window, xlabel='ene_st', ylabel='zratio'):
        # Remove all curves
        legends = self.plotWindow.getAllCurves(just_legend=True)
        self.plotWindow.removeCurves(legends, replot=False)
        if (self.xmcdData is None) or (self.xasData is None):
            # Nothing to do
            return
        xyList  = []
        mapToY2 = False
        window = window.lower()
        if window == self.__tabElem:
            if self.xmcdCorrData is not None:
                if DEBUG:
                    print('plotOnDemand -- __tabElem: Using self.xmcdCorrData')
                xmcdX, xmcdY = self.xmcdCorrData
                xmcdLabel = 'xmcd corr'
            else:
                if DEBUG:
                    print('plotOnDemand -- __tabElem: Using self.xmcdData')
                xmcdX, xmcdY = self.xmcdData
                xmcdLabel = 'xmcd'
            xasX,  xasY  = self.xasData
            xyList = [(xmcdX, xmcdY, xmcdLabel),
                      (xasX, xasY, 'xas')]
            # At least one of the curve is going
            # to get plotted on secondary y axis
            mapToY2 = True 
        elif window == self.__tabBG:
            xasX, xasY= self.xasData
            xyList = [(xasX, xasY, 'xas')]
            if self.xasDataBG is not None:
                xasBGX, xasBGY = self.xasDataBG
                xyList += [(xasBGX, xasBGY, self.__xasBGmodel)]
        elif window == self.__tabInt:
            if (self.xasDataBG is None):
                self.xmcdInt = None
                self.xasInt  = None
                return
            # Calculate xasDataCorr
            xBG, yBG = self.xasDataBG
            x, y = self.xasData
            self.xasDataCorr = x, y-yBG
            if self.xmcdCorrData is not None:
                if DEBUG:
                    print('plotOnDemand -- __tabInt: Using self.xmcdCorrData')
                xmcdX, xmcdY = self.xmcdCorrData
                xmcdIntLabel = 'xmcd corr Int'
            else:
                if DEBUG:
                    print('plotOnDemand -- __tabInt: Using self.xmcdData')
                xmcdX, xmcdY = self.xmcdData
                xmcdIntLabel = 'xmcd Int'
            mathObj = Calculations()
            xasX,  xasY  = self.xasDataCorr
            xmcdIntY = mathObj.cumtrapz(y=xmcdY, x=xmcdX)
            xmcdIntX = .5 * (xmcdX[1:] + xmcdX[:-1])
            xasIntY  = mathObj.cumtrapz(y=xasY,  x=xasX)
            xasIntX  = .5 * (xasX[1:] + xasX[:-1])
            ylabel += ' integral'
            xyList = [(xmcdIntX, xmcdIntY, xmcdIntLabel),
                      (xasX,     xasY,     'xas corr'),
                      (xasIntX,  xasIntY,  'xas Int')]
            self.xmcdInt = xmcdIntX, xmcdIntY
            self.xasInt = xasIntX, xasIntY
        for x,y,legend in xyList:
            self.plotWindow.newCurve(
                    x=x, 
                    y=y,
                    legend=legend,
                    xlabel=xlabel, 
                    #xlabel=xlabel, 
                    ylabel=None, 
                    #ylabel=ylabel,  
                    info={}, 
                    replot=False, 
                    replace=False)
            if mapToY2:
                specLegend = self.plotWindow.dataObjectsList[-1]
                self.plotWindow.graph.mapToY2(specLegend)
                mapToY2 = False
        self.plotWindow.graph.replot()
        
    def addMarker(self, window, label='X MARKER', xpos=None, unit=''):
        # Add spinbox controlling the marker
        spinbox = MarkerSpinBox(window, self.plotWindow, label)

        # Connects
        self.tabChangedSignal.connect(spinbox._setMarkerFollowMouse)
        
        if len(unit) > 0:
            text = label + ' ' + unit
        else:
            text = label
        
        # Widget & Layout
        spinboxWidget = qt.QWidget()
        spinboxLayout = qt.QHBoxLayout()
        spinboxLayout.addWidget(qt.QLabel(text))
        spinboxLayout.addWidget(qt.HorizontalSpacer())
        spinboxLayout.addWidget(spinbox)
        spinboxWidget.setLayout(spinboxLayout)
         
        return spinboxWidget, spinbox

    def _handleGraphSignal(self, ddict):
        #if 'marker' not in ddict:
        if ddict['event'] == 'markerMoved':
            if self.tabWidget.currentIndex() == 1: # 1 -> BG tab
                self.estimateBG()

    def _handleTabChangedSignal(self, idx):
        if idx >= len(self.tabList):
            print('Tab changed -- Index out of range')
            return
        tab = self.tabList[idx]
        self.plotOnDemand(window=tab)
        # Hide/Show markers depending on the selected tab
        # Problem: MarkerLabels are stored in markerList,
        # however the MarkerSpinBoxes are stores in
        # self.valuesDict ...
        # edgeMarkers & xasMarkers -> BACKGROUND  tab
        # xmcdMarker               -> INTEGRATION tab
        markerList = self.xasMarkerList\
                   + self.edgeMarkerList\
                   + self.xmcdMarkerList
        if tab == self.__tabBG:
            self.buttonEstimate.setEnabled(True)
            for marker in markerList:
                if (marker in self.xasMarkerList) or\
                   (marker in self.edgeMarkerList):
                    sb = self.valuesDict[self.__tabBG][marker]
                    sb.showMarker()
                else:
                    sb = self.valuesDict[self.__tabInt][marker]
                    sb.hideMarker()
            keys = [key for key in self.valuesDict[self.__tabElem].keys()\
                    if key.endswith('Transition')]
            ratioSB = self.valuesDict[self.__tabBG]['Step Ratio']
            for idx, keyElem in enumerate(keys):
                keyBG = 'Edge %d'%(idx+1)
                sb = self.valuesDict[self.__tabBG][keyBG]
                parentWidget  = sb.parent()                
                # If edge combobox does not show empty string ''
                transition = str(self.valuesDict[self.__tabElem]\
                                [keyElem].currentText())
                if len(transition) > 0:
                    parentWidget.setEnabled(True)
                else:
                    parentWidget.setEnabled(False)
                    sb.hideMarker()
                    ratioSB.setEnabled(False)
                    ratioSB.setValue(1.0)
        elif tab == self.__tabInt:
            self.buttonEstimate.setEnabled(True)
            for marker in markerList:
                if marker in self.xmcdMarkerList:
                    sb = self.valuesDict[self.__tabInt][marker]
                    sb.showMarker()
                else:
                    sb = self.valuesDict[self.__tabBG][marker]
                    #sb.setValue(0.0) # Should be consistent with estimateBG
                    sb.hideMarker()
        else: # tab == self.__tabElem:
            self.buttonEstimate.setEnabled(False)
            for marker in markerList:
                if marker in self.xmcdMarkerList:
                    sb = self.valuesDict[self.__tabInt][marker]
                else:
                    sb = self.valuesDict[self.__tabBG][marker]
                sb.showMarker()
        self.tabChangedSignal.emit(tab)
    
    def keyPressEvent(self, event):
        if event.key() == qt.Qt.Key_F2:
            # Switch to tab Element
            idx = self.tabList.index(self.__tabElem)
            self.tabWidget.setCurrentIndex(idx)
        elif event.key() == qt.Qt.Key_F3:
            # Switch to tab Background
            idx = self.tabList.index(self.__tabBG)
            self.tabWidget.setCurrentIndex(idx)
        elif event.key() == qt.Qt.Key_F4:
            # Switch to tab Integration
            idx = self.tabList.index(self.__tabInt)
            self.tabWidget.setCurrentIndex(idx)
        elif event.key() == qt.Qt.Key_F5:
            # Trigger estimation
            self.estimate()
        else:
            qt.QWidget.keyPressEvent(self, event)
        
def getData(fn='/home/truter/lab/datasets/sum_rules/Fe_L23/xld_analysis.spec'):
    analysis = sf.specfile.Specfile(fn)
    xmcdArr = []
    xasArr  = []
    avgA, avgB = [], []
    spec = analysis[0]
    x = spec[0][:]
    avgA = spec[1][:]
    avgB = spec[2][:]
    xmcdArr = spec[3][:]
    xasArr  = spec[4][:]
    return x, avgA, avgB, xmcdArr, xasArr

class LoadDichorismDataDialog(qt.QFileDialog):
    
    dataInputSignal = qt.pyqtSignal(object)
    
    def __init__(self, parent=None):
        #qt.QDialog.__init__(self, parent)
        qt.QFileDialog.__init__(self, parent)
        self.dataDict = {}
        self.validated = False
        
        self.setWindowTitle('Load Dichorism Data')
        self.setFilter('Spec Files (*.spec);;'
                      +'All Files (*.*)')
        
        # Take the QSpecFileWidget class as used
        # in the main window to select data and
        # insert it into a QFileDialog. Emit the
        # selected data at acceptance
        self.specFileWidget = QSpecFileWidget.QSpecFileWidget(
                                        parent=parent,
                                        autoreplace=False)
        # Hide the widget containing the Auto Add/Replace
        # checkboxes
        self.specFileWidget.autoAddBox.parent().hide()
        # Remove the tab widget, only the counter widget
        # is needed. Remember: close() only hides a widget
        # however the widget persists in the memory.
        #self.specFileWidget.mainTab.removeTab(1)
        self.specFileWidget.mainTab.hide()
        #self.counterTab = self.specFileWidget.mainTab.widget(0)
        self.specFileWidget.mainLayout.addWidget(self.specFileWidget.cntTable)
        self.specFileWidget.cntTable.show()
        # Change the table headers in cntTable
        # Note: By conicidence, the original SpecFileCntTable
        # has just enough columns as we need. Here, we rename
        # the last two:
        # 'y'   -> 'XAS'
        # 'mon' -> 'XMCD'
        labels = ['Counter', 'X', 'XAS', 'XMCD']
        table  = self.specFileWidget.cntTable
        for idx in range(len(labels)):
            item = table.horizontalHeaderItem(idx)
            if item is None:
                item = qt.QTableWidgetItem(labels[idx],
                                           qt.QTableWidgetItem.Type)
            item.setText(labels[idx])
            table.setHorizontalHeaderItem(idx,item)

        
        # Hide the widget containing the Add, Replace, ...
        # PushButtons
        self.specFileWidget.buttonBox.hide()
        
        # Change selection behavior/mode in the scan list so
        # that only a single scan can be selected at a time
        self.specFileWidget.list.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.specFileWidget.list.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        
        # Tinker with the native layout of QFileDialog
        mainLayout = self.layout()
        mainLayout.addWidget(self.specFileWidget, 0, 4, 4, 1)
        
        #
        # Signals
        #
        self.currentChanged.connect(self.setDataSource)
    
    def setDataSource(self, filename):
        # Opens a spec file and allows to browse its
        # contents in the top right widget
        filename = str(filename)
        if osPathIsDir(filename) or (not filename.endswith('.spec')):
            return
        src = SpecFileDataSource.SpecFileDataSource(filename)
        self.specFileWidget.setDataSource(src)
    
    def accept(self):
        llist = self.selectedFiles()
        if len(llist) == 1:
            filename = str(llist[0])
        else:
            return
        self.processSelectedFile(filename)
        if self.validated:
            qt.QDialog.accept(self)
        
    def processSelectedFile(self, filename):
        self.dataDict = {}
        filename = str(filename)
            
        if (not filename.endswith('.spec')):
            return
            
        scanList = self.specFileWidget.list.selectedItems()
        if len(scanList) == 0:
            self.errorMessageBox('No scan selected!')
            return
        else:
            scan   = scanList[0]
            scanNo = str(scan.text(1))
            
        table = self.specFileWidget.cntTable
        # ddict['x'] -> 'X'
        # ddict['y'] -> 'XAS'
        # ddict['m'] -> 'XMCD'
        ddict   = table.getCounterSelection()
        colX    = ddict['x']
        colXas  = ddict['y']
        colXmcd = ddict['m']
        # Check if only one is selected
        if len(colX) != 1:
            self.errorMessageBox('Single counter must be set as X')
            return
        else:
            colX = colX[0]
            
        if len(colXas) != 1:
            self.errorMessageBox('Single counter must be set as XAS')
            return
        else:
            colXas = colXas[0]
            
        if len(colXmcd) != 1:
            self.errorMessageBox('Single counter must be set as XMCD')
            return
        else:
            colXmcd = colXmcd[0]
            
        if colXas == colX:
            self.errorMessageBox('X and XAS use the same counter')
            return
        elif colX == colXmcd:
            self.errorMessageBox('X and XMCD use the same counter')
            return
        elif colXmcd == colXas:
            self.errorMessageBox('XAS and XMCD use the same counter')
            return
            
        # Extract data
        dataObj = self.specFileWidget.data.getDataObject(scanNo)
        # data has format (rows, cols) -> (steps, counters)
        self.dataDict['fn']   = filename
        self.dataDict['x']    = dataObj.data[:, colX]
        self.dataDict['xas']  = dataObj.data[:, colXas]
        self.dataDict['xmcd'] = dataObj.data[:, colXmcd]
        
        self.validated = True
        self.dataInputSignal.emit(self.dataDict)
        #self.destroy() ?
        #self.close()
        
    def errorMessageBox(self, msg):
        box = qt.QMessageBox()
        box.setWindowTitle('Sum Rules Load Data Error')
        box.setIcon(qt.QMessageBox.Warning)
        box.setText(msg)
        box.exec_()
        
        

if __name__ == '__main__':
   
    app = qt.QApplication([])
    win = SumRulesWindow()
    #x, avgA, avgB, xmcd, xas = getData()
    #win.plotWindow.newCurve(x,xmcd, legend='xmcd', xlabel='ene_st', ylabel='zratio', info={}, replot=False, replace=False)
    #win.setRawData(x,xmcd, identifier='xmcd')
    #win.plotWindow.newCurve(x,xas, legend='xas', xlabel='ene_st', ylabel='zratio', info={}, replot=False, replace=False)
    #win.setRawData(x,xas, identifier='xas')
    #win = LoadDichorismDataDialog()
    win.show()
    app.exec_()
