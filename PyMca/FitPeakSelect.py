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
import types
from EnergyTable import qt
from QPeriodicTable import QPeriodicTable
#from QPeriodicTable import ElementList
import EnergyTable
import Elements
import copy
DEBUG = 0
QTVERSION = qt.qVersion()
ElementList = Elements.ElementList
__revision__ = "$Revision: 1.12 $"
class PeakButton(qt.QPushButton):
    def __init__(self, parent, peak):
        qt.QPushButton.__init__(self, parent)
        #, peak)
        self.peak= peak

        font= self.font()
        font.setBold(1)
        self.setText(peak)
        self.setFlat(1)
        if qt.qVersion() < '4.0.0':
            self.setToggleButton(0)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding))

        self.selected= 0
        self.brush= qt.QBrush(qt.QColor(qt.Qt.yellow))

        self.connect(self, qt.SIGNAL("clicked()"), self.clickedSlot)

    def toggle(self):
        self.selected= not self.selected
        self.update()

    def setSelected(self, b):
        self.selected= b
        if QTVERSION > '4.0.0':
            if b:
                role = self.backgroundRole()
                palette = self.palette()
                palette.setBrush( role,self.brush)
                self.setPalette(palette)
            else:
                role = self.backgroundRole()
                palette = self.palette()
                palette.setBrush( role, qt.QBrush())
                self.setPalette(palette)
        self.update()

    def isSelected(self):
        return self.selected

    def clickedSlot(self):
        self.toggle()
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("peakClicked"), (self.peak,))
        else:
            self.emit(qt.SIGNAL("peakClicked(QString)"), self.peak)

    def paintEvent(self, pEvent):
        if qt.qVersion() < '4.0.0':
            qt.QPushButton.paintEvent(self, pEvent)
        else:
            p = qt.QPainter(self)
            wr= self.rect()
            pr= qt.QRect(wr.left()+1, wr.top()+1, wr.width()-2, wr.height()-2)
            if self.selected:
                p.fillRect(pr, self.brush)
            p.setPen(qt.Qt.black)
            p.drawRoundRect(pr)
            p.end()
            qt.QPushButton.paintEvent(self, pEvent)

    def drawButton(self, p):
        wr= self.rect()
        pr= qt.QRect(wr.left()+1, wr.top()+1, wr.width()-2, wr.height()-2)
        if self.selected:
                p.fillRect(pr, self.brush)
        qt.QPushButton.drawButtonLabel(self, p)
        p.setPen(qt.Qt.black)
        if qt.qVersion() < '3.0.0':        
            p.drawRoundRect(pr,25,25)
        else:
            p.drawRoundRect(pr)

class PeakButtonList(qt.QWidget):
    def __init__(self, parent=None, name="PeakButtonList",
                 peaklist=['K','Ka','Kb','L','L1','L2','L3','M'],
                 fl=0):
        qt.QWidget.__init__(self,parent)
        self.peaklist = peaklist

        if qt.qVersion() < '4.0.0':
            layout= qt.QHBoxLayout(self, 0, 5)
        else:
            layout= qt.QHBoxLayout(self)
            layout.setMargin(0)
            layout.setSpacing(5)
            #, 0, 5)
            
        layout.addStretch(2)

        self.buttondict={}
        for key in peaklist:
            self.buttondict[key] = PeakButton(self, key)
            layout.addWidget(self.buttondict[key])
            if qt.qVersion() < '4.0.0':
                self.connect(self.buttondict[key],
                             qt.PYSIGNAL("peakClicked"), self.__selection)
            else:
                self.connect(self.buttondict[key],
                             qt.SIGNAL("peakClicked(QString)"), self.__selection)

        layout.addStretch(1)

        #Reset 
        self.resetBut = qt.QPushButton(self)
        self.resetBut.setText("Reset")
        layout.addWidget(self.resetBut)
        self.connect(self.resetBut,qt.SIGNAL('clicked()'),self.__resetBut)

        layout.addStretch(2)        

    def __resetBut(self):
        for key in self.peaklist:
                    self.buttondict[key].setSelected(0)
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('selectionChanged'),([],))
        else:
            self.emit(qt.SIGNAL('selectionChanged'),([]))

    def __selection(self, peak):
        selection= []
        for key in self.peaklist:
                if self.buttondict[key].isSelected():
                        selection.append(key)
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("selectionChanged"), (selection,))
        else:
            self.emit(qt.SIGNAL("selectionChanged"), (selection))
            

    def setSelection(self, selection=[]):
        for key in self.peaklist:
                if key in selection:
                        self.buttondict[key].setSelected(1)
                else:        self.buttondict[key].setSelected(0)

    def setDisabled(self,selection=[]):
        for key in self.peaklist:
            if key in selection:
                self.buttondict[key].setEnabled(0)
            else:        self.buttondict[key].setEnabled(1)
        

class FitPeakSelect(qt.QWidget):
    def __init__(self, parent=None, name="FitPeakSelect",peakdict = {}, fl=0, energyTable = None):
        qt.QWidget.__init__(self,parent)

        if QTVERSION < '4.0.0':
            self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Minimum,
                                              qt.QSizePolicy.Minimum))

        layout=qt.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(10)
        hbox = qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.setMargin(0)
        hboxLayout.setSpacing(20)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        l1=MyQLabel(hbox, bold=True, color=qt.QColor(0,0,0))
        hboxLayout.addWidget(l1)

        self.energyValue = None
        if energyTable is not None:
            text = '<b><nobr>Excitation Energy (keV)</nobr></b>'
            l1.setFixedWidth(l1.fontMetrics().width("##"+text+"####"))
            l1.setText(text)
            self.energyTable = energyTable
            add = 0
            self.energy = MyQLabel(hbox)
            hboxLayout.addWidget(self.energy)
            self.energy.setFixedWidth(self.energy.fontMetrics().width('########.###'))
            self.energy.setAlignment(qt.Qt.AlignLeft)
            #self.energy.setForegroundColor(qt.Qt.red)
        else:
            l1.setText('<b><nobr>Excitation Energy (keV)</nobr></b>')
            self.energyTable = EnergyTable.EnergyTable(self)
            add = 1
            self.energy = qt.QLineEdit(hbox)
            hboxLayout.addWidget(self.energy)
            self.energy.setFixedWidth(self.energy.fontMetrics().width('########.###'))
            self.energyButton = qt.QPushButton(hbox)
            hboxLayout.addWidget(self.energyButton)
            self.energyButton.setText("Update")
            self.connect(self.energyButton, qt.SIGNAL('clicked()'),
                         self._energyClicked)

        hboxLayout.addWidget(HorizontalSpacer(hbox))
        layout.addSpacing(20)
        layout.addWidget(hbox)

        self.table = QPeriodicTable(self)
        line= qt.QFrame(self)
        line.setFrameShape(qt.QFrame.HLine)
        line.setFrameShadow(qt.QFrame.Sunken)
        
        self.peaks = PeakButtonList(self)
        self.peaks.setDisabled(['K','Ka','Kb','L','L1','L2','L3','M'])

        if qt.qVersion() < '4.0.0':
            self.connect(self.energyTable, qt.PYSIGNAL("EnergyTableSignal"),
                         self._energyTableAction)
            self.connect(self.table, qt.PYSIGNAL("elementClicked"),
                         self.elementClicked)
            self.connect(self.peaks, qt.PYSIGNAL("selectionChanged"),
                         self.peakSelectionChanged)
        else:
            self.connect(self.energyTable, qt.SIGNAL("EnergyTableSignal"),
                         self._energyTableAction)
            self.connect(self.table, qt.SIGNAL("elementClicked"),
                         self.elementClicked)
            self.connect(self.peaks, qt.SIGNAL("selectionChanged"),
                         self.peakSelectionChanged)
            #Reset All
            self.resetAllButton = qt.QPushButton(self.peaks)
            palette = qt.QPalette(self.resetAllButton.palette())
            role = self.resetAllButton.foregroundRole()
            palette.setColor(role, qt.Qt.red)
            self.resetAllButton.setPalette(palette)
            self.resetAllButton.setText("Reset All")
            self.peaks.layout().addWidget(self.resetAllButton)

            self.connect(self.resetAllButton, qt.SIGNAL("clicked()"),
                         self.__resetAll)

        layout.addWidget(self.table)
        layout.addWidget(line)
        layout.addWidget(self.peaks)
        if add:layout.addWidget(self.energyTable)
        layout.addStretch(1)

        self.current= None 
        self.setSelection(peakdict)

    def __resetAll(self):
        msg=qt.QMessageBox.warning( self, "Clear selection",
                      "Do you want to reset the selection for all elements?",
                      qt.QMessageBox.Yes,qt.QMessageBox.No)
        if msg == qt.QMessageBox.No:
            return

        self.peakdict = {}
        self.table.setSelection(list(self.peakdict.keys()))
        self.peaks.setSelection([])
        self.peakSelectionChanged([])

    def __getZ(self,element):
        return ElementList.index(element) + 1
       
    def setSelection(self,peakdict):
        self.peakdict = {}
        self.peakdict.update(peakdict)
        for key in list(self.peakdict.keys()):
                if type(self.peakdict[key])!= type([]):
                        self.peakdict[key]= [ self.peakdict[key] ]
        self.table.setSelection(list(self.peakdict.keys()))
        
    def getSelection(self):
        ddict={}
        for key in list(self.peakdict.keys()):
                if len(self.peakdict[key]):
                        ddict[key]= self.peakdict[key]
        return ddict 

    def peakSelectionChanged(self,selection):
        if self.current is None: return
        if type(selection) != type([]):
            selection=selection.list
        self.peakdict[self.current] = selection
        if len(self.peakdict[self.current]):
            self.table.setElementSelected(self.current,1)
        else:
            self.table.setElementSelected(self.current,0)
        sel= self.getSelection()
        sel['current'] = self.current
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("FitPeakSelect"), (sel,))
        else:
            self.emit(qt.SIGNAL("FitPeakSelect"), (sel))

    def elementClicked(self,symbol):
        if qt.qVersion() > '4.0.0':symbol = str(symbol)
        if not (symbol in self.peakdict):
            self.peakdict[symbol] = []
        self.current = symbol
        if len(self.peakdict[self.current]):
            self.table.setElementSelected(self.current,1)
        else:
            self.table.setElementSelected(self.current,0)
        for ele in list(self.peakdict.keys()):
            if ele != symbol:
                if not len(self.peakdict[ele]):
                    del self.peakdict[ele]
        sel= self.getSelection()
        sel['current'] = self.current
        self.setPeaksDisabled(symbol)
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("FitPeakSelect"), (sel,))
        else:
            self.emit(qt.SIGNAL("FitPeakSelect"),(sel))
        self.peaks.setSelection(self.peakdict[symbol])

    def setPeaksDisabled(self,symbol):
        z = self.__getZ(symbol)
        if (z > 47) and (Elements.getomegam5('Cd') > 0.0):
            #we have data available to support that
            disabled = []
        elif z > 66:
            #self.peaks.setDisabled(['Ka','Kb'])
            #disabled = ['Ka','Kb']
            disabled = []
        elif z > 17:
            #self.peaks.setDisabled(['Ka','Kb','M'])
            #disabled = ['Ka','Kb','M']
            disabled = ['M']
        elif z > 2:
            #self.peaks.setDisabled(['Ka','Kb','L','L1','L2','L3','M'])
            #disabled = ['Ka','Kb','L','L1','L2','L3','M']
            disabled = ['L','L1','L2','L3','M']
        else:
            #self.peaks.setDisabled(['K','Ka','Kb','L','L1','L2','L3','M'])
            #disabled = ['Ka','Kb','L','L1','L2','L3','M']
            disabled = ['Ka', 'Kb','L','L1','L2','L3','M']

        ele = symbol
        if self.energyValue is not None:
            for peak in ['K', 'Ka', 'Kb', 'L','L1','L2','L3','M']:
                if peak not in disabled:
                    if peak == 'L':
                        if Elements.Element[ele]['binding']['L3'] > self.energyValue:
                            disabled.append(peak)
                    elif peak == 'M':
                        if Elements.Element[ele]['binding']['M5'] > self.energyValue:
                            disabled.append(peak)
                    elif peak == 'Ka':
                        if Elements.Element[ele]['binding']['K'] > self.energyValue:
                            disabled.append(peak)
                    elif peak == 'Kb':
                        if Elements.Element[ele]['binding']['K'] > self.energyValue:
                            disabled.append(peak)
                    elif Elements.Element[ele]['binding'][peak] > self.energyValue:
                            disabled.append(peak)
                    else:
                        pass
        self.peaks.setDisabled(disabled)    

    def setEnergy(self, energy):
        if (energy is None) or (energy == []):
            self.energyValue = energy
            self.energy.setText("None")
        elif energy == "None":
            self.energyValue = None
            self.energy.setText("None")
        elif type(energy) == types.ListType:
            self.energyValue = max(energy)
        else:
            self.energyValue = energy
            self.energy.setText("%.4f" % energy)
        self._energyClicked()

    def _energyTableAction(self, ddict):
        if DEBUG:
            print("_energyTableAction called",)
            print("ddict = ",ddict.dict)
        elist, wlist, flist, slist= self.energyTable.getParameters()
        maxenergy = 0.0
        for i in range(len(flist)):
            if flist[i]:
                if elist[i] is not None:
                    if wlist[i] > 0.0:
                        if elist[i] > maxenergy:
                            maxenergy = elist[i]
        if maxenergy == 0.0:maxenergy = None
        self.setEnergy(maxenergy)

    def _energyClicked(self):
        string = str(self.energy.text())
        string.replace(" ","")
        if (string != "None") and len(string):
            try:
                value = float(string)
                self.energyValue = value
                if False:
                    self.energyButton.setFocus()   
            except:
                msg=qt.QMessageBox(self.energy)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Invalid Float")
                msg.exec_loop()
                self.energy.setFocus()
        else:
            self.energyValue = None
            if False:
                self.energyButton.setFocus()
        self.__updateSelection()
        
    def __updateSelection(self):
        if self.energyValue is not None:
            changed = 0
            for ele in list(self.peakdict.keys()):
                for peak in self.peakdict[ele]:
                    if   peak in self.peakdict[ele]:
                        index = self.peakdict[ele].index(peak)
                    if   peak == 'L':
                        if Elements.Element[ele]['binding']['L3'] > self.energyValue:
                            del self.peakdict[ele][index]
                    elif peak == 'M':
                        if Elements.Element[ele]['binding']['M5'] > self.energyValue:
                            del self.peakdict[ele][index]
                    elif peak == "Ka":
                        if Elements.Element[ele]['binding']['K'] > self.energyValue:
                            del self.peakdict[ele][index]
                    elif peak == "Kb":
                        if Elements.Element[ele]['binding']['K'] > self.energyValue:
                            del self.peakdict[ele][index]
                    elif Elements.Element[ele]['binding'][peak] > self.energyValue:
                        del self.peakdict[ele][index]
                    else:
                        pass
                if ele == self.current:
                    self.peaks.setSelection(self.peakdict[ele])
                    self.peakSelectionChanged(self.peakdict[ele])
                    self.elementClicked(ele)
                if not len(self.peakdict[ele]): del self.peakdict[ele]
        dict = copy.deepcopy(self.peakdict)
        self.setSelection(dict)
        
class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=None):
        qt.QLineEdit.__init__(self,parent,name)

    def focusInEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('yellow'))

    def focusOutEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('white'))

    def setPaletteBackgroundColor(self, qcolor):
        if qt.qVersion() < '3.0.0':
            palette = self.palette()
            palette.setColor(qt.QColorGroup.Base,qcolor)
            self.setPalette(palette)
            text = self.text()
            self.setText(text)
        else:
            qt.QLineEdit.setPaletteBackgroundColor(self,qcolor)

class MyQLabel(qt.QLabel):
    def __init__(self,parent=None,name=None,fl=0,bold=True, color= qt.Qt.red):
        qt.QLabel.__init__(self,parent)
        if qt.qVersion() <'4.0.0':
            self.color = color
            self.bold  = bold
        else:
            palette = self.palette()
            role = self.foregroundRole()
            palette.setColor(role,color)
            self.setPalette(palette)
            self.font().setBold(bold)


    if qt.qVersion() < '4.0.0':
        def drawContents(self, painter):
            painter.font().setBold(self.bold)
            pal =self.palette()
            pal.setColor(qt.QColorGroup.Foreground,self.color)
            self.setPalette(pal)
            qt.QLabel.drawContents(self,painter)
            painter.font().setBold(0)

def testwidget():
    import sys
    def change(ddict):
        print("New selection:",)
        print(ddict)
    a = qt.QApplication(sys.argv)
    qt.QObject.connect(a,qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))

    w = qt.QTabWidget()

    if qt.qVersion() < '4.0.0':
        f = FitPeakSelect(w)
        w.addTab(f, "QPeriodicTable")
        qt.QObject.connect(f, qt.PYSIGNAL("FitPeakSelect"), change)
        w.show()
        a.exec_loop()
    else:
        f = FitPeakSelect()
        w.addTab(f, "QPeriodicTable")
        qt.QObject.connect(f, qt.SIGNAL("FitPeakSelect"), change)
        w.show()
        a.exec_()

if __name__ == "__main__":
    testwidget()
