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
from PyMca import DoubleSlider
qt = DoubleSlider.qt

QTVERSION = qt.qVersion()
DEBUG = 0
    
class RGBCorrelatorSlider(qt.QWidget):
    def __init__(self, parent = None, scale = False, autoscalelimits=None):
        qt.QWidget.__init__(self, parent)
        self.__emitSignals = True
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self._buttonBox = qt.QWidget(self)
        self._buttonBoxLayout = qt.QGridLayout(self._buttonBox)
        self._buttonBoxLayout.setMargin(0)
        self._buttonBoxLayout.setSpacing(0)

        if autoscalelimits is None:
            self.fromA =  5.0
            self.toB   = 95.0
        else:
            self.fromA = autoscalelimits[0]
            self.toB   = autoscalelimits[1]
            if self.fromA > self.toB:
                self.fromA = autoscalelimits[1]
                self.toB   = autoscalelimits[0]
            
        self.autoScaleButton = qt.QPushButton(self._buttonBox)
        self.autoScaleButton.setText("Autoscale")
        self.autoScaleFromAToBButton = qt.QPushButton(self._buttonBox)
        self.autoScaleFromAToBButton.setText("Autoscale %d-%d" %
                                             (int(self.fromA), int(self.toB)))
        self.autoScale90Button = qt.QPushButton(self._buttonBox)
        self.autoScale90Button.setText("Autoscale 0-90")
        self._buttonBoxLayout.addWidget(self.autoScaleButton, 0, 0)
        self._buttonBoxLayout.addWidget(self.autoScaleFromAToBButton, 0, 1)
        self._buttonBoxLayout.addWidget(self.autoScale90Button, 0, 2)

        self._gridBox = qt.QWidget(self)
        self._gridBoxLayout = qt.QGridLayout(self._gridBox)
        self._gridBoxLayout.setMargin(0)
        self._gridBoxLayout.setSpacing(0)

        redLabel = MyQLabel(self._gridBox, color = qt.Qt.red)
        redLabel.setText("RED")
        self.redSlider = DoubleSlider.DoubleSlider(self._gridBox)
        
        greenLabel = MyQLabel(self._gridBox, color = qt.Qt.green)
        greenLabel.setText("GREEN")
        self.greenSlider = DoubleSlider.DoubleSlider(self._gridBox)

        blueLabel = MyQLabel(self._gridBox, color = qt.Qt.blue)
        blueLabel.setText("BLUE")
        self.blueSlider = DoubleSlider.DoubleSlider(self._gridBox, scale = True)

        self._gridBoxLayout.addWidget(redLabel, 0, 0)
        self._gridBoxLayout.addWidget(self.redSlider, 0, 1)

        self._gridBoxLayout.addWidget(greenLabel, 1, 0)
        self._gridBoxLayout.addWidget(self.greenSlider, 1, 1)

        self._gridBoxLayout.addWidget(blueLabel, 2, 0)
        self._gridBoxLayout.addWidget(self.blueSlider, 2, 1)

        self.mainLayout.addWidget(self._buttonBox)
        self.mainLayout.addWidget(self._gridBox)


        self.connect(self.redSlider,
                     qt.SIGNAL("doubleSliderValueChanged"),
                     self._redSliderChanged)

        self.connect(self.greenSlider,
                     qt.SIGNAL("doubleSliderValueChanged"),
                     self._greenSliderChanged)

        self.connect(self.blueSlider,
                     qt.SIGNAL("doubleSliderValueChanged"),
                     self._blueSliderChanged)

        self.connect(self.autoScaleButton,
                     qt.SIGNAL("clicked()"),
                     self.autoScale)
        
        self.connect(self.autoScaleFromAToBButton,
                     qt.SIGNAL("clicked()"),
                     self.autoScaleFromAToB)
        
        self.connect(self.autoScale90Button,
                     qt.SIGNAL("clicked()"),
                     self.autoScale90)
    
    def autoScale(self):
        self.__emitSignals = False
        self.redSlider.setMinMax(0., 100.)
        self.greenSlider.setMinMax(0.0, 100.)
        self.blueSlider.setMinMax(0., 100.)
        self.__emitSignals = True
        self._allChangedSignal()
        
    def autoScaleFromAToB(self):
        self.__emitSignals = False
        self.redSlider.setMinMax( self.fromA, self.toB)
        self.greenSlider.setMinMax(self.fromA, self.toB)
        self.blueSlider.setMinMax(self.fromA, self.toB)
        self.__emitSignals = True
        self._allChangedSignal()
        
    def autoScale90(self):
        self.__emitSignals = False
        self.redSlider.setMinMax(0., 90.)
        self.greenSlider.setMinMax(0.0, 90.)
        self.blueSlider.setMinMax(0., 90.)
        self.__emitSignals = True
        self._allChangedSignal()

    def _allChangedSignal(self):
        ddict = {}
        ddict['event'] = "allChanged"
        ddict['red']   = self.redSlider.getMinMax()
        ddict['green'] = self.greenSlider.getMinMax()
        ddict['blue']  = self.blueSlider.getMinMax()
        self.emit(qt.SIGNAL("RGBCorrelatorSliderSignal"),
                      ddict)

    def _redSliderChanged(self, ddict):
        if DEBUG:
            print("RGBCorrelatorSlider._redSliderChanged()")
        if self.__emitSignals:
            ddict['event'] = "redChanged"
            self.emit(qt.SIGNAL("RGBCorrelatorSliderSignal"),
                      ddict)
            

    def _greenSliderChanged(self, ddict):
        if DEBUG:
            print("RGBCorrelatorSlider._greenSliderChanged()")
        if self.__emitSignals:
            ddict['event'] = "greenChanged"
            self.emit(qt.SIGNAL("RGBCorrelatorSliderSignal"),
                      ddict)

    def _blueSliderChanged(self, ddict):
        if DEBUG:
            print("RGBCorrelatorSlider._blueSliderChanged()")
        if self.__emitSignals:
            ddict['event'] = "blueChanged"
            self.emit(qt.SIGNAL("RGBCorrelatorSliderSignal"),
                      ddict)

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

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit'))

    def slot(ddict):
        print("received dict = ", ddict)
    w = RGBCorrelatorSlider()
    app.connect(w, qt.SIGNAL("RGBCorrelatorSliderSignal"), slot)
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
