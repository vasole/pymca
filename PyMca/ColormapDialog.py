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
import QtBlissGraph
qt = QtBlissGraph.qt
qwt = QtBlissGraph.qwt
import DoubleSlider
QTVERSION = qt.qVersion()
QWTVERSION4 = QtBlissGraph.QWTVERSION4

import os
import numpy.oldnumeric as Numeric
DEBUG = 0

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=""):
        qt.QLineEdit.__init__(self,parent)

    def focusInEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('yellow'))

    def focusOutEvent(self,event):
        self.setPaletteBackgroundColor(qt.QColor('white'))
        if QTVERSION < '4.0.0':
            self.emit(qt.SIGNAL("returnPressed()"),())
        else:
            self.emit(qt.SIGNAL("returnPressed()"))
        
    def setPaletteBackgroundColor(self, qcolor):
        if qt.qVersion() < '3.0.0':
            palette = self.palette()
            palette.setColor(qt.QColorGroup.Base,qcolor)
            self.setPalette(palette)
            text = self.text()
            self.setText(text)
        elif QTVERSION < '4.0.0':
            qt.QLineEdit.setPaletteBackgroundColor(self,qcolor)
        else:
            pass
"""
Manage colormap Widget class
"""
class ColormapDialog(qt.QDialog):
    def __init__(self, parent=None, name="Colormap Dialog", slider = False):
        if QTVERSION < '4.0.0':
            qt.QDialog.__init__(self, parent, name)
            self.setCaption(name)
        else:
            qt.QDialog.__init__(self, parent)
            self.setWindowTitle(name)
        self.title = name
                 
        
        self.colormapList = ["Greyscale", "Reverse Grey", "Temperature",
                             "Red", "Green", "Blue", "Many"]
                         

        # default values
        self.dataMin   = -10
        self.dataMax   = 10
        self.minValue  = 0
        self.maxValue  = 1

        self.colormapIndex  = 2
        self.colormapType   = 0

        self.autoscale   = False
        self.autoscale90 = False
        # main layout
        if QTVERSION < '4.0.0':
            vlayout = qt.QVBoxLayout(self, 0, -1, "Main ColormapDialog Layout")
        else:
            vlayout = qt.QVBoxLayout(self)
        vlayout.setMargin(10)
        vlayout.setSpacing(0)

        # layout 1 : -combo to choose colormap
        #            -autoscale button
        #            -autoscale 90% button
        hbox1    = qt.QWidget(self)
        hlayout1 = qt.QHBoxLayout(hbox1)
        vlayout.addWidget(hbox1)
        hlayout1.setMargin(0)
        hlayout1.setSpacing(10)

        # combo
        self.combo = qt.QComboBox(hbox1)
        for colormap in self.colormapList:
            if QTVERSION < '4.0.0':
                self.combo.insertItem(colormap)
            else:
                self.combo.addItem(colormap)
        self.connect(self.combo,
                     qt.SIGNAL("activated(int)"),
                     self.colormapChange)
        hlayout1.addWidget(self.combo)

        # autoscale
        self.autoScaleButton = qt.QPushButton("Autoscale", hbox1)
        if QTVERSION < '4.0.0':
            self.autoScaleButton.setToggleButton(True)
        else:
            self.autoScaleButton.setCheckable(True)
        self.autoScaleButton.setAutoDefault(False)    
        self.connect(self.autoScaleButton,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscaleChange)
        hlayout1.addWidget(self.autoScaleButton)

        # autoscale 90%
        self.autoScale90Button = qt.QPushButton("Autoscale 90%", hbox1)
        if QTVERSION < '4.0.0':
            self.autoScale90Button.setToggleButton(True)
        else:
            self.autoScale90Button.setCheckable(True)
        self.autoScale90Button.setAutoDefault(False)    
                
        self.connect(self.autoScale90Button,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscale90Change)
        hlayout1.addWidget(self.autoScale90Button)

        # hlayout
        if QTVERSION > '4.0.0':
            hbox0    = qt.QWidget(self)
            self.__hbox0 = hbox0
            hlayout0 = qt.QHBoxLayout(hbox0)
            hlayout0.setMargin(0)
            hlayout0.setSpacing(0)
            vlayout.addWidget(hbox0)
            #hlayout0.addStretch(10)

            self.buttonGroup = qt.QButtonGroup()
            g1 = qt.QCheckBox(hbox0)
            g1.setText("Linear")
            g2 = qt.QCheckBox(hbox0)
            g2.setText("Logarithmic")
            g3 = qt.QCheckBox(hbox0)
            g3.setText("Gamma")
            self.buttonGroup.addButton(g1, 0)
            self.buttonGroup.addButton(g2, 1)
            self.buttonGroup.addButton(g3, 2)
            self.buttonGroup.setExclusive(True)
            if self.colormapType == 1:
                self.buttonGroup.button(1).setChecked(True)
            elif self.colormapType == 2:
                self.buttonGroup.button(2).setChecked(True)
            else:
                self.buttonGroup.button(0).setChecked(True)
            hlayout0.addWidget(g1)
            hlayout0.addWidget(g2)
            hlayout0.addWidget(g3)
            vlayout.addWidget(hbox0)
            self.connect(self.buttonGroup,
                         qt.SIGNAL("buttonClicked(int)"),
                         self.buttonGroupChange)


        vlayout.addSpacing(20)

        hboxlimits = qt.QWidget(self)
        hboxlimitslayout = qt.QHBoxLayout(hboxlimits)
        hboxlimitslayout.setMargin(0)
        hboxlimitslayout.setSpacing(0)

        if slider:
            self.slider = DoubleSlider.DoubleSlider(hboxlimits, scale = False)
            hboxlimitslayout.addWidget(self.slider)
        else:
            self.slider = None

        vlayout.addWidget(hboxlimits)

        vboxlimits = qt.QWidget(hboxlimits)
        vboxlimitslayout = qt.QVBoxLayout(vboxlimits)
        vboxlimitslayout.setMargin(0)
        vboxlimitslayout.setSpacing(0)
        hboxlimitslayout.addWidget(vboxlimits)

        # hlayout 2 : - min label
        #             - min texte
        hbox2    = qt.QWidget(vboxlimits)
        self.__hbox2 = hbox2
        hlayout2 = qt.QHBoxLayout(hbox2)
        hlayout2.setMargin(0)
        hlayout2.setSpacing(0)
        #vlayout.addWidget(hbox2)
        vboxlimitslayout.addWidget(hbox2)
        hlayout2.addStretch(10)
        
        self.minLabel  = qt.QLabel(hbox2)
        self.minLabel.setText("Minimum")
        hlayout2.addWidget(self.minLabel)
        
        hlayout2.addSpacing(5)
        hlayout2.addStretch(1)
        self.minText  = MyQLineEdit(hbox2)
        self.minText.setFixedWidth(150)
        self.minText.setAlignment(qt.Qt.AlignRight)
        self.connect(self.minText,
                     qt.SIGNAL("returnPressed()"),
                     self.minTextChanged)
        hlayout2.addWidget(self.minText)
        
        # hlayout 3 : - min label
        #             - min text
        hbox3    = qt.QWidget(vboxlimits)
        self.__hbox3 = hbox3
        hlayout3 = qt.QHBoxLayout(hbox3)
        hlayout3.setMargin(0)
        hlayout3.setSpacing(0)
        #vlayout.addWidget(hbox3)
        vboxlimitslayout.addWidget(hbox3)
        
        hlayout3.addStretch(10)
        self.maxLabel = qt.QLabel(hbox3)
        self.maxLabel.setText("Maximum")
        hlayout3.addWidget(self.maxLabel)
        
        hlayout3.addSpacing(5)
        hlayout3.addStretch(1)
                
        self.maxText = MyQLineEdit(hbox3)
        self.maxText.setFixedWidth(150)
        self.maxText.setAlignment(qt.Qt.AlignRight)

        self.connect( self.maxText,
                      qt.SIGNAL("returnPressed()"),
                      self.maxTextChanged)
        hlayout3.addWidget(self.maxText)


        # Graph widget for color curve...
        self.c = QtBlissGraph.QtBlissGraph(self)
        self.c.xlabel("Data Values")
        self.c.enableZoom(False)
        self.c.setCanvasBackground(qt.Qt.white)
        self.c.canvas().setMouseTracking(1)

        self.c.enableAxis(qwt.QwtPlot.xBottom)
        
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge

        self.c.setx1axislimits(self.minmd, self.maxpd)
        self.c.sety1axislimits(-11.5, 11.5)
        if QWTVERSION4:
            self.c.setAutoLegend(0)
            self.c.enableLegend(0)
            self.c.enableXTopAxis(0)
            self.c.enableYLeftAxis(0)
            self.c.enableYRightAxis(0)
        else:
            self.c.picker.setSelectionFlags(qwt.QwtPicker.NoSelection)

        x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        y = [-10, -10, 10, 10 ]
        self.c.newCurve("ConstrainedCurve", x, y)
        self.markers = []
        self.__x = x
        self.__y = y
        for i in range(4):
            index = self.c.insertx1marker(x[i], y[i], noline=True)
            marker = self.c.markersdict[index]['marker']
            if i in [1,2]:
                self.c.setmarkerfollowmouse(index, 1)
            if QWTVERSION4:
                self.c.setMarkerPen(marker,qt.QPen(qt.Qt.green, 2, qt.Qt.DashDotLine))
                self.c.setMarkerSymbol(marker,qwt.QwtSymbol(qwt.QwtSymbol.Diamond, 
                                           qt.QBrush(qt.Qt.blue),
                                           qt.QPen(qt.Qt.red), qt.QSize(15,15)))

            else:
                marker.setLinePen(qt.QPen(qt.Qt.green, 2, qt.Qt.DashDotLine))
                marker.setSymbol(qwt.QwtSymbol(qwt.QwtSymbol.Diamond, 
                                           qt.QBrush(qt.Qt.blue),
                                           qt.QPen(qt.Qt.red), qt.QSize(15,15)))
            self.markers.append(index)
            
        #self.c.enablemarkermode()
        self.c.setMinimumSize(qt.QSize(250,200))
        vlayout.addWidget(self.c)

        if QTVERSION < '4.0.0':
            self.connect (self.c , qt.PYSIGNAL("QtBlissGraphSignal"),
                          self.chval)
            self.connect (self.c , qt.PYSIGNAL("QtBlissGraphSignal"),
                          self.chmap)
            if slider:
                self.connect (self.slider,
                          qt.PYSIGNAL("doubleSliderValueChanged"),
                          self._sliderChanged)
        else:
            self.connect (self.c , qt.SIGNAL("QtBlissGraphSignal"),
                          self.chval)
            self.connect (self.c , qt.SIGNAL("QtBlissGraphSignal"),
                          self.chmap)

            if slider:
                self.connect (self.slider,
                          qt.SIGNAL("doubleSliderValueChanged"),
                          self._sliderChanged)

        # colormap window can not be resized
        self.setFixedSize(vlayout.minimumSize())

    def _sliderChanged(self, ddict):
        if not self.__sliderConnected: return
        delta = (self.dataMax - self.dataMin) * 0.01
        xmin = self.dataMin + delta * ddict['min']
        xmax = self.dataMin + delta * ddict['max']
        self.setDisplayedMinValue(xmin)
        self.setDisplayedMaxValue(xmax)
        self.__x[1] = xmin
        self.__x[2] = xmax
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)
        for i in range(4):
            self.c.setMarkerXPos(self.c.markersdict[self.markers[i]]['marker'],
                                 self.__x[i])
            self.c.setMarkerYPos(self.c.markersdict[self.markers[i]]['marker'],
                                 self.__y[i])

        self.c.replot()
        if DEBUG:
            print("Slider asking to update colormap")
        #self._update()
        self.sendColormap()

    def _update(self):
        if DEBUG:
            print("colormap _update called")
        self.marge = (abs(self.dataMax) + abs(self.dataMin)) / 6.0
        self.minmd = self.dataMin - self.marge
        self.maxpd = self.dataMax + self.marge
        self.c.setx1axislimits(self.minmd, self.maxpd)
        self.c.sety1axislimits( -11.5, 11.5)

        self.__x = [self.minmd, self.dataMin, self.dataMax, self.maxpd]
        self.__y = [-10, -10, 10, 10]
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)
        for i in range(4):
            self.c.setMarkerXPos(self.c.markersdict[self.markers[i]]['marker'],
                                 self.__x[i])
            self.c.setMarkerYPos(self.c.markersdict[self.markers[i]]['marker'],
                                 self.__y[i])
        self.c.replot()
        self.sendColormap()

    def buttonGroupChange(self, val):
        self.colormapType = val
        if DEBUG:
            print("buttonGroup asking to update colormap")
        self._update()

    def chval(self, ddict):
        if ddict['event'] == 'markerMoving':
            if ddict['marker'] in self.markers:
                markerIndex = self.markers.index(ddict['marker'])
            else:
                print("Unknown marker")
                return
        else:
            return
        diam = markerIndex + 1
        x = ddict['x']
        if diam == 2:
            self.setDisplayedMinValue(x)
        if diam == 3:
            self.setDisplayedMaxValue(x)

    def chmap(self, ddict):
        if ddict['event'] == 'markerMoved':
            if ddict['marker'] in self.markers:
                markerIndex = self.markers.index(ddict['marker'])
            else:
                print("Unknown marker")
                return
        else:
            return
        diam = markerIndex + 1

        x = ddict['x']
        if diam == 2:
            self.setMinValue(x)
        if diam == 3:
            self.setMaxValue(x)
        
    """
    Colormap
    """
    def setColormap(self, colormap):
        self.colormapIndex = colormap
        if QTVERSION < '4.0.0':
            self.combo.setCurrentItem(colormap)
        else:
            self.combo.setCurrentIndex(colormap)
    
    def colormapChange(self, colormap):
        self.colormapIndex = colormap
        self.sendColormap()

    # AUTOSCALE
    """
    Autoscale
    """
    def autoscaleChange(self, val):
        self.autoscale = val
        self.setAutoscale(val)        
        self.sendColormap()

    def setAutoscale(self, val):
        if DEBUG:
            print("setAutoscale called", val)
        if val:
            if QTVERSION < '4.0.0':
                self.autoScaleButton.setOn(True)
                self.autoScale90Button.setOn(False)
            else:
                self.autoScaleButton.setChecked(True)
                self.autoScale90Button.setChecked(False)
                #self.autoScale90Button.setDown(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax)
            if self.slider is not None:
                self.__sliderConnected = False
                self.slider.setMinMax(0, 100)
                self.slider.setEnabled(False)
                self.__sliderConnected = True

            self.maxText.setEnabled(0)
            self.minText.setEnabled(0)
            self.c.setEnabled(0)
            self.c.disablemarkermode()
        else:
            if QTVERSION < '4.0.0':
                self.autoScaleButton.setOn(False)
                self.autoScale90Button.setOn(False)
            else:
                self.autoScaleButton.setChecked(False)
                self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            if self.slider:self.slider.setEnabled(True)
            self.c.setEnabled(1)
            self.c.enablemarkermode()

    """
    set rangeValues to dataMin ; dataMax-10%
    """
    def autoscale90Change(self, val):
        self.autoscale90 = val
        self.setAutoscale90(val)
        self.sendColormap()

    def setAutoscale90(self, val):
        if val:
            if QTVERSION < '4.0.0':
                self.autoScaleButton.setOn(False)
            else:
                self.autoScaleButton.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax - abs(self.dataMax/10))
            if self.slider is not None:
                self.__sliderConnected = False
                self.slider.setMinMax(0, 90)
                self.slider.setEnabled(0)
                self.__sliderConnected = True

            self.minText.setEnabled(0)
            self.maxText.setEnabled(0)
            self.c.setEnabled(0)
            self.c.disablemarkermode()
        else:
            if QTVERSION < '4.0.0':
                self.autoScale90Button.setOn(False)
            else:
                self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            if self.slider:self.slider.setEnabled(True)
            self.c.setEnabled(1)
            self.c.enablemarkermode()
            self.c.setFocus()



    # MINIMUM
    """
    change min value and update colormap
    """
    def setMinValue(self, val):
        v = float(str(val))
        self.minValue = v
        self.minText.setText("%g"%v)
        self.__x[1] = v
        self.c.setMarkerXPos(self.c.markersdict[self.markers[1]]['marker'], v)
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)
        self.sendColormap()

    """
    min value changed by text
    """
    def minTextChanged(self):
        text = str(self.minText.text())
        if not len(text):return
        val = float(text)
        self.setMinValue(val)
        if self.minText.hasFocus():
            self.c.setFocus()
        
    """
    change only the displayed min value
    """
    def setDisplayedMinValue(self, val):
        self.minValue = val
        self.minText.setText("%g"%val)
        self.__x[1] = val
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)

    # MAXIMUM
    """
    change max value and update colormap
    """
    def setMaxValue(self, val):
        v = float(str(val))
        self.maxValue = v
        self.maxText.setText("%g"%v)
        self.__x[2] = v
        self.c.setMarkerXPos(self.c.markersdict[self.markers[2]]['marker'], v)
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)
        self.sendColormap()

    """
    max value changed by text
    """
    def maxTextChanged(self):
        text = str(self.maxText.text())
        if not len(text):return
        val = float(text)
        self.setMaxValue(val)
        if self.maxText.hasFocus():
            self.c.setFocus()
            
    """
    change only the displayed max value
    """
    def setDisplayedMaxValue(self, val):
        self.maxValue = val
        self.maxText.setText("%g"%val)
        self.__x[2] = val
        self.c.newCurve("ConstrainedCurve", self.__x, self.__y)

    # DATA values
    """
    set min/max value of data source
    """
    def setDataMinMax(self, minVal, maxVal, update=True):
        if minVal is not None:
            vmin = float(str(minVal))
            self.dataMin = vmin
        if maxVal is not None:
            vmax = float(str(maxVal))
            self.dataMax = vmax

        if update:
            # are current values in the good range ?
            self._update()

    """
    send 'ColormapChanged' signal
    """
    def sendColormap(self):
        if DEBUG:
            print("sending colormap")
        #prevent unexpected behaviour because of bad limits
        if self.minValue > self.maxValue:
            vmax = self.minValue
        else:
            vmax = self.maxValue
        try:
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL("ColormapChanged"),
                        (self.colormapIndex, self.autoscale,
                         self.minValue, vmax,
                         self.dataMin, self.dataMax,
                         self.colormapType))
            else:
                self.emit(qt.SIGNAL("ColormapChanged"),
                        self.colormapIndex, self.autoscale,
                        self.minValue, vmax,
                        self.dataMin, self.dataMax,
                        self.colormapType)
            
        except:
            sys.excepthook(sys.exc_info()[0],
                           sys.exc_info()[1],
                           sys.exc_info()[2])

def test():
    app = qt.QApplication(sys.argv)
    app.connect(app,qt.SIGNAL("lastWindowClosed()"), app.quit)
    demo = ColormapDialog()

    demo.setAutoscale(1)
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
    demo.show()
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()


if __name__ == "__main__":
    test()
