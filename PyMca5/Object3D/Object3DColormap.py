import sys
import Object3DQt   as qt
import PyQt4.Qwt5 as Qwt5

DEBUG = 0

class Object3DColormap(qt.QGroupBox):
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Colormap')
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

        self.__disconnected = False
        self.build()
        self._update()

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        # the ComboBox
        self.comboBox = qt.QComboBox(self)
        for colormap in self.colormapList:
            self.comboBox.addItem(colormap)

        self.mainLayout.addWidget(self.comboBox, 0, 0)
        self.connect(self.comboBox,
                     qt.SIGNAL("activated(int)"),
                     self.colormapChanged)

        # autoscale
        self.autoScaleButton = qt.QPushButton("Autoscale", self)
        self.autoScaleButton.setCheckable(True)
        self.autoScaleButton.setAutoDefault(False)    
        self.connect(self.autoScaleButton,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscaleChanged)
        self.mainLayout.addWidget(self.autoScaleButton, 0, 1)

        # autoscale 90%
        self.autoScale90Button = qt.QPushButton("Autoscale 90%", self)
        self.autoScale90Button.setCheckable(True)
        self.autoScale90Button.setAutoDefault(False)    
                
        self.connect(self.autoScale90Button,
                     qt.SIGNAL("toggled(bool)"),
                     self.autoscale90Changed)
        
        self.mainLayout.addWidget(self.autoScale90Button, 0, 2)

        #the checkboxes
        self.buttonGroup = qt.QButtonGroup()
        g1 = qt.QCheckBox(self)
        g1.setText("Linear")
        g2 = qt.QCheckBox(self)
        g2.setText("Logarithmic")
        g3 = qt.QCheckBox(self)
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
        self.mainLayout.addWidget(g1, 1, 0)
        self.mainLayout.addWidget(g2, 1, 1)
        self.mainLayout.addWidget(g3, 1, 2)
        self.connect(self.buttonGroup,
                         qt.SIGNAL("buttonClicked(int)"),
                         self.buttonGroupChanged)

        # The max line
        label = qt.QLabel(self)
        label.setText('Maximum')
        self.maxText = qt.QLineEdit(self)
        self.maxText.setText("%f" % self.maxValue)
        self.maxText.setAlignment(qt.Qt.AlignRight)
        self.maxText.setFixedWidth(self.maxText.fontMetrics().width('######.#####'))
        v = qt.QDoubleValidator(self.maxText)
        self.maxText.setValidator(v)
        self.mainLayout.addWidget(label, 0, 3)
        self.mainLayout.addWidget(self.maxText, 0, 4)
        self.connect(self.maxText,
                     qt.SIGNAL("editingFinished()"),
                     self.textChanged)
        # The min line
        label = qt.QLabel(self)
        label.setText('Minimum')
        self.minText = qt.QLineEdit(self)
        self.minText.setFixedWidth(self.minText.fontMetrics().width('######.#####'))
        self.minText.setAlignment(qt.Qt.AlignRight)
        self.minText.setText("%f" % self.minValue)
        v = qt.QDoubleValidator(self.minText)
        self.minText.setValidator(v)
        self.mainLayout.addWidget(label, 1, 3)
        self.mainLayout.addWidget(self.minText, 1, 4)
        self.connect(self.minText,
                     qt.SIGNAL("editingFinished()"),
                     self.textChanged)

        # The sliders
        self.dataMin   = -10
        self.dataMax   = 10
        self.minValue  = 0
        self.maxValue  = 1
        self.sliderList = []
        delta =  (self.dataMax-self.dataMin) / 200.
        for i in [0, 1]:
            slider = Qwt5.QwtSlider(self, qt.Qt.Horizontal)
            slider.setRange(self.dataMin, self.dataMax, delta)
            if i == 0:
                slider.setValue(self.maxValue)
            else:
                slider.setValue(self.minValue)
            self.mainLayout.addWidget(slider, i, 5)
            self.connect(slider,
                         qt.SIGNAL("valueChanged(double)"),
                         self.sliderChanged)
            self.sliderList.append(slider)

    def colormapChanged(self, value):
        self.colormapIndex = value
        self._emitSignal()

    def autoscaleChanged(self, value):
        self.autoscale = value
        self.__disconnected = True
        self.setAutoscale(value)
        self.__disconnected = False
        self._emitSignal()

    def autoscale90Changed(self, value):
        self.autoscale90 = value
        self.__disconnected = True
        self.setAutoscale90(value)
        self.__disconnected = False
        self._emitSignal()

    def buttonGroupChanged(self, value):
        self.colormapType = value
        self._emitSignal()

    def textChanged(self):
        valueMax = float(str(self.maxText.text()))
        valueMin = float(str(self.minText.text()))
        a = min(valueMax, valueMin)
        b = max(valueMax, valueMin)
        self.maxText.setText("%f" % b)
        self.minText.setText("%f" % a)
        oldMin = min(self.sliderList[0].value(),
                     self.sliderList[1].value())
        oldMax = max(self.sliderList[0].value(),
                     self.sliderList[1].value())
        emit = False
        self.__disconnected = True
        if (oldMax - valueMax) != 0.0:
            emit = True
            self.sliderList[0].setValue(valueMax)
        if (oldMin - valueMin) != 0.0:
            emit = True
            self.sliderList[1].setValue(valueMax)
        self.minValue = valueMin
        self.maxValue = valueMax
        self.__disconnected = False
        if emit: self._emitSignal()

    def sliderChanged(self, value):
        if self.__disconnected:return
        if DEBUG:print "sliderChanged"
        value0 = self.sliderList[0].value()
        value1 = self.sliderList[1].value()
        self.maxText.setText("%f" % max(value0, value1))
        self.minText.setText("%f" % min(value0, value1))
        self.minValue = min(value0, value1)
        self.maxValue = max(value0, value1)

        self._emitSignal()

    def _update(self):
        if DEBUG: print "colormap _update called"
        self.__disconnected = True
        delta = (self.dataMax - self.dataMin)/ 200.
        self.sliderList[0].setRange(self.dataMin, self.dataMax, delta)
        self.sliderList[0].setValue(self.maxValue)
        self.sliderList[1].setRange(self.dataMin, self.dataMax, delta)
        self.sliderList[1].setValue(self.minValue)
        self.maxText.setText("%g" % self.maxValue)
        self.minText.setText("%g" % self.minValue)
        self.comboBox.setCurrentIndex(self.colormapIndex)
        self.buttonGroup.button(self.colormapType).setChecked(True)
        self.setAutoscale(self.autoscale)
        self.setAutoscale90(self.autoscale90)
        self.__disconnected = False

    def _emitSignal(self, event = None):
        if self.__disconnected:return
        if event is None:event = 'ColormapChanged'
        if DEBUG:print "sending colormap"
        ddict = self.getParameters()
        ddict['event'] = event
        self.emit(qt.SIGNAL("Object3DColormapSignal"),ddict)

    def setAutoscale(self, val):
        if DEBUG:
            print "setAutoscale called", val
        if val:
            self.autoScaleButton.setChecked(True)
            self.autoScale90Button.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax)
            for slider in self.sliderList:
                self.__disconnected = True
                delta = (self.dataMax - self.dataMin)/200.
                slider.setRange(self.dataMin, self.dataMax, delta)
                slider.setEnabled(False)
                self.__disconnected = False

            self.maxText.setEnabled(0)
            self.minText.setEnabled(0)
        else:
            self.autoScaleButton.setChecked(False)
            self.autoScale90Button.setChecked(False)
            self.minText.setEnabled(1)
            self.maxText.setEnabled(1)
            for slider in self.sliderList:
                slider.setEnabled(True)

    def setAutoscale90(self, val):
        if val:
            self.autoScaleButton.setChecked(False)
            self.setMinValue(self.dataMin)
            self.setMaxValue(self.dataMax - abs(self.dataMax/10.))
            for slider in self.sliderList:
                self.__disconnected = True
                delta = (self.dataMax - self.dataMin)/200.
                slider.setRange(self.dataMin, self.dataMax, delta)
                slider.setEnabled(False)
                self.__disconnected = False
            self.minText.setEnabled(0)
            self.maxText.setEnabled(0)
        else:
            self.autoScale90Button.setChecked(False)
            if not self.autoscale:
                self.minText.setEnabled(1)
                self.maxText.setEnabled(1)
                for slider in self.sliderList:
                    self.__disconnected = True
                    delta = (self.dataMax - self.dataMin)/200.
                    slider.setRange(self.dataMin, self.dataMax, delta)
                    slider.setEnabled(True)
                    self.__disconnected = False


    # MINIMUM
    """
    change min value and update colormap
    """
    def setMinValue(self, val):
        v = float(str(val))
        self.minValue = v
        self.minText.setText("%g"%v)
        self.__disconnected = True
        self.sliderList[1].setValue(v)
        self.__disconnected = False

    # MAXIMUM
    """
    change max value and update colormap
    """
    def setMaxValue(self, val):
        v = float(str(val))
        self.maxValue = v
        self.maxText.setText("%g"%v)
        self.__disconnected = True
        self.sliderList[0].setValue(v)
        self.__disconnected = False

    def getParameters(self):
        ddict = {}
        if self.minValue > self.maxValue:
            vMax = self.minValue
        else:
            vMax = self.maxValue
        ddict['colormap'] = [self.colormapIndex, self.autoscale,
                        self.minValue, vMax,
                        self.dataMin, self.dataMax,
                        self.colormapType]
        return ddict

    def setParameters(self, ddict):
        if ddict.has_key('colormap'):
            self.__disconnected = True
            self.colormapIndex,  self.autoscale, \
                 self.minValue, self.maxValue, \
                 self.dataMin, self.dataMax, \
                        self.colormapType = ddict['colormap']
            self._update()
            self.__disconnected = False
            

def test():
    app = qt.QApplication(sys.argv)
    app.connect(app,qt.SIGNAL("lastWindowClosed()"), app.quit)

    def slot(ddict):
        print "ddict= ", ddict
    demo = Object3DColormap()
    qt.QObject.connect(demo, qt.SIGNAL("Object3DColormapSignal"),slot)
    demo.show()
    app.exec_()


if __name__ == "__main__":
    test()
