import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        from PyQt4 import Qwt5 as qwt
    except:
        import qt
        try:
            import Qwt4 as qwt 
        except:
            import qwt
else:
    import qt
    try:
        import Qwt4 as qwt 
    except:
        import qwt

QTVERSION = qt.qVersion()
DEBUG = 0
    
if QTVERSION < '4.0.0':
    if qwt.QWT_VERSION_STR[0] > '4':
        raise "ValueError","Unsupported combination %s and %s" % (QTVERSION, 
                                                                qwt.QWT_VERSION_STR)

if qwt.QWT_VERSION_STR[0] > '4':
    Qwt = qwt
    qwt.QwtPlotMappedItem = Qwt.QwtPlotItem
    qwt.QwtCurve          = Qwt.QwtPlotCurve


class DoubleSlider(qt.QWidget):
    def __init__(self, parent = None, scale = False):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        orientation = qt.Qt.Horizontal
        
        self.minSliderContainer = MySlider(self, orientation)
        self.minSlider = self.minSliderContainer.slider
        if scale:
            self.minSlider.setScalePosition(qwt.QwtSlider.BottomScale)
        self.minSlider.setRange(0.0, 100.0, 0.01)
        self.minSlider.setValue(0.0)
        self.maxSliderContainer = MySlider(self, orientation)
        self.maxSlider = self.maxSliderContainer.slider
        self.maxSlider.setRange(0.0, 100.0, 0.01)
        self.maxSlider.setValue(100.)
        self.mainLayout.addWidget(self.maxSliderContainer)
        self.mainLayout.addWidget(self.minSliderContainer)
        self.connect(self.minSlider,
                     qt.SIGNAL("valueChanged(double)"),
                     self._sliderChanged)
        self.connect(self.maxSlider,
                     qt.SIGNAL("valueChanged(double)"),
                     self._sliderChanged)

    def __getDict(self):
        ddict = {}
        ddict['event'] = "doubleSliderValueChanged"
        m   = self.minSlider.value()
        M   = self.maxSlider.value()
        if m > M:
            ddict['max'] = m
            ddict['min'] = M
        else:
            ddict['min'] = m
            ddict['max'] = M
        return ddict

    def _sliderChanged(self, value):
        if DEBUG: print "DoubleSlider._sliderChanged()"
        ddict = self.__getDict()
        self.emit(qt.SIGNAL("doubleSliderValueChanged"),ddict)

    def setMinMax(self, m, M):
        self.minSlider.setValue(m)
        self.maxSlider.setValue(M)

    def getMinMax(self):
        m = self.minSlider.value()
        M = self.maxSlider.value()
        if m > M:
            return M, m
        else:
            return m, M
        

class MySlider(qt.QWidget):
    def __init__(self, parent = None, orientation=qt.Qt.Horizontal):
        qt.QWidget.__init__(self, parent)
        if orientation == qt.Qt.Horizontal:
            alignment = qt.Qt.AlignHCenter | qt.Qt.AlignTop
            layout = qt.QHBoxLayout(self)
        else:
            alignment = qt.Qt.AlignVCenter | qt.Qt.AlignLeft
            layout = qt.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        self.slider = qwt.QwtSlider(self, orientation)
        self.label  = qt.QLabel("0", self)
        self.label.setAlignment(alignment)
        self.label.setFixedWidth(self.label.fontMetrics().width('100.99'))

        
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.connect(self.slider,
                     qt.SIGNAL('valueChanged(double)'),
                     self.setNum)

    def setNum(self, value):
        self.label.setText('%s' % value)
        

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit'))
    
    w = DoubleSlider()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
