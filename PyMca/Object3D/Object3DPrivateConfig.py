import Object3DQt as qt
from HorizontalSpacer import HorizontalSpacer
from VerticalSpacer import VerticalSpacer
import weakref
DEBUG = 0

class Object3DPrivateConfig(qt.QWidget):
    def __init__(self, parent = None, name=""):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("%s private configuration" % name)
        self._configuration = {}
        self._configuration['widget'] = weakref.proxy(self)
        self.callBack = None
        self._name = name
        self.build()

    def __del__(self):
        if DEBUG:
            print "Object3DPrivateConfig %s deleted" % self._name

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)
        self.button = qt.QPushButton(self)
        self.button.setText("Test Signal")
        self.mainLayout.addWidget(VerticalSpacer(self), 0, 0, 1, 3)
        self.mainLayout.addWidget(HorizontalSpacer(self), 1, 0)
        self.mainLayout.addWidget(self.button, 1, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self), 1, 2)
        self.mainLayout.addWidget(VerticalSpacer(self), 2, 0, 1, 3)
        self.connect(self.button, qt.SIGNAL('clicked()'), self.updateCallBack)

    def setParameters(self, ddict):
        for key in ddict.keys():
            if self._configuration.has_key(key):
                if key != 'widget':
                    self._configuration[key].update(ddict[key])
        self._updateWidget()
        return True

    def _updateWidget(self):
        return
    
    def getParameters(self):
        return self._configuration

    def setCallBack(self, f):
        self.callBack = f

    def updateCallBack(self):
        if self.callBack is not None:
            self.callBack()
 
if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot():
        print "Callback called"
    w = Object3DPrivateConfig()
    w.setCallBack(myslot)
    w.show()    
    app.exec_()
