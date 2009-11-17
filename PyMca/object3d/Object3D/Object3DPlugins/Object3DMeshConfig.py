from __future__ import absolute_import
try:
    import PyMca.Object3D
    __package__ = "PyMca.Object3D.Object3DPlugins"
except ImportError:
    import Object3D
    __package__ = "Object3D.Object3DPlugins"
from .. import Object3DQt as qt
from .. import Object3DPrivateConfig
from .. import PrivateConfigTools
from ..HorizontalSpacer import HorizontalSpacer
from ..VerticalSpacer import VerticalSpacer
import weakref
DEBUG = 0

class Object3DMeshConfig(Object3DPrivateConfig.Object3DPrivateConfig):
    def __init__(self, parent = None, name=""):
        Object3DPrivateConfig.Object3DPrivateConfig.__init__(self, parent, name)
        self._configuration = {}
        self._configuration['widget'] = weakref.proxy(self)

    def build(self):
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setMargin(4)
        self.mainLayout.setSpacing(4)

        #info
        self.infoLabel = PrivateConfigTools.InfoLabel(self)

        #color filtering
        self.colorFilter = PrivateConfigTools.ColorFilter(self)
        self.connect(self.colorFilter,
                     qt.SIGNAL('ColorFilterSignal'),
                     self.updateCallBack)

        #value filtering
        self.valueFilter = PrivateConfigTools.ValueFilter(self)
        
        #isosurfaces
        self.isosurfaces = PrivateConfigTools.Isosurfaces(self)


        #actions
        self.updateButton = qt.QPushButton(self)
        self.updateButton.setText("Update")
        self.updateButton.setAutoDefault(False)

        #self.mainLayout.addWidget(HorizontalSpacer(self), 0, 0)
        self.mainLayout.addWidget(self.infoLabel,   0, 0, 1, 4)
        #self.mainLayout.addWidget(HorizontalSpacer(self), 0, 2)
        self.mainLayout.addWidget(self.colorFilter, 1, 0)
        self.mainLayout.addWidget(self.valueFilter, 1, 1)
        self.mainLayout.addWidget(self.isosurfaces, 1, 2)
        self.mainLayout.addWidget(HorizontalSpacer(self), 2, 0)
        self.mainLayout.addWidget(self.updateButton, 2, 1)
        self.mainLayout.addWidget(HorizontalSpacer(self), 2, 3)
        self.mainLayout.addWidget(VerticalSpacer(self), 3, 0)

        #connect
        self.connect(self.colorFilter,
                     qt.SIGNAL('ColorFilterSignal'),
                     self.updateCallBack)
        self.connect(self.updateButton,
                     qt.SIGNAL('clicked()'),
                     self.updateCallBack)

    def setParameters(self, ddict):
        #if ddict.has_key('widget'):
        #    del ddict['widget']
        self._configuration.update(ddict)
        self._updateWidget()
        return True

    def _updateWidget(self):
        self.infoLabel.setParameters(self._configuration)
        self.colorFilter.setParameters(self._configuration)
        self.valueFilter.setParameters(self._configuration)
        self.isosurfaces.setParameters(self._configuration)
        return
    
    def getParameters(self):
        self._configuration.update(self.infoLabel.getParameters())
        self._configuration.update(self.colorFilter.getParameters())
        self._configuration.update(self.valueFilter.getParameters())
        self._configuration.update(self.isosurfaces.getParameters())
        return self._configuration

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def myslot():
        print "Callback called"
    w = Object3DMeshConfig()
    w.setCallBack(myslot)
    w.show()    
    app.exec_()
