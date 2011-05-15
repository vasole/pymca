import sys
import Object3DQt as qt
import Scene
import SceneWidget
import SceneCoordinates
import Object3DMovement
import Object3DConfig
from HorizontalSpacer import HorizontalSpacer
from VerticalSpacer import VerticalSpacer
DEBUG = 0

class SceneControl(qt.QWidget):
    def __init__(self, parent = None, scene = None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Scene Control Widget')
        if scene is None:
            self.scene = Scene.Scene(name='Scene')
        else:
            self.scene = scene

        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.sceneWidget = SceneWidget.SceneWidget(self, scene=self.scene)
        self.selectedObjectControl = Object3DConfig.Object3DConfig(self)
        self.mainTab = self.selectedObjectControl.mainTab
        self.tabScene = qt.QWidget(self.mainTab)
        self.tabScene.mainLayout = qt.QGridLayout(self.tabScene) 
        self.tabScene.mainLayout.setMargin(0)
        self.tabScene.mainLayout.setSpacing(0)
        self.coordinatesWidget = SceneCoordinates.SceneCoordinates(self.tabScene)
        self.coordinatesWidget.observerWidget.hide()
        #self.movementsWidget  = Object3DMovement.Object3DMovement(self.tabScene)
        #self.movementsWidget.anchorWidget.hide()
        self.tabScene.mainLayout.addWidget(self.coordinatesWidget, 0, 0)
        #self.tabScene.mainLayout.addWidget(self.movementsWidget, 1, 0)
        vspacer = VerticalSpacer(self.tabScene) 
        self.tabScene.mainLayout.addWidget(vspacer, 2, 0)
        self.mainTab.addTab(self.tabScene, "SCENE")
        self.mainLayout.addWidget(self.sceneWidget)
        self.mainLayout.addWidget(self.selectedObjectControl)
        configDict = self.scene[self.scene.name()].root[0].getConfiguration()
        self.selectedObjectControl.setConfiguration(configDict)

        self.connect(self.sceneWidget,
                     qt.SIGNAL('SceneWidgetSignal'),
                     self._sceneWidgetSignal)

        self.connect(self.coordinatesWidget,
                     qt.SIGNAL('SceneCoordinatesSignal'),
                     self._sceneCoordinatesSlot)
        return
        self.connect(self.movementsWidget,
                     qt.SIGNAL('Object3DMovementSignal'),
                     self._movementsSlot)


    def _sceneWidgetSignal(self, ddict):
        self.emitSignal(ddict['event'], ddict)

    def _movementsSlot(self, ddict):
        self.emitSignal(ddict['event'], ddict)

    def _sceneCoordinatesSlot(self, ddict):
        objectList = self.scene.tree.getList()
        event = ddict['event']
        xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()
        self.scene.setAutoScale(ddict['autoscale'])
        if ddict['autoscale']:
            ddict['xmin'] = xmin
            ddict['ymin'] = ymin
            ddict['zmin'] = zmin
            ddict['xmax'] = xmax
            ddict['ymax'] = ymax
            ddict['zmax'] = zmax
            self.coordinatesWidget.setParameters(ddict)
        else:
            xmin = ddict['xmin']
            ymin = ddict['ymin']
            zmin = ddict['zmin']
            xmax = ddict['xmax']
            ymax = ddict['ymax']
            zmax = ddict['zmax']
        self.scene.setLimits([xmin, ymin, zmin, xmax, ymax, zmax])
        if ddict['autolocate']:
            diagonal2 = pow((xmax - xmin), 2) + pow((ymax - ymin), 2) + pow((zmax - zmin), 2)
            position = [xmin + diagonal2,
                        ymin + diagonal2,
                        zmin + diagonal2]
            ddict['observer'] = position
            self.coordinatesWidget.setParameters(ddict)                        
        self.emitSignal(ddict['event'], ddict)

    def emitSignal(self, event=None, ddict = None):
        if DEBUG:
            print "SceneControl emit signal ", ddict
        if ddict is None:
            ddict = {}
        if event is not None:
            ddict['event'] = event
        qt.QObject.emit(self,
                        qt.SIGNAL('SceneControlSignal'),
                        ddict)

    def updateView(self):
        self.sceneWidget.updateView()
        xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()
        autoscale = self.scene.getAutoScale()
        ddict = {}
        ddict['autoscale'] = autoscale
        ddict['xmin'] = xmin
        ddict['ymin'] = ymin
        ddict['zmin'] = zmin
        ddict['xmax'] = xmax
        ddict['ymax'] = ymax
        ddict['zmax'] = zmax
        self.coordinatesWidget.setParameters(ddict)

if __name__ == "__main__":
    import Object3DBase
    app = qt.QApplication([])
    w = SceneControl()
    def slot(ddict):
        print " ddict = ", ddict
        objectList = w.scene.tree.getList()
        selected = []
        for item in objectList:
            if hasattr(item, 'name'):
                if hasattr(item, 'selected'):
                    if item.selected():
                        selected.append(item.name())
        print selected
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    qt.QObject.connect(w, qt.SIGNAL("SceneControlSignal"),
                       slot)
    o0 = Object3DBase.Object3D("DummyObject0")
    o0.setLimits(-100, -200, -300, 100, 200, 300)
    o1 = Object3DBase.Object3D("DummyObject1")
    o01 = Object3DBase.Object3D("DummyObject01")
    w.scene.addObject(o0)
    w.scene.addObject(o1)
    w.scene.addObject(o01)
    w.show()
    w.updateView()
    app.exec_()
