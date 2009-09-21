import sys
import Object3DQt as qt
import SceneTree

QTVERSION = qt.qVersion()
DEBUG = 0 

class SceneWidget(qt.QWidget):
    def __init__(self, parent = None, scene = None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Scene Widget')
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setMargin(0)
        if scene is None:
            tree = None
        else:
            tree = scene.tree
        self.treeWidget = SceneTree.Object3DObjectTree(self, tree=tree)
        self.tree = self.treeWidget.tree
        self.mainLayout.addWidget(self.treeWidget)

        self.connect(self.treeWidget,
                     qt.SIGNAL('ObjectTreeSignal'),
                     self._treeWidgetSignal)

    def _treeWidgetSignal(self, ddict):
        self.emitSignal(ddict['event'], ddict)

    def emitSignal(self, event, ddict = None):
        if ddict is None:
            ddict = {}
        ddict['event'] = event
        qt.QObject.emit(self,
                        qt.SIGNAL('SceneWidgetSignal'),
                        ddict)

    def updateView(self, expand=True):
        return self.treeWidget.updateView(expand=expand)

    def setSelectedObject(self, name=None):
        return self.treeWidget.setSelectedObject(name)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = SceneWidget()
    w.show()
    app.exec_()
