import GLToolBar
import SceneControl
qt = SceneControl.qt
import SceneGLWindow
import weakref
from VerticalSpacer import VerticalSpacer
from HorizontalSpacer import HorizontalSpacer

DEBUG = 0

class ToolBar(GLToolBar.GLToolBar):
    def __init__(self, parent, glwindow):
        GLToolBar.GLToolBar.__init__(self, parent)
        self.glWindow = glwindow

    def applyCube(self, cubeFace):
        position = self.glWindow.scene.applyCube(cubeFace)
        self.glWindow.glWidget.setCurrentViewPosition(position)

class SceneManager(qt.QWidget):
    def __init__(self, parent=None, glwindow=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Scene Manager')
        if glwindow is None:
            self.glWindow = SceneGLWindow.SceneGLWindow(manager=self)
            self.glWindow.setWindowTitle('Scene')
            self.glWindow.show()
        else:
            self.glWindow = weakref.proxy(glwindow)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        #self.toolBar = ToolBar(self, self.glWindow)
        #self.toolBar.layout().addWidget(HorizontalSpacer(self.toolBar))
        self.menuBar = qt.QMenuBar(self)
        self.addFileMenu()
        self.sceneControl = SceneControl.SceneControl(self, self.glWindow.scene)
        if glwindow is None:
            #connect the window to the control
            self.glWindow.sceneControl = self.sceneControl
            self.glWindow.connectSceneControl()
        #self.mainLayout.addWidget(self.toolBar)
        self.mainLayout.addWidget(self.menuBar)
        self.mainLayout.addWidget(self.sceneControl)
        self.mainLayout.addWidget(VerticalSpacer(self))

    def addFileMenu(self):
        self.fileMenu = qt.QMenu("File", self.menuBar)
        self.fileMenu.addAction(qt.QString('Add Object'), self.addObjectSignal)
        self.menuBar.addMenu(self.fileMenu)

    def addObjectSignal(self):
        ddict={}
        ddict['event'] = 'addObject'
        ddict['object'] = None
        ddict['legend'] = None
        self.emit(qt.SIGNAL('SceneManagerSignal'), ddict)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = SceneManager()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    app.exec_()
