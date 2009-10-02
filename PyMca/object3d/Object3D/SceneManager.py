import os
import GLToolBar
import SceneControl
qt = SceneControl.qt
import SceneGLWindow
import weakref
from VerticalSpacer import VerticalSpacer
from HorizontalSpacer import HorizontalSpacer
CONFIGDICT = True
try:
    from PyMca import ConfigDict    
except ImportError:
    try:
        import ConfigDict
    except ImportError:
        CONFIGDICT = False

try:
    from PyMca import PyMcaDirs as Object3DDirs
except ImportError:
    try:
        import PyMcaDirs as Object3DDirs
    except ImportError:
        import Object3DDirs

    
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
        if CONFIGDICT:
            self.fileMenu.addSeparator()
            self.fileMenu.addAction(qt.QString('Load Configuration'),
                                    self.loadConfiguration)
            self.fileMenu.addAction(qt.QString('Save Configuration'),
                                    self.saveConfiguration)
        self.menuBar.addMenu(self.fileMenu)

    def loadConfiguration(self):
        wdir = Object3DDirs.inputDir
        message = "Enter input scene configuration file name"
        filename = qt.QFileDialog.getOpenFileName(self,
                    message,
                    wdir,
                    "*.scene")
        filename = str(filename)
        if not len(filename):
            return
        Object3DDirs.inputDir = os.path.dirname(filename)
        d = ConfigDict.ConfigDict()
        d.read(filename)
        self.glWindow.scene.setConfiguration(d)
        #This partially works but it is awful
        current = self.glWindow.scene.getSelectedObject()
        if current is None:
            current = 'Scene'
        self.sceneControl.sceneWidget.setSelectedObject(current)
        self.sceneControl.sceneWidget.treeWidget.emitSignal('objectSelected')
        ddict={}
        ddict['event'] = 'configurationLoaded'
        ddict['object'] = None
        ddict['legend'] = None
        self.emit(qt.SIGNAL('SceneManagerSignal'), ddict)
        
    def saveConfiguration(self):
        wdir = Object3DDirs.outputDir
        message = "Enter output scene configuration file name"
        filename = qt.QFileDialog.getSaveFileName(self,
                    message,
                    wdir,
                    "*.scene")
        filename = str(filename)
        if not len(filename):
            return
        Object3DDirs.outputDir = os.path.dirname(filename)
        config =  self.glWindow.scene.getConfiguration()
        d = ConfigDict.ConfigDict()
        if os.path.exists(filename):
            os.remove(filename)
        d.update(config)
        d.write(filename)

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
