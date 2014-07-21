#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import os
import weakref
from . import GLToolBar
from . import SceneControl
qt = SceneControl.qt
if hasattr(qt, "QString"):
    qtQString = qt.QString
else:
    qtQString = str
from .VerticalSpacer import VerticalSpacer
from .HorizontalSpacer import HorizontalSpacer
CONFIGDICT = True
try:
    from PyMca5.PyMcaIO import ConfigDict
except ImportError:
    try:
        from . import ConfigDict
    except ImportError:
        CONFIGDICT = False

try:
    from PyMca5 import PyMcaDirs as Object3DDirs
except ImportError:
    try:
        import PyMcaDirs as Object3DDirs
    except ImportError:
        from . import Object3DDirs


DEBUG = 0

class ToolBar(GLToolBar.GLToolBar):
    def __init__(self, parent, glwindow):
        GLToolBar.GLToolBar.__init__(self, parent)
        self.glWindow = glwindow

    def applyCube(self, cubeFace):
        position = self.glWindow.scene.applyCube(cubeFace)
        self.glWindow.glWidget.setCurrentViewPosition(position)

class SceneManager(qt.QWidget):
    sigSceneManagerSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, glwindow=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Scene Manager')
        if glwindow is None:
            from . import SceneGLWindow
            self.glWindow = SceneGLWindow.SceneGLWindow(manager=self)
            self.glWindow.setWindowTitle('Scene')
            self.glWindow.show()
        else:
            self.glWindow = weakref.proxy(glwindow)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
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
        self.fileMenu.addAction(qtQString('Add Object'), self.addObjectSignal)
        if CONFIGDICT:
            self.fileMenu.addSeparator()
            self.fileMenu.addAction(qtQString('Load Configuration'),
                                    self.loadConfiguration)
            self.fileMenu.addAction(qtQString('Save Configuration'),
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
        self.sigSceneManagerSignal.emit(ddict)

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
        self.sigSceneManagerSignal.emit(ddict)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = SceneManager()
    app.lastWindowClosed.connect(app.quit)
    w.show()
    app.exec_()
