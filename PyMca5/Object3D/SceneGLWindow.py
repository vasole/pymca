#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
import sys
import numpy
import weakref
import glob
import OpenGL.GL  as GL
import OpenGL.GLU as GLU
#This is to simplify freezing
try:
    from PyMca5.PyMcaIO import EDFStack
    from PyMca5 import spslut
except:
    try:
        import EDFStack
        import spslut
    except:
        pass
from . import Object3DCTools
from . import Object3DQhull
# End of freeze options
#This hould be in PyMca, I put it here for the time being
try:
    from .Object3DPlugins import Object3DMesh
    from .Object3DPlugins import Object3DStack
except:
    pass
#End of PyMca specific imports
from . import Object3DConfig
from . import SceneGLWidget
qt=SceneGLWidget.qt
from . import Object3DIcons
#import Object3D
from . import Object3DSlider
from . import SceneControl
from .HorizontalSpacer import HorizontalSpacer
from .VerticalSpacer import VerticalSpacer
from . import SceneManager
from . import GLToolBar
from . import Object3DPrintPreview

from PyMca5.PyMcaGui import PyMcaFileDialogs

if hasattr(qt, "QString"):
    qtQString = qt.QString
else:
    qtQString = qt.safe_str

DEBUG = 0

def getObject3DModules(directory=None):
    if directory is None:
        directory = os.path.dirname(SceneGLWidget.__file__)
        if (os.path.basename(directory) == 'library.zip') or\
            SceneGLWidget.__file__.endswith('.exe'):
                #handle frozen versions
                directory = os.path.dirname(directory)
        if len(directory):
            directory = os.path.join(directory, "Object3DPlugins")
        else:
            directory = os.path.join(directory,".", "Object3DPlugins")
        if not os.path.exists(directory):
            raise IOError("Directory:\n%s\ndoes not exist." % directory)
    if directory not in sys.path:
        sys.path.append(directory)
    fileList = glob.glob(os.path.join(directory, "*.py"))
    moduleList = []
    for module in fileList:
        try:
            m = __import__(os.path.basename(module)[:-3])
            if hasattr(m, 'getObject3DInstance'):
                moduleList.append(m)
        except:
            if DEBUG:
                print("Problem importing module %s" % module)
    if not len(moduleList):
        raise IOError("No plugins found in directory %s"  % directory)
    return moduleList

class WheelAndSlider(qt.QWidget):
        def __init__(self, parent = None, orientation = qt.Qt.Horizontal):
            qt.QWidget.__init__(self, parent)
            if orientation == qt.Qt.Horizontal:
                self.mainLayout = qt.QHBoxLayout(self)
            else:
                orientation = qt.Qt.Vertical
                self.mainLayout = qt.QVBoxLayout(self)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)
            #self.wheel  = Qwt5.QwtWheel(self)
            #self.wheel.setOrientation(orientation)
            self.slider = Object3DSlider.Object3DSlider(self,
                                         orientation)

            if orientation == qt.Qt.Horizontal:
                #self.mainLayout.addWidget(self.wheel)
                self.mainLayout.addWidget(self.slider)
            else:
                self.mainLayout.addWidget(self.slider)
                #self.mainLayout.addWidget(self.wheel)

class WheelAndLineEdit(qt.QWidget):
        def __init__(self, parent = None, orientation=qt.Qt.Horizontal):
            qt.QWidget.__init__(self, parent)
            self.mainLayout = qt.QHBoxLayout(self)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)
            self.wheel  = Object3DSlider.Object3DSlider(self, orientation)
            self.lineEdit = qt.QLineEdit(self)
            self.lineEdit.setText("")
            self.lineEdit.setReadOnly(True)
            self.mainLayout.addWidget(self.wheel)
            self.mainLayout.addWidget(self.lineEdit)

class WheelAndSpacer(qt.QWidget):
        def __init__(self, parent = None, orientation = qt.Qt.Horizontal):
            qt.QWidget.__init__(self, parent)
            if orientation == qt.Qt.Horizontal:
                self.mainLayout = qt.QHBoxLayout(self)
                self.spacer = HorizontalSpacer(self)
            else:
                orientation = qt.Qt.Vertical
                self.mainLayout = qt.QVBoxLayout(self)
                self.spacer = VerticalSpacer(self)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)
            self.wheel  = Object3DSlider.Object3DSlider(self, orientation)
            if orientation == qt.Qt.Horizontal:
                self.mainLayout.addWidget(self.wheel)
                self.mainLayout.addWidget(self.spacer)
            else:
                self.mainLayout.addWidget(self.spacer)
                self.mainLayout.addWidget(self.wheel)

class SceneGLWindow(qt.QWidget):
    def __init__(self, parent=None, manager=None, printpreview=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout   = qt.QGridLayout(self)
        if printpreview is None:
            self.printPreview = Object3DPrintPreview.Object3DPrintPreview(modal = 0)
        else:
            self.printPreview = printpreview

        self.buildToolBar()

        self.wheelSlider10  = WheelAndSpacer(self, orientation = qt.Qt.Vertical)

        # Wheel
        self.__applyingCube = False

        #self.wheelSlider10.wheel.setMass(0.5)
        self.wheelSlider10.wheel.setRange(-360., 360., 0.5)
        self.wheelSlider10.wheel.setValue(0.0)
        #self.wheelSlider10.wheel.setTotalAngle(360.)
        self.wheelSlider10.wheel.valueChanged[float].connect(self.setTheta)


        self.glWidget = SceneGLWidget.SceneGLWidget(self)
        self.scene = weakref.proxy(self.glWidget.scene)
        self.glWidget.setObjectSelectionMode(True)
        self.wheelSlider12 = Object3DSlider.Object3DSlider(self,
                                                 qt.Qt.Vertical)

        #self.axesObject = BlissGLAxesObject.BlissGLAxesObject("3D Axes")
        #self.glWidget.addObject3D(self.axesObject)
        #self.wheelSlider12.setScaleEngine(Qwt5.QwtLog10ScaleEngine())
        #self.wheelSlider12.setThumbWidth(20)
        self.wheelSlider12.setRange(-10, 10, 0.05)
        #self.wheelSlider12.setScale(0.01, 100)
        self.wheelSlider12.setValue(0.0)
        #self.wheelSlider12.setScaleMaxMinor(10)
        self.wheelSlider12.valueChanged.connect(self.setZoomFactor)

        self.wheelSlider21  = WheelAndLineEdit(self, orientation = qt.Qt.Horizontal)
        #wheel
        #self.wheelSlider21.wheel.setMass(0.5)
        self.wheelSlider21.wheel.setRange(-360., 360., 0.5)
        self.wheelSlider21.wheel.setValue(0.0)
        #self.wheelSlider21.wheel.setTotalAngle(360.)
        self.infoLine = self.wheelSlider21.lineEdit
        self.infoLine.setText("Scene is in object selection mode.")
        self.wheelSlider21.wheel.valueChanged.connect(self.setPhi)

        self.mainLayout.addWidget(self.toolBar, 0, 1)
        self.mainLayout.addWidget(self.wheelSlider10, 1, 0)
        self.mainLayout.addWidget(self.glWidget, 1, 1)
        self.mainLayout.addWidget(self.wheelSlider12, 1, 2)
        self.mainLayout.addWidget(self.wheelSlider21, 2, 1)

        if manager is None:
            self.manager = SceneManager.SceneManager(None, glwindow=self)
            self.sceneControl=self.manager.sceneControl
            #self.sceneControl = SceneControl.SceneControl(None, self.glWidget.scene)
            #self.manager.sceneControl
            self.connectSceneControl()
            #self.manager.show()
            #self.sceneControl.show()
            self.manager.sigSceneManagerSignal.connect(self.sceneManagerSlot)
        else:
            self.manager=weakref.proxy(manager)

        self.activeObject = None

        self.glWidget.sigObjectSelected.connect(self.objectSelectedSlot)

        self.glWidget.sigVertexSelected.connect(self.vertexSelectedSlot)

        self.setWindowTitle("Object3D Scene")

    def connectSceneControl(self):
        self.selectedObjectControl = self.sceneControl.selectedObjectControl
        self.sceneControl.updateView()
        self.sceneControl.sigSceneControlSignal.connect(self.sceneControlSlot)
        self.selectedObjectControl.sigObject3DConfigSignal.connect(\
                         self.objectControlSlot)


    def buildToolBar(self):
        self.toolBar = GLToolBar.GLToolBar(self)
        self.toolBar.sigGLToolBarSignal.connect(self.applyCube)
        IconDict = Object3DIcons.IconDict
        self.normalIcon = qt.QIcon(qt.QPixmap(IconDict["cursor_normal"]))
        self.sizeallIcon = qt.QIcon(qt.QPixmap(IconDict["cursor_sizeall"]))
        self.pointingHandIcon = qt.QIcon(qt.QPixmap(IconDict["cursor_pointinghand"]))
        self.whatIcon = qt.QIcon(qt.QPixmap(IconDict["cursor_what"]))

        # the object selection toggle
        text  = "Object selection Mode"
        text += "\nIt may be faster to select objects\n"
        text += "objects through the control window."
        tb = self._addToolButton(self.normalIcon,
                                 self.setObjectSelectionModeSlot,
                                 text,
                                 toggle = True,
                                 state = True)
        self.buttonObjectSelectionMode = tb
        self.objectSelectionMode = False

        # the vertex toggle
        tb = self._addToolButton(self.pointingHandIcon,
                                 self.setVertexSelectionModeSlot,
                                 "Vertex selection Mode",
                                 toggle = True,
                                 state = False)
        self.buttonVertexSelectionMode = tb
        self.vertexSelectionMode = False

        # the panning toggle
        tb = self._addToolButton(self.sizeallIcon,
                                 self.setScenePanningModeSlot,
                                 "Scene panning Mode",
                                 toggle = True,
                                 state = False)
        self.buttonScenePanningMode = tb
        self.scenePanningMode = False

        #give empty space
        spacer = HorizontalSpacer(self.toolBar)
        self.toolBar.layout().addWidget(spacer)


        #the possibility to load an object from a file
        self.controlIcon = qt.QIcon()
        tb = self._addToolButton(self.controlIcon,
                                 self.showSceneControlWindow,
                                 "Show control window")


        #the possibility to load an object from a file
        self.openIcon = qt.QIcon(qt.QPixmap(IconDict["file_open"]))
        tb = self._addToolButton(self.openIcon,
                                 self.addObjectFromFileDialog,
                                 "Add 3DObject from file")


        #the possibility to save a "photo" of the 3D scene
        self.saveIcon = qt.QIcon(qt.QPixmap(IconDict["file_save"]))
        tb = self._addToolButton(self.saveIcon,
                                 self.saveImage,
                                 "Save 3DScene to a file")

        #print the 3D scene
        self.printIcon = qt.QIcon(qt.QPixmap(IconDict["print"]))
        tb = self._addToolButton(self.printIcon,
                                 self.printWidget,
                                 "Print widget")

    def applyCube(self, ddict):
        if 'face' in ddict:
            position = self.scene.applyCube(ddict['face'])
            self.glWidget.setCurrentViewPosition(position)
            self.__applyingCube = True
            self.wheelSlider10.wheel.setValue(0.0)
            self.wheelSlider21.wheel.setValue(0.0)
            self.__applyingCube = False


    def setObjectSelectionModeSlot(self):
        self.vertexSelectionMode = False
        self.objectSelectionMode = True
        self.scenePanningMode    = False
        self.buttonVertexSelectionMode.setChecked(self.vertexSelectionMode)
        self.buttonObjectSelectionMode.setChecked(self.objectSelectionMode)
        self.buttonScenePanningMode.setChecked(self.scenePanningMode)
        self.glWidget.setObjectSelectionMode(self.objectSelectionMode)
        self.glWidget.setVertexSelectionMode(self.vertexSelectionMode)
        self.glWidget.setCursor(qt.QCursor(qt.Qt.ArrowCursor))
        text = "Scene is in object selection mode."
        current = self.scene.getSelectedObject()
        if current not in [None, self.scene.name()]:
            text += " Object %s currently selected." % current
        self.infoLine.setText(text)

    def setVertexSelectionModeSlot(self):
        self.vertexSelectionMode = True
        self.objectSelectionMode = False
        self.scenePanningMode    = False
        self.buttonVertexSelectionMode.setChecked(self.vertexSelectionMode)
        self.buttonObjectSelectionMode.setChecked(self.objectSelectionMode)
        self.buttonScenePanningMode.setChecked(self.scenePanningMode)
        self.glWidget.setObjectSelectionMode(self.objectSelectionMode)
        self.glWidget.setVertexSelectionMode(self.vertexSelectionMode)
        #self.glWidget.setCursor(qt.QCursor(qt.Qt.WhatsThisCursor))
        self.glWidget.setCursor(qt.QCursor(qt.Qt.PointingHandCursor))
        current = self.scene.getSelectedObject()
        o3d = self.scene.getObject3DProxy(current)
        if current in [None, self.scene.name()]:
            text = ("Vertex selection mode is useless without a selected object.")
        elif not o3d.isVertexSelectionModeSupported():
                text = "Object %s does not support vertex selection mode." % current
        else:
            text = ("Vertex selection mode for object %s." % current)
        self.infoLine.setText(text)

    def setScenePanningModeSlot(self):
        if self.scenePanningMode:
           # nothing to be done
           return
        self.vertexSelectionMode = False
        self.objectSelectionMode = False
        self.scenePanningMode    = True
        self.buttonVertexSelectionMode.setChecked(self.vertexSelectionMode)
        self.buttonObjectSelectionMode.setChecked(self.objectSelectionMode)
        self.buttonScenePanningMode.setChecked(self.scenePanningMode)
        self.glWidget.setObjectSelectionMode(self.objectSelectionMode)
        self.glWidget.setVertexSelectionMode(self.vertexSelectionMode)
        self.glWidget.setCursor(qt.QCursor(qt.Qt.SizeAllCursor))
        self.infoLine.setText("Scene is in panning mode.")

    def saveImage(self):
        filelist = PyMcaFileDialogs.getFileList(
            parent=self,
            filetypelist=["Image files (*.png)"],
            message="Please give output file name",
            mode="SAVE",
            getfilter=False)
        if len(filelist):
            self.glWidget.saveImage(filelist[0])

    def printWidget(self):
        #pixmap = qt.QPixmap.grabWidget(self)
        #self.printPreview.addPixmap(pixmap)
        qimage = self.glWidget.getQImage()
        self.printPreview.addImage(qimage)
        if self.printPreview.isHidden():
            self.printPreview.show()
        self.printPreview.raise_()

    def _addToolButton(self, icon, action, tip, toggle=None, state=None, position=None):
        tb      = qt.QToolButton(self.toolBar)
        tb.setIcon(icon)
        tb.setToolTip(tip)
        if toggle is not None:
            if toggle:
                tb.setCheckable(1)
                if state is not None:
                    if state:
                        tb.setChecked(state)
                else:
                    tb.setChecked(False)
        if position is not None:
            self.toolBar.mainLayout.insertWidget(position, tb)
        else:
            self.toolBar.mainLayout.addWidget(tb)
        if action is not None:
            # The action should not accept any parameter because
            # the [()] trick does not any longer work
            tb.clicked.connect(action)
        return tb

    def sceneManagerSlot(self, ddict):
        if DEBUG:
            print("sceneManagerSlot", ddict)
        if ddict['event'] == 'addObject':
            if ddict['object'] is not None:
                self.addObject(ddict['object'], ddict['legend'])
            else:
                self.addObjectFromFileDialog()
        elif ddict['event'] == 'configurationLoaded':
            theta, phi = self.scene.getThetaPhi()
            self.__applyingCube = True
            self.wheelSlider10.wheel.setValue(theta)
            self.wheelSlider21.wheel.setValue(phi)
            zoomFactor = self.scene.getZoomFactor()
            self.wheelSlider12.setValue(numpy.log(zoomFactor)/numpy.log(2))
            self.__applyingCube = False
        else:
            if DEBUG:
                print("DOING NOTHING")

    def showSceneControlWindow(self):
        if self.manager is not None:
            self.manager.show()
            self.manager.raise_()

    def _getObject3D(self):
        moduleList = getObject3DModules()
        if not len(moduleList):
            return None
        actionList = []
        menu = qt.QMenu(self)
        for m in moduleList:
            function = m.getObject3DInstance
            if hasattr(m, 'MENU_TEXT'):
                text = qtQString(m.MENU_TEXT)
            else:
                text = os.path.basename(m.__file__)
                if text.endswith('.pyc'):
                    text = text[:-4]
                elif text.endswith('.py'):
                    text = text[:-3]
                text = qtQString(text)
            menu.addAction(text)
            actionList.append(text)
        a = menu.exec_(qt.QCursor.pos())
        if a is None:
            return None
        idx = actionList.index(a.text())
        object3D = moduleList[idx].getObject3DInstance()
        return object3D

    def addObjectFromFileDialog(self):
        try:
            object3D = self._getObject3D()
            if object3D is not None:
                try:
                    self.addObject(object3D, object3D.name())
                except:
                    self.addObject(object3D)
        except:
            qt.QMessageBox.critical(self, "Error adding object",
                "%s\n %s" % (sys.exc_info()[0], sys.exc_info()[1]),
                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                            qt.QMessageBox.NoButton)


    def sceneControlSlot(self, ddict):
        if DEBUG:
            print("sceneControlSlot", ddict)
        if ddict['event'] in ["objectSelected",
                              "objectDeleted",
                              "objectReplaced"]:
            ddict['legend'] = ddict['current']
            self.objectSelectedSlot(ddict, update_scene=False)
        elif ddict['event'] in ["SceneLimitsChanged"]:
            if ddict['autoscale'] or self.scene.getAutoScale():
                #force limits update
                self.scene.setAutoScale(True)
                self.scene.getLimits()
                self.sceneControl.updateView()
            self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())
            return
        #self.glWidget.updateGL()
        self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())

    def setSelectedObject(self, name, update_scene=True):
        self.objectSelectedSlot({'legend':name}, update_scene=update_scene)

    def objectSelectedSlot(self, ddict, update_scene=True):
        if DEBUG:
            print("objectSelectedSlot", ddict)
        if self.selectedObjectControl is None:
            self.selectedObjectControl = Object3DConfig.Object3DConfig()
            self.selectedObjectControl.sigObject3DConfigSignal.connect(\
                         self.objectControlSlot)
        # It is not necessary to show the manager
        if 0:
            if self.manager.isHidden():
                self.manager.show()
        legend = ddict['legend']
        if legend is None:
            return
        #Should I deselect object?
        self.activeObject = legend
        if update_scene:
            self.scene.setSelectedObject(self.activeObject)
            #print "BEFORE = ",self.scene.getSelectedObject()
            self.sceneControl.updateView()
            #print "AFTER = ",self.scene.getSelectedObject()
        #self.glWidget.objectsDict[legend]['object3D'].setSelected(True)

        configDict = self.scene[legend].root[0].getConfiguration()
        try:
            ddict['pointsizecapabilities'] = [self.glWidget._pointSizes[0],
                                              self.glWidget._pointSizes[-1],
                                              self.glWidget._pointSizeStep]
            ddict['linewidthcapabilities'] = [self.glWidget._lineWidths[0],
                                              self.glWidget._lineWidths[-1],
                                              self.glWidget._lineWidthStep]
        except:
            print("Error reading point and line capabilities. GL widget not initialized yet?")
            pass


        configDict['common'].update(ddict)
        self.selectedObjectControl.setWindowTitle(legend)
        self.selectedObjectControl.setConfiguration(configDict)
        self.selectedObjectControl.show()
        # This does not seem to be a problem any longer
        if 0:
            #make sure we do not mix modes
            self.setObjectSelectionModeSlot()
        text = "Object %s selected." % self.activeObject
        if 'event' in ddict:
            if ddict['event'] == "objectDeleted":
                text = ("Object %s deleted. " % ddict['previous']) + text
        self.infoLine.setText(text)
        if DEBUG:
            print("WHAT IS SCENE SAYING?")
            print("ACTIVE IS ", self.scene.getSelectedObject())

    def objectControlSlot(self, ddict):
        if DEBUG:
            print("objectControlSlot", ddict)
        if self.activeObject is None:
            ndict = {}
            ndict['legend'] = self.scene.name()
            self.objectSelectedSlot(ndict, update_scene=True)
            self.activeObject = self.scene.name()
        legend = self.activeObject
        if legend not in self.scene.getObjectList():
            self.selectedObjectControl.hide()
            return
        configDict = {'common':{}, 'private':{}}
        if 'private' in ddict:
            configDict['private'] = ddict['private']
        oldScale = self.scene[legend].root[0].getConfiguration()['common']['scale']
        configDict['common'].update(ddict['common'])
        self.scene[legend].root[0].setConfiguration(configDict)
        rootEvent = ddict.get('event', None)
        if legend == self.scene.name():
            sceneEventProcessed = False
            if rootEvent is not None:
                if rootEvent in ['AspectUpdated', 'DrawModeUpdated']:
                    # only properties passed to all objects are those related to drawing
                    objectList = self.scene.getObjectList()
                    i = 0
                    for o3dName in objectList:
                        if o3dName == self.scene.name():
                            continue
                        o3d = self.scene.getObject3DProxy(o3dName)
                        cfg = {}
                        cfg['common'] = o3d.getConfiguration()['common']
                        for key in ['pointsize', 'linewidth', 'transparency', 'mode']:
                            if key in ['mode']:
                                #drawing mode changed only if explicetly requested
                                if rootEvent in ['DrawModeUpdated']:
                                    cfg['common'][key] = ddict['common'][key]
                            else:
                                cfg['common'][key] = ddict['common'][key]
                        o3d.setConfiguration(cfg)
                    sceneEventProcessed = True
                    #self.scene[legend].root[0].setConfiguration(configDict)
                    #self.selectedObjectControl.setConfiguration(configDict)

            if not sceneEventProcessed:
                if 'common' in ddict:
                    if 'event' in ddict['common']:
                        if ddict['common']['event'] == 'ColormapChanged':
                            #print scene colormap changed
                            newColormap = ddict['common']['colormap']
                            objectList = self.scene.getObjectList()
                            i = 0
                            for o3dName in objectList:
                                if o3dName == self.scene.name():
                                    continue
                                o3d = self.scene.getObject3DProxy(o3dName)
                                cfg = {}
                                cfg['common'] = o3d.getConfiguration()['common']
                                minCmap = cfg['common']['colormap'][4]
                                maxCmap = cfg['common']['colormap'][5]
                                if (i == 0) or (newColormap[4] > minCmap):
                                    newColormap[4] = minCmap
                                if (i == 0) or (newColormap[5] > maxCmap):
                                    newColormap[5] = maxCmap
                                i = 1
                                cfg['common']['colormap'][0:4] = newColormap[0:4]
                                cfg['common']['colormap'][-1]  = newColormap[-1]
                                cfg['common']['event'] = ddict['common']['event']
                                o3d.setConfiguration(cfg)
                            #this is to get the proper limits for the scene
                            configDict['common']['colormap'] = newColormap
                            del configDict['common']['event']
                            cfg = self.scene[legend].root[0].getConfiguration()
                            cfg['common']['colormap'] = newColormap
                            self.scene[legend].root[0].setConfiguration(cfg)
                            self.selectedObjectControl.setConfiguration(cfg)

        if legend == self.scene.name() or self.scene.getAutoScale():
            newScale = self.scene[legend].root[0].getConfiguration()['common']['scale']
            if newScale != oldScale:
                #force cube calculation
                if self.scene.getAutoScale():
                    self.scene.getLimits()
                    self.sceneControl.updateView()
                position=self.scene.applyCube()
                self.glWidget.setCurrentViewPosition(position,
                                                     rotation_reset=False)
                return
        self.glWidget.cacheUpdateGL()

    def vertexSelectedSlot(self, ddict):
        self.infoLine.setText(ddict['info'])

    def setZoomFactor(self, value):
        if self.__applyingCube:
            return
        self.glWidget.setZoomFactor(pow(2, value))
        self.wheelSlider12.label.setText("%.2f" % pow(2, value))

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360
        angle = angle % 360
        return angle

    def setPhi(self, value):
        if self.__applyingCube:
            return
        value = self.normalizeAngle(value)
        theta, phi = self.scene.getThetaPhi()
        if value != phi:
            self.scene.setThetaPhi(theta, value)
            self.glWidget.cacheUpdateGL()

    def setTheta(self, value):
        if self.__applyingCube:
            return
        value = self.normalizeAngle(value)
        theta, phi = self.scene.getThetaPhi()
        if value != theta:
            self.scene.setThetaPhi(value, phi)
            self.glWidget.cacheUpdateGL()

    def selectObject(self):
        if self.glWidget.objectSelectionMode():
            self.glWidget.setObjectSelectionMode(False)
        else:
            self.glWidget.setObjectSelectionMode(True)

    def selectVertex(self):
        if self.glWidget.vertexSelectionMode():
            self.glWidget.setVertexSelectionMode(False)
        else:
            self.glWidget.setVertexSelectionMode(True)

    def setAlpha(self, value):
        self.glWidget.setSelectedObjectAlpha(value/10.)
        #self.glWidget.updateGL()
        self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())

    def addObject(self, ob, legend = None, update_scene=True):
        self.sceneControl.scene.addObject(ob, legend)
        self.sceneControl.scene.getLimits()
        self.sceneControl.updateView()
        if self.activeObject in [None, 'None']:
            ndict = {}
            if 0:
                #the default is the scene
                ndict['legend'] = self.scene.name()
            else:
                #the default is the selected object
                ndict['legend'] = legend
            self.objectSelectedSlot(ndict, update_scene=update_scene)
            self.scene.setSelectedObject(self.activeObject)
        if update_scene:
            self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())

    def removeObject(self, legend, update_scene=True):
        if self.activeObject == legend:
            self.activeObject = None
        self.sceneControl.scene.removeObject(legend)
        self.sceneControl.scene.getLimits()
        self.sceneControl.updateView()
        if update_scene:
            self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())

    def clear(self, update_scene=True):
        self.activeObject = None
        self.sceneControl.scene.clearTree()
        self.sceneControl.scene.getLimits()
        self.sceneControl.updateView()
        if update_scene:
            self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())

    def closeEvent(self, event):
        objectList = self.scene.getObjectList()
        for object3D in objectList:
            if object3D == "Scene":
                continue
            self.scene.removeObject(object3D)
        self.manager.close()
        self.glWidget.close()
        # This is needed to force the destruction of the Object3D(s)
        del self.manager
        #del self.scene
        qt.QWidget.closeEvent(self, event)

if __name__ == '__main__':
    import sys
    import Object3DBase
    app = qt.QApplication(sys.argv)

    window = SceneGLWindow()
    window.show()
    if 0:
        class MyObject(Object3DBase.Object3D):
            def drawObject(self):
                #GL.glShadeModel(GL.GL_FLAT)
                GL.glShadeModel(GL.GL_SMOOTH) #in order not to have just blue face
                GL.glBegin(GL.GL_TRIANGLE_STRIP)
                alpha = 1.0 - self._configuration['common']['transparency']
                GL.glColor4f(1., 0., 0., alpha)      # Red
                GL.glVertex3f(-25., 0., 0.)
                GL.glColor4f(0., 1., 0., alpha)      # Green
                GL.glVertex3f(25., 0., 0.)
                GL.glColor4f(0., 0., 1., alpha)      # Blue
                GL.glVertex3f(0, 25, 0.)
                GL.glEnd()

        ob3D1 = MyObject(name="Object1")
        ob3D1.setLimits(-25, 0.0, 0.0, 25, 25, 0.0)

        ob3D2 = MyObject(name="Object2")
        ob3D2.setLimits(-25, 0.0, 0.0, 25, 25, 0.0)


        #translate
        config = ob3D2.getConfiguration()
        config['common']['translation'] = [0.0, 0.0, 0.0]
        ob3D2.setConfiguration(config)

        window.setWindowTitle('Object3D Window')
        window.addObject(ob3D1, "Object1")
        window.addObject(ob3D2, "Object2")
        window.glWidget.setObjectSelectionMode(True)
        window.show()
        window.glWidget.setZoomFactor(1.0)
    sys.exit(app.exec_())
