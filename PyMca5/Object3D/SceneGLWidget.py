#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
import OpenGL.GL  as GL
import OpenGL.GLU as GLU
import sys
import numpy
import weakref
from . import Object3DQt as qt
from . import Object3DCoordinates
from . import Object3DRedBookFont
from . import Scene
from . import ObjectTree
from . import GLWidgetCachePixmap
from .HorizontalSpacer import HorizontalSpacer
QTVERSION = qt.qVersion()

DEBUG = 0
SCENE_MATRIX = True


if hasattr(qt, 'QOpenGLWidget'):  # PyQt>=5.4
    _BaseOpenGLWidget = qt.QOpenGLWidget
    USING_QOPENGLWIDGET = True
elif hasattr(qt, 'QGLWidget'):
    _BaseOpenGLWidget = qt.QGLWidget
    USING_QOPENGLWIDGET = False
else:
    raise ImportError("QOpenGLWidget not available.")


class SceneGLWidget(_BaseOpenGLWidget):

    sigScaleChanged = qt.pyqtSignal(object)
    sigObjectSelected = qt.pyqtSignal(object)
    sigVertexSelected = qt.pyqtSignal(object)
    sigMouseMoved = qt.pyqtSignal(object)

    def __init__(self, parent = None, scene=None):
        #_BaseOpenGLWidget.__init__(self, qt.QGLFormat(qt.QGL.SampleBuffers), parent)
        _BaseOpenGLWidget.__init__(self, parent)
        if 1:
            self.__test = None
        else:
            self.__test = [200, 0, 200, 400]
        if scene is None:
            self.scene = Scene.Scene()
        else:
            self.scene = weakref.proxy(scene)
        self.__ownTree =  ObjectTree.ObjectTree("__Scene__", "_Scene_")
        self.__ownTree.addChildTree(self.scene.tree)

        self._visualVolume = [-100., 100., -100., 100., -100.0, 100.0]
        if SCENE_MATRIX:
            self.__currentViewPosition = self.scene.getCurrentViewMatrix()
        else:
            self.__currentViewPosition = numpy.zeros((4,4), numpy.float32)
            for i in [0, 1, 2, 3]:
                self.__currentViewPosition[i, i] = 1
            self.scene.setCurrentViewMatrix(self.__currentViewPosition)
        self.__sceneModelViewMatrix = numpy.zeros((4,4), numpy.float64)
        for i in [0, 1, 2, 3]:
            self.__sceneModelViewMatrix[i, i] = 1
        self.__sceneProjectionMatrix = self.__sceneModelViewMatrix * 1.0
        self.__selectedModelViewMatrix  = self.__sceneModelViewMatrix * 1.0
        self.__selectedProjectionMatrix = self.__sceneModelViewMatrix * 1.0


        self.__zoomFactor = 1.0/self.scene.getZoomFactor()

        self.scale   = 1.0
        self._objectSelectionMode = False
        self._vertexSelectionMode = False
        self.__selectingVertex    = False
        if hasattr(self, "setAutoBufferSwap"):
            self.setAutoBufferSwap(False)
        self.autoScale = True
        self.coordinates = Object3DCoordinates.Object3DCoordinates(self)
        self.lastPos = qt.QPoint()


        #cache pixmap
        self.__cacheEnabled = True
        self.__usingCache = False
        self.__outOfSelectMode = False
        self.__cacheTexture = GLWidgetCachePixmap.GLWidgetCachePixmap()

    if USING_QOPENGLWIDGET:
        def updateGL(self):
            return self.update()

        def swapBuffers(self):
            pass
            # no need to get the context with self.context() to swap buffers

    def setCurrentViewPosition(self, position, rotation_reset=None):
        if rotation_reset is None:
            rotation_reset = True
        if rotation_reset:
            self.scene.setThetaPhi(0, 0)
        if position.shape == (3, 4):
            self.__currentViewPosition[0:3, :] = position[0:3, :]
        else:
            self.__currentViewPosition[:, :] = position[:, :]
        if SCENE_MATRIX:
            self.scene.setCurrentViewMatrix(self.__currentViewPosition)
        self.cacheUpdateGL()

    def getCurrentViewPosition(self):
        if SCENE_MATRIX:
            return self.scene.getCurrentViewMatrix()
        else:
            return self.__currentViewPosition

    def addObject3D(self, ob, legend = None, plot=True):
        self.scene.addObject(ob, legend)
        if plot:
            #this is to recalculate the projection limits
            xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()
            self._visualVolume = [xmin, xmax, ymin, ymax, zmin, zmax]
            self.cacheUpdateGL()
        return

    def setVisualizationVolume(self, xmin, xmax,
                                     ymin, ymax,
                                     zmin, zmax):
        if xmax == xmin: xmax = xmin + 1
        if ymax == ymin: ymax = ymin + 1
        if zmax == zmin: zmax = zmin + 1
        self._visualVolume = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.cacheUpdateGL()

    def visualizationVolume(self):
        return self._visualVolume * 1

    def minimumSizeHint(self):
        return qt.QSize(50, 50)

    def sizeHint(self):
        return qt.QSize(400, 400)

    def setScale(self, value):
        self.scale = value
        self.sigScaleChanged.emit(value)
        self.cacheUpdateGL()

    def setZoomFactor(self, value):
        # I have to update the viewport
        self.__zoomFactor = 1.0/value
        self.scene.setZoomFactor(value)
        self.cacheUpdateGL()

    def getZoomFactor(self):
        value = self.scene.getZoomFactor()
        self.__zoomFactor = 1.0/value
        return value

    def getScale(self):
        return self.scale * 1

    def setObjectSelectionMode(self, value):
        self._objectSelectionMode = value

    def objectSelectionMode(self):
        return self._objectSelectionMode

    def setVertexSelectionMode(self, value):
        self._vertexSelectionMode = value

    def vertexSelectionMode(self):
        return self._vertexSelectionMode

    def setSelectedObjectAlpha(self, alpha):
        i = 0
        for legend in self.objectsList:
            ob = self.objectsDict[legend]['object3D']
            if ob.selected():
                if DEBUG:
                    print("setting alpha = ", alpha)
                ob.setAlpha(alpha)
            else:
                if DEBUG:
                    print(" object %d not selected" % i)
                i += 1

    def initializeGL(self):
        if DEBUG:
            print("OpenGL version = ", GL.glGetString(GL.GL_VERSION))
            print("Supported extensions = ", GL.glGetString(GL.GL_EXTENSIONS))
        GL.glClearDepth(1.0)
        ##########GL.glEnable(GL.GL_DEPTH_TEST)
        self.clearColor = [0.5, 0.5, 0.5, 1.0]
        #self.clearColor = [0.0, 0.0, 0.0, 1.0]
        #self.clearColor = [0.1, 0.1, 0.1, 1.0]
        GL.glClearColor(*self.clearColor)
        #enable blending for transparencies
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        if 0:
            #This line gave me problems on the BSIN when using
            #transparency
            GL.glShadeModel(GL.GL_SMOOTH)
        else:
            GL.glShadeModel(GL.GL_FLAT)
        GL.glDisable(GL.GL_DITHER) # no dithering, please
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        #initialize own font lists
        self.redBookFont = Object3DRedBookFont.Object3DRedBookFont()
        self.redBookFont.initialize()
        ####

        if 0:
            #initialize lighting
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION,  (0.0, 0.0, 1, 0.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            GL.glEnable(GL.GL_LIGHT0)

            GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, (0.0, 0.0, -1.0, 0.0))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
            GL.glEnable(GL.GL_LIGHT1)

            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            if 1:
                #Light off
                GL.glDisable(GL.GL_LIGHTING)
            else:
                #Light on
                GL.glEnable(GL.GL_LIGHTING)
        if 1:
            #lighting from marching cubes example
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION,  (1.0, 1.0, 1.0, 0.0))
            self.afAmbientWhite  = [0.25, 0.25, 0.25, 1.00]
            self.afAmbientRed    = [0.25, 0.00, 0.00, 1.00]
            self.afAmbientGreen  = [0.00, 0.25, 0.00, 1.00]
            self.afAmbientBlue   = [0.00, 0.00, 0.25, 1.00]
            self.afDiffuseWhite  = [0.75, 0.75, 0.75, 1.00]
            self.afDiffuseRed    = [0.75, 0.00, 0.00, 1.00]
            self.afDiffuseGreen  = [0.00, 0.75, 0.00, 1.00]
            self.afDiffuseBlue   = [0.00, 0.00, 0.75, 1.00]
            self.afSpecularWhite = [1.00, 1.00, 1.00, 1.00]
            self.afSpecularRed   = [1.00, 0.25, 0.25, 1.00]
            self.afSpecularGreen = [0.25, 1.00, 0.25, 1.00]
            self.afSpecularBlue  = [0.25, 0.25, 1.00, 1.00]
            self.afPropertiesAmbient  = [0.50, 0.50, 0.50, 1.00]
            self.afPropertiesDiffuse  = [0.75, 0.75, 0.75, 1.00]
            self.afPropertiesSpecular = [1.00, 1.00, 1.00, 1.00]
            GL.glLightfv( GL.GL_LIGHT0, GL.GL_AMBIENT,  self.afPropertiesAmbient)
            GL.glLightfv( GL.GL_LIGHT0, GL.GL_DIFFUSE,  self.afPropertiesDiffuse)
            GL.glLightfv( GL.GL_LIGHT0, GL.GL_SPECULAR, self.afPropertiesSpecular)
            GL.glLightModelf(GL.GL_LIGHT_MODEL_TWO_SIDE, 1.0)

            GL.glEnable( GL.GL_LIGHT0 )

            GL.glMaterialfv(GL.GL_BACK,  GL.GL_AMBIENT,   self.afAmbientGreen)
            GL.glMaterialfv(GL.GL_BACK,  GL.GL_DIFFUSE,   self.afDiffuseGreen)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT,   self.afAmbientBlue)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_DIFFUSE,   self.afDiffuseBlue)
            GL.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR,  self.afSpecularWhite)
            GL.glMaterialf( GL.GL_FRONT, GL.GL_SHININESS, 25.0)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
        else:
            #lighting from appli
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION,  (0, 0, 1000, 0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.4, 0.4, 0.4, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.6, 0.6, 0.6, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            GL.glEnable(GL.GL_LIGHT0)

            GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION,  (0, 1000, 0, 0))
            #GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, (0.2, 0.2, 0.2, 0.2))
            GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            GL.glEnable(GL.GL_LIGHT1)

            GL.glLightfv(GL.GL_LIGHT2, GL.GL_POSITION,  (1000, 0, 0, 0))
            #GL.glLightfv(GL.GL_LIGHT2, GL.GL_AMBIENT, (0.05, 0.05, 0.05, 0.05))
            #GL.glLightfv(GL.GL_LIGHT2, GL.GL_DIFFUSE, (0.2, 0.2, 0.2, 1.0))
            GL.glLightfv(GL.GL_LIGHT2, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            GL.glEnable(GL.GL_LIGHT2)

            GL.glLightfv(GL.GL_LIGHT3, GL.GL_POSITION,  (0, 0, -1000, 0))
            #GL.glLightfv(GL.GL_LIGHT3, GL.GL_AMBIENT, (0.05, 0.05, 0.05, 0.05))
            #GL.glLightfv(GL.GL_LIGHT3, GL.GL_DIFFUSE, (0.2, 0.2, 0.2, 1.0))
            GL.glLightfv(GL.GL_LIGHT3, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            GL.glEnable(GL.GL_LIGHT3)

            GL.glLightfv(GL.GL_LIGHT4, GL.GL_POSITION,  (-1000, 0, 0, 0))
            #GL.glLightfv(GL.GL_LIGHT4, GL.GL_DIFFUSE, (0.2, 0.2, 0.2, 0.2))
            GL.glLightfv(GL.GL_LIGHT4, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            #GL.glLightfv(GL.GL_LIGHT4, GL.GL_AMBIENT, (0.2, 0.2, 0.2, 0.2))
            GL.glEnable(GL.GL_LIGHT4)

            GL.glLightfv(GL.GL_LIGHT5, GL.GL_POSITION,  (0, -1000, 0, 0))
            #GL.glLightfv(GL.GL_LIGHT5, GL.GL_DIFFUSE, (0.2, 0.2, 0.2, 0.2))
            #GL.glLightfv(GL.GL_LIGHT5, GL.GL_AMBIENT, (0.05, 0.05, 0.05, 0.05))
            GL.glLightfv(GL.GL_LIGHT5, GL.GL_SPECULAR, (0.2, 0.2, 0.2, 1.0))
            GL.glEnable(GL.GL_LIGHT5)
            #GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
        if 1:
            #Light off
            GL.glDisable(GL.GL_LIGHTING)
        else:
            #Light on
            GL.glEnable(GL.GL_LIGHTING)


        if 0:
            GL.glEnable(GL.GL_CULL_FACE) #hides bottom face
        else:
            GL.glDisable(GL.GL_CULL_FACE) #shows opposite face

        #this is slower, but much better when performing rotations
        GL.glEnable(GL.GL_DEPTH_TEST)

        # get supported point size and step size
        self._pointSizes    = GL.glGetFloatv(GL.GL_POINT_SIZE_RANGE)
        self._pointSizeStep = GL.glGetFloatv(GL.GL_POINT_SIZE_GRANULARITY)

        # get supported line width and step size
        self._lineWidths    = GL.glGetFloatv(GL.GL_LINE_WIDTH_RANGE)
        self._lineWidthStep = GL.glGetFloatv(GL.GL_LINE_WIDTH_GRANULARITY)
        if not hasattr(self._lineWidths, '__iter__'):
            #some versions give back a single value instead
            self._lineWidths = [self._lineWidthStep, self._lineWidths]

        #This should be at least 256
        self.__maximumTextureLength = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)

        #This should be at least 2
        self.__maximumTextureDepth = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_STACK_DEPTH)

        #print "point sizes = " , self._pointSizes
        #print "granularity =  ", self._pointSizeStep

        #nice lines
        #it could be blending is necessary to get them nice ...
        #GL.glEnable (GL.GL_LINE_SMOOTH)
        #GL.glLineWidth (4)
        #GL.glHint (GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)

    def forceRedrawing(self):
        self.__usingCache = False

    def setCacheEnabled(self, value):
        if value:
            self.__cacheEnabled = True
        else:
            self.__cacheEnabled = False

    def drawCacheTexture(self):
        if DEBUG:
            print("USING CACHE TEXTURE!!!!!!!!!!!!")
        GL.glClearColor(*self.clearColor)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width(),
                   0, self.height(),
                   -1, 1)
        self.__cacheTexture.drawObject()
        #
        self.swapBuffers()

    def paintGL(self):
        if self.__selectingVertex:
            GL.glClearColor(1.0, 1.0, 1.0, 1.0) #white
            #GL.glClearColor(0.0, 0.0, 0.0, 0.0)  #black
        else:
            GL.glClearColor(*self.clearColor)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        if self.__outOfSelectMode and self.__cacheEnabled and self.__usingCache:
            self.drawCacheTexture()
            return
        elif self.__cacheEnabled and\
           self.__usingCache and\
           (not self.__selectingVertex) and\
           (GL.glGetIntegerv(GL.GL_RENDER_MODE) != GL.GL_SELECT):
            self.drawCacheTexture()
            return
        else:
            #make sure texturing is disabled
            GL.glDisable(GL.GL_TEXTURE_2D)
        #setup the projection
        width = self.width()
        height = self.height()
        #self.setupViewport(width, height)
        self.setupProjection(width, height)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        if SCENE_MATRIX:
            GL.glMultMatrixf(self.scene.getTransformationMatrix())
        else:
            #apply the selected face
            GL.glMultMatrixf(self.__currentViewPosition)

            # center of the scene
            xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()
            centerX = 0.5 * (xmax + xmin)
            centerY = 0.5 * (ymax + ymin)
            centerZ = 0.5 * (zmax + zmin)
            #zenith angle theta in spherical coordinates z = r * cos(theta)
            #rotate theta around Y axis
            #azimuthal angle phi in spherical coordinates
            #rotate phi around Z axis
            theta, phi = self.scene.getThetaPhi()
            sceneConfig = self.scene.tree.root[0].getConfiguration()
            #I have to rotate around the center of the scene
            #taking into account the scale it will use
            scale = sceneConfig['common']['scale']
            anchor = [centerX*scale[0], centerY*scale[1], centerZ*scale[2]]
            #print "ANCHOR = ", anchor
            #print "ZENITH = %f, AZIMUTH = %f " % (theta, phi)
            #M = self.getRotationMatrix(0, theta, phi, anchor)
            #print "M = ", M
            GL.glTranslated(anchor[0], anchor[1], anchor[2])
            GL.glRotated(theta, 0.0, 1.0, 0.0)
            GL.glRotated(phi, 0.0, 0.0, 1.0)
            GL.glTranslated(-anchor[0], -anchor[1], -anchor[2])

        self.drawScene()

        #prepare a pure virtual method for derived classes ???
        self.userPaintGL()
        #keep a copy of the current image
        #self.__finalImage = GL.glReadPixelsub(0,0, self.width(),self.height(),

        if GL.glGetIntegerv(GL.GL_RENDER_MODE) != GL.GL_SELECT and\
           (not self.__selectingVertex):
            try:
                # next line crashes on windows with intel HD 5500 with Qt 5.10.1
                GL.glReadBuffer(GL.GL_BACK)
                self.__finalImage = GL.glReadPixelsub(0,0,
                                    self.width(), self.height(),
                                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
                if not hasattr(self.__finalImage, "dtype"):
                    # we did not receive an array (python 3) ...
                    self.__finalImage = numpy.fromstring(self.__finalImage,
                                                    dtype=numpy.uint8)

                self.__cacheTexture.setPixmap(self.__finalImage,
                                          self.width(), self.height())
                self.__usingCache = True
                #self.saveImage()
            except:
                self.__usingCache = False
        

        if hasattr(self, "doubleBuffer"):
            if self.doubleBuffer():
                if not self.autoBufferSwap():
                    if not self.__selectingVertex:
                        if GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_RENDER:
                            self.swapBuffers()
                else:
                    print("WARNING: Expected to work with autoBufferSwap off")
        else:
            if not self.__selectingVertex:
                if GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_RENDER:
                    self.swapBuffers()
        if 0:
            #keep a hardcopy of the image
            image = GL.glReadPixels(0,0,self.width(),self.height(),
                                    GL.GL_BGRA,GL.GL_UNSIGNED_BYTE)
            qimage=qt.QImage(image, self.width(),
                             self.height(),
                             qt.QImage.Format_RGB32).mirrored(0, 1)
            qimage.save('Object3DGLWidget2.png')
        return

    def getQImage(self):
        qimage = qt.QImage(self.__finalImage,
                           self.width(),
                           self.height(),
                           qt.QImage.Format_ARGB32).mirrored(0, 1)
        a=qimage.rgbSwapped()
        return a

    def saveImage(self, filename=None):
        if filename is None:
            filename = 'Object3DGLWidget.png'
        qimage = qt.QImage(self.__finalImage,
                           self.width(),
                           self.height(),
                           qt.QImage.Format_ARGB32).mirrored(0, 1)
        a=qimage.rgbSwapped()
        return a.save(filename)

    def drawTree(self, tree):
        childList = tree.childList()
        for subTree in childList:
            name = subTree.name()
            object3D = subTree.root[0]
            GL.glPushMatrix()
            GL.glPushName(self.scene.getIndex(name))
            object3D.setVertexSelectionMode(False)
            configDict = object3D.getConfiguration()['common']
            #This call could be made at the object if needed ...
            GL.glPointSize(configDict['pointsize'])
            SCALE_BEFORE = True
            if SCALE_BEFORE:
                #print "SCALING BEFORE"
                GL.glScalef(*configDict['scale'])
            if self.__selectingVertex:
                if object3D.selected():
                    #force Object3D GL_SELECT equivalent drawing
                    object3D.setVertexSelectionMode(True)
                else:
                    #We'll only draw the object that can be selected ????
                    #or the different objects will take care?
                    # I take care below
                    pass

            ########### Should this be made at the object itself ?? #
            if 1:
                anchor = configDict['anchor']
                anchorPosition = [0.0, 0.0, 0.0]

                #print all this made by the object itself???
                limits = object3D.getLimits()

                for i in range(3):
                    xmin =limits[0][i] * 1
                    xmax =limits[1][i] * 1
                    if anchor[i] == 1:
                        anchorPosition[i] = xmin
                        continue
                    if anchor[i] == 2:
                        anchorPosition[i] = 0.5 * (xmax + xmin)
                        continue
                    if anchor[i] == 3:
                        anchorPosition[i] = xmax
                        continue

                TRANSLATE_BEFORE = True
                if TRANSLATE_BEFORE:
                    # object translation in the parent system
                    GL.glTranslated(*configDict['translation'])
                    anchorPosition[0] += configDict['translation'][0]
                    anchorPosition[1] += configDict['translation'][1]
                    anchorPosition[2] += configDict['translation'][2]

                GL.glTranslated(anchorPosition[0],
                                anchorPosition[1],
                                anchorPosition[2])

                #this works
                #RotX
                angle = configDict['rotation'][0]*numpy.pi/180.
                cs = numpy.cos(angle)
                sn = numpy.sin(angle)
                rotX = numpy.zeros((4,4), numpy.float64)
                rotX[0,0] =  1
                rotX[1,1] =  1
                rotX[2,2] =  1
                rotX[3,3] =  1
                rotX[1,1] =  cs; rotX[1,2] = sn
                rotX[2,1] = -sn; rotX[2,2] = cs

                #RotY
                angle = configDict['rotation'][1]*numpy.pi/180.
                cs = numpy.cos(angle)
                sn = numpy.sin(angle)
                rotY = numpy.zeros((4,4), numpy.float64)
                rotY[0,0] =  1
                rotY[1,1] =  1
                rotY[2,2] =  1
                rotY[3,3] =  1
                rotY[0,0] =  cs; rotY[0,2] = -sn   #inverted respect to the others
                rotY[2,0] =  sn; rotY[2,2] =  cs

                #RotZ
                angle = configDict['rotation'][2]*numpy.pi/180.
                cs = numpy.cos(angle)
                sn = numpy.sin(angle)
                rotZ = numpy.zeros((4,4), numpy.float64)
                rotZ[0,0] =  1
                rotZ[1,1] =  1
                rotZ[2,2] =  1
                rotZ[3,3] =  1
                rotZ[0,0] =  cs; rotZ[0,1] = sn
                rotZ[1,0] = -sn; rotZ[1,1] = cs

                #The final matrix
                rotMatrix = numpy.dot(rotZ,numpy.dot(rotY, rotX))

                #perform the inplace rotation
                GL.glMultMatrixd(rotMatrix)

                #find out where the anchor goes under that rotation
                trans = numpy.zeros((4,4), numpy.double)
                trans[0,0] = 1.0
                trans[1,1] = 1.0
                trans[2,2] = 1.0
                trans[3,3] = 1.0
                trans[3,0] = anchorPosition[0]
                trans[3,1] = anchorPosition[1]
                trans[3,2] = anchorPosition[2]
                distance = numpy.dot(rotMatrix, trans)

                # and subtract it
                GL.glTranslated(-distance[3,0],
                                -distance[3,1],
                                -distance[3,2])

                if not TRANSLATE_BEFORE:
                    # object translation to its position
                    # The translation is in the object reference system
                    GL.glTranslated(*configDict['translation'])
                else:
                    #find out the displacement under that rotation
                    trans = numpy.zeros((4,4), numpy.double)
                    trans[0,0] = 1.0
                    trans[1,1] = 1.0
                    trans[2,2] = 1.0
                    trans[3,3] = 1.0
                    trans[3,0] = configDict['translation'][0]
                    trans[3,1] = configDict['translation'][1]
                    trans[3,2] = configDict['translation'][2]
                    distance = numpy.dot(rotMatrix, trans)
                    GL.glTranslated(distance[3,0],
                                    distance[3,1],
                                    distance[3,2])

                if not SCALE_BEFORE:
                    #print "SCALING AFTER"
                    GL.glScalef(*configDict['scale'])

                #get the current matrix
                if name.upper() == "SCENE":
                    self.__sceneModelViewMatrix = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
                    self.__sceneProjectionMatrix = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)

                if object3D.selected():
                    self.__selectedModelViewMatrix = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
                    self.__selectedProjectionMatrix = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)

                if self.__selectingVertex:
                    if object3D.selected():
                        # draw the object
                        object3D.draw()
                else:
                    # draw the object
                    object3D.draw()

                if object3D.selected() and not self.__selectingVertex:
                    self.coordinates.setLimits(limits)
                    self.coordinates.setFlags(*configDict['showlimits'])
                    #self.coordinates.setLimits(object3D.getLimits())
                    self.coordinates.draw()
            ########################################################
            else:
                self.scene[name].root[0].draw()
            GL.glPopName()
            self.drawTree(subTree)
            GL.glPopMatrix()

    def drawScene(self):
        #be ready for selections
        #get the maximum number of names
        self.maxNumberNames = GL.glGetIntegerv(GL.GL_MAX_NAME_STACK_DEPTH)  #The very minimum is 64
        objectCounter = 1
        if GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_SELECT:
            GL.glInitNames()    #reset the name list
            #GL.glPushName(-1)   #the red book says -1, but it is not accepted here

        if 1:
            self.drawTree(self.__ownTree)
        else:
            #This misses drawing the scene bounding box
            self.drawTree(self.scene.tree)

    def getRotationMatrix(self, xRot, yRot, zRot, anchor=None):
        """
        Angles given in degrees!!!!
        """
        M = numpy.zeros((4,4), numpy.float64)
        M[0, 0] = 1
        M[1, 1] = 1
        M[2, 2] = 1
        M[3, 3] = 1
        if (xRot == 0) and (yRot == 0) and (zRot == 0):
            return M

        if anchor is None:
            anchorPosition = [0.0, 0.0, 0.0]
        else:
            anchorPosition = anchor

        trans = M * 1
        rotX  = M * 1
        rotY  = M * 1
        rotZ  = M * 1

        #translation
        M[3, 0] = anchorPosition[0]
        M[3, 1] = anchorPosition[1]
        M[3, 2] = anchorPosition[2]

        #this works
        #RotX
        angle = xRot * numpy.pi/180.
        cs = numpy.cos(angle)
        sn = numpy.sin(angle)
        rotX = numpy.zeros((4,4), numpy.float64)
        rotX[0,0] =  1
        rotX[1,1] =  1
        rotX[2,2] =  1
        rotX[3,3] =  1
        rotX[1,1] =  cs; rotX[1,2] = sn
        rotX[2,1] = -sn; rotX[2,2] = cs

        #RotY
        angle = yRot * numpy.pi/180.
        cs = numpy.cos(angle)
        sn = numpy.sin(angle)
        rotY = numpy.zeros((4,4), numpy.float64)
        rotY[0,0] =  1
        rotY[1,1] =  1
        rotY[2,2] =  1
        rotY[3,3] =  1
        rotY[0,0] =  cs; rotY[0,2] = -sn   #inverted respect to the others
        rotY[2,0] =  sn; rotY[2,2] =  cs

        #RotZ
        angle = zRot * numpy.pi/180.
        cs = numpy.cos(angle)
        sn = numpy.sin(angle)
        rotZ = numpy.zeros((4,4), numpy.float64)
        rotZ[0,0] =  1
        rotZ[1,1] =  1
        rotZ[2,2] =  1
        rotZ[3,3] =  1
        rotZ[0,0] =  cs; rotZ[0,1] = sn
        rotZ[1,0] = -sn; rotZ[1,1] = cs

        #The final rotation matrix
        rotMatrix = numpy.dot(rotZ,numpy.dot(rotY, rotX))

        #perform the in-place rotation
        #GL.glMultMatrixd(rotMatrix)

        #find out where the anchor goes under that rotation
        trans = numpy.zeros((4,4), numpy.double)
        trans[0,0] = 1.0
        trans[1,1] = 1.0
        trans[2,2] = 1.0
        trans[3,3] = 1.0
        trans[3,0] = anchorPosition[0]
        trans[3,1] = anchorPosition[1]
        trans[3,2] = anchorPosition[2]
        distance = numpy.dot(rotMatrix, trans)

        # and subtract it
        trans[3,0] = -distance[3,0]
        trans[3,1] = -distance[3,1]
        trans[3,2] = -distance[3,2]
        M = numpy.dot(trans, numpy.dot(rotMatrix,M))
        return M

    def userPaintGL(self):
        pass

    def resizeGL(self, width, height):
        if self.__outOfSelectMode == False:
            self.forceRedrawing()
        else:
            pass
            #print "OUT OF SELECT MODE"
        self.setupViewport(width, height)
        if 0:
            #moved to paintGL
            self.setupProjection(width, height)
        self.updateGL()        #Do I need to ask for the update? YES!!!
        if self.__outOfSelectMode:
            self.__outOfSelectMode = False

    def setupProjection(self, width, height):
        # I can also apply a zoom based on the visual volume
        if self.scene is None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._visualVolume
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()

        #zmax = zmax + 2 * deltaz
        #zmin = zmin - 2 * deltaz
        if 1:
            zmean = 0.5 * (zmax + zmin)
        else:
            zmin = min(-abs(zmin), -abs(zmax))
            zmax = max(abs(zmin), abs(zmax))

        deltaz = zmax - zmin
        #the first zoom should be 1.0
        deltax = (xmax - xmin) * 0.5
        deltay = (ymax - ymin) * 0.5
        deltaz = (zmax - zmin) * 0.5
        deltax = deltax * self.__zoomFactor
        deltay = deltay * self.__zoomFactor
        deltaz = deltaz * self.__zoomFactor

        #What if the scene has a different scale than 1, 1, 1?
        #The calculation works quite nicely but, if the user
        #changes the z scale to -1, we may loose the whole image because
        #of the change in sign of zmax and zmin
        xScale, yScale, zScale = self.scene.tree.root[0].getConfiguration()['common']['scale']
        if xScale < 0:
            t = xmax
            xmax = xmin * xScale
            xmin = t * xScale
        elif xScale > 0:
            xmax *= xScale
            xmin *= xScale
        if yScale < 0:
            t = ymax
            ymax = ymin * yScale
            ymin = t * yScale
        elif yScale > 0:
            ymax *= yScale
            ymin *= yScale
        if zScale < 0:
            t = zmax
            zmax = zmin * zScale
            zmin = t * zScale
        elif zScale > 0:
            zmax *= zScale
            zmin *= zScale
        #end of scene scale correction

        xmean = 0.5 * (xmax + xmin)
        ymean = 0.5 * (ymax + ymin)
        zmean = 0.5 * (zmax + zmin)
        xmin = xmean - deltax
        xmax = xmean + deltax
        ymin = ymean - deltay
        ymax = ymean + deltay
        #zmin = zmean - deltaz
        #zmax = zmean + deltaz
        if (width < height):
            #This is to center at least the first plot at zoom 1
            #GL.glTranslatef(0.0, (height-width)/float(height), 0.0)
            ratio = height / (1.0 * width)
            ymin = ymean - ratio * deltay
            ymax = ymean + ratio * deltay
        else:
            ratio = width / (1.0 * height)
            xmin = xmean - deltax * ratio
            xmax = xmean + deltax * ratio

        if 0:
            #This is the minimum not to cut when plotting the XY plane
            GL.glOrtho(xmin, xmax,
                       ymin, ymax,
                      -zmax, -zmin)
        else:
            radius = numpy.sqrt(deltax * deltax + deltay * deltay)
            zmax += radius
            zmin -= radius
            if (xmin == xmax):
                xmax += 0.5
                xmin -= 0.5
            if (ymin == ymax):
                ymax += 0.5
                ymin -= 0.5
            GL.glOrtho(xmin, xmax,
                       ymin, ymax,
                      -zmax, -zmin)
        #If the limits are set here, it does not seem to be correct
        #to store them there ...
        self.scene.setOrthoLimits(xmin,
                                  ymin,
                                  zmin,
                                  xmax,
                                  ymax,
                                  zmax)
        self.__orthoLimits = [xmin, ymin, zmin, xmax, ymax, zmax]
        #print "PROJECTION zmin, zmax = ", zmin, zmax

        # reset model matrix
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def setupViewport(self, width, height):
        """
        Done here to be able to do the same transformations in object selection
        mode without code duplication
        """
        if 0:
            side = min(width, height)
            GL.glViewport((width - side) / 2, (height - side) / 2, side, side)


            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            #GL.glOrtho(-100., 100., 100., -100., -100.0, 100.0)
            GL.glOrtho(*self._visualVolume)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        else:
            if (height == 0):
                height = 1
            if (width == 0):
                width = 1

            #match the viewport to the window size
            GL.glViewport(0, 0, width, height)
            #reset projection matrix
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()

            #I can apply zoom below or now
            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
            if 0 and self.__test is None:
                GLU.gluPickMatrix(viewport[2]*0.5,
                              viewport[3]*0.5,
                              viewport[2]*0.5*self.__zoomFactor,
                              viewport[3]*0.5*self.__zoomFactor,
                              viewport)
            elif 0 and self.__test is not None:
                GLU.gluPickMatrix(self.__test[0],
                              self.__test[1],
                              self.__test[2],
                              self.__test[3],
                              viewport)
                self.__test = None
            #gluPickMatrix code
            """
gluPickMatrix(GLdouble x, GLdouble y, GLdouble deltax, GLdouble deltay,
          GLint viewport[4])
{
    if (deltax <= 0 || deltay <= 0) {
        return;
    }

    /* Translate and scale the picked region to the entire window */
    glTranslatef((viewport[2] - 2 * (x - viewport[0])) / deltax,
        (viewport[3] - 2 * (y - viewport[1])) / deltay, 0);
    glScalef(viewport[2] / deltax, viewport[3] / deltay, 1.0);
}
            """

    if 0 and QTVERSION >= '4.3.0':
        def renderText(self,  x, y, z, text, font = None, listbase = 2000):
            GL.glGetError()
            GL.glGetError()
            _BaseOpenGLWidget.renderText(self, x, y, z, text, font, listbase)
            GL.glGetError()
            GL.glGetError()

    def renderText(self, x, y, z, text, font = None, listbase = 2000):
        if font is None: font=self.font()
        if (QTVERSION < '4.3.2') or (QTVERSION > '4.4.0'):
            _BaseOpenGLWidget.renderText(self, x, y, z, text, font, listbase)
        else:
            if 0:
                GL.glRasterPos3d(x, y, z)
                self.redBookFont.printString(text)
            else:
                GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
                GL.glDisable(GL.GL_TEXTURE_1D)
                GL.glDisable(GL.GL_TEXTURE_2D)
                GL.glDisable(GL.GL_CULL_FACE)

                GL.glRasterPos3d(x, y, z)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glEnable(GL.GL_BLEND)

                GL.glAlphaFunc(GL.GL_GREATER, 0.0)
                GL.glEnable(GL.GL_ALPHA_TEST)
                GL.glListBase(self.redBookFont.fontOffset)
                GL.glCallLists(text)
                GL.glPopAttrib()


    def mousePressEvent(self, event):
        if DEBUG:
            print("pressEvent")
        self.lastPos = qt.QPoint(event.pos())
        #get the color
        x = self.lastPos.x()
        y = self.lastPos.y()
        width  = self.width()
        height = self.height()
        """
        if event.buttons() == qt.Qt.MidButton:
            if GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_SELECT:
                GL.glRenderMode(GL.GL_RENDER)
            else:
                print "setting selection mode"
                GL.glRenderMode(GL.GL_SELECT)
        """
        if event.buttons() & qt.Qt.RightButton:
            if DEBUG:
                print("Right button clicked")
            return
        if self._objectSelectionMode:
            if DEBUG:
                print("in object selection mode")
            y = self.height()- y

            self.makeCurrent()
            #change to selection mode
            # setup a result buffer for GL_SELECT mode
            GL.glSelectBuffer(10000)

            # start GL_SELECT mode : no rendering, just listing objects
            GL.glRenderMode(GL.GL_SELECT)

            #setup a small 5x5 pixel square region
            viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()



            # changed viewport size to 10x10
            w = 10
            self.__test = [x,
                           y,
                           w,
                           w]
            GLU.gluPickMatrix(self.__test[0],
                             self.__test[1],
                             self.__test[2],
                             self.__test[3],
                             viewport)

            # draw the small square (but not visible)
            self.paintGL()

            # returns to usual mode and get the select buffer
            self.selectBuffer = list(GL.glRenderMode(GL.GL_RENDER))

            objectsList = self.scene.getObjectList()
            ddict = {}
            ddict['legend'] = None
            if len(self.selectBuffer):
                # search for the nearest object
                nearest = (1e20,())
                for (near, far, index) in self.selectBuffer:
                    if (near < nearest[0]) and (index != ()):
                        #if index > 0: #do not select the scene???
                            nearest = (near,index)
                    #print "near = ", near, "far = ", far, index
                if DEBUG:
                    print(" Object index   = ", nearest[1][0] - 1)
                if len(nearest[1]):
                    index = nearest[1][0]
                    if index <= len(objectsList):
                        ddict['legend'] = objectsList[index]
            else:
                ddict['legend'] = self.scene.name()

            if DEBUG:
                print("In SceneGLWidget")
                print("current = ", self.scene.getSelectedObject())
                print("selected = ", ddict['legend'])
            if ddict['legend'] == self.scene.getSelectedObject():
                self.__outOfSelectMode = True
            else:
                #Just to draw the proper bounding box ...
                #otherways one could just redraw the texture
                self.__outOfSelectMode = False

            self.scene.setSelectedObject(ddict['legend'])

            if not self.__outOfSelectMode:
                self.sigObjectSelected.emit(ddict)
            else:
                print("no signal")


            if hasattr(self, "doneCurrent"):
                self.doneCurrent()
            qt.QApplication.postEvent(self,
                              qt.QResizeEvent(qt.QSize(width,height),self.size()))
        elif self._vertexSelectionMode:
            #self.setCacheEnabled(False)
            if DEBUG:
                print("vertexSelectionMode")
            selected = False
            legend = self.scene.getSelectedObject()
            if legend is None:
                if DEBUG:
                    print(" NO OBJECT SELECTED")
                return
            else:
                selected = True
            object3D = self.scene[legend].root[0]
            if not object3D.selected():
                if DEBUG:
                    print("%s NOT SELECTED" % legend)
                print("THIS SHOULD NOT HAPPEN")
                return
            if not object3D.isVertexSelectionModeSupported():
                #emit info
                ddict= {}
                ddict['legend'] = legend
                ddict['index'] = None
                txt = "Object %s does not support vertex selection." % legend
                ddict['info'] = txt
                ddict['vertex'] = None
                ddict['value']  = None
                self.sigVertexSelected.emit(ddict)
                return

            #print "glu pro before height correction",GLU.gluUnProject(x, y, 0.0)
            y = self.height()- y
            #print "glu pro after height correction",GLU.gluUnProject(x, y, 0.0)

            #I should try to do all this in paintGL to avoid make current ...
            self.makeCurrent()

            #make sure I am in render mode
            GL.glRenderMode(GL.GL_RENDER)

            #draw only the active object
            lightFlag = GL.glGetBooleanv(GL.GL_LIGHTING)
            try:
                if lightFlag:
                    GL.glDisable(GL.GL_LIGHTING)
                self.__selectingVertex = True
                self.paintGL()
            finally:
                if lightFlag:
                    GL.glEnable(GL.GL_LIGHTING)
                self.__selectingVertex = False
            GL.glFlush()
            #GL.glFinish()

            #this assumes I draw on white background when selecting vertices
            backgroundIndex = 16777215

            # returns to usual mode and get the select buffer
            color = GL.glReadPixelsub(x, y, 1, 1, GL.GL_RGBA)
            #workaround a couple of PyOpenGL bugs
            # sometimes I get an int8 and sometimes a string
            if hasattr(color, "dtype"):
                if color.dtype == 'int8':
                    if DEBUG:
                        print('######### workaround pyopengl bug #########')
                    color = color.astype(numpy.uint8)
            elif hasattr(color, "decode") and (not hasattr(color, "encode")):
                # received a bytes string
                color0 = color
                color = numpy.zeros((1,1,4), dtype=numpy.uint8)
                color[0][0][0] = color0[0]
                color[0][0][1] = color0[1]
                color[0][0][2] = color0[2]
                color[0][0][3] = color0[3]
            else:
                #assume to have received a string
                color0 = color
                color = numpy.zeros((1,1,4), dtype=numpy.uint8)
                color[0][0][0] = ord(color0[0])
                color[0][0][1] = ord(color0[1])
                color[0][0][2] = ord(color0[2])
                color[0][0][3] = ord(color0[3])

            index =  color[0][0][0] + \
                    (color[0][0][1] << 8) +\
                    (color[0][0][2] << 16)

            index += pow(2,24) * (255 - color[0][0][3])
            if index == backgroundIndex:
                searchRegion = range(10)
                for i in searchRegion:
                    if index != backgroundIndex:break
                    for k in range(2):
                        if index != backgroundIndex:break
                        if k == 1:
                            i = -i
                        if (x+i) >= width:continue
                        if (x+i) < 0:    continue
                        for j in searchRegion:
                            if index != backgroundIndex:break
                            for k in range(2):
                                if k == 1:
                                    j = -j
                                if (y+j) >= height:continue
                                if (y+j) < 0:    continue
                                color = GL.glReadPixelsub(x+i, y+j, 1, 1, GL.GL_RGBA)
                                #workaround a couple of PyOpenGL bugs
                                # sometimes I get an int8 and sometimes a string
                                if hasattr(color, "dtype"):
                                    if color.dtype == 'int8':
                                        if DEBUG:
                                            print('######### workaround pyopengl bug #########')
                                        color = color.astype(numpy.uint8)
                                elif hasattr(color, "decode") and (not hasattr(color, "encode")):
                                    # received a bytes string
                                    color0 = color
                                    color = numpy.zeros((1,1,4), dtype=numpy.uint8)
                                    color[0][0][0] = color0[0]
                                    color[0][0][1] = color0[1]
                                    color[0][0][2] = color0[2]
                                    color[0][0][3] = color0[3]
                                else:
                                    #assume to have received a string
                                    color0 = color
                                    color = numpy.zeros((1,1,4), dtype=numpy.uint8)
                                    color[0][0][0] = ord(color0[0])
                                    color[0][0][1] = ord(color0[1])
                                    color[0][0][2] = ord(color0[2])
                                    color[0][0][3] = ord(color0[3])
                                index = color[0][0][0] + \
                                        (color[0][0][1] << 8) + \
                                        (color[0][0][2] << 16)
                                index += pow(2,24) * (255 - color[0][0][3])
                                if index != backgroundIndex:
                                    if DEBUG:
                                        print("found with x, y = ", i, j)
                                        print("color = ", color)
                                        print("index = ", index)
                                        if hasattr(object3D, "getIndexValues"):
                                            print("VERTEX = ",\
                                                 object3D.getIndexValues(index))
                                    break
            if DEBUG:
                if index == backgroundIndex:
                    print("click too far away")
                else:
                    print("INDEX =  ", index)
                    if hasattr(object3D, "getIndexValues"):
                        print("VERTEX = ", object3D.getIndexValues(index))

            #make sure everything is fine ...
            qt.QApplication.postEvent(self,
                    qt.QResizeEvent(qt.QSize(width,height),self.size()))

            #emit info
            ddict= {}
            ddict['legend'] = None
            ddict['index'] = index
            if index == backgroundIndex:
                ddict['info'] ="Clicked too far away"
                ddict['vertex'] = None
                ddict['value']  = None
            else:
                try:
                    values = object3D.getIndexValues(index)
                    ddict['vertex'] = [values[0], values[1], values[2]]
                    ddict['value']  = values[-1]
                    ddict['legend'] = object3D._configuration['common']['name']
                    ddict['info'] = "X = %f   Y = %f   Z = %f   I = %f"  %\
                                    (values[0], values[1], values[2], values[3])
                except:
                    ddict['info'] = "ERROR: %s" % (sys.exc_info()[1])
            self.sigVertexSelected.emit(ddict)

    def mouseReleaseEvent(self, event):
        if DEBUG:
            print("Release event = L", event.button() & qt.Qt.LeftButton)
            print("Release event = M", event.button() & qt.Qt.MidButton)
            print("Release event = R", event.button() & qt.Qt.RightButton)
        if self._objectSelectionMode:
            self.setCacheEnabled(True)
            return
        #This does not work: event.buttons() excludes the button that caused the event
        if event.buttons() & qt.Qt.MidButton:
            pass

        #This does not work: event.buttons() excludes the button that caused the event
        if event.buttons() & qt.Qt.RightButton:
            if DEBUG:
                print("Right button released")
            pass

    def mouseMoveEvent(self, event):
        if event.buttons() & qt.Qt.LeftButton:
            if self._objectSelectionMode:
                return
            if self._vertexSelectionMode:
                return
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & qt.Qt.LeftButton:
            #try to move the scene
            # I need the viewport size and the orthographic limits
            w = self.width()
            h = self.height()
            # I need the orthographic limits
            if dx != 0:
                dx = (dx/float(w)) * (self.__orthoLimits[3]-self.__orthoLimits[0])
            if dy != 0:
                dy = (dy/float(h)) * (self.__orthoLimits[4]-self.__orthoLimits[1])
            #probably the zoom also plays a role
            if SCENE_MATRIX:
                self.__currentViewPosition = self.scene.getCurrentViewMatrix()
            self.__currentViewPosition[3,0] += dx
            self.__currentViewPosition[3,1] -= dy
            self.scene.setCurrentViewMatrix(self.__currentViewPosition)
            self.cacheUpdateGL()
        elif event.buttons() & qt.Qt.RightButton:
            self.setCacheEnabled(False)
            angleX =  0.3*dy
            angleZ =  0.3*dx
            xmin, ymin, zmin, xmax, ymax, zmax = self.scene.getLimits()
            centerX = 0.5 * (xmax + xmin)
            centerY = 0.5 * (ymax + ymin)
            centerZ = 0.5 * (zmax + zmin)
            scale = self.scene.tree.root[0].getConfiguration()['common']['scale']
            anchor = [centerX*scale[0], centerY*scale[1], centerZ*scale[2]]
            M = self.getRotationMatrix(angleX, 0, angleZ, anchor)
            if SCENE_MATRIX:
                self.__currentViewPosition = self.scene.getCurrentViewMatrix()
            self.__currentViewPosition = numpy.dot(M,
                                            self.__currentViewPosition)
            self.scene.setCurrentViewMatrix(self.__currentViewPosition)
            self.cacheUpdateGL()
            viewMatrix = self.scene.getCurrentViewMatrix()
        elif event.buttons() & qt.Qt.MidButton:
            #Z translation
            #in orthographic projection is almost senseless
            h = self.height()
            #the zoom plays a minor role
            #but the scene scale plays a big one
            if dy != 0:
                dy = (dy/float(h)) * (self.__orthoLimits[5]-self.__orthoLimits[2])
                if SCENE_MATRIX:
                    self.__currentViewPosition = self.scene.getCurrentViewMatrix()
                self.__currentViewPosition[3,2] -= dy
                self.scene.setCurrentViewMatrix(self.__currentViewPosition)
                self.cacheUpdateGL()
        else:
            if DEBUG:
                print("I can only be here is mouse tracking is enabled")
            xPixel = event.x()
            yPixel = event.y()
            width  = self.width()
            height = self.height()
            x = xPixel
            y = self.height()- yPixel
            if 0:
                # I am not sure about the value of Z being the correct one
                # but I expect this to be used when in "2D" mode
                z = 0.0
            else:
                # The correct way (NeHe "Using gluUnproject" tutorial)
                # glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z)
                z = GL.glReadPixels(x, int(y), 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
                z = z[0]

            #print numpy.dot(numpy.linalg.inv(self.__sceneModelViewMatrix), xyz)
            #x, y, z, w = numpy.dot(numpy.linalg.inv(self.__sceneModelViewMatrix), xyz)
            ddict = {}
            ddict['event']  = 'mouseMoved'
            ddict['xpixel'] = xPixel
            ddict['ypixel'] = yPixel
            ddict['zpixel'] = z
            view = GL.glGetIntegerv(GL.GL_VIEWPORT)
            glX, glY, glZ   = GLU.gluUnProject(x, y, z,
                                   model=self.__sceneModelViewMatrix,
                                   proj=self.__sceneProjectionMatrix,
                                   view=view)
            ddict['x'] = glX
            ddict['y'] = glY
            ddict['z'] = glZ
            glX, glY, glZ   = GLU.gluUnProject(x, y, z,
                                   model=self.__selectedModelViewMatrix,
                                   proj=self.__selectedProjectionMatrix,
                                   view=view)
            ddict['xselected'] = glX
            ddict['yselected'] = glY
            ddict['zselected'] = glZ
            if DEBUG:
                print("Emitting mouseMoved signal", ddict)
            self.sigMouseMoved.emit(ddict)
        self.lastPos = qt.QPoint(event.pos())

    def cacheUpdateGL(self):
        qt.QApplication.postEvent(self,
                 qt.QResizeEvent(self.size(),self.size()))

    def print3D(self):
        """
        #This worked on Qt3
        printer = qt.QPrinter()
        if printer.setup(None):
            painter = qt.QPainter()
            if not(painter.begin(printer)):
                return 0
            image = GL.glReadPixels(0,0,self.width,self.height,GL.GL_BGRA,opengl.GL_UNSIGNED_BYTE)
            a=Numeric.array(image,'c')
            qimage=qt.QImage(image, self.width, self.height, 32, None, 0, qt.QImage.IgnoreEndian)

            painter.drawImage(0,0,qimage)
            painter.end()
        """
        pass

    def closeEvent(self, event):
        self.__cacheTexture.openGLCleanup()
        self.setCacheEnabled(False)
        _BaseOpenGLWidget.closeEvent(self, event)

if __name__ == '__main__':
    import sys
    import Object3DBase
    app = qt.QApplication(sys.argv)
    class MyObject(Object3DBase.Object3D):
        def drawObject(self):
            #GL.glShadeModel(GL.GL_FLAT)
            GL.glShadeModel(GL.GL_SMOOTH) #in order not to have just blue face
            GL.glBegin(GL.GL_TRIANGLE_STRIP)
            GL.glColor3f(1., 0., 0.)      # Red
            GL.glVertex3f(-25., 0., 0.)
            GL.glColor3f(0., 1., 0.)      # Green
            GL.glVertex3f(25., 0., 0.)
            GL.glColor3f(0., 0., 1.)      # Blue
            GL.glVertex3f(0, 25, 0.)
            GL.glEnd()

    ob3D1 = MyObject()
    ob3D1.setLimits(-25, 0.0, 0.0, 25, 25, 0.0)

    ob3D2 = MyObject()
    ob3D2.setLimits(-25, 0.0, 0.0, 25, 25, 0.0)


    #translate
    config = ob3D2.getConfiguration()
    config['common']['translation'] = [0.0, -25, 0.0]
    ob3D2.setConfiguration(config)

    if 0:
        import SceneWindow
        window = SceneWindow.SceneWindow()
        window.show()
        window.addObject(ob3D1, "Object1")
        window.show()
        sys.exit(app.exec_())


    window = SceneGLWidget()
    window.setWindowTitle('Object3DGLWidget')
    window.scene.setAutoScale(False)
    window.addObject3D(ob3D1, "Object1", plot=False)
    window.addObject3D(ob3D2, "Object2", plot=True)
    window.setObjectSelectionMode(True)
    window.setZoomFactor(1)
    def mySlot(ddict):
        print("Selected = ", ddict['legend'])
        print("Object %s selected" % ddict['legend'])
        if ddict['legend'] in [None, 'Scene']:
            print("Come on! It is not so difficult to hit one triangle")
            window.setZoomFactor(window.getZoomFactor() / 1.1)
        else:
            window.setZoomFactor(window.getZoomFactor() * 1.1)

        print("NEW ZOOM = ", window.getZoomFactor())
    window.sigObjectSelected.connect(mySlot)
    window.show()
    sys.exit(app.exec_())

