#/*##########################################################################
# Copyright (C) 2004-2019 V.A. Sole, European Synchrotron Radiation Facility
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
import logging
import numpy
import weakref
import sys
from . import Object3DQt as qt
from . import Object3DPrivateConfig

try:
    import OpenGL.GL  as GL
except ImportError:
    raise ImportError("OpenGL must be installed to use these functionalities")

_logger = logging.getLogger(__name__)


DRAW_MODES = ['NONE',
              'POINT',
              'WIRE',
              'SURFACE']
              #'LIGHT',
              #'POINT_SELECTION']

class Object3D(object):
    def __init__(self, name = "Object3D"):
        self._configuration ={}

        #default limits
        self._limits = numpy.zeros((2,3), numpy.float32)
        self._limits[1,0] = 1.0
        self._limits[1,1] = 1.0
        self._limitsChanged = False
        self.__name = name

        #the possible private configuration widget
        self._privateConfigurationWidget = None

        self.initCommonConfiguration(name)
        self.initPrivateConfiguration(name)

        # the object should know if it is the active object
        self._selected = False

        # vertex selection mode
        self._vertexSelectionMode = False

        #bounding box gl list
        self.boundingBoxList = 0

        #bounding box indices
        #Bottom XY plane
        self._bottomPlane = numpy.zeros(5).astype(numpy.uint32)
        self._bottomPlane[0] = 0
        self._bottomPlane[1] = 2
        self._bottomPlane[2] = 6
        self._bottomPlane[3] = 4
        self._bottomPlane[4] = 0

        #Top XY plane
        self._topPlane = numpy.zeros(5).astype(numpy.uint32)
        self._topPlane[0] = 1
        self._topPlane[1] = 3
        self._topPlane[2] = 7
        self._topPlane[3] = 5
        self._topPlane[4] = 1

        #Parallel edges
        self._edges = numpy.arange(8).astype(numpy.uint32)


    def __del__(self):
        _logger.debug("DELETING Object3d base")
        if GL is not None:
            if self.boundingBoxList != 0:
                GL.glDeleteLists(self.boundingBoxList, 1)
        _logger.debug("%s DELETED", self.name())

    def name(self):
        return self.__name

    def isVertexSelectionModeSupported(self):
        return False

    def setVertexSelectionMode(self, flag):
        # This is to tell the widget the application is trying
        # to get information about a vertex.
        self._vertexSelectionMode = flag

    def initCommonConfiguration(self, name):
        """
        Fills the default configuration features
        found in all objects
        """

        ddict = {}

        ddict['name'] = name

        ddict['pointsize']    = 1.0     #always supported
        ddict['linewidth']    = 1.0     #always supported
        ddict['transparency'] = 0.0     #solid color

        #scaling parameters
        ddict['scale']        = [1.0, 1.0, 1.0]
        ddict['scalefactor']  = 1.0

        #default anchoring is the origin
        ddict['anchor']       = [0, 0, 0] #warning:these are flags

        #translation
        ddict['translation']  = [0.0, 0.0, 0.0]

        #rotation in (degrees)
        ddict['rotation']     = [ 0.0, 0.0, 0.0]

        #drawing modes
        ddict['drawingmodes']   = DRAW_MODES

        #current mode
        ddict['mode'] = 0

        #supported modes
        supportedModes = []
        for i in range(len(DRAW_MODES)):
            supportedModes.append(i)
        supportedModes [0] = 1          #no drawing is always possible
        ddict['supportedmodes'] = supportedModes

        #show bounding box
        ddict['bboxflag']  = 0

        #limits
        ddict['showlimits'] = [0, 0, 0]

        #clipping planes
        #clipping planes have the form [flag, A, B, C, D]
        ddict['clippingplanes'] = [[0, 0.0, 0.0, 1.0, 0.0], #XY
                                   [0, 0.0, 1.0, 0.0, 0.0], #XZ
                                   [0, 1.0, 0.0, 0.0, 0.0], #YZ
                                   [0, 0.0, 0.0, 1.0, 0.0]] #U0


        ddict['limits'] = self.getLimits()

        ddict['colormap'] = [2, True, 0, 1, -10, 10, 0]

        self._configuration['common'] = ddict

    def initPrivateConfiguration(self, name):
        """
        Specific configuration. To be overwitten
        """
        self._configuration['private'] = {}
        if self._privateConfigurationWidget is None:
            self._privateConfigurationWidget = Object3DPrivateConfig.Object3DPrivateConfig(None,
                                                                                       name=name)
        self._configuration['private']['widget'] = weakref.proxy(self._privateConfigurationWidget)

    def getConfiguration(self):
        return self._configuration

    def setConfiguration(self, ddict):
        """
        You will very likely overwrite this method
        """
        if "common" in ddict:
            self._configuration['common'].update(ddict['common'])
        if "private" in ddict:
            #do not overwrite widget!!!
            widget = self._configuration['private']['widget']
            self._configuration['private'].update(ddict['private'])
            if widget is not None:
                #restore former widget
                self._configuration['private']['widget'] = widget

    def setSelected(self, flag = True):
        self._selected = flag

    def selected(self):
        return self._selected

    def getLimits(self):
        """
        This method returns the limits of the object
        Typically will be its bounding box.
        The form is a 2 row numpy array:
        [[xmin, ymin, zmin],
         [xmax, ymax, zmax]]
        """
        return self._limits

    def setLimits(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self._limits[0,:] = xmin, ymin, zmin
        self._limits[1,:] = xmax, ymax, zmax
        self._limitsArray = numpy.zeros((8,3), numpy.float32)
        i = 0
        for x in [xmin, xmax]:
            for y in [ymin, ymax]:
                for z in [zmin, zmax]:
                    self._limitsArray[i, 0] = x
                    self._limitsArray[i, 1] = y
                    self._limitsArray[i, 2] = z
                    i += 1

        self._limitsChanged = True
        self._configuration['common']['limits'] = self._limits

    def draw(self):
        """
        This is the method called to perform all the openGL stuff.
        The default implementation calls drawObject.
        If the objects is selected, it also calls drawBoundingBox.
        Perhaps, you should consider overwriting just drawObject in
        your application.
        """
        self.enableClippingPlanes()
        self.drawObject()
        self.disableClippingPlanes()
        #Default implementattion:
        #A selected object draws its bounding box
        #unless we are selecting but in that case it does not get here.
        if self._selected:
            if not self._vertexSelectionMode:
                self.drawBoundingBox()

    def enableClippingPlanes(self):
        plane =  self._configuration['common']['clippingplanes']
        GL.glClipPlane(GL.GL_CLIP_PLANE0, plane[0][1:])
        GL.glClipPlane(GL.GL_CLIP_PLANE1, plane[1][1:])
        GL.glClipPlane(GL.GL_CLIP_PLANE2, plane[2][1:])
        GL.glClipPlane(GL.GL_CLIP_PLANE3, plane[3][1:])
        if plane[0][0]:
            GL.glEnable(GL.GL_CLIP_PLANE0)
        if plane[1][0]:
            GL.glEnable(GL.GL_CLIP_PLANE1)
        if plane[2][0]:
            GL.glEnable(GL.GL_CLIP_PLANE2)
        if plane[3][0]:
            GL.glEnable(GL.GL_CLIP_PLANE3)

    def disableClippingPlanes(self):
        plane =  self._configuration['common']['clippingplanes']
        GL.glDisable(GL.GL_CLIP_PLANE0)
        GL.glDisable(GL.GL_CLIP_PLANE1)
        GL.glDisable(GL.GL_CLIP_PLANE2)
        GL.glDisable(GL.GL_CLIP_PLANE3)

    def drawObject(self):
        pass

    def drawCoordinates(self):
        if self._coordinates is not None:
            self._coordinates.draw()

    def drawBoundingBox(self):
        if 0:
            self.drawBoundingBoxNew()
        else:
            self.drawBoundingBoxOld()

    def drawBoundingBoxNew(self):
        #should I consider the alpha?
        alpha = 1. - self._configuration['common']['transparency']
        GL.glColor4f(0., 0., 0., alpha)
        GL.glVertexPointerf(self._limitsArray)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDrawElements(GL.GL_LINE_STRIP,
                          5,
                          GL.GL_UNSIGNED_INT,
                          self._bottomPlane)
        GL.glDrawElements(GL.GL_LINE_STRIP,
                          5,
                          GL.GL_UNSIGNED_INT,
                          self._topPlane)
        GL.glDrawElements(GL.GL_LINES,
                          8,
                          GL.GL_UNSIGNED_INT,
                          self._edges)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def drawBoundingBoxOld(self):
        if self.boundingBoxList == 0:
            self.buildBoundingBoxList()
        else:
            if self._limitsChanged:
                #the old bounding box list is not valid
                GL.glDeleteLists(self.boundingBoxList, 1)
                self.buildBoundingBoxList()

        #should I consider the alpha?
        alpha = 1. - self._configuration['common']['transparency']
        GL.glColor4f(0., 0., 0., alpha)
        GL.glCallList(self.boundingBoxList)

    def buildBoundingBoxList(self):
        (xmin, ymin, zmin) = self._limits[0,:]
        (xmax, ymax, zmax) = self._limits[1,:]

        self.boundingBoxList = GL.glGenLists(1)
        GL.glColor3f(0.5, 0.0, 0.0)

        # Bottom XY plane
        GL.glNewList(self.boundingBoxList, GL.GL_COMPILE)
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f(xmin, ymin, zmin)
        GL.glVertex3f(xmax, ymin, zmin)
        GL.glVertex3f(xmax, ymax, zmin)
        GL.glVertex3f(xmin, ymax, zmin)
        GL.glVertex3f(xmin, ymin, zmin)
        GL.glEnd()

        # Top XY plane
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f(xmin, ymin, zmax)
        GL.glVertex3f(xmax, ymin, zmax)
        GL.glVertex3f(xmax, ymax, zmax)
        GL.glVertex3f(xmin, ymax, zmax)
        GL.glVertex3f(xmin, ymin, zmax)
        GL.glEnd()

        # Parallel edges
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(xmin, ymin, zmin)
        GL.glVertex3f(xmin, ymin, zmax)

        GL.glVertex3f(xmax, ymin, zmin)
        GL.glVertex3f(xmax, ymin, zmax)

        GL.glVertex3f(xmin, ymax, zmin)
        GL.glVertex3f(xmin, ymax, zmax)

        GL.glVertex3f(xmax, ymax, zmin)
        GL.glVertex3f(xmax, ymax, zmax)

        GL.glEnd()
        #self.buildTicks()
        GL.glEndList()


    def buildTicks(self):
        (xmin, ymin, zmin) = self._limits[0,:]
        (xmax, ymax, zmax) = self._limits[1,:]
        xdelta = self.getTickDelta(xmin, xmax)
        ydelta = self.getTickDelta(ymin, ymax)
        zdelta = self.getTickDelta(zmin, zmax)
        p = 30.
        xTickSize = (xmax - xmin)/p
        yTickSize = (ymax - ymin)/p
        zTickSize = (zmax - zmin)/p
        if xdelta > 0:
            GL.glBegin(GL.GL_LINES)
            x = xmin + xdelta
            while x <= xmax:
                GL.glVertex3f(x, ymin, zmin)
                GL.glVertex3f(x, ymin + yTickSize, zmin)
                GL.glVertex3f(x, ymax, zmin)
                GL.glVertex3f(x, ymax - yTickSize, zmin)
                x += xdelta
            GL.glEnd()

        if ydelta > 0:
            GL.glBegin(GL.GL_LINES)
            y = ymin + ydelta
            while y <= ymax:
                GL.glVertex3f(xmin, y, zmin)
                GL.glVertex3f(xmin + xTickSize, y, zmin)
                GL.glVertex3f(xmax, y, zmin)
                GL.glVertex3f(xmax - xTickSize, y, zmin)
                y += ydelta
            GL.glEnd()

        if zdelta > 0:
            GL.glBegin(GL.GL_LINES)
            z = zmin + zdelta
            while z <= zmax:
                GL.glVertex3f(xmin, ymin, z)
                GL.glVertex3f(xmin + xTickSize, ymin, z)
                GL.glVertex3f(xmax, ymin, z)
                GL.glVertex3f(xmax - xTickSize, ymin, z)

                GL.glVertex3f(xmin, ymin, z)
                GL.glVertex3f(xmin, ymin + yTickSize, z)
                GL.glVertex3f(xmin, ymax, z)
                GL.glVertex3f(xmin, ymax - yTickSize, z)
                z += zdelta
            GL.glEnd()

    def getTickDelta(self, minval0, maxval0, nticks = 5):
        if minval0 < 0:
            minval = -minval0
        else:
            minval = minval0
        if maxval0 < 0:
            maxval = -maxval0
        else:
            maxval = maxval0
        if minval > maxval:
            temp = minval * 1.0
            minval = maxval * 1.0
            maxval = temp
        #both are positive now
        delta = maxval - minval
        if delta <= 0:
            stepInterval = 1
        else:
            stepInterval = pow(10.0, int(numpy.log10(maxval - minval))+1)
        finalInterval = stepInterval * 1
        ticks = (maxval - minval)/ stepInterval
        counter = 0
        divider = 1.0
        while ticks < nticks:
            if (counter % 3) == 0:
                divider = 2.
            elif (counter % 3) == 1:
                divider = 5.
            elif (counter % 3) == 2:
                divider = 1.
                stepInterval /= 10.
            counter += 1
            finalInterval = stepInterval/divider
            ticks = (maxval-minval)/ finalInterval
        return finalInterval

    def getIndexValues(self, index):
        """
        To be overwritten.
        Expected to give back x, y, z, I for index
        """
        return None, None, None, None

    def updatePrivateConfigurationWidget(self):
        # This rarely be called from here unless for initialization.
        # The graphic interface should deal with the configuration.
        if hasPrivateConfigurationWidget():
            return self._privateConfigurationWidget.setConfiguration(self._configuration)
        else:
            return True

    def hasPrivateConfigurationWidget(self):
        if self._privateConfigurationWidget is not None:
            return True
        else:
            return False

def getObject3DInstance(config=None):
    return Object3D()

if __name__ == "__main__":
    app = qt.QApplication(sys.argv)
    name = "Base 3D-Object"
    object3D = Object3D(name)
    object3D.setLimits(10.0, 10.0, 10., 30., 30., 30)
    object3D.setSelected(1) #otherways we'll see nothing
    if 0:
        import SceneGLWidget
        window = SceneGLWidget.SceneGLWidget()
        window.addObject3D(object3D, name)
        window.show()
    else:
        import SceneGLWindow
        window = SceneGLWindow.SceneGLWindow()
        #Needed to initialize SceneGLWidget
        window.show()
        window.addObject(object3D, name, update_scene=False)
        #window.glWidget.setZoomFactor(window.glWidget.getZoomFactor())
        window.glWidget.setZoomFactor(0.9)
    app.exec_()
