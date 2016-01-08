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
import numpy
try:
    import OpenGL.GL  as GL
    from OpenGL.GL import glDeleteLists
except ImportError:
    raise ImportError("OpenGL must be installed to use these functionalities")

try:
    from PyMca5 import spslut
    from PyMca5.PyMcaIO import EdfFile
except ImportError:
    import spslut
    import EdfFile

try:
    from PyMca5.Object3D import Object3DBase
except ImportError:
    from Object3D import Object3DBase

try:
    from PyMca5.Object3D import Object3DCTools
except ImportError:
    try:
        from Object3D import Object3DCTools
    except:
        import Object3DCTools

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui import PyMcaFileDialogs

import weakref

import time
DEBUG = 0

DRAW_MODES = ['NONE',
              'POINT',
              'WIRE',
              'SURFACE']

COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]

class Object3DPixmap(Object3DBase.Object3D):
    def __init__(self, name="Pixmap"):
        Object3DBase.Object3D.__init__(self, name)
        Object3DBase.Object3D.__init__(self, name)

        self.pixmap = None
        self._qt = False
        self._xCalibration = [0.0, 1.0, 0.0]
        self._yCalibration = [0.0, 1.0, 0.0]
        self._zValue = 0.0
        self._imageData = None
        self.textureId = -1
        self.drawList = 0
        self._forceListCalculation = True
        self._forceTextureCalculation = True
        self.__xMirror = False
        self.__yMirror = False  # OpenGL origin is bottom left ...
        self.__widthStep  = 1.0
        self.__heightStep = 1.0
        self._linear = True
        self.pointSelectionGridList = 0
        self._forcePointSelectionGridListCalculation = False

        #draw points by default
        self._configuration['common']['mode'] = 1
        #centered on XY plane and on Z
        self._configuration['common']['anchor'] = [2, 2, 2]

    def setConfiguration(self, ddict):
        Object3DBase.Object3D.setConfiguration(self, ddict)
        if self._qt:
            return
        if self._imageData is None:
            return
        if 'event' in ddict['common']:
            if ddict['common']['event'] == 'ColormapChanged':
                self._forceTextureCalculation = True
        if not self._forceTextureCalculation:
            return
        colormap = self._configuration['common']['colormap']

        #avoid recalculating min and max values
        if colormap[1]:
            vMin = colormap[4]
            vMax = colormap[5]
        else:
            vMin = colormap[2]
            vMax = colormap[3]
        if not self._meshImage:
            (pixmap,size,minmax)= spslut.transform(self._imageData,
                                              (1,0),
                                              (colormap[6],3.0),
                                              "RGBX",
                                              COLORMAPLIST[int(str(colormap[0]))],
                                              0,
                                              (vMin, vMax),
                                              (0, 255),1)
            width = size[0]
            height = size[1]
            pixmap.shape = -1, 4
            tjump = self.__tWidth
            pjump = self.__width
            for i in range(height):
                self.pixmap[i*tjump:(i*tjump+pjump), :] = pixmap[(i*pjump):(i+1)*pjump,:]
        else:
            (pixmap,size,minmax)= spslut.transform(self._imageData.T,
                                              (1,0),
                                              (colormap[6],3.0),
                                              "RGBX",
                                              COLORMAPLIST[int(str(colormap[0]))],
                                              0,
                                              (vMin, vMax),
                                              (0, 255),1)
            pixmap.shape = -1, 4
            self.pixmap[:,0:3] = pixmap[:,0:3]
        self.pixmap[:,3] = 255
        return

    def setQPixmap(self, qpixmap):
        if not isinstance(qpixmap, qt.QPixmap):
            raise TypeError("This does not seem to be a QPixmap")
        qimage = qpixmap.toImage()
        return self.setQImage(qimage)

    def setQImage(self, qimage):
        """
        QImage
        """
        if not isinstance(qimage, qt.QImage):
            raise TypeError("This does not seem to be a QImage")
        height = qimage.height()
        width  = qimage.width()
        image = qimage.convertToFormat(qt.QImage.Format_ARGB32)
        pixmap = numpy.fromstring(qimage.bits().asstring(width * height * 4),
                             dtype = numpy.uint8)
        self._qt = True
        return self.setPixmap(pixmap, width, height,
                           xmirror=False, ymirror=True)

    def setPixmap(self, pixmap, width, height, xmirror = False, ymirror = False, z=0.0):
        """
        spslut string output
        """
        if type(pixmap) == type(""):
            raise ValueError("Input pixmap has to be an uin8 array")

        self._imageData = None
        self._meshImage = False
        self._forceTextureCalculation = True

        #make sure we work with integers
        self.__width  = int(width)
        self.__height = int(height)

        # some cards still need to pad to powers of 2
        self.__tWidth  = self.getPaddedValue(self.__width)
        self.__tHeight = self.getPaddedValue(self.__height)
        maximum_texture = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)
        #print "MAXIMUM  = ", maximum_texture
        if (self.__tWidth > maximum_texture) or (self.__tHeight > maximum_texture):
            raise ValueError("Invalid final texture size: %d x %d" % (self.__tWidth,
                                                                 self.__tHeight))

        if (self.__tWidth != self.__width) or (self.__tHeight != self.__height):
            #I have to zero padd the texture to make sure it works on all cards ...
            self.pixmap = numpy.zeros((self.__tWidth*self.__tHeight, 4), numpy.uint8)
            pixmap.shape = [width*height, 4]
            tjump = self.__tWidth
            pjump = self.__width
            for i in range(height):
                self.pixmap[i*tjump:(i*tjump+pjump), :] = pixmap[(i*pjump):(i+1)*pjump,:]
        else:
            self.pixmap = pixmap
            self.pixmap.shape = [width*height, 4]
        self.pixmap[:,3] = 255 #alpha
        self.__xMirror = xmirror
        self.__yMirror = ymirror
        self.zPosition = z
        aspectRatio = False
        if aspectRatio:
            if height > width:
                delta = 0.5*(height - width)
                self.setLimits(-delta, 0.0, self.zPosition,
                           width+delta, height,
                           self.zPosition)
            else:
                delta = 0.5*(width - height)
                self.setLimits(0.0, -delta, self.zPosition,
                           width, height+delta,
                           self.zPosition)
        else:
            self.setLimits(0.0, 0.0, self.zPosition,
                       width, height, self.zPosition)

    def setLimits(self, *var):
        Object3DBase.Object3D.setLimits(self, *var)
        if DEBUG:
            t0 = time.time()
        self._buildPointSelectionVertices()
        if DEBUG:
            print("POINT SELECTION ELAPSED = ", time.time() - t0)
        self._forcePointSelectionGridListCalculation = True

    def setImage(self, *var, **kw):
        return self.setPixmap(*var, **kw)


    def updateImageData(self, data):
        if self._imageData is None:
            return self.setImageData(data)
        if (self._imageData.shape[0] != data.shape[0]) or\
           (self._imageData.shape[1] != data.shape[1]):
            return self.setImageData(data)
        self._imageData = data
        self._dataMin = data.min()
        self._dataMax = data.max()
        self._configuration['common']['colormap'][4]=self._dataMin
        self._configuration['common']['colormap'][5]=self._dataMax
        ddict = {'common':{'event':'ColormapChanged'}}
        self.setConfiguration(ddict)

    def setImageData(self, data):
        """
        setImageData(self, data)
        data is a numpy array
        """
        self._qt = False
        maxTextureSize = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)
        shape = data.shape
        self._dataMin = data.min()
        self._dataMax = data.max()
        if (shape[0] > maxTextureSize) or\
           (shape[1] > maxTextureSize):
            #very slow
            self._imageData = data.astype(numpy.float32)
            self._meshImage = True
            #self._imageData = data.astype(numpy.float32)
            self.__width  = self._imageData.shape[1]
            self.__height = self._imageData.shape[0]
            self.zPosition = 0.0
            self._xValues = numpy.arange(self.__width).astype(numpy.float32)
            self._yValues = numpy.arange(self.__height).astype(numpy.float32)
            self._zValues  = numpy.zeros(self._imageData.shape, numpy.float32)
            self.setLimits(0.0, 0.0, self.zPosition,
                       self.__width-1, self.__height-1, self.zPosition)
            (image,size,minmax)= spslut.transform(self._imageData.T, (1,0),
                                      (spslut.LINEAR,3.0),
                                      "RGBX", spslut.TEMP,
                                       0,
                                      (self._dataMin, self._dataMax),
                                      (0, 255), 1)
            self.pixmap = image
            self.pixmap.shape = -1, 4
            self.pixmap[:,3]  = 255
        else:
            self._meshImage = False
            (image,size,minmax)= spslut.transform(data, (1,0),
                                      (spslut.LINEAR,3.0),
                                      "RGBX", spslut.TEMP,
                                       0,
                                      (self._dataMin, self._dataMax),
                                      (0, 255), 1)
            self.setPixmap(image,
                           size[0],
                           size[1],
                           xmirror = False,
                           ymirror = False)
            self._imageData = data
        self._configuration['common']['colormap'][2]=self._dataMin
        self._configuration['common']['colormap'][3]=self._dataMax
        self._configuration['common']['colormap'][4]=self._dataMin
        self._configuration['common']['colormap'][5]=self._dataMax

    def getPaddedValue(self, v):
        a = 2
        while (a<v):
            a *=2
        return int(a)

    def __mirrorRotator(self, xMirror, yMirror):
        if yMirror or xMirror:
            if yMirror:
                rotX = numpy.zeros((4,4), numpy.float64)
                rotX[0,0] =  1
                rotX[1,1] =  1
                rotX[2,2] =  1
                rotX[3,3] =  1
                cs = -1.0
                sn =  0.0
                rotX[1,1] =  cs; rotX[1,2] = sn
                rotX[2,1] = -sn; rotX[2,2] = cs
                GL.glMultMatrixd(rotX)
                GL.glTranslated(0.0, -self.__height, 0.0)

            if xMirror:
                #RotY
                cs = -1.0
                sn = 0.0
                rotY = numpy.zeros((4,4), numpy.float64)
                rotY[0,0] =  1
                rotY[1,1] =  1
                rotY[2,2] =  1
                rotY[3,3] =  1
                rotY[0,0] =  cs; rotY[0,2] = -sn   #inverted respect to the others
                rotY[2,0] =  sn; rotY[2,2] =  cs
                GL.glMultMatrixd(rotY)
                GL.glTranslated(-self.__width, 0.0, 0.0)

    def drawObject(self):
        if DEBUG:e0=time.time()
        if self.pixmap is None:
            if self._imageData is None:
                return

        if self.__xMirror or self.__yMirror:
            self.__mirrorRotator(self.__xMirror, self.__yMirror)

        if self._vertexSelectionMode:
            if (self.pointSelectionGridList < 0 ) or self._forcePointSelectionGridListCalculation:
                self.buildPointSelectionGridList()

            GL.glTranslate(0.5 * self.__widthStep, 0.5 * self.__heightStep, 0.0)
            # drawing lines the height is not a problem but the points are displaced
            # and I would have to add one point ...
            GL.glCallList(self.pointSelectionGridList)
            GL.glTranslate(-0.5 * self.__widthStep, -0.5 * self.__heightStep, 0.0)
        else:
            if self._meshImage:
                self._drawMesh()
            else:
                #draw pixmap
                if self._forceTextureCalculation:
                    self.buildTexture()
                GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
                GL.glDisable(GL.GL_DEPTH_TEST)
                alpha = 1.0 - self._configuration['common']['transparency']
                GL.glColor4f(1.0, 1.0, 1.0, alpha)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                if (self.drawList <= 0) or self._forceListCalculation:
                    self.buildQuad()
                GL.glCallList(self.drawList)
                GL.glPopAttrib()
                """
                if (self.pointSelectionGridList < 0 ) or self._forcePointSelectionGridListCalculation:
                    self.buildPointSelectionGridList()
                GL.glTranslate(0.5 * self.__widthStep, 0.5 * self.__heightStep, 0.0)
                GL.glCallList(self.pointSelectionGridList)
                GL.glTranslate(-0.5 * self.__widthStep, -0.5 * self.__heightStep, 0.0)
                """

        if DEBUG:
            print("elapsed = ", time.time() - e0)

    def _drawMesh(self):
        alpha = 1.0 - self._configuration['common']['transparency']
        if alpha < 0:
            alpha = 0
        elif alpha >= 1.0:
            alpha = 255
        else:
            alpha = int(255 * alpha)
        self.pixmap[:, 3] = alpha
        shape = self._imageData.shape
        self._imageData.shape = -1,1
        if DRAW_MODES[self._configuration['common']['mode']] == "POINT":
            if False:
                Object3DCTools.drawXYZPoints(self.vertices,
                                         self.pixmap)
            else:
                Object3DCTools.draw2DGridPoints(self._xValues,
                           self._yValues,
                           self._zValues,
                           self.pixmap)
        elif DRAW_MODES[self._configuration['common']['mode']] == "WIRE":
            Object3DCTools.draw2DGridLines(self._xValues,
                           self._yValues,
                           self._zValues,
                           self.pixmap)
        elif DRAW_MODES[self._configuration['common']['mode']] == "SURFACE":
            Object3DCTools.draw2DGridQuads(self._xValues,
                           self._yValues,
                           self._zValues,
                           self.pixmap)
        self._imageData.shape = shape

    def buildTexture(self):
        """
        Normal procedure:
        Generate a texture name with glGenTextures.
        Select the new texture with glBindTexture.
        Fill the texture with an image using glTexImage2D.
        Set the texture's minification and magnification filters using glTexParameteri.
        Enable 2D textures with glEnable.
        When producing geometry, bind the texture before starting the polygon and then
        set a texture coordinate with glTexCoord before each glVertex.
        """
        if self.textureId >= 0:
            #I should use texture subimage whenever possible ...
            #glDeleteTextures(GLsizei n, const GLuint *textureNames);
            GL.glDeleteTextures(self.textureId)
        self.textureId = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureId)
        GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT )
        GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT )
        linear = self._linear
        if linear:
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST )
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR )
        else:
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST )
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST )
        if self._qt:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                        self.__tWidth,
                        self.__tHeight,
                        0, GL.GL_BGRA,
                        GL.GL_UNSIGNED_BYTE, self.pixmap)
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                        self.__tWidth,
                        self.__tHeight,
                        0, GL.GL_RGBA,
                        GL.GL_UNSIGNED_BYTE, self.pixmap)
        GL.glEnable(GL.GL_TEXTURE_2D)
        self._forceTextureCalculation = False

    def buildQuad(self):
        if self.drawList > 0:
            GL.glDeleteLists(self.drawList, 1)
        aspectRatio = False
        if aspectRatio:
            xmin, ymin, zmin = 0.0, 0.0, self.zPosition
            xmax, ymax, zmax = self.__width, self.__height, self.zPosition
        else:
            xmin, ymin, zmin = self._limits[0]
            xmax, ymax, zmax = self._limits[1]
        tx0 = 0.0
        tx1 = (1.0 * self.__width)/self.__tWidth
        ty0 = 0.0
        ty1 = (1.0 * self.__height)/self.__tHeight
        self.drawList = GL.glGenLists(1)
        GL.glNewList(self.drawList, GL.GL_COMPILE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textureId)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2d(tx0, ty0)
        GL.glVertex3f(xmin, ymin, zmin)
        GL.glTexCoord2d(tx0, ty1)
        GL.glVertex3f(xmin, ymax, zmin)
        GL.glTexCoord2d(tx1, ty1)
        GL.glVertex3f(xmax, ymax, zmin)
        GL.glTexCoord2d(tx1, ty0)
        GL.glVertex3f(xmax, ymin, zmin)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEndList()
        self._forceListCalculation = False

    def isVertexSelectionModeSupported(self):
        return True

    def _buildPointSelectionVertices(self):
        # I assume self.__width * self.__height points
        # distributed in the limits interval
        xsize = self.__width
        ysize = self.__height

        xmin, ymin, zmin = self._limits[0]
        xmax, ymax, zmax = self._limits[1]

        self.__widthStep  = (xmax - xmin)/xsize
        self.__heightStep = (ymax - ymin)/ysize

        x = xmin + numpy.arange(xsize) * self.__widthStep
        y = ymin + numpy.arange(ysize) * self.__heightStep

        if DEBUG:e0 = time.time()
        self.vertices = numpy.zeros((xsize * ysize, 3), numpy.float32)
        if 0:
            #fast method to generate the vertices
            A=numpy.outer(x, numpy.ones(len(y), numpy.float32))
            B=numpy.outer(y, numpy.ones(len(x), numpy.float32))

            self.vertices[:,0]=A.flatten()
            self.vertices[:,1]=B.transpose().flatten()
            self.vertices[:,2]=0.5 * (zmax - zmin)
        else:
            #this is faster
            A, B = numpy.meshgrid(y.astype(numpy.float32), x.astype(numpy.float32))
            A.shape = -1
            B.shape = -1
            self.vertices[:,0]=B
            self.vertices[:,1]=A
        self.zdata = self.vertices[:,2]
        if DEBUG:
            print("vertex generation elapsed = ", time.time()-e0)

        #get the associated selection colors
        if DEBUG:
            e0 = time.time()
        i = numpy.arange(len(self.vertices))
        self.__useUInt8 = True
        if self.__useUInt8:
            self.vertexSelectionColors = numpy.zeros((len(self.vertices),3), numpy.uint8)
            self.vertexSelectionColors[:,0] = (i & 255)
            self.vertexSelectionColors[:,1] = ((i >> 8) & 255)
            self.vertexSelectionColors[:,2] = ((i >> 16) & 255)
        else:
            self.vertexSelectionColors = numpy.zeros((len(self.vertices),3), numpy.float32)
            self.vertexSelectionColors[:,0] = (i & 255)/255.
            self.vertexSelectionColors[:,1] = ((i >> 8) & 255)/255.
            self.vertexSelectionColors[:,2] = ((i >> 16) & 255)/255.
        if DEBUG:
            print("vertex selection color elapsed = ", time.time()-e0)

    def buildPointSelectionGridList0(self):
        #in fact I am using a line selection
        if self.pointSelectionGridList > 0:
            GL.glDeleteLists(self.pointSelectionGridList, 1)
        self.pointSelectionGridList = GL.glGenLists(1)
        GL.glNewList(self.pointSelectionGridList, GL.GL_COMPILE)
        GL.glVertexPointerf(self.vertices)
        GL.glColorPointerf(self.vertexSelectionColors)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        for j in range(self.__width):
            GL.glDrawArrays(GL.GL_LINE_STRIP, j*self.__height, self.__height)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEndList()
        self._forcePointSelectionGridListCalculation = False

    def buildPointSelectionGridList(self):
        if self.pointSelectionGridList > 0:
            GL.glDeleteLists(self.pointSelectionGridList, 1)

        self.pointSelectionGridList = GL.glGenLists(1)
        GL.glNewList(self.pointSelectionGridList, GL.GL_COMPILE)
        GL.glVertexPointerf(self.vertices)
        if self.__useUInt8:
            GL.glColorPointerub(self.vertexSelectionColors)
        else:
            GL.glColorPointerf(self.vertexSelectionColors)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.__width*self.__height)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY);
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY);
        GL.glEndList()
        self._forcePointSelectionGridListCalculation = False

    def getIndexValues(self, index):
        """
        x,y,z, I
        """
        if DEBUG:
            print("INDEX = ",index)
            print("Width, Height =", self.__width, self.__height)
        xindex = int(index/self.__height)
        yindex = index % (self.__height)
        xvalue = self._xCalibration[0] +\
                 self._xCalibration[1] * xindex+\
                 self._xCalibration[2] * xindex * xindex
        yvalue = self._yCalibration[0] +\
                 self._yCalibration[1] * yindex+\
                 self._yCalibration[2] * yindex * yindex
        try:
            z = self._zValue[index]
        except TypeError:
            z = self._zValue
        if self._imageData is None:
            return xvalue, yvalue, z, z
        else:
            return xvalue, yvalue, z, self._imageData[yindex, xindex]


MENU_TEXT = 'Pixmap'
def getObject3DInstance(config=None):
    fileTypeList = ['Picture Files (*jpg *jpeg *tif *tiff *png)',
                    'EDF Files (*edf)',
                    'EDF Files (*ccd)',
                    'ADSC Files (*img)',
                    'EDF Files (*)']
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select one object data file",
        mode="OPEN",
        getfilter=True)
    if not len(fileList):
        return
    fname = fileList[0]
    if filterUsed.split()[0] == "Picture":
        qimage = qt.QImage(fname)
        if qimage.isNull():
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot read file %s as an image" % fname)
            msg.exec_()
            return
        object3D = Object3DPixmap(os.path.basename(fname))
        object3D.setQImage(qimage)
        return object3D
    if filterUsed.split()[0] in ["EDF", "ADSC"]:
        edf = EdfFile.EdfFile(fname)
        data = edf.GetData(0)
        if True:
            object3D = Object3DPixmap(os.path.basename(fname))
            object3D.setImageData(data)
        else:
            (image,size,minmax)= spslut.transform(data, (1,0),
                                      (spslut.LINEAR,3.0),
                                      "RGBX", spslut.TEMP,
                                      1,
                                      (0, 1),
                                      (0, 255), 1)
            object3D = Object3DPixmap(os.path.basename(fname))
            object3D.setPixmap(image, size[0], size[1], xmirror = False, ymirror = False)
        return object3D
    return None


if __name__ == "__main__":
    import sys
    import os
    from PyMca5.Object3D import SceneGLWindow

    app = qt.QApplication(sys.argv)
    window = SceneGLWindow.SceneGLWindow()
    window.show()
    object3D=getObject3DInstance()
    if object3D is None:
        name = "125 rows x 80 columns array"
        data = numpy.arange(10000.).astype(numpy.float32)
        data.shape = [125, 80]
        data[120:125, 70:80]  = 0
        (image,size,minmax)= spslut.transform(data, (1,0),
                                      (spslut.LINEAR,3.0),
                                      "RGBX", spslut.TEMP,
                                      1,
                                      (0, 1),
                                      (0, 255), 1)
        object3D = Object3DPixmap(os.path.basename(name))
        object3D.setPixmap(image, size[0], size[1],
                           xmirror = False,
                           ymirror = False)
        object3D.setImageData(data)
    window.addObject(object3D)
    window.glWidget.setZoomFactor(1.0)
    window.show()
    while(0):
        time.sleep(0.01)
        data = numpy.random.random((2048, 2048))*1000
        #data = data.astype(numpy.float32)
        data = data.astype(numpy.int16)
        #data = data.astype(numpy.int8)
        t0 = time.time()
        object3D.updateImageData(data)
        window.glWidget.setZoomFactor(1.0)
        print("elapsed = ",  time.time() - t0)
        app.processEvents()
    object3D = None
    app.exec_()

