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
try:
    import OpenGL.GL  as GL
except ImportError:
    raise ImportError("OpenGL must be installed to use these functionalities")
from . import Object3DQt as qt
import numpy
import logging

_logger = logging.getLogger(__name__)


class GLWidgetCachePixmap(object):
    def __init__(self, name="Unnamed"):
        self.__name = name
        self.__pixmap = None
        self.__textureId = None
        self.drawList = 0
        self.__alpha = 1.0
        self._limits = numpy.zeros((2,3), numpy.float32)

    def getTextureId(self):
        return self.__textureId

    def openGLCleanup(self):
        _logger.debug("CLEANING OPENGL")
        if self.drawList <= 0:
            GL.glDeleteLists(self.drawList, 1)
            self.drawList = 0
        if self.__textureId is not None:
            GL.glDeleteTextures([self.__textureId])
            self.__textureId = None

    def setPixmap(self, pixmap, width, height, xmirror = False, ymirror = False, z=0.0):
        if not hasattr(pixmap, "dtype"):
            raise ValueError("Input pixmap has to be an uint8 array")

        useNewTexture = True
        if self.__pixmap is not None:
            if (width == self.__width) and (height == self.__height):
                useNewTexture = False
        #I force always for the time being
        useNewTexture = True

        #make sure we work with integers
        self.__width  = int(width)
        self.__height = int(height)

        # some cards still need to pad to powers of 2
        self.__tWidth  = self.getPaddedValue(self.__width)
        self.__tHeight = self.getPaddedValue(self.__height)

        if (self.__tWidth != self.__width) or (self.__tHeight != self.__height):
            #I have to zero padd the texture to make sure it works on all cards ...
            self.__pixmap = numpy.zeros((self.__tWidth*self.__tHeight, 4), numpy.uint8)
            pixmap.shape = [width*height, 4]
            tjump = self.__tWidth
            pjump = self.__width
            for i in range(height):
                self.__pixmap[i*tjump:(i*tjump+pjump), :] = pixmap[(i*pjump):(i+1)*pjump,:]
        else:
            self.__pixmap = pixmap
            self.__pixmap.shape = [width*height, 4]
        self.__pixmap[:,3] = 255 #alpha
        self.__xMirror = xmirror
        self.__yMirror = ymirror
        self._forceListCalculation = True
        self._forceTextureCalculation = True
        self._useNewTexture = useNewTexture
        self.setLimits(0.0, 0.0, z, self.__width, self.__height, z)

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
        self._forceListCalculation = True

    def drawObject(self):
        if self.__pixmap is None:
            return
        #if self.__xMirror or self.__yMirror:
        #    self.__mirrorRotator(self.__xMirror, self.__yMirror)

        if self._forceTextureCalculation:
            self.buildTexture()
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if self._forceListCalculation:
            self.buildQuad()
        if self.drawList:
            GL.glCallList(self.drawList)
        #else:
        #    self.buildQuad()
        GL.glPopAttrib()

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
        gl.glTexSubImage2D(texture.getTarget(),
            0, // no support for mipmapping
            x, y + row, // in texture
            bounds.width, 1,
            GL.GL_BGRA,
            type,
            dataBuffer
            );


        """
        if self._useNewTexture:
            if self.__textureId is not None:
                GL.glDeleteTextures([self.__textureId])
            self.__textureId = GL.glGenTextures(1)
        else:
            if self.__textureId is None:
               self.__textureId = GL.glGenTextures(1)
            else:
                GL.glDeleteTextures([self.__textureId])
                self.__textureId = GL.glGenTextures(1)
        if self.__textureId is None:
            _logger.info("no valid texture id?")
            return
        if self._useNewTexture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.__textureId)
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT )
            GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT )
            linear = 0
            if linear:
                GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR )
                GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR )
            else:
                #Nearest on magnification
                GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST )
                #Linear when minimizing
                GL.glTexParameteri( GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR )

            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
                            self.__tWidth,
                            self.__tHeight,
                            0, GL.GL_RGBA,
                            GL.GL_UNSIGNED_BYTE, self.__pixmap)
            GL.glEnable(GL.GL_TEXTURE_2D)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.__textureId)
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D,
            0,
            0, 0,
            self.__tWidth,
            self.__tHeight,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            self.__pixmap)
            GL.glEnable(GL.GL_TEXTURE_2D)

        self._forceTextureCalculation = False

    def buildQuad(self):
        if self.drawList > 0:
            GL.glDeleteLists(self.drawList, 1)
        xmin, ymin, zmin = self._limits[0]
        xmax, ymax, zmax = self._limits[1]
        tx0 = 0.0
        tx1 = (1.0 * self.__width)/self.__tWidth
        ty0 = 0.0
        ty1 = (1.0 * self.__height)/self.__tHeight
        self.drawList = GL.glGenLists(1)
        GL.glNewList(self.drawList, GL.GL_COMPILE)
        #The texture gets multiplied by this color!!
        GL.glColor4f(1.0, 1.0, 1.0, self.__alpha)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.__textureId)
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

    def getPaddedValue(self, v):
        a = 2
        while (a<v):
            a *=2
        return int(a)


    def __mirrorRotator(self, xmirror, ymirror):
        if xmirror:
            x = -1.0
        else:
            x = 1.0
        if ymirror:
            y = -1.0
        else:
            y = 1.0
        GL.glScalef(x, y, 1.0)
