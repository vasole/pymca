#/*##########################################################################
# Copyright (C) 2004-2020 European Synchrotron Radiation Facility
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
try:
    import OpenGL.GL  as GL
    import OpenGL.GLU as GLU
except ImportError:
    raise ImportError("OpenGL must be installed to use these functionalities")
import numpy
try:
    from PyMca5 import spslut
except:
    import spslut
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

try:
    from PyMca5.Object3D.Object3DPlugins import Object3DMeshConfig
except ImportError:
    try:
        from Object3D.Object3DPlugins import Object3DMeshConfig
    except:
        import Object3DMeshConfig

from PyMca5.PyMcaGui import PyMcaFileDialogs

qt = Object3DMeshConfig.qt
import weakref
#import buffers
DEBUG = 0
import time
DRAW_MODES = ['NONE',
              'POINT',
              'WIRE',
              'SURFACE',
              'LIGHT',
              'POINT_SELECTION']

COLORMAPLIST = [spslut.GREYSCALE, spslut.REVERSEGREY, spslut.TEMP,
                spslut.RED, spslut.GREEN, spslut.BLUE, spslut.MANY]

class Object3DStack(Object3DBase.Object3D):
    def __init__(self, name = "3D-Array"):
        Object3DBase.Object3D.__init__(self, name)
        self._alpha = 255

        self.drawListDict = {}
        self._forceListCalculation = {}
        self.vertices = None
        self.vertexColors = None
        self.vertexSelectionColors = None
        self._selected     = False
        self._vertexSelectionMode = False
        self.drawMode = 'POINT'
        self.__isosurfacesDict = {}
        for i in range(5):
            self.__isosurfacesDict[i] = {}
            self.__isosurfacesDict[i]['list'] = 0
            self.__isosurfacesDict[i]['value'] = 0
            self.__isosurfacesDict[i]['color'] = 'red'
            self.__isosurfacesDict[i]['r'] = 0xFF
            self.__isosurfacesDict[i]['g'] = 0
            self.__isosurfacesDict[i]['b'] = 0
            self.__isosurfacesDict[i]['a'] = 0xFF
        self._configuration['common']['supportedmodes'] = [1, 1, 1, 1]
        self._configuration['common']['mode'] = 1

        #centered on XY plane and on Z
        self._configuration['common']['anchor'] = [2, 2, 2]

        #self._verticesBufferObject = None
        #self._vertexColorBufferObject = None
        #self._vertexSelectionColorBufferObject = None

    def initPrivateConfiguration(self, name):
        """
        Specific configuration
        """
        self._configuration['private'] = {}
        if self._privateConfigurationWidget is None:
            self._privateConfigurationWidget = Object3DMeshConfig.\
                                               Object3DMeshConfig(None, name)
        self._configuration['private']['widget'] = weakref.proxy(self._privateConfigurationWidget)
        self._configuration['private']['colorfilter'] = 1
        self._configuration['private']['isosurfaces'] = [[1, 20, 'green', 0, 0xFF, 0, 0xFF]] #green
        #self._configuration['private']['isosurfaces'] = [[1, 10, None, 0, 0, 0, 0xFF]] #auto
        self._configuration['private']['useminmax']    = [0, 100, 200]
        self._configuration['private']['infolabel'] = "Object3DStack %s" % name

    def __del__(self):
        for key in self.drawListDict.keys():
            if key.upper() != "NONE":
                if self.drawListDict[key] > 0:
                    GL.glDeleteLists(self.drawListDict[key], 1)

        for key in self.__isosurfacesDict.keys():
            if self.__isosurfacesDict[key]['list'] > 0:
                GL.glDeleteLists(self.__isosurfacesDict[key]['list'], 1)

        try:
            Object3DBase.Object3D.__del__(self)
        except AttributeError:
            pass

    def setConfiguration(self, ddict):
        old_alpha = 1.0 - self._configuration['common']['transparency']
        Object3DBase.Object3D.setConfiguration(self, ddict)
        new_alpha = 1.0 - self._configuration['common']['transparency']

        if (new_alpha != old_alpha):
            self._setAlpha(new_alpha)

        self.drawMode = DRAW_MODES[self._configuration['common']['mode']]

        if 'event' in ddict['common']:
            if ddict['common']['event'] == 'ColormapChanged':
                self.getColors()

    def _setAlpha(self, alpha):
        if alpha < 0:
            alpha = 0
        elif alpha >= 1.0:
            alpha = 255
        else:
            self._alpha = int(255 * alpha)
        if self.vertexColors is None:
            return
        self.vertexColors[:, 3] = self._alpha

    def setData(self, *args, **kw):
        return self.setStack(*args, **kw)

    def setStack(self, data, x=None, y=None, z=None, xyz=None):
        """
        setStack(data, data, xyz=None)
        data    is the array of vertex values.
        xyz = [x,y,z] are three arrays with the grid coordinates
        """
        if hasattr(data, "info") and hasattr(data, "data"):
            #It is an actual stack
            self._actualStack = True
            self.values = data.data[:]
        else:
            self._actualStack = False
            self.values = data[:]

        if self.values.dtype != numpy.float32:
            print("WARNING: Converting to float32")
            self.values = self.values.astype(numpy.float32)
        if (x is None) and (y is None) and (xyz is None):
            xsize, ysize, zsize = self.values.shape
            self._x = numpy.arange(xsize).astype(numpy.float32)
            self._y = numpy.arange(ysize).astype(numpy.float32)
            self._z = numpy.arange(zsize).astype(numpy.float32)
            if self._actualStack:
                xCal = list(map(float, eval(data.info.get('CAxis0CalibrationParameters', '[0., 1.0, 0.0]'))))
                yCal = list(map(float, eval(data.info.get('CAxis1CalibrationParameters', '[0., 1.0, 0.0]'))))
                zCal = list(map(float, eval(data.info.get('CAxis2CalibrationParameters', '[0., 1.0, 0.0]'))))
                self._x[:] = xCal[0] + self._x * (xCal[1] + xCal[2] * self._x)
                self._y[:] = yCal[0] + self._y * (yCal[1] + yCal[2] * self._y)
                self._z[:] = zCal[0] + self._z * (zCal[1] + zCal[2] * self._z)
            self.xSize, self.ySize, self.zSize = xsize, ysize, zsize
        elif xyz is not None:
            self.xSize, self.ySize, self.zSize = self.values.shape
            self._x[:] = xyz[0][:]
            self._y[:] = xyz[1][:]
            self._z[:] = xyz[2][:]
        elif (x is not None) and (y is not  None):
            #regular mesh
            self._x = numpy.array(x).astype(numpy.float32)
            self._y = numpy.array(y).astype(numpy.float32)
            self._x.shape = -1, 1
            self._y.shape = -1, 1
            self.xSize = self._x.shape[0]
            self.ySize = self._y.shape[0]
            if z is not None:
                self._z = numpy.array(z).astype(numpy.float32)
                if len(self._z.shape) == 0:
                    #assume just a number
                    self.zSize = 1
                else:
                    self._z.shape = -1, 1
                    self.zSize = self._z.shape[0]
            else:
                a=1
                for v in self.values.shape:
                    a *= v
                zsize = int(a/(self.xSize * self.ySize))
                self._z = numpy.arange(zsize).astype(numpy.float32)
                self.zSize = zsize
        else:
            raise ValueError("Unhandled case")

        old_shape = self.values.shape
        self.nVertices = self.xSize * self.ySize * self.zSize
        self.values.shape = self.nVertices, 1

        self.getColors()
        self._obtainLimits()
        #restore original shape
        self.values.shape = old_shape

    def getColors(self):
        old_shape = self.values.shape
        self.values.shape = -1, 1
        self._configuration['common']['colormap'][4]=self.values.min()
        self._configuration['common']['colormap'][5]=self.values.max()
        colormap = self._configuration['common']['colormap']
        (self.vertexColors,size,minmax)= spslut.transform(self.values,
                                              (1,0),
                                              (colormap[6],3.0),
                                              "RGBX",
                                              COLORMAPLIST[int(str(colormap[0]))],
                                              colormap[1],
                                              (colormap[2], colormap[3]),
                                              (0, 255),1)
        self.values.shape = old_shape
        self.vertexColors.shape = self.nVertices, 4
        self.vertexColors[:, 3] = self._alpha
        #selection colors
        # if I have more than pow(2, 24) vertices
        # the vertex with number pow(2, 24) will never be selected
        return
        i = numpy.arange(self.nVertices)
        self.vertexSelectionColors = numpy.zeros((self.nVertices,4),
                                                 numpy.uint8)
        self.vertexSelectionColors[:,0] = (i & 255)
        self.vertexSelectionColors[:,1] = ((i >> 8) & 255)
        self.vertexSelectionColors[:,2] = ((i >> 16) & 255)
        self.vertexSelectionColors[:,3] = 255 - (i >> 24)

    def _obtainLimits(self):
        xmin, ymin, zmin =  self._x.min(), self._y.min(), self._z.min()
        xmax, ymax, zmax =  self._x.max(), self._y.max(), self._z.max()
        self.setLimits(xmin, ymin, zmin, xmax, ymax, zmax)

    def drawObject(self):
        if self.values is None:
            return
        if DEBUG:
            t0=time.time()
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        GL.glShadeModel(GL.GL_FLAT)
        if self.drawMode == 'NONE':
            pass
        elif (GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_SELECT) or \
           self._vertexSelectionMode:
            self.buildPointList(selection=True)
        elif self.drawMode == 'POINT':
            self.buildPointList(selection=False)
            #self.buildPointListNEW(selection=False)
        elif self.drawMode == 'POINT_SELECTION':
            self.buildPointList(selection=True)
        elif self.drawMode in ['LINES', 'WIRE']:
            Object3DCTools.draw3DGridLines(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexColors,
                                       self.values,
                                       self._configuration['private']['colorfilter'],
                                       self._configuration['private']['useminmax'])
        elif self.drawMode == "SURFACE":
            flag = 1
            i = 0
            for use, value, label, cr, cg, cb, ca in self._configuration['private']['isosurfaces']:
                color = (cr, cg, cb, ca)
                if None in color:
                    color = None
                if use:
                    flag = 0
                    GL.glEnable(GL.GL_LIGHTING)
                    if color is not None:
                        GL.glColor4ub(color[0],
                                      color[1],
                                      color[2],
                                      self._alpha)
                    colorflag = False
                    if self.__isosurfacesDict[i]['list'] > 0:
                        if self.__isosurfacesDict[i]['color'] == color:
                            colorflag = True
                        elif (self.__isosurfacesDict[i]['color'] != None) and\
                             (color != None):
                            colorflag = True
                    if self.__isosurfacesDict[i]['list'] > 0:
                        if (self.__isosurfacesDict[i]['value'] == value) and\
                           colorflag:
                            GL.glCallList(self.__isosurfacesDict[i]['list'])
                            i += 1
                            continue
                        GL.glDeleteLists(self.__isosurfacesDict[i]['list'],
                                            1)
                    self.__isosurfacesDict[i]['value']= value
                    self.__isosurfacesDict[i]['color']= color
                    self.__isosurfacesDict[i]['list'] = GL.glGenLists(1)
                    GL.glNewList(self.__isosurfacesDict[i]['list'],
                                                 GL.GL_COMPILE)

                    GL.glBegin(GL.GL_TRIANGLES)
                    Object3DCTools.gridMarchingCubes(self._x, self._y, self._z, self.values, value, color, (1, 1, 1), 1)
                    #Object3DCTools.gridMarchingCubes(self._x, self._y, self._z, self.values, value, None, (1, 1, 1), 1)
                    GL.glEnd()
                    GL.glEndList()
                    GL.glCallList(self.__isosurfacesDict[i]['list'])
                    GL.glDisable(GL.GL_LIGHTING)
                i += 1
            if flag:
                #This is useless, only isosurfaces makes sense
                Object3DCTools.draw3DGridQuads(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexColors,
                                       self.values,
                                       self._configuration['private']['colorfilter'],
                                       self._configuration['private']['useminmax'])
        else:
            print("UNSUPPORTED MODE")
        GL.glPopAttrib()
        if DEBUG:
            print("Drawing takes ", time.time() - t0)

    def _getVertexSelectionColors(self):
        self.vertexSelectionColors = numpy.zeros((self.nVertices,4),
                                                 numpy.uint8)

        #split the color generation in two blocks
        #to reduce the amount of memory needed
        half = int(self.nVertices/2)
        i = numpy.arange(0, half)
        self.vertexSelectionColors[:half,0] = (i & 255)
        self.vertexSelectionColors[:half,1] = ((i >> 8) & 255)
        self.vertexSelectionColors[:half,2] = ((i >> 16) & 255)
        self.vertexSelectionColors[:half,3] = 255 - (i >> 24)

        i = numpy.arange(half, self.nVertices)
        self.vertexSelectionColors[half:,0] = (i & 255)
        self.vertexSelectionColors[half:,1] = ((i >> 8) & 255)
        self.vertexSelectionColors[half:,2] = ((i >> 16) & 255)
        self.vertexSelectionColors[half:,3] = 255 - (i >> 24)

    def isVertexSelectionModeSupported(self):
        return True

    def buildPointList(self, selection=False):
        if selection:
            if self.vertexSelectionColors is None:
                self._getVertexSelectionColors()
            if self._configuration['private']['colorfilter']:
                tinyNumber = 1.0e-10
                minValue = self._configuration['common']['colormap'][2] + tinyNumber
                maxValue = self._configuration['common']['colormap'][3] - tinyNumber
                Object3DCTools.draw3DGridPoints(self._x,
                                           self._y,
                                           self._z,
                                           self.vertexSelectionColors,
                                           self.values,
                                           0,
                                           [1, minValue, maxValue])
            else:
                Object3DCTools.draw3DGridPoints(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexSelectionColors,
                                       self.values,
                                       0,
                                       self._configuration['private']['useminmax'])
        else:
            Object3DCTools.draw3DGridPoints(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexColors,
                                       self.values,
                                       self._configuration['private']['colorfilter'],
                                       self._configuration['private']['useminmax'])

    def buildWireList(self):
        Object3DCTools.draw3DGridLines(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexColors)

    def __fillVerticesBufferObject(self):
        if self.vertices is None:
            self.vertices = Object3DCTools.get3DGridFromXYZ(self._x,
                                                   self._y,
                                                   self._z)
            self.indices = numpy.arange(self.nVertices)
        self._verticesBufferObject = buffers.VertexBuffer(self.vertices,
                                                        GL.GL_STATIC_DRAW)
        self.vertices = None

    def __fillVertexColorsBufferObject(self):
        if self.vertexColors is None:
            if self.vertexSelectionColors is None:
                i = numpy.arange(self.nVertices)
                self.vertexSelectionColors = numpy.zeros((self.nVertices,4),
                                                         numpy.uint8)
                self.vertexSelectionColors[:,0] = (i & 255)
                self.vertexSelectionColors[:,1] = ((i >> 8) & 255)
                self.vertexSelectionColors[:,2] = ((i >> 16) & 255)
                self.vertexSelectionColors[:,3] = 255 - (i >> 24)
        self._vertexColorsBufferObject = buffers.VertexBuffer(self.vertexSelectionColors,
                                                        GL.GL_STATIC_DRAW)

    def buildPointListNEW(self, selection=False):
        if self._verticesBufferObject is None:
            self.__fillVerticesBufferObject()

        if self._vertexColorsBufferObject is None:
            self.__fillVertexColorsBufferObject()

        #self._vertexSelectionColorBufferObject = None
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        self._verticesBufferObject.bind()
        self._vertexColorsBufferObject.bind()
        GL.glDrawElements(GL.GL_POINTS, self.indices)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)

    def buildPointListOLD(self):
        if self.vertices is None:
            self.vertices = Object3DCTools.get3DGridFromXYZ(self._x,
                                                       self._y,
                                                       self._z)
        GL.glVertexPointerf(self.vertices)
        GL.glColorPointerub(self.vertexColors)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.nVertices)

    def buildPointList0(self):
        """
        This is just to test memory and speed
        """
        n1, n2, n3 = 256, 256, 256
        zdata = numpy.arange(n1*n2*n3).astype(numpy.float32)
        zdata.shape= -1, 1
        (image,size,minmax)= spslut.transform(zdata,
                                          (1,0),
                                          (spslut.LINEAR,3.0),
                                          "RGBX",
                                          spslut.TEMP,
                                          1,
                                          (0, 1),
                                          (0, 255),1)
        image.shape = -1, 4
        image[:,3] = 255
        #self.vertexColors = image.astype(numpy.float32)
        x = numpy.arange(n1).astype(numpy.float32)
        y = numpy.arange(n2).astype(numpy.float32)
        z = numpy.arange(n3).astype(numpy.float32)
        #Object3DCTools.draw3DGridQuads(x, y, y)
        #Object3DCTools.draw3DGridLines(x, y, z, image)
        Object3DCTools.draw3DGridPoints(x, y, z, image)
        self.zdata = zdata

    def getIndexValues(self, index):
        """
        x,y,z, I
        """
        xindex = int(index/(self.ySize*self.zSize))
        yindex = int((index % (self.ySize*self.zSize))/self.zSize)
        zindex = index % self.zSize
        #print "index = ", index, "xindex = ", xindex, "yindex = ", yindex, "zindex = ", zindex
        if len(self.values.shape) == 3:
            value =  self.values[xindex, yindex, zindex]
        else:
            value = self.values[index]
        return self._x[xindex], self._y[yindex], self._z[zindex], value

MENU_TEXT = '4D Stack'
def getObject3DInstance(config=None):
    #for the time being a former configuration
    #for serializing purposes is not implemented

    #I do the import here for the case PyMca is not installed
    #because the modules could be instanstiated without using
    #this method
    try:
        from PyMca5.PyMcaIO import EDFStack
        from PyMca5.PyMcaIO import TiffStack
    except ImportError:
        import EDFStack
        import TiffStack

    fileTypeList = ['EDF Z Stack (*edf *ccd)',
                    'EDF X Stack (*edf *ccd)',
                    'TIFF Stack (*tif *tiff)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = False
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select the object file(s)",
        mode="OPEN",
        getfilter=True)
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = old
    if not len(fileList):
        return None
    if filterUsed == fileTypeList[0]:
        fileindex = 2
    else:
        fileindex = 1
    #file index is irrelevant in case of an actual 3D stack.
    filename = fileList[0]
    legend = os.path.basename(filename)
    if filterUsed == fileTypeList[2]:
        #TIFF
        stack = TiffStack.TiffStack(dtype=numpy.float32, imagestack=False)
        stack.loadFileList(fileList, fileindex=1)
    elif len(fileList) == 1:
        stack = EDFStack.EDFStack(dtype=numpy.float32, imagestack=False)
        stack.loadIndexedStack(filename, fileindex=fileindex)
    else:
        stack = EDFStack.EDFStack(dtype=numpy.float32, imagestack=False)
        stack.loadFileList(fileList, fileindex=fileindex)
    if stack is None:
        raise IOError("Problem reading stack.")
    object3D = Object3DStack(name=legend)
    object3D.setStack(stack)
    return object3D

if __name__ == "__main__":
    import sys
    try:
        from PyMca5.Object3D import SceneGLWindow
    except ImportError:
        from Object3D import SceneGLWindow
    import os
    try:
        from PyMca5.PyMcaIO import EDFStack
        from PyMca5.PyMcaIO import EdfFile
    except ImportError:
        import EDFStack
        import EdfFile
    import getopt
    options = ''
    longoptions = ["fileindex=","begin=", "end="]
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except:
        print(sys.exc_info()[0])
        sys.exit(1)
    fileindex = 2
    begin = None
    end = None
    for opt, arg in opts:
        if opt in '--begin':
            begin = int(arg)
        elif opt in '--end':
            end = int(arg)
        elif opt in '--fileindex':
            fileindex = int(arg)
    app = qt.QApplication(sys.argv)
    window = SceneGLWindow.SceneGLWindow()
    window.show()
    if len(sys.argv) == 1:
        object3D = getObject3DInstance()
        if object3D is not None:
            window.addObject(object3D)
    else:
        if len(sys.argv) > 1:
            stack = EDFStack.EDFStack(dtype=numpy.float32, imagestack=False)
            filename = args[0]
        else:
            stack = EDFStack.EDFStack(dtype=numpy.float32, imagestack=False)
            filename = r"..\COTTE\ch09\ch09__mca_0005_0000_0070.edf"
        if os.path.exists(filename):
            print("fileindex = ", fileindex)
            stack.loadIndexedStack(filename, begin=begin, end=end, fileindex=fileindex)
            object3D = Object3DStack()
            object3D.setStack(stack)
            stack = 0
        else:
            print("filename %s does not exists" % filename)
            sys.exit(1)
        time.sleep(1)
        print("START ADDING")
        window.addObject(object3D, "STACK")
        window.setSelectedObject("STACK")
        print("END ADDING")

    window.glWidget.setZoomFactor(1.0)
    window.show()
    app.exec_()
