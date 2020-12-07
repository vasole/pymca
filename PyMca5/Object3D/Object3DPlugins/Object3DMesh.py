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
try:
    import OpenGL.GL  as GL
    import OpenGL.GLU as GLU
    from OpenGL.GL import glDeleteLists
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
    from PyMca5.Object3D import Object3DQhull
except ImportError:
    try:
        from Object3D import Object3DCTools
        from Object3D import Object3DQhull
    except ImportError:
        import Object3DCTools
        import Object3DQhull

from PyMca5.PyMcaGui import PyMcaFileDialogs

try:
    from PyMca5.Object3D.Object3DPlugins import Object3DMeshConfig
except ImportError:
    try:
        from Object3D.Object3DPlugins import Object3DMeshConfig
    except:
        import Object3DMeshConfig

qt = Object3DMeshConfig.qt
import weakref

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

class Object3DMesh(Object3DBase.Object3D):
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
        self.__flat = True
        self._xyz = False
        self.facets = None
        self.drawMode = 'POINT'
        self._configuration['common']['supportedmodes'] = [1, 1, 1, 1]
        self._configuration['common']['mode'] = 1

        #centered on XY plane and on Z
        self._configuration['common']['anchor'] = [2, 2, 2]

    def initPrivateConfiguration(self, name):
        """
        Specific configuration
        """
        self._configuration['private'] = {}
        if self._privateConfigurationWidget is None:
            self._privateConfigurationWidget = Object3DMeshConfig.\
                                               Object3DMeshConfig(None, name)
        self._configuration['private']['widget'] = weakref.proxy(self._privateConfigurationWidget)
        self._configuration['private']['colorfilter'] = 0
        self._configuration['private']['isosurfaces'] = [[0, 10, 'green', 0, 0xFF, 0, 0xFF]] #green
        self._configuration['private']['useminmax']    = [0, 100, 200]
        self._configuration['private']['infolabel'] = "Object3DMesh %s" % name

    def __del__(self):
        if DEBUG:
            print("Deleting object %s" % self.name())
        for key in self.drawListDict.keys():
            if key.upper() != "NONE":
                if self.drawListDict[key] > 0:
                    glDeleteLists(self.drawListDict[key], 1)
        del self._privateConfigurationWidget
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

    def setData(self, data, x=None, y=None, z=None, xyz=None):
        self.values = data[:]
        self._configuration['private']['useminmax'] = [0, self.values.min(), self.values.max()]
        if (xyz is not None) and (x is None) and (y is None):
            arr = Object3DCTools.getVertexArrayMeshAxes(xyz)
            if arr is not None:
                doit = True
                x = arr[0]
                if len(x) > 1:
                    if abs(x[1] - xyz[0,0]) > 1.0e-10:
                        #not proper C order
                        #prevent bad plotting of a regular grid
                        doit = False
                        x = None
                if doit:
                    y = arr[1]
                    z = numpy.zeros(x.shape[0]*y.shape[0], numpy.float64)
                    z[:] = xyz[:, 2]
                    xyz = None
        if (x is None) and (y is None) and (xyz is None):
            #regular mesh
            self.xSize, self.ySize = data.shape
            self.zSize = 1
            self._x = numpy.arange(self.xSize).astype(numpy.float32)
            self._y = numpy.arange(self.ySize).astype(numpy.float32)
            if z is not None:
                self._z = numpy.array(z).astype(numpy.float32)
                if len(self._z.shape) == 0:
                    #assume just a number
                    self._z.shape = 1, 1
                    self.zSize = 1
                else:
                    self._z.shape = -1, 1
                    self.zSize = self._z.shape[0]
                    self.__flat = False
            else:
                self._z = numpy.arange(self.zSize).astype(numpy.float32)
                self._z.shape = 1, 1
                self.zSize = 1
        elif xyz is not None:
            #full irregular mesh
            self.__setXYZArray(xyz, values=data)
            self.xSize = self.vertices.shape[0]
            self.ySize = 1
            self._x = self.vertices[:,0]
            self._y = self.vertices[:,1]
            self._z = self.vertices[:,2]
            self._xyz = True
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
                    self._z.shape = 1, 1
                    self.zSize = 1
                else:
                    self._z.shape = -1, 1
                    self.zSize = self._z.shape[0]
                    self.__flat = False
            else:
                self._z = numpy.arange(self.zSize).astype(numpy.float32)
                self._z.shape = 1, 1
                self.zSize = 1
        else:
            raise ValueError("Unhandled case")
        self.nVertices = self.xSize * self.ySize
        self.values.shape = self.nVertices, 1

        self.getColors()
        self._obtainLimits()

    def __setXYZArray(self, xyz, values=None):
        if values is None:
            #This case could use a 1D texture
            self.values = xyz[:,2]
        else:
            #This case cannot use 1D texture
            self.values = numpy.array(values, numpy.float32)
        self.vertices  = xyz
        self.nVertices = self.vertices.shape[0]

    def getColors(self):
        if DEBUG:
            t0 = time.time()
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
        self.vertexColors.shape = self.nVertices, 4
        self.vertexColors[:, 3] = self._alpha
        if DEBUG:
            print("colors elapsed = ", time.time() - t0)
        #selection colors
        # if I have more than pow(2, 24) vertices
        # the vertex with number pow(2, 24) will never be selected
        if 0:
            i = numpy.arange(self.nVertices)
            self.vertexSelectionColors = numpy.zeros((self.nVertices,4),
                                                     numpy.uint8)
            self.vertexSelectionColors[:,0] = (i & 255)
            self.vertexSelectionColors[:,1] = ((i >> 8) & 255)
            self.vertexSelectionColors[:,2] = ((i >> 16) & 255)
            self.vertexSelectionColors[:,3] = 255 - (i >> 24)
        return

    def _obtainLimits(self):
        xmin, ymin, zmin =  self._x.min(), self._y.min(), self._z.min()
        xmax, ymax, zmax =  self._x.max(), self._y.max(), self._z.max()
        self.setLimits(xmin, ymin, zmin, xmax, ymax, zmax)

    def drawObject(self):
        if self.values is None:
            return
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        GL.glShadeModel(GL.GL_FLAT)
        if DEBUG:
            t0=time.time()
        if self.drawMode == 'NONE':
            pass
        elif (GL.glGetIntegerv(GL.GL_RENDER_MODE) == GL.GL_SELECT) or \
           self._vertexSelectionMode:
            GL.glPointSize(self._configuration['common']['pointsize'])
            if self._xyz:
                self.buildPointListXYZ(selection=True)
            else:
                self.buildPointList(selection=True)
        elif self.drawMode == 'POINT':
            GL.glShadeModel(GL.GL_FLAT)
            GL.glPointSize(self._configuration['common']['pointsize'])
            if self._xyz:
                self.buildPointListXYZ(selection=False)
            else:
                self.buildPointList(selection=False)
        elif self.drawMode == 'POINT_SELECTION':
            GL.glShadeModel(GL.GL_FLAT)
            GL.glPointSize(self._configuration['common']['pointsize'])
            self.buildPointList(selection=True)
        elif self.drawMode in ['LINES', 'WIRE']:
            GL.glLineWidth(self._configuration['common']['linewidth'])
            GL.glShadeModel(GL.GL_SMOOTH)
            if self._xyz:
                if self.facets is None:
                    self._getFacets()
                Object3DCTools.drawXYZLines(self.vertices,
                             self.vertexColors,
                             self.values,
                             self.facets,
                             self._configuration['private']['colorfilter'],
                             self._configuration['private']['useminmax'])
                #sys.exit(1)
            elif self.__flat:
                Object3DCTools.draw3DGridLines(self._x,
                                self._y,
                                self._z,
                                self.vertexColors,
                                self.values,
                                self._configuration['private']['colorfilter'],
                                self._configuration['private']['useminmax'])
            else:
                Object3DCTools.draw2DGridLines(self._x,
                                self._y,
                                self._z,
                                self.vertexColors,
                                self.values,
                                self._configuration['private']['colorfilter'],
                                self._configuration['private']['useminmax'])
        elif self.drawMode == "SURFACE":
            GL.glShadeModel(GL.GL_SMOOTH)
            if self._xyz:
                if self.facets is None:
                    self._getFacets()
                Object3DCTools.drawXYZTriangles(self.vertices,
                             self.vertexColors,
                             self.values,
                             self.facets,
                             self._configuration['private']['colorfilter'],
                             self._configuration['private']['useminmax'])

            elif self.__flat:
                Object3DCTools.draw3DGridQuads(self._x,
                            self._y,
                            self._z,
                            self.vertexColors,
                            self.values,
                            self._configuration['private']['colorfilter'],
                            self._configuration['private']['useminmax'])
            else:
                Object3DCTools.draw2DGridQuads(self._x,
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
        i = numpy.arange(self.nVertices)
        self.vertexSelectionColors = numpy.zeros((self.nVertices,4),
                                                 numpy.uint8)
        self.vertexSelectionColors[:,0] = (i & 255)
        self.vertexSelectionColors[:,1] = ((i >> 8) & 255)
        self.vertexSelectionColors[:,2] = ((i >> 16) & 255)
        self.vertexSelectionColors[:,3] = 255 - (i >> 24)

    def isVertexSelectionModeSupported(self):
        return True

    def _getFacets(self):
        if self.vertices is None:
            self.facets = None
        #if self.vertices.dtype == numpy.float32:
        #    self.facets = Object3DQhullf.delaunay(self.vertices[:,0:2],
        #                         "qhull d Qbb QJ Qc Po")
        #else:
        if DEBUG:
            e0 = time.time()
        self.facets = Object3DQhull.delaunay(self.vertices[:,0:2],
                                 "qhull d Qbb QJ Qc")
        if DEBUG:
            print("delaunay elapsed = ", time.time() -e0)
            print("facets 1st= ",self.facets[0,:])
            print("vertices 1st =", self.vertices[self.facets[0,0]],\
                                self.vertices[self.facets[0,1]],\
                                self.vertices[self.facets[0,2]])
            print("COLORS = ", self.vertexColors[self.facets[0,0]],\
                           self.vertexColors[self.facets[0,1]],\
                           self.vertexColors[self.facets[0,2]])
            print("COLORS = ", self.vertexColors[self.facets[1,0]],\
                           self.vertexColors[self.facets[1,1]],\
                           self.vertexColors[self.facets[1,2]])

    def buildPointList(self, selection=False):
        if selection:
            if self.vertexSelectionColors is None:
                self._getVertexSelectionColors()
            if self.__flat:
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
                #if self._configuration['private']['colorfilter'] is selected
                # I should get the min and max values of the colormap ...
                if self._configuration['private']['colorfilter']:
                    tinyNumber = 1.0e-10
                    minValue = self._configuration['common']['colormap'][2] + tinyNumber
                    maxValue = self._configuration['common']['colormap'][3] - tinyNumber
                    Object3DCTools.draw2DGridPoints(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexSelectionColors,
                                       self.values,
                                       0,
                                       [1, minValue, maxValue])
                else:
                    Object3DCTools.draw2DGridPoints(self._x,
                                       self._y,
                                       self._z,
                                       self.vertexSelectionColors,
                                       self.values,
                                       0,
                                       self._configuration['private']['useminmax'])
        else:
            if self.__flat:
                Object3DCTools.draw3DGridPoints(self._x,
                            self._y,
                            self._z,
                            self.vertexColors,
                            self.values,
                            self._configuration['private']['colorfilter'],
                            self._configuration['private']['useminmax'])
            else:
                Object3DCTools.draw2DGridPoints(self._x,
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


    def buildPointListXYZ(self, selection=False):
        if 1:
            if selection:
                if self.vertexSelectionColors is None:
                    self._getVertexSelectionColors()
                if self._configuration['private']['colorfilter']:
                    tinyNumber = 1.0e-10
                    minValue = self._configuration['common']['colormap'][2] + tinyNumber
                    maxValue = self._configuration['common']['colormap'][3] - tinyNumber
                    Object3DCTools.drawXYZPoints(self.vertices,
                             self.vertexSelectionColors,
                             self.values,
                             None,
                             0,
                             [1, minValue, maxValue])
                else:
                    Object3DCTools.drawXYZPoints(self.vertices,
                             self.vertexSelectionColors,
                             self.values,
                             None,
                             0,
                             self._configuration['private']['useminmax'])
            else:
                Object3DCTools.drawXYZPoints(self.vertices,
                             self.vertexColors,
                             self.values,
                             None,
                             self._configuration['private']['colorfilter'],
                             self._configuration['private']['useminmax'])
        return


        GL.glVertexPointerf(self.vertices)
        if selection:
            GL.glColorPointerub(self.vertexSelectionColors)
        else:
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
        if self._xyz:
            xindex = index
            yindex = index
        else:
            xindex = int(index/self.ySize)
            yindex = index % (self.ySize)
        if self.__flat:
            zindex = 0
        else:
            zindex = index
        return self._x[xindex], self._y[yindex], self._z[zindex], self.values[index]

MENU_TEXT = '3D Mesh'
def getObject3DInstance(config=None):
    #for the time being a former configuration
    #for serializing purposes is not implemented


    #I do the import here for the case PyMca is not installed
    #because the modules could be instanstiated without using
    #this method
    try:
        from PyMca5.PyMcaIO import EdfFile
    except ImportError:
        import EdfFile

    fileTypeList = ['EDF Files (*edf)',
                    'EDF Files (*ccd)',
                    'ADSC Files (*img)',
                    'All Files (*)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = False
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select one object data file",
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
    edf = EdfFile.EdfFile(filename, access='rb')
    data = edf.GetData(0).astype(numpy.float32)
    object3D = Object3DMesh(os.path.basename(filename))
    object3D.setData(data, z=data[:])
    return object3D

if __name__ == "__main__":
    import sys
    import os
    from PyMca5.Object3D import SceneGLWindow
    try:
        from PyMca5.PyMcaIO import EdfFile
    except ImportError:
        import EdfFile
    app = qt.QApplication(sys.argv)
    window = SceneGLWindow.SceneGLWindow()
    window.show()
    if len(sys.argv) > 1:
        flist = []
        for i in range(1, len(sys.argv)):
            flist.append(sys.argv[i])
        for f in flist:
            edf = EdfFile.EdfFile(f, access='rb')
            data = edf.GetData(0)
            object3D = Object3DMesh(os.path.basename(f))
    else:
        data = numpy.arange(200.).astype(numpy.float32)
        data.shape = [40, 5]
        object3D = Object3DMesh('builtin')

    #several options: regular grid, irregular grid
    if len(sys.argv) > 1:
        #print "IMPOSSING A 1000 OFFSET"
        #offset = 1000.0
        offset = 0
        #irregular grid
        xSize, ySize = data.shape[0:2]
        zSize = 1
        xyz = Object3DCTools.get3DGridFromXYZ(numpy.arange(xSize).astype(numpy.float32)-offset,
                                       numpy.arange(xSize).astype(numpy.float32)+offset,
                                       numpy.arange(1)+1)
        a = xyz[:,0] * 1
        xyz[:,0] = xyz[:,1] * 1
        xyz[:,1] = a[:]
        #print xyz[0:3,:]
        #print xyz.shape
        #print Object3DCTools.getVertexArrayMeshAxes(xyz)
        #sys.exit(0)
        data.shape = 1, xSize * ySize
        xyz[:,2] = data[:]
        object3D.setData(data, xyz=xyz)
    elif 0:
        #flat
        object3D.setData(data, z=4)
    else:
        #not flat
        object3D.setData(data, z=data)
    window.addObject(object3D, "Mesh")
    object3D = None

    window.glWidget.setZoomFactor(1.0)
    window.show()
    app.exec_()
