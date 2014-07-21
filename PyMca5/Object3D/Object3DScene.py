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
    from PyMca5.Object3D import SceneGLWindow
    #This is needed for the mesh plots
    from PyMca5.Object3D.Object3DPlugins import Object3DMesh
    #This is needed for the stack plots
    from PyMca5.Object3D.Object3DPlugins import Object3DStack
except:
    from . import SceneGLWindow
    #This is needed for the mesh plots
    from .Object3DPlugins import Object3DMesh
    #This is needed for the stack plots
    from .Object3DPlugins import Object3DStack

class Object3DScene(SceneGLWindow.SceneGLWindow):
    """
    This class just adds a set of simple commands that can be accessed
    in interactive mode from ipython -q4thread
    """
    def mesh(self, data, x=None, y=None, z=None, xyz=None, legend=None,update_scene=True):
        """
        mesh(self, data, x=None, y=None, z=None, xyz=None, legend=None)
        The legend is optional. It is the name of the generated mesh.

        mesh(self, data)

        Generate a mesh plot according to data dimensions

        mesh(self, data, x=x, y=y)

        Generate a REGULAR 2D mesh plot on the x y grid.
        length_of_x *  length_of_y = length_of_data
        the z coordinate is set to zero

        mesh(self, data, x=x, y=y, z=10)
        Idem to previous with the z coordinate set to 10

        mesh(self, data, x=x, y=y, z=data[:])
        Mesh plot in 3D
        length_of_x *  length_of_x = length_of_data
        length_of_z = length_of_data

        mesh(self, data, xyz=vertices)
        data has n 3D points
        vertices.shape = n, 3
        """
        if legend is None:
            legend = "Mesh"
        o3d = Object3DMesh.Object3DMesh(legend)
        o3d.setData(data, x=x, y=y, z=z, xyz=xyz)
        self.addObject(o3d, legend, update_scene=update_scene)
        return o3d

    def stack(self, data, x=None, y=None, z=None, legend=None, update_scene=True):
        """
        I should find a better name for this method ...
        stack(self, data, x=None, y=None, z=None, legend=None)

        The goal of this method is to provide a way to handle a
        scalar value measured in a 3-dimensional grid.

        data contains the measured values (use floats)

        The x, y and z values give the values used to generate the grid

        x_length * y_length * z_length = data_length

        If those values are not present, they are generated from the
        dimensions of the data array.

        The legend is optional. It is the name of the generated object.
        """
        o3d = Object3DStack.Object3DStack(legend)
        o3d.setData(data, x=x, y=y, z=z)
        self.addObject(o3d, legend, update_scene=update_scene)
        return o3d


if __name__ == "__main__":
    import numpy
    import sys
    qt = SceneGLWindow.qt
    app = qt.QApplication([])
    scene = Object3DScene()
    scene.show()
    if len(sys.argv) > 1:
        x = numpy.arange(-2,2,0.01).astype(numpy.float32)
        y = numpy.arange(-2,2,0.01).astype(numpy.float32)
        xsize = len(x)
        ysize = len(y)
        #generate the vertices
        vertices = numpy.zeros((xsize * ysize, 3), numpy.float32)

        #generate the grid using simple math operations
        A=numpy.outer(x, numpy.ones(len(y), numpy.float32))
        B=numpy.outer(y, numpy.ones(len(x), numpy.float32))

        #fill the x and y values
        vertices[:,0]=A.flatten()
        vertices[:,1]=B.transpose().flatten()

        #calculate the data values
        for i in range(xsize):
            vertices[(i*ysize):((i+1)*ysize),2]= 2 * x[i] * numpy.exp(-x[i]**2-y**2)
        z = vertices[:,2]
        scene.mesh(z, x=x, y=y, z=z)
    else:
        #example from:
        #http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/mlab.html#id1
        x, y, z = numpy.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
        s = numpy.sin(x*y*z)/(x*y*z)
        scene.stack(s, x=x, y=y, z=z, legend='demo')
    app.exec_()
