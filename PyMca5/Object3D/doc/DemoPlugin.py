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
from Object3D import Object3DBase
from OpenGL.GL import *

class Object3DDemo(Object3DBase.Object3D):
    def __init__(self, name="Demo"):
        Object3DBase.Object3D.__init__(self, name)

        # I have to give the limits I am going to use in order
        # to calculate a proper bounding box
        self.setLimits(-1.0, 0.0, 0.0, 1.0, 1.0, 0.0)

    def drawObject(self):
        # this is to handle transparency
        alpha = 1. - self._configuration['common']['transparency']

        #some simple drawing
        glShadeModel(GL_SMOOTH)
        glBegin(GL_TRIANGLES)
        glColor4f(  1.0, 0.0, 0.0, alpha)
        glVertex3f(-1.0, 0.0, 0.0)
        glColor4f(  0.0, 1.0, 0.0, alpha)
        glVertex3f( 0.0, 1.0, 0.0)
        glColor4f(  0.0, 0.0, 1.0, alpha)
        glVertex3f( 1.0, 0.0, 0.0)
        glEnd()

MENU_TEXT = 'Demo'
def getObject3DInstance(config=None):
    return Object3DDemo()
