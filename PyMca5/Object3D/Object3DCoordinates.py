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
import OpenGL.GL  as GL
import OpenGL.GLU as GLU
import weakref

class Object3DCoordinates(object):
    def __init__(self, parent, limits = None):

        """
        The parent has to be a QGLWidget
        """

        self.parent = weakref.proxy(parent)
        self.limits = limits
        self.drawX  = True
        self.drawY  = True
        self.drawZ  = True
        self.delta  = 0.1

    def setLimits(self, limits):
        self.limits = limits

    def setFlags(self, xflag, yflag, zflag):
        self.drawX = xflag
        self.drawY = yflag
        self.drawZ = zflag

    def draw(self):
        if self.limits is None:
            return

        if self.drawX or self.drawY or self.drawZ:
            #get the viewport limits
            vlimits = self.convertToViewportLimits(self.limits)

            if self.drawX:
                self.drawXAxis(vlimits)
            if self.drawY:
                self.drawYAxis(vlimits)
            if self.drawZ:
                self.drawZAxis(vlimits)

    def convertToViewportLimits(self, limits):
        xmin, ymin, zmin = limits[0]
        xmax, ymax, zmax = limits[1]

        limits = []
        limits.append(GLU.gluProject(xmin, ymin, zmin))
        limits.append(GLU.gluProject(xmin, ymin, zmax))
        limits.append(GLU.gluProject(xmin, ymax, zmin))
        limits.append(GLU.gluProject(xmin, ymax, zmax))
        limits.append(GLU.gluProject(xmax, ymin, zmin))
        limits.append(GLU.gluProject(xmax, ymin, zmax))
        limits.append(GLU.gluProject(xmax, ymax, zmin))
        limits.append(GLU.gluProject(xmax, ymax, zmax))

        lminx, lminy, lminz = limits[0]
        lmaxx, lmaxy, lmaxz = limits[0]
        for i in range(len(limits)):
            x, y, z = limits[i]
            if x < lminx:
                lminx = x
            elif x > lmaxx:
                lmaxx = x
            if y < lminy:
                lminy = y
            elif y > lmaxy:
                lmaxy = y
            if z < lminz:
                lminz = z
            elif z > lmaxz:
                lmaxz = z

        return [[lminx, lminy, lminz], [lmaxx, lmaxy, lmaxz]]

    def drawXAxis(self, vlimits):
        xmin, ymin, zmin = self.limits[0]
        xmax, ymax, zmax = self.limits[1]
        difference_vector = self.limits[1] - self.limits[0]
        deltax, deltay, deltaz = self.delta * (difference_vector)


        begin = [[xmin, ymin - deltay, zmin - deltaz],
                 [xmin, ymax + deltay, zmin - deltaz],
                 [xmin, ymin - deltay, zmax + deltaz],
                 [xmin, ymax + deltay, zmax + deltaz]]

        end   = [[xmax, ymin - deltay, zmin - deltaz],
                 [xmax, ymax + deltay, zmin - deltaz],
                 [xmax, ymin - deltay, zmax + deltaz],
                 [xmax, ymax + deltay, zmax + deltaz]]
        X = []
        for i in range(4):
            X.append([GLU.gluProject(*begin[i]),
                      GLU.gluProject(*end[i])])

        #print "0->0, 1", vlimits[0][0], vlimits[1][0]
        #print "1->0, 1", vlimits[0][1], vlimits[1][1]

        possible = [1, 1, 1, 1]
        i = 0
        for item in X:
            if (item[0][0] > vlimits[0][0]) and (item[0][0] < vlimits[1][0]) and \
               (item[0][1] > vlimits[0][1]) and (item[0][1] < vlimits[1][1]):
                possible[i] = 0
            elif (item[1][0] > vlimits[0][0]) and (item[1][0] < vlimits[1][0]) and \
                 (item[1][1] > vlimits[0][1]) and (item[1][1] < vlimits[1][1]):
                possible[i] = 0
            i += 1

        #print "POSSIBLE = ", possible

        j = None
        for i in range(4):
            if possible[i]:
                if j is None:
                    j = i
                else:
                    if X[i][0][0] > X[j][0][0]:
                        j = i

        if j is not None:
            i = j
            GL.glColor3f(0.5, 0.0, 0.0) #RED for X Axis
            self.parent.renderText(begin[i][0], begin[i][1], begin[i][2],
                        "%.3f" % (begin[i][0]),
                        self.parent.font(), 2000)

            self.parent.renderText(end[i][0], end[i][1], end[i][2],
                        "%.3f" % (end[i][0]),
                        self.parent.font(), 2000)
            return


    def drawYAxis(self, vlimits):
        xmin, ymin, zmin = self.limits[0]
        xmax, ymax, zmax = self.limits[1]
        difference_vector = self.limits[1] - self.limits[0]
        deltax, deltay, deltaz = self.delta * (difference_vector)


        begin = [[xmin - deltax, ymin, zmin - deltaz],
                 [xmin - deltax, ymin, zmax + deltaz],
                 [xmax + deltax, ymin, zmin - deltaz],
                 [xmax + deltax, ymin, zmax + deltaz]]

        end   = [[xmin - deltax, ymax, zmin - deltaz],
                 [xmin - deltax, ymax, zmax + deltaz],
                 [xmax + deltax, ymax, zmin - deltaz],
                 [xmax + deltax, ymax, zmax + deltaz]]

        X = []
        for i in range(4):
            X.append([GLU.gluProject(*begin[i]),
                      GLU.gluProject(*end[i])])

        possible = [1, 1, 1, 1]
        i = 0
        for item in X:
            if (item[0][0] > vlimits[0][0]) and (item[0][0] < vlimits[1][0]) and \
               (item[0][1] > vlimits[0][1]) and (item[0][1] < vlimits[1][1]):
                possible[i] = 0
            elif (item[1][0] > vlimits[0][0]) and (item[1][0] < vlimits[1][0]) and \
                 (item[1][1] > vlimits[0][1]) and (item[1][1] < vlimits[1][1]):
                possible[i] = 0
            i += 1

        j = None
        for i in range(4):
            if possible[i]:
                if j is None:
                    j = i
                else:
                    if X[i][0][0] > X[j][0][0]:
                        j = i

        if j is not None:
                i = j
                GL.glColor3f(0.0, 0.5, 0.0) #GREEN for Y Axis
                self.parent.renderText(begin[i][0], begin[i][1], begin[i][2],
                            "%.3f" % (begin[i][1]),
                            self.parent.font(), 2000)

                self.parent.renderText(end[i][0], end[i][1], end[i][2],
                            "%.3f" % (end[i][1]),
                            self.parent.font(), 2000)
                return

    def drawZAxis(self, vlimits):
        xmin, ymin, zmin = self.limits[0]
        xmax, ymax, zmax = self.limits[1]
        difference_vector = self.limits[1] - self.limits[0]
        deltax, deltay, deltaz = self.delta * (difference_vector)


        begin = [[xmin - deltax, ymin - deltay, zmin],
                 [xmin - deltax, ymax + deltay, zmin],
                 [xmax + deltax, ymin - deltay, zmin],
                 [xmax + deltax, ymax + deltay, zmin]]

        end   = [[xmin - deltax, ymin - deltay, zmax],
                 [xmin - deltax, ymax + deltay, zmax],
                 [xmax + deltax, ymin - deltay, zmax],
                 [xmax + deltax, ymax + deltay, zmax]]

        X = []
        for i in range(4):
            X.append([GLU.gluProject(*begin[i]),
                      GLU.gluProject(*end[i])])

        possible = [1, 1, 1, 1]
        i = 0
        for item in X:
            if (item[0][0] > vlimits[0][0]) and (item[0][0] < vlimits[1][0]) and \
               (item[0][1] > vlimits[0][1]) and (item[0][1] < vlimits[1][1]):
                possible[i] = 0
            elif (item[1][0] > vlimits[0][0]) and (item[1][0] < vlimits[1][0]) and \
                 (item[1][1] > vlimits[0][1]) and (item[1][1] < vlimits[1][1]):
                possible[i] = 0
            i += 1

        j = None
        for i in range(4):
            if possible[i]:
                #if pow(pow(X[i][1][0] - X[i][0][0], 2) + pow(X[i][1][1] - X[i][0][1], 2),0.5) > 30:
                    if j is None:
                            j = i
                    else:
                        if X[i][0][0] > X[j][0][0]:
                            j = i

        if j is not None:
                i = j
                GL.glColor3f(0.0, 0.0, 0.5) #BLUE for Z Axis
                self.parent.renderText(begin[i][0], begin[i][1], begin[i][2],
                            "%.3f" % (begin[i][2]),
                            self.parent.font(), 2000)

                self.parent.renderText(end[i][0], end[i][1], end[i][2],
                            "%.3f" % (end[i][2]),
                            self.parent.font(), 2000)
                return



