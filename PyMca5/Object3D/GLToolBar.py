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
import sys
from . import Object3DQt as qt
from . import Object3DIcons
from .HorizontalSpacer import HorizontalSpacer


class GLToolBar(qt.QWidget):

    sigGLToolBarSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.build()

    def build(self):
        IconDict = Object3DIcons.IconDict
        self.cubeFrontIcon = qt.QIcon(qt.QPixmap(IconDict["cube_front"]))
        self.cubeBackIcon = qt.QIcon(qt.QPixmap(IconDict["cube_back"]))
        self.cubeTopIcon = qt.QIcon(qt.QPixmap(IconDict["cube_top"]))
        self.cubeBottomIcon = qt.QIcon(qt.QPixmap(IconDict["cube_bottom"]))
        self.cubeRightIcon = qt.QIcon(qt.QPixmap(IconDict["cube_right"]))
        self.cubeLeftIcon = qt.QIcon(qt.QPixmap(IconDict["cube_left"]))
        self.cube45Icon = qt.QIcon(qt.QPixmap(IconDict["cube_45"]))

        #the buttons
        self.cubeFront = qt.QToolButton(self)
        self.cubeFront.setIcon(self.cubeFrontIcon)

        self.cubeBack = qt.QToolButton(self)
        self.cubeBack.setIcon(self.cubeBackIcon)

        self.cubeTop = qt.QToolButton(self)
        self.cubeTop.setIcon(self.cubeTopIcon)

        self.cubeBottom = qt.QToolButton(self)
        self.cubeBottom.setIcon(self.cubeBottomIcon)

        self.cubeRight = qt.QToolButton(self)
        self.cubeRight.setIcon(self.cubeRightIcon)

        self.cubeLeft = qt.QToolButton(self)
        self.cubeLeft.setIcon(self.cubeLeftIcon)

        self.cube45 = qt.QToolButton(self)
        self.cube45.setIcon(self.cube45Icon)

        #the tool tips

        self.cubeFront.setToolTip("See from front (X+)")
        self.cubeBack.setToolTip("See from back (X-)")

        self.cubeTop.setToolTip("See from top (Z+)")
        self.cubeBottom.setToolTip("See from bottom (Z-)")

        self.cubeRight.setToolTip("See from right (Y+)")
        self.cubeLeft.setToolTip("See from left (Y-)")

        self.cube45.setToolTip("See from diagonal ( 1, 1, 1)")

        self.mainLayout.addWidget(self.cubeFront)
        self.mainLayout.addWidget(self.cubeBack)
        self.mainLayout.addWidget(self.cubeTop)
        self.mainLayout.addWidget(self.cubeBottom)
        self.mainLayout.addWidget(self.cubeRight)
        self.mainLayout.addWidget(self.cubeLeft)
        self.mainLayout.addWidget(self.cube45)

        self.cubeFront.clicked.connect(self.cubeFrontSlot)
        self.cubeBack.clicked.connect(self.cubeBackSlot)
        self.cubeTop.clicked.connect(self.cubeTopSlot)
        self.cubeBottom.clicked.connect(self.cubeBottomSlot)
        self.cubeRight.clicked.connect(self.cubeRightSlot)
        self.cubeLeft.clicked.connect(self.cubeLeftSlot)
        self.cube45.clicked.connect(self.cube45Slot)

    def cubeFrontSlot(self):
        self.applyCube('front')

    def cubeBackSlot(self):
        self.applyCube('back')

    def cubeTopSlot(self):
        self.applyCube('top')

    def cubeBottomSlot(self):
        self.applyCube('bottom')

    def cubeRightSlot(self):
        self.applyCube('right')

    def cubeLeftSlot(self):
        self.applyCube('left')

    def cube45Slot(self):
        self.applyCube('d45')

    def applyCube(self, cubeFace):
        ddict = {}
        ddict['event'] = 'ApplyCubeClicked'
        ddict['face'] = cubeFace
        self.sigGLToolBarSignal.emit(ddict)
        #print "to be implemented"

if __name__ == "__main__":
    app = qt.QApplication([])
    w = GLToolBar()
    app.lastWindowClosed.connect(app.quit)
    w.show()
    app.exec_()
