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
import sys
from . import Object3DQt as qt
from . import SceneTree

QTVERSION = qt.qVersion()
DEBUG = 0

class SceneWidget(qt.QWidget):
    sigSceneWidgetSignal = qt.pyqtSignal(object)

    def __init__(self, parent = None, scene = None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle('Scene Widget')
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        if scene is None:
            tree = None
        else:
            tree = scene.tree
        self.treeWidget = SceneTree.Object3DObjectTree(self, tree=tree)
        self.tree = self.treeWidget.tree
        self.mainLayout.addWidget(self.treeWidget)

        self.treeWidget.sigObjectTreeSignal.connect(self._treeWidgetSignal)

    def _treeWidgetSignal(self, ddict):
        self.emitSignal(ddict['event'], ddict)

    def emitSignal(self, event, ddict = None):
        if ddict is None:
            ddict = {}
        ddict['event'] = event
        self.sigSceneWidgetSignal.emit(ddict)

    def updateView(self, expand=True):
        return self.treeWidget.updateView(expand=expand)

    def setSelectedObject(self, name=None):
        return self.treeWidget.setSelectedObject(name)

if __name__ == "__main__":
    app = qt.QApplication([])
    w = SceneWidget()
    w.show()
    app.exec_()
