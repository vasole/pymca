# /*#########################################################################
# Copyright (C) 2017-2019 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""If positioners data is available for this stack, this plugin opens
a stack plot with a widget displaying motor positions at the current mouse
position in the plot."""

__authors__ = ["P. Knobel"]
__contact__ = "sole@esrf.fr"
__license__ = "MIT"


from PyMca5 import StackPluginBase
import PyMca5.PyMcaGui.PyMcaQt as qt
from PyMca5.PyMcaGui.pymca import StackROIWindow
try:
    from PyMca5.PyMcaPlugins import MotorInfoWindow
except ImportError:
    from . import MotorInfoWindow

from PyMca5.PyMcaGui.plotting import MaskImageTools


class PointInfoWindow(qt.QWidget):
    """Display an image next to a MotorInfoWindow showing the values
    from `info["positioners"]` associated with the data underneath
    the mouse cursor."""
    def __init__(self, parent=None, plugin=None):
        super(PointInfoWindow, self).__init__(parent)
        assert isinstance(plugin, StackPluginBase.StackPluginBase)
        self.plugin = plugin

        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.maskImageWidget = StackROIWindow.StackROIWindow(self,
                                                        crop=False,
                                                        rgbwidget=None,
                                                        selection=True,
                                                        colormap=True,
                                                        imageicons=True,
                                                        standalonesave=True,
                                                        profileselection=True)
        self.maskImageWidget.setSelectionMode(True)

        # self.maskImageWidget.setWindowFlags(qt.Qt.W)

        self.motorPositionsWindow = MotorInfoWindow.MotorInfoDialog(self,
                                                                    ["Stack"],
                                                                    [{}])
        self.motorPositionsWindow.setMaximumHeight(120)

        self.maskImageWidget.sigMaskImageWidgetSignal.connect( \
                                    self.onMaskImageWidgetSignal)
        self.maskImageWidget.graph.sigPlotSignal.connect(self._updateMotors)

        layout.addWidget(self.maskImageWidget)
        layout.addWidget(self.motorPositionsWindow)

        self._first_update = True

    def onMaskImageWidgetSignal(self, ddict):
        """triggered by self.widget.sigMaskImageWidget"""
        if ddict['event'] == "selectionMaskChanged":
            self.plugin.setStackSelectionMask(ddict["current"])
        elif ddict['event'] == "removeImageClicked":
            self.plugin.removeImage(ddict['title'])
        elif ddict['event'] == "addImageClicked":
            self.plugin.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.plugin.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.plugin.setStackSelectionMask(None)

    def _updateMotors(self, ddict):
        if not ddict["event"] == "mouseMoved":
            return

        motorsValuesAtCursor = self.plugin.getPositionersFromXY(ddict["x"],
                                                                ddict["y"])

        self.motorPositionsWindow.table.updateTable(
                legList=["Stack"],
                motList=[motorsValuesAtCursor])

        if self._first_update:
            self._select_motors()
            self._first_update = False

    def _select_motors(self):
        """This methods sets the motors in the comboboxes when the widget
        is first initialized."""
        for i, combobox in enumerate(self.motorPositionsWindow.table.header.boxes):
            # First item (index 0) in combobox is "", so first motor name is at index 1.
            # First combobox in header.boxes is at index 1 (boxes[0] is None).
            if i == 0:
                continue
            if i < combobox.count():
                combobox.setCurrentIndex(i)


class StackMotorInfoPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow)
        self.methodDict = {'Show motor positions':
                               [self._showWidgets, "Show motor positions in a popup window"], }
        self.__methodKeys = ['Show motor positions']
        self.widget = None

    def stackClosed(self):
        if self.widget is not None:
            self.widget.close()

    def _getStackOriginDelta(self):
        """Return (originX, originY) and (deltaX, deltaY)
        """
        info = self.getStackInfo()

        xscale = info.get("xScale", [0.0, 1.0])
        yscale = info.get("yScale", [0.0, 1.0])

        origin = xscale[0], yscale[0]
        delta = xscale[1], yscale[1]

        return origin, delta

    def stackUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return

        images, names = self.getStackROIImagesAndNames()

        image_shape = list(self.getStackOriginalImage().shape)

        info = self.getStackInfo()
        xScale = (0.0, 1.0) # info["xScale"]
        yScale = (0.0, 1.0) # info["yScale"]

        self.widget.maskImageWidget.setImageList(images,
                                    imagenames=names,
                                    #xScale=xScale,
                                    #yScale=yScale,
                                    dynamic=False)

        self.widget.maskImageWidget.setSelectionMask(self.getStackSelectionMask())

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        mask = self.getStackSelectionMask()
        if not self.widget.maskImageWidget.isHidden():
            self.widget.maskImageWidget.setSelectionMask(mask)

    def _showWidgets(self):
        if "positioners" not in self.getStackInfo():
            msg = qt.QMessageBox()
            msg.setWindowTitle("No positioners")
            msg.setIcon(qt.QMessageBox.Information)
            msg.setInformativeText("No positioners are set for this stack.")
            msg.raise_()
            msg.exec()
            return
        if self.widget is None:
            self.widget = PointInfoWindow(plugin=self)

        # Show
        self.widget.show()
        self.widget.raise_()

        self.stackUpdated()    # fixme: is this necessary?

    def getPositionersFromXY(self, x, y):
        """Return positioner values for a stack pixel identified
        by it's (x, y) coordinates.
        """
        nRows, nCols = self.getStackOriginalImage().shape
        
        info = self.getStackInfo()
        xScale = (0.0, 1.0) # info["xScale"]
        yScale = (0.0, 1.0) # info["yScale"]
        r, c = MaskImageTools.convertToRowAndColumn(x, y,
                                                    shape=(nRows, nCols),
                                                    xScale=xScale,
                                                    yScale=yScale,
                                                    safe=True)
        idx1d = r * nCols + c
        return self._stackWindow.getPositionersFromIndex(idx1d)

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        if len(self.methodDict[name]) < 3:
            return None
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()


MENU_TEXT = "Stack Motor Positions"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackMotorInfoPlugin(stackWindow)
    return ob
