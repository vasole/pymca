#/*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
#############################################################################*/
"""
This plugin opens a widget to view a stack as a scatter plot, by using
positioner data as X and Y coordinates.

"""
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import logging
import numpy
from contextlib import contextmanager

from PyMca5.PyMcaGui import PyMcaQt as qt    # just to be sure PyMcaQt is imported before silx.gui.qt
from silx.gui import qt as silx_qt

from PyMca5 import StackPluginBase

from silx.gui.plot.ScatterView import ScatterView
from silx.gui.widgets.BoxLayoutDockWidget import BoxLayoutDockWidget

_logger = logging.getLogger(__name__)
# _logger.setLevel(logging.DEBUG)


# Probe OpenGL availability and widget
backend = "mpl"
try:
    import OpenGL
except ImportError:
    backend = "mpl"
    _logger.debug("pyopengl not installed")
else:
    # sanity check from silx.gui._glutils.OpenGLWidget
    if not hasattr(silx_qt, 'QOpenGLWidget') and\
            (not silx_qt.HAS_OPENGL or
             silx_qt.QApplication.instance() and not silx_qt.QGLFormat.hasOpenGL()):
        backend = "mpl"
        _logger.debug("qt has a QOpenGLWidget: %s", hasattr(silx_qt, 'QOpenGLWidget'))
        _logger.debug("qt.HAS_OPENGL: %s", silx_qt.HAS_OPENGL)
        _logger.debug("silx_qt.QGLFormat.hasOpenGL(): %s",
                      silx_qt.QApplication.instance() and not silx_qt.QGLFormat.hasOpenGL())

    else:
        backend = "gl"

_logger.debug("Using backend %s", backend)


class AxesPositionersSelector(qt.QWidget):
    sigSelectionChanged = qt.pyqtSignal(object, object)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        hlayout = qt.QHBoxLayout()
        self.setLayout(hlayout)

        xlabel = qt.QLabel("X:", parent=parent)
        self.xPositioner = qt.QComboBox(parent)
        self.xPositioner.currentIndexChanged.connect(self._emitSelectionChanged)

        ylabel = qt.QLabel("Y:", parent=parent)
        self.yPositioner = qt.QComboBox(parent)
        self.yPositioner.currentIndexChanged.connect(self._emitSelectionChanged)

        hlayout.addWidget(xlabel)
        hlayout.addWidget(self.xPositioner)
        hlayout.addWidget(ylabel)
        hlayout.addWidget(self.yPositioner)

        self._nPoints = None
        """If set to an integer, only motors with this number of data points
        can be added."""

        self._initComboBoxes()

    def _initComboBoxes(self):
        self.xPositioner.clear()
        self.xPositioner.insertItem(0, "None")
        self.yPositioner.clear()
        self.yPositioner.insertItem(0, "None")

    def _emitSelectionChanged(self, idx):
        self.sigSelectionChanged.emit(*self.getSelectedPositioners())

    def setNumPoints(self, n):
        self._nPoints = n

    def unsetNumPoints(self):
        self._nPoints = None

    def fillPositioners(self, positioners):
        """

        :param dict positioners: Dictionary of positioners
            The key is the motor name, the value are the motor's position data
        """
        self._initComboBoxes()
        i = 0
        for motorName, motorValues in positioners.items():
            if not numpy.isscalar(motorValues) and self._nPoints is not None and self._nPoints != motorValues.size:
                continue
            else:
                i += 1
                self.xPositioner.insertItem(i, motorName)
                self.yPositioner.insertItem(i, motorName)

    def getSelectedPositioners(self):
        """

        :return: 2-tuple of selected positioner names (or None)
        """
        selected = [None, None]
        if self.xPositioner.currentText() != "None":
            selected[0] = self.xPositioner.currentText()
        if self.yPositioner.currentText() != "None":
            selected[1] = self.yPositioner.currentText()
        return selected


class MaskScatterViewPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show': [self._showWidget,
                                    "Show ROIs",
                                    None]}
        self.__methodKeys = ['Show']
        self._scatterView = None

        self._xdata = None
        self._ydata = None

    def _buildWidget(self):
        self._scatterView = ScatterView(parent=None, backend=backend)
        self._setData()
        self._scatterView.resetZoom()
        self._scatterView.getMaskToolsWidget().sigMaskChanged.connect(
                self._scatterMaskChanged)

        self._axesSelector = AxesPositionersSelector(parent=self._scatterView)
        self._axesSelectorDock = BoxLayoutDockWidget()
        self._axesSelectorDock.setWindowTitle('Axes selection')
        self._axesSelectorDock.setWidget(self._axesSelector)
        self._scatterView.addDockWidget(qt.Qt.BottomDockWidgetArea, self._axesSelectorDock)

        self._axesSelector.fillPositioners(self._getStackPositioners())
        self._axesSelector.sigSelectionChanged.connect(self._setAxesData)

    def _showWidget(self):
        if self._scatterView is None:
            self._buildWidget()

        # Show
        self._scatterView.show()
        self._scatterView.raise_()

    def _getStackPositioners(self):
        # info = self.getStackInfo()
        # return info.get("positioners", {})
        stack_images, stack_names = self.getStackROIImagesAndNames()
        shape2d = stack_images[0].shape
        return {"toto": numpy.arange(stack_images[0].size) ** 1.2,
                "tata": 3.14,
                "pipo": numpy.arange(stack_images[0].size).reshape(shape2d) ** 0.5}

    def _setAxesData(self, xPositioner, yPositioner):
        """

        :param str xPositioner: motor name, or None
        :param str yPositioner: motor name, or None
        :return:
        """
        positioners = self._getStackPositioners()
        if xPositioner is not None:
            assert xPositioner in positioners
            self._xdata = positioners[xPositioner]
        else:
            self._xdata = None
        if yPositioner is not None:
            assert yPositioner in positioners
            self._ydata = positioners[yPositioner]
        else:
            self._ydata = None
        self._setData()

    @contextmanager
    def _scatterMaskDisconnected(self):
        # This context manager allows to call self.setStackSelectionMask
        # without entering an infinite loop, by temporarily disconnecting
        # callbacks from our mask signals.
        self._scatterView.getMaskToolsWidget().sigMaskChanged.disconnect(
                self._scatterMaskChanged)
        try:
            yield
        finally:
            self._scatterView.getMaskToolsWidget().sigMaskChanged.connect(
                    self._scatterMaskChanged)

    def _setData(self):
        stack_images, stack_names = self.getStackROIImagesAndNames()
        nrows, ncols = stack_images[0].shape

        # flatten image
        stackValues = stack_images[0].reshape((-1,))
        # get regular grid coordinates as a 1D array
        if self._xdata is None or self._ydata is None:
            defaultX, defaultY = numpy.meshgrid(numpy.arange(ncols),
                                                numpy.arange(nrows))
            defaultX.shape = stackValues.shape
            defaultY.shape = stackValues.shape

        xdata = self._xdata if self._xdata is not None else defaultX
        ydata = self._ydata if self._ydata is not None else defaultY

        if numpy.isscalar(xdata):
            xdata = xdata * numpy.ones_like(stackValues)
            _logger.debug("converting scalar to constant 1D array for x")
        elif len(xdata.shape) > 1:
            _logger.debug("flattening %s array", str(xdata.shape))
            xdata = xdata.reshape((-1,))

        if numpy.isscalar(ydata):
            ydata = ydata * numpy.ones_like(stackValues)
            _logger.debug("converting scalar to constant 1D array for y")
        elif len(ydata.shape) > 1:
            _logger.debug("flattening %s array", str(ydata.shape))
            ydata = ydata.reshape((-1,))

        self._scatterView.setData(xdata, ydata, stackValues,
                                  copy=False)

    def _isScatterViewVisible(self):
        if self._scatterView is None:
            return False
        if self._scatterView.isHidden():
            return False
        return True

    def _scatterMaskChanged(self):
        scattermask = self._scatterView.getMaskToolsWidget().getSelectionMask()
        if scattermask is not None:
            shape = self.getStackOriginalImage().shape
            mask = scattermask.reshape(shape)
        else:
            mask = scattermask
        with self._scatterMaskDisconnected():
            self.setStackSelectionMask(mask)

    def stackUpdated(self):
        if not self._isScatterViewVisible():
            return
        self._setData()
        self._axesSelector.fillPositioners(self._getStackPositioners())

    def selectionMaskUpdated(self):
        if not self._isScatterViewVisible():
            return
        mask = self.getStackSelectionMask()
        scatterMask = mask.reshape((-1,))
        self._scatterView.getMaskToolsWidget().setSelectionMask(scatterMask)

    def stackClosed(self):
        if self._scatterView is not None:
            self._scatterView.close()

    def stackROIImageListUpdated(self):
        self.stackUpdated()

    # Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()


MENU_TEXT = "Mask Scatter View"


def getStackPluginInstance(stackWindow, **kw):
    ob = MaskScatterViewPlugin(stackWindow)
    return ob


