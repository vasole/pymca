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
isGlAvailable = False
try:
    import OpenGL
except ImportError:
    _logger.debug("pyopengl not installed")
else:
    # sanity check from silx.gui._glutils.OpenGLWidget
    if not hasattr(silx_qt, 'QOpenGLWidget') and\
            (not silx_qt.HAS_OPENGL or
             silx_qt.QApplication.instance() and not silx_qt.QGLFormat.hasOpenGL()):
        _logger.debug("qt has a QOpenGLWidget: %s", hasattr(silx_qt, 'QOpenGLWidget'))
        _logger.debug("qt.HAS_OPENGL: %s", silx_qt.HAS_OPENGL)
        _logger.debug("silx_qt.QGLFormat.hasOpenGL(): %s",
                      silx_qt.QApplication.instance() and not silx_qt.QGLFormat.hasOpenGL())

    else:
        isGlAvailable = True

_logger.debug("Gl availability: %s", isGlAvailable)


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
                # checks consistency of number of data points (but accepts scalars)
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
        tooltip = "Stack as a scatter plot, using positioners as axes"
        self.methodDict = {'Show (mpl)': [self._showWidgetMpl,
                                          tooltip + " (matplotlib backend)",
                                          None]}
        self.__methodKeys = ['Show (mpl)']
        self._backends = ['mpl']
        if isGlAvailable:
            self.methodDict['Show (gl)'] = [self._showWidgetGl,
                                            tooltip + " (OpenGL backend)",
                                            None]
            self.__methodKeys.append('Show (gl)')
            self._backends.append('gl')

        self._scatterViews = {"gl": None,
                              "mpl": None}
        self._axesSelectors = {"gl": None,
                               "mpl": None}

        self._xdata = {"gl": None,
                       "mpl": None}
        self._ydata = {"gl": None,
                       "mpl": None}

    def _buildWidget(self, backend):
        scatterView = ScatterView(parent=None, backend=backend)
        self._scatterViews[backend] = scatterView

        self._setData(backend)
        scatterView.resetZoom()
        callback = self._scatterMaskChangedGl if backend == "gl" else self._scatterMaskChangedMpl
        scatterView.getMaskToolsWidget().sigMaskChanged.connect(
                callback)

        self._axesSelectors[backend] = AxesPositionersSelector(parent=scatterView)
        _axesSelectorDock = BoxLayoutDockWidget()
        _axesSelectorDock.setWindowTitle('Axes selection')
        _axesSelectorDock.setWidget(self._axesSelectors[backend])
        scatterView.addDockWidget(qt.Qt.BottomDockWidgetArea, _axesSelectorDock)

        self._axesSelectors[backend].setNumPoints(self._getNumStackPoints())
        self._axesSelectors[backend].fillPositioners(self._getStackPositioners())

        callback = self._setAxesDataGl if backend == "gl" else self._setAxesDataMpl
        self._axesSelectors[backend].sigSelectionChanged.connect(callback)

    def _showWidgetMpl(self):
        self._showWidget(backend="mpl")

    def _showWidgetGl(self):
        self._showWidget(backend="gl")

    def _showWidget(self, backend):
        if self._scatterViews[backend] is None:
            self._buildWidget(backend=backend)

        # Show
        self._scatterViews[backend].show()
        self._scatterViews[backend].raise_()

        # Draw mask, if any
        self.selectionMaskUpdated()

    def _setAxesDataMpl(self, xPositioner, yPositioner):
        self._setAxesData(xPositioner, yPositioner, "mpl")

    def _setAxesDataGl(self, xPositioner, yPositioner):
        self._setAxesData(xPositioner, yPositioner, "gl")

    def _setAxesData(self, xPositioner, yPositioner, backend):
        """

        :param str xPositioner: motor name, or None
        :param str yPositioner: motor name, or None
        :return:
        """
        positioners = self._getStackPositioners()
        if xPositioner is not None:
            assert xPositioner in positioners
            self._xdata[backend] = positioners[xPositioner]
        else:
            self._xdata[backend] = None
        if yPositioner is not None:
            assert yPositioner in positioners
            self._ydata[backend] = positioners[yPositioner]
        else:
            self._ydata[backend] = None
        self._setData(backend)

    @contextmanager
    def _scatterMaskDisconnected(self, backend):
        # This context manager allows to call self.setStackSelectionMask
        # without entering an infinite loop, by temporarily disconnecting
        # callbacks from our mask signals.
        callback = self._scatterMaskChangedGl if backend == "gl" else self._scatterMaskChangedMpl
        self._scatterViews[backend].getMaskToolsWidget().sigMaskChanged.disconnect(
                callback)
        try:
            yield
        finally:
            self._scatterViews[backend].getMaskToolsWidget().sigMaskChanged.connect(
                    callback)

    def _setData(self, backend):
        stack_images, stack_names = self.getStackROIImagesAndNames()
        nrows, ncols = stack_images[0].shape

        # flatten image
        stackValues = stack_images[0].reshape((-1,))
        # get regular grid coordinates as a 1D array
        if self._xdata[backend] is None or self._ydata[backend] is None:
            defaultX, defaultY = numpy.meshgrid(numpy.arange(ncols),
                                                numpy.arange(nrows))
            defaultX.shape = stackValues.shape
            defaultY.shape = stackValues.shape

        xdata = self._xdata[backend] if self._xdata[backend] is not None else defaultX
        ydata = self._ydata[backend] if self._ydata[backend] is not None else defaultY

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

        self._scatterViews[backend].setData(xdata, ydata, stackValues,
                                            copy=False)
        self._scatterViews[backend].resetZoom()

    def _isScatterViewVisible(self, backend):
        if self._scatterViews[backend] is None:
            return False
        if self._scatterViews[backend].isHidden():
            return False
        return True

    def _scatterMaskChangedGl(self):
        self._scatterMaskChanged("gl")

    def _scatterMaskChangedMpl(self):
        self._scatterMaskChanged("mpl")

    def _scatterMaskChanged(self, backend):
        scattermask = self._scatterViews[backend].getMaskToolsWidget().getSelectionMask()
        if scattermask is not None:
            shape = self.getStackOriginalImage().shape
            mask = scattermask.reshape(shape)
        else:
            mask = scattermask
        with self._scatterMaskDisconnected(backend):
            self.setStackSelectionMask(mask)

    def _getNumStackPoints(self):
        stack_images, stack_names = self.getStackROIImagesAndNames()
        return stack_images[0].size

    def _getStackPositioners(self):
        info = self.getStackInfo()
        return info.get("positioners", {})

    def stackUpdated(self):
        for backend in self._backends:
            if not self._isScatterViewVisible(backend):
                return
            self._setData(backend)
            self._axesSelectors[backend].setNumPoints(self._getNumStackPoints())
            self._axesSelectors[backend].fillPositioners(self._getStackPositioners())

    def selectionMaskUpdated(self):
        for backend in self._backends:
            if not self._isScatterViewVisible(backend):
                return
            mask = self.getStackSelectionMask()
            if mask is not None:
                scatterMask = mask.reshape((-1,))
            else:
                scatterMask = None
            self._scatterViews[backend].getMaskToolsWidget().setSelectionMask(scatterMask)

    def stackClosed(self):
        for sv in self._scatterViews.values():
            if sv is not None:
                sv.close()

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


