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

from PyMca5.PyMcaGui import PyMcaQt    # just to be sure PyMcaQt is imported before silx.gui.qt
from silx.gui import qt as silx_qt

from PyMca5 import StackPluginBase

from silx.gui.plot.ScatterView import ScatterView

_logger = logging.getLogger(__name__)


# Probe OpenGL availability and widget
backend = "mpl"
try:
    import OpenGL
except ImportError:
    backend = "mpl"
else:
    # sanity check from silx.gui._glutils.OpenGLWidget
    if not hasattr(silx_qt, 'QOpenGLWidget') or\
            not silx_qt.HAS_OPENGL or \
            (silx_qt.QApplication.instance() and not silx_qt.QGLFormat.hasOpenGL()):
        backend = "mpl"
    else:
        backend = "gl"

_logger.info("Using backend %s", backend)


class MaskScatterViewPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'Show': [self._showWidget,
                                    "Show ROIs",
                                    None]}
        self.__methodKeys = ['Show']
        self._scatterView = None

    def _showWidget(self):
        if self._scatterView is None:
            self._scatterView = ScatterView(parent=None, backend=backend)
            self._setData()
            self._scatterView.resetZoom()
            self._scatterView.getMaskToolsWidget().sigMaskChanged.connect(
                    self._scatterMaskChanged)

        # Show
        self._scatterView.show()
        self._scatterView.raise_()

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
        defaultX, defaultY = numpy.meshgrid(numpy.arange(ncols),
                                            numpy.arange(nrows))
        defaultX.shape = stackValues.shape
        defaultY.shape = stackValues.shape

        self._scatterView.setData(defaultX, defaultY, stackValues,
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


