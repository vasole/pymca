#/*##########################################################################
# Copyright (C) 2018 European Synchrotron Radiation Facility
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
__license__ = "MIT"

import numpy
import logging

from PyMca5.PyMcaGui import PyMcaQt as qt

from silx.gui.plot3d import SceneWindow
from silx.gui import icons

from silx.math.calibration import ArrayCalibration

from PyMca5.Object3D.Object3DPlugins.ChimeraStack import getObject3DInstance as getChimeraObj3D
from PyMca5.Object3D.Object3DPlugins.Object3DStack import getObject3DInstance as get4DStackObj3D
from PyMca5.Object3D.Object3DPlugins.Object3DPixmap import getObject3DInstance as getPixmapObj3D
from PyMca5.Object3D.Object3DPlugins.Object3DMesh import getObject3DInstance as get3DMeshObj3D





_logger = logging.getLogger(__name__)


class OpenAction(qt.QAction):
    """This action opens a menu with sub-actions to load data from a file,
    build a dataObject and add it to a :class:`SceneGlWindow`.
    """
    def __init__(self, parent=None, sceneGlWindow=None):
        """

        :param QWidget parent: Parent widget
        :param SceneGLWindow sceneGlWindow: :class:`SceneGlWindow` displaying
            the data.
        """
        super(OpenAction, self).__init__(parent)
        self._sceneGlWindow = sceneGlWindow

        self.setIcon(icons.getQIcon("document-open"))
        self.setText("Load data from a file")
        self.setCheckable(False)
        self.triggered[bool].connect(self._openMenu)

    def _openMenu(self, checked):
        # note: opening a context menu over a QGLWidget causes a warning (fixed in Qt 5.4.1)
        #       See: https://bugreports.qt.io/browse/QTBUG-42464
        menu = qt.QMenu(self._sceneGlWindow)

        loadPixmapAction = qt.QAction("Pixmap", self)
        loadPixmapAction.triggered[bool].connect(self._onLoadPixmap)
        menu.addAction(loadPixmapAction)

        load3DMeshAction = qt.QAction("3D mesh", self)
        load3DMeshAction.triggered[bool].connect(self._onLoad3DMesh)
        menu.addAction(load3DMeshAction)

        load4DStackAction = qt.QAction("4D stack", self)
        load4DStackAction.triggered[bool].connect(self._onLoad4DStack)
        menu.addAction(load4DStackAction)

        loadChimeraAction = qt.QAction("4D chimera", self)
        loadChimeraAction.triggered[bool].connect(self._onLoadChimeraStack)
        menu.addAction(loadChimeraAction)

        a = menu.exec_(qt.QCursor.pos())

    def _onLoadPixmap(self, checked):
        self._load(getPixmapObj3D)

    def _onLoad3DMesh(self, checked):
        self._load(get3DMeshObj3D)

    def _onLoad4DStack(self, checked):
        self._load(get4DStackObj3D)

    def _onLoadChimeraStack(self, checked):
        self._load(getChimeraObj3D)

    def _load(self, method):
        """

        :param method: Callable returning an Object3D instance or None
        """
        ob3d = method()
        if ob3d is not None:
            self._sceneGlWindow.addDataObject(ob3d)


class SceneGLWindow(SceneWindow.SceneWindow):
    def __init__(self, parent=None):
        super(SceneGLWindow, self).__init__(parent)

        self._openAction = OpenAction(parent=self.getOutputToolBar(),
                                      sceneGlWindow=self)
        # insert before first action
        self.getOutputToolBar().insertAction(
                self.getOutputToolBar().actions()[0],
                self._openAction)

    def _addSelection(self, selectionlist):
        _logger.debug("addSelection(self, selectionlist=%s)", selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            legend = sel['legend']  # expected form sourcename + scan key
            dataObject = sel['dataobject']
            # one-dimensional selections not considered
            if dataObject.info["selectiontype"] == "1D":
                continue

            # there must be something to plot
            if not hasattr(dataObject, 'y'):
                continue
            # there must be an x for a scan selection to reach here
            if not hasattr(dataObject, 'x'):
                continue

            # we have to loop for all y values
            for ycounter, ydata in enumerate(dataObject.y):
                ylegend = 'y%d' % ycounter
                if sel['selection'] is not None:
                    if type(sel['selection']) == type({}):
                        if 'y' in sel['selection']:
                            ilabel = sel['selection']['y'][ycounter]
                            ylegend = dataObject.info['LabelNames'][ilabel]
                object3Dlegend = legend + " " + ylegend
                self.addDataObject(dataObject,
                                   legend=object3Dlegend)

    def _removeSelection(self, selectionlist):
        _logger.debug("_removeSelection(self, selectionlist=%s)", selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        items = self.getSceneWidget().getItems()

        for sel in sellist:
            legend = sel['legend']
            if 'LabelNames' in sel['selection']:
                labelNames = sel['selection']['LabelNames']
            else:
                labelNames = sel['selection']['cntlist']
            for ycounter in sel['selection']['y']:
                ylegend = labelNames[ycounter]
                object3Dlegend = legend + " " + ylegend
                for it in items:
                    if it.getLabel() == object3Dlegend:
                        self.getSceneWidget().removeItem(it)

    def _replaceSelection(self, selectionlist):
        _logger.debug("_replaceSelection(self, selectionlist=%s)", selectionlist)
        self.getSceneWidget().clearItems()
        self._addSelection(selectionlist)

    def addDataObject(self, dataObject, legend=None):
        if legend is None:
            legend = dataObject.info['legend']

        if dataObject.m is None or dataObject.m == []:
            data = dataObject.y[0]
        else:
            # I would have to check for the presence of zeros in monitor
            data = dataObject.y[0] / dataObject.m[0]

        if dataObject.x is None:
            if len(data.shape) == 3:
                item3d = self.getSceneWidget().add3DScalarField(data)
            elif len(data.shape) == 2:
                item3d = self.getSceneWidget().addImage(data)
            else:
                item3d = self.getSceneWidget().mesh(data)    # TODO: add image as height map
            item3d.setLabel(legend)
            return

        ndata = numpy.prod(data.shape)

        xDimList = []
        for dataset in dataObject.x:
            xdim = numpy.prod(dataset.shape)
            xDimList.append(xdim)

        # case with one axis per signal dimension
        if len(dataObject.x) == len(data.shape) and \
                numpy.prod(xDimList) == ndata:
            for axis_dim, data_dim in zip(xDimList, data.shape):
                if axis_dim != data_dim:
                    text = "Dimensions mismatch: axes %s, data %s" % (xDimList, data.shape)
                    raise ValueError(text)

            if len(data.shape) == 3:
                _logger.debug("CASE 1: 3D data with 3 axes")
                # 3D scalar field convention is ZYX
                zcal = ArrayCalibration(dataObject.x[0])
                ycal = ArrayCalibration(dataObject.x[1])
                xcal = ArrayCalibration(dataObject.x[2])

                item3d = self.getSceneWidget().add3DScalarField(data)
                scales = [1., 1., 1.]
                origins = [0., 0., 0.]
                for i, cal in enumerate((xcal, ycal, zcal)):
                    arr = cal.calibration_array
                    origins[i] = arr[0]
                    if not cal.is_affine() and len(arr) > 1:
                        _logger.warning("axis is not linear. "
                                        "deltaX will be estimated")
                        scales[i] = (arr[-1] - arr[0]) / (len(arr) - 1)
                    else:
                        scales[i] = cal.get_slope()
                    # todo: check != 0
                item3d.setScale(*scales)
                item3d.setTranslation(*origins)
            elif len(data.shape) == 2:
                _logger.debug("CASE 2: 2D data with 2 axes")
                xcal = ArrayCalibration(dataObject.x[0])
                ycal = ArrayCalibration(dataObject.x[1])

                item3d = self.getSceneWidget().addImage(data)
                origins = [xcal(0), ycal(0)]
                scales = [1., 1.]
                for i, cal in enumerate((xcal, ycal)):
                    arr = cal.calibration_array
                    if not cal.is_affine() and len(arr) > 1:
                        _logger.warning("axis is not linear. "
                                        "deltaX will be estimated")
                        scales[i] = (arr[-1] - arr[0]) / (len(arr) - 1)
                    else:
                        scales[i] = cal.get_slope()
                item3d.setTranslation(*origins)
                item3d.setScale(*scales)

            elif len(data.shape) == 1:
                _logger.debug("CASE 3: 1D scatter (x and values)")
                item3d = self.getSceneWidget().add3DScatter(value=data,
                                                            x=dataObject.x[0],
                                                            y=numpy.zeros_like(data),
                                                            z=data)
            else:
                # this case was ignored in the original code,
                # so it probably cannot happen
                raise TypeError("Could not understand data dimensionality")
            item3d.setLabel(legend)
            return
        elif len(data.shape) == 3 and len(xDimList) == 2:
            _logger.warning("Assuming last dimension")
            _logger.debug("CASE 1.1")
            if list(xDimList) != list(data.shape[0:1]):
                text = "Wrong dimensions:"
                text += " %dx%d != (%d, %d, %d)" % (xDimList[0],
                                                    xDimList[1],
                                                    data.shape[0],
                                                    data.shape[1],
                                                    data.shape[2])
                raise ValueError(text)
            item3d = self.getSceneWidget().add3DScalarField(data)
            zcal = ArrayCalibration(dataObject.x[0])
            ycal = ArrayCalibration(dataObject.x[1])
            scales = [1., 1., 1.]
            origins = [0., 0., 0.]
            for i, cal in enumerate((ycal, zcal)):
                arr = cal.calibration_array
                origins[i + 1] = arr[0]
                if not cal.is_affine() and len(arr) > 1:
                    _logger.warning("axis is not linear. "
                                    "deltaX will be estimated")
                    scales[i + 1] = (arr[-1] - arr[0]) / (len(arr) - 1)
                else:
                    scales[i + 1] = cal.get_slope()
            item3d.setScale(*scales)
            item3d.setTranslation(*origins)
            item3d.setLabel(legend)
            return

        # I have to assume all the x are of 1 element or of as many elements as data
        axes = [numpy.zeros_like(data),
                numpy.zeros_like(data),
                numpy.zeros_like(data)]
        # overwrite initialized axes, if provided
        for xdataCounter, xdata in enumerate(dataObject.x):
            assert xdataCounter <= 2, \
                "Wrong scatter dimensionality (more than 3 axes)"
            if numpy.prod(xdata.shape) == 1:
                axis = xdata * numpy.ones(ndata)
            else:
                axis = xdata
            axes[xdataCounter] = axis

        if len(dataObject.x) == 2:
            item3d = self.getSceneWidget().add2DScatter(x=axes[0],
                                                        y=axes[1],
                                                        value=data)

            item3d.setVisualization("solid")
            # item3d.setHeightMap(True)
        else:
            # const_axes_indices = []
            # for i, axis in axes:
            #     if numpy.all(axis == axis[0]):
            #         const_axes_indices.append(i)
            # if len(const_axes_indices) == 1:
            #     item3d = self.getSceneWidget().add2DScatter(x=axes[0],  ????? TODO
            #                                                 y=axes[1],  ?????
            #                                                 value=data)
            #     # TODO: rotate adequately
            # else:
            item3d = self.getSceneWidget().add3DScatter(x=axes[0],
                                                        y=axes[1],
                                                        z=axes[2],
                                                        value=data)
        item3d.setLabel(legend)
