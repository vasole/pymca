#/*##########################################################################
# Copyright (C) 2018-2019 European Synchrotron Radiation Facility
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
import h5py
import logging
import os.path

from PyMca5.PyMcaIO import EdfFile
from PyMca5.PyMcaIO import EDFStack
from PyMca5.PyMcaIO import TiffStack

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaGui.io import PyMcaFileDialogs

from silx.gui.plot3d import SceneWindow
from silx.gui import icons
from silx.gui.utils.image import convertQImageToArray
from silx.gui.colors import Colormap
from silx.gui.plot3d import items

from silx.math.calibration import ArrayCalibration


_logger = logging.getLogger(__name__)


def getPixmap():
    """
    Open an image file and return the filename and the data.

    Return ``None, None`` in case of failure.
    """
    fileTypeList = ['Picture Files (*jpg *jpeg *tif *tiff *png)',
                    'EDF Files (*edf)',
                    'EDF Files (*ccd)',
                    'ADSC Files (*img)',
                    'EDF Files (*)']
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select one object data file",
        mode="OPEN",
        getfilter=True)
    if not fileList:
        return None, None
    fname = fileList[0]
    if filterUsed.split()[0] == "Picture":
        qimage = qt.QImage(fname)
        if qimage.isNull():
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot read file %s as an image" % fname)
            msg.exec()
            return None, None
        return os.path.basename(fname), convertQImageToArray(qimage)
    if filterUsed.split()[0] in ["EDF", "ADSC"]:
        edf = EdfFile.EdfFile(fname)
        data = edf.GetData(0)
        return os.path.basename(fname), data
    return None, None


def get4DStack():
    """
    Open a stack of image files in EDF or TIFF format, and return
    the data with metadata.

    :returns: legend, data, xScale, yScale, fileindex
        Legend and data are both ``None`` in case of failure.
        Scales are 2-tuples (originX, deltaX).
        fileindex indicates the dimension/axis in the data corresponding to
        the Z-axis.
    :raise IOError: If the data could not be read
    """
    fileTypeList = ['EDF Z Stack (*edf *ccd)',
                    'EDF X Stack (*edf *ccd)',
                    'TIFF Stack (*tif *tiff)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = False
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select the object file(s)",
        mode="OPEN",
        getfilter=True)
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = old
    if not fileList:
        return None, None
    if filterUsed.startswith('EDF Z'):
        fileindex = 2
    else:
        fileindex = 1
    filename = fileList[0]
    legend = os.path.basename(filename)
    if filterUsed.startswith('TIFF'):
        stack = TiffStack.TiffStack(dtype=numpy.float32, imagestack=False)
        stack.loadFileList(fileList, fileindex=1)
    else:
        stack = EDFStack.EDFStack(dtype=numpy.float32, imagestack=False)
        if len(fileList) == 1:
            stack.loadIndexedStack(filename, fileindex=fileindex)
        else:
            stack.loadFileList(fileList, fileindex=fileindex)
    if stack is None:
        raise IOError("Problem reading stack.")
    xScale = stack.info.get("xScale")
    yScale = stack.info.get("yScale")
    return legend, stack.data, xScale, yScale, fileindex


def getChimeraStack():
    """
    Open an chimera file and return the filename and the data.

    Return ``None, None`` in case the user cancelled the file dialog.
    :raise IOError: If the data is not a 3D stack
    """
    fileTypeList = ['Chimera Stack (*cmp)',
                    'Chimera Stack (*)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select the object file(s)",
        mode="OPEN",
        getfilter=True)
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = old
    if not fileList:
        return None, None
    filename = fileList[0]
    with h5py.File(filename, mode='r') as f:
        stack = f['Image/data'][...]
    if not isinstance(stack, numpy.ndarray) or stack.ndim != 3:
        raise IOError("Problem reading stack.")
    return os.path.basename(filename), stack


def getMesh():
    """
    Read an image data file (EDF, ADSC), return the data and image name.
    This is then used to display the image as a height map.
    Returns *None, None* if the file dialog is cancelled or loaing fails.

    :return: legend, data
    """
    fileTypeList = ['EDF Files (*edf)',
                    'EDF Files (*ccd)',
                    'ADSC Files (*img)',
                    'All Files (*)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = False
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select one object data file",
        mode="OPEN",
        getfilter=True)
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = old
    if not fileList:
        return None, None

    filename = fileList[0]
    edf = EdfFile.EdfFile(filename, access='rb')
    data = edf.GetData(0).astype(numpy.float32)
    return os.path.basename(filename), data


def mean_isolevel(data):
    """Compute a default isosurface level: mean + 1 std

    :param numpy.ndarray data: The data to process
    :rtype: float
    """
    data = data[numpy.isfinite(data)]
    if len(data) == 0:
        return 0
    else:
        return numpy.mean(data) + numpy.std(data)


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
        legend, data = getPixmap()
        if legend is not None and data.ndim in [2, 3]:
            item3d = self._sceneGlWindow.getSceneWidget().addImage(data)
            item3d.setLabel(legend)
            if not isinstance(item3d, items.ImageRgba):
                item3d.setColormap(Colormap(name="temperature"))

    def _onLoad3DMesh(self, checked):
        legend, data = getMesh()
        if legend is None or data.ndim != 2:
            return

        xSize, ySize = data.shape
        x, y = numpy.meshgrid(numpy.arange(xSize), numpy.arange(ySize))
        x = x.reshape(-1)
        y = y.reshape(-1)
        item3d = self._sceneGlWindow.getSceneWidget().add2DScatter(x=x,
                                                                   y=y,
                                                                   value=data)
        item3d.setVisualization("solid")   # this is expensive for large images
        item3d.setHeightMap(True)
        item3d.setColormap(Colormap(name="temperature"))

    def _onLoad4DStack(self, checked):
        # todo: use fileIndex to decide the slicing direction of the cube
        legend, stackData, xScale, yScale, fileIndex = get4DStack()
        if legend is None:
            return
        origin = [0., 0., 0.]
        delta = [1., 1., 1.]
        if xScale is not None:
            origin[0] = xScale[0]
            delta[0] = xScale[1]
        if yScale is not None:
            origin[1] = yScale[0]
            delta[1] = yScale[1]

        # Uncomment this block for a stack of images (may be slow)
        # group = items.GroupItem()
        # group.setLabel(legend)
        # for i in range(stackData.shape[0]):
        #     item3d = items.ImageData()
        #     item3d.setData(stackData[i])
        #     item3d.setLabel("frame %d" % i)
        #     origin[2] = i                  # shift each frame by 1
        #     item3d.setTranslation(*origin)
        #     item3d.setScale(*delta)
        #     group.addItem(item3d)
        # self._sceneGlWindow.getSceneWidget().addItem(group)

        item3d = self._sceneGlWindow.getSceneWidget().add3DScalarField(stackData)
        item3d.setLabel(legend)
        item3d.setTranslation(*origin)
        item3d.setScale(*delta)
        item3d.addIsosurface(mean_isolevel, "blue")
        for cp in item3d.getCutPlanes():
            cp.setColormap(Colormap(name="temperature"))

    def _onLoadChimeraStack(self, checked):
        legend, data = getChimeraStack()
        if legend is None:
            return
        item3d = self._sceneGlWindow.getSceneWidget().add3DScalarField(data)
        item3d.setLabel(legend)
        item3d.addIsosurface(mean_isolevel, "blue")
        for cp in item3d.getCutPlanes():
            cp.setColormap(Colormap(name="temperature"))


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
            # note: In addDataObject we currently only ever access y[0].
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

        nItemsBefore = len(self.getSceneWidget().getItems())
        # we need to remove existing items with the same legend
        to_be_removed = []
        for it in self.getSceneWidget().getItems():
            if it.getLabel() == legend:
                to_be_removed.append(it)
        for _i in range(len(to_be_removed)):
            self.getSceneWidget().removeItem(to_be_removed.pop())

        if dataObject.m is None or dataObject.m == []:
            data = dataObject.y[0]
        else:
            # I would have to check for the presence of zeros in monitor
            data = dataObject.y[0] / dataObject.m[0]

        if dataObject.x is None:
            # note: this does not seem to be possible if data is sent from the main selector,
            #       at least 2 axs must be selected for the data to be sent to this widget
            if len(data.shape) == 3:
                item3d = self.getSceneWidget().add3DScalarField(data)
                item3d.addIsosurface(mean_isolevel, "blue")
                for cp in item3d.getCutPlanes():
                    cp.setColormap(Colormap(name="temperature"))
            elif len(data.shape) == 2:
                item3d = self.getSceneWidget().addImage(data)
                item3d.setColormap(Colormap(name="temperature"))
            else:
                raise NotImplementedError("case dataObject.x is None and ndim not in [2, 3]")
                # item3d = self.getSceneWidget().mesh(data)
            item3d.setLabel(legend)
            if (not nItemsBefore) or \
               (len(self.getSceneWidget().getItems()) == 1):
                self.getSceneWidget().centerScene()
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
                item3d.addIsosurface(mean_isolevel, "blue")
                for cp in item3d.getCutPlanes():
                    cp.setColormap(Colormap(name="temperature"))
            elif len(data.shape) == 2:
                _logger.debug("CASE 2: 2D data with 2 axes")
                ycal = ArrayCalibration(dataObject.x[0])
                xcal = ArrayCalibration(dataObject.x[1])

                item3d = self.getSceneWidget().addImage(data)
                origins = [xcal(0), ycal(0)]
                scales = [1., 1.]
                for i, cal in enumerate((xcal, ycal)):
                    arr = cal.calibration_array
                    if not cal.is_affine() and len(arr) > 1:
                        _logger.warning("axis is not linear. "
                                        "deltaX will be estimated")
                        scales[i] = (arr[-1] - arr[0]) / (len(arr) - 1)    # TODO: do a scatter instead with numpy.meshgrid
                    else:
                        scales[i] = cal.get_slope()
                item3d.setTranslation(*origins)
                item3d.setScale(*scales)
                item3d.setColormap(Colormap(name="temperature"))

            elif len(data.shape) == 1:
                _logger.debug("CASE 3: 1D scatter (x and values)")
                item3d = self.getSceneWidget().add3DScatter(value=data,
                                                            x=dataObject.x[0],
                                                            y=numpy.zeros_like(data),
                                                            z=data)
                item3d.setColormap(Colormap(name="temperature"))
            else:
                # this case was ignored in the original code,
                # so it probably cannot happen
                raise TypeError("Could not understand data dimensionality")
            item3d.setLabel(legend)
            if (not nItemsBefore) or \
               (len(self.getSceneWidget().getItems()) == 1):
                self.getSceneWidget().centerScene()
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
            item3d.addIsosurface(mean_isolevel, "blue")
            for cp in item3d.getCutPlanes():
                cp.setColormap(Colormap(name="temperature"))
            if (not nItemsBefore) or \
               (len(self.getSceneWidget().getItems()) == 1):
                self.getSceneWidget().centerScene()
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
            item3d.setColormap(Colormap(name="temperature"))
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
            item3d.setColormap(Colormap(name="temperature"))
        item3d.setLabel(legend)
        if (not nItemsBefore) or \
           (len(self.getSceneWidget().getItems()) == 1):
            self.getSceneWidget().centerScene()
