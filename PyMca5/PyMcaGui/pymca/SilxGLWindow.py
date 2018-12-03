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

from silx.gui.plot3d import SceneWindow

from silx.math.calibration import ArrayCalibration


_logger = logging.getLogger(__name__)


class SceneGLWindow(SceneWindow.SceneWindow):

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
                    if not cal.is_affine() and len(arr):
                        _logger.warning("axis is not affine. "
                                        "deltaX will be estimated")
                        scales[i] = (arr[-1] - arr[0]) / (len(arr) - 1)
                        # todo: check != 0
                    else:
                        scales[i] = cal.get_slope()
                item3d.setScale(*scales)
                item3d.setTranslation(*origins)
            elif len(data.shape) == 2:
                _logger.debug("CASE 2: 2D data with 2 axes")
                zcal = ArrayCalibration(dataObject.x[0])
                ycal = ArrayCalibration(dataObject.x[1])
                xcal = ArrayCalibration(dataObject.x[2])

                item3d = self.getSceneWidget().addImage(data)
                # TODO: item3d.setScale() setOrigin()
                #           x=dataObject.x[0],
                #           y=dataObject.x[1],

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
            if (xDimList[0] != data.shape[0]) or\
               (xDimList[1] != data.shape[1]):
                text = "Wrong dimensions:"
                text += " %dx%d != (%d, %d, %d)" % (xDimList[0],
                                                    xDimList[1],
                                                    data.shape[0],
                                                    data.shape[1],
                                                    data.shape[2])
                raise ValueError(text)
            # z = numpy.arange(data.shape[2])
            item3d = self.getSceneWidget().add3DScalarField(data)
            # TODO: setScale()   setTranslation()
            # TODO: or if axes are not regular, add3DScatter
            #     x=dataObject.x[0],
            #     y=dataObject.x[1],
            #     z=z,
            item3d.setLabel(legend)
            return

        # I have to assume all the x are of 1 element or of as many elements as data
        # TODO: add3DScatter

        axes = [numpy.zeros_like(data),
                numpy.zeros_like(data),
                numpy.zeros_like(data)]
        # overwrite initialized axes, if provided
        for xdataCounter, xdata in enumerate(dataObject.x):
            assert xdataCounter <= 2, "Wrong data dimensionality"
            ndim = numpy.prod(xdata.shape)
            if ndim == 1:
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
            # TODO: if one axis is constant, add it as a 2D scatter to
            #       be able to benefit from solid visualisation
            item3d = self.getSceneWidget().add3DScatter(x=axes[0],
                                                        y=axes[1],
                                                        z=axes[2],
                                                        value=data)
        item3d.setLabel(legend)
