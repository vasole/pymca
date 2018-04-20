#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import numpy
import logging
_logger = logging.getLogger(__name__)

try:
    from PyMca5 import Object3D
    from PyMca5.Object3D import Object3DScene
except ImportError:
    _logger.debug("PyMcaGLWindow imports Object3D direcly. Frozen version?")
    import Object3D
    from Object3D import Object3DScene

class SceneGLWindow(Object3D.Object3DScene.Object3DScene):
    def _addSelection(self, selectionlist, replot=True):
        _logger.debug("addSelection(self, selectionlist=%s)", selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            dataObject = sel['dataobject']
            #one-dimensional selections not considered
            if dataObject.info["selectiontype"] == "1D":
                continue

            #there must be something to plot
            if not hasattr(dataObject, 'y'):
                continue
            #there must be an x for a scan selection to reach here
            if not hasattr(dataObject, 'x'):
                continue

            if dataObject.x is None:
                numberOfXAxes = 0
            else:
                numberOfXAxes = len(dataObject.x)
                if numberOfXAxes > 1:
                    _logger.debug("Mesh plots")
                else:
                    xdata = dataObject.x[0]

            #we have to loop for all y values
            ycounter = -1
            for ydata in dataObject.y:
                ycounter += 1
                ylegend = 'y%d' % ycounter
                if sel['selection'] is not None:
                    if type(sel['selection']) == type({}):
                        if 'y' in sel['selection']:
                            ilabel = sel['selection']['y'][ycounter]
                            ylegend = dataObject.info['LabelNames'][ilabel]
                object3Dlegend = legend + " " + ylegend
                ndata = len(ydata)
                object3D = self.addDataObject(dataObject,
                                       legend=object3Dlegend,
                                       update_scene=False)
        self.sceneControl.updateView()
        self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())


    def _removeSelection(self, selectionlist):
        _logger.debug("_removeSelection(self, selectionlist=%s)", selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]

        for sel in sellist:
            source = sel['SourceName']
            key    = sel['Key']
            legend = sel['legend'] #expected form sourcename + scan key
            if 'LabelNames' in sel['selection']:
                labelNames = sel['selection']['LabelNames']
            else:
                labelNames = sel['selection']['cntlist']
            for ycounter in sel['selection']['y']:
                ylegend = labelNames[ycounter]
                object3Dlegend = legend + " " + ylegend
            self.removeObject(object3Dlegend, update_scene=False)
        self.sceneControl.updateView()
        self.glWidget.setZoomFactor(self.glWidget.getZoomFactor())


    def _replaceSelection(self, selectionlist):
        _logger.debug("_replaceSelection(self, selectionlist=%s)", selectionlist)
        if type(selectionlist) == type([]):
            sellist = selectionlist
        else:
            sellist = [selectionlist]
        self.clear(update_scene=False)
        self._addSelection(selectionlist)

    def addDataObject(self, dataObject, legend=None, update_scene=True):
        if legend is None:
            legend = dataObject.info['legend']

        if (dataObject.m is None) or (dataObject.m == []):
            data = dataObject.y[0]
        else:
            #I would have to check for the presence of zeros in monitor
            data = dataObject.y[0]/dataObject.m[0]

        if dataObject.x is None:
            if len(data.shape) == 3:
                object3D=self.stack(data,
                           legend=legend,
                           update_scene=False)
            else:
                object3D=self.mesh(data,
                          legend=legend,
                          update_scene=False)
            return object3D

        ndata = 1
        for dimension in data.shape:
            ndata *= dimension

        ndim = 1
        xDimList = []
        for dataset in dataObject.x:
            xdim = 1
            for dimension in dataset.shape:
                xdim *= dimension
            xDimList.append(xdim)
            ndim *=xdim

        if len(dataObject.x) == len(data.shape):
            #two possibilities, the product is equal to the dimension
            #or not
            if ndim == ndata:
                if len(data.shape) == 3:
                    _logger.debug("CASE 1")
                    if (xDimList[0] != data.shape[0]) or\
                       (xDimList[1] != data.shape[1]) or\
                       (xDimList[2] != data.shape[2]):
                        text = "Wrong dimensions:"
                        text += " %dx%dx%d != (%d, %d, %d)" % (xDimList[0],
                                                               xDimList[1],
                                                               xDimList[2],
                                                               data.shape[0],
                                                               data.shape[1],
                                                               data.shape[2])
                        raise ValueError(text)
                    object3D = self.stack(data,
                                          x=dataObject.x[0],
                                          y=dataObject.x[1],
                                          z=dataObject.x[2],
                                          legend=legend,
                                          update_scene=update_scene)
                elif len(data.shape) == 2:
                    _logger.debug("CASE 2")
                    object3D = self.mesh(data,
                                         x=dataObject.x[0],
                                         y=dataObject.x[1],
                                         z=0, #This is 2D
                                         #z=data[:], #This is 3D
                                         legend=legend,
                                         update_scene=update_scene)
                elif len(data.shape) == 1:
                    _logger.debug("CASE 3")
                    object3D = self.mesh(data,
                                         x=dataObject.x[0],
                                         y=numpy.zeros((1,1), numpy.float32),
                                         z=data[:],
                                         legend=legend,
                                         update_scene=update_scene)
                return object3D
        elif (len(data.shape) == 3) and (len(xDimList) == 2):
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
            z = numpy.arange(data.shape[2])
            object3D = self.stack(data,
                                  x=dataObject.x[0],
                                  y=dataObject.x[1],
                                  z=z,
                                  legend=legend,
                                  update_scene=update_scene)
            return object3D


        #I have to assume all the x are of 1 element or of as many elements as data
        xyzData = numpy.zeros((ndata, 3), numpy.float32)
        values  = numpy.zeros((ndata, 1), numpy.float32)
        values[:,0]  = data
        xdataCounter = 0
        for xdata in dataObject.x:
            ndim = 1
            for dimension in xdata.shape:
                ndim *= dimension
            if ndim == 1:
                xyzData[:,xdataCounter] = xdata * numpy.ones(ndata)
            else:
                xyzData[:,xdataCounter] = xdata
            xdataCounter += 1

        object3D = Object3D.Object3DScene.Object3DMesh.Object3DMesh(legend)
        #if the number of points is reasonable
        #I force a surface plot.
        if ndata < 200000:
            cfg = object3D.setConfiguration({'common':{'mode':3}})
        _logger.debug("DEFAULT CASE")
        object3D.setData(values, xyz=xyzData)
        self.addObject(object3D, legend, update_scene=update_scene)
        return object3D
