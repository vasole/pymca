#/*##########################################################################
# Copyright (C) 2004-2015 V.A. Sole, European Synchrotron Radiation Facility
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
"""

A Stack plugin is a module that will be automatically added to the PyMca stack windows
in order to perform user defined operations on the data stack.

These plugins will be compatible with any stack window that provides the functions:
    #data related
    getStackDataObject
    getStackData
    getStackInfo
    setStack

    #images related
    addImage
    removeImage
    replaceImage

    #mask related
    setSelectionMask
    getSelectionMask

    #displayed curves
    getActiveCurve
    getGraphXLimits
    getGraphYLimits

    #information method
    stackUpdated
    selectionMaskUpdated
"""
import numpy
from PyMca5 import StackPluginBase
# Add support for normalization by data
from PyMca5.PyMca import PyMcaFileDialogs
from PyMca5.PyMca import EdfFile
from PyMca5.PyMca import specfilewrapper
try:
    import h5py
    HDF5 = True
except:
    HDF5 = False

DEBUG = 0

class StackNormalizationPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {}
        text  = "Stack/I0 where I0 is the active curve\n"
        function = self.divideByCurve
        info = text
        icon = None
        self.methodDict["I/I0 Normalization"] =[function,
                                                info,
                                                icon]

        text  = "-log(Stack/I0) Normalization where I0 is the active curve\n"
        function = self.logNormalizeByCurve
        info = text
        icon = None
        self.methodDict["-log(I/I0) Normalization"] =[function,
                                                      info,
                                                      icon]

        text  = "External Image I/I0 Normalization where\n"
        text += "I0 is an image read from file\n"
        function = self.divideByExternalImage
        info = text
        icon = None
        self.methodDict["Image I/I0 Normalization"] =[function,
                                                info,
                                                icon]

        text  = "External Image (I/I0) * max(I0) Normalization where\n"
        text += "I0 is an image read from file\n"
        function = self.scaleByExternalImage
        info = text
        icon = None
        self.methodDict["Image I * (max(I0)/I0) Scaling"] =[function,
                                                info,
                                                icon]

        text  = "External Image -log(Stack/I0) Normalization\n"
        text += "where I0 is an image read from file\n"
        function = self.logNormalizeByExternalImage
        info = text
        icon = None
        self.methodDict["Image -log(I/I0) Normalization"] =[function,
                                                info,
                                                icon]
        self.__methodKeys = ["I/I0 Normalization",
                             "-log(I/I0) Normalization",
                             "Image I/I0 Normalization",
                             "Image I * (max(I0)/I0) Scaling",
                             "Image -log(I/I0) Normalization"]

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _loadExternalData(self):
        getfilter = True
        fileTypeList = ["EDF Files (*edf *ccd *tif)"]
        if HDF5:
            fileTypeList.append('HDF5 Files (*.h5 *.nxs *.hdf)')
        fileTypeList.append('ASCII Files (*)') 
        fileTypeList.append("EDF Files (*)")
        message = "Open data file"
        filenamelist, ffilter = PyMcaFileDialogs.getFileList(parent=None,
                                    filetypelist=fileTypeList,
                                    message=message,
                                    getfilter=getfilter,
                                    single=True,
                                    currentfilter=None)
        if len(filenamelist) < 1:
            return
        filename = filenamelist[0]
        if ffilter.startswith('HDF5'):
            data = HDF5Widget.getDatasetDialog(filename, value=True,
                            message='Select your data set by a double click')
        elif ffilter.startswith("EDF"):
            edf = EdfFile.EdfFile(filename, "rb")
            if edf.GetNumImages() > 1:
                # TODO: A dialog showing the different images
                # based on the external images browser needed
                print("WARNING: Taking first image")
            data = edf.GetData(0)
            edf = None
        elif ffilter.startswith("ASCII"):
            #data=numpy.loadtxt(filename)
            sf = specfilewrapper.Specfile(filename)
            targetScan = sf[0]
            data = numpy.array(targetScan.data().T, copy=True)
            targetScan = None
            sf = None
        return data

    def scaleByExternalImage(self):
        self._externalImageOperation("scale")

    def divideByExternalImage(self):
        self._externalImageOperation("divide")

    def logNormalizeByExternalImage(self):
        self._externalImageOperation("log")

    def _externalImageOperation(self, operation="divide"):
        if operation == "log":
            operator = numpy.log
        stack = self.getStackDataObject()
        if stack is None:
            return
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks yet"
            raise TypeError(text)
        normalizationData = self._loadExternalData()
        if normalizationData is None:
            return
        mcaIndex = stack.info.get('McaIndex', -1)
        stackShape = stack.data.shape
        if mcaIndex < 0:
            mcaIndex = len(stackShape) - mcaIndex
        imageSize = stack.data.size / stackShape[mcaIndex]
        if normalizationData.size != imageSize:
            if normalizationData.shape[0] == imageSize:
                if len(normalizationData.shape) == 2:
                    # assume the last column are the normalization data
                    normalizationData = normalizationData[:, -1]
        if normalizationData.size != imageSize:
                raise ValueError("Loaded data size does not match required size")
        if normalizationData.dtype not in [numpy.float,
                                           numpy.float32,
                                           numpy.float64]:
            normalizationData = normalizationData.astype(numpy.float32)
        if operation == "scale":
            normalizationData /= normalizationData.max()
        # TODO: Use an intermediate array and set divisions 0/0 to 0.
        if stack.data.dtype in [numpy.int32, numpy.uint32]:
            view = stack.data.view(numpy.float32)
        elif stack.data.dtype in [numpy.int64, numpy.uint64]:
            view = stack.data.view(numpy.float64)
        else:
            view = stack.data
        if mcaIndex == 0:
            normalizationData.shape = stackShape[1:]
            if operation in ["divide", "scale"]:
                for i in range(stackShape[mcaIndex]):
                    view[i] = stack.data[i] / normalizationData
            elif operation == "log":
                for i in range(stackShape[mcaIndex]):
                    view[i] = -operator(stack.data[i]/normalizationData)
        elif mcaIndex == 2:
            normalizationData.shape = stackShape[:2]
            if operation in ["divide", "scale"]:
                for i in range(stackShape[mcaIndex]):
                    view[:, :, i] = stack.data[:, :, i] / normalizationData
            else:
                for i in range(stackShape[mcaIndex]):
                    view[:, :, i] = -operator(stack.data[:, :, i]/ \
                                                    normalizationData)
        elif mcaIndex == 1:
            normalizationData.shape = stackShape[0], stackShape[2]
            if operation in ["divide", "scale"]:
                for i in range(stackShape[mcaIndex]):
                    view[:, i, :] = stack.data[:, i, :] / normalizationData
            else:
                for i in range(stackShape[mcaIndex]):
                    view[:, i, :] = -operator(stack.data[:, i, :]/ \
                                                    normalizationData)
        else:
            raise ValueError("Unsupported 1D index %d" % mcaIndex)
        self.setStack(view)

    def divideByExternalCurve(self):
        stack = self.getStackDataObject()
        if stack is None:
            return
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks yet"
            raise TypeError(text)

    def divideByCurve(self):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        curve = self.getActiveCurve()
        if curve is None:
            text = "Please make sure to have an active curve"
            raise TypeError(text)
        x, y, legend, info = self.getActiveCurve()
        yWork = y[y!=0].astype(numpy.float)
        mcaIndex = stack.info.get('McaIndex', -1)
        if mcaIndex in [-1, 2]:
            for i, value in enumerate(yWork):
                stack.data[:, :, i] = stack.data[:,:,i]/value
        elif mcaIndex == 0:
            for i, value in enumerate(yWork):
                stack.data[i, :, :] = stack.data[i,:,:]/value
        elif mcaIndex == 1:
            for i, value in enumerate(yWork):
                stack.data[:, i, :] = stack.data[:,i,:]/value
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        self.setStack(stack)

    def logNormalizeByCurve(self):
        stack = self.getStackDataObject()
        if not isinstance(stack.data, numpy.ndarray):
            text = "This method does not work with dynamically loaded stacks"
            raise TypeError(text)
        curve = self.getActiveCurve()
        if curve is None:
            text = "Please make sure to have an active curve"
            raise TypeError(text)
        x, y, legend, info = self.getActiveCurve()
        yWork = y[y>0].astype(numpy.float)
        mcaIndex = stack.info.get('McaIndex', -1)
        if mcaIndex in [-1, 2]:
            for i, value in enumerate(yWork):
                stack.data[:, :, i] = -numpy.log(stack.data[:,:,i]/value)
        elif mcaIndex == 0:
            for i, value in enumerate(yWork):
                stack.data[i, :, :] = -numpy.log(stack.data[i,:,:]/value)
        elif mcaIndex == 1:
            for i, value in enumerate(yWork):
                stack.data[:, i, :] = -numpy.log(stack.data[:,i,:]/value)
        else:
            raise ValueError("Invalid 1D index %d" % mcaIndex)
        self.setStack(stack)

MENU_TEXT = "Stack Normalization"
def getStackPluginInstance(stackWindow, **kw):
    ob = StackNormalizationPlugin(stackWindow)
    return ob
