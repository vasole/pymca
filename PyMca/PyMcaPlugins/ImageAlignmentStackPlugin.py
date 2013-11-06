#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
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
import os
import numpy
try:
    from PyMca import StackPluginBase
    from PyMca import PyMcaQt as qt
    from PyMca import EDFStack
    from PyMca import PyMcaFileDialogs
    from PyMca import StackPluginResultsWindow
    from PyMca import FFTAlignmentWindow
    from PyMca import ExternalImagesWindow
    from PyMca import ImageRegistration
    import PyMca.PyMca_Icons as PyMca_Icons
    from PyMca import SpecfitFuns
    from PyMca import CalculationThread    
except ImportError:
    print("ExternalImagesWindow importing from somewhere else")
    import StackPluginBase
    import PyMcaQt as qt
    import EDFStack
    import PyMcaFileDialogs
    import StackPluginResultsWindow
    import ExternalImagesWindow
    import PyMca_Icons
    import ImageRegistration
    import FFTAlignmentWindow
    import SpecfitFuns
    import CalculationThread
try:
    from PyMca import sift
    SIFT = True
except:
    SIFT = False

DEBUG = 0
class ImageAlignmentStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow, **kw):
        StackPluginBase.DEBUG = DEBUG
        StackPluginBase.StackPluginBase.__init__(self, stackWindow, **kw)
        self.methodDict = {'FFT Alignment':[self._fftAlignment,
                                            "Align using FFT",
                                            None]}
        self.__methodKeys = ['FFT Alignment']
        if SIFT:
            key = 'SIFT Alignment'
            self.methodDict[key] = [self._siftAlignment,
                                    "Align using SIFT Algorithm",
                                    None]
            self.__methodKeys.append(key) 
        self.widget = None

    def stackUpdated(self):
        self.widget = None

    """

    def selectionMaskUpdated(self):
        if self.widget is None:
            return
        if self.widget.isHidden():
            return
        mask = self.getStackSelectionMask()
        self.widget.setSelectionMask(mask)
    def mySlot(self, ddict):
        if DEBUG:
            print("mySlot ", ddict['event'], ddict.keys())
        if ddict['event'] == "selectionMaskChanged":
            self.setStackSelectionMask(ddict['current'])
        elif ddict['event'] == "addImageClicked":
            self.addImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "removeImageClicked":
            self.removeImage(ddict['title'])
        elif ddict['event'] == "replaceImageClicked":
            self.replaceImage(ddict['image'], ddict['title'])
        elif ddict['event'] == "resetSelection":
            self.setStackSelectionMask(None)
    """

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _fftAlignment(self):
        stack = self.getStackDataObject()
        if stack is None:
            return
        mcaIndex = stack.info.get('McaIndex')
        if mcaIndex != 0:
            raise IndexError("For the time being only stacks of images supported")

        if self.widget is None:
            self.widget = FFTAlignmentWindow.FFTAlignmentDialog()
        self.widget.setStack(stack)
        ret = self.widget.exec_()
        if ret:
            ddict = self.widget.getParameters()
            self.widget.setDummyStack()
            offsets = [ddict['Dim 0']['offset'], ddict['Dim 1']['offset']] 
            widths = [ddict['Dim 0']['width'], ddict['Dim 1']['width']]
            reference = stack.data[ddict['reference_index']]
            crop = False
            if ddict['file_use']:
                filename = ddict['file_name']
            else:
                filename = None
            if DEBUG:
                shifts = self.calculateShiftsFFT(stack,
                                                 reference,
                                                 offsets=offsets,
                                                 widths=widths,
                                                 crop=crop)
                result = self.shiftStack(stack,
                                         -shifts,
                                         crop=crop,
                                         filename=filename)

            else:
                result = self.__calculateShiftsFFT(stack,
                                                   reference,
                                                   offsets=offsets,
                                                   widths=widths,
                                                   crop=crop)
                if result[0] == 'Exception':
                    # exception occurred
                    raise Exception(result[1], result[2], result[3])
                else:
                    shifts = result
                result = self.__shiftStack(stack,
                                           -shifts,
                                           crop=crop,
                                           filename=filename)
                if result is not None:
                    # exception occurred
                    raise Exception(result[1], result[2], result[3])
            self.setStack(stack)

    def __calculateShiftsFFT(self, *var, **kw):
        self._progress = 0.0
        thread = CalculationThread.CalculationThread(\
                calculation_method=self.calculateShiftsFFT,
                calculation_vars=var,
                calculation_kw=kw)
        thread.start()
        CalculationThread.waitingMessageDialog(thread,
                                               message="Please wait. Calculation going on.",
                                               parent=self.widget,
                                               modal=True,
                                               update_callback=self._waitingCallback)
        return thread.result

    def __shiftStack(self, *var, **kw):
        self._progress = 0.0
        thread = CalculationThread.CalculationThread(\
                calculation_method=self.shiftStack,
                calculation_vars=var,
                calculation_kw=kw)
        thread.start()
        CalculationThread.waitingMessageDialog(thread,
                                               message="Please wait. Calculation going on.",
                                               parent=self.widget,
                                               modal=True,
                                               update_callback=self._waitingCallback)
        return thread.result


    def __calculateShiftsSIFT(self, *var, **kw):
        self._progress = 0.0
        thread = CalculationThread.CalculationThread(\
                calculation_method=self.calculateShiftsSIFT,
                calculation_vars=var,
                calculation_kw=kw)
        thread.start()
        CalculationThread.waitingMessageDialog(thread,
                                               message="Please wait. Calculation going on.",
                                               parent=self.widget,
                                               modal=True,
                                               update_callback=self._waitingCallback)
        return thread.result

    def _waitingCallback(self):
        ddict = {}
        ddict['message'] = "Calculation Progress = %d %%" % self._progress
        return ddict        

    def _siftAlignment(self):
        if sift.opencl.ocl is None:
            raise ImportError("PyOpenCL does not seem to be installed on your system")
        stack = self.getStackDataObject()
        if stack is None:
            return
        mcaIndex = stack.info.get('McaIndex')
        if mcaIndex != 0:
            raise IndexError("For the time being only stacks of images supported")
        from PyMca import SIFTAlignmentWindow
        widget = SIFTAlignmentWindow.SIFTAlignmentDialog()
        widget.setStack(stack)
        mask = self.getStackSelectionMask()
        widget.setSelectionMask(mask)
        ret = widget.exec_()
        if ret:
            ddict = widget.getParameters()
            widget.setDummyStack()
            reference = stack.data[ddict['reference_index']] * 1
            mask = ddict['mask']
            if ddict['file_use']:
                filename = ddict['file_name']
            else:
                filename = None
            crop = False
            device = ddict['opencl_device']
            if DEBUG:
                result = self.calculateShiftsSIFT(stack, reference, mask=mask, device=device, crop=crop, filename=filename)
            else:
                result = self.__calculateShiftsSIFT(stack, reference, mask=mask, device=device, crop=crop, filename=filename)
                
            self.setStack(stack)
        
    def calculateShiftsSIFT(self, stack, reference, mask=None, device=None, crop=None, filename=None):
        mask = self.getStackSelectionMask()
        if mask is not None:
            if mask.sum() == 0:
                mask = None
        if device is None:
            siftInstance = sift.LinearAlign(reference.astype(numpy.float32), devicetype="cpu")
        else:
            siftInstance = sift.LinearAlign(reference.astype(numpy.float32), device=device)
        data = stack.data
        mcaIndex = stack.info['McaIndex']
        if mcaIndex != 0:
             raise IndexError("For the time being only stacks of images supported")
        total = float(data.shape[mcaIndex])
        for i in range(data.shape[mcaIndex]):
            if DEBUG:
                print("SIFT Shifting image %d" % i)
            result = siftInstance.align(data[i].astype(numpy.float32), shift_only=True, return_all=True)
            if DEBUG:
                print("Index = %d shift = %.4f, %.4f" % (i, result['offset'][0], result['offset'][1]))
            stack.data[i] = result['result']
            self._progress = (100 * i) / total

    def calculateShiftsFFT(self, stack, reference, offsets=None, widths=None, crop=False):
        if DEBUG:
            print("Offsets = ", offsets)
            print("Widths = ", widths)
        data = stack.data
        if offsets is None:
            offsets = [0.0, 0.0]
        if widths is None:
            widths = [reference.shape[0], reference.shape[1]]
        fft2Function = numpy.fft.fft2
        if 1:
            DTYPE = numpy.float32
        else:
            DTYPE = numpy.float64
        image2 = numpy.zeros((widths[0], widths[1]), dtype=DTYPE)
        shape = image2.shape

        USE_APODIZATION_WINDOW = False
        apo = [10, 10]
        if USE_APODIZATION_WINDOW:
            # use apodization window
            window = numpy.outer(SpecfitFuns.slit([0.5, shape[0] * 0.5, shape[0] - 4 * apo[0], apo[0]],
                                          numpy.arange(float(shape[0]))),
                                 SpecfitFuns.slit([0.5, shape[1] * 0.5, shape[1] - 4 * apo[1], apo[1]],
                                          numpy.arange(float(shape[1])))).astype(DTYPE)
        else:
            window = numpy.zeros((shape[0], shape[1]), dtype=DTYPE)
            window[apo[0]:shape[0] - apo[0], apo[1]:shape[1] - apo[1]] = 1
        image2[:,:] = window * reference[offsets[0]:offsets[0]+widths[0],
                                         offsets[1]:offsets[1]+widths[1]]
        image2fft2 = fft2Function(image2)
        shifts = numpy.zeros((data.shape[0], 2), numpy.float)
        image1 = numpy.zeros(image2.shape, dtype=DTYPE)
        total = float(data.shape[0])
        for i in range(data.shape[0]):
            image1[:,:] = window * data[i][offsets[0]:offsets[0]+widths[0],
                                           offsets[1]:offsets[1]+widths[1]]
               
            image1fft2 = fft2Function(image1)
            shifts[i] = ImageRegistration.measure_offset_from_ffts(image1fft2,
                                                                   image2fft2)
            if DEBUG:
                print("Index = %d shift = %.4f, %.4f" % (i, shifts[i][0], shifts[i][1]))
            self._progress = (100 * i) / total
        return shifts

    def shiftStack(self, stack, shifts, crop=False, filename=None):
        """
        """
        data = stack.data
        mcaIndex = stack.info['McaIndex']
        if mcaIndex != 0:
             raise IndexError("For the time being only stacks of images supported")
        shape = data[mcaIndex].shape
        d0_start, d0_end, d1_start, d1_end = ImageRegistration.get_crop_indices(shape,
                                                                                shifts[:, 0],
                                                                                shifts[:, 1])
        window = numpy.zeros(shape, numpy.float32)
        window[d0_start:d0_end, d1_start:d1_end] = 1.0
        self._progress = 0.0
        total = float(data.shape[mcaIndex])
        for i in range(data.shape[mcaIndex]):
            #tmpImage = ImageRegistration.shiftFFT(data[i], shifts[i])
            tmpImage = ImageRegistration.shiftBilinear(data[i], shifts[i])
            print("Index %d bilinear shifted" % i)
            if filename is None:
                stack.data[i] = tmpImage * window
            self._progress = (100 * i) / total

MENU_TEXT = "Image Alignment Tool"
def getStackPluginInstance(stackWindow, **kw):
    ob = ImageAlignmentStackPlugin(stackWindow)
    return ob
