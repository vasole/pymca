#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
import unittest
import numpy

class DummyArray(object):
    def __init__(self, data):
        """
        This class forces ROI and spectra calculation to be performed as
        it is made for a dynamically loaded array.

        This allows detection of the PyMca bug track issue 3544665
        """
        self.data = numpy.array(data, copy=True)

    def __getitem__(self, *var):
        if len(var) == 1:
            return self.data[var[0]]
        elif len(var) == 2:
            return self.data[var[0], var[1]]
        elif len(var) == 3:
            return self.data[var[0], var[1], var[2]]

    def getShape(self):
        return self.data.shape

    def getDType(self):
        return self.data.dtype

    shape = property(getShape)
    dtype = property(getDType)

class testStackBase(unittest.TestCase):
    def setUp(self):
        return
        if not hasattr(self, '_data1D'):
            print "DOING IT"
            nrows = 100
            ncolumns = 200
            nchannels = 1024
            a = numpy.ones((nrows, ncolumns), numpy.float)
            self._data1D = numpy.zeros((nrows, ncolumns, nchannels),
                                       numpy.float)
            self._data2D = numpy.zeros((nchannels, nrows, ncolumns),
                                       numpy.float)
            for i in range(nchannels):
                self._data1D[:, :, i] = a * i
                self._data2D[i, :, :] = a * i
            self._dummyDataObject1D = DummyDataObject(self._data1D)
            self._dummyDataObject2D = DummyDataObject(self._data2D)
            self._dummyDataObject2D.info['mcaIndex'] = 0
        
    def testStackBaseImport(self):
        from PyMca import StackBase

    def testStackBaseStack1DDataHandling(self):
        from PyMca import StackBase
        nrows = 100
        ncolumns = 200
        nchannels = 1024
        a = numpy.ones((nrows, ncolumns), numpy.float)
        referenceData = numpy.zeros((nrows, ncolumns, nchannels),
                                   numpy.float)
        for i in range(nchannels):
            referenceData[:, :, i] = a * i
        mask = numpy.zeros((nrows, ncolumns), numpy.uint8)
        mask[20:30, 15:50] = 1

        dummyArray = DummyArray(referenceData)

        defaultMca = referenceData.sum(axis=0, dtype=numpy.float64).sum(axis=0)
        maskedMca = referenceData[mask>0, :].sum(axis=0)


        j = 0
        for data in [referenceData, dummyArray]:
            if j == 0:
                dynamic = ""
                j = 1
            else:
                dynamic = "dynamic "
            stackBase = StackBase.StackBase()
            stackBase.setStack(data, mcaindex=2)
            channels, counts = stackBase.getActiveCurve()[0:2]
            self.assertTrue(numpy.allclose(defaultMca, counts),
                                           "Incorrect %sdefault mca" % dynamic)

            # set mask
            stackBase.setSelectionMask(mask)
            self.assertTrue(numpy.allclose(stackBase.getSelectionMask(), mask),
                                           "Incorrect mask set and get")

            # get mca from mask
            mcaDataObject = stackBase.calculateMcaDataObject()
            self.assertTrue(numpy.allclose(mcaDataObject.y[0], maskedMca),
                                    "Incorrect %smca from mask calculation" % dynamic)

            #get image from roi
            i0 = 100
            imiddle = 200
            i1 = 500
            # calculate
            imageDict = stackBase.calculateROIImages(100, 500, imiddle=200)
            self.assertTrue(numpy.allclose(imageDict['ROI'], data[:,:,i0:i1].sum(axis=-1)),
                    "Incorrect ROI image from %sROI calculation"  % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Left'], data[:,:,i0]),
                    "Incorrect Left image from %sROI calculation" % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Right'], data[:,:,i1-1]),
                    "Incorrect Right image from %sROI calculation"  % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Middle'], data[:,:,imiddle]),
                    "Incorrect Middle image from %sROI calculation" % dynamic)

    def testStackBaseStack2DDataHandling(self):
        from PyMca import StackBase
        nrows = 100
        ncolumns = 200
        nchannels = 1024
        a = numpy.ones((nrows, ncolumns), numpy.float)
        referenceData = numpy.zeros((nchannels, nrows, ncolumns),
                                   numpy.float)
        for i in range(nchannels):
            referenceData[i, :, :] = a * i
        mask = numpy.zeros((nrows, ncolumns), numpy.uint8)
        mask[20:30, 15:50] = 1

        dummyArray = DummyArray(referenceData)

        defaultMca = referenceData.sum(axis=2, dtype=numpy.float64).sum(axis=1)
        maskedMca = referenceData[:,mask>0].sum(axis=1)


        j = 0
        for data in [referenceData, dummyArray]:
            if j == 0:
                dynamic = ""
                j = 1
            else:
                dynamic = "dynamic "
            stackBase = StackBase.StackBase()
            stackBase.setStack(data, mcaindex=0)
            channels, counts = stackBase.getActiveCurve()[0:2]
            self.assertTrue(numpy.allclose(defaultMca, counts),
                                           "Incorrect %sdefault mca" % dynamic)

            # set mask
            stackBase.setSelectionMask(mask)
            self.assertTrue(numpy.allclose(stackBase.getSelectionMask(), mask),
                                           "Incorrect mask set and get")

            # get mca from mask
            mcaDataObject = stackBase.calculateMcaDataObject()
            self.assertTrue(numpy.allclose(mcaDataObject.y[0], maskedMca),
                                    "Incorrect %smca from mask calculation" % dynamic)

            #get image from roi
            i0 = 100
            imiddle = 200
            i1 = 500
            # calculate
            imageDict = stackBase.calculateROIImages(100, 500, imiddle=200)
            self.assertTrue(numpy.allclose(imageDict['ROI'], data[i0:i1, :,:].sum(axis=0)),
                    "Incorrect ROI image from %sROI calculation"  % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Left'], data[i0,:,:]),
                    "Incorrect Left image from %sROI calculation" % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Right'], data[i1-1,:,:]),
                    "Incorrect Right image from %sROI calculation"  % dynamic)
            self.assertTrue(numpy.allclose(imageDict['Middle'], data[imiddle,:,:]),
                    "Incorrect Middle image from %sROI calculation" % dynamic)

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testStackBase))
    else:
        # use a predefined order
        testSuite.addTest(testStackBase("testStackBaseImport"))
        testSuite.addTest(testStackBase("testStackBaseStack1DDataHandling"))
        testSuite.addTest(testStackBase("testStackBaseStack2DDataHandling"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
