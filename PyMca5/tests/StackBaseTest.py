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
        self.data = numpy.array(data, copy=False)

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

    def getSize(self):
        s = 1
        for item in self.__shape:
            s *= item
        return s

    shape = property(getShape)
    dtype = property(getDType)
    size = property(getSize)

class testStackBase(unittest.TestCase):
    def testStackBaseImport(self):
        from PyMca5.PyMcaCore import StackBase

    def testStackBaseStack1DDataHandling(self):
        from PyMca5.PyMcaCore import StackBase
        nrows = 50
        ncolumns = 100
        nchannels = 500
        a = numpy.ones((nrows, ncolumns), numpy.float)
        referenceData = numpy.zeros((nrows, ncolumns, nchannels),
                                   numpy.float)
        for i in range(nchannels):
            referenceData[:, :, i] = a * i
        a = None
        mask = numpy.zeros((nrows, ncolumns), numpy.uint8)
        mask[20:30, 15:50] = 1

        dummyArray = DummyArray(referenceData)

        defaultMca = referenceData.sum(axis=0, dtype=numpy.float64).sum(axis=0)
        maskedMca = referenceData[mask>0, :].sum(axis=0)


        for fileindex in [0, 1]:
            #usually only one file index case is used but
            #we test both to have a better coverage
            j = 0
            for data in [referenceData, dummyArray]:
                if j == 0:
                    dynamic = ""
                    j = 1
                else:
                    dynamic = "dynamic "
                stackBase = StackBase.StackBase()
                stackBase.setStack(data, mcaindex=2, fileindex=fileindex)
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
                i1 = 400
                # calculate
                imageDict = stackBase.calculateROIImages(i0, i1, imiddle=imiddle)
                self.assertTrue(numpy.allclose(imageDict['ROI'], data[:,:,i0:i1].sum(axis=-1)),
                        "Incorrect ROI image from %sROI calculation"  % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Left'], data[:,:,i0]),
                        "Incorrect Left image from %sROI calculation" % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Right'], data[:,:,i1-1]),
                        "Incorrect Right image from %sROI calculation"  % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Middle'], data[:,:,imiddle]),
                        "Incorrect Middle image from %sROI calculation" % dynamic)
        stackBase = None
        data = None
        dummyArray = None
        referenceData = None

    def testStackBaseStack2DDataHandling(self):
        from PyMca5.PyMcaCore import StackBase
        nrows = 50
        ncolumns = 100
        nchannels = 500
        a = numpy.ones((nrows, ncolumns), numpy.float)
        referenceData = numpy.zeros((nchannels, nrows, ncolumns),
                                   numpy.float)
        for i in range(nchannels):
            referenceData[i] = a * i
        a = None
        mask = numpy.zeros((nrows, ncolumns), numpy.uint8)
        mask[20:30, 15:50] = 1

        dummyArray = DummyArray(referenceData)

        defaultMca = referenceData.sum(axis=2, dtype=numpy.float64).sum(axis=1)
        maskedMca = referenceData[:,mask>0].sum(axis=1)

        for fileindex in [1, 2]:
            #usually only one file index case is used but
            #we test both to have a better coverage
            j = 0
            for data in [referenceData, dummyArray]:
                if j == 0:
                    dynamic = ""
                    j = 1
                else:
                    dynamic = "dynamic "
                stackBase = StackBase.StackBase()
                stackBase.setStack(data, mcaindex=0, fileindex=fileindex)
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
                i1 = 400
                # calculate
                imageDict = stackBase.calculateROIImages(i0, i1, imiddle=imiddle)
                self.assertTrue(numpy.allclose(imageDict['ROI'], data[i0:i1, :,:].sum(axis=0)),
                        "Incorrect ROI image from %sROI calculation"  % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Left'], data[i0,:,:]),
                        "Incorrect Left image from %sROI calculation" % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Right'], data[i1-1,:,:]),
                        "Incorrect Right image from %sROI calculation"  % dynamic)
                self.assertTrue(numpy.allclose(imageDict['Middle'], data[imiddle,:,:]),
                        "Incorrect Middle image from %sROI calculation" % dynamic)
        stackBase = None
        data = None
        dummyArray = None
        referenceData = None

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
