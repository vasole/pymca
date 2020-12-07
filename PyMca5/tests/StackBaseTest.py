#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
__author__ = "V. Armando Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
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
        a = numpy.ones((nrows, ncolumns), numpy.float64)
        referenceData = numpy.zeros((nrows, ncolumns, nchannels),
                                   numpy.float64)
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
        a = numpy.ones((nrows, ncolumns), numpy.float64)
        referenceData = numpy.zeros((nchannels, nrows, ncolumns),
                                   numpy.float64)
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
