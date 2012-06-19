#/*##########################################################################
# Copyright (C) 2004 - 2012 European Synchrotron Radiation Facility
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
import sys
import os
import gc
import tempfile
import numpy

class testEdfFile(unittest.TestCase):
    def setUp(self):
        """
        import the EdfFile module
        """
        tmpFile = tempfile.mkstemp(text=False)
        os.close(tmpFile[0])
        self.fname = tmpFile[1]
        try:
            from PyMca.EdfFile import EdfFile
            self.fileClass = EdfFile
        except:
            self.fileClass = None

    def tearDown(self):
        """clean up any possible files"""
        gc.collect()
        if self.fileClass is not None:
            if os.path.exists(self.fname):
                os.remove(self.fname)

    def testEdfFileImport(self):
        #"""Test successful import"""
        self.assertTrue(self.fileClass is not None)

    def testEdfFileReadWrite(self):
        # create a file
        self.assertTrue(self.fileClass is not None)
        data = numpy.arange(10000).astype(numpy.int32)
        data.shape = 100, 100
        edf = self.fileClass(self.fname, 'wb+')
        edf.WriteImage({'Title': "title",
                        'key': 'key'}, data)
        edf = None

        # read it
        edf = self.fileClass(self.fname, 'rb')
        # the number of images
        nImages = edf.GetNumImages()
        self.assertEqual(nImages, 1)
        # the header information
        header = edf.GetHeader(0)
        self.assertEqual(header['Title'], "title")
        self.assertEqual(header['key'], "key")

        #the data information
        readData = edf.GetData(0)
        self.assertEqual(readData.dtype, numpy.int32,
                         'Read type %s instead of %s' %\
                        (readData.dtype, numpy.int32))
        self.assertEqual(readData[10,20], data[10,20])
        self.assertEqual(readData[20,10], data[20,10])
        edf =None

        # add a second Image
        edf = self.fileClass(self.fname, 'rb+')
        edf.WriteImage({'Title': "title2", 'key': 'key2'},
                       data.astype(numpy.float32), Append=1)
        edf = None

        # read it
        edf = self.fileClass(self.fname, 'rb')
        # the number of images
        nImages = edf.GetNumImages()
        self.assertEqual(nImages, 2)

        # the header information
        header = edf.GetHeader(1)
        self.assertEqual(header['Title'], "title2")
        self.assertEqual(header['key'], "key2")

        # the data information
        readData = edf.GetData(1)
        self.assertEqual(readData.dtype, numpy.float32)
        self.assertTrue(abs(readData[10,20]-data[10,20]) < 0.00001)
        self.assertTrue(abs(readData[20,10]-data[20,10]) < 0.00001)
        edf =None
        gc.collect()

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testEdfFile))
    else:
        # use a predefined order
        testSuite.addTest(testEdfFile("testEdfFileImport"))
        testSuite.addTest(testEdfFile("testEdfFileReadWrite"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
