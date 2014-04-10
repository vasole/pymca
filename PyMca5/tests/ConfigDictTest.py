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

class testConfigDict(unittest.TestCase):
    def setUp(self):
        """
        import the module
        """
        try:
            from PyMca5.PyMcaIO import ConfigDict
            self._module = ConfigDict
        except:
            self._module = None
        self._tmpFileName = None

    def tearDown(self):
        """clean up any possible files"""
        gc.collect()
        if self._tmpFileName is not None:
            if os.path.exists(self._tmpFileName):
                os.remove(self._tmpFileName)

    def testConfigDictImport(self):
        #"""Test successful import"""
        self.assertTrue(self._module is not None,\
                        "Unsuccessful PyMca.ConfigDict import")

    def testConfigDictIO(self):
        # create a dictionnary
        from PyMca5.PyMcaIO import ConfigDict
        testDict = {}
        testDict['simple_types'] = {} 
        testDict['simple_types']['float'] = 1.0
        testDict['simple_types']['int'] = 1
        testDict['simple_types']['string'] = "Hello World"
        testDict['containers'] = {}
        testDict['containers']['list'] = [-1, 'string', 3.0]
        if ConfigDict.USE_NUMPY:
            import numpy
            testDict['containers']['array'] = numpy.array([1.0, 2.0, 3.0])
        testDict['containers']['dict'] = {'key1': 'Hello World',
                                          'key2': 2.0}

        tmpFile = tempfile.mkstemp(text=False)
        os.close(tmpFile[0])
        self._tmpFileName = tmpFile[1]

        writeInstance = ConfigDict.ConfigDict(initdict=testDict)
        writeInstance.write(self._tmpFileName)

        #read the data back
        readInstance = ConfigDict.ConfigDict()
        readInstance.read(self._tmpFileName)

        # get read key list
        testDictKeys = list(testDict.keys()) 
        readKeys = list(readInstance.keys())
        self.assertTrue(len(readKeys) == len(testDictKeys),
                    "Number of read keys not equal to number of written keys")

        topKey = 'simple_types'
        for key in testDict[topKey]:
            original = testDict[topKey][key]
            read = readInstance[topKey][key]
            self.assertTrue( read == original,
                            "Read <%s> instead of <%s>" % (read, original))

        topKey = 'containers'
        for key in testDict[topKey]:
            original = testDict[topKey][key]
            read = readInstance[topKey][key]
            if key == 'array':
                self.assertTrue( read.all() == original.all(),
                            "Read <%s> instead of <%s>" % (read, original))
            else:
                self.assertTrue( read == original,
                            "Read <%s> instead of <%s>" % (read, original))

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testConfigDict))
    else:
        # use a predefined order
        testSuite.addTest(testConfigDict("testConfigDictImport"))
        testSuite.addTest(testConfigDict("testConfigDictIO"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
