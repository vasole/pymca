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
        self.assertIsNotNone(self.fileClass)

    def testEdfFileReadWrite(self):
        # create a file
        self.assertIsNotNone(self.fileClass)
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
        self.assertEqual(readData.dtype, numpy.int32)
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

def getSuite():
    suite = unittest.TestLoader().loadTestsFromTestCase(testEdfFile)
    return suite

if __name__ == '__main__':
    #unittest.main()
    unittest.TextTestRunner(verbosity=2).run(getSuite())
