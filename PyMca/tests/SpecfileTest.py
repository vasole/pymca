import unittest
import sys
import os
import gc
import tempfile

class testSpecfile(unittest.TestCase):
    def setUp(self):
        """
        import the specfile module
        """
        try:
            from PyMca import specfile
            self.specfileClass = specfile
        except:
            self.specfileClass = None
        if self.specfileClass is not None:
            text  = "#F \n"
            text += "\n"
            text += "#S 10  Undefined command 0\n"
            text += "#N 3\n"
            text += "#L First label  Second label  Third label\n"
            text += "10  100  1000\n"
            text += "20  400  8000\n"
            text += "30  900  270000\n"
            text += "\n"
            text += "#S 20  Undefined command 1\n"
            text += "#N 3\n"
            text += "#L First  Second  Third\n"
            text += "1.3  1  1\n"
            text += "2.5  4  8\n"
            text += "3.7  9  27\n"
            text += "\n"
            tmpFile = tempfile.mkstemp(text=False)
            if sys.version < '3.0':
                os.write(tmpFile[0], text)
            else:
                os.write(tmpFile[0], bytes(text, 'utf-8'))
            os.close(tmpFile[0])
            self.fname = tmpFile[1]

    def tearDown(self):
        """clean up any possible files"""
        gc.collect()
        if self.specfileClass is not None:
            if os.path.exists(self.fname):
                os.remove(self.fname)

    def testSpecfileImport(self):
        #"""Test successful import"""
        self.assertTrue(self.specfileClass is not None)

    def testSpecfileReading(self):
        #"""Test specfile readout"""
        self.assertTrue(self.specfileClass is not None)
        sf = self.specfileClass.Specfile(self.fname)
        # test the number of found scans
        self.assertEqual(len(sf), 2)
        self.assertEqual(sf.scanno(), 2)
        # test scan iteration selection method
        scan = sf[1]
        labels = scan.alllabels()
        expectedLabels = ['First', 'Second', 'Third']
        self.assertEqual(len(labels), 3)
        for i in range(3):
            self.assertEqual(labels[i], expectedLabels[i])
        # test scan number selection method
        scan = sf.select('20.1')
        labels = scan.alllabels()
        sf = None
        expectedLabels = ['First', 'Second', 'Third']
        self.assertEqual(len(labels), 3)
        for i in range(3):
            self.assertEqual(labels[i], expectedLabels[i])
        gc.collect()

    def testSpecfileReadingCompatibleWithUserLocale(self):
        #"""Test specfile compatible with C locale"""
        self.assertTrue(self.specfileClass is not None)
        sf = self.specfileClass.Specfile(self.fname)
        scan = sf[1]
        datacol = scan.datacol(1)
        data = scan.data()
        sf = None
        self.assertEqual(datacol[0], 1.3)
        self.assertEqual(datacol[1], 2.5)
        self.assertEqual(datacol[2], 3.7)
        self.assertEqual(datacol[1], data[0][1])
        gc.collect()

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testSpecfile))
    else:
        # use a predefined order
        testSuite.addTest(testSpecfile("testSpecfileImport"))
        testSuite.addTest(testSpecfile("testSpecfileReading"))
        testSuite.addTest(\
            testSpecfile("testSpecfileReadingCompatibleWithUserLocale"))
    return testSuite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=False))
