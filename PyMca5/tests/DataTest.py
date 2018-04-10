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
import os
import sys
import numpy

class testData(unittest.TestCase):
    ELEMENTS = ['H', 'He',
                'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
                'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
                'Bh', 'Hs', 'Mt']

    def setUp(self):
        """
        Get the data directory
        """
        self._importSuccess = False
        try:
            from PyMca5 import PyMcaDataDir
            self._importSuccess = True
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None

        try:
            self.docDir = PyMcaDataDir.PYMCA_DOC_DIR
        except:
            self.docDir = None

    def testDataDirectoryPresence(self):
        self.assertTrue(self._importSuccess,
                        'Unsuccessful PyMca5.PyMcaDataDir import')
        self.assertTrue(self.dataDir is not None,
                        'Unassigned PyMca5.PyMcaDataDir.PYMCA_DATA_DIR')
        self.assertTrue(os.path.exists(self.dataDir),
                        'Directory "%s" does not exist' % self.dataDir)
        self.assertTrue(os.path.isdir(self.dataDir),
                        '"%s" expected to be a directory' % self.dataDir)


    def testFisxDataDirectoryPresence(self):
        try:
            from fisx import DataDir
            dataDir = DataDir.FISX_DATA_DIR
        except:
            dataDir = None
        self.assertTrue(dataDir is not None,
                        'fisx module not properly installed')
        self.assertTrue(os.path.exists(dataDir),
                        'fisx directory "%s" does not exist' % dataDir)
        self.assertTrue(os.path.isdir(dataDir),
                        '"%s" expected to be a directory' % dataDir)
        for fname in ['BindingEnergies.dat',
                      'EADL97_BindingEnergies.dat',
                      'EADL97_KShellConstants.dat',
                      'EADL97_LShellConstants.dat',
                      'EADL97_MShellConstants.dat',
                      'EPDL97_CrossSections.dat',
                      'KShellConstants.dat',
                      'KShellRates.dat',
                      'LShellConstants.dat',
                      'LShellRates.dat',
                      'MShellConstants.dat',
                      'MShellRates.dat',
                      'XCOM_CrossSections.dat']:
            actualName = os.path.join(dataDir, fname)
            self.assertTrue(os.path.exists(actualName),
                            'File "%s" does not exist.' % actualName)
            self.assertTrue(os.path.isfile(actualName),
                            'File "%s" is not an actual file.' % actualName)

    def testDataFilePresence(self):
        # Testing file presence
        self.testDataDirectoryPresence()
        for fname in ['LShellRatesCampbell.dat',
                      'LShellRatesScofieldHS.dat',
                      'McaTheory.cfg',
                      'Scofield1973.dict',
                      'XRFSpectrum.mca',
                      'Steel.cfg', 'Steel.spe']:
            actualName = os.path.join(self.dataDir, fname)
            self.assertTrue(os.path.exists(actualName),
                            'File "%s" does not exist.' % actualName)
            self.assertTrue(os.path.isfile(actualName),
                            'File "%s" is not an actual file.' % actualName)
        actualName = os.path.join(self.dataDir, 'attdata')
        self.assertTrue(os.path.exists(actualName),
                        'Directory "%s" does not exist' % actualName)
        self.assertTrue(os.path.isdir(actualName),
                        '"%s" expected to be a directory' % actualName)
        for i in range(92):
            fname = "%s%s" % (testData.ELEMENTS[i], ".mat")
            actualName = os.path.join(self.dataDir, 'attdata', fname)
            self.assertTrue(os.path.exists(actualName),
                            'File "%s" does not exist.' % actualName)
            self.assertTrue(os.path.isfile(actualName),
                            'File "%s" is not an actual file.' % actualName)

    def testDocDirectoryPresence(self):
        self.assertTrue(self._importSuccess,
                        'Unsuccessful PyMca5.PyMcaDataDir import')
        self.assertTrue(self.docDir is not None,
                        'Unassigned PyMca5.PyMcaDataDir.PYMCA_DOC_DIR')
        self.assertTrue(os.path.exists(self.dataDir),
                        'Directory "%s" does not exist' % self.docDir)
        self.assertTrue(os.path.isdir(self.dataDir),
                        '"%s" expected to be a directory' % self.docDir)

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testData))
    else:
        # use a predefined order
        testSuite.addTest(testData("testFisxDataDirectoryPresence"))
        testSuite.addTest(testData("testDataDirectoryPresence"))
        testSuite.addTest(testData("testDataFilePresence"))
        testSuite.addTest(testData("testDocDirectoryPresence"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
