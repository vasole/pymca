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
            from PyMca import PyMcaDataDir
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
                        'Unsuccessful PyMca.PyMcaDataDir import')
        self.assertTrue(self.dataDir is not None,
                        'Unassigned PyMca.PyMcaDataDir.PYMCA_DATA_DIR')
        self.assertTrue(os.path.exists(self.dataDir),
                        'Directory "%s" does not exist' % self.dataDir)
        self.assertTrue(os.path.isdir(self.dataDir),
                        '"%s" expected to be a directory' % self.dataDir)

    def testDataFilePresence(self):
        # Testing file presence
        self.testDataDirectoryPresence()
        for fname in ['BindingEnergies.dat',
                      'EADL97_BindingEnergies.dat',
                      'EADL97_KShellConstants.dat',
                      'EADL97_LShellConstants.dat',
                      'EADL97_MShellConstants.dat',
                      'EPDL97_CrossSections.dat',
                      'KShellConstants.dat',
                      'KShellRates.dat',
                      'KShellRatesScofieldHS.dat',
                      'LShellConstants.dat',
                      'LShellRates.dat',
                      'LShellRatesCampbell.dat',
                      'LShellRatesScofieldHS.dat',
                      'McaTheory.cfg',
                      'MShellConstants.dat',
                      'MShellRates.dat',
                      'Scofield1973.dict',
                      'XCOM_CrossSections.dat',
                      'XRFSpectrum.mca']:
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
                        'Unsuccessful PyMca.PyMcaDataDir import')
        self.assertTrue(self.docDir is not None,
                        'Unassigned PyMca.PyMcaDataDir.PYMCA_DOC_DIR')
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
        testSuite.addTest(testData("testDataDirectoryPresence"))
        testSuite.addTest(testData("testDataFilePresence"))
        testSuite.addTest(testData("testDocDirectoryPresence"))
    return testSuite

def test(auto=False):
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=auto))

if __name__ == '__main__':
    test()
