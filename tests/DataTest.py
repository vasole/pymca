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
        try:
            from PyMca import PyMcaDataDir
            self.dataDir = PyMcaDataDir.PYMCA_DATA_DIR
        except:
            self.dataDir = None

    def testDataDirectoryPresence(self):
        # Testing directory presence
        try:
            self.assertTrue(self.dataDir is not None)
            self.assertTrue(os.path.exists(self.dataDir))
            self.assertTrue(os.path.isdir(self.dataDir))
        except:
            print("\n Cannot find PyMcaData directory: %s" % self.dataDir)
            raise

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
                      'XCOM_CrossSections.dat']:
            actualName = os.path.join(self.dataDir, fname)
            try:
                self.assertTrue(os.path.exists(actualName))
                self.assertTrue(os.path.isfile(actualName))
            except:
                print('File "%s" does not exist.' % actualName)
                raise                
        self.assertTrue(os.path.isdir(os.path.join(self.dataDir, 'attdata')))
        for i in range(92):
            fname = "%s%s" % (testData.ELEMENTS[i], ".mat")
            actualName = os.path.join(self.dataDir, 'attdata', fname)
            try:
                self.assertTrue(os.path.exists(actualName))
                self.assertTrue(os.path.isfile(actualName))
            except:
                print('File "%s" does not exist.' % fname)
                raise
        
def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(unittest.TestLoader().loadTestsFromTestCase(testData))
    else:
        # use a predefined order
        testSuite.addTest(testData("testDataDirectoryPresence"))
        testSuite.addTest(testData("testDataFilePresence"))
    return testSuite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=False))
