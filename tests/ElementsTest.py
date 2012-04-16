import unittest
import os
import sys
import numpy    

class testElements(unittest.TestCase):
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
        from PyMca import Elements
        self._elements = Elements

    def testDataDirectoryPresence(self):
        # Testing directory presence
        try:
            self.assertIsNotNone(self.dataDir)
            self.assertTrue(os.path.exists(self.dataDir))
            self.assertTrue(os.path.isdir(self.dataDir))
        except:
            print("\n Cannot find PyMcaData directory: %s" % self.dataDir)
            raise

    def testPeakIdentification(self):
        # energy in keV
        energy = 5.9
        # 10 eV threshold
        threshold = 0.010
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K'])
        self.assertTrue(len(lines[0]['elements']) == 1)
        self.assertTrue(lines[0]['energy'] == energy)
        self.assertTrue(lines[0]['elements'][0] == 'Mn')

        energy = 10.550
        threshold = 0.030
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K'])
        self.assertTrue(len(lines[0]['elements']) == 1)
        self.assertTrue(lines[0]['energy'] == energy)
        self.assertTrue('As' in lines[0]['elements'])
        self.assertTrue('Pb' not in lines[0]['elements'])

        # Test K and L lines
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold,
                                             targetrays=['K', 'L'])
        self.assertTrue(len(lines[0]['elements']) > 1)
        self.assertTrue('As' in lines[0]['elements'])
        self.assertTrue('Pb' in lines[0]['elements'])

        # Test all
        energy = 2.280
        threshold = 0.030
        lines = self._elements.getcandidates(energy,
                                             threshold=threshold)
        self.assertTrue(len(lines[0]['elements']) > 1)
        self.assertTrue('As' not in lines[0]['elements'])
        self.assertTrue('Pb' not in lines[0]['elements'])
        self.assertTrue('S' in lines[0]['elements'])
        self.assertTrue('Hg' in lines[0]['elements'])

def getSuite(auto=True):
    testSuite = unittest.TestSuite()
    if auto:
        testSuite.addTest(\
            unittest.TestLoader().loadTestsFromTestCase(testElements))
    else:
        testSuite.addTest(testElements("testDataDirectoryPresence"))
        testSuite.addTest(testElements("testPeakIdentification"))
    return testSuite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(getSuite(auto=False))
